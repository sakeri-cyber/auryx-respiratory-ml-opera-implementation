"""OPERA-GT encoder, reimplemented from scratch to match the released checkpoint.

OPERA-GT is a masked-autoencoder Vision Transformer pretrained on 404 hours of
respiratory audio (Zhang et al., NeurIPS 2024 D&B). The published checkpoint
contains both encoder and decoder; only the encoder is needed for representation
extraction, so the decoder is dropped at load time.

Architecture recovered from `encoder-operaGT.ckpt`:

    patch_embed.proj      Conv2d(1, 384, kernel=4, stride=4)
    cls_token             (1, 1, 384)
    pos_embed             (1, 1025, 384)          1024 patches + CLS
    blocks.0 .. blocks.11 12 x standard pre-norm ViT block, dim 384, MLP ratio 4
    norm                  LayerNorm(384)
    -- decoder (unused) --
    decoder_embed         Linear(384, 256)
    decoder_blocks.0..15  16 x ViT block, dim 256
    decoder_pred          Linear(256, 16)         16 = 4*4*1, one patch

1024 patches at 4x4 with 64 mel bins implies an input of (1, 64, 256):
    (64/4) * (256/4) = 16 * 64 = 1024.

No timm, torchvision or torchaudio dependency — every layer is written out, both
so it runs on a bare runtime and so the weight-shape correspondence is explicit
and checkable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GTConfig:
    img_size: tuple[int, int] = (64, 256)  # (n_mels, n_frames)
    patch_size: int = 4
    in_chans: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6  # ViT-Small convention for dim 384; 384/6 = 64 per head
    mlp_ratio: float = 4.0

    @property
    def grid(self) -> tuple[int, int]:
        return self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid[0] * self.grid[1]


class PatchEmbed(nn.Module):
    """Split the spectrogram into non-overlapping 4x4 patches and project each to `embed_dim`.

    A strided convolution is exactly equivalent to slicing patches and applying a
    shared linear layer, but is far faster and is how the pretrained weights are
    stored.
    """

    def __init__(self, cfg: GTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(cfg.in_chans, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> (B, num_patches, embed_dim)."""
        if x.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W), got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % self.cfg.patch_size or w % self.cfg.patch_size:
            raise ValueError(
                f"Input {h}x{w} is not divisible by patch size {self.cfg.patch_size}"
            )
        x = self.proj(x)  # (B, D, gh, gw)
        return x.flatten(2).transpose(1, 2)  # (B, gh*gw, D)


class Attention(nn.Module):
    """Standard multi-head self-attention with a fused qkv projection."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        # (B, N, 3D) -> (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Fused SDPA: numerically stable and materially faster than a hand-rolled
        # softmax(QK^T/sqrt(d))V, which matters because this runs 12 times per clip.
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x)).

    Pre-norm (rather than post-norm) is what MAE and modern ViTs use; it keeps the
    residual path clean and makes deep stacks trainable without warmup tricks.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class OperaGTEncoder(nn.Module):
    """The OPERA-GT encoder. Frozen feature extractor for linear probing."""

    def __init__(self, cfg: GTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or GTConfig()

        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches + 1, cfg.embed_dim))
        self.blocks = nn.ModuleList(
            [Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

    @property
    def feature_dim(self) -> int:
        return self.cfg.embed_dim

    def forward(self, x: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
        """(B, 1, n_mels, n_frames) -> (B, embed_dim).

        `pooling`:
            "mean" — average of patch tokens, excluding CLS. The MAE convention:
                     MAE has no contrastive objective training the CLS token, so
                     mean-pooled patch tokens are the stronger representation.
            "cls"  — the CLS token alone.
            "both" — concatenation, giving 2*embed_dim.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.patch_embed(x)  # (B, N, D)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)

        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Token count {x.shape[1]} != positional embedding {self.pos_embed.shape[1]}. "
                f"Input must be {self.cfg.img_size} for a {self.cfg.patch_size}x{self.cfg.patch_size} patch grid."
            )
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if pooling == "cls":
            return x[:, 0]
        if pooling == "mean":
            return x[:, 1:].mean(dim=1)
        if pooling == "both":
            return torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=-1)
        raise ValueError(f"Unknown pooling {pooling!r}")


def load_opera_gt(ckpt_path: Path | str, cfg: GTConfig | None = None, strict: bool = True) -> OperaGTEncoder:
    """Load released OPERA-GT weights into the encoder, discarding the decoder.

    Verifies that every encoder parameter was populated. A silent partial load
    would produce randomly-initialised layers and plausible-looking but meaningless
    embeddings — the failure mode most likely to waste a week, so it raises instead.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    # Keep encoder-side tensors only.
    encoder_state = {
        k: v
        for k, v in state.items()
        if not k.startswith(("decoder_", "mask_token"))
    }

    model = OperaGTEncoder(cfg)
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)

    if strict and missing:
        raise RuntimeError(
            f"{len(missing)} encoder parameters were not found in the checkpoint: {missing[:8]}"
        )
    if unexpected:
        logger.info("Ignored %d non-encoder tensors from checkpoint", len(unexpected))

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded OPERA-GT encoder: %s parameters, feature dim %d", f"{n_params:,}", model.feature_dim)
    return model.eval()


@torch.inference_mode()
def extract_features(
    model: OperaGTEncoder,
    spectrograms: torch.Tensor,
    batch_size: int = 16,
    device: str = "cpu",
    pooling: str = "mean",
) -> torch.Tensor:
    """Batched feature extraction over a stack of spectrograms.

    Run once and cache. Every downstream experiment operates on the cached matrix,
    so a lost session costs nothing after this step completes.
    """
    model = model.to(device).eval()
    outputs = []
    for start in range(0, spectrograms.shape[0], batch_size):
        batch = spectrograms[start : start + batch_size].to(device)
        outputs.append(model(batch, pooling=pooling).cpu())
    return torch.cat(outputs, dim=0)
