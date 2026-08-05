"""Tests for the from-scratch OPERA-GT encoder.

The checkpoint tests are marked `slow` because they need the 394 MB download.
Everything else runs on a randomly-initialised model in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from respnet.models.opera_gt import (
    Attention,
    Block,
    GTConfig,
    OperaGTEncoder,
    PatchEmbed,
    load_opera_gt,
)

CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts/checkpoints/encoder-operaGT.ckpt"


@pytest.fixture
def cfg() -> GTConfig:
    return GTConfig()


class TestGTConfig:
    def test_patch_grid(self, cfg):
        assert cfg.grid == (16, 64)

    def test_num_patches_matches_released_positional_embedding(self, cfg):
        """pos_embed in the checkpoint is (1, 1025, 384) = 1024 patches + CLS."""
        assert cfg.num_patches == 1024

    def test_head_dim_divides_evenly(self, cfg):
        assert cfg.embed_dim % cfg.num_heads == 0


class TestPatchEmbed:
    def test_output_shape(self, cfg):
        out = PatchEmbed(cfg)(torch.randn(2, 1, 64, 256))
        assert out.shape == (2, 1024, 384)

    def test_rejects_non_divisible_input(self, cfg):
        with pytest.raises(ValueError, match="divisible"):
            PatchEmbed(cfg)(torch.randn(1, 1, 63, 256))

    def test_rejects_wrong_rank(self, cfg):
        with pytest.raises(ValueError, match="Expected"):
            PatchEmbed(cfg)(torch.randn(1, 64, 256))


class TestAttention:
    def test_shape_preserved(self):
        attn = Attention(dim=384, num_heads=6)
        x = torch.randn(2, 100, 384)
        assert attn(x).shape == x.shape

    def test_rejects_indivisible_head_count(self):
        with pytest.raises(ValueError, match="divisible"):
            Attention(dim=384, num_heads=5)

    def test_permutation_equivariance(self):
        """Self-attention without positional information must be permutation-equivariant.

        This is *why* pos_embed exists. If this test fails, the attention block is
        leaking positional information it shouldn't have.
        """
        torch.manual_seed(0)
        attn = Attention(dim=64, num_heads=4).eval()
        x = torch.randn(1, 10, 64)
        perm = torch.randperm(10)
        with torch.inference_mode():
            assert torch.allclose(attn(x)[:, perm], attn(x[:, perm]), atol=1e-5)


class TestBlock:
    def test_shape_preserved(self):
        assert Block(384, 6, 4.0)(torch.randn(2, 50, 384)).shape == (2, 50, 384)

    def test_is_residual(self):
        """With zeroed output projections the block must be the identity."""
        block = Block(384, 6, 4.0).eval()
        with torch.no_grad():
            block.attn.proj.weight.zero_(); block.attn.proj.bias.zero_()
            block.mlp.fc2.weight.zero_(); block.mlp.fc2.bias.zero_()
        x = torch.randn(1, 20, 384)
        with torch.inference_mode():
            assert torch.allclose(block(x), x, atol=1e-6)


class TestEncoder:
    def test_forward_shapes(self, cfg):
        model = OperaGTEncoder(cfg).eval()
        x = torch.randn(2, 1, 64, 256)
        with torch.inference_mode():
            assert model(x, pooling="mean").shape == (2, 384)
            assert model(x, pooling="cls").shape == (2, 384)
            assert model(x, pooling="both").shape == (2, 768)

    def test_accepts_unbatched_channel_dim(self, cfg):
        model = OperaGTEncoder(cfg).eval()
        with torch.inference_mode():
            assert model(torch.randn(2, 64, 256)).shape == (2, 384)

    def test_wrong_input_size_raises_informative_error(self, cfg):
        model = OperaGTEncoder(cfg).eval()
        with pytest.raises(ValueError, match="positional embedding"):
            model(torch.randn(1, 1, 64, 128))

    def test_unknown_pooling_raises(self, cfg):
        with pytest.raises(ValueError, match="Unknown pooling"):
            OperaGTEncoder(cfg)(torch.randn(1, 1, 64, 256), pooling="nonsense")

    def test_deterministic_in_eval(self, cfg):
        model = OperaGTEncoder(cfg).eval()
        x = torch.randn(1, 1, 64, 256)
        with torch.inference_mode():
            assert torch.equal(model(x), model(x))


@pytest.mark.slow
class TestCheckpointLoading:
    @pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not downloaded")
    def test_loads_strictly_with_no_missing_parameters(self):
        """Strict load is the guard against a silent partial load.

        If the architecture were wrong, some layers would stay randomly
        initialised and the embeddings would be meaningless but plausible-looking.
        """
        model = load_opera_gt(CHECKPOINT, strict=True)
        assert isinstance(model, OperaGTEncoder)

    @pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not downloaded")
    def test_parameter_count_matches_published_figure(self):
        """The paper states 21M parameters for the OPERA-GT encoder."""
        model = load_opera_gt(CHECKPOINT)
        n = sum(p.numel() for p in model.parameters())
        assert 21_000_000 < n < 22_000_000, f"got {n:,}"

    @pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not downloaded")
    def test_pretrained_weights_are_not_random(self):
        """Loaded weights must differ from a fresh initialisation."""
        loaded = load_opera_gt(CHECKPOINT)
        fresh = OperaGTEncoder()
        assert not torch.allclose(loaded.pos_embed, fresh.pos_embed)

    def test_missing_checkpoint_raises(self):
        with pytest.raises(FileNotFoundError):
            load_opera_gt("/nonexistent/model.ckpt")
