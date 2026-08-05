"""A small supervised CNN trained directly on the task — component B.

The question this exists to answer: **does 404 hours of self-supervised pretraining
actually beat a small task-specific model on this task?**

OPERA's benchmark compares its encoders against *other pretrained models*
(OpenSMILE, VGGish, AudioMAE, CLAP). It never compares against a supervised
baseline trained directly on the downstream data. That is the practitioner's
actual question and the paper does not answer it.

Kept deliberately small (~200k parameters) because ICBHI T7 has 126 patients.
Anything larger memorises the training patients and tells us nothing.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool.

    BatchNorm before the activation is the original formulation and matters more
    than usual here: ICBHI's four recording devices have very different gains, and
    normalising per batch keeps early activations comparable across them.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class BaselineCNN(nn.Module):
    """Four conv blocks, global average pooling, linear head.

    Global average pooling rather than a flatten: it makes the model invariant to
    input length and removes the large fully-connected layer that would otherwise
    dominate the parameter count and overfit immediately on 126 patients.
    """

    def __init__(self, n_classes: int = 2, channels: tuple[int, ...] = (16, 32, 64, 64), dropout: float = 0.3) -> None:
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.classifier(self.dropout(h))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_baseline(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 10,
) -> tuple[BaselineCNN, dict]:
    """Train the baseline and return (model, history).

    Class weighting is applied in the loss rather than by resampling, because
    resampling would duplicate recordings from the same patient and worsen the
    memorisation problem the small architecture is already guarding against.
    """
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    model = BaselineCNN().to(device)

    counts = torch.bincount(y_train, minlength=2).float()
    weights = (counts.sum() / (2 * counts.clamp_min(1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    history: dict[str, list[float]] = {"loss": [], "test_auroc": []}
    n = x_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            logits = torch.cat(
                [model(x_test[i : i + batch_size].to(device)).cpu() for i in range(0, len(x_test), batch_size)]
            )
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        try:
            auroc = float(roc_auc_score(y_test.numpy(), probs))
        except ValueError:  # single class present in the test split
            auroc = float("nan")

        history["loss"].append(epoch_loss / n)
        history["test_auroc"].append(auroc)

        if log_every and (epoch + 1) % log_every == 0:
            logger.info("  epoch %3d  loss %.4f  test AUROC %.3f", epoch + 1, history["loss"][-1], auroc)

    return model, history
