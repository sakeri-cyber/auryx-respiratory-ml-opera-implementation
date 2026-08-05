"""End-to-end experiment pipeline.

Stages, in dependency order:

    0. Index ICBHI recordings + diagnoses
    1. Waveform -> log-mel spectrograms          (cached to artifacts/)
    2. Spectrograms -> OPERA-GT embeddings       (cached to artifacts/)
    A. T7 COPD linear probe                       -- reproduction
    C1. Nuisance probes (device / patient / site) -- the novel experiment
    C2. Deployment profile                        -- unpublished by the paper
    C3. Supervised CNN baseline                   -- the comparison the paper omits

Everything after stage 2 runs on cached matrices in seconds, so an interrupted
session never costs more than the extraction it already completed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data.audio import AudioConfig, LogMelSpectrogram, clip_waveform, load_wav, lowpass
from .data.icbhi import (
    DEVICES,
    Recording,
    bandwidth_matched_subset,
    load_recordings,
    subject_wise_split,
    summarise,
)
from .models.opera_gt import OperaGTEncoder, extract_features, load_opera_gt
from .probe import linear_probe, shuffled_control

logger = logging.getLogger(__name__)


def pick_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------ stages 1 & 2


def build_spectrograms(
    recordings: list[Recording],
    cfg: AudioConfig,
    cache_path: Path | None = None,
    bandwidth_limit_hz: float | None = None,
) -> torch.Tensor:
    """Load every recording and convert it to one fixed-size log-mel spectrogram.

    One clip per recording, taken from the centre. The centre is a deliberate
    choice: ICBHI recordings frequently begin with stethoscope placement noise,
    and starting at t=0 would sample that handling transient rather than lung sound.

    `bandwidth_limit_hz` low-passes every recording to a common bandwidth before
    analysis. Used for the bandwidth-controlled arm of C1 — see `docs/findings.md`.
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached spectrograms from %s", cache_path)
        return torch.load(cache_path)

    mel = LogMelSpectrogram(cfg)
    need = int(round(cfg.clip_seconds * cfg.sample_rate))
    out = torch.empty(len(recordings), cfg.n_mels, cfg.n_frames)

    start_time = time.perf_counter()
    for i, rec in enumerate(recordings):
        waveform, _ = load_wav(rec.wav_path, cfg.sample_rate)
        if bandwidth_limit_hz is not None:
            waveform = lowpass(waveform, cfg.sample_rate, bandwidth_limit_hz)
        offset = max(0, (len(waveform) - need) // 2)
        clip = clip_waveform(waveform, cfg, offset=offset)
        out[i] = mel(torch.from_numpy(clip.copy()))[0]

        if (i + 1) % 200 == 0:
            logger.info("  %d/%d spectrograms", i + 1, len(recordings))

    logger.info("Built %d spectrograms in %.1fs", len(recordings), time.perf_counter() - start_time)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache_path)
    return out


def build_features(
    model: OperaGTEncoder,
    spectrograms: torch.Tensor,
    cache_path: Path | None = None,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Run the frozen encoder over every spectrogram and cache the result."""
    if cache_path and cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        return np.load(cache_path)

    start_time = time.perf_counter()
    feats = extract_features(model, spectrograms, batch_size=batch_size, device=device).numpy()
    logger.info(
        "Extracted %s features in %.1fs on %s", feats.shape, time.perf_counter() - start_time, device
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, feats)
    return feats


def sanity_check_embeddings(features: np.ndarray, recordings: list[Recording], n_pairs: int = 200, seed: int = 0) -> dict:
    """Are two clips from the same *patient* more similar than clips from different patients?

    If not, preprocessing is wrong and every downstream number is noise. This is the
    cheapest possible guard against a silent input-format mismatch — the failure
    mode that would otherwise cost days.
    """
    rng = np.random.default_rng(seed)
    norm = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)

    by_patient: dict[int, list[int]] = {}
    for i, rec in enumerate(recordings):
        by_patient.setdefault(rec.patient_id, []).append(i)
    eligible = [p for p, idx in by_patient.items() if len(idx) >= 2]

    same, diff = [], []
    for _ in range(n_pairs):
        p = eligible[rng.integers(len(eligible))]
        i, j = rng.choice(by_patient[p], size=2, replace=False)
        same.append(float(norm[i] @ norm[j]))

        q = p
        while q == p:
            q = eligible[rng.integers(len(eligible))]
        k = by_patient[q][rng.integers(len(by_patient[q]))]
        diff.append(float(norm[i] @ norm[k]))

    result = {
        "same_patient_cosine": float(np.mean(same)),
        "diff_patient_cosine": float(np.mean(diff)),
        "separation": float(np.mean(same) - np.mean(diff)),
        "passed": bool(np.mean(same) > np.mean(diff)),
    }
    logger.info(
        "Sanity check: same-patient %.4f vs different-patient %.4f (separation %+.4f) -> %s",
        result["same_patient_cosine"], result["diff_patient_cosine"], result["separation"],
        "PASS" if result["passed"] else "FAIL",
    )
    return result


# ---------------------------------------------------------------------- labels


def labels_for(recordings: list[Recording], target: str) -> np.ndarray:
    """Extract a label vector for one of the probe targets."""
    if target == "copd":
        return np.array([r.is_copd for r in recordings])
    if target == "device":
        return np.array([r.device_index for r in recordings])
    if target == "patient":
        pids = sorted({r.patient_id for r in recordings})
        lookup = {p: i for i, p in enumerate(pids)}
        return np.array([lookup[r.patient_id] for r in recordings])
    if target == "chest_location":
        locs = sorted({r.chest_location for r in recordings})
        lookup = {c: i for i, c in enumerate(locs)}
        return np.array([lookup[r.chest_location] for r in recordings])
    raise ValueError(f"Unknown probe target {target!r}")


def split_indices(recordings: list[Recording], test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Subject-wise split, returned as index arrays into the recording list."""
    position = {id(r): i for i, r in enumerate(recordings)}
    split = subject_wise_split(recordings, test_fraction=test_fraction, seed=seed)
    train = np.array([position[id(r)] for r in split["train"]])
    test = np.array([position[id(r)] for r in split["test"]])
    return train, test


def random_split_indices(n: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Recording-wise random split, ignoring patient identity.

    Included only so the subject-leakage effect can be *quantified* rather than
    asserted: the gap between this and the subject-wise split is the size of the
    optimism a naive protocol would buy you.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(n * test_fraction))
    return perm[cut:], perm[:cut]
