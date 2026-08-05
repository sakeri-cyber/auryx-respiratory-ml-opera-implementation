#!/usr/bin/env python3
"""Run every experiment and write results to artifacts/results.json.

Usage:
    PYTHONPATH=src python3 scripts/run_experiments.py --data-root "/path/to/Respiratory_Sound_Database"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from respnet.data.audio import AudioConfig  # noqa: E402
from respnet.data.icbhi import DEVICES, bandwidth_matched_subset, load_recordings, summarise  # noqa: E402
from respnet.models.baseline_cnn import BaselineCNN, train_baseline  # noqa: E402
from respnet.models.opera_gt import load_opera_gt  # noqa: E402
from respnet.pipeline import (  # noqa: E402
    build_features,
    build_spectrograms,
    labels_for,
    pick_device,
    random_split_indices,
    sanity_check_embeddings,
    split_indices,
)
from respnet.probe import linear_probe, shuffled_control  # noqa: E402
from respnet.profiling import profile_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run")

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TEST_FRACTION = 0.25
SEEDS = (0, 1, 2, 3, 4)


def banner(text: str) -> None:
    logger.info("\n%s\n%s\n%s", "=" * 78, text, "=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "checkpoints/encoder-operaGT.ckpt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    device = args.device or pick_device()
    cfg = AudioConfig()
    results: dict = {
        "config": {
            "audio": asdict(cfg),
            "device": device,
            "test_fraction": TEST_FRACTION,
            "seeds": list(SEEDS),
            "clip_seconds": cfg.clip_seconds,
        }
    }

    # ---------------------------------------------------------------- stage 0
    banner("STAGE 0 — Dataset")
    audio_dir = args.data_root / "audio_and_txt_files"
    recordings = load_recordings(audio_dir, args.data_root / "patient_diagnosis.csv", probe_sample_rates=True)
    stats = summarise(recordings)
    results["dataset"] = stats
    logger.info(json.dumps(stats, indent=2))

    # ---------------------------------------------------------------- stage 1
    banner("STAGE 1 — Spectrograms")
    specs = build_spectrograms(recordings, cfg, cache_path=ARTIFACTS / "spectrograms.pt")
    logger.info("Spectrogram tensor: %s", tuple(specs.shape))

    # ---------------------------------------------------------------- stage 2
    banner("STAGE 2 — OPERA-GT features")
    model = load_opera_gt(args.checkpoint)
    results["config"]["encoder_params"] = sum(p.numel() for p in model.parameters())
    feats = build_features(model, specs, cache_path=ARTIFACTS / "features_operaGT.npy", device=device)
    logger.info("Feature matrix: %s", feats.shape)

    results["sanity_check"] = sanity_check_embeddings(feats, recordings)

    # ---------------------------------------------------------------- A
    banner("EXPERIMENT A — T7 COPD detection (reproduction)")
    y_copd = labels_for(recordings, "copd")
    tr, te = split_indices(recordings, TEST_FRACTION, seed=0)

    a_subject = linear_probe(
        feats[tr], y_copd[tr], feats[te], y_copd[te], name="COPD | subject-wise split", seeds=SEEDS
    )
    rtr, rte = random_split_indices(len(recordings), TEST_FRACTION, seed=0)
    a_random = linear_probe(
        feats[rtr], y_copd[rtr], feats[rte], y_copd[rte], name="COPD | random split (leaky)", seeds=SEEDS
    )
    a_control = shuffled_control(feats[tr], y_copd[tr], feats[te], y_copd[te], name="COPD | shuffled control")

    results["experiment_A"] = {
        "subject_wise": asdict(a_subject),
        "random_split": asdict(a_random),
        "shuffled_control": asdict(a_control),
        "leakage_inflation_auroc": a_random.auroc_mean - a_subject.auroc_mean,
        "opera_paper_reported_auroc": None,
        "note": (
            "NOT COMPARABLE to the published T7 number, and no published value is quoted "
            "here because we have not read it from the paper's table. Our protocol differs "
            "in at least: clip selection (one centred 8.2s clip vs their unspecified "
            "scheme), pooling (mean patch tokens), and split construction (our own "
            "stratified subject-wise splits). Matching their src/benchmark/ T7 definition "
            "is prerequisite to any comparison. See docs/08-architecture.md section 5."
        ),
    }

    # ---------------------------------------------------------------- C1
    banner("EXPERIMENT C1 — Nuisance probes (device / patient / chest location)")
    c1: dict = {"naive": {}, "bandwidth_controlled": {}}

    for target in ("device", "chest_location", "patient"):
        y = labels_for(recordings, target)
        # Device and site are properties of the *recording*, not the patient, so a
        # random split is the correct protocol for them. Patient identity obviously
        # requires a split where the same patient appears on both sides.
        itr, ite = random_split_indices(len(recordings), TEST_FRACTION, seed=0)
        res = linear_probe(feats[itr], y[itr], feats[ite], y[ite], name=f"{target} | naive", seeds=SEEDS)
        c1["naive"][target] = asdict(res)

    c1["naive"]["shuffled_control"] = asdict(
        shuffled_control(
            feats[rtr], labels_for(recordings, "device")[rtr],
            feats[rte], labels_for(recordings, "device")[rte],
            name="device | shuffled control",
        )
    )

    # Bandwidth-controlled arm: drop natively-low-rate recordings so every remaining
    # file shares an effective bandwidth. Without this, Litt3200 (all 4 kHz native)
    # is identifiable from its empty upper spectrum alone.
    banner("EXPERIMENT C1b — Bandwidth-controlled device probe")
    matched = bandwidth_matched_subset(recordings)
    matched_ids = {r.stem for r in matched}
    keep = np.array([i for i, r in enumerate(recordings) if r.stem in matched_ids])
    feats_m = feats[keep]
    recs_m = [recordings[i] for i in keep]

    present_devices = sorted({r.device for r in recs_m})
    lookup = {d: i for i, d in enumerate(present_devices)}
    y_dev_m = np.array([lookup[r.device] for r in recs_m])

    mtr, mte = random_split_indices(len(recs_m), TEST_FRACTION, seed=0)
    res_m = linear_probe(
        feats_m[mtr], y_dev_m[mtr], feats_m[mte], y_dev_m[mte],
        name="device | bandwidth-matched", seeds=SEEDS,
    )
    c1["bandwidth_controlled"]["device"] = asdict(res_m)
    c1["bandwidth_controlled"]["devices_present"] = present_devices
    c1["bandwidth_controlled"]["n_recordings"] = len(recs_m)

    # Does COPD detection survive on the same matched subset?
    y_copd_m = np.array([r.is_copd for r in recs_m])
    mtr2, mte2 = split_indices(recs_m, TEST_FRACTION, seed=0)
    c1["bandwidth_controlled"]["copd"] = asdict(
        linear_probe(
            feats_m[mtr2], y_copd_m[mtr2], feats_m[mte2], y_copd_m[mte2],
            name="COPD | bandwidth-matched", seeds=SEEDS,
        )
    )
    results["experiment_C1"] = c1

    # ---------------------------------------------------------------- C2
    banner("EXPERIMENT C2 — Deployment profile")
    example = torch.randn(1, 1, cfg.n_mels, cfg.n_frames)
    results["experiment_C2"] = profile_report(
        model, example, out_path=ARTIFACTS / "profile.json", include_quantized=True
    )

    # ---------------------------------------------------------------- C3
    if not args.skip_baseline:
        banner("EXPERIMENT C3 — Supervised CNN baseline")
        x_tr = specs[tr].unsqueeze(1)
        x_te = specs[te].unsqueeze(1)
        y_tr = torch.from_numpy(y_copd[tr]).long()
        y_te = torch.from_numpy(y_copd[te]).long()

        cnn, history = train_baseline(
            x_tr, y_tr, x_te, y_te, epochs=30, device=device, seed=0, log_every=5
        )
        final = [v for v in history["test_auroc"][-5:] if not np.isnan(v)]
        results["experiment_C3"] = {
            "n_parameters": cnn.n_parameters(),
            "encoder_parameters": results["config"]["encoder_params"],
            "final_test_auroc_mean_last5": float(np.mean(final)) if final else float("nan"),
            "best_test_auroc": float(np.nanmax(history["test_auroc"])),
            "history": history,
            "opera_probe_auroc": a_subject.auroc_mean,
        }
        logger.info(
            "Baseline CNN (%s params) best AUROC %.3f vs OPERA-GT probe %.3f",
            f"{cnn.n_parameters():,}",
            results["experiment_C3"]["best_test_auroc"],
            a_subject.auroc_mean,
        )

    # ---------------------------------------------------------------- save
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    banner(f"Wrote {out}")


if __name__ == "__main__":
    main()
