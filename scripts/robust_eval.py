#!/usr/bin/env python3
"""Re-run the probes with variance estimated from resampled splits.

The first pass reported std = 0.000 on every probe. That is not a sign of a
stable result — logistic regression on fixed data is deterministic, so varying
only the classifier's `random_state` varies nothing. This is precisely the
criticism `docs/02-OPERA-deep-dive.md` §6.1 makes of OPERA's own protocol, and
our first implementation reproduced it exactly.

This script resamples the *split* instead, which is where the real uncertainty
lives: with ~30 test patients, which patients land in the test half dominates.

Runs on cached features, so it takes seconds.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from respnet.data.icbhi import bandwidth_matched_subset, load_recordings  # noqa: E402
from respnet.pipeline import labels_for, random_split_indices, split_indices  # noqa: E402
from respnet.probe import multi_split_probe  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("robust")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SPLIT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
TEST_FRACTION = 0.25


def main() -> None:
    data_root = Path(sys.argv[1])
    recordings = load_recordings(
        data_root / "audio_and_txt_files", data_root / "patient_diagnosis.csv", probe_sample_rates=True
    )
    feats = np.load(ARTIFACTS / "features_operaGT.npy")
    assert len(recordings) == len(feats), "cached features do not match the recording index"

    out: dict = {"split_seeds": list(SPLIT_SEEDS), "test_fraction": TEST_FRACTION}

    logger.info("\n--- COPD (T7), subject-wise, resampled splits ---")
    y_copd = labels_for(recordings, "copd")
    out["copd_subject_wise"] = asdict(
        multi_split_probe(
            feats, y_copd,
            lambda s: split_indices(recordings, TEST_FRACTION, seed=s),
            name="COPD | subject-wise", split_seeds=SPLIT_SEEDS,
        )
    )

    logger.info("\n--- COPD, random (patient-leaky) split ---")
    out["copd_random"] = asdict(
        multi_split_probe(
            feats, y_copd,
            lambda s: random_split_indices(len(recordings), TEST_FRACTION, seed=s),
            name="COPD | random (leaky)", split_seeds=SPLIT_SEEDS,
        )
    )

    logger.info("\n--- Nuisance probes, naive ---")
    for target in ("device", "chest_location"):
        y = labels_for(recordings, target)
        out[f"{target}_naive"] = asdict(
            multi_split_probe(
                feats, y,
                lambda s: random_split_indices(len(recordings), TEST_FRACTION, seed=s),
                name=f"{target} | naive", split_seeds=SPLIT_SEEDS,
            )
        )

    logger.info("\n--- Device, bandwidth-matched ---")
    matched_ids = {r.stem for r in bandwidth_matched_subset(recordings)}
    keep = np.array([i for i, r in enumerate(recordings) if r.stem in matched_ids])
    recs_m = [recordings[i] for i in keep]
    feats_m = feats[keep]
    devices = sorted({r.device for r in recs_m})
    lookup = {d: i for i, d in enumerate(devices)}
    y_dev_m = np.array([lookup[r.device] for r in recs_m])

    out["device_bandwidth_matched"] = asdict(
        multi_split_probe(
            feats_m, y_dev_m,
            lambda s: random_split_indices(len(recs_m), TEST_FRACTION, seed=s),
            name="device | bandwidth-matched", split_seeds=SPLIT_SEEDS,
        )
    )
    out["device_bandwidth_matched"]["devices_present"] = devices

    logger.info("\n--- COPD on the bandwidth-matched subset ---")
    y_copd_m = np.array([r.is_copd for r in recs_m])
    out["copd_bandwidth_matched"] = asdict(
        multi_split_probe(
            feats_m, y_copd_m,
            lambda s: split_indices(recs_m, TEST_FRACTION, seed=s),
            name="COPD | bandwidth-matched", split_seeds=SPLIT_SEEDS,
        )
    )

    # Headline comparison: is device more decodable than the clinical target?
    dev = out["device_bandwidth_matched"]
    copd = out["copd_bandwidth_matched"]
    out["headline"] = {
        "device_lift_over_chance": dev["balanced_acc_mean"] - dev["majority_baseline"],
        "copd_lift_over_chance": copd["balanced_acc_mean"] - copd["majority_baseline"],
        "leakage_inflation_auroc": out["copd_random"]["auroc_mean"] - out["copd_subject_wise"]["auroc_mean"],
    }

    path = ARTIFACTS / "robust_results.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    logger.info("\nWrote %s", path)


if __name__ == "__main__":
    main()
