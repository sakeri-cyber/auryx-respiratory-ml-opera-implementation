"""Tests for ICBHI indexing, splitting and the bandwidth control.

`test_no_patient_appears_in_both_splits` is the important one. Patient-level
leakage is the standard way respiratory-sound results get silently inflated, and
it matters more than usual here: COPD patients contribute ~12 recordings each
(793 COPD recordings from 64 patients), so a recording-wise split would let the
model memorise a patient in training and be graded on that same patient at test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from respnet.data.icbhi import (
    DEVICES,
    Recording,
    bandwidth_matched_subset,
    subject_wise_split,
    summarise,
    _parse_stem,
)


def make(patient_id: int, diagnosis: str = "Healthy", device: str = "Meditron", rate: int | None = 44100) -> Recording:
    return Recording(
        patient_id=patient_id,
        stem=f"{patient_id}_1b1_Al_sc_{device}",
        wav_path=Path(f"/nonexistent/{patient_id}_{device}.wav"),
        device=device,
        chest_location="Al",
        mode="sc",
        diagnosis=diagnosis,
        native_sample_rate=rate,
    )


@pytest.fixture
def recordings() -> list[Recording]:
    """40 patients, 5 recordings each; half COPD."""
    out = []
    for p in range(100, 140):
        dx = "COPD" if p % 2 == 0 else "Healthy"
        for _ in range(5):
            out.append(make(p, diagnosis=dx))
    return out


class TestParseStem:
    def test_valid_stem(self):
        assert _parse_stem("101_1b1_Al_sc_Meditron") == (101, "Al", "sc", "Meditron")

    @pytest.mark.parametrize("bad", ["101_1b1_Al_sc", "101_1b1_Al_sc_Meditron_extra", "junk"])
    def test_malformed_stem_raises(self, bad):
        with pytest.raises(ValueError, match="Expected 5 fields"):
            _parse_stem(bad)


class TestRecording:
    def test_is_copd(self):
        assert make(1, "COPD").is_copd == 1
        assert make(1, "Healthy").is_copd == 0
        assert make(1, "Asthma").is_copd == 0

    def test_device_index_matches_registry(self):
        for i, device in enumerate(DEVICES):
            assert make(1, device=device).device_index == i


class TestSubjectWiseSplit:
    def test_no_patient_appears_in_both_splits(self, recordings):
        split = subject_wise_split(recordings, test_fraction=0.25, seed=7)
        train = {r.patient_id for r in split["train"]}
        test = {r.patient_id for r in split["test"]}
        assert train.isdisjoint(test), f"Patient leakage: {sorted(train & test)}"

    def test_every_recording_assigned_exactly_once(self, recordings):
        split = subject_wise_split(recordings, test_fraction=0.2, seed=0)
        assert len(split["train"]) + len(split["test"]) == len(recordings)

    def test_deterministic_for_a_seed(self, recordings):
        a = subject_wise_split(recordings, seed=42)
        b = subject_wise_split(recordings, seed=42)
        assert [r.stem for r in a["test"]] == [r.stem for r in b["test"]]

    def test_different_seeds_differ(self, recordings):
        a = subject_wise_split(recordings, seed=1)
        b = subject_wise_split(recordings, seed=2)
        assert {r.patient_id for r in a["test"]} != {r.patient_id for r in b["test"]}

    def test_stratification_preserves_class_balance(self, recordings):
        """Both halves must contain COPD and non-COPD patients.

        With only 126 patients an unstratified draw can produce a test split with
        one class absent, which makes AUROC undefined.
        """
        split = subject_wise_split(recordings, test_fraction=0.25, seed=3, stratify_by_label=True)
        for half in ("train", "test"):
            labels = {r.is_copd for r in split[half]}
            assert labels == {0, 1}, f"{half} split missing a class"

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
    def test_invalid_fraction_raises(self, recordings, bad):
        with pytest.raises(ValueError, match="test_fraction"):
            subject_wise_split(recordings, test_fraction=bad)


class TestBandwidthMatchedSubset:
    def test_keeps_only_the_modal_sample_rate(self):
        recs = [make(i, device="AKGC417L", rate=44100) for i in range(10)]
        recs += [make(100 + i, device="Litt3200", rate=4000) for i in range(3)]
        kept = bandwidth_matched_subset(recs)
        assert len(kept) == 10
        assert all(r.native_sample_rate == 44100 for r in kept)

    def test_removes_the_device_that_is_uniquely_low_rate(self):
        """Litt3200 is 100% 4 kHz in ICBHI, so controlling for bandwidth removes it."""
        recs = [make(i, device="AKGC417L", rate=44100) for i in range(10)]
        recs += [make(100 + i, device="Litt3200", rate=4000) for i in range(3)]
        assert "Litt3200" not in {r.device for r in bandwidth_matched_subset(recs)}

    def test_raises_when_rates_were_not_probed(self):
        with pytest.raises(ValueError, match="probe_sample_rates"):
            bandwidth_matched_subset([make(1, rate=None)])


class TestSummarise:
    def test_counts_are_consistent(self, recordings):
        s = summarise(recordings)
        assert s["n_recordings"] == len(recordings)
        assert s["n_patients"] == 40
        assert sum(s["by_device"].values()) == len(recordings)

    def test_patient_and_recording_copd_counts_differ_when_unbalanced(self):
        """One COPD patient with many recordings must not read as many COPD patients."""
        recs = [make(1, "COPD") for _ in range(10)] + [make(2, "Healthy")]
        s = summarise(recs)
        assert s["copd_patients"] == 1
        assert s["copd_recordings"] == 10
