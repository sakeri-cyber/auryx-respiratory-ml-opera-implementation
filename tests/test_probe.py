"""Tests for linear probing.

`test_shuffled_control_lands_at_chance` and `test_scaler_is_fit_on_train_only`
are the ones that protect the validity of every reported number. A probe that
scores above chance on permuted labels means something leaks, and on datasets
this small a scaler fitted over train+test measurably inflates results.
"""

from __future__ import annotations

import numpy as np
import pytest

from respnet.probe import ProbeResult, linear_probe, shuffled_control


@pytest.fixture
def separable():
    """Two well-separated Gaussian blobs — a probe must nearly solve this."""
    rng = np.random.default_rng(0)
    n, d = 200, 32
    x0 = rng.normal(-1.5, 1.0, size=(n, d))
    x1 = rng.normal(+1.5, 1.0, size=(n, d))
    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]
    return x[:300], y[:300], x[300:], y[300:]


@pytest.fixture
def unlearnable():
    """Pure noise with random labels — a probe must not beat chance."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 32)).astype(np.float32)
    y = rng.integers(0, 2, size=400)
    return x[:300], y[:300], x[300:], y[300:]


class TestLinearProbe:
    def test_recovers_separable_signal(self, separable):
        result = linear_probe(*separable, name="separable", seeds=(0,))
        assert result.auroc_mean > 0.95
        assert result.balanced_acc_mean > 0.90

    def test_does_not_beat_chance_on_noise(self, unlearnable):
        result = linear_probe(*unlearnable, name="noise", seeds=(0,))
        assert result.auroc_mean < 0.65

    def test_reports_correct_class_count(self, separable):
        assert linear_probe(*separable, seeds=(0,)).n_classes == 2

    def test_multiclass_gives_no_auroc_but_gives_balanced_accuracy(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(300, 16)).astype(np.float32)
        y = rng.integers(0, 4, size=300)
        result = linear_probe(x[:200], y[:200], x[200:], y[200:], seeds=(0,))
        assert result.n_classes == 4
        assert np.isnan(result.auroc_mean)
        assert not np.isnan(result.balanced_acc_mean)

    def test_majority_baseline_is_inverse_class_count(self):
        """Balanced accuracy's chance level is 1/n_classes regardless of skew.

        Matters for the device probe: the raw device distribution is 70%
        AKGC417L, so plain accuracy would flatter a constant predictor.
        """
        rng = np.random.default_rng(3)
        x = rng.normal(size=(200, 8)).astype(np.float32)
        y = np.array([0] * 180 + [1] * 20)
        perm = rng.permutation(200)  # ensure both classes reach the training half
        x, y = x[perm], y[perm]
        assert linear_probe(x[:150], y[:150], x[150:], y[150:], seeds=(0,)).majority_baseline == pytest.approx(0.5)

    def test_single_class_training_split_raises_clearly(self):
        """An unstratified split on few patients can drop a class entirely."""
        x = np.random.randn(40, 8).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20)
        with pytest.raises(ValueError, match="only class"):
            linear_probe(x[:20], y[:20], x[20:], y[20:])

    def test_scaler_is_fit_on_train_only(self, separable):
        """Shifting the test features must change the score.

        If the scaler were fitted on train+test, a constant offset applied to the
        test half would be absorbed and the result would not move — which is
        exactly the leak this asserts against.
        """
        x_tr, y_tr, x_te, y_te = separable
        base = linear_probe(x_tr, y_tr, x_te, y_te, seeds=(0,)).auroc_mean
        shifted = linear_probe(x_tr, y_tr, x_te + 50.0, y_te, seeds=(0,)).auroc_mean
        assert base != pytest.approx(shifted)

    def test_mismatched_lengths_raise(self):
        x = np.random.randn(10, 4).astype(np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            linear_probe(x, np.zeros(9, dtype=int), x, np.zeros(10, dtype=int))

    def test_empty_test_split_raises(self):
        x = np.random.randn(10, 4).astype(np.float32)
        with pytest.raises(ValueError, match="Empty test split"):
            linear_probe(x, np.zeros(10, dtype=int), x[:0], np.zeros(0, dtype=int))

    def test_multiple_seeds_are_recorded(self, separable):
        result = linear_probe(*separable, seeds=(0, 1, 2))
        assert len(result.per_seed_balanced_acc) == 3


class TestShuffledControl:
    def test_lands_at_chance(self, separable):
        """The experiment's smoke alarm — must not detect signal in permuted labels."""
        result = shuffled_control(*separable, name="control")
        assert 0.35 < result.auroc_mean < 0.65


class TestProbeResult:
    def test_lift_over_chance(self):
        r = ProbeResult(name="x", n_classes=4, n_train=10, n_test=10,
                        balanced_acc_mean=0.625, majority_baseline=0.25)
        # (0.625 - 0.25) / (1 - 0.25) = 0.5
        assert r.lift_over_chance == pytest.approx(0.5)

    def test_lift_is_zero_at_chance(self):
        r = ProbeResult(name="x", n_classes=2, n_train=10, n_test=10,
                        balanced_acc_mean=0.5, majority_baseline=0.5)
        assert r.lift_over_chance == pytest.approx(0.0)

    def test_str_includes_the_headline_numbers(self):
        r = ProbeResult(name="probe", n_classes=2, n_train=100, n_test=50,
                        auroc_mean=0.9, auroc_std=0.01,
                        balanced_acc_mean=0.8, balanced_acc_std=0.02, majority_baseline=0.5)
        s = str(r)
        assert "0.900" in s and "0.800" in s
