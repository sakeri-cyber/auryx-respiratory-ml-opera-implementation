"""Linear probing — the OPERA evaluation protocol, plus the nuisance-variable probes.

Two uses, same machinery:

1. **Task probe (component A).** OPERA's protocol: a single fully-connected layer
   on frozen features. Deliberately weak, so that good performance is evidence
   about the representation rather than the classifier.

2. **Nuisance probe (component C1).** The same probe pointed at variables that
   *should not* matter — recording device, patient identity, chest location. If
   device is highly decodable, the representation carries a device fingerprint.

The theoretical motivation for (2) is in `docs/02-OPERA-deep-dive.md` §3.1: OPERA's
contrastive variants build positive pairs from two crops of the *same recording*,
so device, subject and session are perfectly predictive of a positive pair and the
objective rewards encoding them. OPERA-GT is generative and carries no such
pressure, which makes CT-vs-GT the natural comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProbeResult:
    """Outcome of one probe, aggregated over seeds."""

    name: str
    n_classes: int
    n_train: int
    n_test: int
    auroc_mean: float = float("nan")
    auroc_std: float = float("nan")
    balanced_acc_mean: float = float("nan")
    balanced_acc_std: float = float("nan")
    majority_baseline: float = float("nan")
    per_seed_auroc: list[float] = field(default_factory=list)
    per_seed_balanced_acc: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        if self.n_classes == 2:
            head = f"AUROC {self.auroc_mean:.3f}±{self.auroc_std:.3f}"
        else:
            head = f"{self.n_classes}-class"
        return (
            f"{self.name:<38} {head}  "
            f"bal-acc {self.balanced_acc_mean:.3f}±{self.balanced_acc_std:.3f}  "
            f"(chance {self.majority_baseline:.3f}, n={self.n_train}/{self.n_test})"
        )

    @property
    def lift_over_chance(self) -> float:
        """Balanced accuracy above chance, as a fraction of the available headroom."""
        headroom = 1.0 - self.majority_baseline
        if headroom <= 0:
            return float("nan")
        return (self.balanced_acc_mean - self.majority_baseline) / headroom


def linear_probe(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    name: str = "probe",
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    max_iter: int = 2000,
    C: float = 1.0,
) -> ProbeResult:
    """Fit a multinomial logistic regression on frozen features.

    Features are standardised using statistics computed on the **training split
    only** — fitting the scaler on all data would leak test statistics into
    training, which on datasets this small measurably inflates results.

    Balanced accuracy is reported alongside AUROC because ICBHI is imbalanced on
    every label we probe. Plain accuracy would be actively misleading: the device
    label is 70% AKGC417L, so a constant predictor scores 0.70.
    """
    if features_train.shape[0] != labels_train.shape[0]:
        raise ValueError(
            f"Feature/label mismatch: {features_train.shape[0]} vs {labels_train.shape[0]}"
        )
    if features_test.shape[0] == 0:
        raise ValueError("Empty test split")
    if len(np.unique(labels_train)) < 2:
        raise ValueError(
            f"Training split contains only class {np.unique(labels_train)[0]}. "
            "With few patients an unstratified split can drop a class entirely — "
            "use stratify_by_label=True or a different seed."
        )

    classes = np.unique(np.concatenate([labels_train, labels_test]))
    n_classes = len(classes)

    # Chance = the majority-class rate under balanced accuracy, which is 1/n_classes
    # for a balanced-accuracy metric regardless of skew. Reporting it explicitly
    # stops a skewed 4-class device probe from looking impressive at 0.70.
    majority = 1.0 / n_classes

    aurocs: list[float] = []
    baccs: list[float] = []

    for seed in seeds:
        scaler = StandardScaler().fit(features_train)
        x_tr = scaler.transform(features_train)
        x_te = scaler.transform(features_test)

        clf = LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=seed,
            multi_class="auto",
            class_weight="balanced",
        )
        clf.fit(x_tr, labels_train)

        pred = clf.predict(x_te)
        baccs.append(float(balanced_accuracy_score(labels_test, pred)))

        if n_classes == 2 and len(np.unique(labels_test)) == 2:
            score = clf.predict_proba(x_te)[:, 1]
            aurocs.append(float(roc_auc_score(labels_test, score)))

    result = ProbeResult(
        name=name,
        n_classes=n_classes,
        n_train=len(labels_train),
        n_test=len(labels_test),
        auroc_mean=float(np.mean(aurocs)) if aurocs else float("nan"),
        auroc_std=float(np.std(aurocs)) if aurocs else float("nan"),
        balanced_acc_mean=float(np.mean(baccs)),
        balanced_acc_std=float(np.std(baccs)),
        majority_baseline=majority,
        per_seed_auroc=aurocs,
        per_seed_balanced_acc=baccs,
    )
    logger.info("%s", result)
    return result


def multi_split_probe(
    features: np.ndarray,
    labels: np.ndarray,
    split_fn,
    name: str = "probe",
    split_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> ProbeResult:
    """Probe across several *independently resampled splits*, not just probe seeds.

    This exists because of a flaw we found in our own first implementation, which
    is the same flaw `docs/02-OPERA-deep-dive.md` §6.1 identifies in OPERA's
    protocol. Logistic regression fitted on fixed data is deterministic, so
    varying only the classifier's `random_state` produces **exactly zero
    variance** — we measured std = 0.000 across five seeds.

    Reported error bars from that procedure describe solver noise, not sampling
    uncertainty, and are therefore far narrower than the true uncertainty. With
    32 test patients the dominant error source is *which patients landed in the
    test split*, which only resampling the split can capture.

    `split_fn(seed) -> (train_idx, test_idx)`.
    """
    aurocs: list[float] = []
    baccs: list[float] = []
    n_train = n_test = 0

    for seed in split_seeds:
        train_idx, test_idx = split_fn(seed)
        single = linear_probe(
            features[train_idx], labels[train_idx],
            features[test_idx], labels[test_idx],
            name=f"{name} [split {seed}]", seeds=(0,),
        )
        if not np.isnan(single.auroc_mean):
            aurocs.append(single.auroc_mean)
        baccs.append(single.balanced_acc_mean)
        n_train, n_test = single.n_train, single.n_test

    n_classes = len(np.unique(labels))
    result = ProbeResult(
        name=name,
        n_classes=n_classes,
        n_train=n_train,
        n_test=n_test,
        auroc_mean=float(np.mean(aurocs)) if aurocs else float("nan"),
        auroc_std=float(np.std(aurocs)) if aurocs else float("nan"),
        balanced_acc_mean=float(np.mean(baccs)),
        balanced_acc_std=float(np.std(baccs)),
        majority_baseline=1.0 / n_classes,
        per_seed_auroc=aurocs,
        per_seed_balanced_acc=baccs,
    )
    logger.info("%s  <- across %d resampled splits", result, len(split_seeds))
    return result


def shuffled_control(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    name: str = "shuffled control",
    seed: int = 0,
) -> ProbeResult:
    """Probe against randomly permuted labels. Must land at chance.

    This is the experiment's smoke alarm. If a shuffled-label probe scores above
    chance, something leaks — the scaler, the split, or duplicated rows — and every
    other number in the run is untrustworthy. Cheap to run, and it has caught real
    bugs in published work.
    """
    rng = np.random.default_rng(seed)
    return linear_probe(
        features_train,
        rng.permutation(labels_train),
        features_test,
        rng.permutation(labels_test),
        name=name,
        seeds=(seed,),
    )
