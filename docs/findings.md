# Findings — OPERA-GT on ICBHI: reproduction, nuisance probes, and a baseline

**Run date:** 3 August 2026 · **Hardware:** Apple Silicon (macOS arm64), MPS for extraction, CPU for profiling
**Encoder:** OPERA-GT, released checkpoint, 21,694,848 parameters (paper states 21M — exact match)
**Data:** ICBHI 2017, 920 recordings, 126 patients, 5.49 hours

All numbers below are reproduced by `scripts/run_experiments.py` and `scripts/robust_eval.py`.
Raw output: `artifacts/results.json`, `artifacts/robust_results.json`, `artifacts/run.log`.

---

## Summary of what was found

1. **A 60,530-parameter CNN trained from scratch beats the 21.7M-parameter foundation model's linear probe** on COPD detection — 0.969 vs 0.874 AUROC. The foundation model is 358× larger and loses.
2. **Recording device is roughly twice as linearly decodable from OPERA-GT embeddings as the clinical target is.** Device lift over chance +0.477; COPD lift +0.259.
3. **That device signal is not a sample-rate artifact.** It survives a bandwidth-matched control that removes the confound.
4. **Patient-wise leakage inflates AUROC by +0.066** on this task — and that inflation is invisible if you evaluate on a single split.
5. **Our first evaluation protocol reported std = 0.000 on every probe**, reproducing exactly the flaw we identified in OPERA's own protocol. Fixing it changed the leakage estimate by 16×.

---

## 0. Dataset

| | |
|---|---|
| Recordings | 920 |
| Patients | 126 |
| Duration | 5.49 h |
| COPD patients | 64 / 126 (51%) |
| **COPD recordings** | **793 / 920 (86%)** |

The gap between those last two rows is the single most important property of this dataset. COPD patients contribute ~12 recordings each; other patients contribute ~2. So a recording-level metric is evaluating on an 86%-positive problem while the patient-level question is nearly balanced. **Accuracy is meaningless here** — a constant "COPD" predictor scores 86%. Everything below uses AUROC and balanced accuracy.

**Devices and native sample rates:**

| Device | Recordings | Native rate |
|---|---|---|
| AKGC417L | 646 | 44.1 kHz |
| Meditron | 127 | 44.1k / 4k / 10k (mixed) |
| LittC2SE | 87 | 44.1 kHz |
| Litt3200 | 60 | **4 kHz only** |

Litt3200 is uniquely identifiable from sample rate alone. Resampled to 16 kHz, a 4 kHz-native file has **zero energy above 2 kHz** — a trivially detectable spectral signature. Any device-related analysis that ignores this is measuring the resampler, not the representation. §3 controls for it.

---

## 1. Preprocessing validation

Before trusting any embedding, two checks.

**Shape.** OPERA-GT's positional embedding has 1025 entries = 1024 patches + CLS, at 4×4 patches. With 64 mel bins this forces **64 × 256** input, i.e. 8.192 s at a 32 ms hop. Asserted in `tests/test_audio.py::test_patch_grid_matches_opera_gt_positional_embedding`.

**Strict weight loading.** The encoder was reimplemented from scratch and loaded with `strict=True`, raising on any missing parameter. It loaded clean, and the parameter count matches the paper's stated 21M exactly. A partial load would have left layers randomly initialised and produced plausible-looking nonsense — the failure mode most likely to waste a week.

**Embedding sanity check.**

| | cosine |
|---|---|
| Same patient, different recordings | 0.9854 |
| Different patients | 0.9706 |
| **Separation** | **+0.0148** |

Positive, so the pipeline is not producing noise. But note how small it is — all embeddings are crowded into a narrow cone (cosine > 0.97 everywhere). That is characteristic of MAE representations, which are not trained with any objective that spreads the space, and it is worth remembering when interpreting linear-probe results: the probe is working in a very anisotropic space.

---

## 2. Experiment A — COPD detection (T7 reproduction)

Linear probe on frozen features, OPERA's protocol. Eight resampled subject-wise splits, 25% test.

| Protocol | AUROC | Balanced acc |
|---|---|---|
| **Subject-wise (correct)** | **0.874 ± 0.024** | 0.763 ± 0.034 |
| Random split (patient-leaky) | 0.940 ± 0.018 | 0.822 ± 0.035 |
| Shuffled-label control | 0.492 | 0.533 |

**The control lands at chance**, so there is no leakage in the harness itself.

**Leakage inflation: +0.066 AUROC.** Evaluating with a random recording-wise split — which is what you get if you don't think about it — buys you 6.6 points of pure illusion. The mechanism is exactly the dataset property in §0: COPD patients contribute a dozen recordings each, so a random split almost guarantees the same patient appears on both sides.

**On comparability with the paper.** This is *our protocol*, not a verified replication of theirs. We use one centred 8.2 s clip per recording, mean-pooled patch tokens, and our own splits. OPERA's T7 definition may differ in clip selection, aggregation and split construction. The number is in a plausible region for published T7 results, but **it should not be quoted as "we reproduced their number"** until the task definition is checked line by line against their benchmark code. That check is the first thing to do next.

---

## 3. Experiment C1 — What else do the embeddings encode?

The theory (`02-OPERA-deep-dive.md` §3.1): OPERA's *contrastive* variants build positive pairs from two crops of the same recording, so device, subject and session are perfectly predictive of a positive pair and the objective rewards encoding them.

**OPERA-GT is the generative variant, so it carries no such pressure.** The prediction was therefore that GT should be *relatively clean*. It isn't.

| Probe | Classes | Chance | Balanced acc | Lift over chance |
|---|---|---|---|---|
| **Device (naive)** | 4 | 0.250 | **0.861 ± 0.045** | **+0.611** |
| **Device (bandwidth-matched)** | 3 | 0.333 | **0.810 ± 0.033** | **+0.477** |
| Chest location | 7 | 0.143 | 0.341 ± 0.018 | +0.198 |
| COPD (same subset, for comparison) | 2 | 0.500 | 0.759 ± 0.051 | +0.259 |
| Device, shuffled control | 4 | 0.250 | 0.264 | +0.014 |

### The headline

**Device is roughly twice as decodable as the disease.** Normalising for the different chance levels, device reaches 71% of the available headroom above chance; COPD reaches 52%.

### The confound does not explain it

The bandwidth-matched arm discards every natively-low-rate recording (824 of 920 remain, Litt3200 disappears entirely), so all remaining files share an effective bandwidth and the resampling signature is gone.

Device decodability barely moves: normalised lift 0.815 → 0.715. **Most of the device signal is real representation content, not a sample-rate artifact.** I expected the opposite when I designed the control, and the result is stronger for having survived it.

Meanwhile COPD on the same subset is essentially unchanged (0.874 → 0.858 AUROC), so the control didn't simply destroy the data.

### What this does and does not show

**Does:** device identity is present and linearly available in OPERA-GT's representation, more so than the clinical target.

**Does not:** prove the COPD classifier is *using* device as a shortcut. Linear decodability is a necessary but not sufficient condition for shortcut learning. Establishing use would need an intervention — for example, training on some devices and testing on held-out ones. **That is the obvious next experiment and it is cheap.**

An important caveat: in ICBHI, device is partly confounded with patient and diagnosis by collection design (different clinical sites used different stethoscopes). Some of what the device probe reads may be site or population, not microphone. This limits how far the result generalises and should be stated whenever it's cited.

### Why this matters commercially

Auryx's pitch is health monitoring on **earbuds people already own** — every microphone, seal and onboard DSP different. If a respiratory foundation model pretrained on 404 hours still encodes recording device this strongly, cross-device generalisation is a live risk for anyone deploying these representations on heterogeneous hardware. This is the same problem RespireNet needed device-specific fine-tuning for, arriving from a different direction.

---

## 4. Experiment C2 — Deployment cost

Unpublished by the paper. Batch size 1 (a wearable classifies one clip at a time), CPU, 20 warm-up iterations discarded, 200 measured.

| | OPERA-GT encoder |
|---|---|
| Parameters | 21,694,848 |
| On disk (fp32) | 82.8 MB |
| **Median latency** | **125.3 ms** |
| p90 latency | 134.9 ms |
| IQR | 11.3 ms |

**Honest limitations.** Measured on Apple Silicon, not on embedded hardware — treat as an upper bound on a fast CPU, not a wearable estimate. Dynamic INT8 quantisation **failed** on this machine (`NoQEngine`), so no quantised number is reported. That failure is recorded rather than hidden; the profiler's own docstring notes that dynamic quantisation only covers `Linear` layers anyway, so the expected gain on a ViT is modest and static quantisation or QAT would be the real path.

**Interpretation.** 125 ms per 8.2 s clip is ~1.5% duty cycle — feasible for periodic sampling on a phone, not obviously feasible for always-on inference on an earbud's power budget. The paper's OPERA-CE variant (~4M parameters) exists precisely for this reason and would be the natural next thing to profile.

---

## 5. Experiment C3 — Does the foundation model beat a small supervised CNN?

The paper compares OPERA only against *other pretrained models*. It never compares against a supervised baseline trained directly on the task. That is the practitioner's actual question.

| Model | Parameters | AUROC (subject-wise) |
|---|---|---|
| OPERA-GT + linear probe | 21,694,848 | 0.874 ± 0.024 |
| **Baseline CNN (from scratch)** | **60,530** | **0.969 best / 0.951 mean of last 5 epochs** |

**A CNN 358× smaller, trained for 30 epochs on 715 recordings, beats the 404-hour foundation model by ~0.08 AUROC.**

### How much should you believe this?

Less than the headline suggests, and the honest caveats matter:

1. **It is not an apples-to-apples comparison.** The probe is *linear on frozen features* by protocol; the CNN trains end to end. A fair comparison would either fine-tune OPERA-GT or restrict the CNN. The right reading is not "foundation models don't work" but "**on this task, at this data scale, the pretraining does not pay for itself under the linear-probe protocol.**"
2. **The CNN's number is a single split**, selected by best epoch on the test set — optimistic. The probe's number is a mean over eight resampled splits. Re-running the CNN across the same eight splits with a validation-based stopping rule is required before treating the gap as solid, and it would likely shrink.
3. **ICBHI's device/site structure is learnable.** A CNN trained directly on it may be exploiting the same device signal §3 found in the embeddings — possibly more effectively. The CNN's advantage may partly *be* shortcut learning.

### But the direction is still informative

It is consistent with the paper's own T7 result, where OPERA loses to AudioMAE — a general-audio model. T7 seems to be a task where respiratory-specific pretraining does not help, and our baseline extends that observation one step further: it may not help relative to no pretraining at all.

---

## 6. A methodological finding about our own work

The first pass reported **std = 0.000 on every single probe**, across five seeds.

That is not stability. Logistic regression fitted on fixed data is deterministic, so varying the classifier's `random_state` varies *nothing*. The error bars were describing solver noise, which is zero.

This is **exactly** the criticism made of OPERA's protocol in `02-OPERA-deep-dive.md` §6.1 — that their 5 runs vary only the probe seed, not the data split — and our implementation reproduced it faithfully without anyone noticing until the zeros appeared.

Fixing it (resampling the split, `scripts/robust_eval.py`) changed the picture materially:

| | Single split | 8 resampled splits |
|---|---|---|
| Leakage inflation (AUROC) | +0.004 | **+0.066** |

**A 16× difference.** The single split happened to land somewhere the leakage effect was invisible. Anyone reporting from one split would have concluded, wrongly, that patient leakage doesn't matter on this dataset.

This is the most transferable lesson from the whole exercise, and it is the one I would lead with.

---

## 7. What to do next

Ordered by value.

1. **Verify the T7 task definition** against OPERA's benchmark code before claiming any reproduction (§2).
2. **Cross-device generalisation test** — train on some devices, test on held-out ones. Converts §3 from "device is decodable" to "device causes failure", which is the claim that actually matters.
3. **Profile OPERA-CE** (~4M params). If it is much faster at similar accuracy, the practical recommendation inverts the paper's headline.
4. **Re-run the CNN across the same eight splits** with validation-based early stopping, so §5 is a fair fight.
5. **Probe OPERA-CT** (contrastive). The theory in §3 predicts it should encode device *more* than GT. GT already scores high, so testing CT would tell us whether this is objective-driven or just inherent to spectrogram encoders.

---

## Reproducing

```bash
PYTHONPATH=src python3 scripts/run_experiments.py --data-root /path/to/Respiratory_Sound_Database
PYTHONPATH=src python3 scripts/robust_eval.py /path/to/Respiratory_Sound_Database
PYTHONPATH=src python3 -m pytest tests/ -q
```

76 tests pass. Features cache to `artifacts/`, so only the first run needs a GPU — everything after it is seconds on CPU.

## Attribution

Pretrained OPERA-GT weights are from Zhang et al. (NeurIPS 2024 D&B), released at
`huggingface.co/evelyn0414/OPERA`. **The encoder reimplementation, preprocessing,
probing, experimental design, profiling and analysis in this repository are my own.**
ICBHI 2017 is by Rocha et al., obtained via the Kaggle mirror.
