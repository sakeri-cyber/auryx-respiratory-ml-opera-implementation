# Implementation Plan — OPERA Reproduction, Critique & Deployment Profile

**Constraints this plan is built around:** one week · free Colab GPU (T4, session limits, disconnects) · you write the code, not an AI · must close the audio and PyTorch gaps · must be defensible as your own understanding in an interview.

---

## 1. What this project is

Three components, in dependency order:

| # | Component | Whose code | Purpose |
|---|---|---|---|
| **A** | **Reproduce** OPERA's linear-probe result on one benchmark task | Their checkpoints, **your** extraction + probe + eval | Proves you can read a paper and land its number |
| **B** | **Baseline** — a small CNN trained from scratch on the same task | **100% yours, hand-written** | Closes the PyTorch gap properly. Answers a question the paper never asks |
| **C** | **Critique** — three experiments the paper omits | **Yours** | The original contribution. What makes this yours rather than homework |

**The critique experiments** (from `02-OPERA-deep-dive.md` §9):

- **C1 — Nuisance probe.** Can a linear probe recover the *recording device* from OPERA embeddings? Theory (§3.1) says the contrastive objective pressures the model to encode recording identity. If device is highly decodable, the representation carries a device fingerprint — the exact obstacle to "works on any earbud."
- **C2 — Deployment profile.** Latency, memory, model size, INT8/ONNX behaviour for OPERA-CT / CE / GT. Entirely unpublished.
- **C3 — Foundation model vs. your baseline.** The paper compares OPERA only against *other pretrained models*. Never against a well-tuned task-specific CNN. That's the practitioner's real question.

**C1 is the headline.** If it comes out strong it is genuinely novel, directly on Auryx's hardest problem, and costs almost nothing to run.

### Explicitly out of scope

- Pretraining anything (404 hours — impossible, and not the point)
- All 19 tasks (one, done properly)
- Beating their number (you're reproducing, not competing)
- Fine-tuning the encoder (linear probing is the paper's protocol; keep it)

---

## 2. The central design decision: don't use their environment

The OPERA repo installs via `conda env create` plus shell scripts that modify `~/.bashrc`. On Colab that will fight you, and you'll burn a day on dependency resolution.

**Instead: pull the checkpoints from HuggingFace and write your own inference path.**

```
Their contribution → pretrained weights (download)
Your contribution  → loading, preprocessing, feature extraction,
                     probing, evaluation, profiling
```

This costs perhaps half a day more than `pip install -e .`, and buys three things that matter more than the time:

1. **You understand the pipeline**, because you built it. That is the whole reason for doing this.
2. **You control preprocessing**, so you can verify your mel-spectrogram parameters match the paper (16 kHz, 64 mels, 64 ms window, 32 ms hop, 1×126×64) rather than trusting a config you never read.
3. **It runs in a notebook**, so no environment fights.

**The one thing you must borrow carefully:** their exact task definition for whichever task you reproduce — split, label mapping, aggregation. Read that code closely. If your task definition differs from theirs, your number is not comparable and the reproduction claim collapses. This is the single highest-risk detail in the project.

---

## 3. Task selection

**Primary target: T7 — COPD detection from ICBHI lung sounds.**

Why:
- ICBHI is small (920 recordings, 5.5 hours) and openly available.
- It's the anomaly task where AudioMAE beats OPERA (§7) — reproducing a *surprising* result is more interesting than reproducing an expected one.
- It has four recording devices, which is what makes **C1 possible at all**. No other openly-available benchmark dataset gives you a clean device variable.
- Patient IDs are in the filenames, so subject-wise splitting is trivial.

**Important:** T7 is **COPD detection at the patient level**, not the 4-class crackle/wheeze cycle classification of the original ICBHI challenge. ICBHI ships both label sets. Make sure you use the diagnosis file, not the cycle annotations. (The scaffold already on disk parses the *cycle* annotations — you'll need the patient diagnosis file too.)

**Fallback if ICBHI registration is slow:** Coswara and CoughVID are directly downloadable and both appear in the benchmark. You lose C1 (no clean device variable), so only fall back if you must. **Start the ICBHI registration today, before anything else.**

---

## 4. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ICBHI access delayed | Medium | Register **day 0**. Fallback to Coswara/CoughVID. |
| Can't match their exact T7 definition | Medium-high | Read their benchmark code before writing yours. If unmatched, report your number as "our protocol" and say so plainly — an honest non-match is fine, a silent one is not. |
| Colab disconnects mid-run | High | Extract features **once**, cache to Drive as `.npy`. Never recompute. Everything downstream runs on cached features in seconds. |
| Checkpoint loading fails | Low-medium | HF hosting is stated. If it breaks, Zenodo is the alternative. Check this on day 1, not day 4. |
| Preprocessing mismatch → garbage embeddings | Medium | Sanity check: embeddings of two crops from the same recording should be more similar than crops from different recordings. If not, preprocessing is wrong. |
| Scope creep | High | C1 is the priority. C2 and C3 are cuttable. B is cuttable. A is not. |

**The Colab caching point deserves emphasis.** Once features are extracted and saved, every experiment in this project is a linear probe over a small matrix — seconds, on CPU. The GPU is needed for perhaps 30 minutes total across the whole week. Structure the work so a disconnect costs you nothing.

---

## 5. Project layout

Reuse what's already on disk where it fits; discard the RespireNet-specific parts.

```
auryx-respiratory-ml/
├── docs/
│   ├── 01-OPERA-beginner.md
│   ├── 02-OPERA-deep-dive.md
│   ├── 03-implementation-plan.md
│   └── findings.md                 ← the write-up, grows daily
├── notebooks/
│   ├── 01_extract_features.ipynb   ← GPU, run once
│   └── 02_experiments.ipynb        ← CPU, run often
├── src/respnet/
│   ├── data/
│   │   ├── icbhi.py                ✅ exists — extend for diagnosis labels
│   │   └── preprocess.py           ⚠️ exists — retune to OPERA's params (16kHz, not 4kHz)
│   ├── models/
│   │   ├── opera.py                🆕 checkpoint loading + feature extraction
│   │   └── baseline_cnn.py         🆕 your from-scratch model (component B)
│   ├── probe.py                    🆕 linear probing + the nuisance probes
│   ├── metrics.py                  ✅ exists — add AUROC
│   └── profiling.py                ✅ exists — reuse as-is for C2
├── tests/                          ✅ ~60 tests exist; extend
└── artifacts/                      cached features, profile.json, figures
```

✅ = written, reusable · ⚠️ = written, needs adjustment · 🆕 = you write

**On `preprocess.py`:** it currently targets RespireNet's 4 kHz / 7-second cycles. OPERA uses **16 kHz, 64 mels, 64 ms window, 32 ms hop, 4-second clips → 1×126×64**. Retuning it is a genuinely useful exercise — you'll have to understand every parameter to change them correctly.

---

## 6. Day by day

### Day 0 (today, 30 minutes)
Register for ICBHI. Everything else waits on this.

### Day 1 — Understand + verify access
- Read the paper properly, with `02-OPERA-deep-dive.md` open alongside.
- Answer the five questions at the end of doc 01 in writing. If you can't, reread rather than proceed.
- **Verify a checkpoint downloads and loads.** One notebook cell, `print(model)`. Do this today — it de-risks the entire week.
- Skim their `src/benchmark/` for the T7 definition.

### Day 2 — Data + preprocessing
- Get ICBHI on disk, in Drive.
- Extend `icbhi.py` to parse patient diagnosis labels.
- Retune `preprocess.py` to OPERA's parameters. **Write a test asserting output shape is exactly (1, 126, 64) for 4s at 16kHz.**
- Sanity check: plot a few spectrograms. They should look like the paper's.

### Day 3 — Feature extraction (the GPU day)
- Write `opera.py`: load checkpoint, batch, extract embeddings.
- Extract for CT, CE and GT. **Cache all three to Drive.**
- **The similarity sanity check from §4.** If same-recording crops aren't more similar than different-recording crops, stop and fix preprocessing before going further.
- Everything after today runs on CPU.

### Day 4 — Component A: reproduce
- Write `probe.py`: single linear layer, 5 seeds, AUROC.
- Run T7. Compare to the paper.
- **Match, near-miss, or miss — all three are reportable.** Write down what you got and what you think explains any gap. A documented near-miss with a hypothesis is a better artifact than a suspiciously exact match.

### Day 5 — Component C1: the nuisance probe ⭐
The important day. Same frozen features, different labels:
- probe → **device** (4 classes: AKGC417L, LittC2SE, Litt3200, Meditron)
- probe → **subject identity**
- probe → **chest location**
- probe → **shuffled labels** (the control — this must come out at chance, and if it doesn't your probe is leaking)

Compare against the T7 accuracy. **If device is more decodable than pathology, that's your headline finding.**

### Day 6 — Components C2 + B
- C2: run `profiling.py` over the three encoders. Latency, memory, size, INT8, ONNX.
- B: your baseline CNN, if time survives. Small, hand-written, trained directly on the task.

### Day 7 — Write up
- `findings.md`: what you reproduced, what you found, what you'd do next, what you're uncertain about.
- Clean the repo. README. Make sure CI passes.

**Cut order if you fall behind:** B first, then C3, then C2. Never cut A or C1.

---

## 7. Experiment designs

### C1 — Nuisance probe (the one that matters)

**Hypothesis.** OPERA's contrastive positive pairs are two crops of the *same recording* (§3.1). Device, subject, and session are all constant within a recording, so they are perfectly predictive of a positive pair. The objective therefore *rewards* encoding them. Prediction: device and subject are highly linearly decodable from OPERA-CT and OPERA-CE embeddings, more so than from OPERA-GT (generative, no instance-discrimination pressure).

**Method.** Frozen embeddings. Multinomial logistic regression. Subject-wise splits for the device probe (so you're measuring *device*, not memorised subjects — this matters and is easy to get wrong). Report balanced accuracy against a majority-class baseline.

**Controls.** Shuffled-label probe must sit at chance. Report the class distribution — ICBHI's devices are unevenly used and an unbalanced probe will flatter you.

**Interpretation.**
- Device highly decodable → representation carries a device fingerprint → cross-device generalisation is at risk → **directly relevant to Auryx's core claim.**
- CT/CE more device-decodable than GT → supports the objective-driven explanation, not merely an architectural one.
- Nothing decodable → hypothesis wrong, and that's a real result too. Report it.

**Honest caveat to state in the write-up:** decodability of device does not automatically mean the *pathology* classifier is using device. It shows the information is present and linearly available, which is a necessary but not sufficient condition for shortcut learning. Don't overclaim — the necessary-condition framing is strong enough on its own.

### C2 — Deployment profile

Batch size 1 (a wearable classifies one clip at a time). CPU (the realistic target). 20 warm-up iterations discarded, 200 measured. **Median and IQR, not mean** — latency distributions are right-skewed. Record the host machine; a latency number without hardware is a rumour.

Report: params, disk size fp32, disk size INT8, median latency, p90, peak memory, ONNX export success. Across CT / CE / GT.

**The interesting axis:** OPERA-CE is ~4M params vs CT's 31M and *beats CT on lung-function tasks*. If CE is also dramatically faster, the practical recommendation for a wearable inverts the paper's headline. That's a real finding.

`profiling.py` already implements this. Read it before you run it — including the honest note about dynamic quantisation only covering Linear layers.

### C3 — Baseline comparison

Your CNN, trained directly on T7, subject-wise split, same metric. Question: does 404 hours of pretraining actually beat a small supervised model on this task? The paper never asks. Given T7 is where OPERA loses to AudioMAE, the answer might be uncomfortable — which makes it worth knowing.

---

## 8. What "done" looks like

A public repo containing:

1. **Working code** you can explain line by line.
2. **A reproduction result** with an honest account of any gap.
3. **At least one novel finding** (C1) that the paper does not report.
4. **`findings.md`** — 800–1200 words, plain, including what you're unsure about.
5. **Passing CI** and a test suite that includes the leakage tests.
6. **A README** stating clearly what's yours and what's theirs.

That last point is not optional. State plainly: *"Pretrained checkpoints are from the OPERA authors. Feature extraction, probing, evaluation, profiling and analysis are mine."*

### The one-line pitch this produces

> "I reproduced OPERA's linear-probe result on ICBHI COPD detection, then tested whether the embeddings encode recording device — they do, at [X]% — which suggests cross-device generalisation is a live risk for anyone deploying these representations on heterogeneous hardware. I also profiled the three encoders for on-device inference, which the paper doesn't report."

For a company whose product must work across every earbud on the market, that is a conversation-starter with their CTO.


