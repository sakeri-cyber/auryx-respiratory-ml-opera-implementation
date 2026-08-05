# OPERA-GT on ICBHI — a reimplementation, and three experiments the paper doesn't run

A from-scratch reimplementation of the **OPERA-GT** respiratory audio foundation-model
encoder ([Zhang et al., *Towards Open Respiratory Acoustic Foundation Models*, NeurIPS 2024
Datasets & Benchmarks](https://arxiv.org/abs/2406.16148)), evaluated on ICBHI 2017 COPD
detection — plus a nuisance-variable probe, a deployment profile, and a from-scratch
baseline, none of which appear in the paper.

> **What this is, precisely:** a *verified reimplementation of the OPERA-GT encoder* and an
> *independent evaluation of it on one dataset*. It is **not** a replication of the OPERA
> benchmark — see [Honest scope](#honest-scope). Independent study, ~1 week, one laptop.

---

## Headline results

| Finding | Number |
|---|---|
| A 60,530-param CNN beats the 21.7M-param foundation-model probe | **0.969 vs 0.874** AUROC |
| Recording **device** is more decodable than the disease | lift **0.715** vs **0.518** |
| …and it survives a bandwidth-matched control (not a sample-rate artifact) | 0.815 → **0.715** |
| Patient-wise leakage inflates AUROC | **+0.066** |
| Encoder inference cost — unreported by the paper | **125 ms** / clip, **83 MB** |

Full results with every caveat: **[`docs/findings.md`](docs/findings.md)**.

---

## What's mine and what's theirs

**Theirs.** The pretrained OPERA-GT weights (`huggingface.co/evelyn0414/OPERA`) and the
ICBHI 2017 dataset (Rocha et al.).

**Mine.** The encoder reimplementation, the audio front end, feature extraction, probing,
the baseline, profiling, the experimental design, and all analysis. The encoder is written
from the checkpoint's tensor shapes with **no `timm`, `torchvision` or `torchaudio`**; it
loads the released weights with `strict=True` and matches the paper's stated **21M**
parameter count exactly (21,694,848).

---

## Documentation

This repo is as much a write-up as it is code. Suggested reading order:

### Understand the paper
| Doc | What it covers |
|---|---|
| [`docs/01-OPERA-beginner.md`](docs/01-OPERA-beginner.md) | OPERA explained from scratch — no background assumed |
| [`docs/02-OPERA-deep-dive.md`](docs/02-OPERA-deep-dive.md) | The maths, the design choices, and where the evaluation is weaker than it looks |

### The wider context (the lab's in-ear work)
| Doc | What it covers |
|---|---|
| [`docs/04-RespEar-beginner.md`](docs/04-RespEar-beginner.md) · [`05-deep-dive`](docs/05-RespEar-deep-dive.md) | RespEar — respiratory rate from RSA and stride coupling |
| [`docs/06-hEARt-beginner.md`](docs/06-hEARt-beginner.md) · [`07-deep-dive`](docs/07-hEARt-deep-dive.md) | hEARt — in-ear heart-rate monitoring under motion |

### The implementation
| Doc | What it covers |
|---|---|
| [`docs/03-implementation-plan.md`](docs/03-implementation-plan.md) | The plan the work followed, and the risk register |
| [`docs/code-walkthrough.md`](docs/code-walkthrough.md) | **Line-by-line** explanation of every module, with the design decisions |
| [`docs/architecture.svg`](docs/architecture.svg) | The implemented architecture, one diagram |
| [`docs/08-the-journey.md`](docs/08-the-journey.md) | Every decision and why — including the dead ends and the mistakes |

### The results, and how to read them
| Doc | What it covers |
|---|---|
| [`docs/findings.md`](docs/findings.md) | Results, caveats, next steps |
| [`docs/09-slides-11-13-explained.md`](docs/09-slides-11-13-explained.md) · [`10`](docs/10-slides-14-16-explained.md) | The architecture, data and evaluation, in plain language |
| [`docs/11-numbers-explained.md`](docs/11-numbers-explained.md) · [`13`](docs/13-tables-explained-rigorously.md) | Every reported number, and every table cell, explained |

A slide deck and narration script live in [`presentation/`](presentation/).

---

## The three experiments the paper doesn't run

- **C1 — Nuisance probe.** Freeze the encoder; train a linear probe to predict *recording
  device* from the embeddings. The paper's contrastive variants build positive pairs from
  two crops of the *same recording*, which pressures the representation to encode recording
  identity. Device turns out **more** decodable than the disease, and it survives a
  bandwidth-matched control — so it isn't just the sample-rate artifact. Directly relevant to
  any product deployed across heterogeneous hardware.
- **C2 — Deployment profile.** Latency, memory and size, which the paper never reports. 125 ms
  per clip, 83 MB. (INT8 quantisation failed on the test machine — recorded, not hidden.)
- **C3 — Supervised baseline.** The paper compares OPERA only against *other pretrained
  models*, never a small model trained directly on the task. A 60k-param CNN beats it here —
  with important caveats about fairness in `findings.md`.

---

## Running it

**1. Install**

```bash
pip install -r requirements.txt
```

**2. Download the OPERA-GT checkpoint** (~394 MB; not included — it's the authors' weights)

```bash
mkdir -p artifacts/checkpoints
curl -sSL -o artifacts/checkpoints/encoder-operaGT.ckpt \
  https://huggingface.co/evelyn0414/OPERA/resolve/main/encoder-operaGT.ckpt
```

**3. Get ICBHI 2017** ([Kaggle mirror](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)),
then point the scripts at its `Respiratory_Sound_Database` folder.

**4. Run everything**

```bash
PYTHONPATH=src python3 scripts/run_experiments.py --data-root /path/to/Respiratory_Sound_Database
PYTHONPATH=src python3 scripts/robust_eval.py     /path/to/Respiratory_Sound_Database
```

**5. Tests**

```bash
PYTHONPATH=src python3 -m pytest tests/ -q      # 76 tests
```

Features cache to `artifacts/` after the first run, so only that run needs a GPU;
everything downstream is seconds on CPU.

---

## Layout

```
src/respnet/
  data/audio.py          mel filterbank + STFT, hand-written (no torchaudio)
  data/icbhi.py          dataset index, subject-wise splits, bandwidth control
  models/opera_gt.py     the OPERA-GT encoder, from scratch
  models/baseline_cnn.py the supervised comparison
  probe.py               linear probing, nuisance probes, resampled-split variance
  profiling.py           latency / memory / quantisation
  pipeline.py            stage orchestration + caching
scripts/
  run_experiments.py     full run  -> artifacts/results.json
  robust_eval.py         8-split variance -> artifacts/robust_results.json
tests/                   76 tests
docs/                    see above
presentation/            slide deck + narration script
```

---

## Honest scope

Stated plainly, because it matters:

- **One encoder of three** (GT), **one task of nineteen** (T7).
- **Not benchmarked against the paper's numbers.** Clip selection, pooling and split
  construction differ; matching their T7 definition is the first next-step. The COPD AUROC
  here is *our protocol*, not a reproduction of theirs.
- **Input orientation unverified** — 128×128 is also consistent with the checkpoint; 64×256
  is chosen because the paper states 64 mel bins.
- **Head count inferred** from the ViT-Small convention, not read from the weights.
- Device is **confounded with clinical site** in ICBHI, so the C1 probe may read site or
  population as well as microphone.
- The CNN baseline is a single split with best-epoch selection — optimistic; not yet a fair
  fight against the 8-split probe.

---

## Attribution & licence

Pretrained OPERA-GT weights © the OPERA authors (Zhang et al., NeurIPS 2024), released at
[huggingface.co/evelyn0414/OPERA](https://huggingface.co/evelyn0414/OPERA). ICBHI 2017 ©
Rocha et al. This repository's **code and documentation** are released under the MIT licence
([`LICENSE`](LICENSE)). The pretrained weights and the dataset are **not** included and
retain their original licences.
