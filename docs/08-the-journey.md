# The Journey — every decision, and why

A non-code narrative of how this project came to be what it is. Read it before the
code; it gives you the context that makes the code obvious rather than arbitrary.

Written honestly, including the decisions that were wrong and the things I got
wrong along the way.

---

## Part 0 — The brief and its constraints

You set five constraints, and almost every decision below traces back to one of them:

| Constraint | Consequence |
|---|---|
| **One week**, Auryx only | Scope had to shrink to one encoder, one task |
| Close the **audio** and **PyTorch** gaps | Ruled out pure signal-processing papers |
| Must be **published by / core to Auryx's research** | Ruled out RespireNet |
| **Small, easy read, easy to implement** | Ruled out reimplementing HTSAT |
| **Free Colab GPU at most** | Ruled out any pretraining |
| You want to **understand it, not have AI write it** | Shaped what I built vs. explained |

Two of these are in direct tension: *"from Auryx's own research"* points at in-ear
signal processing, while *"close the PyTorch gap"* points at deep learning. Most of
Part 1 is resolving that tension.

---

## Part 1 — Choosing the paper

### Attempt 1: RespireNet — abandoned

My first choice was **RespireNet** (Gairola et al., EMBC 2021) on ICBHI. I built about
1,200 lines of scaffold for it.

**Why it looked right:** public dataset, small, published baselines to beat, and its
device-specific fine-tuning is conceptually the same problem Auryx faces across earbud
models.

**Why it was wrong:** you pointed out the actual criterion — it had to be *Auryx's own
research*. RespireNet is Microsoft Research India. No connection to Cambridge, Mascolo,
or the company.

I discarded that scaffold. Some of it survived (the testing patterns, the profiling
module); most did not.

**Lesson worth keeping:** I optimised for *feasibility* before confirming *relevance*.
The order should have been the other way round.

### Attempt 2: the in-ear papers — read, not implemented

You found three papers on Auryx's site — **RespEar**, **hEARt**, and hEARt's journal
extension. All authored by Butkow (the CTO) and Mascolo.

These are maximally relevant. They are the company's actual technology. And I recommended
*against* implementing them, for three reasons:

1. **No public data.** hEARt is 20 subjects, RespEar 18, both collected in-lab with
   custom hardware. You cannot reproduce what you cannot download, and access requests
   don't resolve in a week.
2. **They're signal-processing papers.** RespEar is filters, Hilbert envelopes, FFT, SSA
   and RLS. Implementing it would teach you real DSP and put **zero PyTorch** on your CV.
3. **Custom hardware.** 3D-printed sealed earbuds with a specific microphone.

So: read them for the interview, implement something else.

### Attempt 3: OPERA — chosen

**OPERA** (Zhang et al., NeurIPS 2024) is Mascolo's lab's flagship open release. She is
senior author *and* an Auryx co-founder.

**Why it resolves the tension:**

- It is genuinely their lab's work → satisfies "Auryx's research"
- It is deep learning in PyTorch → closes the PyTorch gap
- It is respiratory audio → closes the audio gap
- **Checkpoints are released** → no pretraining needed → fits free Colab
- **Linear probing** is the evaluation protocol → frozen encoder, trivial compute

And a strategic argument I only articulated later, which I think is the strongest one:
Auryx's own job posting says their mission is *"the world's best foundation model for
turning sound into health insights."* That is the OPERA direction in their own words.
Butkow's DSP papers are their **present**; OPERA is their **stated future**. The job asks
for PyTorch, not DSP. So the project aims where they say they're going.

### The honest reframe

You cannot implement OPERA from scratch — they pretrained on 404 hours. So "from scratch"
was redefined:

- **Their contribution:** the pretrained weights
- **My contribution:** the encoder architecture, the entire audio front end, feature
  extraction, probing, evaluation, profiling, and the experimental design

That is still a substantial from-scratch build — 1,859 lines — and it's the part where
understanding lives.

---

## Part 2 — Before writing any code

### Verification first, because a week is short

Before committing you to the plan I checked three things that could each have killed it:

1. **Are checkpoints actually released?** → HuggingFace API returned 200, three files:
   `encoder-operaCE.ckpt`, `-CT`, `-GT`. Yes.
2. **Is ICBHI one of the benchmark datasets?** → Yes, T7 is COPD detection on ICBHI.
3. **Is there a linear-probing path?** → Yes, `extract_opera_feature`.

**Why this mattered:** each is a single query, and any one coming back negative would
have meant a different project. Spending twenty minutes here saved potentially days.

### The dataset arrived — five checks before trusting it

When you sent the Kaggle download I ran five checks. Each was chosen because failing it
silently would corrupt everything downstream:

| Check | Result | Why I checked |
|---|---|---|
| File counts | 920 wav + 920 txt ✓ | A partial download looks identical to a complete one until results are wrong |
| Patient count | 126 ✓ | Confirms it's the full release |
| Total duration | 5.49 h ✓ | Matches the published 5.5 h |
| Filename convention | matches parser ✓ | My parser assumes 5 underscore-separated fields |
| Diagnosis labels | present, 64 COPD / 126 ✓ | **T7 needs `patient_diagnosis.csv`, which not every mirror includes** |

That last one was the real risk. The original ICBHI challenge is about *cycle-level*
crackles and wheezes; T7 is *patient-level* COPD. Different label file. A mirror without
it would have forced a different task.

### The discovery that changed the experiment

Then I checked something nobody asked for: **sample rate per device.**

| Device | Native rate |
|---|---|
| AKGC417L | 100% @ 44.1 kHz |
| LittC2SE | 100% @ 44.1 kHz |
| **Litt3200** | **100% @ 4 kHz** |
| Meditron | mixed |

Litt3200 is *uniquely identifiable from sample rate alone*. Resample 4 kHz audio to 16 kHz
and it has zero energy above 2 kHz — about half the mel bands (30 of 64) are simply empty.

**Why this mattered enormously:** the headline experiment (C1) asks whether OPERA's
embeddings encode recording device. Without this check, the answer would have been "yes,
overwhelmingly" — and it would have been *an artifact of the resampler*, not a finding
about the representation. I would have written up a false result.

**The decision:** run C1 twice — naive, and on a bandwidth-matched subset with all
natively-low-rate files removed. Report both. **The gap between them is the actual
science.**

This turned out to be the single most valuable twenty minutes in the project — and note
that it came from checking a property of the data that had nothing to do with the task.

---

## Part 3 — Design decisions during the build

### Decision: which of the three encoders

| Encoder | Architecture | Verdict |
|---|---|---|
| OPERA-CT | HTSAT transformer from CLAP, 32.4M | **No** — HTSAT is complex, needs CLAP code, has a built-in STFT frontend. Days of work. |
| OPERA-CE | EfficientNet-B0, 4.97M | **No** — needs `efficientnet_pytorch`, and the `_blocks` naming means reimplementing MBConv blocks faithfully |
| **OPERA-GT** | **MAE ViT, 21M** | **Yes** — a standard ViT. Writable in pure PyTorch, no dependencies. |

**Cost of this choice, stated plainly:** GT is the *generative* model. My theoretical
prediction (doc 02 §3.1) was that the **contrastive** models CT/CE would encode device
most strongly, because their positive pairs are two crops of the same recording. So I
tested the encoder *least* likely to show the effect.

**That turned out to strengthen the result.** Device is strongly encoded even in the model
with no instance-discrimination pressure. If it shows up in GT, CT is likely worse — but
that is a prediction, not a measurement, and it's the top item in the next-steps list.

### Decision: don't use their environment

The OPERA repo installs via `conda env create` plus shell scripts that modify `~/.bashrc`.
I chose to skip it entirely and write my own inference path.

**Rationale:** on Colab that fights you; you learn nothing from `pip install -e .`; and
depending on their preprocessing config means you don't know what your preprocessing does.

**This is the decision that cost us benchmarkability. See Part 5.**

### Decision: no torchaudio, no torchvision, no timm

Only `torch`, `numpy`, `scipy`, `scikit-learn`.

**Rationale:** runs on a bare runtime; forces the mel filterbank to be written by hand
(five lines, and it's the part people most often treat as a black box); avoids silent
mismatches between torchaudio's defaults and OPERA's stated parameters.

**Cost:** ~80 extra lines. Worth it, and it also meant the environment needed nothing
installed — torchaudio was in fact missing from your machine.

### Decision: derive the input shape rather than guess it

The paper says 64 mel bins. The checkpoint says 1024 patches at 4×4. Therefore
`(64/4) × (n_frames/4) = 1024` → **n_frames = 256** → 8.192 s clips.

**But here is an honest gap:** `128 × 128` also gives 1024 patches. Both load, both run.
I chose 64×256 because the paper states 64 mel bins, which makes it much more likely — but
**I did not verify it**, and if it's wrong the embeddings are systematically degraded
without any error being raised. It's flagged in amber on the architecture diagram.

### Decision: mean-pool patch tokens, not CLS

In contrastive models the CLS token is trained directly by the objective. **In an MAE,
nothing trains CLS** — the loss is reconstruction of masked patches. So mean-pooled patch
tokens should be the stronger representation. I implemented all three options (`mean`,
`cls`, `both`) so it's testable rather than assumed, and defaulted to mean.

### Decision: strict weight loading

`load_state_dict(strict=False)` to tolerate leftover decoder keys, then **manually check
`missing` and raise**.

**Rationale:** this is the highest-consequence failure mode in the project. A silent
partial load leaves layers randomly initialised and produces embeddings that look
completely plausible and mean nothing. You could lose a week to it. This converts it into
an immediate exception.

**It also became the correctness proof:** strict load succeeded with zero missing
parameters and the total came to **21,694,848** against the paper's stated 21M.

---

## Part 4 — What went wrong, and what it taught

### Bug 1: 24-bit wav files

`scipy.io.wavfile.read(..., mmap=True)` crashed: *"mmap=True not compatible with 3-byte
container size."* ICBHI contains 24-bit audio; scipy's memory-mapped reader can't handle
3-byte samples.

Fixed with stdlib `wave`, which reads only the header, plus a scipy fallback.

**Lesson:** real datasets contain format variety that clean tutorials don't prepare you
for. Read headers, don't decode, when you only need metadata.

### Bug 2: the variance flaw — the most instructive thing here

The first full run reported **`std = 0.000` on every single probe** across five seeds.

That looks like remarkable stability. It is nothing of the sort. Logistic regression
fitted on fixed data is **deterministic** — varying the classifier's `random_state`
varies nothing at all. The error bars were describing solver noise, which is zero.

**The uncomfortable part:** this is *precisely* the criticism I had written about OPERA's
own protocol two documents earlier (doc 02 §6.1 — "their 5 runs vary only the probe seed,
not the data split"). I wrote the critique, then implemented the identical flaw, and only
caught it because the zeros were conspicuous.

**Fixing it changed a headline number by 16×:**

| | Single split | 8 resampled splits |
|---|---|---|
| Patient-leakage inflation | +0.004 AUROC | **+0.066 AUROC** |

The single split had landed somewhere the leakage effect was invisible. Anyone reporting
from one split would have concluded — wrongly — that patient leakage doesn't matter here.

**This is the most transferable lesson in the project**, and it's the one I'd lead with in
an interview. It's also a demonstration that the critique in doc 02 was correct, verified
on my own code rather than asserted about someone else's.

### Error 3: I asserted a number I hadn't verified

I wrote `"opera_paper_reported_auroc": 0.855` into the results file, with a note saying
"confirm against the paper's table". I never read that number from the paper. It was a
placeholder that would have read as a finding.

Removed and replaced with `None` plus an explicit statement that no published value is
quoted because none was verified. Flagging it here because it's exactly the kind of thing
that becomes an embarrassing correction in an interview.

---

## Part 5 — Why we can't benchmark against their numbers

You asked this directly, and it deserves a direct answer.

**It was a deliberate trade, correctly identified as the top risk, and then not
executed on.**

In the implementation plan (doc 03 §2) I wrote:

> *"The one thing you must borrow carefully: their exact task definition for whichever
> task you reproduce — split, label mapping, aggregation. If your task definition differs
> from theirs, your number is not comparable and the reproduction claim collapses. **This
> is the single highest-risk detail in the project.**"*

I named it as the top risk and then didn't do it, because when you asked me to build
under time pressure I prioritised getting the pipeline working end to end.

**What specifically differs, and therefore what would need matching:**

| Dimension | Mine | Theirs |
|---|---|---|
| Clip selection | one centred 8.192 s clip | unspecified — likely multiple clips + aggregation |
| Aggregation | none (one clip = one prediction) | probably recording- or patient-level pooling |
| Pooling | mean patch tokens | unspecified |
| Split | my stratified subject-wise, 25% | "participant-independent random" — different seed, different fraction |
| Label mapping | COPD vs all | probably the same, unverified |
| Metric | AUROC ✓ | AUROC ✓ |

**Was the trade defensible?** For the *stated* goals — closing the audio and PyTorch gaps,
understanding the pipeline — yes. You now understand every stage because you can read
every line, which `pip install -e .` would not have given you.

**Was it right?** Partly. The mistake wasn't writing my own pipeline; it was not spending
the extra hour reading their `src/benchmark/` T7 definition *before* running, so the
pipeline could have matched it from the start. The two goals were not actually in
conflict — I could have had both.

**The good news: it's cheap to fix.** Features are cached. Matching their task definition
and re-running the probe is a few hours, not a rebuild. It is item 1 in the next-steps
list for exactly that reason.

**And a nuance worth holding onto:** components C1, C2 and C3 — the device probe, the
deployment profile, the baseline comparison — **do not require benchmark comparability at
all.** They are internally-controlled experiments. The device probe compares device
decodability against COPD decodability *on the same features under the same protocol*.
That comparison is valid regardless of whether my absolute AUROC matches theirs. So the
novel findings survive this limitation entirely; only the word "reproduction" doesn't.

---

## Part 6 — Is this a good implementation?

Split the question, because the answer differs.

### As an encoder reimplementation: yes, and it's verified

- Architecture recovered purely from tensor shapes, no reference implementation consulted
- Strict load, zero missing parameters
- **21,694,848 parameters vs the paper's stated 21M — exact match**
- Preprocessing derived from the paper's stated values, not copied from a config
- Behavioural tests: permutation equivariance, residual structure, gain invariance
- Sanity check passed (same-patient similarity > different-patient)

That parameter match is strong evidence. A wrong architecture almost certainly either
fails to load or gives a different count.

### As a reproduction of the paper: no, and it doesn't claim to be

- 1 of 3 encoders
- 1 of 19 tasks
- Task definition not matched (Part 5)
- Input orientation unverified (64×256 vs 128×128)
- Head count inferred from ViT-Small convention, not stated
- No fine-tuning comparison

**Honest label: a verified reimplementation of the OPERA-GT encoder, and an independent
evaluation of it on ICBHI — not a replication of the OPERA benchmark.** The README and
findings both say this explicitly, and you should too.

### As a piece of engineering: reasonable

76 tests, caching at both expensive boundaries, controls on every experiment, honest
recording of the quantisation failure rather than omitting it, and every result written
to JSON so the write-up quotes data rather than memory.

**Weakest parts:** one clip per recording discards most of the audio for long files
(recordings run up to 86 s; we use 8.2). The CNN baseline is a single split with
best-epoch selection, which is optimistic and not a fair fight against the 8-split probe.

---

## Part 7 — What critique are we actually making?

Four distinct claims, and they are not equally strong. It matters to keep them separate.

### 1. Device encoding — a genuinely novel empirical finding ⭐

Nobody has published this. It is motivated by theory (contrastive positives are
same-recording crops), tested with a control (bandwidth matching), and it survived that
control.

**Claim:** recording device is roughly twice as linearly decodable from OPERA-GT
embeddings as the clinical target — device lift 0.715 of available headroom, COPD 0.518.

**Deliberately not claimed:** that the COPD classifier *uses* device as a shortcut.
Decodability is necessary but not sufficient. Proving use needs a cross-device
generalisation experiment.

**Also stated as a limitation:** in ICBHI, device is partly confounded with clinical site,
so some of what the probe reads may be site or population rather than microphone.

### 2. No deployment cost reported — a gap we filled

The paper reports no latency, memory or energy figures. For a mobile-systems lab whose
motivation is ubiquitous health sensing, and whose OPERA-CE variant exists *purely* for
efficiency, that is a conspicuous omission. We measured it: 125.3 ms, 82.8 MB.

Not a novel technique — just a measurement nobody made.

### 3. No supervised baseline — a gap we filled

The paper compares OPERA only against *other pretrained models*. Never against a model
trained directly on the task. A 60,530-parameter CNN beats the 21.7M probe 0.969 to 0.874.

**Weakest of the four**, because it isn't apples-to-apples (linear probe on frozen
features vs end-to-end training) and the CNN number is a single split. The defensible
version is: *"on this task at this data scale, the pretraining does not pay for itself
under the linear-probe protocol."*

### 4. Variance methodology — a methodological critique, demonstrated

Reporting variance across probe seeds measures nothing, because the fit is deterministic.
The uncertainty that matters is which patients land in the test split.

**Strong because we proved it on ourselves:** our own implementation had the flaw, produced
exactly zero variance, and fixing it moved a headline number 16×.

### And one finding that is just useful

Patient-wise leakage inflates AUROC by +0.066 on this dataset, and that inflation is
invisible from a single split.

---

## Part 8 — What I'd do differently

1. **Confirm relevance before feasibility.** I built RespireNet scaffold before checking
   whose paper it was.
2. **Read their benchmark definition before running anything.** One hour, and it was the
   risk I had myself identified as the largest.
3. **Design the variance protocol properly at the start** rather than noticing zeros.
4. **Verify the input orientation** by testing 64×256 against 128×128 empirically — the
   sanity-check separation score would probably discriminate between them.
5. **Never write a placeholder number into a results file.** It reads as a finding.

---

## Part 9 — How to talk about this

The three things to lead with, in order:

1. **The device finding** — a respiratory foundation model pretrained on 404 hours encodes
   recording device about twice as strongly as the disease, and it survives a
   bandwidth-matched control. For a company shipping on heterogeneous consumer earbuds,
   that's a live risk. Then immediately state the limitation: decodability isn't proof of
   use, and the cross-device experiment is the one that would settle it.

2. **The variance bug** — because owning a methodological error you found in your own work,
   and quantifying its impact, demonstrates more than a clean result does.

3. **The baseline** — stated with caveats.

And when they ask why you can't compare to their number, the answer is the honest one
from Part 5: *"I wrote my own pipeline to understand it, which cost comparability. I
should have read their T7 definition first — the two weren't actually in conflict. It's a
few hours to fix and it's top of my list. But the device and deployment findings don't
depend on it, because they're internally controlled."*

That answer is better than a clean reproduction would have been, because it shows you
know exactly what your work does and doesn't support.
