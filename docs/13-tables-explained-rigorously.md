# The tables, explained in full — every column, every cell

Slides 16, 18, 20, 21, 22. For each table: what every **column** measures, then what every
**cell** means, why it is high or low, and how it compares to the others. Numbers are the
exact values from `artifacts/robust_results.json` and `artifacts/results.json`.

---

# SLIDE 16 — COPD detection

```
Protocol                          AUROC          Balanced acc.
Subject-wise, 8 resampled splits  0.874 ± 0.024  0.763 ± 0.034
Random split (patient-leaky)      0.940 ± 0.018  0.822 ± 0.035
Shuffled-label control            0.492          0.533
```

## The columns

**Column 1 — Protocol.** *How the data was divided into a training set and a test set.*
This is not a property of the model — it's a property of the experiment. All three rows use
the **same** model and the **same** embeddings. Only the splitting rule changes. That is
the entire point of the table: to show that the *split*, not the model, moves the score.

- *Subject-wise* — every recording from a given patient goes entirely to train or entirely
  to test. A patient is never on both sides.
- *Random (patient-leaky)* — recordings are shuffled and split individually, so a patient's
  12 recordings scatter across both train and test.
- *Shuffled-label control* — subject-wise split, but the COPD/healthy labels are randomly
  permuted so they carry no information.

**Column 2 — AUROC.** *The probability that the model scores a randomly chosen COPD
recording higher than a randomly chosen healthy one.* Scale: 0.5 = random guessing (the
floor), 1.0 = perfect. This is the primary metric because it is **immune to class
imbalance** — it only ever compares one-of-each, so the 86%-COPD skew in the recordings
cannot inflate it.

**Column 3 — Balanced accuracy.** *The average of two recall rates: how many COPD cases it
caught, and how many healthy cases it caught, averaged with equal weight.* Scale: 0.5 =
chance (for 2 classes), 1.0 = perfect. Included alongside AUROC because they answer
different questions. AUROC asks "can it *rank* / order cases?" Balanced accuracy asks "at a
fixed decision threshold, how many does it actually get right, counting both classes
equally?" A model can rank well (high AUROC) yet make many hard calls wrong (lower balanced
accuracy) — which is exactly the pattern here.

**The ± value** on each: the standard deviation across the 8 re-runs. It is the measurement
error. Two numbers separated by less than their ± are statistically indistinguishable.

## The cells

**Row 1, subject-wise — AUROC 0.874.** The honest headline. On the 0.5-to-1.0 scale it is
**strong** (past 0.8, near 0.9). Why is it this high? Because the OPERA embeddings — even
though the encoder was trained on unlabelled coughs and breaths and never on COPD — carry
information that separates diseased from healthy lungs, and a single linear layer can
extract it. Why is it not higher, near 1.0? Two reasons: (a) COPD is genuinely hard to call
from a short lung-sound clip in some patients; (b) the encoder was never *tuned* for this —
we froze it and put one linear layer on top, deliberately the weakest possible read-out.

**Row 1 — balanced accuracy 0.763.** Notice it is *lower* than the AUROC of 0.874. This is
the informative gap. It means the model **orders** cases well (0.874) but, forced to draw a
single yes/no line, it still misclassifies about a quarter of cases counting both classes
equally. The disease signal is real but not cleanly separable — there is overlap between
the two classes that ranking tolerates and a hard threshold does not.

**Row 1 — the ± 0.024 / 0.034.** Small relative to the scores. The 8 per-split AUROCs ranged
0.842 to 0.919. So the result is stable: it does not depend on one lucky division of
patients. This small number is also the yardstick for judging every difference below.

**Row 2, random split — AUROC 0.940.** Higher than row 1 by **0.066**. Is that a real
difference or noise? 0.066 is about **2.7×** the ± 0.024, so it is real, not luck. But it is
real *illusion*: the model is not better, it is being tested partly on patients it already
trained on, so it can recognise the person rather than the disease. The 0.066 gap is a
direct measurement of how much a careless split would flatter you.

**Row 2 — balanced accuracy 0.822.** Also inflated versus row 1 (0.763), by the same
mechanism, and by a similar margin (0.059). Both metrics move together, which confirms the
inflation is systematic, not an artifact of one metric.

**Row 3, shuffled control — AUROC 0.492.** The safety check. Labels are randomised, so the
only honest score is 0.5. It landed at 0.492 — within measurement noise of exactly chance.
Why does this matter? It rules out leakage *in the machinery itself* — a bug in the split,
duplicated rows, statistics computed across train and test. If this had come back at, say,
0.65, every other number in the study would be untrustworthy, because it would mean the
pipeline manufactures signal from noise. 0.492 certifies it does not.

**Row 3 — balanced accuracy 0.533.** Essentially chance (0.5) too, with the tiny excess
being ordinary noise from a single run. Consistent with the AUROC reading.

## What the table proves as a whole

Same model, three splits: **0.874 honest → 0.940 when cheating → 0.492 on nonsense.** The
spread across rows is far larger than the spread within any row (the ± values). Therefore
the dominant driver of the score is *the experimental protocol*, and the honest protocol
gives 0.874.

---

# SLIDE 18 — What else do the embeddings encode?

```
Probe target                 Classes  Chance  Balanced acc.  Lift of headroom
Device — naive               4        0.250   0.861 ± 0.045  0.815
Device — bandwidth-matched   3        0.333   0.810 ± 0.033  0.715
COPD — same subset           2        0.500   0.759 ± 0.051  0.518
Chest location               7        0.143   0.341 ± 0.018  0.231
Device — shuffled control    4        0.250   0.264          0.019
```

Every row uses the **same frozen embeddings**. Only the *label* being predicted changes.
The question is no longer "how good is the model at COPD" — it is "what information is
sitting inside these embeddings, whether it should be there or not."

## The columns

**Column 1 — Probe target.** The thing a fresh linear layer is asked to predict from the
embeddings. Device = which of the stethoscopes recorded it. COPD = the disease. Chest
location = where on the body the recording was taken.

**Column 2 — Classes.** How many possible answers. This is why the raw accuracies are not
directly comparable: guessing 1-of-4 is harder than guessing 1-of-2. This column is the
reason the last column exists.

**Column 3 — Chance.** The score a model gets for pure guessing, which is **1 ÷ classes**.
4 classes → 0.25. 2 classes → 0.50. 7 classes → 0.143. This is the floor for *that specific
row*. A balanced accuracy of 0.34 sounds poor until you see its floor is 0.143.

**Column 4 — Balanced accuracy.** Same metric as slide 16, but now most rows have more than
2 classes: it is the average recall across all the classes, so that a common class cannot
dominate. Its floor is the Chance column; its ceiling is 1.0.

**Column 5 — Lift of headroom.** *Of the distance from guessing to perfect, what fraction
did the probe actually cover.* Formula:

```
lift = (balanced_acc − chance) / (1 − chance)
```

This is the only fair cross-row comparison, because it cancels out the different numbers of
classes. 0 = no better than guessing; 1 = perfect. **Read this column to compare rows.**

## The cells

**Row 1, device naive — balanced acc 0.861, lift 0.815.** From the raw embeddings, a linear
layer identifies which of 4 stethoscopes recorded a clip 86% of the time against a 25%
floor — capturing **81.5%** of the available skill. Almost perfectly readable. But this
number is *contaminated*: one device (Litt3200) recorded at 4 kHz, so half its spectrogram
is blank, and the probe can spot that trivially. So this row overstates the real effect —
which is precisely why row 2 exists.

**Row 2, device bandwidth-matched — balanced acc 0.810, lift 0.715.** After removing every
low-sample-rate file so all remaining devices share a bandwidth (which also drops the count
from 4 classes to 3, hence chance rises to 0.333), device is **still** readable at 0.715
lift. Compare to row 1: 0.815 → 0.715. It barely fell. **This is the load-bearing number of
the whole study.** If the device signal had been *only* the blank-spectrogram artifact, this
would have collapsed toward 0. It did not — so the encoder genuinely encodes recording
device for real reasons (microphone frequency response, noise floor), not just the obvious
cheat.

**Row 3, COPD same subset — balanced acc 0.759, lift 0.518.** The disease, measured on the
*exact same recordings and protocol* as row 2, so it is directly comparable. It captured
51.8% of the available headroom. Now the headline comparison: **device 0.715 vs disease
0.518.** The embeddings encode *which microphone was used* about **1.4× more strongly** than
they encode *the disease the model is supposed to detect*. (0.715 ÷ 0.518 ≈ 1.38.)

**Row 4, chest location — balanced acc 0.341, lift 0.231.** The control that makes the
device result meaningful. Chest location is only weakly decodable — 0.231 lift, low. Why
does this matter? Because it proves the probe does **not** simply read *everything* off the
embeddings. If chest location had also scored ~0.7, the device result would be worthless —
it would just mean "linear probes read anything." Instead, device stands out sharply against
chest location, so the strong device number is a real, specific property of the
representation, not a quirk of the method.

**Row 5, device shuffled control — balanced acc 0.264, lift 0.019.** The safety check, same
role as slide 16's 0.492. Scramble the device labels and the probe manages lift 0.019 —
essentially zero. This certifies that when rows 1–2 report device is readable, that is real
signal, not the probe hallucinating structure in noise.

## What the table proves as a whole

Ordered by lift: **device (0.715) > disease (0.518) > chest location (0.231) > noise
(0.019).** The encoder carries a strong, real, specifically-device fingerprint that outranks
the clinical target — with the controls (rows 2, 4, 5) each closing off an alternative
explanation (Nyquist artifact, "probe reads everything," and hallucinated signal
respectively).

---

# SLIDE 20 — Deployment cost

```
Metric               Value
Median latency       125.3 ms   (batch 1, CPU, 200 iterations)
On disk (fp32)        82.8 MB   (21.7M parameters)
Inter-quartile range  11.3 ms
```

Different in kind: these measure **speed and size**, not correctness. There is no
"good"/"bad" scale — the significance is contextual (can it run on the target hardware?)
and comparative (the paper reports none of it).

## The measurements

**Median latency — 125.3 ms.** Time to convert one 8.192-second clip into its 384-number
embedding. *Median* means the middle value of 200 timed runs (100 faster, 100 slower).

- Why median, not mean: run times are **right-skewed** — bounded below by the actual
  compute (~106 ms, the observed minimum) and unbounded above by random OS interruptions. A
  few slow outliers drag a *mean* upward and misrepresent typical speed. The median ignores
  them, so it reports what a normal run actually costs.
- What 125 ms *implies*: on a laptop, trivial. Analysed once every 30 seconds, it is a 0.4%
  duty cycle. But an earbud's chip is far weaker than a laptop CPU and runs on a tiny
  battery, so for *always-on* inference this is likely too slow. The number is neither good
  nor bad in the abstract — it is a **design constraint** that decides which use-cases are
  feasible.

**Inter-quartile range — 11.3 ms.** The spread of the middle 50% of runs (the gap between
the 25th and 75th percentile). It is a measure of *consistency*.

- 11.3 ms is under 10% of the 125 ms median → the timing is tight and repeatable, not
  jumping around. This is what makes the 125 ms trustworthy rather than a fluke. Reported
  instead of a standard deviation for the same reason median beats mean: it is not distorted
  by the skew.
- (Supporting figures from the run: p90 = 134.9 ms, minimum = 106.0 ms. The bulk of runs sit
  in a narrow 106–135 ms band.)

**On disk (fp32) — 82.8 MB.** The 21.7M parameters stored at full precision (float32 = 4
bytes each: 21.7M × 4 ≈ 83 MB). *Implies:* sizeable for a small wearable's memory as-is,
which is why quantisation (storing each number in 1 byte instead of 4, ~4× smaller) would
be the natural next step — and why it mattered that quantisation **failed** on this machine
(`NoQEngine` — no quantised backend in this PyTorch build), leaving that optimisation
unmeasured. Recorded as a failure rather than hidden.

## What the table proves as a whole

The model is comfortably runnable on a laptop, of questionable feasibility always-on on a
wearable, and its size points straight at quantisation as the first optimisation. None of
this appears in the paper — which, for a company whose product is on-device sensing, is the
gap the table fills.

---

# SLIDE 21 — Foundation model vs a small CNN

```
Model                       Parameters   AUROC (subject-wise)
OPERA-GT + linear probe     21,694,848   0.874 ± 0.024
Small CNN, trained scratch     60,530    0.969 best · 0.951 mean of last 5
```

## The columns

**Column 1 — Model.** Two fundamentally different strategies. *OPERA-GT + probe* = take the
big encoder pretrained on 404 hours, freeze it, add one linear layer (transfer learning).
*Small CNN* = a tiny network with no pretraining, trained directly on the 920 ICBHI
recordings from random initialisation (task-specific learning).

**Column 2 — Parameters.** The count of learned numbers = model size/capacity. 21,694,848
vs 60,530 is a **358×** ratio. This column frames the surprise: the far smaller model is
about to win.

**Column 3 — AUROC.** Same metric and same subject-wise protocol as slide 16, so the two
rows are comparable on the metric axis (with important caveats below).

## The cells

**Row 1 — 21.7M params, AUROC 0.874.** The foundation-model number, carried over from slide
16 row 1. It is the *mean over 8 splits*.

**Row 2 — 60,530 params, AUROC 0.969 / 0.951.** The from-scratch CNN. Two numbers because:
*0.969 best* is its single best training epoch; *0.951 mean* is the average of its last 5
epochs. The gap between them (0.018) is ordinary epoch-to-epoch wobble during training. The
honest figure to compare is **0.951**, because quoting the single best epoch cherry-picks
the luckiest moment.

**The comparison — 0.951 vs 0.874.** A difference of 0.077, about **3×** the ± 0.024, so it
is outside the noise: the tiny model really did score higher. *Implies:* on this task, at
this data scale, 404 hours of pretraining bought no advantage over training a small model
directly.

**But three reasons not to over-read it, in order of importance:**
1. **Not like-for-like.** The OPERA row is a *linear probe on frozen features* — the
   weakest possible read-out, by protocol. The CNN trains *end to end*. A fair fight would
   fine-tune the encoder or restrict the CNN. So this compares two different *usage
   strategies*, not two encoders.
2. **Different variance basis.** 0.874 is a mean over 8 resampled splits; 0.951/0.969 come
   from a *single* split. The CNN number has not been stress-tested the same way, and would
   likely come down on averaging.
3. **Possible shortcut.** From slide 18, device is strongly encoded and ICBHI confounds
   device with diagnosis. A from-scratch CNN may exploit that shortcut *more* efficiently
   than frozen features do, so part of its edge may be learning the microphone, not the
   disease.

## What the table proves as a whole

Narrowly and defensibly: *on this one task, at this data scale, under the linear-probe
protocol, pretraining did not pay for itself.* It does **not** prove "foundation models
don't work" — and the value of the slide is stating that distinction rather than taking the
flattering headline.

---

# SLIDE 22 — The variance mistake

```
Patient-leakage inflation   One split   Eight resampled splits
AUROC difference            +0.004      +0.066
```

Plus the observation that triggered it: the first run reported **standard deviation = 0.000**
on every probe.

## The columns

**Column 1 — the quantity being measured.** "Leakage inflation" = how much the AUROC goes
*up* when you switch from a correct subject-wise split to a cheating patient-leaky split. It
is the slide-16 gap (row 2 − row 1), i.e. the size of the illusion a careless split buys.

**Column 2 — One split.** That same gap, estimated the flawed way: run each protocol on a
*single* fixed division of the data, subtract.

**Column 3 — Eight resampled splits.** The same gap, estimated correctly: run each protocol
on 8 *different* random patient divisions, average, then subtract.

## The cells

**std = 0.000 (the trigger).** The first run's error bars were exactly zero on every score.
This is the alarm. Real measurements always wobble; a *perfect* zero means you are not
varying the thing that is supposed to vary. The cause: the "8 repeats" changed only the
classifier's random seed — but logistic regression on fixed data is **deterministic**, so
all 8 runs were byte-for-byte identical. I was averaging one run with itself and calling the
spread "stability."

**+0.004 (one split).** Estimated the flawed way, patient leakage looks negligible — four
thousandths of an AUROC point, ignorable. A single split happened to land where train and
test patients overlapped little, so leakage barely showed.

**+0.066 (eight splits).** Estimated correctly, by re-rolling *which patients* go where —
the thing that genuinely varies — leakage is +0.066. That is a **16×** larger estimate
(0.066 ÷ 0.004 ≈ 16), and it is a serious effect, not a negligible one.

## What the table proves as a whole

*What you randomise determines whether your error bars mean anything.* Randomising the wrong
thing (the deterministic classifier seed) produced fake zero-variance and a 16×
underestimate of a real effect. Randomising the right thing (the patient split) revealed it.
The learning is methodological, and it is the most persuasive item in the deck because it is
a demonstration of self-auditing: a suspiciously clean number was caught, diagnosed, and
fixed, and the fix changed a conclusion.

---

# The through-line

- **Slide 16**: the split — not the model — drives the score; the honest number is 0.874.
- **Slide 18**: the embeddings encode the microphone (0.715) more than the disease (0.518),
  and it is real (survives the control) and specific (chest location doesn't, noise
  doesn't).
- **Slide 20**: it costs 125 ms and 83 MB — a constraint the paper never reported.
- **Slide 21**: a 358×-smaller model won here, but only under an unfair-to-OPERA protocol.
- **Slide 22**: the wrong randomisation hid a real effect 16-fold, until a too-clean number
  gave it away.
