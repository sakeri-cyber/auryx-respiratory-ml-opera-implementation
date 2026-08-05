# Every number, explained for a beginner

Slides 16, 18, 20, 21, 22. For each number: what it literally measures, what scale it
lives on, and what it actually tells you.

First, three ideas that show up everywhere. Get these and the rest is easy.

---

## The three ideas behind almost every number

### Idea 1 — AUROC: "how often does it rank the sick person higher?"

Most of the scores are **AUROC**. The one-sentence definition:

> Pick one COPD recording and one healthy recording at random. Does the model give the
> COPD one a higher score? AUROC is the fraction of times it gets that ordering right.

The scale is the important part:

| AUROC | Meaning |
|---|---|
| **0.5** | Coin flip. The model knows nothing. |
| **0.7** | Weak but real signal. |
| **0.8** | Solid. |
| **0.9** | Strong. |
| **1.0** | Perfect — never wrong. |

**0.5 is the floor, not 0.** A score of 0.5 means useless. You basically never see below
0.5 unless something is broken. So don't read 0.874 as "87% of the way to perfect" — read
it as "most of the way from useless (0.5) to perfect (1.0)."

### Idea 2 — the ± number: "how much does the answer wobble?"

Every headline score has a companion like **± 0.024**. That's the **standard deviation** —
a measure of how much the number jumps around when you re-run the experiment on a
different random split of patients.

> **The analogy:** measuring your height 8 times with a slightly bent ruler. You'd report
> "172 cm, give or take 1 cm." The "give or take" is the ± number. Small = the measurement
> is stable and trustworthy. Large = it depends heavily on luck.

Rule of thumb: **if two scores are closer together than their ± values, you can't tell
them apart.** The difference might just be noise.

### Idea 3 — "lift over chance": making unfair comparisons fair

Some tasks are naturally easier to guess than others. Guessing between 2 options (COPD:
yes/no) starts you at 50%. Guessing between 7 options (which chest location) starts you at
about 14%. Comparing their raw accuracies would be unfair.

**Lift over chance** fixes this. It asks: *of the room for improvement above pure
guessing, how much did the model actually capture?*

```
lift  =  (score − chance) / (1 − chance)
```

- 0.0 means "no better than guessing"
- 1.0 means "perfect"

It puts every task on the same 0-to-1 ruler regardless of how many options it had. This is
the number that lets slide 18 compare "device" (4 options) against "disease" (2 options)
honestly.

---

# SLIDE 16 — COPD detection

| Protocol | AUROC |
|---|---|
| Subject-wise, 8 splits | **0.874 ± 0.024** |
| Random split (leaky) | 0.940 ± 0.018 |
| Shuffled control | 0.492 |

### 0.874 — the honest headline

The model, using a frozen encoder plus a single linear layer, ranks a COPD recording above
a healthy one **87.4% of the time**.

On the scale from Idea 1, that's **strong**. It's well past "solid" (0.8), approaching
"strong" (0.9). For a model that was **never trained on lung disease** — it was trained on
unlabelled coughs and breaths, then frozen — getting to 0.874 on COPD is a genuinely good
result. It says the embeddings carry real respiratory-health information.

### ± 0.024 — and why it's the number that keeps you honest

Across 8 different random draws of patients, the score wobbled by about 0.024.

This tells you two things:
1. **The result is stable.** 0.024 is small relative to 0.874. It's not a fluke of one
   lucky split.
2. **It sets the "can you even tell?" threshold.** Any two scores within ~0.024 of each
   other are, for practical purposes, the same. Hold onto this — it matters immediately
   below.

### 0.940 vs 0.874 — the cost of cheating

The leaky version scores **0.066 higher**. Is that real or just noise?

Compare it to the wobble: 0.066 is nearly **three times** the ± 0.024. So it's **well
outside the noise** — a real effect, not luck.

What it means: when you let the same patient appear in both training and testing, the model
scores 6.6 points higher **without being any better**. It's just recognising people it has
already seen. That 0.066 is **pure illusion** — the exact amount you'd fool yourself by if
you split the data carelessly.

### 0.492 — the smoke alarm

For this run I scrambled the labels so they mean nothing. A working pipeline **must** score
0.5 (pure chance) on nonsense labels.

It scored 0.492 — essentially exactly 0.5.

> **The analogy:** a smoke alarm going quiet in a room with no fire. It doesn't tell you
> the room is nice. It tells you nothing is secretly burning.

If this had come out at, say, 0.65, it would mean the pipeline was somehow finding "signal"
in random noise — which is impossible unless there's a leak (duplicated rows, contaminated
splits). Every other number would then be suspect. 0.492 means the machinery is clean.

---

# SLIDE 18 — What else do the embeddings encode?

This is the headline experiment, and it's all about comparing **lift over chance** (Idea
3) across different targets.

| Probe target | Options | Chance | Balanced acc | **Lift** |
|---|---|---|---|---|
| Device — naive | 4 | 0.250 | 0.861 | **0.815** |
| Device — bandwidth-matched | 3 | 0.333 | 0.810 | **0.715** |
| COPD — same subset | 2 | 0.500 | 0.759 | **0.518** |
| Chest location | 7 | 0.143 | 0.341 | 0.231 |
| Device — shuffled control | 4 | 0.250 | 0.264 | **0.019** |

Read the **Lift** column — that's the fair, apples-to-apples one.

### 0.815 — device is almost perfectly readable (naive)

Before any correction, the model can tell which of the 4 stethoscopes recorded a clip with
**0.815 lift** — it captured 81.5% of the possible skill above guessing. Almost perfectly
readable.

But we know part of this is the Nyquist cheat (the 4 kHz device with half its spectrogram
blank). So this number is contaminated. Which is why the next row exists.

### 0.715 — the number that matters

After removing the sample-rate cheat (dropping the 4 kHz files so all remaining recordings
share a bandwidth), device is **still** readable at **0.715 lift**.

**This is the single most important number in the whole study**, so sit with it:

- It barely dropped from 0.815. If the device signal had been *only* the Nyquist artifact,
  this would have collapsed toward 0. It didn't.
- So the model genuinely encodes **which microphone was used**, in some deep way that has
  nothing to do with the obvious blank-bins trick.

### 0.715 vs 0.518 — the headline comparison

Now put device (0.715) next to the disease you actually care about (0.518), on the same
subset, same protocol.

> **The model encodes the recording device about 1.4× more strongly than it encodes the
> disease.**

Think about what that means for Auryx. A model meant to detect health conditions is paying
*more* attention to what hardware recorded the sound than to the health condition itself.
For a product that has to work across every earbud on the market, that's exactly the thing
you'd want to know about. (With all the caveats from slide 19 — this shows the information
is *present*, not that the disease classifier necessarily *leans on* it.)

### 0.231 — the reassuring contrast

Chest location gets only **0.231 lift** — weakly readable. This is a useful sanity check:
the embeddings *don't* strongly encode everything. Device is special. If chest location had
also scored 0.7, you'd worry the probe just reads anything and the device result is
meaningless. It doesn't — so the device result stands out for a reason.

### 0.019 — the smoke alarm again

Scramble the device labels and the probe scores **0.019 lift** — essentially zero, pure
chance. Same purpose as the 0.492 on slide 16: it confirms the device probe isn't finding
fake signal in noise. When it says device is readable at 0.715, that's real.

---

# SLIDE 20 — What does it cost to run?

| | |
|---|---|
| Median latency | **125.3 ms** |
| On disk (fp32) | **82.8 MB** |
| Inter-quartile range | **11.3 ms** |

Different kind of number here — not accuracy, but **speed and size**. This is about whether
the model could actually run on a device.

### 125.3 ms — a tenth of a second per clip

To turn one 8-second clip into its 384-number embedding takes **125 milliseconds** — about
an eighth of a second — on a laptop CPU.

Is that fast or slow? Depends entirely on the use:
- **Fine** if you analyse a clip every 30 seconds. 125 ms of work every 30,000 ms is
  nothing — a 0.4% duty cycle.
- **Probably too slow** for always-on, continuous monitoring on a tiny earbud chip, which
  is far weaker than a laptop and running on a battery.

So the number itself is neutral. Its *significance* is that **the paper never reports it**,
and for a company building wearables it's a first-order question. You measured the thing
nobody measured.

### "median," not "average" — and why 11.3 ms matters

**Median** = the middle value when you line all 200 runs up in order. Half were faster, half
slower.

Why median instead of average? Because timing has occasional hiccups — the OS pauses your
program, a run takes 300 ms for no reason. A few big outliers drag the *average* upward and
mislead. The median ignores them.

**11.3 ms IQR (inter-quartile range)** is the spread of the middle half of the runs. It's
small — under 10% of the median — which means the timing is **consistent**, not jumping all
over the place. A tight IQR is what makes the 125 ms trustworthy rather than a lucky pick.

### 82.8 MB — the model's footprint

The 21.7 million numbers, stored at full precision (4 bytes each), take **82.8 MB** on disk.

Significance: too big to sit comfortably in a cheap earbud's memory as-is. This is why
techniques like quantisation (storing numbers in 1 byte instead of 4) exist — and why it
mattered that quantisation *failed* on my machine, leaving that optimisation unmeasured.

---

# SLIDE 21 — Foundation model vs a small CNN

| Model | Parameters | AUROC |
|---|---|---|
| OPERA-GT + probe | **21,694,848** | 0.874 ± 0.024 |
| Small CNN from scratch | **60,530** | **0.969** best · 0.951 mean |

### 21,694,848 vs 60,530 — a 358× size gap

The foundation model has **21.7 million** learned numbers. My little CNN has **60 thousand**
— about **358 times smaller**.

Feel the scale: if the foundation model were a 358-page book, the CNN is a single page.

### 0.969 vs 0.874 — the small one wins

And the single page scored **higher**: 0.969 vs 0.874.

Is 0.095 a real gap? It's about **4× the ± 0.024 wobble**, so yes — not noise. The tiny
model genuinely beat the giant.

**What this suggests** (carefully): on *this specific task, at this data scale, under this
protocol*, 404 hours of expensive pretraining did not buy you anything over a small model
trained directly on the 920 recordings. The pretraining didn't pay for itself here.

### Why "0.969 best · 0.951 mean" has two numbers — and this is a fairness point

- **0.951 mean** = the average AUROC over the last 5 training epochs.
- **0.969 best** = the single best epoch.

Reporting "best" is slightly generous — you're cherry-picking the model's luckiest moment.
The honest comparison is probably 0.951 vs 0.874, still a clear win but smaller.

**And the bigger fairness caveat:** even 0.951 isn't a clean fight. The CNN's number comes
from a *single* split; the OPERA number is a mean over *8* splits. And the CNN might be
exploiting the same device shortcut from slide 18. So the right takeaway is "pretraining
didn't obviously help here," **not** "foundation models are useless." The presentation says
exactly this, and saying it is what keeps the claim credible.

---

# SLIDE 22 — The mistake

| | |
|---|---|
| First run's spread | **std = 0.000** |
| Leakage, one split | **+0.004** |
| Leakage, 8 resampled splits | **+0.066** |

This slide has the fewest numbers and the most important lesson.

### 0.000 — the number that was too good

My first run reported a standard deviation of **exactly zero** on every score. No wobble at
all.

Why that's alarming, not reassuring: **real measurements always wobble a little.** A perfect
0.000 doesn't mean "incredibly precise" — it means "I'm not actually measuring the thing
that varies."

The cause: I was re-running with a different *random seed on the classifier*, but the
classifier is deterministic — same data in, same answer out, every time. So all 8 "repeats"
were identical. I was measuring nothing and calling it stability.

> **The analogy:** weighing yourself on the same broken scale 8 times, getting 70.0 kg each
> time, and proudly reporting "70.0 kg with zero variation!" The zero variation isn't
> precision — it's the scale being stuck.

### +0.004 vs +0.066 — a 16× difference in one conclusion

Once fixed (re-running with genuinely different *patient splits*, which is what actually
varies), the estimate of how much patient-leakage inflates the score changed dramatically:

- Old, broken way (one split): leakage looked like **+0.004** — negligible, ignorable.
- Correct way (8 real splits): leakage is **+0.066** — a serious effect.

**16 times larger.** The broken method had accidentally landed on a split where leakage
happened to be invisible, and concluded — wrongly — that it didn't matter.

**Why this is the best slide in the deck:** it's not a result, it's a demonstration of
carefulness. You caught your own mistake because a number looked *too* clean, you understood
why, you fixed it, and the fix changed a real conclusion. That's the single most convincing
thing you can show a research team — more than any accuracy score.

---

# The whole thing, in one table

| Number | One-line significance |
|---|---|
| **0.874** | Strong COPD detection from a frozen, never-trained-on-disease encoder |
| **± 0.024** | The wobble — sets the "can you even tell two scores apart?" threshold |
| **0.940** | Same model, cheating via patient leakage |
| **0.492** | Smoke alarm: pipeline finds nothing in scrambled labels ✓ |
| **0.815 → 0.715** | Device signal *survives* removing the Nyquist cheat — it's real |
| **0.715 vs 0.518** | Model encodes the microphone ~1.4× more than the disease |
| **0.231** | Chest location only weakly readable — device is genuinely special |
| **125.3 ms** | Cost to run — fine periodically, likely too slow always-on |
| **11.3 ms IQR** | Timing is consistent, so 125 ms is trustworthy |
| **21.7M vs 60k** | The foundation model is 358× bigger than the CNN that beat it |
| **0.969 / 0.951** | The tiny model won — pretraining didn't pay off here |
| **0.000** | Impossibly clean → I was measuring the wrong kind of variation |
| **+0.004 → +0.066** | Fixing that changed a conclusion 16-fold |

**If you internalise one thing:** the raw scores (0.874, 0.969) show competence. But the
*comparisons* — 0.715-vs-0.518, and 0.004-vs-0.066 — are where the actual insight lives.
Those two comparisons are what make this a study rather than a set of benchmark rows.
