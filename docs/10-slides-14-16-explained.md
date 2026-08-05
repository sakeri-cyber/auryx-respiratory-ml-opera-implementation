# Slides 14, 15, 16 — explained slowly

Continues from `09-slides-11-13-explained.md`. Same plain-language approach.

> **⚠️ One correction to the script before you record — see §15.4.** The line "the top
> third of the mel bins are simply empty" understates it. It's roughly the top **half**.

---

# SLIDE 14 — How do I know it's actually right?

I claimed I rebuilt someone's neural network correctly, from nothing but the numbers.
That's a bold claim. Here are the three things that back it up, and — at the end — what
they *don't* prove.

## Check 1: the parameter count

### What is a "parameter"?

A parameter is **one single number that the model learned during training**. Every weight
in every matrix, every bias. OPERA-GT's encoder has 21,694,848 of them.

### Why counting them proves anything

This is the part worth really understanding.

**The parameter count is completely determined by the architecture.** You don't choose it —
it falls out of the structure. Change anything about the design and the count changes.

So it works like a **fingerprint**. If my rebuilt model has exactly the same count as
theirs, my structure is almost certainly identical.

### Let's actually do the arithmetic

Here's the whole 21,694,848, built up from the shapes on slide 11.

**One transformer block:**

| Piece | Calculation | Parameters |
|---|---|---|
| norm1 (LayerNorm) | 384 weights + 384 biases | 768 |
| attn.qkv | 1152 × 384 weights + 1152 biases | 443,520 |
| attn.proj | 384 × 384 weights + 384 biases | 147,840 |
| norm2 (LayerNorm) | 384 + 384 | 768 |
| mlp.fc1 | 1536 × 384 + 1536 | 591,360 |
| mlp.fc2 | 384 × 1536 + 384 | 590,208 |
| **One block total** | | **1,774,464** |

**Twelve blocks:**

```
1,774,464 × 12 = 21,293,568
```

**Everything outside the blocks:**

| Piece | Calculation | Parameters |
|---|---|---|
| patch_embed | 384 × 1 × 4 × 4 + 384 biases | 6,528 |
| cls_token | one vector of 384 | 384 |
| pos_embed | 1025 × 384 | 393,600 |
| final norm | 384 + 384 | 768 |

**Add it all up:**

```
21,293,568  (blocks)
+    6,528  (patch embed)
+      384  (cls token)
+  393,600  (positional embeddings)
+      768  (final norm)
─────────────
21,694,848
```

**Exactly the number my model reports.** And the paper says "21M".

Now notice how sensitive this is. If I'd guessed **11 blocks** instead of 12, I'd be
1.77 million off. If I'd guessed width **320** instead of 384, everything changes. There
is essentially no way to get a wrong architecture that lands on the right count.

## Check 2: strict loading with zero missing keys

### How loading weights works

Both the checkpoint and my model hold a **dictionary** — names paired with blocks of
numbers:

```
"blocks.0.attn.qkv.weight"  →  [a 1152 × 384 block]
"blocks.0.mlp.fc1.weight"   →  [a 1536 × 384 block]
```

Loading means: **match them up by name and copy the numbers across.**

### What "missing keys" means

If my model has a slot called `blocks.5.mlp.fc1.weight` but the checkpoint has no entry
with that name, that's a **missing key**. Nothing gets copied. That layer keeps whatever
random numbers it was born with.

> **The analogy:** filling in a form by copying from another form. **Strict mode** = "every
> field must be filled, tell me loudly if one isn't." **Non-strict** = "fill in what you
> can, leave the rest blank, say nothing."

PyTorch's default behaviour will happily leave blanks and carry on.

### Why blanks are so dangerous

Here's the thing: **a randomly-initialised layer still works.** It still multiplies
matrices. It still outputs numbers of the right shape.

So if three of my twelve blocks silently failed to load, the model would still run. I'd
still get 384 numbers out per recording. They would look completely normal — same range,
same distribution, nothing suspicious.

They would just be **meaningless**. And every experiment afterwards would be measuring
noise while looking exactly like real results.

**That's the single most expensive failure available in this project** — you could lose a
week to it and never know. So I made the loader **raise an exception** if even one
parameter is missing. Turn a silent disaster into a loud, immediate stop.

Zero missing keys means: every single one of my 21,694,848 slots was filled from their
file.

## Check 3: the sanity check

The first two checks prove the *structure* is right. They say nothing about whether the
model actually produces sensible **output**.

### What an "embedding" is

Push a recording through the encoder and you get **384 numbers**. That's the embedding —
the model's compressed description of that recording.

### What "embedding space" means

Think of those 384 numbers as coordinates. Just as 2 numbers place a point on a map and 3
place it in a room, 384 numbers place it in a 384-dimensional space.

You can't picture it, but the maths works the same: **similar recordings should land near
each other.**

### The test

Two recordings from the **same patient** should be more similar than two from
**different patients**. Same person means the same lungs, the same chest, the same
stethoscope, the same room.

If the encoder produces anything sensible at all, that must hold. If it doesn't, something
upstream is broken.

I measure "similar" with **cosine similarity** — how closely two points' directions align.
1.0 means identical direction, 0 means perpendicular.

**Result:**

| | Cosine |
|---|---|
| Same patient, different recordings | 0.9854 |
| Different patients | 0.9706 |
| **Separation** | **+0.0148** |

It passes: same-patient really is more similar.

### But be honest about how small that is

Everything sits above 0.97. **All 920 recordings are crammed into a very narrow cone** of
the space, and the gap I'm relying on is a 1.5% difference.

That's normal for this kind of model — a masked autoencoder is trained to *reconstruct*
input, never to *spread things apart* — but it's worth saying, because it means the
separation is real but not dramatic.

## What these three checks do NOT prove

Worth having ready, because it's the honest limit:

- They don't prove the **input orientation** is right (slide 12). A 128 × 128 input would
  pass all three of these checks.
- They don't prove the **head count** is 6.
- They don't prove anything about **matching their evaluation protocol**.

They prove the architecture is right and the weights loaded. That's it — but that's the
foundation everything else stands on.

---

# SLIDE 15 — Two things about the data

## 15.1 The imbalance exists at two different levels

Here's the whole thing in one table:

| | COPD | Not COPD | Total | COPD share |
|---|---|---|---|---|
| **Patients** | 64 | 62 | 126 | **51%** |
| **Recordings** | 793 | 127 | 920 | **86%** |

Count as *people*, it's a coin flip. Count as *recordings*, it's overwhelmingly COPD.

**Why?** Because COPD patients were recorded far more:

```
COPD patients:      793 recordings ÷ 64 people  ≈  12.4 each
Everyone else:      127 recordings ÷ 62 people  ≈   2.0 each
```

Six times as many recordings per person.

## 15.2 Why that makes accuracy useless

Imagine the laziest possible model. It ignores the audio entirely and answers **"COPD"**
every single time.

```
It gets all 793 COPD recordings right.
It gets all 127 non-COPD recordings wrong.
Accuracy = 793 / 920 = 86%
```

**86% accuracy, and it has never once identified a healthy person.** It doesn't even read
the input.

So reporting accuracy here would be actively misleading.

## 15.3 What AUROC does instead

AUROC sidesteps imbalance completely. Here's the whole idea:

> Pick **one COPD recording** and **one non-COPD recording** at random. Ask the model to
> score both. **Did it give the COPD one the higher score?**
>
> Do that for every possible pair. AUROC is the fraction of times it got the ordering
> right.

- **0.5** = coin flip — no better than guessing
- **1.0** = perfect — always ranks correctly
- **0.874** = ranks correctly about 87% of the time

Because it always compares **one of each**, the 86/14 split can't inflate it. Our lazy
always-COPD model would score exactly 0.5.

## 15.3b And why the splitting has to be by patient

If you split **randomly by recording**, here's what happens to a COPD patient with 12
recordings:

```
~9 of their recordings go to training
~3 go to testing
```

The model sees that specific person's chest, their room, their stethoscope, their
breathing rhythm during training. Then you test it on **the same person**.

> **The analogy:** studying for an exam using the actual exam questions. You'll score well.
> It proves nothing about whether you learned the subject.

So I split by **patient**: every one of a person's recordings goes entirely to train or
entirely to test, never both.

**We measured what the wrong way costs: +0.066 AUROC** — 0.940 instead of 0.874. Six and a
half points of pure illusion.

## 15.4 The trap I nearly walked into

### Sample rate, in one line

Sample rate is **how many times per second the microphone measured the air pressure**.
44,100 Hz means 44,100 measurements every second.

### The hard physical limit

There's a rule (the Nyquist limit) that says:

> A recording made at rate **R** can only contain frequencies up to **R ÷ 2**.

So a **4,000 Hz** recording physically cannot contain anything above **2,000 Hz**. Not
"quietly" — it's simply not there. The microphone never sampled fast enough to capture it.

### What I found

I checked every file's original sample rate, grouped by which stethoscope recorded it:

| Device | Files | Original rate | Max possible frequency |
|---|---|---|---|
| AKGC417L | 646 | 44,100 Hz | 22,050 Hz |
| LittC2SE | 87 | 44,100 Hz | 22,050 Hz |
| **Litt3200** | **60** | **4,000 Hz** | **2,000 Hz** |
| Meditron | 127 | mixed | mixed |

**Litt3200 is 100% four-kilohertz. No other device is.**

### Why that's a trap

My pipeline resamples everything to 16,000 Hz. But **resampling doesn't create
information** — it just re-expresses what's there. Content above 2,000 Hz was never
captured, so it stays absent.

My mel spectrogram covers 50 Hz to 8,000 Hz across 64 bands. Work out where 2,000 Hz falls
on the mel scale and it sits about **52% of the way up**.

> **⚠️ Correction for your script:** you currently say "the top third of the mel bins are
> simply empty." It's closer to **the top half — roughly 30 of the 64 bands.** Say "the top
> half" or "about half the mel bands".

So every Litt3200 file arrives at the model with **half its picture blank**, in exactly the
same place, every time.

### Why I had to control for it

The headline experiment (slide 18) asks: *do the embeddings encode which device was used?*

Without controlling for this, the answer would trivially be **yes** — but for a boring
reason that has nothing to do with the model. Any system can spot "half the image is
blank." I'd be measuring the resampler, not the representation.

So I ran it twice: once naively, once after **dropping every low-sample-rate file** so all
remaining recordings share the same bandwidth. **The gap between those two runs is the
actual measurement.**

And it's what makes the slide-18 result meaningful: the device signal *survived* the
control.

---

# SLIDE 16 — Experiment A, and the big caveat

## The three numbers

| Protocol | AUROC | What it means |
|---|---|---|
| **Subject-wise, 8 splits** | **0.874 ± 0.024** | The honest number |
| Random split | 0.940 ± 0.018 | Inflated by patient leakage |
| Shuffled control | 0.492 | The safety check |

### "8 resampled draws"

I ran the entire experiment **8 times**, each with a different random division of patients
into train and test, and report the average plus the spread.

**Why bother?** There are only about 30 patients in any test split. *Which* 30 you happen
to draw moves the result a lot. One split tells you about that split; eight tells you about
the method.

The `± 0.024` is that spread — how much the answer wobbles depending on the draw.

### The shuffled control

I take the labels — COPD, not-COPD — and **scramble them randomly**, so they no longer
correspond to anything. Then run the identical experiment.

The labels now carry **zero information**, so a correct pipeline must score **0.5**.

It scored **0.492**. Essentially exactly chance.

> **The analogy:** a smoke alarm. It doesn't tell you the house is nice. It tells you the
> house isn't on fire.

If this had come out at, say, 0.65, it would mean information was leaking somewhere — a
bug in the splitting, duplicated rows, statistics computed across train and test. Every
other number in the study would be worthless. It's cheap and it has caught real bugs in
published work.

## The caveat — why my number can't be compared to theirs

This is the most important thing on the slide, so it's worth being precise about *what*
differs.

### Difference 1 — which audio I feed in

Recordings run from 8 seconds to 86 seconds. I take **one 8.192-second clip from the
middle** of each.

The paper doesn't specify its scheme, and there are many reasonable ones — several clips
per recording averaged together, clips from the start, a sliding window. **Each would give
a different number.**

Taking one centred clip from an 86-second recording throws away 90% of the audio. If they
use all of it, they have more information than I do.

### Difference 2 — how I turn 1024 patches into one description

The encoder outputs 1024 patch descriptions plus the CLS token. To get one embedding per
recording, you must combine them. I **average the 1024 patches** and ignore CLS.

They might use CLS, or a weighted combination, or something else. Different choice,
different embedding, different number.

### Difference 3 — how the data is divided

I use my own patient-wise splits, 25% held out, stratified, with my own random seeds. They
describe theirs as "participant-independent random" — the right idea, but a different
fraction and different seeds means a different set of test patients.

Given the `± 0.024` spread we measured across splits, this alone can move the result by
several points.

### So what does that add up to?

**Three different choices, each of which independently moves the number.** Two numbers
produced under different protocols aren't comparable, even if both are honestly computed.

Which is why I don't quote a published figure anywhere — not out of excessive caution, but
because a comparison would be **meaningless**, and presenting it as one would mislead.

## Why the fix is cheap

The expensive step in this whole pipeline is running 920 recordings through a 21-million
parameter transformer. That produces a **920 × 384 table of numbers**.

**I save that table to disk.** Everything after it — every probe, every experiment on
these slides — is arithmetic on that saved table, and takes seconds.

So matching their protocol means: read their code, change how clips are picked and how
splits are built, re-run. Depending on whether clip selection changes, either a few
seconds or a few minutes of recomputation. **Hours of work, not a rebuild.**

That's why it's item one on the next-steps list.

---

# The one-line version of each

**Slide 14** — Three checks. The parameter count is a fingerprint of the architecture and
mine matches theirs exactly (and the arithmetic adds up by hand). Strict loading proves
every one of 21.7M slots was filled from their file, with an exception rather than a silent
blank. And the embeddings behave sensibly — same patient looks more like same patient.

**Slide 15** — Half the *patients* have COPD but 86% of the *recordings* do, so accuracy is
meaningless and splits must be by person. And one stethoscope recorded everything at 4 kHz,
which leaves half the spectrogram blank — a giveaway that has nothing to do with the model,
and which I had to control for before the device experiment meant anything.

**Slide 16** — 0.874 with correct splitting, 0.940 with leaky splitting, 0.492 on scrambled
labels. But my clip selection, pooling and splits all differ from theirs, so the number
isn't comparable to their published one — and I say so rather than implying a reproduction
I didn't do.
