# Slides 11, 12, 13 — explained slowly

Plain language, building from the beginning. Read in order; each part uses the one before.

---

# SLIDE 11 — Recovering the architecture from the weights

## First: what is a "checkpoint"?

A neural network is, physically, just a very large pile of numbers.

When someone "releases a model", they release a file containing those numbers. That file
is called a **checkpoint**. OPERA-GT's is 394 MB — that's 34 million numbers.

But the numbers aren't in one big heap. They're in **named groups**. Open the file and you
get something like a dictionary:

```
"patch_embed.proj.weight"     →  a block of numbers
"blocks.0.attn.qkv.weight"    →  a block of numbers
"blocks.0.mlp.fc1.weight"     →  a block of numbers
...
```

Each block has a **shape** — its dimensions. Like a spreadsheet being "1152 rows by 384
columns".

## The problem I had

The checkpoint gives you the numbers and the names. It does **not** give you the code that
says how to wire them together.

> **The analogy:** someone hands you a box of Lego with no instruction booklet. But every
> piece is labelled, and you can measure each one. From the labels and the measurements,
> you can work out what the finished thing must have been.

That's what "recovering the architecture from the weights" means. I read the shapes and
worked backwards to the structure.

## Why shapes tell you the structure

Here's the key idea, and everything else follows from it.

A neural network layer is mostly **a matrix multiplication**. If a layer takes 384 numbers
in and gives 1152 numbers out, its weight matrix must be **1152 × 384**. There's no choice
about it — that's just how the arithmetic works.

**So: read the shape → know the input and output sizes → know what the layer does.**

## Now let's read each one

### 1. `patch_embed.proj.weight` — shape (384, 1, 4, 4)

Four numbers in the shape means this is a **convolution**. Convolution weights always have
the shape:

```
(output channels, input channels, kernel height, kernel width)
```

So reading it off:

| Position | Value | Meaning |
|---|---|---|
| output channels | 384 | produces 384 numbers |
| input channels | **1** | takes a **1-channel** image |
| kernel height | 4 | looks at a 4-pixel-tall region |
| kernel width | 4 | ...and 4 pixels wide |

**In plain English:** take the input picture, chop it into little 4×4 squares, and turn
each square into 384 numbers.

Why 1 input channel? A colour photo has 3 channels (red, green, blue). A spectrogram is a
single greyscale picture — so 1.

Each 4×4 square is called a **patch**. This layer is the "patch embedder": it converts
picture-patches into number-lists that a transformer can chew on.

### 2. `pos_embed` — shape (1, 1025, 384)

**Positional embedding.** To understand why this exists, you need one fact about
transformers:

> A transformer has **no built-in sense of order**. If you shuffle the input pieces, it
> produces the same answer, just shuffled. It genuinely cannot tell which piece came first.

That's a problem — in a spectrogram, *where* a sound happens obviously matters.

The fix: before processing, **add a "you are here" vector to every piece**. Each position
gets its own learned signature. Position 1 gets one set of 384 numbers, position 2 gets a
different set, and so on. The network learns these during training.

So the shape means: **1025 positions, each with 384 numbers.**

**But why 1025 and not a round 1024?**

Because there's one extra piece that isn't part of the picture. It's called the **CLS
token** (short for "classification").

> **The analogy:** imagine 1024 people in a meeting, each with something to say. The CLS
> token is an extra empty notepad you put on the table. It isn't a person, it doesn't have
> its own opinion — but as the meeting runs, notes get written on it. At the end, you can
> read the notepad to get a summary of the whole meeting.

So: **1024 real patches + 1 summary notepad = 1025.**

*(A detail that becomes important on slide 14: in OPERA-GT the notepad is never actually
read during training — the training task is reconstructing hidden patches, not
summarising. So I average the 1024 real patches instead of using the notepad. In a
contrastive model like OPERA-CT the opposite would be true.)*

### 3. `blocks.0.attn.qkv.weight` — shape (1152, 384)

This one's in the **attention** part, which is the heart of a transformer.

The way attention works, every piece of the input needs **three different versions of
itself**:

| Name | What it's for | Loose analogy |
|---|---|---|
| **Query** | what this piece is looking for | "I'm searching for X" |
| **Key** | what this piece advertises about itself | "I contain X" |
| **Value** | what this piece actually passes on | the content itself |

Attention then matches everyone's Query against everyone else's Key, and wherever there's
a good match, it pulls across that piece's Value.

> **The analogy:** a room full of people. Each person has a question (Query), a badge
> saying what they know (Key), and actual knowledge (Value). Everyone reads everyone's
> badge, finds who can answer their question, and listens mainly to those people.

Now the shape. Each of Query, Key and Value is 384 numbers. Three of them:

```
3 × 384 = 1152
```

So instead of three separate layers, they built **one layer that outputs all three at
once**, and the code slices the 1152 into three chunks of 384. That's what "**fused
QKV**" means. It's done because one big matrix multiply is faster than three small ones.

**Seeing 1152 = 3 × 384 is exactly how you know it's fused.** If it had been three separate
weights of 384 × 384 each, you'd have seen three separate entries in the checkpoint.

### 4. `blocks.0.mlp.fc1.weight` — shape (1536, 384)

After attention, each piece goes through a small two-layer network on its own. It's a
sandwich:

```
384 numbers  →  expand to 1536  →  squeeze back to 384
```

Why expand then squeeze? Roughly: the wide middle gives the network **room to think**, and
squeezing back keeps the size consistent so you can stack many blocks.

The **MLP ratio** is just how much it expands:

```
1536 ÷ 384 = 4
```

Ratio 4 is the standard, used by almost every transformer.

### 5. `blocks.11.*` exists — and `blocks.12.*` doesn't

The checkpoint has entries named `blocks.0.…` through `blocks.11.…`, then stops.

Counting from zero, that's **12 blocks**. So the encoder stacks the
attention-then-MLP sandwich 12 times.

## Putting it together

Here's what I now knew, purely from shapes:

| Property | Value |
|---|---|
| Depth (blocks) | 12 |
| Width (dimension) | 384 |
| MLP ratio | 4 |
| Patch size | 4 × 4 |

**This exact combination has a name: ViT-Small.** It's a standard published model size,
like "medium" on a coffee menu. There's ViT-Tiny, ViT-Small, ViT-Base, ViT-Large. Anyone
building a ViT picks one of these rather than inventing dimensions.

And ViT-Small conventionally uses **6 attention heads**.

## The one thing I could NOT read — and why

This is the part worth understanding properly, because it's the honest caveat on the
slide.

**What are heads?** Attention doesn't do one big comparison. It splits the 384 dimensions
into groups, and each group looks for a *different kind* of relationship independently.

With 6 heads: 384 ÷ 6 = **64 dimensions per head**.

> **The analogy:** six people read the same document, each looking for something different
> — one for names, one for dates, one for tone. Then you combine their notes. You learn
> more than one person reading for everything at once.

**Now the crucial bit: the head count is invisible in the weights.**

Look at what happens with different head counts:

| Heads | Dimensions per head | QKV weight shape |
|---|---|---|
| 6 | 64 | **1152 × 384** |
| 8 | 48 | **1152 × 384** |
| 12 | 32 | **1152 × 384** |

**All identical.** The splitting into heads happens *after* the matrix multiply, inside the
code — it's a reshape, not a separate set of numbers. So the checkpoint simply doesn't
record it.

I chose 6 because that's the ViT-Small convention and the rest of the architecture matches
ViT-Small exactly. It is very probably right. But **"very probably" is not "verified"**,
so I mark it as inferred everywhere.

---

# SLIDE 12 — Where the input shape comes from

## The setup

I now know the model chops its input into **4 × 4 patches**, and that there are exactly
**1024** of them.

What I *don't* yet know is what size picture goes in.

## The arithmetic

If the input picture is **H tall and W wide**, then chopping into 4×4 squares gives:

```
patches down  =  H ÷ 4
patches across =  W ÷ 4
total patches  =  (H ÷ 4) × (W ÷ 4)
```

And we know the total must be 1024.

The paper states OPERA uses **64 mel bins** — meaning the picture is 64 tall. So:

```
(64 ÷ 4) × (W ÷ 4)  =  1024
      16 × (W ÷ 4)  =  1024
          (W ÷ 4)   =  64
                W   =  256
```

**So the input is 64 tall by 256 wide.**

## Turning that into seconds

The width is time. Each column of the spectrogram is one "frame", and the paper says
frames step every **32 milliseconds**:

```
256 frames × 32 ms  =  8192 ms  =  8.192 seconds
```

That's why every clip in my pipeline is 8.192 seconds. **I didn't pick that number — it
fell out of the checkpoint.** That's what "derived, not chosen" means on the slide.

## Now the honest gap

Here's the problem. Try a square picture instead:

```
128 tall, 128 wide
(128 ÷ 4) × (128 ÷ 4)  =  32 × 32  =  1024
```

**Also exactly 1024 patches.**

So both 64 × 256 and 128 × 128 fit the checkpoint perfectly. The model loads either one.
It runs on either one. Neither produces an error message.

I chose 64 × 256 because the paper explicitly says 64 mel bins, which makes it far more
likely. But I never confirmed it.

## Why being wrong here would be nasty

This is the important part.

The model doesn't *know* what the numbers mean. It just multiplies matrices. Feed it
anything with 1024 patches and it will happily compute and hand you back 384 numbers that
look completely normal.

But remember the positional embeddings from slide 11 — position #37 has a learned
signature meaning *"this is the region at this particular time and this particular
frequency"*. Those were learned for **one specific layout**.

If my layout is different, then every single patch gets given the wrong "you are here"
vector. The model is being systematically lied to about where everything is.

> **The analogy:** a jigsaw with the right number of pieces but the wrong picture. Every
> piece slots in. Nothing jams. But the image comes out wrong — and you only notice if you
> step back and look.

The output wouldn't crash, wouldn't warn, wouldn't look obviously broken. It would just be
**quietly worse**. That's the most dangerous kind of bug, which is why I flag it rather
than let it pass.

**And it's checkable in about ten minutes** — the sanity check I already have (same-patient
recordings should look more similar than different-patient ones) would very likely score
noticeably better with the correct layout. That's why I say it's the first thing I'd want
to confirm.

---

# SLIDE 13 — The audio front end

## What this stage does

Before the model sees anything, the sound file has to become a picture. That conversion is
the "front end":

```
.wav file → resample to 16 kHz → take an 8.192 s clip → STFT
          → mel filterbank → log → standardise → the picture
```

## Why not just use torchaudio?

**torchaudio** is PyTorch's audio library. It has a ready-made function that makes mel
spectrograms in one line. Using it would have been much less work.

The problem is that the function has **many options**, and where you don't specify one, it
picks a default. There are several different-but-valid ways to build a mel filterbank,
several ways to normalise, several ways to pad the edges.

If any of torchaudio's defaults differ from what OPERA actually used, my pictures come out
subtly different from the ones the model was trained on.

**And the model would not complain.** It would give me embeddings that look completely
normal and are quietly wrong — the same silent-failure problem as slide 12.

So I wrote it myself, which meant I had to decide every parameter deliberately rather than
inherit it without noticing.

The bit I was most worried about — the mel filterbank — turned out to be about five lines.

## What a mel filterbank actually is

After the FFT, you have the energy at about **513 evenly-spaced frequencies**, from 0 Hz up
to 8000 Hz.

Two problems with that:

1. **513 numbers per time-slice is a lot**, and much of it is redundant.
2. **Even spacing doesn't match hearing.** You can easily tell 100 Hz from 200 Hz. You
   cannot tell 5000 Hz from 5100 Hz — even though both are a 100 Hz gap. Low frequencies
   deserve fine detail; high frequencies don't.

So we group the 513 into **64 bands** — narrow bands down low, wide bands up high.

**Building them, in three steps:**

1. Take the frequency range and re-express it on the **mel scale** — a warped ruler where
   equal steps sound equally different to a human ear. On this ruler, low frequencies get
   stretched out and high ones get squashed together.
2. Place 66 evenly-spaced marks along that warped ruler, then convert them back to Hz.
   Because of the warping, the low-frequency marks come out close together and the
   high-frequency ones far apart — exactly what we wanted.
3. Between each set of three consecutive marks, build a **triangle**: weight rises from
   zero at the left mark to one at the middle, then falls back to zero at the right.

Each triangle is one band. Multiply the 513 energies by a triangle and add them up, and you
get that band's total energy. Do it 64 times → 64 numbers.

> **The analogy:** 64 overlapping buckets catching rain. Narrow buckets where you care
> about detail, wide buckets where you don't.

## Log compression — and why it's clever

Sound energy covers an enormous range. A loud sound might be a million times the energy of
a quiet one. Taking the **logarithm** squashes that into something manageable.

But there's a second reason, and it's the one that matters here.

**Logarithms turn multiplication into addition:**

```
log(a × b)  =  log(a) + log(b)
```

Now think about what "turning up the volume" does. Doubling the volume **multiplies** every
energy value by the same factor.

After taking the log, that multiplication becomes **adding the same constant everywhere**.

```
before log:  every value × 10
after  log:  every value + log(10)
```

**A volume change stops being a stretch and becomes a shift.** That sets up the next step.

## Per-example standardisation — the line that matters most

Now, for each spectrogram separately, I:

1. work out its average value,
2. subtract that average from every pixel,
3. divide by its standard deviation.

Because volume differences are now just "everything shifted up or down by a constant",
**subtracting the mean removes them exactly**. A quiet recording and a loud recording of
the same sound become *identical* after this step.

## Why this matters so much for this dataset

ICBHI was recorded with **four different stethoscopes**. They have very different
sensitivities — one device might record everything much louder than another.

Without standardisation, the loudness *is* the device. The model would find the easiest
possible pattern:

> "These pictures are bright → AKGC417L. These are dim → Litt3200."

It would learn to recognise **equipment**, not **disease**. And it would score well on the
training data while learning nothing medically useful.

Standardisation removes the loudness cue entirely, forcing the model to look at the
*shape* of the sound instead.

## The test on the slide

```
Take any audio.
Make version A = the audio.
Make version B = the audio × 0.01   (that's 40 dB quieter)
Run both through the front end.
Assert the two outputs are identical.
```

If that ever fails, loudness is leaking through, and everything downstream is suspect. It's
a two-line test that guards the most important property of the whole pipeline.

*(Worth being precise about the limit: this removes **volume** differences. It does **not**
remove **frequency-response** differences — if one stethoscope genuinely picks up more
high frequencies than another, that survives standardisation. Which is very likely part of
why the device probe on slide 18 still finds so much device information.)*

---

# The one-line version of each slide

**Slide 11** — The checkpoint contained numbers but no blueprint. I read the blueprint off
the numbers' dimensions, because a layer's shape tells you exactly what it does. Everything
recovered cleanly except the head count, which is genuinely invisible in the weights.

**Slide 12** — I didn't choose the input size; I solved for it. 1024 patches at 4×4 with 64
mel bins forces 64 × 256, which is 8.192 seconds. But a square 128 × 128 also gives 1024
patches, and nothing would error if I'd picked wrong — it would just be quietly worse.

**Slide 13** — I built the sound-to-picture conversion by hand so no library default could
silently differ from the paper. The most important line takes the log and then subtracts
each picture's own average, which removes volume completely — because otherwise the model
would just learn to recognise which stethoscope was used.
