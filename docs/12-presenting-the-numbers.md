# How to present the numbers — a delivery guide

Slides 16, 18, 20, 21, 22. This is about *saying it out loud*, not the maths (that's in
`11-numbers-explained.md`). For each slide: the one thing to land, exact spoken lines, and
the **imply → infer → learn** breakdown you asked for.

---

## The one technique that carries all five slides

**Say the meaning first. Let the number land second, as proof.**

Beginners switch off at a bare number and switch on at a plain sentence. So never open
with "0.874." Open with what it *means*, then drop the number to back it up.

| Don't say | Say instead |
|---|---|
| "AUROC was 0.874." | "It told sick from healthy about 87 times out of 100 — 0.874." |
| "Device lift was 0.715." | "It could still spot the microphone almost as well as ever — even after I took the shortcut away." |
| "Latency was 125 ms." | "It takes about an eighth of a second per clip." |

Three more habits:
- **One number per breath.** Never stack three in a sentence.
- **Give the ruler before the measurement.** "Half is a coin-flip, one is perfect —
  we got 0.87." Now the listener has a scale to place it on.
- **For comparisons, the comparison IS the point.** Say "device beats disease," not two
  separate numbers and let them do the subtraction.

---

# SLIDE 16 — COPD detection

### The one thing to land
"It works — and I can prove I'm not fooling myself."

### How to say the numbers
> "The scale first: a half is a coin-flip, a one is perfect. We got **0.87** — so it tells
> a sick recording from a healthy one about 87 times out of 100. And remember, this model
> was never actually trained on lung disease. That's the encouraging part.
>
> Now, if I let the *same patient* sneak into both training and testing, the score jumps to
> 0.94. That looks better, but it's fake — the model is just recognising people it's
> already met, not learning disease.
>
> And the last one is my safety check. I scrambled the labels into nonsense and re-ran. It
> scored 0.49 — a coin-flip. Which is exactly what I want: it proves the pipeline can't
> find a pattern that isn't there."

### Imply → Infer → Learn
- **Implies:** the frozen embeddings genuinely carry respiratory-health information.
- **Infer:** how you split the data changes the score by ~6 points *with no change in
  skill* — so the split matters as much as the model.
- **Learning:** *"The honest number is 0.87. The higher one is a trap, and I built the
  test that proves my honest number is clean."*

---

# SLIDE 18 — What else do the embeddings encode? (the headline)

### The one thing to land
"The model pays more attention to the microphone than to the disease."

### How to say the numbers
Slow down here — this is the centrepiece.

> "I asked a different question. Forget disease for a second — can the model tell which
> *stethoscope* recorded the clip? It shouldn't need to, for a health task.
>
> It could — almost perfectly. But part of that was a cheat: one device recorded at low
> quality, so it's obvious from the blank half of the picture. So I removed those files and
> asked again.
>
> And it *still* identified the device almost as well. That's the key moment — the cheat is
> gone, and the signal survived. So the model really does encode which microphone was used,
> for real reasons.
>
> Here's the part that matters. On the same data, it reads the *device* more strongly than
> it reads the *disease* — by roughly one-and-a-half times. A health model paying more
> attention to the hardware than to the health.
>
> And just to be sure this probe isn't reading everything — I tried chest location, and it
> barely could. So device isn't a fluke. It's special."

### Imply → Infer → Learn
- **Implies:** the encoder has baked recording-device identity deep into its
  representation, not just as an obvious artifact.
- **Infer:** for a product running on many different earbuds, that's a real risk — the
  model may key off *what recorded the sound* rather than *what's in it*.
- **Learning:** *"The device signal survived the control, and it's stronger than the
  disease signal. That's the one finding I'd most want your eyes on."*

> **Tone note:** this is a question *for them*, not a verdict *on them*. End on "I'd love
> to know if this is something you've already accounted for," not "your model has a flaw."

---

# SLIDE 20 — What does it cost to run?

### The one thing to land
"I measured something the paper didn't — can this actually run on an earbud?"

### How to say the numbers
> "This one's about speed and size, not accuracy. On a laptop, turning one clip into its
> numbers takes about an eighth of a second — 125 milliseconds.
>
> Is that fast or slow? Depends. If you check someone's breathing once every 30 seconds,
> it's nothing. But for *always-on* monitoring on a tiny earbud chip running off a battery,
> it's probably too slow. So it's not a good or bad number — it's a *design constraint*
> nobody had written down.
>
> And the model is about 83 megabytes, which is quite big for a device that small."

Skip the IQR out loud unless asked — it's a trust detail, not a headline. If someone asks
why "median": *"a couple of runs randomly hiccup and drag an average up, so I report the
middle value — it's the honest one."*

### Imply → Infer → Learn
- **Implies:** the model is comfortably runnable on a laptop, questionable on a wearable.
- **Infer:** getting from a benchmark model to an on-device product has a real
  speed-and-size gap to close.
- **Learning:** *"The paper optimises for accuracy and never reports cost. For a wearables
  company, cost is a first-order question — so I measured it."*

---

# SLIDE 21 — Foundation model vs a tiny CNN

### The one thing to land
"A model 358 times smaller beat it — but I'll tell you why not to over-read that."

### How to say the numbers
> "I trained a tiny model from scratch — about 60,000 numbers — and put it next to the big
> foundation model, which has 21 *million*. So one is roughly 358 times bigger than the
> other. Picture a 358-page book versus a single page.
>
> The single page won. It scored about 0.95, against the big model's 0.87.
>
> But I want to be careful, because this is easy to over-read. It's not a totally fair
> fight — the big model was tested with one hand tied behind its back, by design. And my
> little model might be using that same device shortcut from earlier. So I would *not* say
> 'foundation models don't work.' I'd say something narrower: *on this one task, with this
> little data, the expensive pretraining didn't pay for itself.*"

### Imply → Infer → Learn
- **Implies:** for this specific task and data size, pretraining bought no advantage.
- **Infer:** foundation models aren't automatically better — the benefit depends on the
  task and how you use them.
- **Learning:** *"The interesting result isn't 'small model wins.' It's that I know exactly
  why that comparison is unfair, and I said so instead of taking the flattering headline."*

> This is where restraint reads as competence. Claiming the small model "beats" the paper
> would sound naive to a research team. Naming the caveats sounds like a colleague.

---

# SLIDE 22 — The mistake

### The one thing to land
"A number looked too perfect, and chasing that caught a real error."

### How to say the numbers
This is the most human slide. Tell it like a short story.

> "My first run gave me a spread of exactly zero. Every repeat, identical. And at first
> that felt great — looks incredibly precise.
>
> But it bothered me, because real measurements always wobble a little. A perfectly flat
> zero usually means you're not measuring the thing that's supposed to vary.
>
> And that's exactly what had happened. I was re-rolling the wrong dice — a setting that
> doesn't actually change the answer — so all my 'repeats' were the same run eight times.
>
> When I fixed it and varied the thing that *does* matter — which patients go in which
> group — one of my conclusions changed by sixteen times. Something I'd have called
> negligible turned out to be a real effect.
>
> I'm showing you this on purpose. I caught it because a number looked *too* clean."

### Imply → Infer → Learn
- **Implies:** the first run's "stability" was an illusion — a stuck measurement.
- **Infer:** *what* you randomise decides whether your error bars mean anything at all.
- **Learning:** *"Be suspicious of results that look too good. The zero wasn't precision —
  it was a broken thermometer, and noticing that changed a real conclusion."*

> **Why lead with your own mistake?** Because it's the most convincing slide you have. Any
> candidate can show a good score. Showing that you audit your own work, distrust
> suspiciously clean numbers, and fix conclusions when the evidence changes — that's what a
> research team is actually hiring for.

---

# The whole arc in five spoken sentences

If you had to compress all five slides to one line each:

1. **(16)** "It detects COPD well — 87 out of 100 — and I built the test that proves that's honest."
2. **(18)** "But it reads the microphone more strongly than the disease, and that signal is real, not an artifact."
3. **(20)** "It's an eighth of a second per clip — fine occasionally, maybe too slow always-on — a cost the paper never reported."
4. **(21)** "A model 358 times smaller beat it here — though I'll tell you exactly why that comparison isn't fully fair."
5. **(22)** "And a suspiciously perfect number turned out to be my own mistake, which — once fixed — changed a conclusion sixteenfold."

Notice the shape: **it works → but here's something worth worrying about → here's what it costs → here's a humbling comparison → and here's me checking my own work.** That arc — competence, then curiosity, then honesty — is far more persuasive than five slides of good scores.

---

# Three delivery rules to carry across all five

1. **Meaning first, number second.** Every single time.
2. **Say what the comparison proves, not the two numbers.** "Device beats disease." "The
   small one won." "It changed sixteenfold." The listener should never have to do the
   subtraction.
3. **End every slide on one plain takeaway sentence**, starting with "So the learning
   here is…". It gives the listener a place to file what they just heard before you move on.
