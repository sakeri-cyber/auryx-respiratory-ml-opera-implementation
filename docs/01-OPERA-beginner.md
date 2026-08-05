# OPERA — The Beginner's Read

**Paper:** *Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking*
Yuwei Zhang, Tong Xia, Jing Han, Yu Wu, Georgios Rizos, Yang Liu, Mohammed Mosuily, Jagmohan Chauhan, Cecilia Mascolo
NeurIPS 2024, Datasets & Benchmarks Track · [arXiv:2406.16148](https://arxiv.org/abs/2406.16148) · [github.com/evelyn0414/OPERA](https://github.com/evelyn0414/OPERA)

> Read this first, in one sitting. It gives you the shape of the paper without the mathematics. Document 02 then goes back over the same ground properly.

---

## 1. The problem, in plain terms

Your body makes noise. Your lungs make noise when you breathe. Your heart makes noise when it beats. When something goes wrong — fluid in the lungs, narrowed airways, a stiff valve — **that noise changes**, and it changes in ways a trained clinician can hear through a stethoscope.

So: could a machine hear it too? And if it could, you would no longer need a clinic visit to be screened, because everyone already carries a microphone.

That's the promise. The obstacle is data.

### Why this is hard: labels are expensive

To train a model to detect COPD from lung sounds, you traditionally need thousands of lung recordings **where someone has already diagnosed COPD**. That means recruiting patients, getting ethics approval, running spirometry, and having a clinician confirm each label. It is slow, expensive, and legally fraught.

The result is that respiratory audio datasets are *tiny*. ICBHI — the standard benchmark in this field — has **920 recordings**. Compare that to ImageNet's 14 million images. You cannot train a large modern neural network on 920 of anything; it will simply memorise them.

And it gets worse. Every research group collects its own small dataset, trains a bespoke model on it, reports a number, and moves on. Nothing transfers. Nobody can build on anyone else's work. The field runs in place.

### The idea: separate "learning what breathing sounds like" from "learning what disease sounds like"

This is the key conceptual move, and everything else in the paper follows from it.

There is a lot of respiratory audio in the world with **no diagnostic label** — coughs, breaths, exhalations collected during COVID-era studies, crowdsourced by the hundred thousand. It is not directly useful for training a COPD detector, because none of it says "this person has COPD."

But it is enormously useful for a different purpose: **learning the general structure of respiratory sound.** What a cough is. How breaths differ from each other. Which acoustic variations are meaningful and which are just microphone noise.

So you split the problem in two:

1. **Pretraining** (no labels needed, done once, expensive): train a model on 400 hours of unlabelled respiratory audio to produce a good general-purpose *representation* of respiratory sound.
2. **Downstream task** (few labels needed, done many times, cheap): take that frozen representation and train a tiny classifier on top of it for your specific question — COPD? smoker? COVID?

The expensive part happens once. Everyone else reuses it. That reusable pretrained model is what people mean by a **foundation model**.

This already existed for text (BERT, GPT) and general audio (AudioMAE, CLAP). **It did not exist, openly, for respiratory audio.** That gap is what OPERA fills.

---

## 2. What "a representation" actually means

This word does a lot of work in the paper, so it's worth being concrete.

A 4-second audio clip at 16 kHz is 64,000 numbers. Those numbers are a terrible description of the sound: shift the recording by a hundredth of a second and every single number changes, though the sound is identical to a human ear.

A **representation** (or *embedding*) is a much shorter list of numbers — say 768 — computed from the audio, chosen so that:

- clips that *sound* similar have similar embeddings;
- clips that sound different have distant embeddings;
- irrelevant variation (timing offsets, volume, background hum) is discarded.

If you have a good representation, downstream tasks become almost trivial. Detecting COPD stops being "learn to hear disease from raw audio with 200 examples" and becomes "draw a line through a cloud of 768-dimensional points" — which 200 examples is plenty for.

**The entire value of a foundation model is that it turns hard problems into easy ones by pre-digesting the input.**

---

## 3. From sound to picture: the spectrogram

Neural networks do not consume raw waveforms well. Almost all audio ML first converts sound into a **spectrogram**, which you should think of as *a picture of the sound*.

Chop the audio into short overlapping windows (OPERA uses 64 ms windows, stepping 32 ms). For each window, ask: how much energy is at each frequency? Stack the answers side by side.

You get a 2-D grid: **time along one axis, frequency along the other, brightness = energy**. A wheeze — a sustained high-pitched whistle — shows up as a bright horizontal streak. A crackle — a brief click — shows up as a thin vertical line.

OPERA uses 64 **mel** frequency bands rather than raw linear frequency. The mel scale spaces bands the way human hearing works: finely at low frequencies, coarsely at high ones. This is a deliberate bias toward what humans find perceptually meaningful, and it's standard practice.

**Concretely in OPERA:** 4 seconds of audio at 16 kHz becomes a spectrogram of shape **1 × 126 × 64** — one channel, 126 time steps, 64 frequency bands.

Once your sound is a picture, you can use everything the computer vision community spent a decade building. That is the whole trick, and it is why the models below are CNNs and vision transformers rather than anything audio-specific.

---

## 4. How do you learn without labels?

We have 400 hours of unlabelled respiratory audio. How does a model learn anything from data with no answers attached?

This is **self-supervised learning**: invent a task where the data supplies its own answer. OPERA uses two different inventions, which is why there are three models.

### Strategy A — Contrastive learning: "which two clips came from the same recording?"

Take a spectrogram. Cut two segments out of it. Those two segments came from the same recording of the same person, so call them a **positive pair**. Segments from *different* recordings are **negative pairs**.

Now train the model on this task: given a segment, pick out its partner from a lineup of unrelated segments.

To succeed, the model has to build a representation where "two bits of the same person's breathing" land close together and "bits of different recordings" land apart. It can't do that by memorising — it has to actually learn what makes a recording distinctive. **Nobody had to label anything**; the answer key came free from how the data was cut.

This is the approach behind **OPERA-CT** and **OPERA-CE**.

### Strategy B — Generative learning: "fill in the parts I've hidden"

Take a spectrogram, chop it into small patches like a jigsaw, and **hide 70% of them**. Ask the model to reconstruct what was behind the mask.

To do that well, the model must understand how respiratory sound is structured over time and frequency — if a wheeze occupies a certain frequency band before and after the hole, it should continue through it. Again, no labels: the answer is the original spectrogram.

This is **masked autoencoding**, the same idea behind BERT and AudioMAE, and it's what **OPERA-GT** uses.

### Why both?

Because they learn different things, and the paper's results show it clearly.

Contrastive learning asks *"what makes recordings different from each other?"* — it learns **discriminative** features, good for telling categories apart.

Masked reconstruction asks *"what does respiratory sound look like in general?"* — it learns **descriptive** features that retain finer detail, because you cannot reconstruct a spectrogram you only vaguely understand.

You'd guess the first is better for classification and the second for fine-grained measurement. **That is exactly what happens**, and it's one of the more satisfying findings in the paper.

---

## 5. The three models

| Model | Full name | Architecture | Parameters | Pretraining |
|---|---|---|---|---|
| **OPERA-CT** | Contrastive Transformer | Transformer encoder | 31M | Contrastive |
| **OPERA-CE** | Contrastive Efficient | EfficientNet-B0 (CNN) | ~4M | Contrastive |
| **OPERA-GT** | Generative Transformer | ViT encoder (21M) + Swin decoder (12M) | 21M encoder | Masked reconstruction, 70% masked |

Two axes are being varied: **what you learn** (contrastive vs generative) and **how big the model is** (31M vs 4M).

Note **OPERA-CE is deliberately tiny** — around 4M parameters, roughly an eighth of OPERA-CT. That is not an accident. This is a mobile-systems lab; they care about models that run on a phone or a wearable. Keep OPERA-CE in mind, because it matters enormously for anything on-device.

---

## 6. The pretraining data

**135,944 samples · 404.1 hours**, assembled from five existing sources:

| Source | Contents |
|---|---|
| COVID-19 Sounds | 40,866 coughs + 36,605 deep breaths |
| UK COVID-19 | 19,533 coughs + 20,719 exhalations |
| COUGHVID | 7,179 coughs |
| HF Lung | 10,554 lung sounds |
| ICBHI | 538 lung sounds |

Everything resampled to 16 kHz mono, converted to 64-band mel spectrograms.

**Two things to notice, because they matter later:**

First, the mix is overwhelmingly **coughs and breaths** — the COVID-era crowdsourced recordings dominate. Actual stethoscope lung sounds are a small minority. That will shape what the models are good at.

Second — and flag this — **ICBHI appears in the pretraining set.** It also appears in the evaluation benchmark. Hold that thought for document 02.

---

## 7. How do you prove a foundation model is any good?

You cannot evaluate a representation directly. So the field uses **linear probing**:

1. Freeze the pretrained encoder completely. No further training of it.
2. Push your labelled task data through it to get embeddings.
3. Train exactly **one fully-connected layer** on those embeddings.
4. Measure how well that single layer does.

Why deliberately hobble yourself with one layer? Because that's the point. A single linear layer can barely do anything on its own — so if it performs well, **the credit belongs to the representation, not the classifier.** It's a clean measurement of embedding quality.

It's also the honest test of the foundation-model promise: *can someone with 200 labelled examples and no GPU get a good result by reusing your encoder?*

### The benchmark: 10 datasets, 19 tasks

**12 health-condition classification tasks (T1–T12)** — COVID status from cough and from exhalation, symptom presence, COPD detection, smoker status, obstructive airway disease, COPD severity, sleep body position from snoring, and gender from cough. Scored with **AUROC**.

**7 lung-function regression tasks (T13–T19)** — predicting spirometry values (FVC, FEV1, FEV1/FVC) from deep breaths and from a sustained vowel, plus respiratory rate. Scored with **MAE**.

That spread is deliberate: different sound types (cough, breath, lung, snore), different diseases, different data sources, and both classification and regression. A model that only wins on coughs would be exposed.

### AUROC in one line

For classification they use **AUROC** — roughly, *"pick one sick person and one healthy person at random; how often does the model score the sick one higher?"* 0.5 is coin-flipping. 1.0 is perfect. It's used instead of accuracy because these datasets are imbalanced, and on imbalanced data accuracy is close to meaningless.

---

## 8. The results

Overall ranking across all 19 tasks (higher is better — it's a Mean Reciprocal Rank, essentially "how often is this model the best one"):

| Model | All tasks | Health inference | Lung function |
|---|---|---|---|
| OpenSMILE (hand-crafted features) | 0.2912 | 0.2190 | 0.4150 |
| VGGish (general audio) | 0.2289 | 0.1714 | 0.3276 |
| AudioMAE (general audio) | 0.2489 | 0.2058 | 0.3228 |
| CLAP (audio-language) | 0.3435 | 0.4319 | 0.1918 |
| **OPERA-CT** | **0.5632** | **0.6944** | 0.3381 |
| OPERA-CE | 0.4412 | 0.4153 | 0.4857 |
| OPERA-GT | 0.5298 | 0.4569 | **0.6548** |

**Headline claims:**
- The best OPERA model beats hand-crafted acoustic features on **17 of 19** tasks.
- It beats every general-audio pretrained baseline on **16 of 19** tasks.
- OPERA models clear AUROC 0.70 on 6 of 12 classification tasks; the best baseline manages 3.

**The interesting structure:** OPERA-CT (contrastive) dominates classification. OPERA-GT (generative) dominates regression. Exactly the split predicted in §4 — discriminative pretraining for discriminative tasks, descriptive pretraining for fine-grained measurement.

**And they report their losses**, which is to their credit. On **T7 — COPD detection from ICBHI lung sounds — AudioMAE, a general-audio model, wins with 0.886.** A general model beating the specialist on lung sounds is a genuinely odd result. Document 02 digs into why.

---

## 9. What to carry into Auryx

Auryx turns earbuds into health monitors using sound. Mascolo is a co-founder and OPERA's senior author. So this paper is close to a public statement of the science underneath the company. Four things to hold onto:

**1. The data bottleneck is the whole game.** Auryx will never have millions of labelled in-ear recordings. Their entire technical strategy has to be about extracting maximum value from limited labels. That's what OPERA is for.

**2. OPERA-CE exists because efficiency matters.** A 4M-parameter model that gets most of the way there is more valuable to a wearable company than a 31M one that scores marginally better. The paper does not report what these models cost to *run* — no latency, no memory, no on-device numbers. For a company whose product is continuous monitoring on an earbud, that missing measurement is conspicuous.

**3. Different pretraining suits different endpoints.** Auryx wants heart rate, HRV, respiration, cardiovascular parameters — mostly *continuous estimation*, which is regression. The paper says generative pretraining wins at regression. That's a directly actionable finding for them.

**4. Domain shift is unsolved and everywhere.** OPERA pretrained mostly on phone-recorded coughs and breaths, and its worst result is on stethoscope lung sounds. Auryx faces the same problem an order of magnitude worse: every earbud model has a different microphone, a different seal, different onboard DSP. **How well do these representations survive a change of recording device?** The paper does not really answer this, and for Auryx it may be the most important question there is.

---

## 10. Read next

- **Document 02** — the same paper, properly: the mathematics, why each design choice was made over the alternatives, and where the evaluation is weaker than it looks.
- **Document 03** — the implementation plan.

Before moving on, check you can answer these in your own words:

1. Why can't you just train a big network directly on ICBHI?
2. What does "self-supervised" actually mean, and where does the training signal come from?
3. Why does linear probing use only one layer?
4. Why is OPERA-CT better at classification while OPERA-GT is better at regression?
5. Why should it bother us that ICBHI is in both the pretraining set and the benchmark?
