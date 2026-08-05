# OPERA — Deep Technical Analysis

*Companion to `01-OPERA-beginner.md`. Assumes you've read it.*

**A note on provenance.** Where a formula or number appears in the paper, it's stated as such. Where I supply standard background the paper assumes (STFT definition, InfoNCE derivation, AUROC properties), it's marked **[background]**. Where I'm reasoning beyond the paper — critique, alternative explanations, proposed experiments — it's marked **[analysis]**. Check anything load-bearing against the source before you repeat it in an interview.

---

## 1. Formal problem setup

Let $\mathcal{D}_u = \{x_i\}_{i=1}^{N}$ be unlabelled respiratory audio, $N = 135{,}944$, total duration 404.1 hours.

Let $\{\mathcal{D}_k\}_{k=1}^{19}$ be labelled downstream datasets, $\mathcal{D}_k = \{(x_j, y_j)\}$, where $|\mathcal{D}_k|$ ranges from a few dozen to a few thousand — **three to four orders of magnitude smaller than $\mathcal{D}_u$**.

We want an encoder $f_\theta: \mathcal{X} \to \mathbb{R}^d$ trained only on $\mathcal{D}_u$, such that for every task $k$ there exists a *linear* map $W_k \in \mathbb{R}^{d \times c_k}$ with

$$\hat{y} = W_k^\top f_\theta(x)$$

achieving good performance. The constraint that $W_k$ is **linear and $f_\theta$ is frozen** is the entire scientific claim. Allow $f_\theta$ to be fine-tuned and you have merely shown that pretraining is a decent initialisation — a far weaker and much less interesting statement.

**Formally, the hypothesis is that respiratory pathology is close to linearly separable in $f_\theta$'s representation space.** That is strong, and it's what the benchmark tests.

---

## 2. The input representation

### 2.1 STFT **[background]**

Audio $s[n]$ at $f_s = 16$ kHz. The short-time Fourier transform with window $w$ of length $L$ and hop $H$:

$$X[m, k] = \sum_{n=0}^{L-1} s[n + mH] \, w[n] \, e^{-j 2\pi k n / L}$$

OPERA's stated parameters: **64 ms Hann window, 32 ms hop**. At 16 kHz that's $L = 1024$ samples, $H = 512$ — i.e. **50% overlap**.

The window length fixes the time–frequency resolution trade-off. $L = 1024$ gives frequency resolution $f_s/L \approx 15.6$ Hz and time resolution 64 ms.

**[analysis] This choice is questionable for some respiratory events.** A crackle is a transient on the order of 5–20 ms. A 64 ms window smears it across the entire frame, and adjacent 50%-overlapping frames smear it further. The parameters are well suited to *sustained* phenomena — wheezes, breath phases, cough envelopes — and poorly suited to *transients*. Given the pretraining corpus is dominated by coughs and breaths, this is internally consistent. But it predicts weakness on crackle-driven tasks, and it's one candidate explanation for the T7 result in §7.

### 2.2 Mel filterbank **[background]**

Linear frequency is mapped to the mel scale:

$$m(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

$B = 64$ triangular filters are spaced uniformly in mel, giving the mel spectrogram

$$M[m, b] = \sum_k |X[m,k]|^2 \, \Lambda_b[k]$$

followed by log compression, $\tilde{M} = \log(M + \epsilon)$.

Log compression matters more than it appears: it converts multiplicative gain into an additive offset. Recording level then becomes a shift rather than a scale, which per-example normalisation removes cleanly. **[analysis]** This is the mechanism by which spectrogram models get partial invariance to microphone gain — but only to *gain*, not to *frequency response*. Two microphones with different frequency responses produce different mel spectrograms that no amount of normalisation will reconcile. Remember this for §8.

### 2.3 Resulting tensor

4 seconds → $\lceil (4000 - 64)/32 \rceil + 1 = 124$ frames; the paper states **1 × 126 × 64** (minor padding conventions account for the difference). One channel, 126 time steps, 64 mel bands.

**[analysis] Note the aspect ratio.** 126 × 64 is not square, and the two axes are not commensurable — adjacency in time means something physically different from adjacency in frequency. Every architecture below inherits this from computer vision, where both axes *are* spatial and translation-equivariance is appropriate on both. On a spectrogram, translation in time is benign (a sound shifted later is the same sound) but translation in frequency is **not** (a wheeze shifted up an octave is a different wheeze). Convolutions and ViTs apply the same operator to both. This is a known, generally-tolerated mismatch in audio ML, and it is a legitimate line of criticism.

---

## 3. Contrastive pretraining (OPERA-CT, OPERA-CE)

### 3.1 Positive pair construction

From one spectrogram, two segments are cropped; they form a positive pair. Segments from different recordings are negatives.

**[analysis] This is a consequential definition and worth dwelling on.** The invariance you get is exactly the invariance you build into the positive pairs. Here the pair differs *only in temporal position within the same recording*. Therefore the model is trained to be invariant to **when** within a recording you listen — and to nothing else.

It is explicitly **not** trained to be invariant to:
- recording device
- microphone placement
- subject identity
- background acoustics
- session

All of those are constant within a recording, so they are **perfectly predictive of the positive pair**. A representation that encoded nothing but "which recording is this" would achieve *zero contrastive loss*.

This is the well-known shortcut problem in contrastive learning. It does not mean the method fails — evidently it doesn't — but it means the representation is under strong pressure to encode recording identity, of which device and subject identity are components. **This is the theoretical basis for the device-probe experiment proposed in §9, and it is the single most Auryx-relevant observation in this document.**

Contrast with SimCLR for images, where positives are aggressive augmentations (crop, colour jitter, blur) *designed* to destroy nuisance factors. OPERA's positives involve no augmentation of this kind — only temporal cropping.

### 3.2 The COLA objective

The repo's `cola_pretraining.py` identifies this as COLA (Saeed et al., 2021). The distinguishing feature is **bilinear similarity** rather than cosine:

$$s(x, y) = g(x)^\top W g(y)$$

with learnable $W \in \mathbb{R}^{d' \times d'}$, where $g = \text{proj} \circ f_\theta$. The paper's phrasing — a projector into a low-dimensional space "where bilinear similarity is calculated" — matches.

The loss is cross-entropy over in-batch negatives **[background]**:

$$\mathcal{L}_{\text{CT}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp\big(s(x_i, x_i^+)\big)}{\sum_{j=1}^{B} \exp\big(s(x_i, x_j^+)\big)}$$

**Why bilinear over cosine? [analysis]** Cosine similarity $\frac{u^\top v}{\|u\|\|v\|}$ imposes an isotropic geometry and typically needs a temperature $\tau$ to be tuned. The bilinear form absorbs both: $W$ learns an anisotropic metric — effectively a learned Mahalanobis distance — and its scale subsumes $\tau$. COLA's original ablation found this materially better for audio. The cost is $d'^2$ extra parameters and a mild risk of degenerate $W$.

**The InfoNCE bound [background].** This objective lower-bounds the mutual information between views:

$$I(X; X^+) \ \geq\ \log B - \mathcal{L}_{\text{CT}}$$

The bound is capped by $\log B$, so batch size directly limits how much information the objective can extract. **[analysis]** With a 31M-parameter transformer on academic hardware, $B$ is likely in the hundreds — meaning $\log B \approx 5$–6 nats. This is a real constraint on contrastive methods and is why MoCo-style memory banks and CLIP-scale batches exist. The paper does not report batch size prominently; **this is worth checking in the config before you draw conclusions from it.**

---

## 4. Generative pretraining (OPERA-GT)

### 4.1 Masked autoencoding

The spectrogram is partitioned into patches; **70% are masked uniformly at random**; a ViT encoder processes only the visible 30%; a Swin-transformer decoder reconstructs the full spectrogram.

Loss on masked patches only **[background]**:

$$\mathcal{L}_{\text{GT}} = \frac{1}{|\mathcal{M}|}\sum_{p \in \mathcal{M}} \big\| \tilde{M}_p - \hat{M}_p \big\|_2^2$$

Computing loss only on $\mathcal{M}$ is essential; including visible patches lets the model win by learning the identity map.

### 4.2 Why 70%?

**[analysis]** The masking ratio sets task difficulty and therefore what gets learned.

- **Too low** (say 15%, BERT's ratio) — local interpolation suffices. Spectrograms are smooth and highly redundant along both axes, so a small hole can be filled by averaging neighbours without any semantic understanding. Text does not have this problem, which is precisely why BERT can use 15% and MAE cannot.
- **Too high** (say 90%) — the task becomes ill-posed; the model learns the dataset mean.

He et al.'s image MAE found 75% optimal; AudioMAE found 80% for general audio. OPERA's 70% is slightly below both. **[analysis] A plausible reading:** respiratory spectrograms are *less* redundant than natural images in the frequency direction — a wheeze occupies specific bands and cannot be inferred from neighbouring bands — so the task is intrinsically harder and needs less masking to stay learnable. That is a hypothesis, not something the paper argues.

### 4.3 Asymmetric encoder/decoder

Encoder 21M, decoder 12M. The decoder is discarded after pretraining. **[background]** Two reasons this design wins: the encoder only sees 30% of patches, so pretraining cost drops roughly 3×; and a *weak* decoder forces the encoder to do the semantic work rather than offloading it.

### 4.4 Why contrastive and generative differ in what they learn

This is the paper's most interesting empirical structure, and it deserves a theoretical account. **[analysis]**

Contrastive learning optimises for **instance discrimination**. Information is retained only insofar as it distinguishes recordings. Anything shared across all recordings — the general spectral envelope of breathing, absolute amplitude relationships — is *compressed away*, because it carries no discriminative signal. The result is a representation that is excellent for "which class?" and lossy about "how much?".

Masked reconstruction optimises for **input recovery**. To reconstruct a spectrogram you must retain enough to redraw it, including absolute magnitudes and fine spectral detail. Nothing is discarded for being non-discriminative.

Predicted consequence: contrastive wins classification, generative wins regression.

Observed: **OPERA-CT health-inference MRR 0.6944 vs OPERA-GT 0.4569. OPERA-GT lung-function MRR 0.6548 vs OPERA-CT 0.3381.** A near-perfect double dissociation.

**This is the most theoretically satisfying result in the paper**, and it generalises well beyond respiratory audio. It is also directly actionable for Auryx: heart rate, HRV and respiratory rate are *estimation* problems, so the generative branch is the relevant one — despite OPERA-CT winning the headline comparison.

---

## 5. Architectures

| | OPERA-CT | OPERA-CE | OPERA-GT |
|---|---|---|---|
| Encoder | Transformer | EfficientNet-B0 | ViT |
| Params | 31M | ~4M | 21M (+12M decoder) |
| Objective | Contrastive (COLA) | Contrastive (COLA) | Masked reconstruction |
| Inductive bias | Global attention | Local conv + depthwise separable | Patch attention |

**[analysis] The CT/CE pair is a controlled comparison** — same objective, same data, different capacity and inductive bias. This is the paper's cleanest ablation and its results are informative:

- Health inference: CT **0.6944** vs CE **0.4153**. Capacity matters enormously.
- Lung function: CT **0.3381** vs CE **0.4857**. **CE beats CT, despite being 8× smaller.**

That reversal is not noise-shaped; it deserves an explanation the paper does not really give. **[analysis] My hypothesis:** EfficientNet's convolutional locality preserves local spectral magnitude structure that global attention discards. Lung-function regression depends on breath-envelope amplitude and duration — local, magnitude-sensitive quantities. This is the same discriminative/descriptive axis as §4.4, arriving via architecture rather than objective. If true, it predicts that a *generative CNN* would be the best lung-function model of all. **That model does not exist in the paper. It is a genuine gap and a cheap thing to propose.**

---

## 6. Evaluation protocol

### 6.1 Linear probing

One fully-connected layer on frozen features. Fixed train/val/test split, **5 independent runs**.

**[analysis] Strengths:** cheap, reproducible, isolates representation quality, matches the low-label deployment scenario.

**Weaknesses, and they're real:**

1. **Linear probing measures linear decodability, not information content.** Information can be present but nonlinearly entangled. A representation could be strictly more informative yet probe worse. Comparing architectures with different geometries under a linear probe conflates *what is encoded* with *how it is arranged*.
2. **5 runs vary only the probe seed, not the data split.** The reported variance therefore reflects probe initialisation, not sampling variability — the dominant error source on datasets of a few hundred examples. **Confidence intervals from this procedure are much narrower than the true uncertainty.**
3. **No multiple-comparison correction** across 19 tasks × 7 models = 133 comparisons.

### 6.2 Metrics

**AUROC [background]:**

$$\text{AUROC} = \mathbb{P}\big(\hat{s}(x^+) > \hat{s}(x^-)\big) = \frac{1}{n_+ n_-}\sum_{i \in \mathcal{P}}\sum_{j \in \mathcal{N}} \mathbb{1}[\hat{s}_i > \hat{s}_j]$$

Threshold-free and insensitive to class balance — appropriate here. **[analysis] But** AUROC is also insensitive to *calibration*, and on small test sets its variance is large: with $n_+ \approx 30$, the standard error is roughly 0.05–0.08. **Many of the paper's between-model gaps are smaller than that.**

**MRR:**

$$\text{MRR} = \frac{1}{|T|}\sum_{t \in T} \frac{1}{\text{rank}_t}$$

**[analysis] MRR is the paper's most questionable choice.** It is a rank aggregation, so it discards effect size entirely: winning by 0.001 AUROC and winning by 0.20 contribute identically. A model that is marginally best on many tasks outranks one that is decisively best on a few. It also cannot express "all models are bad here" — some model always ranks first. **When you read Table 3, remember it says who wins, not by how much.** Always cross-reference the per-task tables.

### 6.3 Splits

Official splits (T1–4, T12–18); participant-independent random splits (T5–11, T19); leave-one-subject-out for the small sets (T13–19).

**[analysis]** Participant-independent splitting is the right call and to the authors' credit — it is the main defence against the subject-leakage failure mode endemic to this literature. Worth noting that the split *strategy varies by task*, which makes cross-task comparison of absolute numbers slightly apples-to-oranges.

---

## 7. The T7 anomaly

**T7 = COPD detection from ICBHI lung sounds. AudioMAE — a general-audio model — achieves 0.886, beating every OPERA variant.**

The domain-specific foundation model loses to a general one on the flagship respiratory dataset. The paper reports this honestly but does not fully explain it. **[analysis] Four candidate explanations, which you should be able to argue between:**

**(a) Corpus composition mismatch.** Of 135,944 pretraining samples, ICBHI contributes **538** — about 0.4%. HF Lung adds 10,554. So stethoscope lung sounds are roughly 8% of pretraining; coughs and breaths are the rest. OPERA is, functionally, a cough-and-breath model. AudioMAE, pretrained on AudioSet's vast general diversity, may simply have broader low-level filters. **This is the explanation I find most likely.**

**(b) STFT parameters.** As argued in §2.1, a 64 ms window is poorly matched to crackles. AudioMAE's front end may resolve transients better.

**(c) Task-representation mismatch.** COPD is a *patient-level* diagnosis, but the input is a *recording-level* clip. The label is only weakly attached to any individual sound. Aggregation strategy may matter more than the encoder.

**(d) ICBHI's own difficulty.** Only 126 patients, heavily imbalanced toward COPD. With so few subjects, participant-independent AUROC has enormous variance and 0.886 vs 0.85 may not be a real difference.

**[analysis] And a fifth issue, which is not an explanation but a confound:** ICBHI is in *both* pretraining and evaluation. If the 538 pretraining clips overlap the T7 test patients, T7 is contaminated — OPERA's number would be optimistic, making the loss to AudioMAE *even more* surprising. If they don't overlap, the comparison is clean. **The paper does not, to my reading, state this explicitly. Checking it in the code is a concrete, valuable, near-free contribution.**

---

## 8. Critical assessment

### What the paper genuinely establishes

1. **Domain-specific pretraining beats general-audio pretraining for respiratory tasks** — 16/19 is a robust margin, not cherry-picking.
2. **The objective/task-type dissociation** (§4.4). Clean, theoretically motivated, generalisable.
3. **A real public good.** Open checkpoints, open code, standardised tasks. The field's fragmentation problem is genuinely reduced by this.
4. **Honest reporting of losses.** T7 is stated plainly.

### Where it is weaker than it looks

1. **Statistical rigour.** 5 seeds, no split resampling, no correction across 133 comparisons, AUROC standard errors comparable to reported gaps. Several conclusions are under-powered.
2. **MRR hides effect sizes** (§6.2).
3. **Pretraining corpus is not representative of the benchmark.** COVID-era crowdsourced phone recordings dominate pretraining; a third of the benchmark is stethoscope and specialist data.
4. **No deployment characterisation whatsoever.** No latency, no memory, no throughput, no energy. For a mobile-systems lab whose stated motivation is ubiquitous health sensing, and whose OPERA-CE exists purely for efficiency reasons, **this is a striking omission.** You cannot tell from the paper whether any of these models can run on a phone, let alone an earbud.
5. **Device and domain shift are not evaluated.** Every task is train-and-test within one dataset. **No cross-device or cross-dataset generalisation experiment appears in the main protocol**, despite generalisation being claimed in the abstract.
6. **The shortcut concern of §3.1 is never tested.** Nobody checks what nuisance variables the contrastive representation encodes.

### Positioning against related work

| | Pretraining domain | Objective | Open weights |
|---|---|---|---|
| VGGish | AudioSet (general) | Supervised | Yes |
| AudioMAE | AudioSet (general) | Masked reconstruction | Yes |
| CLAP | Audio–text pairs | Cross-modal contrastive | Yes |
| OpenSMILE | None (hand-crafted) | — | Yes |
| **OPERA** | **Respiratory** | **Contrastive + generative** | **Yes** |

**[analysis]** The genuinely novel axis is *domain*, not method — COLA and MAE are both off-the-shelf. That is a fair contribution for a Datasets & Benchmarks paper, which is what this is; it should not be read as a methods paper. Judged as a methods paper it would be thin. Judged as infrastructure, it's valuable.

**The conspicuous absent baseline is wav2vec 2.0 / HuBERT** — self-supervised models that operate on raw waveforms rather than spectrograms. Their inclusion would test whether the spectrogram front end is itself a limitation. **[analysis] Their absence is defensible on compute grounds but leaves the front-end question open.**

---

## 9. Open questions worth pursuing

Ordered by (value × feasibility) on a free Colab GPU.

**Q1 — What nuisance variables do the embeddings encode?**
Freeze OPERA. Train linear probes to predict **recording device**, **subject identity**, and **chest location** from the embeddings. §3.1 predicts these are highly decodable. Costs one feature-extraction pass and a few linear fits.
*Why it matters:* if device is linearly decodable at high accuracy, the representation carries a device fingerprint — which is precisely the obstacle to Auryx's "works on any earbud" claim. **This is the highest-value experiment available to you.**

**Q2 — What do these models cost to run?**
Latency (batch 1, CPU), peak memory, parameter count, INT8 and ONNX behaviour, across CT / CE / GT. Unpublished. Directly relevant to any wearable deployment. Pairs naturally with Q1.

**Q3 — Does the pretraining/benchmark ICBHI overlap contaminate T7?**
Read the code, compare identifiers. Near-zero cost, potentially significant finding either way.

**Q4 — How much of the gain survives a harder split?**
Re-evaluate a task under strict subject-wise splitting with proper resampling and honest confidence intervals.

**Q5 — Does a foundation model beat a small supervised CNN trained directly on the task?**
The paper compares OPERA against *other pretrained models*, never against a well-tuned task-specific baseline. This is the practitioner's actual question and it is unanswered.

**Q6 — Generative CNN.** §5 predicts an EfficientNet-MAE would be the best lung-function model. Expensive to pretrain; propose rather than run.

---

## 10. Interview-ready summary

If asked "what did you make of the OPERA paper?", something like:

> It's a Datasets & Benchmarks contribution rather than a methods one — the objectives are COLA and MAE off the shelf; the novelty is applying them to a curated 400-hour respiratory corpus and standardising 19 downstream tasks, which the field badly needed.
>
> The result I found most interesting isn't the headline. It's the double dissociation between objective and task type: contrastive pretraining wins classification, generative wins regression, and the MRRs almost invert between the two groups. That has a clean theoretical explanation — instance discrimination compresses away non-discriminative information, reconstruction has to preserve it — and it implies that for continuous vital-sign estimation the generative branch is the right starting point, even though OPERA-CT wins overall.
>
> The gap I kept coming back to is that a paper motivated by ubiquitous health sensing reports no deployment cost at all. OPERA-CE exists specifically because efficiency matters, and there's no latency or memory number anywhere. I also think the contrastive positive-pair construction — two crops from the same recording — puts the representation under strong pressure to encode recording identity, including device, and nobody tests that. Those two things seemed worth measuring, so I did.

Honest, specific, respectful of the work, and it ends on something you built rather than something you read.
