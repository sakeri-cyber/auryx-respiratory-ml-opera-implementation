# hEARt — Deep Technical Analysis

*Companion to `06-hEARt-beginner.md`.*

> **⚠️ Scope limitation, stated plainly.** The hEARt PDF exceeded what I could retrieve, so this document is built from the **verified abstract** plus standard theory, not from the full method section. Sections marked **[GAP]** are things you must fill in from the PDF yourself. I have not invented architecture details to fill them — an honest hole is more useful than a plausible fabrication you might repeat to the author.
>
> Tags: **[paper]** verified from the abstract · **[background]** standard theory · **[analysis]** my reasoning · **[GAP]** unverified, read the PDF.

---

## 1. Problem formalisation

Observed in-ear audio under motion:

$$y(t) = h(t) * s_{\text{cardiac}}(t) + n_{\text{motion}}(t) + n_{\text{speech}}(t) + n_{\text{ambient}}(t)$$

where $h(t)$ is the ear-canal transfer function (dominated by the occlusion effect) and the noise terms are, crucially, **non-stationary and structured** — not white.

Goal: estimate instantaneous heart rate $\text{HR}(t)$ from $y(t)$.

**[analysis] Why this is not a standard denoising problem.** The interference is not additive white noise; it is *another physiological signal with its own temporal structure*. Footstep impacts are quasi-periodic at 1.5–3 Hz. Heartbeats are quasi-periodic at 1–3 Hz. **These overlap.** Classical spectral subtraction fails because you cannot separate two quasi-periodic sources by frequency alone when their rates coincide — which they do, precisely during running, when stride rate and heart rate can both sit near 3 Hz.

This is the structural reason a learned denoiser is justified here. It's also the strongest argument for the paper's two-stage design.

---

## 2. The acoustic front end

### 2.1 Occlusion effect, quantitatively **[background]**

Sealing the ear canal converts it from an open tube (quarter-wave resonator, radiating impedance at the entrance) into a closed cavity. Bone-conducted vibration of the canal walls compresses a fixed volume of air rather than displacing air out of the canal.

For a closed cavity of volume $V$ with wall displacement $\Delta V$, pressure rises as

$$\Delta p = -\gamma P_0 \frac{\Delta V}{V}$$

**[background]** so the smaller the trapped volume, the larger the pressure gain. Gain is strongly frequency-dependent — typically 10–25 dB below ~1 kHz, falling off above — because at higher frequencies the canal walls' compliance and the eardrum's impedance dominate.

**[analysis] Three consequences that matter commercially:**

1. **The gain is where the signal is.** Cardiac acoustic energy is concentrated 20–150 Hz. Maximum occlusion gain is below ~500 Hz. The mechanism is well matched to the target by luck of anatomy.
2. **Gain depends on trapped volume**, which depends on insertion depth and ear anatomy. Two users with identical hearts produce different signal levels. **This is the per-subject variance source**, and it is anatomical rather than algorithmic — it cannot be engineered away, only calibrated around.
3. **Seal quality is binary-ish.** A leak converts the closed cavity back toward an open tube and the low-frequency gain collapses. This does not degrade gracefully. **[analysis] For Auryx's "works on earbuds you already own" claim, seal variation across commodity earbuds is, in my view, the single largest productisation risk — larger than the algorithms.**

### 2.2 Cardiac acoustics **[background]**

| Component | Frequency | Origin |
|---|---|---|
| **S1** ("lub") | 20–150 Hz, peak ~30–50 Hz | Mitral + tricuspid valve closure, ventricular systole onset |
| **S2** ("dub") | 20–200 Hz, peak ~50–70 Hz | Aortic + pulmonic valve closure, diastole onset |
| S3 / S4 | 20–70 Hz | Usually pathological in adults; low amplitude |
| Murmurs | up to 600 Hz | Turbulent flow |

**[analysis]** For heart *rate*, you need only S1 detection — S1-to-S1 intervals give the beat period. S2 is a confound: if your detector fires on both, you double the apparent rate. The S1–S2 interval (systole, ~300 ms) is shorter and less variable than S2–S1 (diastole), so a well-designed detector uses that asymmetry to disambiguate. **[GAP] Whether hEARt does this is in the PDF.**

### 2.3 The interference, characterised **[analysis]**

| Source | Character | Band | Why it's hard |
|---|---|---|---|
| Footsteps | Impulsive, quasi-periodic 1.5–3 Hz | Broadband, low-freq dominant | **Rate overlaps HR during running** |
| Speech (own voice) | Sustained, harmonic | 85–255 Hz F0 + harmonics | Occlusion effect amplifies it *more* than the heart; F0 overlaps S1/S2 |
| Jaw / swallowing | Impulsive | Low | Also seal-disturbing |
| Ambient | Varied | Broad | Attenuated by the seal — the one case where occlusion helps twice |

**[analysis] Speech deserves emphasis.** Own-voice is amplified by the same occlusion mechanism, and male F0 (85–180 Hz) sits directly on top of S2 and the upper S1 band. It is not merely loud — it is *spectrally coincident*. That explains why speaking (9.39 BPM) is nearly as damaging as running (11.23), which is otherwise surprising given speaking involves no motion at all.

---

## 3. Stage 1 — learned artefact mitigation

**[paper]** "A novel deep learning framework to denoise the in-ear audio signals."

**[GAP] Not verified: architecture, layer counts, loss function, input representation (waveform vs spectrogram), whether it is masking-based or mapping-based, parameter count, inference cost.**

### 3.1 The training-data problem — the interesting question

**[analysis]** Supervised denoising needs paired $(y_{\text{noisy}}, y_{\text{clean}})$. But you **cannot simultaneously record a clean and a corrupted version of the same heartbeat** — the subject is either running or not.

Standard resolutions, any of which hEARt might use:

**(a) Synthetic mixing.** Record clean cardiac audio at rest; separately record motion noise; add them at controlled SNR. Gives exact ground truth. *Weakness:* additive mixing assumes the corruption is additive and independent, but motion also changes the **seal** and hence the transfer function $h(t)$ — a multiplicative, not additive, effect. Models trained on synthetic mixtures often disappoint on real motion for exactly this reason.

**(b) Reference-signal supervision.** Use a simultaneously recorded ECG or chest-strap signal to derive beat timings, and train the network to produce a signal whose beats match. Supervises the *task* rather than the waveform.

**(c) Self-supervised / unpaired.** Cycle-consistency or noise2noise variants. Avoids the pairing problem entirely at the cost of weaker supervision.

**[analysis] This is the question I would most want answered from the PDF, and the best technical question you could ask Butkow.** "How did you construct supervision for the denoiser, given you can't record clean and noisy simultaneously?" is a question only someone who has thought about the problem asks.

### 3.2 Why denoise rather than predict end-to-end

**[analysis]** The obvious alternative is a single network mapping raw audio → BPM. hEARt deliberately doesn't. Reasons this is the better call:

1. **Data efficiency.** 20 subjects is far too few to learn an end-to-end regression that generalises. Denoising is a lower-level, more transferable task with far more effective training signal per sample (every timestep supervises, not one number per window).
2. **Interpretability and debuggability.** You can *listen* to the denoiser's output. You can plot it. When it fails you can see why. An end-to-end regressor gives you a wrong number and no recourse.
3. **Regulatory tractability.** For eventual CE/FDA work — which Auryx's job posting explicitly mentions — a pipeline whose final step is deterministic peak detection is far easier to characterise and validate than a black box.
4. **Failure mode.** A denoiser that fails produces visibly bad audio, and downstream confidence measures can catch it. An end-to-end regressor fails *confidently*.

**[analysis] This division — learn the unmodellable, compute the modellable — is the single most transferable idea in the paper, and the thing I'd lead with if asked what you took from it.**

---

## 4. Stage 2 — HR estimation

**[paper]** "An HR estimation algorithm" applied post-denoising.

**[GAP] Not verified: filter bands, detection method, window length, whether time- or frequency-domain, S1/S2 disambiguation.**

**[background] The standard approaches, so you know the design space:**

**Time domain.** Bandpass ~20–150 Hz → Hilbert or Shannon-energy envelope → adaptive-threshold peak picking → S1-to-S1 intervals → HR. *Advantage:* gives beat-to-beat intervals, hence HRV, hence (via RSA) respiration — this is the route RespEar depends on. *Disadvantage:* individual missed or spurious beats corrupt the interval series badly.

**Frequency domain.** Envelope → FFT or autocorrelation over a window → dominant periodicity → HR. *Advantage:* far more robust to individual missed beats. *Disadvantage:* **no beat-to-beat intervals**, so no HRV, so no RSA-based respiration. Also poor temporal resolution.

**[analysis] This choice has product-level consequences.** Auryx claims HR **and HRV** — and RespEar's whole sedentary pipeline needs accurate IBIs. So the estimator must be time-domain, or at minimum produce beat timings. That is the harder, less robust option, and it constrains the design. Verify which hEARt uses — it tells you a lot about how the pieces fit together.

**Autocorrelation, why it's natural here [background]:** for a quasi-periodic envelope $e(t)$, $R_{ee}(\tau) = \mathbb{E}[e(t)e(t+\tau)]$ peaks at the beat period. Footstep interference at a *different* rate produces a peak at a different lag, so the two are separable in the autocorrelation domain **provided the rates differ** — which fails, again, precisely when stride and heart rate converge during running.

---

## 5. Results, examined

**[paper]**

| Condition | MAE ± SD (BPM) |
|---|---|
| Stationary | 3.02 ± 2.97 |
| Walking | 8.12 ± 6.74 |
| Running | 11.23 ± 9.20 |
| Speaking | 9.39 ± 6.97 |

### 5.1 Contextualising

**[background]** Reference points: chest-strap ECG < 1 BPM. Consumer wrist PPG at rest ~2–3 BPM, degrading to 10–20+ BPM during vigorous exercise. ANSI/AAMI EC13 for cardiac monitors requires ±10% or ±5 BPM.

**[analysis]**
- **Stationary (3.02)** is competitive with consumer PPG at rest, not better. It clears the AAMI bar at typical resting rates.
- **Running (11.23)** does not clear it. But the honest comparison is against in-ear PPG under the same motion, and the paper claims — plausibly — that it wins there.
- **The claim being made is relative, not absolute.** "Better than in-ear PPG under motion", not "solved". Reading it as the latter would be a mistake.

### 5.2 The standard deviations

**[analysis]** SD ≈ MAE across every condition (e.g. 11.23 ± 9.20). For an error distribution bounded below by zero, that ratio indicates a **heavy right tail** — a subpopulation of subjects or windows where the method fails badly, dragging the mean.

Given §2.1, the obvious candidate explanation is **anatomical**: subjects whose ear geometry gives a poor seal, hence weak occlusion gain, hence low cardiac SNR before any algorithm runs. If true, per-subject performance would be bimodal rather than continuous.

**[analysis] The evaluation I'd want:** per-subject MAE, ranked. If a handful of subjects dominate the error, that is a *hardware/fit* finding, not an algorithm finding, and it changes what you'd work on. **[GAP] The journal paper (2024), being an extended evaluation "under motion", is the most likely place this analysis lives — check there first.**

### 5.3 The speaking result

**[analysis]** Speaking (9.39) approaching running (11.23) is the most informative number in the table. It confirms the interference is **spectral, not mechanical** — no motion is involved. It follows that whatever the denoiser learned about motion artefacts transfers imperfectly to voice, which are structurally different (harmonic and sustained vs impulsive and broadband).

For a consumer product this may matter more than running. People talk far more than they run.

---

## 6. Critical assessment

### Strengths **[analysis]**

1. **A genuinely novel sensing modality**, with a sound physical basis rather than a hopeful one.
2. **Failure-mode complementarity with PPG** is a real argument, not marketing — the two degrade for unrelated reasons, which also makes them good fusion candidates.
3. **Speaking is included.** Most lab studies quietly omit the conditions that break them.
4. **The two-stage architecture is the right engineering call** (§3.2).
5. **Honest relative framing.** They don't claim to have solved motion.

### Limitations **[analysis]**

1. **N = 20**, laboratory conditions. Standard for the venue; insufficient to characterise the anatomical variance the method is exposed to.
2. **Motion error remains large.** 11 BPM is not a product; it's a research result pointing at the remaining work.
3. **Custom hardware.** Results are obtained with a purpose-built sealed earbud and a dedicated microphone. **Commodity earbuds differ in microphone type, position, seal, and — critically — apply proprietary DSP (ANC, beamforming, noise suppression) that may destroy exactly the low-frequency content the method depends on.** The transfer from lab hardware to consumer hardware is unproven and is the largest gap between this paper and Auryx's product claim.
4. **No cross-device evaluation** — same concern as OPERA, arriving from the hardware side.
5. **Deployment cost not reported in the abstract.** RespEar reports latency, battery and memory; the abstract for hEARt doesn't. **[GAP]** Check the full text.
6. **[GAP] Denoiser generalisation is uncharacterised** — at least from what I can verify. If trained on synthetic mixtures, real-motion performance is the open question.

### Relation to the wider literature **[analysis]**

The in-ear sensing field (Ferlini, Ma, Mascolo and others) has explored in-ear PPG, IMU-based sensing, and audio. hEARt's contribution is establishing **audio as viable under motion**, which was not previously demonstrated. The natural next step — and a plausible research direction for Auryx — is **multimodal fusion**: in-ear audio + PPG + IMU, where the IMU provides a motion reference that lets you cancel the artefact you can independently observe. That is a well-trodden pattern in wearable sensing and I'd expect it on their roadmap.

---

## 7. Questions worth asking Butkow

1. **"How did you construct supervision for the denoiser, given you can't record clean and corrupted audio simultaneously?"** — the best question in this document. It is *the* hard problem in the paper and only someone who thought about the method would ask.
2. **"Is the error distribution across subjects bimodal? Does it track seal quality or ear geometry?"** — probes whether the residual problem is algorithmic or anatomical.
3. **"Speaking is nearly as damaging as running despite no motion — does that suggest the denoiser needs a separate treatment for harmonic interference?"** — shows you read the results table, not just the headline.
4. **"Does the HR estimator produce beat-to-beat intervals? RespEar's RSA pipeline needs them, but interval-based estimation is much less robust under motion."** — demonstrates you connected the two papers. **This is the one I'd actually lead with**, because it shows systems thinking across their whole stack rather than close reading of one paper.
5. **"How much of the low-frequency content survives commodity earbud DSP and ANC?"** — the productisation question. Ask carefully; it may be commercially sensitive.

---

## 8. What to fill in from the PDF

Before any interview, resolve these:

- [ ] Which version's numbers are current (§ version warning)
- [ ] Denoiser architecture and input representation
- [ ] Denoiser loss function
- [ ] **How training pairs were constructed**
- [ ] HR estimator: time or frequency domain? Does it yield IBIs?
- [ ] S1/S2 disambiguation strategy
- [ ] Window length and update rate
- [ ] Hardware specification
- [ ] Ground-truth device
- [ ] On-device cost, if reported
- [ ] What the 2024 journal paper adds beyond the conference version

That checklist is roughly two hours of reading and it is the highest-value preparation available to you for this specific interview.
