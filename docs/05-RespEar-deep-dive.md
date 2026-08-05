# RespEar — Deep Technical Analysis

*Companion to `04-RespEar-beginner.md`.*

**Provenance tags:** **[paper]** stated in the source · **[background]** standard theory the paper assumes · **[analysis]** my reasoning beyond the paper. Verify anything load-bearing before repeating it to an author.

---

## 1. Problem formalisation

Given in-ear audio $s(t)$ sampled at $f_s$, estimate respiratory rate $\text{RR}(t)$ in breaths per minute, continuously, across sedentary and ambulatory conditions.

The direct approach — detect breath sounds, count them — fails because the breath-to-noise ratio in the ear canal is very poor and collapses entirely under motion.

**The indirect formulation.** Suppose there exists an observable signal $u(t)$ that is (a) high-SNR in the ear canal and (b) modulated by respiration through a known physiological coupling $\mathcal{C}$:

$$u(t) = \mathcal{C}\big(\text{RR}(t)\big) + \eta(t)$$

Then estimating RR becomes an inverse problem in $\mathcal{C}$ rather than a detection problem in a weak signal. RespEar instantiates this twice:

| Regime | Observable $u(t)$ | Coupling $\mathcal{C}$ |
|---|---|---|
| Sedentary | IBI series (HRV) | Respiratory Sinus Arrhythmia |
| Active | Footstep rhythm | Locomotor Respiratory Coupling |

**[analysis] The design insight worth naming:** the SNR of $u$ and the SNR of direct breath detection are *anti-correlated with activity*. Motion destroys breath sounds but makes footsteps trivially detectable. So the two regimes are complementary by construction, and the regime switch is not a hack — it's forced by the physics.

---

## 2. The acoustic front end

### 2.1 Occlusion effect **[background]**

Sealing the ear canal creates a closed cavity. Bone-conducted vibration that would normally radiate out is trapped, raising low-frequency SPL inside the canal by roughly 20 dB below ~1 kHz. This is why an occluded in-ear microphone can hear the heart at all — cardiac sound is essentially all below 150 Hz, exactly where the occlusion gain is largest.

**[analysis]** This makes seal quality a first-order variable. A poor seal doesn't degrade the signal gracefully; it removes the amplification mechanism the whole system depends on. For Auryx — whose pitch is that it works on earbuds people *already own*, with wildly varying seal quality — this is arguably the central productisation risk, and neither this paper nor hEARt characterises seal-quality sensitivity.

### 2.2 Band allocation **[paper]**

| Target | Band | Rationale |
|---|---|---|
| Heart sounds | LPF 30 Hz | S1/S2 energy concentrated 20–150 Hz; 30 Hz keeps the fundamental, discards the rest of the world |
| Footsteps | LPF 50 Hz | Impact transients, bone-conducted |
| Breath (walking) | BPF 300–1800 Hz | Turbulent airway noise |
| Breath (running) | BPF 2000–9000 Hz | Harmonics where breath exceeds footstep energy |

**[analysis] The 2–9 kHz choice for running is the interesting one.** The naive move is to keep the same breath band and fight harder to suppress footsteps. Instead they move *up* to a band where the interferer is intrinsically weak. Footstep energy is impulsive and low-frequency; breath at running intensity is loud and broadband. Rather than improving the filter, they changed the question. That's a good instinct and it generalises.

---

## 3. Heartbeat detection and the IBI series

### 3.1 Envelope-based peak detection **[paper]**

1. LPF at 30 Hz.
2. Hilbert envelope: for analytic signal $s_a(t) = s(t) + j\,\mathcal{H}\{s\}(t)$, the envelope is $e(t) = |s_a(t)|$. **[background]** This gives instantaneous amplitude independent of the carrier's phase — appropriate because we want *when* a beat occurred, not its waveform detail.
3. Smooth $e(t)$.
4. Adaptive threshold $\theta(t) = \text{MA}(e, W)$ — a moving average, so the threshold tracks amplitude drift from posture, seal shift and gain changes.
5. Regions of interest = intervals where $e(t) > \theta(t)$; peak = argmax within each ROI.

**[analysis]** A moving-average threshold is a high-pass operation on the decision statistic. It confers robustness to slow gain drift — significant in-ear, where jaw movement and seal shift cause exactly that — but it is vulnerable to sudden broadband transients, which is precisely why stage 3 (§5) exists.

### 3.2 The IBI series and its sampling problem

$$\text{IBI}_k = t_{k} - t_{k-1}$$

giving samples $\{(t_k, \text{IBI}_k)\}$.

**[background] This is the subtlety most people miss.** The IBI series is **non-uniformly sampled** — one sample per heartbeat, at instants determined by the heart itself. You cannot FFT it directly, because the FFT assumes uniform sampling. Standard practice is to resample onto a uniform grid (cubic spline at 4 Hz is conventional in the HRV literature) before spectral analysis.

**[analysis]** The paper's description doesn't foreground this step, but it must be happening. It matters because resampling interacts with the very modulation you're measuring: the sampling rate *is* the heart rate, which RSA modulates. At HR 60 you get 1 sample/s; Nyquist is 0.5 Hz, and you are trying to resolve a 0.15–0.4 Hz oscillation. **The margin is thin.** At low heart rates with fast breathing (say HR 50, RR 30 = 0.5 Hz), RSA-based estimation is approaching or violating Nyquist. This is a genuine, structural limitation of every RSA method and a good question to raise.

Reported IBI accuracy: **3% MAPE at beat-to-beat level** **[paper]**.

### 3.3 Channel fusion **[paper]**

Select the channel with lower IBI standard deviation. Left alone 1.8 BPM, right alone 1.79, fused **1.42** — a 20% improvement.

**[analysis]** The selection statistic is a proxy: lower IBI variance is treated as evidence of cleaner detection. It's not a pure quality metric, because genuine HRV *is* variance — the very thing being measured. A subject with strong RSA has high IBI standard deviation for physiological rather than noise reasons, so this criterion may systematically prefer the channel with *weaker* RSA. **A cleaner statistic would be beat-detection confidence** (envelope prominence, template correlation) rather than IBI dispersion. Worth raising; it's a real design critique that respects the result.

---

## 4. Adaptive HF-band localisation — the core contribution

### 4.1 Why the fixed band fails

Conventional HRV analysis **[background]** partitions the IBI power spectrum:

- VLF: < 0.04 Hz
- LF: 0.04–0.15 Hz (baroreflex, sympathetic + parasympathetic)
- **HF: 0.15–0.40 Hz (respiratory, vagally mediated)**

Naive method: bandpass IBI to HF, FFT, take $\arg\max$.

Two failure modes **[analysis]**:

1. **Range violation.** HF spans 9–24 BPM. Post-exercise or during stress, RR exceeds 24 and the true peak lies *outside* the band entirely.
2. **Poor selectivity.** A 0.25 Hz-wide band around a weak sinusoid admits substantial LF spillover and broadband HRV noise. The FFT peak becomes unreliable.

**[paper]** Fixed [0.15, 0.35] Hz yields **2.4× higher error** than an adapted band.

The bind: narrowing requires knowing the centre; the centre is the unknown.

### 4.2 The fixed-point search **[paper]**

Define candidates $r_i \in \{7.5, 8.0, \dots, 42.5\}$ BPM, restricted to $[r_c - w/2,\; r_c + w/2]$ where $r_c$ comes from an initial fixed-band FFT.

For each $r_i$, construct a **proportional** band:

$$l_i = 0.65 \cdot \frac{r_i}{60}, \qquad h_i = 1.35 \cdot \frac{r_i}{60} \quad \text{[Hz]}$$

Filter the HRV signal with $[l_i, h_i]$, FFT, take the dominant component $\hat{r}_i$, and form

$$\Delta^F_i = \big|\hat{r}_i - r_i\big|$$

min-max normalised across $i$.

**[analysis] The principle: the correct rate is a fixed point of the filter-and-estimate operator.**

$$\hat{r} = \arg\min_{r} \big| \Phi(r) - r \big|, \qquad \Phi(r) = \text{FFT-peak}\big(\text{BPF}_{[0.65r,\,1.35r]}(\text{HRV})\big)$$

If $r_i$ is right, a narrow band centred on it passes the true oscillation and $\Phi(r_i) \approx r_i$. If $r_i$ is wrong, the band either excludes the true peak (returning a noise artefact) or is off-centre (returning something displaced). Self-consistency identifies the truth.

**Note the band is proportional, not absolute** — $[0.65r, 1.35r]$ means constant *relative* bandwidth, i.e. constant Q. **[analysis]** That's the right choice: it gives uniform selectivity in log-frequency, matching how the estimator's difficulty scales.

### 4.3 Time-domain cross-check **[paper]**

Frequency-domain self-consistency alone can lock onto a harmonic or a persistent noise line. So:

1. Count zero crossings of each filtered breathing signal → time-domain rate $\Delta^T_i$.
2. Take the three candidates with smallest $\Delta^F$.
3. Smooth the T-difference list.
4. Choose $\arg\min_i (\Delta^F_i + \Delta^T_i)$ over those three.

**[analysis]** Zero-crossing counting and FFT peak-picking fail in *different* ways — the former is sensitive to noise-induced spurious crossings, the latter to harmonic confusion. Requiring agreement is a cheap, effective ensemble. Restricting to the top-3 keeps the noisier time-domain statistic as a *tiebreaker* rather than a primary, which is the right weighting.

**[analysis] Cost:** roughly 70 candidates × (filter + FFT) per window. Trivially parallel, embarrassingly cheap, and it needs zero training data — which is exactly why it works on 18 subjects.

---

## 5. Interference rejection **[paper]**

- Split the 60 s window into 3 s segments.
- Compute per-segment standard deviation.
- Flag segments exceeding an empirical threshold as interfered (head motion, speech, cough).
- Repair with an **adaptive RLS filter** using the nearest non-interfered segment as reference.

**[background] RLS** minimises exponentially-weighted least squares:

$$\mathbf{w}_n = \arg\min_{\mathbf{w}} \sum_{i=1}^{n} \lambda^{n-i} \big| d(i) - \mathbf{w}^\top \mathbf{x}(i) \big|^2$$

with forgetting factor $\lambda \in (0,1]$. Converges faster than LMS at higher computational cost — appropriate here, since interference events are short and an adaptive filter that converges slowly is useless.

**[analysis]** Using a neighbouring clean segment as reference assumes local stationarity of the cardiac signal — reasonable over a few seconds. It's fragile if interference is *sustained* (a long conversation), because then there is no clean neighbour. The paper's in-the-wild office test (1.56 BPM) suggests it holds up in practice, but sustained-speech performance isn't isolated. Worth asking about.

---

## 6. The LRC pipeline

### 6.1 Template matching **[paper]**

Template: one user, loud breathing, quiet room. 40 ms frames, 20 ms overlap. Periodogram per frame, frequency range divided into 15 bins, energy summed per bin → $\mathbf{v}_f \in \mathbb{R}^{15}$. Average across frames → template $\mathbf{T}$.

Per live frame: cosine similarity $S_f = \dfrac{\mathbf{v}_f^\top \mathbf{T}}{\|\mathbf{v}_f\|\|\mathbf{T}\|}$, then

$$P(f) = \begin{cases} \dfrac{S_f - T_{\text{thr}}}{1 - T_{\text{thr}}} & S_f > T_{\text{thr}} \\[2mm] 0 & \text{otherwise} \end{cases}$$

**[analysis] A single-subject template is the weakest link in this pipeline.** Breath spectra vary with airway anatomy, sex, fitness and effort. 15 bins is a coarse enough representation that it may capture only "broadband turbulent noise", which is generic — that coarseness is probably load-bearing rather than a limitation. But the paper doesn't ablate template choice, and "whose breathing did you record?" is a fair question. A per-user calibration breath, or a template averaged over subjects, would be the obvious robustness fix.

Note $P(f)$ is a **rectified, rescaled similarity**, not a probability. Cosine similarity is not calibrated, and the linear rescaling above threshold doesn't make it so. Harmless — only its periodicity is used — but the naming is loose.

### 6.2 SSA decomposition **[background]**

Singular Spectrum Analysis: embed the series into a trajectory (Hankel) matrix of window length $L$, take the SVD, group eigentriples, diagonally average to reconstruct additive components. It's a data-adaptive decomposition — no basis assumed, unlike Fourier — which suits a quasi-periodic signal whose rate drifts.

**[analysis]** SSA is the right tool here precisely because step and breath rhythms are non-stationary within a 60 s window. A Fourier basis would smear a drifting rhythm across bins; SSA tracks it.

### 6.3 Ratio-agnostic aggregation **[paper]**

Valid RR bounds from stride frequency:

$$\text{RR}_{\min} = \frac{\text{SF}_{\text{est}} \cdot (N/f_s)}{\text{LRC}_{\max}}, \qquad \text{RR}_{\max} = \frac{\text{SF}_{\text{est}} \cdot (N/f_s)}{\text{LRC}_{\min}}$$

with LRC ∈ [1.9, 4.9] walking, [1.8, 5.6] running. Components whose peak count falls outside are discarded; the rest are summed; final peak detection gives RR.

**[analysis] This is the pipeline's best idea.** Prior LRC work assumes a *fixed* ratio, which real runners violate within a single window. By keeping every physiologically admissible component and summing, they avoid committing to a ratio at all. Stride frequency is used as a *constraint* rather than a *predictor* — a much weaker and therefore much safer assumption.

---

## 7. The regime selector **[paper]**

Two-stage SVM on MFCC features, 5 s segments, majority vote over 12 segments per 60 s window, ≥75% agreement required or the window is discarded.

Accuracy: 100% sedentary/active; 99% per-segment and 100% post-voting for walking/running.

**[analysis]** 100% should prompt scepticism, not admiration — it usually signals a task that is trivially separable rather than a strong classifier. Distinguishing "sitting" from "running on a treadmill" from in-ear audio *is* trivially separable: footstep transients are enormous. The real question is performance on **boundary states** — slow walking, fidgeting, standing up, climbing stairs slowly — which is where the regime switch actually matters and where a misclassification routes the signal to the wrong pipeline. The reported activities are all unambiguous. **This is the evaluation gap I'd probe hardest.**

The abstain mechanism (<75% agreement → discard) is good engineering, but the paper doesn't report the **abstention rate**. A system with excellent accuracy on the windows it answers and a high discard rate is a different product from one that answers everything. Coverage should be reported alongside accuracy.

---

## 8. Results and evaluation quality

**[paper]** Overall 1.71 BPM / 9.68% MAPE. Sedentary 1.48. Active 2.28. Running 3.12 (14.01% MAPE). Bland–Altman bias −0.02, LoA [−4.8, +4.76].

Ground truth: Zephyr BioHarness 3.0 @ 25 Hz.

**On-device:** 3.11 s (RSA) / 12.27 s (LRC) per window; 4% / 14% battery per hour; ~49 MB memory.

### Strengths **[analysis]**

1. **Bland–Altman rather than correlation alone.** Correct choice for method-comparison studies; correlation can be high with severe bias, and clinicians read LoA.
2. **Deployment cost measured and reported.** Latency, battery, memory. Compare with OPERA, which reports none.
3. **Genuinely broad robustness testing** — music, outdoor construction noise, ANC simulation, speed variation, single-channel, an hour in the wild. This is unusually thorough for the venue.
4. **The abstain mechanism exists at all.**

### Weaknesses **[analysis]**

1. **N = 18, ages 22–51, treadmill-based.** RSA amplitude declines markedly with age and is blunted in cardiac disease, diabetes and autonomic neuropathy — i.e. **exactly the populations who most need respiratory monitoring**. A method that depends on RSA may degrade precisely where it would be most valuable. This is the most important limitation and it is inherent to the approach, not fixable by more engineering.
2. **Ground truth is a chest strap**, itself an indirect measure with its own error. Reported MAE is against BioHarness, not against truth. Absolute accuracy is bounded by the reference.
3. **Nyquist margin** (§3.2) is not discussed and constrains the high-RR regime.
4. **No abstention rate reported** (§7).
5. **Empirical thresholds throughout** — interference detection, $T_{\text{thr}}$, LRC ranges. Tuned on the same 18 subjects that supply the results, with no held-out subject protocol described for threshold selection. Some optimism should be assumed.
6. **Custom hardware** — 3D-printed earbud, Knowles SPU1410LR5H-QB, dedicated amplifier, Raspberry Pi. **Results do not transfer to commodity earbuds**, whose microphones sit in different positions with different responses and aggressive proprietary DSP. The ANC simulation is a nod at this, not an answer.

---

## 9. Questions worth asking Butkow

Ordered by how much they'd reveal that you'd thought about it.

1. **"How does RSA-based estimation hold up in older or autonomically impaired subjects, where RSA amplitude is reduced?"** — the deepest question about the method's ceiling.
2. **"What's the abstention rate of the selector, and does it rise on boundary activities?"** — shows you read the evaluation design, not just the results.
3. **"How sensitive is it to ear-canal seal quality?"** — the occlusion effect is the whole front end, and seal varies enormously across commodity earbuds.
4. **"At low HR and high RR you're close to Nyquist on the IBI series — where does that bound the operating range?"** — a genuine signal-processing observation.
5. **"Channel selection uses IBI standard deviation, but genuine RSA increases that. Does it bias toward the weaker-RSA channel?"** — specific, technical, and constructive.

**[analysis]** Any one of these signals that you engaged with the method rather than the abstract. Ask one or two, not five.

---

## 10. What generalises to your own work

1. **Indirect measurement beats amplification when SNR is hopeless.** Find a coupled high-SNR observable.
2. **Fixed-point / self-consistency search** is a powerful trick when a parameter is needed to estimate itself. No training data required.
3. **Constrain, don't predict.** Stride frequency bounds the admissible RR range rather than predicting RR. Weaker assumptions fail more gracefully.
4. **Change the band, not the filter.** When an interferer dominates, look for a region where it's intrinsically weak.
5. **Build in abstention.** Systems allowed to say "I don't know" are far more deployable — and report coverage alongside accuracy.
