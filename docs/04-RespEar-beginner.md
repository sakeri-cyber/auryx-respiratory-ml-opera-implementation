# RespEar — The Beginner's Read

**Paper:** *RespEar: Earable-Based Robust Respiratory Rate Monitoring*
Yang Liu, **Kayla-Jade Butkow**, Jake Stuchbury-Wass, Adam Pullin, Dong Ma, **Cecilia Mascolo**
[arXiv:2407.06901](https://arxiv.org/abs/2407.06901) · [free PDF](https://mobile-systems.cl.cam.ac.uk/papers/respear.pdf)

> Butkow is Auryx's CTO. Mascolo is a co-founder. This is the closest public description of what Auryx actually does.

---

## 1. The problem

Respiratory rate is one of the best early-warning vital signs in medicine — it deteriorates before heart rate or blood pressure in sepsis, respiratory failure and cardiac arrest. It is also the one nobody measures continuously, because measuring it usually means strapping something to your chest.

So: can an earbud do it? Continuously, while you sit, work, walk and run?

### Why this is genuinely hard

**Breathing is quiet.** Your heartbeat produces a distinct low-frequency thump in the ear canal. Breathing produces almost nothing — a faint turbulent hiss, easily below the noise floor.

**Motion destroys it.** Walking and running generate footstep impacts conducted through your skeleton straight into the ear canal, orders of magnitude louder than breath. The signal you want is buried under the signal you don't.

**Existing approaches fail in exactly this regime.** Chest straps work but nobody wears them. IMU-based methods (detecting chest wall motion) collapse when the whole body moves. Audio methods that listen directly for breath sounds work in a silent room and nowhere else.

---

## 2. The central idea

Here is the move that makes the paper work, and it's genuinely clever:

> **Don't listen for breathing. Listen for breathing's fingerprint on signals that are loud.**

Breathing is quiet, but breathing *modulates* things that are loud. Your heartbeat is loud. Your footsteps are loud. And both are coupled to your breathing by physiology. So measure the loud thing precisely, and recover breathing from how it wobbles.

This is an **indirect measurement** strategy, and it is a much better fit for a noisy wearable than trying to amplify a whisper.

### Coupling 1 — RSA: breathing modulates your heart rate

**Respiratory Sinus Arrhythmia** is a real, well-characterised physiological phenomenon: your heart speeds up when you inhale and slows down when you exhale. Everyone does this; it's strongest in young, healthy, relaxed people.

So if you can measure the time between consecutive heartbeats accurately enough, that series of intervals will oscillate — and **the frequency of that oscillation is your breathing rate**.

You never hear the breath at all. You hear the heart, time it precisely, and read the breathing off the rhythm.

This is what RespEar uses when you're **sedentary**.

### Coupling 2 — LRC: breathing locks to your stride

**Locomotor Respiratory Coupling** is the tendency, when running or walking, to synchronise breathing with footfalls — one breath every 2, 3 or 4 steps. Ratios like 4:3, 3:2 and 2:1 are common. It's partly mechanical (the impact of landing helps expel air) and partly neural.

So during **activity**, RespEar counts footsteps — which are deafeningly loud in the ear canal, an advantage rather than a problem — and uses the coupling to constrain the breathing rate.

The thing that ruins the sedentary method is the thing that enables the active method. That inversion is the paper's nicest idea.

---

## 3. System shape

Because the two couplings apply in different regimes, RespEar needs three parts:

```
in-ear audio (60s window, 30s overlap)
        │
        ▼
   ┌─────────────┐
   │  SELECTOR   │  SVM on MFCC features
   │             │  sedentary? walking? running?
   └──────┬──────┘
          │
     ┌────┴─────┐
     ▼          ▼
 ┌────────┐  ┌────────┐
 │  RSA   │  │  LRC   │
 │pipeline│  │pipeline│
 └────┬───┘  └───┬────┘
      └─────┬────┘
            ▼
    respiratory rate (BPM)
```

The **selector** is a two-stage SVM: first sedentary vs. active, then (if active) walking vs. running. It votes across twelve 5-second segments per window and needs 75% agreement, otherwise it declines to answer. Reported accuracy is 100% sedentary-vs-active and 99–100% walking-vs-running.

**Note the design instinct there** — the system is allowed to say "I don't know". For a medical-adjacent device that's not a limitation, it's a feature.

---

## 4. The sedentary pipeline, step by step

1. **Low-pass filter at 30 Hz.** Heart sounds are low-frequency; almost everything else isn't. This single step removes most of the world.
2. **Find the heartbeats.** Compute the smoothed Hilbert envelope (the outline of the signal's energy), take a moving average as an adaptive threshold, and pick the peak within each region where the envelope exceeds the threshold. Adaptive, not fixed, so it survives amplitude drift.
3. **Compute inter-beat intervals (IBIs)** — the time gaps between consecutive beats. This series is your HRV signal.
4. **Pick the better ear.** Left and right earbuds both record; choose whichever gives lower IBI standard deviation. Fusing them cuts error from ~1.8 to 1.42 BPM — a 20% improvement essentially for free.
5. **Extract the breathing oscillation** from the IBI series. This is the paper's main technical contribution and §5 covers why it's harder than it sounds.
6. **Reject interference.** Split the window into 3-second chunks, flag any whose standard deviation is anomalous (head movement, speech, a cough), and repair them with an adaptive RLS filter using a clean neighbouring segment as reference.

---

## 5. The clever bit: adaptive band selection

Textbook HRV analysis says respiratory information lives in the "high frequency" band, conventionally **0.15–0.4 Hz** (9–24 breaths/min). So the obvious approach is: bandpass the IBI series to that fixed band, take an FFT, and the dominant peak is your breathing rate.

**This works badly.** The paper reports that a fixed [0.15, 0.35] Hz band gives **2.4× higher error** than a band adapted to the actual rate.

Why? Because the band is wide and the signal is weak. A fixed band admits a lot of non-respiratory HRV variation along with the breathing, and the FFT peak can easily land on the wrong thing. But you can't narrow the band without knowing the answer first — and the answer is what you're trying to find.

**RespEar's solution is a search.** Propose every candidate respiratory rate from 7.5 to 42.5 BPM in 0.5 BPM steps. For each candidate, build a *narrow* band centred on it (0.65× to 1.35× the candidate), filter with it, FFT the result, and see what rate comes out. If a candidate is correct, filtering around it should return that same value — **self-consistency**. If it's wrong, the filter returns something else.

Then pick the candidate with the smallest disagreement, cross-checked against a time-domain estimate (counting zero crossings) for robustness.

This is a **fixed-point search**: the right answer is the one that reproduces itself. Elegant, and it needs no training data at all.

---

## 6. The active pipeline

1. **Low-pass at 50 Hz** and detect footsteps with the same peak-finding machinery. Gets stride frequency to within 3%.
2. **Bandpass for breath sounds** — 300–1800 Hz for walking, but **2000–9000 Hz for running**. Higher, because at running intensity the breath harmonics up there actually exceed the footstep energy. A nice empirical detail.
3. **Template matching.** Record one person breathing loudly in a quiet room, split into 40 ms frames, compute a 15-bin spectral energy vector per frame, average them into a "this is what breathing looks like" template. Then score every frame of live audio by cosine similarity to the template, giving a **breathing probability curve** over time.
4. **Decompose with SSA.** Singular Spectrum Analysis splits that curve into components at different timescales — one will track the step rhythm, another the breathing.
5. **Aggregate valid components.** Rather than assuming a fixed step:breath ratio, keep every component whose peak count falls inside the physiologically plausible LRC range (1.9–4.9 walking, 1.8–5.6 running) and sum them. This handles the ratio *changing* mid-window, which real runners do.

---

## 7. Results

| Condition | MAE | MAPE |
|---|---|---|
| **Overall** | **1.71 BPM** | 9.68% |
| Sedentary | 1.48 BPM | 9.12% |
| Active | 2.28 BPM | 11.04% |

Per activity: sitting/standing ~1.3–1.4, working in-the-wild 1.56, listening to music 1.98, walking 1.75, **running 3.12** (the hardest case, as expected).

Ground truth is a Zephyr BioHarness chest strap at 25 Hz. Bland–Altman shows mean bias −0.02 BPM with limits of agreement −4.8 to +4.76.

Beats prior IMU-based, audio-envelope and LRC-based methods across the board.

**On-device cost** — and note that unlike OPERA, they actually report this:

| | Sedentary (RSA) | Active (LRC) |
|---|---|---|
| Latency per 60s window | 3.11 s | 12.27 s |
| Battery per hour | 4% | 14% |
| Memory | ~49 MB | ~49 MB |

Real-time capable, since a new estimate is only needed every 30 seconds.

They also test robustness properly: music playing, outdoor construction noise, an hour of ordinary office activity, simulated active noise cancellation, varying speeds, and single-earbud operation.

---

## 8. What this tells you about Auryx

**The product is indirect measurement.** They aren't building better microphones to hear quieter sounds. They're exploiting physiological couplings to recover what can't be heard directly. That's a science-led strategy, and it's why the founders are PhDs.

**HRV is infrastructure, not a feature.** The entire sedentary pipeline rests on beat-to-beat interval accuracy — reported at 3% MAPE. If you can time heartbeats precisely you get HR, HRV *and* respiration. That is why Auryx's marketing lists all three: they fall out of one measurement.

**Robustness is the actual product.** Look at how much of the paper is spent on motion, noise, music, ANC and in-the-wild testing. Getting 1.5 BPM in a quiet lab is a student project. Holding 2–3 BPM while someone runs outdoors is a company.

**They care about deployment cost** and measure it. Latency, battery, memory, all reported. Compare with the OPERA paper, which reports none of it. That contrast is worth remembering.

**This is mostly classical signal processing.** Filters, Hilbert envelopes, FFTs, SSA, RLS, one small SVM. Very little machine learning, no deep learning at all. Hold that thought — it matters for how you position yourself, and it's discussed in the DSP-vs-neural-networks note.

---

## 9. Check yourself

1. Why measure breathing *indirectly* rather than just amplifying breath sounds?
2. What is RSA, and why does it let you find breathing rate in a heartbeat signal?
3. Why does the fixed [0.15, 0.35] Hz band fail, and how does the candidate search fix it?
4. Why does running use a 2–9 kHz band while walking uses 300–1800 Hz?
5. Why is the *selector* necessary at all — why not run both pipelines always?
