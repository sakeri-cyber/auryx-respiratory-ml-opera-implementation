# hEARt — The Beginner's Read

**Papers (one line of work, two publications):**

1. *hEARt: Motion-resilient Heart Rate Monitoring with In-ear Microphones*
   **Kayla-Jade Butkow**, Ting Dang, Andrea Ferlini, Dong Ma, **Cecilia Mascolo**
   IEEE PerCom 2023 · [IEEE 10099317](https://ieeexplore.ieee.org/abstract/document/10099317) · [free on arXiv](https://arxiv.org/abs/2108.09393)

2. *An evaluation of heart rate monitoring with in-ear microphones under motion*
   Pervasive and Mobile Computing, 2024 · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1574119224000397) · [ACM](https://dl.acm.org/doi/10.1016/j.pmcj.2024.101913)

> These are the same research programme — the journal paper is the extended evaluation of the conference system — so they're covered together rather than in separate documents.

> **⚠️ Version warning.** Different sources report different subject counts and error figures for hEARt, which suggests the arXiv preprint was revised between 2021 and the 2023 PerCom publication. The figures below come from the **current arXiv abstract**. Some secondary sources cite 15 subjects and MAE 1.88 / 6.83 / 13.19 BPM. **Read the PDF and confirm which numbers are current before quoting any of them to Butkow.** Quoting superseded numbers to their author is a bad way to open.

---

## 1. The problem

Every wearable that measures your heart rate — watch, ring, earbud — uses **PPG**: photoplethysmography. Shine a green LED into the skin, measure how much bounces back. Blood absorbs light, blood volume pulses with each beat, so the reflected light pulses too.

PPG is cheap, small and works well when you are sitting still.

**It falls apart when you move.** Motion changes the sensor's contact pressure against the skin, shifts its position, and lets ambient light leak in. The resulting artefacts land in the same frequency range as the heartbeat itself, which makes them very hard to filter out. It also performs worse on darker skin, because melanin absorbs green light — a well-documented equity problem in consumer wearables.

Motion is not an edge case. Exercise is exactly when people most want heart rate.

## 2. The idea: listen instead of looking

Your heart makes sound. The classic *lub-dub* is S1 and S2 — the heart valves snapping shut. A stethoscope on the chest hears this easily.

Could a microphone in your ear hear it?

Ordinarily, barely. But something useful happens when you **seal** the ear canal.

### The occlusion effect

Sound reaches your inner ear two ways: through the air, and through your skull as vibration (bone conduction). Normally, low-frequency bone-conducted sound escapes out of the open ear canal.

**Seal the canal and it can't escape.** It's trapped in a small closed cavity, and the sound pressure inside rises — by roughly 20 dB at low frequencies.

This is why your own voice sounds boomy and strange when you plug your ears, and why chewing sounds enormous with earplugs in. It's a nuisance in hearing-aid design.

**hEARt turns it into a sensor.** Heart sounds are almost entirely below 150 Hz — precisely where the occlusion effect gives the most amplification. Put a microphone inside a sealed earbud and the heartbeat becomes audible.

### Why audio might beat PPG under motion

The paper's bet is that **acoustic sensing and optical sensing fail differently**.

PPG's motion artefacts come from the sensor's mechanical relationship with the skin — pressure, position, light leakage. An in-ear microphone in a sealed canal isn't subject to those failure modes in the same way.

Audio has its own motion problem, of course — footsteps conduct through the skeleton and are extremely loud. But footstep noise is *impulsive and broadband*, quite unlike the rhythmic *lub-dub* of a heartbeat. Different structure means it may be separable, whereas PPG artefacts occupy the same band as the signal.

**That's the hypothesis. The paper tests it.**

---

## 3. What the system does

Two stages:

```
in-ear audio (noisy, motion-corrupted)
        │
        ▼
┌────────────────────┐
│ STAGE 1            │   deep learning
│ artefact mitigation│   denoise the audio
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ STAGE 2            │   signal processing
│ HR estimation      │   find beats, compute rate
└─────────┬──────────┘
          ▼
   heart rate (BPM)
```

**Stage 1 is a neural network** that learns to strip motion artefacts out of the in-ear audio. This is worth noticing: unlike RespEar, hEARt *does* use deep learning — but for **denoising**, not for predicting heart rate. The network's job is to clean the signal; a classical algorithm then reads the answer off it.

**Stage 2 is signal processing** — filter to the cardiac band, detect the beats, convert intervals to a rate.

This division is deliberate and it's the pattern worth remembering: **learn the part you can't model, compute the part you can.** Nobody has a clean analytical model of how running corrupts ear-canal audio, so learn it. Everybody knows how to turn beat timings into BPM, so just do that.

---

## 4. The data

**20 subjects**, four conditions:

| Condition | Why it's included |
|---|---|
| **Stationary** | Baseline — the easy case |
| **Walking** | Moderate, rhythmic motion |
| **Running** | Severe motion, loud footstep interference |
| **Speaking** | The awkward one — your own voice is enormously loud in an occluded ear canal |

Including **speaking** is a good sign of practical thinking. It's not motion at all; it's a completely different kind of interference that any real earbud product hits constantly, and most lab studies ignore it.

---

## 5. Results

| Condition | MAE (BPM) |
|---|---|
| Stationary | 3.02 ± 2.97 |
| Walking | 8.12 ± 6.74 |
| Running | 11.23 ± 9.20 |
| Speaking | 9.39 ± 6.97 |

**How to read these honestly:**

**Stationary at 3 BPM is decent but not clinical.** A chest-strap ECG is under 1 BPM. Consumer wearables at rest claim 2–3 BPM. So this is competitive at rest, not better.

**The degradation under motion is severe** — nearly 4× from stationary to running. The paper's headline claim is that this **still beats reported in-ear PPG performance**, which is the honest framing: not "we solved motion", but "we fail less badly than the incumbent".

**Speaking is nearly as damaging as running.** That should be intuitive once you think about the occlusion effect — the same mechanism that amplifies your heartbeat also amplifies your own voice, massively. A product that degrades whenever the user talks has a real problem.

**The variances are large.** ±9.20 on running means some subjects or windows are far worse than the mean. High variance in a physiological measurement usually indicates a subgroup for whom the method simply doesn't work — anatomy, seal quality, or motion style.

---

## 6. Why this matters for Auryx

**This is the founding technology.** Butkow's PhD was *In-ear Audio for Physiological Monitoring*. hEARt is the core of it, and Auryx is the commercialisation. If you understand this paper you understand what the company is.

**The motion gap is the company's central problem.** 3 BPM at rest is a product. 11 BPM while running is not. Everything between those numbers is the engineering roadmap — and closing it is presumably a large part of what the ML engineer they're hiring will work on.

**They already use deep learning where it earns its place.** Not for the end-to-end prediction, but for the denoising step where no analytical model exists. That's a mature engineering judgement and it tells you how they think.

**The hardest problems are the unglamorous ones:** speech interference, seal quality, per-subject variability. Not architecture choices.

**And note what's missing:** the paper reports accuracy but — unlike RespEar — no on-device latency, battery or memory figures in the abstract. Worth checking in the full text.

---

## 7. Read this before an interview

You will very likely be interviewed by Butkow. Read the actual PDF, not just this summary — particularly:

- the denoising architecture and how they built training data (you cannot record clean and noisy versions of the same heartbeat simultaneously, so how did they construct supervision? this is the interesting engineering question)
- the HR estimation algorithm after denoising
- the hardware setup
- the journal paper's extended evaluation, which is where the deeper analysis of motion lives

## 8. Check yourself

1. What is the occlusion effect, and why does it make in-ear heart sound detection possible?
2. Why might audio beat PPG under motion? What's the structural argument?
3. Why is deep learning used for denoising but not for the heart rate estimate itself?
4. Why is *speaking* such a hard condition — harder than walking?
5. What would you need to improve to turn 11 BPM running error into a shippable product?
