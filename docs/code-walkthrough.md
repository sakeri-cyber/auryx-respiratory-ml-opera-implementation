# Code Walkthrough

Every module, explained. Written so you can defend any line of it in an interview.

**How to use this:** read a section, then open the file and read the code with the
explanation beside it. Where a design decision had alternatives, the alternatives and
the reason for rejecting them are given — those are what you'll actually be asked about.

---

# Table of contents

1. [The shape of the whole thing](#1-the-shape-of-the-whole-thing)
2. [`data/audio.py` — waveform to spectrogram](#2-dataaudiopy)
3. [`data/icbhi.py` — the dataset index](#3-dataicbhipy)
4. [`models/opera_gt.py` — the encoder](#4-modelsopera_gtpy)
5. [`models/baseline_cnn.py` — the comparison](#5-modelsbaseline_cnnpy)
6. [`probe.py` — linear probing](#6-probepy)
7. [`profiling.py` — deployment cost](#7-profilingpy)
8. [`pipeline.py` — orchestration](#8-pipelinepy)
9. [`scripts/` — the entry points](#9-scripts)
10. [`tests/` — what each test protects](#10-tests)
11. [Questions you should be able to answer](#11-questions-you-should-be-able-to-answer)

---

# 1. The shape of the whole thing

```
920 .wav files
      │  data/icbhi.py      index + labels + metadata
      ▼
  [Recording] × 920
      │  data/audio.py      load → resample → clip → log-mel
      ▼
  (920, 64, 256) spectrograms          ← cached to artifacts/
      │  models/opera_gt.py  frozen encoder
      ▼
  (920, 384) embeddings                ← cached to artifacts/
      │  probe.py           logistic regression on frozen features
      ▼
  AUROC / balanced accuracy
```

Two design decisions govern everything:

**Cache at the two expensive boundaries.** Spectrograms and embeddings are written to
disk. Extraction takes ~46 s total; every experiment afterwards runs on cached matrices
in seconds. On a Colab free tier that disconnects, this is the difference between a
usable project and an unusable one.

**Minimal dependencies.** `torch`, `numpy`, `scipy`, `scikit-learn`. No `torchaudio`, no
`torchvision`, no `timm`. Partly so it runs on a bare runtime, but mainly because
depending on a library's audio defaults means you don't know what your preprocessing is
doing — and a silent preprocessing mismatch produces embeddings that look fine and mean
nothing.

---

# 2. `data/audio.py`

Turns a `.wav` file into the exact tensor the encoder expects.

## 2.1 `AudioConfig`

```python
sample_rate: int = 16_000
n_mels: int = 64
win_ms: float = 64.0
hop_ms: float = 32.0
n_frames: int = 256
```

The first four come straight from the paper: *"resampled to 16 kHz and merged into a mono
channel"*, *"64 Mel filter banks with a 64 ms Hann window that shifts every 32 ms"*.

`n_frames = 256` is **derived, not chosen**. Here's the reasoning, and it's worth being
able to reproduce it on a whiteboard:

- The checkpoint's `pos_embed` has shape `(1, 1025, 384)`.
- 1025 = 1024 patches + 1 CLS token.
- `patch_embed.proj` is `Conv2d(1, 384, kernel_size=4, stride=4)` → 4×4 patches.
- So `(n_mels / 4) × (n_frames / 4) = 1024`.
- With `n_mels = 64`: `16 × (n_frames / 4) = 1024` → `n_frames = 256`.

**Anything else fails loudly** with a positional-embedding mismatch — which is by design;
see §4.5.

```python
@property
def win_length(self) -> int:
    return int(round(self.sample_rate * self.win_ms / 1000.0))
```

64 ms × 16 kHz = **1024 samples**. Hop: 32 ms → **512 samples**, i.e. 50% overlap.

```python
@property
def n_fft(self) -> int:
    return 1 << (self.win_length - 1).bit_length()
```

Rounds the window length up to the next power of two. `(1024-1).bit_length()` is 10, so
`1 << 10` = 1024 — already a power of two, so no padding here. The property exists so
that changing `win_ms` doesn't silently give you a slow non-power-of-two FFT.

`clip_seconds` = 256 × 32 ms = **8.192 s**. Every recording contributes exactly one clip
of this length.

## 2.2 `load_wav` — three subtleties

```python
if data.dtype.kind == "i":
    data = data.astype(np.float32) / np.iinfo(data.dtype).max
```

**Normalise by the dtype's full scale, not by the observed maximum.** Dividing by
`data.max()` would gain up quiet recordings to full scale, destroying the relative level
information between recordings — and level differs systematically across ICBHI's four
devices. That would be quietly injecting a device signal into the input.

```python
if data.ndim > 1:
    data = data.mean(axis=1)
```

Mono downmix. ICBHI is mono, but some tools emit duplicated stereo.

```python
gcd = math.gcd(int(native_sr), int(target_sr))
data = resample_poly(data, target_sr // gcd, native_sr // gcd)
```

`resample_poly` is **polyphase** resampling — it applies an anti-aliasing filter and
resamples in one rational-ratio step. The alternative, `scipy.signal.resample`, is
FFT-based and assumes the signal is periodic, which produces edge artifacts on
non-periodic audio. Reducing by the GCD keeps the polyphase filter small: 44100→16000
becomes 160/441 rather than 16000/44100.

The function **returns the native sample rate** alongside the audio. That's not
incidental — in ICBHI, native rate is confounded with device (§3.4), and the whole
bandwidth-control experiment depends on knowing it.

## 2.3 The mel filterbank, built by hand

```python
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
```

The mel scale. Roughly linear below 1 kHz, logarithmic above — modelling the fact that
human pitch discrimination is finer at low frequencies. The constants 2595 and 700 are
the standard HTK formulation.

```python
mel_edges = np.linspace(_hz_to_mel(cfg.f_min), _hz_to_mel(cfg.f_max), cfg.n_mels + 2)
hz_edges = _mel_to_hz(mel_edges)
```

**Space the edges uniformly in mel, then convert back to Hz.** That's the entire idea:
uniform in mel = non-uniform in Hz = more filters where the ear (and lung sounds) have
more detail. `n_mels + 2` because each triangle needs a lower edge, a centre and an upper
edge, and adjacent triangles share them.

```python
for i in range(cfg.n_mels):
    lower, centre, upper = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
    rising  = (fft_freqs - lower)  / max(centre - lower, 1e-9)
    falling = (upper - fft_freqs)  / max(upper - centre, 1e-9)
    fb[i] = np.maximum(0.0, np.minimum(rising, falling))
```

Each filter is a triangle: rise linearly from `lower` to `centre`, fall from `centre` to
`upper`. `min(rising, falling)` gives the triangle; `max(0, ·)` clips everything outside
the support. The `1e-9` guards against a degenerate zero-width triangle if the mel
spacing collapses.

Writing this out rather than importing it is deliberate — this is the part of audio ML
people most often treat as a black box, and it's five lines.

## 2.4 `LogMelSpectrogram`

```python
self.register_buffer("fb", mel_filterbank(cfg), persistent=False)
self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)
```

**Buffers, not parameters.** They move with `.to(device)` but carry no gradient.
`persistent=False` keeps them out of `state_dict()` — they're deterministic functions of
the config, so saving them would be redundant and would make checkpoints
config-dependent.

```python
spec = torch.stft(..., center=True, pad_mode="reflect", return_complex=True)
power = spec.real.pow(2) + spec.imag.pow(2)
```

`center=True` pads by `n_fft // 2` so frame *k* is centred at sample *k × hop* — meaning
frame indices map cleanly onto time. Reflect padding rather than zeros, for the same
reason as everywhere else in this codebase: silence is not a sound that occurs in these
recordings, and networks learn to key off it.

Power spectrum (magnitude²), not magnitude. That's what the mel filterbank convention
expects and what the paper implies.

```python
log_mel = torch.log(mel + 1e-6)
```

The `1e-6` prevents `log(0) = -inf`. **Test `test_silence_produces_no_nans` exists
because of this line** — a fully silent input would otherwise poison the whole batch.

```python
mean = log_mel.mean(dim=(-2, -1), keepdim=True)
std = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
return (log_mel - mean) / std
```

**Per-example standardisation, and this is the most important line in the file.**

Log compression turns a multiplicative gain into an additive offset:
`log(a·x) = log(a) + log(x)`. So subtracting the per-example mean removes recording gain
entirely, and dividing by the std removes dynamic-range differences.

Why it matters here specifically: ICBHI's four devices have very different sensitivities.
Without this, the loudest device is trivially identifiable and the encoder learns
equipment rather than pathology. `test_gain_invariance` asserts that a 40 dB level change
produces identical output.

`clamp_min(1e-5)` handles the constant-input case where std is 0.

## 2.5 `_fit_width` and `smart` padding

```python
if t > target:
    start = (t - target) // 2
    return x[..., start : start + target]
```

Centre-crop. ICBHI recordings often begin with stethoscope placement noise, so taking
from the middle samples lung sound rather than a handling transient.

```python
reps = -(-target // max(t, 1))
tiled = torch.cat([x, x.flip(-1)] * reps, dim=-1)
```

`-(-a // b)` is integer ceiling division. Alternating forward and reversed copies gives
**reflection tiling** — continuous at every seam, unlike simple repetition which
introduces a discontinuity at each boundary that reads as a broadband click.

---

# 3. `data/icbhi.py`

## 3.1 `Recording`

```python
@dataclass(frozen=True, slots=True)
class Recording:
```

`frozen=True` makes it hashable and prevents accidental mutation mid-pipeline.
`slots=True` drops the per-instance `__dict__` — with 920 instances it's marginal, but
it also catches typo'd attribute assignments at runtime instead of silently creating them.

```python
@property
def is_copd(self) -> int:
    return int(self.diagnosis == COPD_LABEL)
```

T7 is **binary COPD vs everything else**, not multi-class diagnosis. Everything that
isn't literally `"COPD"` — Healthy, URTI, Asthma, Bronchiectasis, Pneumonia,
Bronchiolitis, LRTI — is the negative class.

## 3.2 `_parse_stem` — failing loudly

```python
parts = stem.split("_")
if len(parts) != 5:
    raise ValueError(f"Expected 5 fields in ICBHI stem, got {len(parts)}: {stem!r}")
```

A filename we can't parse means the dataset isn't what we think it is. Skipping it
silently would train on a subset without telling you. **Raise.**

## 3.3 `read_sample_rate` — a bug worth remembering

The original implementation used `scipy.io.wavfile.read(..., mmap=True)` to get the rate
cheaply. It crashed:

```
ValueError: mmap=True not compatible with 3-byte container size
```

ICBHI contains **24-bit** wav files, and scipy's memory-mapped reader can't handle 3-byte
samples. The fix uses stdlib `wave`, which reads only the header:

```python
with wave.open(str(wav_path), "rb") as fh:
    return fh.getframerate()
```

with a scipy fallback for anything `wave` rejects. Worth knowing about: real audio
datasets contain format variety that clean examples don't prepare you for.

## 3.4 `bandwidth_matched_subset` — the confound control

```python
modal_rate = max(set(rates), key=rates.count)
kept = [r for r in recordings if r.native_sample_rate == modal_rate]
```

Short function, important idea.

ICBHI's native sample rates are almost perfectly predictive of device:

| Device | Native rate |
|---|---|
| AKGC417L | 100% @ 44.1 kHz |
| LittC2SE | 100% @ 44.1 kHz |
| **Litt3200** | **100% @ 4 kHz** |
| Meditron | mixed |

Resample a 4 kHz file to 16 kHz and it has **zero energy above 2 kHz** — 30 of the 64 mel
bands (about half) are empty, since 2 kHz sits at 52% of the mel range from 50 Hz to 8 kHz. A classifier can identify Litt3200 from that alone, learning nothing
about audio content.

Keeping only 44.1 kHz-native files (824 of 920) removes the artifact. Litt3200 disappears
entirely, leaving three devices that share a bandwidth.

**Why this matters:** without it, the C1 result would have been uninterpretable. With it,
the device signal *survived* — which is what makes the finding real rather than an
artifact of the resampler.

## 3.5 `subject_wise_split`

```python
by_patient: dict[int, Recording] = {}
for rec in recordings:
    by_patient.setdefault(rec.patient_id, rec)
```

Splits at **patient** level, so no patient appears on both sides.

Why this matters more than usual here: **793 of 920 recordings are COPD, but only 64 of
126 patients.** COPD patients contribute ~12 recordings each. A random recording-wise
split therefore near-guarantees the same patient on both sides, and the model can
memorise a patient's recording conditions rather than learn pathology.

We measured the cost of getting this wrong: **+0.066 AUROC of pure illusion**.

```python
if stratify_by_label:
    groups = {0: [], 1: []}
    ...
    for label, pids in groups.items():
        n_test = max(1, round(len(pids) * test_fraction))
        test_patients.update(pids[:n_test])
```

Splits COPD and non-COPD patients **separately**, so both classes appear in both halves in
roughly the right proportion. With only 126 patients an unstratified draw can produce a
test split missing a class entirely, which makes AUROC undefined.
`test_stratification_preserves_class_balance` asserts this.

---

# 4. `models/opera_gt.py`

The core of the project — OPERA-GT reimplemented from the checkpoint's tensor shapes.

## 4.1 How the architecture was recovered

No architecture file was available, so it was read off the weights:

| Checkpoint tensor | Shape | What it implies |
|---|---|---|
| `patch_embed.proj.weight` | `(384, 1, 4, 4)` | Conv2d(1→384), 4×4 patches |
| `cls_token` | `(1, 1, 384)` | embed dim 384 |
| `pos_embed` | `(1, 1025, 384)` | 1024 patches + CLS |
| `blocks.0.attn.qkv.weight` | `(1152, 384)` | 1152 = 3 × 384, fused QKV |
| `blocks.0.mlp.fc1.weight` | `(1536, 384)` | MLP ratio 4 |
| `blocks.11.*` | — | depth 12 |
| `decoder_pred.weight` | `(16, 256)` | 16 = 4×4×1, one patch |

Depth 12 + dim 384 + MLP ratio 4 is **ViT-Small**, whose convention is 6 heads (384/6 =
64 per head). That's the one number inferred rather than read, and it's in `GTConfig` so
it can be changed.

**The verification that this is right:** loading with `strict=True` succeeded with zero
missing parameters, and the total came to **21,694,848** — the paper states 21M for the
GT encoder. If the architecture were wrong, either the load would fail or the count
wouldn't match.

## 4.2 `PatchEmbed`

```python
self.proj = nn.Conv2d(cfg.in_chans, cfg.embed_dim,
                      kernel_size=cfg.patch_size, stride=cfg.patch_size)
```

**Kernel size == stride** means non-overlapping windows. This is exactly equivalent to
slicing the image into 4×4 patches, flattening each to a 16-vector, and applying a shared
`Linear(16, 384)` — but as a strided convolution it's one fused, fast op, and it's how the
weights are stored.

```python
x = self.proj(x)                      # (B, 384, 16, 64)
return x.flatten(2).transpose(1, 2)   # (B, 1024, 384)
```

`flatten(2)` collapses the two spatial dims into one sequence dim; `transpose` moves it
to position 1 to give the `(batch, sequence, features)` layout transformers expect.

Note **row-major flattening**: the sequence runs along frequency-within-time. The
positional embedding is learned, so the model figures out the geometry — but it means
"adjacent tokens" doesn't uniformly mean "adjacent in time".

## 4.3 `Attention`

```python
qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
q, k, v = qkv.unbind(0)
```

One `Linear(384, 1152)` produces Q, K and V together — one matmul instead of three, and
it matches the checkpoint layout. The reshape splits that 1152 into `(3, heads,
head_dim)`, and the permute reorders to `(3, batch, heads, seq, head_dim)` so `unbind(0)`
gives three tensors shaped for attention.

```python
out = F.scaled_dot_product_attention(q, k, v)
```

PyTorch's fused SDPA rather than a hand-written
`softmax(QKᵀ/√d)V`. It's numerically more stable (online softmax, no materialised
`(seq, seq)` logits at full precision) and materially faster. With 1025 tokens the
attention matrix would be 1025×1025 per head per layer — worth not materialising.

```python
out = out.transpose(1, 2).reshape(b, n, d)
```

Concatenates the heads back together.

## 4.4 `Block` — pre-norm

```python
x = x + self.attn(self.norm1(x))
return x + self.mlp(self.norm2(x))
```

**Pre-norm**: LayerNorm *inside* the residual branch, so the residual path from input to
output is unnormalised and gradients flow through it untouched. Post-norm (the original
2017 transformer) puts LayerNorm after the addition and needs learning-rate warmup to
train deep stacks. MAE and every modern ViT use pre-norm — and critically, the
checkpoint's tensor ordering (`norm1` before `attn`) confirms this is what OPERA used.
Getting it backwards would load without error and produce garbage.

`test_is_residual` verifies the structure: zero the output projections and the block must
become the identity.

## 4.5 `OperaGTEncoder.forward`

```python
cls = self.cls_token.expand(x.shape[0], -1, -1)
x = torch.cat([cls, x], dim=1)
```

`expand` rather than `repeat` — it creates a broadcast view with no copy.

```python
if x.shape[1] != self.pos_embed.shape[1]:
    raise ValueError(f"Token count {x.shape[1]} != positional embedding ...")
```

**A deliberate guard.** Without it, a wrong-sized input would hit a broadcasting error
somewhere deeper with an unhelpful message. This says exactly what's wrong and what the
input should be. `test_wrong_input_size_raises_informative_error` covers it.

```python
if pooling == "mean":
    return x[:, 1:].mean(dim=1)
```

**Mean of patch tokens, excluding CLS — and the exclusion matters.** In contrastive
models (CLIP, and OPERA-CT) the CLS token is trained directly by the objective and is the
natural representation. In an MAE there is **no objective that trains CLS** — the loss is
reconstruction of masked patches. So the CLS token in a generative model carries much
less, and mean-pooled patch tokens are the stronger choice. `"cls"` and `"both"` are
available so this is testable rather than assumed.

## 4.6 `load_opera_gt`

```python
encoder_state = {k: v for k, v in state.items()
                 if not k.startswith(("decoder_", "mask_token"))}
```

The checkpoint holds encoder *and* decoder (34.8M total). The decoder exists only to
provide a reconstruction target during pretraining and is useless for representation
extraction. Dropping it leaves 21.7M.

```python
missing, unexpected = model.load_state_dict(encoder_state, strict=False)
if strict and missing:
    raise RuntimeError(f"{len(missing)} encoder parameters were not found ...")
```

Loads with `strict=False` (so leftover decoder keys don't error) but then **checks
`missing` manually and raises**. This is the important bit: a silent partial load would
leave layers randomly initialised, producing embeddings that look plausible and mean
nothing. That is the single most expensive failure mode available here, and this converts
it from a week of confusion into an immediate exception.

## 4.7 `extract_features`

```python
@torch.inference_mode()
```

Stronger than `no_grad()` — also disables autograd version tracking, so tensors can't
later be used in a graph. Slightly faster and appropriate for pure inference.

Batches, moves each batch to the device, and brings results back to CPU immediately so
GPU memory doesn't accumulate across 920 recordings.

---

# 5. `models/baseline_cnn.py`

The comparison OPERA's paper never makes: **does 404 hours of pretraining beat a small
model trained directly on the task?**

## 5.1 Deliberately tiny

```python
channels: tuple[int, ...] = (16, 32, 64, 64)
```

**60,530 parameters** against the encoder's 21.7M. Sized down on purpose: ICBHI T7 has
126 patients. Anything larger memorises them.

## 5.2 `ConvBlock`

```python
self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
self.bn = nn.BatchNorm2d(out_ch)
```

`bias=False` because BatchNorm immediately subtracts the mean, making the conv bias
mathematically redundant — it would be trained and then cancelled.

BatchNorm matters more than usual here for the same reason as §2.4: four devices with
different gains, and normalising per batch keeps early activations comparable.

## 5.3 Global average pooling

```python
self.pool = nn.AdaptiveAvgPool2d(1)
self.classifier = nn.Linear(in_ch, n_classes)
```

Collapses `(B, 64, H, W)` to `(B, 64)`. The alternative — flatten then a big dense layer —
would be `64 × 4 × 16 = 4096` inputs, adding ~8k parameters that would dominate the model
and overfit instantly. GAP also makes the model input-length invariant.

## 5.4 Class weighting rather than resampling

```python
counts = torch.bincount(y_train, minlength=2).float()
weights = (counts.sum() / (2 * counts.clamp_min(1))).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

Training recordings are ~86% COPD. Inverse-frequency weighting in the loss corrects this.

**Why weighting rather than oversampling:** oversampling would duplicate recordings, and
since COPD patients contribute many recordings each, duplication would compound the
memorisation risk the small architecture is already guarding against.

`clamp_min(1)` prevents division by zero if a class is absent.

---

# 6. `probe.py`

## 6.1 `linear_probe`

```python
scaler = StandardScaler().fit(features_train)
x_tr = scaler.transform(features_train)
x_te = scaler.transform(features_test)
```

**Fit on train only, transform both.** Fitting on the concatenation would leak test
statistics into training. On 920 samples that's a measurable inflation, and it's one of
the most common silent errors in small-data ML.

`test_scaler_is_fit_on_train_only` catches it by shifting the test features and asserting
the score changes — if the scaler saw the test data, the shift would be absorbed.

```python
clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced", ...)
```

A single linear layer, which is OPERA's protocol. Deliberately weak: if it performs well,
the credit belongs to the representation. `class_weight="balanced"` because every label we
probe is imbalanced.

```python
majority = 1.0 / n_classes
```

**Chance level under *balanced* accuracy is 1/n_classes regardless of skew.** This is the
line that keeps the device probe honest — the raw device distribution is 70% AKGC417L, so
plain accuracy would score a constant predictor at 0.70 and look impressive. Balanced
accuracy against 0.25 is the fair comparison.

## 6.2 `shuffled_control` — the smoke alarm

```python
rng = np.random.default_rng(seed)
return linear_probe(features_train, rng.permutation(labels_train), ...)
```

Probe against permuted labels. **Must land at chance.** If it doesn't, something leaks —
the scaler, the split, duplicated rows — and every other number in the run is worthless.

It did land at chance: 0.492 AUROC for COPD, 0.264 balanced accuracy for device (chance
0.250). Cheap insurance, and it has caught real bugs in published work.

## 6.3 `multi_split_probe` — and why it exists

This function was added **because the first implementation was wrong**, and it's the most
interesting thing in the file.

The initial run reported `std = 0.000` on every probe across five seeds. That looked like
remarkable stability. It isn't — logistic regression fitted on fixed data is
deterministic, so varying the classifier's `random_state` varies **nothing**. The error
bars were describing solver noise, which is zero.

This is *exactly* the criticism made of OPERA's own protocol in
`02-OPERA-deep-dive.md` §6.1 — that their 5 runs vary the probe seed, not the split — and
this implementation reproduced it faithfully.

```python
for seed in split_seeds:
    train_idx, test_idx = split_fn(seed)
    single = linear_probe(features[train_idx], labels[train_idx], ...)
```

Resamples the **split**, which is where the uncertainty actually lives: with ~30 test
patients, *which* patients land in the test half dominates everything else.

The fix changed the leakage estimate from **+0.004 to +0.066 AUROC — a 16× difference.**
The single split had landed somewhere the effect was invisible.

---

# 7. `profiling.py`

## 7.1 Measurement discipline

```python
for _ in range(n_warmup):
    model(example_input)
```

Discarded warm-up. The first passes pay for lazy kernel selection and allocator growth.
Including them inflates the median by an amount that varies by machine — which is how
latency claims become irreproducible.

```python
timings.sort()
quartiles = statistics.quantiles(timings, n=4)
return LatencyResult(median_ms=statistics.median(timings), ...)
```

**Median and IQR, not mean and standard deviation.** Latency distributions are
right-skewed — bounded below by the actual compute, unbounded above by scheduling noise.
A mean is dragged upward by a few outliers and a standard deviation implies a symmetry
that isn't there.

```python
if device.startswith("cuda"):
    torch.cuda.synchronize()
```

CUDA kernels are asynchronous; without synchronising you'd time the *launch*, not the
execution. Needed both after warm-up and inside the loop.

## 7.2 Honest reporting

```python
report: dict = {"host": {"platform": platform.platform(), ...}}
```

Records the machine. A latency number without the hardware it was measured on isn't a
result.

```python
except Exception as exc:
    logger.warning("Quantisation failed: %s", exc)
    report["quantisation_error"] = str(exc)
```

Quantisation **failed on this machine** (`NoQEngine` — no quantised backend on this
Apple Silicon build). That's recorded in the output rather than swallowed. The docstring
already notes that dynamic quantisation only covers `Linear` layers, so on a ViT the
expected gain was modest anyway and static quantisation or QAT would be the real path.

---

# 8. `pipeline.py`

## 8.1 Caching

```python
if cache_path and cache_path.exists():
    logger.info("Loading cached features from %s", cache_path)
    return np.load(cache_path)
```

Both expensive stages check for a cache first. This is what makes the project survivable
on a free Colab tier: extraction happens once, everything after runs on cached matrices.

## 8.2 Centre-clipping

```python
offset = max(0, (len(waveform) - need) // 2)
```

One clip per recording, from the centre — ICBHI recordings frequently open with
stethoscope placement noise.

**A real limitation:** recordings run 7.9 s to 86.2 s, so for long ones we use 8.192 s out
of 86 and discard the rest. Multiple clips per recording with aggregation would use the
data better. Listed in `findings.md` §7 as future work.

## 8.3 `sanity_check_embeddings`

```python
norm = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)
```

Are two recordings from the same patient more similar than two from different patients?
If not, preprocessing is wrong and everything downstream is noise.

Result: **0.9854 vs 0.9706, separation +0.0148.** Passes — but note how tightly clustered
everything is (cosine > 0.97 across the board). MAE representations have no objective
that spreads the space, so the whole embedding cloud sits in a narrow cone. Worth
remembering when interpreting probe results.

## 8.4 `random_split_indices`

Exists **only** to quantify what subject-wise splitting buys you. The gap between it and
the correct split *is* the leakage measurement. Keeping the wrong protocol around
deliberately, and reporting both, is how you turn a methodological point into a number.

---

# 9. `scripts/`

**`run_experiments.py`** — stages 0–2 then experiments A, C1, C1b, C2, C3, writing
`artifacts/results.json`. Every result is captured to JSON so the write-up quotes data
rather than remembered numbers.

**`robust_eval.py`** — re-runs the probes with resampled splits (§6.3). Runs on cached
features in seconds.

```python
assert len(recordings) == len(feats), "cached features do not match the recording index"
```

Guards against a stale cache after the dataset or ordering changes — a subtle way to
produce completely wrong results while everything appears to work.

---

# 10. `tests/`

76 tests. The ones that carry real weight:

| Test | What it protects |
|---|---|
| `test_patch_grid_matches_opera_gt_positional_embedding` | Input shape derivation. Wrong → encoder errors |
| `test_gain_invariance` | 40 dB level change must not alter output. Wrong → model learns device gain |
| `test_silence_produces_no_nans` | `log(0)` and `std=0` paths |
| `test_no_patient_appears_in_both_splits` | The +0.066 AUROC illusion |
| `test_stratification_preserves_class_balance` | Undefined AUROC on small splits |
| `test_scaler_is_fit_on_train_only` | Test-statistics leakage |
| `test_shuffled_control_lands_at_chance` | Validity of every reported number |
| `test_parameter_count_matches_published_figure` | Architecture correctness vs the paper's 21M |
| `test_loads_strictly_with_no_missing_parameters` | Silent partial load |
| `test_permutation_equivariance` | Attention isn't leaking position |
| `test_is_residual` | Pre-norm block structure |
| `test_removes_the_device_that_is_uniquely_low_rate` | The bandwidth confound control |

Two tests exist because of bugs found during the build: `read_sample_rate` (24-bit wav
files) and `test_single_class_training_split_raises_clearly` (unstratified splits dropping
a class).

---

# 11. Questions you should be able to answer

Work through these before any interview. If one is shaky, reread that section.

**On preprocessing**
1. Why is `n_frames` 256 and not something else? *(Derive it from `pos_embed`.)*
2. Why per-example standardisation, and what would break without it?
3. Why reflect-pad rather than zero-pad?
4. Why `resample_poly` rather than `scipy.signal.resample`?

**On the model**
5. How did you recover the architecture without an architecture file?
6. How do you know your reimplementation is correct?
7. Why mean-pool patch tokens instead of using CLS? Would your answer change for OPERA-CT?
8. Why pre-norm rather than post-norm, and how would you tell from the checkpoint?
9. Why `strict=True` on the load — what's the failure mode you're preventing?

**On the experiments**
10. Why AUROC and balanced accuracy rather than accuracy?
11. Why is the chance level for the device probe 0.25 and not 0.70?
12. What does the bandwidth-matched control remove, and what did it show?
13. Does device decodability prove shortcut learning? What experiment would?
14. Why did `std = 0.000` in the first run, and why does that matter?

**On the results**
15. A 60k CNN beat a 21.7M foundation model. How much do you believe it, and why?
16. What would you do next, and why that order?

---

## The three things to lead with

If you only remember three things from this codebase:

1. **The device finding.** A respiratory foundation model pretrained on 404 hours encodes
   recording device about twice as strongly as it encodes the disease — and it survives a
   bandwidth-matched control. For a company shipping on heterogeneous consumer earbuds,
   that's a live risk.

2. **The variance bug.** We criticised OPERA's protocol for reporting variance from probe
   seeds rather than splits, then reproduced the same flaw, caught it because the std was
   exactly zero, and fixed it — which changed a headline number 16×.

3. **The baseline.** A 60,530-parameter CNN beat the 21.7M-parameter model's linear probe.
   Stated with its caveats, that's a more useful contribution than another benchmark row.
