# Narration Script

**Runtime: ~17.5 minutes.** Written to be spoken, not read — short sentences, no
subordinate clauses you'll trip over. Timings are approximate; they assume an unhurried
pace with real pauses.

---

## Before you record

**The one thing that matters most: tone.** You are a candidate who spent a week on their
lab's work and is offering it for correction. You are not a reviewer. Every observation
should sound like *"here's something I noticed, and I'd like to know if I've got it
wrong"* — never *"here's a gap in your paper."*

Three habits that keep it there:

- Say **"I think"**, **"it appears"**, **"as far as I can tell"** on every claim.
- When you state a limitation, state it *before* anyone could raise it.
- Credit them generously and mean it. They released weights and code; that's why this was possible.

**Practical notes**

- Open `deck.html` in a browser, press **F** for fullscreen, arrow keys to advance.
- Record in one take if you can. Small stumbles are fine and read as human; over-editing sounds rehearsed.
- **Slow down on slides 18, 22 and 25.** They carry the most weight.
- Don't read this script verbatim — internalise each beat and speak it. Bracketed notes are direction, not words.

---

## Slide 1 · Title — 35s

> Hello. My name is Rohan Sakeri, and this is a short walkthrough of a small study I did
> over about a week.
>
> I rebuilt the encoder from OPERA — the open respiratory acoustic foundation model from
> the Cambridge Mobile Systems Lab — from scratch, and then ran three small experiments
> that came out of trying to understand it.
>
> I want to say at the very start that this is a study and not a result. I'm one week into
> a field that some of you have spent a decade in. I've tried to be careful about what I
> can and can't claim, and I'll be flagging the limitations as I go rather than saving
> them for the end.

*[Warm, unhurried. Don't rush the disclaimer — it sets the frame for everything after.]*

---

## Slide 2 · Why I did this — 50s

> The honest reason I did this is that I wanted to understand your work properly before
> asking to be part of it.
>
> It's easy to read a paper and say you found it interesting. I wasn't sure whether I'd
> actually understood OPERA, and the only test I could think of that would give me a real
> answer was to rebuild the encoder from the released weights and see whether it worked.
>
> Everything here was done on a laptop, in about a week, using the public checkpoints and
> the public ICBHI dataset. I've almost certainly made mistakes somewhere in it. If any
> of you spot one, I'd genuinely rather know.

---

## Slide 3 · The paper, and what I did — 1m 15s

*[This slide is the frame for everything after it. Give it the time — a viewer who only
watches ninety seconds should still leave knowing what the paper is and what you did.]*

> Before anything else, let me put the whole thing on one slide.
>
> The paper first, in three sentences.
>
> Respiratory audio has almost no labelled data, because getting a diagnosis attached to a
> recording means recruitment, ethics approval and a clinician. So every group ends up
> training its own model on its own tiny dataset, and nothing transfers between them.
>
> OPERA's answer is to pretrain encoders on four hundred hours of *unlabelled* coughs,
> breaths and lung sounds — no diagnoses needed — and then evaluate those encoders by
> freezing them completely and training a single linear layer on nineteen downstream
> health tasks.
>
> It releases three pretrained encoders, and shows that this domain-specific pretraining
> beats general-purpose audio models on sixteen of those nineteen tasks.
>
> Now what I did, also in three sentences.
>
> I rebuilt one of those three encoders — OPERA-GT — from scratch in PyTorch, recovering
> its architecture from the tensor shapes in the released checkpoint.
>
> I evaluated it on one of the nineteen tasks: COPD detection from ICBHI lung sounds, with
> subject-wise splits and controls.
>
> And then I asked three things the paper doesn't ask. What else, besides disease, do the
> embeddings encode? What does the model cost to run? And does a small supervised CNN
> trained directly on the task do better?
>
> I want the scope on the record from the very start. One encoder of three. One task of
> nineteen. No pretraining — I used their released weights. And not benchmarked against
> their published numbers, which I'll come back to properly later on.

---

## Slide 4 · RespEar and hEARt — 1m 10s

> Before OPERA, I want to explain what I *didn't* do, because I think the reasoning
> matters.
>
> I started with RespEar and hEARt. Those are the papers I most wanted to work on, and
> they're the ones on your website. They're also the ones I couldn't responsibly attempt
> in a week.
>
> I read them properly regardless, and I'm glad I did. RespEar's central idea — that
> rather than trying to hear breathing, which is quiet, you recover it from its
> fingerprint on things that are loud, through respiratory sinus arrhythmia when someone's
> sitting still and through stride coupling when they're moving — that's the most elegant
> thing I read all month. And hEARt's structure, where a learned model does the denoising
> and a classical algorithm does the estimation, is a design pattern I've genuinely taken
> away from this.
>
> But the datasets are in-lab and not public — eighteen and twenty subjects, custom
> hardware. I can't reproduce what I can't download. They're also principally
> signal-processing work, so implementing them would have taught me DSP but written no
> PyTorch. And they need a sealed in-ear microphone I don't have.
>
> So: read carefully, not implemented.

*[This slide does a lot of work. It proves you engaged with their actual research rather
than just the one paper you could run. Don't rush it.]*

---

## Slide 5 · Why OPERA — 55s

> Which brought me to OPERA, as the one piece of the lab's output I could actually run.
>
> Professor Mascolo is senior author, so it's genuinely the lab's work rather than
> something adjacent. The checkpoints are released, so no pretraining is needed — four
> hundred hours was never within reach for me. Linear probing is the paper's own protocol,
> and a frozen encoder costs almost nothing to evaluate. And ICBHI is public and small.
>
> There was also a simpler reason. Your job posting describes the goal as building the
> world's best foundation model for turning sound into health insights. OPERA seemed like
> the closest public thing to that direction, so it felt like the most useful place to
> spend the week.

---

## Slide 6 · The problem OPERA solves — 1m

> Now let me go back over that first half properly, because the experiments later only
> make sense against the theory.
>
> The problem OPERA addresses is that labels are the bottleneck in this field. Training a
> COPD detector traditionally needs recordings that a clinician has already diagnosed —
> which means recruitment, ethics approval, spirometry. It's slow and expensive, so
> respiratory datasets stay small. ICBHI, which is the standard benchmark, has nine
> hundred and twenty recordings. You can't train a modern network on that.
>
> And because every group trains its own bespoke model on its own small dataset, nothing
> transfers.
>
> The move the paper makes is to separate two things: learning what breathing *sounds
> like*, from learning what *disease* sounds like. The first needs no labels at all, and
> there are hundreds of thousands of unlabelled coughs and breaths in the world. You do
> that once, expensively. Everyone else reuses it. That reusable encoder is the foundation
> model.

---

## Slide 7 · Two learning strategies — 1m 15s

> There are two ways to learn without labels, and OPERA uses both — which is why there are
> three models.
>
> The contrastive approach takes two crops from the same spectrogram and calls them a
> positive pair; crops from different recordings are negatives. The model has to pick out
> the partner from a lineup. Nobody labels anything — the answer key comes free from how
> you cut the data. That's OPERA-CT and CE.
>
> The generative approach chops the spectrogram into patches, hides seventy percent of
> them, and asks the model to reconstruct what was behind the mask. That's OPERA-GT.
>
> These learn different things. Contrastive learning asks what makes recordings different
> from each other, so it keeps discriminative features. Masked reconstruction asks what
> respiratory sound looks like in general, so it keeps descriptive detail — you can't
> redraw something you only vaguely understand.
>
> And the paper's results show almost exactly that. Contrastive wins classification,
> generative wins regression, and the two nearly invert between task groups. I found that
> the most satisfying thing in the paper. It has a clean theoretical explanation and it
> generalises well beyond audio.

*[Genuine enthusiasm here is good. This is you showing you read for understanding, not
for talking points.]*

---

## Slide 8 · Linear probing — 55s

> The evaluation protocol is linear probing. Freeze the encoder completely, push your
> labelled data through it, and train exactly one fully-connected layer on the
> embeddings.
>
> Deliberately hobbling yourself with one layer is the point. A single linear layer can
> barely do anything on its own — so if it performs well, the credit belongs to the
> representation rather than the classifier. It's a clean measurement of embedding
> quality.
>
> It's also the honest test of the foundation-model promise. Can someone with two hundred
> labelled examples and no GPU get a good result by reusing your encoder?
>
> The benchmark has nineteen tasks across ten datasets. I worked on exactly one — T7, COPD
> detection from ICBHI lung sounds. Which is also, I noticed, one of the three where the
> paper reports OPERA being beaten by a general-audio baseline. That seemed like a more
> interesting place to look than one where everything already works.

---

## Slide 9 · The three encoders — 1m

> There are three released encoders, and I had to choose one.
>
> OPERA-CT is an HTSAT transformer inherited from CLAP, thirty-two million parameters.
> Rebuilding that faithfully wasn't a one-week job. CE is EfficientNet-B0, which would
> have meant reimplementing MBConv blocks correctly. GT is a masked-autoencoder Vision
> Transformer — a standard architecture I could write in pure PyTorch. So I chose GT.
>
> I want to flag a consequence of that choice early, because it cuts against me.
>
> GT is the generative model. My own reasoning — which I'll come to on slide seventeen —
> was that the *contrastive* models would be the ones most likely to encode nuisance
> variables, because their positive pairs come from the same recording. So I ended up
> testing the encoder that was *least* likely to show the effect I later went looking for.

*[Flagging this against yourself, unprompted, is exactly the right instinct. Say it
plainly.]*

---

## Slide 10 · Mine vs yours — 40s

> To be completely clear about attribution.
>
> Yours: the pretrained weights, and the ICBHI dataset. Mine: the encoder architecture,
> the entire audio front end including a hand-written mel filterbank, the feature
> extraction, the probing, the profiling, and the experimental design.
>
> I deliberately didn't use the repository's conda environment. That decision cost me
> something important, and I'll come back to it on slide sixteen.

---

## Slide 11 · Architecture recovery — 1m 5s

> There was no architecture file with the checkpoint, so I read the architecture off the
> tensor shapes.
>
> The patch embedding weight is three-eighty-four by one by four by four — so it's a
> convolution from one channel to three-eighty-four, with non-overlapping four-by-four
> patches. The positional embedding has one thousand and twenty-five entries, which is a
> thousand and twenty-four patches plus one CLS token. The QKV weight is eleven
> fifty-two by three eighty-four, and eleven fifty-two is three times three eighty-four,
> so it's a fused query-key-value projection. The MLP first layer gives a ratio of four.
> And the blocks run from zero to eleven, so depth twelve.
>
> Depth twelve, dimension three eighty-four, MLP ratio four — that's ViT-Small, whose
> convention is six attention heads.
>
> That head count is the one number I inferred rather than read. I've marked it as
> unverified everywhere it appears.

*[Steady pace. Reading numbers aloud is where people rush — don't.]*

---

## Slide 12 · Input shape — 55s

> The input shape I didn't choose. I derived it.
>
> The positional embedding tells you there are a thousand and twenty-four patches. Patches
> are four by four. The paper states sixty-four mel bins. So sixty-four over four, times
> n-frames over four, equals a thousand and twenty-four — which gives two hundred and
> fifty-six frames, or clips of eight point one nine two seconds.
>
> But there's an honest gap here that I want to put on the record.
>
> A hundred-and-twenty-eight by a hundred-and-twenty-eight input also gives a thousand and
> twenty-four patches. Both load. Both run. Neither raises an error. I picked
> sixty-four by two-fifty-six because the paper states sixty-four mel bins, which makes it
> much the likelier of the two — but I haven't verified it. And if I'm wrong, the
> embeddings would be quietly degraded rather than obviously broken.
>
> That's the first thing I'd want to check with you.

---

## Slide 13 · Audio front end — 1m

> The audio front end I wrote by hand rather than using torchaudio.
>
> The reason is that torchaudio's defaults differ subtly from the paper's stated
> parameters — the mel scale variant, the normalisation, the padding. Depending on them
> silently seemed like a good way to end up with a preprocessing mismatch that produces
> embeddings that look completely fine and mean nothing. And writing the mel filterbank
> turns out to be about five lines: space the filter edges uniformly on the mel scale,
> convert back to hertz, build triangles.
>
> The line I'd point to as mattering most is the per-example standardisation at the end.
>
> Log compression turns a multiplicative gain into an additive offset. So subtracting the
> per-example mean removes recording level entirely. That matters here specifically
> because ICBHI's four devices have very different sensitivities — without it, the model
> would be looking at equipment rather than pathology.
>
> There's a test that asserts a forty-decibel level change produces identical output.

---

## Slide 14 · Verification — 55s

> So how do I know the reimplementation is actually right?
>
> Three things.
>
> My encoder has twenty-one million, six hundred and ninety-four thousand, eight hundred
> and forty-eight parameters. The paper states twenty-one million. A wrong architecture
> would almost certainly either fail to load or give a different count.
>
> It loads with strict mode on, and zero missing keys. I made that deliberate — the loader
> raises if any parameter is missing, because a silent partial load would leave layers
> randomly initialised and produce embeddings that look entirely plausible and mean
> nothing. That was the failure mode I was most afraid of, and it seemed worth converting
> into an immediate exception.
>
> And the sanity check passes: two recordings from the same patient are more similar in
> embedding space than recordings from different patients.

---

## Slide 15 · The data — 1m 10s

> Two things about ICBHI shaped everything that follows.
>
> The first is that sixty-four of the hundred and twenty-six patients have COPD — about
> half — but seven hundred and ninety-three of the nine hundred and twenty recordings do.
> Eighty-six percent. COPD patients contribute roughly twelve recordings each; everyone
> else about two.
>
> That means accuracy is meaningless — a model that always says COPD scores eighty-six
> percent. And it means a random recording-wise split almost guarantees the same patient
> appears on both sides. So everything here uses AUROC and subject-wise splitting.
>
> The second thing I nearly missed entirely, and I want to describe it because catching it
> was probably the most valuable twenty minutes of the week.
>
> I checked the native sample rate of every file, broken down by device. And it turns out
> the Litt3200 recordings are all at four kilohertz, while the AKG and Littmann Classic
> ones are all at forty-four-one.
>
> Resample four kilohertz audio up to sixteen, and it has zero energy above two kilohertz.
> About half the mel bands — thirty of the sixty-four — are simply empty. Which means the Litt3200 is identifiable
> from that alone — nothing to do with the learned representation at all.

*[The pivot from "here's a dataset stat" to "here's a trap I nearly fell into" is the
interesting beat. Pause before it.]*

---

## Slide 16 · Experiment A — 1m 10s

> So, the first experiment: COPD detection.
>
> With subject-wise splits across eight resampled draws, the linear probe gets AUROC of
> zero point eight-seven-four. With a random split that lets patients leak across, it gets
> zero point nine-four-zero. And the shuffled-label control sits at zero point
> four-nine-two, which at least tells me the harness itself isn't leaking.
>
> Now — I want to be very clear about something here, because it's the biggest limitation
> in this whole study.
>
> **I have not compared this to your published T7 number, and I've deliberately not quoted
> one.** My clip selection, my pooling, and my split construction all differ from yours.
> Until I've read your benchmark definition line by line, my number and your number are
> not measuring the same thing, and calling this a reproduction would be overclaiming.
>
> I made a decision early on to write my own pipeline rather than use your environment,
> because I wanted to actually understand each stage. I think that was right for learning.
> But I should have read your T7 definition first, so I could have matched it from the
> start. Those two goals were never actually in conflict — that was just my mistake.
>
> The good news is it's cheap to fix. The features are cached, so it's a few hours rather
> than a rebuild.

*[Slow here. This is the honesty that makes everything else credible. Don't apologise
excessively — state it once, clearly, and move on.]*

---

## Slide 17 · C1, the question — 55s

> The second experiment came out of something I noticed while reading about the
> contrastive objective.
>
> If your positive pairs are two crops of the same recording, then the device, the
> subject, and the session are all constant within that recording. Which means they're
> perfectly predictive of a positive pair. A representation that encoded nothing except
> "which recording is this" would achieve zero contrastive loss.
>
> That made me curious about what else, besides disease, the embeddings might carry. So I
> pointed the same linear probe at variables that shouldn't matter clinically — recording
> device, chest location, patient identity.
>
> ICBHI happens to be recorded with four different stethoscopes, which is the only reason
> this was possible on public data at all.
>
> And because of the sample-rate problem I mentioned, I had to run it twice. Once
> naively, and once on a bandwidth-matched subset where I dropped every natively-low-rate
> file. The gap between those two is the actual measurement.

---

## Slide 18 · C1, the result — 1m 15s

> Here's what came out.
>
> The naive device probe gets eighty-six percent balanced accuracy on a four-class problem
> where chance is twenty-five percent.
>
> Then the control. Dropping every low-sample-rate file leaves three devices that share a
> bandwidth. And device is still recoverable at eighty-one percent, against a chance level
> of thirty-three.
>
> Normalising for the different chance levels, device reaches about seventy-one percent of
> the available headroom. COPD, on the same subset, under the same protocol, reaches about
> fifty-two.
>
> So device appears to be roughly twice as linearly decodable as the disease is.
>
> I should say — I expected the bandwidth control to explain most of that. I designed it
> assuming it would. It didn't; the signal barely moved. That was genuinely surprising to
> me, and it's the reason I think this is worth mentioning rather than something I should
> quietly drop.
>
> The shuffled control sits at chance, so I don't think it's an artefact of the probe.

*[This is the centrepiece. Slow, measured, no triumph in the voice. You're reporting
something that surprised you, not scoring a point.]*

---

## Slide 19 · C1, the caveats — 1m 5s

> But I want to be careful about what that does and doesn't show, because I think it's
> easy to overread.
>
> It is not evidence of shortcut learning. Linear decodability is a necessary condition,
> not a sufficient one. Showing that device information is *present* is not the same as
> showing the COPD classifier *uses* it. Proving that would need a cross-device
> generalisation test — train on some devices, hold others out — and I haven't run that.
>
> And there's a confound I can't remove with this dataset. In ICBHI, different clinical
> sites used different stethoscopes. So some of what my probe is reading might be site, or
> patient population, rather than microphone.
>
> I'm raising this as a question rather than a finding. It seemed potentially relevant
> given that the product has to work across a lot of different earbuds — but you'll know
> far better than I do whether that's a real concern or something you've long since
> accounted for.

*[The last sentence is important. Deliver it as genuine deference, not false modesty.]*

---

## Slide 20 · C2, deployment — 50s

> The third thing I looked at was what the encoder costs to run, because I couldn't find
> it reported anywhere and it seemed like the number that decides what's possible on a
> device.
>
> A hundred and twenty-five milliseconds median, at batch size one on CPU. Eighty-three
> megabytes in fp32.
>
> I used batch size one because a wearable classifies one clip at a time; I discarded
> warm-up iterations, because the first passes pay for kernel selection and including them
> inflates the median; and I report median and inter-quartile range rather than mean and
> standard deviation, because latency distributions are right-skewed and a mean flatters
> them.
>
> Two limitations. This is Apple Silicon, not embedded hardware — so treat it as an upper
> bound on a fast CPU, not a wearable estimate. And INT8 quantisation actually failed on
> my machine, so I have no quantised number. I've recorded that failure in the output
> rather than leaving it out.

---

## Slide 21 · C3, the baseline — 1m 10s

> The last experiment was the one I was most unsure about including.
>
> I trained a small CNN from scratch on the same task — sixty thousand parameters against
> the encoder's twenty-one point seven million. And it reached AUROC of zero point
> nine-six-nine, against the probe's zero point eight-seven-four.
>
> I don't think that should be read as "the foundation model doesn't work," and I want to
> give three reasons why.
>
> First, it isn't a like-for-like comparison. The probe is linear on frozen features,
> because that's the protocol. The CNN trains end to end. Fine-tuning the encoder would be
> the fair fight, and I haven't run it.
>
> Second, the CNN figure is a single split with best-epoch selection, which is optimistic.
> The probe figure is a mean over eight resampled splits.
>
> And third — the CNN may well be exploiting exactly the same device structure my earlier
> probe found, possibly more effectively than the frozen features do. So its advantage
> might partly *be* shortcut learning.
>
> The most I'd claim is: on this task, at this data scale, under the linear-probe
> protocol, the pretraining didn't pay for itself. Which is at least consistent with T7
> being one of the tasks your own paper reports OPERA losing.

---

## Slide 22 · The mistake — 1m 10s

> I want to spend a minute on something I got wrong, because it changed my conclusions
> more than any of my results did.
>
> My first full run reported zero variance. Not small variance — exactly zero, on every
> single probe.
>
> It looked like remarkable stability. It was nothing of the sort. Logistic regression
> fitted on fixed data is deterministic, so varying the classifier's random seed varies
> nothing at all. My error bars were describing solver noise, and solver noise is zero.
>
> The uncomfortable part is that I'd written a note to myself a few days earlier observing
> that measuring variance across probe seeds, rather than across data splits, would
> understate the real uncertainty. And then I implemented exactly that. I only caught it
> because the zeros were conspicuous.
>
> When I fixed it — resampling the split instead — the estimate of how much patient
> leakage inflates the result went from plus zero point zero-zero-four to plus zero point
> zero-six-six. A sixteen-fold difference. The single split had happened to land somewhere
> the leakage effect was invisible.
>
> If I'd reported from one split, I'd have concluded that patient leakage doesn't matter
> on this dataset. It does.

*[This is the slide that separates you from other candidates. Deliver it calmly — no
self-flagellation, no defensiveness. Just: here's what happened, here's what it taught me.]*

---

## Slide 23 · Limitations — 55s

> Let me put the limitations together in one place, because I'd rather state them than
> have them found.
>
> One encoder of three — and, as I said, the one my own reasoning suggested was least
> likely to show the device effect. One task of nineteen. Not benchmarked against your
> numbers, because the task definition is unmatched. The input orientation is unverified.
> The head count is inferred. I use one eight-second clip per recording, centred, when
> recordings run up to eighty-six seconds — so I'm discarding most of the audio. There's
> no fine-tuning comparison. And the latency is measured on a laptop.
>
> The honest label for all of this is: a verified reimplementation of the OPERA-GT
> encoder, and an independent evaluation of it on one dataset. Not a replication of your
> benchmark. I've tried to say that everywhere it appears, including in the code.

---

## Slide 24 · Next steps — 50s

> If I were carrying this further, in order:
>
> Match your T7 definition and re-run — that's the prerequisite for any comparison at all,
> and it's a few hours because the features are cached.
>
> Then the cross-device generalisation test, which is what would turn "device is
> decodable" into "device causes failure" — the claim that would actually matter.
>
> Then probe OPERA-CT, because my reasoning predicts the contrastive model should encode
> device more strongly than the generative one, and GT already scores high. That would
> tell you whether this is driven by the objective or is just inherent to spectrogram
> encoders.
>
> Then profile CE, which at four million parameters might invert the practical
> recommendation for a wearable.
>
> And give the CNN baseline a fair fight — same eight splits, proper early stopping.

---

## Slide 25 · Close — 50s

> Last slide, and I mostly want to say thank you.
>
> None of this would have been possible if you hadn't released the weights and the code.
> Being able to take a NeurIPS model, rebuild its encoder, and test my own understanding
> against it inside a week is a direct consequence of that decision, and I'm genuinely
> grateful for it.
>
> I'm very aware that I'm a week into something you've spent a decade on, and that some of
> what I've said here is probably naive, or already well understood, or just wrong. I'd
> welcome being told where — honestly, I'd learn more from that conversation than from
> anything in these slides.
>
> Thank you very much for watching.

*[Warm. Slow the last two lines. Then hold the slide for two seconds before you stop
recording — don't cut on the final syllable.]*

---

## If you need a shorter version

**~9 minutes:** slides 1, 3, 5, 7, 9, 14, 15, 16, 18, 19, 22, 25.
Keeps the framing, the verification, the headline finding with its caveats, and the
mistake. Drops most of the theory and the implementation detail.

**Do not cut slides 16, 19, 22 or 25 under any circumstances.** They are what make the
whole thing credible rather than boastful.

---

## Likely questions, and honest answers

**"Why didn't you just match our benchmark?"**
> That was my mistake. I wrote my own pipeline to make sure I understood every stage,
> which I think was right for learning, but I should have read your T7 definition first so
> I could have matched it from the start. The two weren't actually in conflict. It's a few
> hours to fix and it's top of my list.

**"Isn't the device result just the dataset?"**
> Possibly, partly — device is confounded with clinical site in ICBHI and I can't separate
> those. What I can say is that it isn't the sample-rate artefact, because the
> bandwidth-matched control removed that and the signal barely moved. Whether it's the
> microphone or the site, I'd need cross-device evaluation to tell.

**"Your CNN beat the foundation model — doesn't that undermine the whole approach?"**
> I don't think so, and I'd be careful with that result. It's not like-for-like, it's a
> single split with best-epoch selection, and the CNN may be exploiting the same device
> structure. The most I'd claim is that on this task at this scale, under linear probing,
> the pretraining didn't pay for itself.

**"How do you know your encoder is right?"**
> The parameter count matches your stated 21M exactly — 21,694,848 — and it loads with
> strict mode on and zero missing keys. A wrong architecture would almost certainly fail
> one of those. What I haven't verified is the input orientation and the head count.

**"What would you do first if you joined?"**
> Match the benchmark definition and run the cross-device test, because that's the
> question I couldn't answer and it seems like the one that matters for shipping on
> hardware you don't control.

**"Why GT and not CT?"**
> Purely tractability — HTSAT wasn't a one-week rebuild. And I'd note it cuts against me:
> GT is the model my own reasoning said was least likely to show the device effect, so
> testing CT is the obvious next step.
