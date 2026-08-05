const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";               // 13.333 x 7.5
pres.author = "Rohan Sakeri";
pres.title = "OPERA-GT on ICBHI";

// ---------------------------------------------------------------- palette
const DARK   = "0A2A33";   // near-black teal — title/close
const TEAL   = "0F5C6B";   // dominant
const SEA    = "00A896";   // accent
const INK    = "16232A";
const MUTED  = "566C78";
const FAINT  = "8CA0AB";
const PANEL  = "F1F6F7";
const PANELB = "DCE7EA";
const WARN   = "9A5B12";
const WARNBG = "FDF4E6";
const WARNBD = "F0DDBE";
const GOOD   = "1C6B45";
const GOODBG = "E9F5EF";
const GOODBD = "C6E3D4";
const WHITE  = "FFFFFF";

const HEAD = "Cambria";
const BODY = "Calibri";

const M = 0.62;            // left margin
const W = 13.333 - M * 2;  // usable width

// ---------------------------------------------------------------- helpers
function kicker(s, text) {
  s.addText(text.toUpperCase(), {
    x: M, y: 0.42, w: W, h: 0.26, fontFace: BODY, fontSize: 11.5, bold: true,
    color: TEAL, charSpacing: 1.6, margin: 0,
  });
}

function title(s, text, y = 0.76, size = 30) {
  s.addText(text, {
    x: M, y, w: W, h: 0.95, fontFace: HEAD, fontSize: size, bold: true,
    color: INK, margin: 0, lineSpacingMultiple: 1.02, valign: "top",
  });
}

function foot(s, left, n) {
  s.addShape(pres.ShapeType.line, {
    x: M, y: 6.92, w: W, h: 0, line: { color: PANELB, width: 0.75 },
  });
  s.addText(left, { x: M, y: 6.98, w: W / 2, h: 0.3, fontFace: BODY, fontSize: 10,
    color: FAINT, margin: 0 });
  s.addText(String(n), { x: M + W / 2, y: 6.98, w: W / 2, h: 0.3, fontFace: BODY,
    fontSize: 10, color: FAINT, align: "right", margin: 0 });
}

function card(s, { x, y, w, h, tone = "plain" }) {
  const map = {
    plain: [PANEL, PANELB], warn: [WARNBG, WARNBD],
    good: [GOODBG, GOODBD], accent: ["E4F2F4", "BFDFE4"],
  };
  const [fill, line] = map[tone];
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.09, fill: { color: fill },
    line: { color: line, width: 1 },
  });
}

function cardHead(s, text, { x, y, w, color = INK }) {
  s.addText(text, { x, y, w, h: 0.3, fontFace: HEAD, fontSize: 15.5, bold: true,
    color, margin: 0 });
}

function para(s, text, { x, y, w, h, size = 13.5, color = MUTED, bold = false }) {
  s.addText(text, { x, y, w, h, fontFace: BODY, fontSize: size, color, bold,
    margin: 0, lineSpacingMultiple: 1.16, valign: "top" });
}

function bullets(s, items, { x, y, w, h, size = 13.5, color = MUTED }) {
  s.addText(
    items.map((t, i) => ({
      text: t, options: { bullet: { code: "2022" }, breakLine: i !== items.length - 1 },
    })),
    { x, y, w, h, fontFace: BODY, fontSize: size, color, margin: 0,
      paraSpaceAfter: 7, lineSpacingMultiple: 1.14, valign: "top" }
  );
}

function numbered(s, items, { x, y, w, gap = 0.72, size = 14 }) {
  items.forEach((t, i) => {
    const yy = y + i * gap;
    s.addShape(pres.ShapeType.ellipse, {
      x, y: yy, w: 0.36, h: 0.36, fill: { color: TEAL },
    });
    s.addText(String(i + 1).padStart(2, "0"), {
      x, y: yy, w: 0.36, h: 0.36, fontFace: BODY, fontSize: 11, bold: true,
      color: WHITE, align: "center", valign: "middle", margin: 0,
    });
    s.addText(t, {
      x: x + 0.54, y: yy - 0.03, w: w - 0.54, h: gap - 0.06, fontFace: BODY,
      fontSize: size, color: MUTED, margin: 0, lineSpacingMultiple: 1.14, valign: "top",
    });
  });
}

function stat(s, { x, y, w, big, cap, color = TEAL }) {
  card(s, { x, y, w, h: 1.85 });
  s.addText(big, { x, y: y + 0.28, w, h: 0.75, fontFace: HEAD, fontSize: 40,
    bold: true, color, align: "center", margin: 0 });
  s.addText(cap, { x: x + 0.16, y: y + 1.06, w: w - 0.32, h: 0.65, fontFace: BODY,
    fontSize: 11.5, color: MUTED, align: "center", margin: 0, lineSpacingMultiple: 1.12 });
}

// Canva's PPTX importer frequently drops or mangles native <a:tbl> tables.
// NO_TABLES=1 draws the same grids from text boxes and hairlines instead, which
// every importer handles, at the cost of the cells no longer being editable as a table.
const NO_TABLES = process.env.NO_TABLES === "1";

function drawGrid(s, rows, { x, y, w, colW, headRow = true, hiRow = -1 }) {
  const rowH = 0.42;
  let cy = y;
  rows.forEach((r, ri) => {
    const isHead = ri === 0 && headRow;
    const isHi = ri === hiRow;
    if (isHi) {
      s.addShape(pres.ShapeType.roundRect, {
        x: x - 0.06, y: cy, w: w + 0.12, h: rowH, rectRadius: 0.05,
        fill: { color: "E4F2F4" }, line: { color: "E4F2F4", width: 0.5 },
      });
    }
    let cx = x;
    r.forEach((cell, ci) => {
      s.addText(String(cell), {
        x: cx, y: cy, w: colW[ci], h: rowH,
        fontFace: BODY,
        fontSize: isHead ? 10.5 : 12.5,
        bold: isHead || isHi,
        color: isHead ? FAINT : (isHi ? INK : MUTED),
        valign: "middle", margin: [0, 5, 0, 5],
      });
      cx += colW[ci];
    });
    cy += rowH;
    if (ri < rows.length - 1) {
      s.addShape(pres.ShapeType.line, {
        x, y: cy, w, h: 0,
        line: { color: PANELB, width: isHead ? 1.25 : 0.6 },
      });
    }
  });
}

function table(s, rows, opts) {
  if (NO_TABLES) return drawGrid(s, rows, opts);
  const { x, y, w, colW, headRow = true, hiRow = -1 } = opts;
  const tRows = rows.map((r, ri) => r.map((c) => ({
    text: String(c),
    options: {
      fontFace: BODY,
      fontSize: ri === 0 && headRow ? 10.5 : 13,
      bold: (ri === 0 && headRow) || ri === hiRow,
      color: ri === 0 && headRow ? FAINT : (ri === hiRow ? INK : MUTED),
      fill: { color: ri === hiRow ? "E4F2F4" : WHITE },
      valign: "middle",
      margin: [5, 8, 5, 8],
    },
  })));
  s.addTable(tRows, {
    x, y, w, colW, border: { type: "solid", color: PANELB, pt: 0.75 },
    autoPage: false,
  });
}

function flow(s, nodes, { x, y, w }) {
  const gap = 0.16;
  const arrowW = 0.22;
  const totalArrows = (nodes.length - 1) * (arrowW + gap * 2);
  const nodeW = (w - totalArrows) / nodes.length;
  let cx = x;
  nodes.forEach((n, i) => {
    const hot = typeof n === "object";
    const label = hot ? n.t : n;
    card(s, { x: cx, y, w: nodeW, h: 0.62, tone: hot ? "accent" : "plain" });
    s.addText(label, { x: cx + 0.06, y, w: nodeW - 0.12, h: 0.62, fontFace: BODY,
      fontSize: 11, color: INK, align: "center", valign: "middle", margin: 0 });
    cx += nodeW;
    if (i < nodes.length - 1) {
      s.addText("→", { x: cx + gap, y, w: arrowW, h: 0.62, fontFace: BODY,
        fontSize: 15, color: FAINT, align: "center", valign: "middle", margin: 0 });
      cx += arrowW + gap * 2;
    }
  });
}

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: DARK };
  return s;
}

// ================================================================ SLIDES

// 1 — Title
{
  const s = darkSlide();
  s.addText("A SMALL INDEPENDENT STUDY  ·  AUGUST 2026", {
    x: M, y: 1.75, w: W, h: 0.3, fontFace: BODY, fontSize: 12, bold: true,
    color: SEA, charSpacing: 1.8, margin: 0,
  });
  s.addText("Reimplementing OPERA-GT\nand probing what it learned", {
    x: M, y: 2.2, w: 10.4, h: 1.9, fontFace: HEAD, fontSize: 40, bold: true,
    color: WHITE, margin: 0, lineSpacingMultiple: 1.06,
  });
  s.addText(
    "An attempt to understand the Cambridge Mobile Systems Lab's respiratory foundation model by rebuilding its encoder from scratch — and three small experiments it prompted.",
    { x: M, y: 4.25, w: 9.4, h: 0.9, fontFace: BODY, fontSize: 15, color: "B7CDD3",
      margin: 0, lineSpacingMultiple: 1.2 }
  );
  s.addText("Rohan Sakeri  ·  ML Engineer", { x: M, y: 5.45, w: W, h: 0.3,
    fontFace: BODY, fontSize: 14, bold: true, color: WHITE, margin: 0 });
  s.addText("Offered with respect, and with all its limitations stated openly.", {
    x: M, y: 5.78, w: W, h: 0.3, fontFace: BODY, fontSize: 12, color: "7E9AA3",
    italic: true, margin: 0 });
  s.addNotes("Hello. My name is Rohan Sakeri, and this is a short walkthrough of a small study I did over about a week. I rebuilt the encoder from OPERA — the open respiratory acoustic foundation model from the Cambridge Mobile Systems Lab — from scratch, and then ran three small experiments that came out of trying to understand it. I want to say at the very start that this is a study and not a result. I'm one week into a field that some of you have spent a decade in.");
}

// 2 — Why
{
  const s = pres.addSlide();
  kicker(s, "Why I did this");
  title(s, "I wanted to understand your work properly\nbefore asking to be part of it", 0.76, 28);
  para(s, "Reading a paper and saying you found it interesting is easy. I wasn't sure whether I had actually understood OPERA, and the only honest test I could think of was to rebuild the encoder from the released weights and see whether it worked.",
    { x: M, y: 2.55, w: 11.4, h: 1.0, size: 15 });
  card(s, { x: M, y: 3.85, w: 11.4, h: 1.55, tone: "accent" });
  para(s, "Everything here was done in about a week, on a laptop, using publicly released checkpoints and the public ICBHI dataset. It is a study, not a result. I have almost certainly made mistakes, and I would genuinely value being corrected on any of it.",
    { x: M + 0.32, y: 4.12, w: 10.75, h: 1.1, size: 14.5, color: INK });
  foot(s, "Rohan Sakeri", 2);
  s.addNotes("The honest reason I did this is that I wanted to understand your work properly before asking to be part of it. It's easy to read a paper and say you found it interesting. The only test I could think of that would give me a real answer was to rebuild the encoder and see whether it worked. I've almost certainly made mistakes somewhere. If any of you spot one, I'd genuinely rather know.");
}

// 3 — The paper, and what I did
{
  const s = pres.addSlide();
  kicker(s, "The whole thing in one slide");
  title(s, "The paper, and what I did with it");
  const cw = (W - 0.45) / 2;

  card(s, { x: M, y: 1.9, w: cw, h: 3.15 });
  cardHead(s, "OPERA, in three sentences", { x: M + 0.3, y: 2.14, w: cw - 0.6 });
  bullets(s, [
    "Respiratory audio has almost no labelled data, so every group trains a bespoke model on a tiny dataset and nothing transfers.",
    "OPERA pretrains encoders on 404 hours of unlabelled coughs, breaths and lung sounds, then evaluates them by freezing them and training a single linear layer on 19 downstream health tasks.",
    "It releases three encoders and shows domain-specific pretraining beats general-audio models on 16 of those 19 tasks.",
  ], { x: M + 0.3, y: 2.58, w: cw - 0.6, h: 2.35, size: 12.5 });

  card(s, { x: M + cw + 0.45, y: 1.9, w: cw, h: 3.15, tone: "accent" });
  cardHead(s, "What I did, in three sentences", { x: M + cw + 0.75, y: 2.14, w: cw - 0.6 });
  bullets(s, [
    "Rebuilt one of the three encoders — OPERA-GT — from scratch in PyTorch, recovering its architecture from the released checkpoint's tensor shapes.",
    "Evaluated it on one task, T7, COPD detection from ICBHI lung sounds, with subject-wise splits and controls.",
    "Then asked three things the paper does not: what else the embeddings encode, what the model costs to run, and whether a small supervised CNN does better.",
  ], { x: M + cw + 0.75, y: 2.58, w: cw - 0.6, h: 2.35, size: 12.5, color: INK });

  card(s, { x: M, y: 5.3, w: 11.4, h: 1.15, tone: "warn" });
  s.addText([
    { text: "Scope, stated up front:  ", options: { bold: true, color: WARN } },
    { text: "1 of 3 encoders  ·  1 of 19 tasks  ·  no pretraining — I used their released weights  ·  not benchmarked against their published numbers", options: { color: INK } },
  ], { x: M + 0.32, y: 5.5, w: 10.75, h: 0.35, fontFace: BODY, fontSize: 13, margin: 0 });
  s.addText("1,859 lines of source  ·  560 lines of tests  ·  76 tests passing", {
    x: M + 0.32, y: 5.9, w: 10.75, h: 0.3, fontFace: BODY, fontSize: 12.5,
    color: MUTED, margin: 0 });
  foot(s, "Summary", 3);
  s.addNotes("Before anything else, here is the whole thing in one slide. The paper first. Respiratory audio has almost no labelled data, so every group trains its own model on a tiny dataset and nothing transfers. OPERA pretrains encoders on 404 hours of unlabelled coughs, breaths and lung sounds, then evaluates them by freezing them and training a single linear layer on nineteen downstream health tasks. It releases three encoders and shows that domain-specific pretraining beats general-audio models on sixteen of those nineteen. What I did: I rebuilt one of the three encoders, OPERA-GT, from scratch in PyTorch, recovering the architecture from the checkpoint's tensor shapes. I evaluated it on one task, COPD detection from ICBHI lung sounds. And then I asked three things the paper doesn't — what else the embeddings encode, what the model costs to run, and whether a small supervised CNN does better. I want the scope on the record from the start: one encoder of three, one task of nineteen, no pretraining, and not benchmarked against their published numbers. I'll come back to that last one.");
}

// 4 — RespEar / hEARt
{
  const s = pres.addSlide();
  kicker(s, "Choosing what to work on");
  title(s, "I started with RespEar and hEARt —\nand decided not to implement them", 0.76, 27);
  para(s, "They are the papers I most wanted to work on. They are also the ones I could not responsibly attempt in a week.",
    { x: M, y: 2.42, w: 11.4, h: 0.4, size: 14 });
  const cw = (W - 0.4) / 2;
  card(s, { x: M, y: 3.0, w: cw, h: 2.6 });
  cardHead(s, "Why I read them anyway", { x: M + 0.28, y: 3.25, w: cw - 0.56 });
  bullets(s, [
    "RespEar's indirect measurement — recovering breathing from RSA and from stride coupling rather than trying to hear it — is the most elegant idea I read all month.",
    "hEARt's split of learned denoising followed by classical estimation is a design pattern I have taken away.",
  ], { x: M + 0.28, y: 3.68, w: cw - 0.56, h: 1.8, size: 12.5 });

  card(s, { x: M + cw + 0.4, y: 3.0, w: cw, h: 2.6, tone: "warn" });
  cardHead(s, "Why I didn't implement them", { x: M + cw + 0.68, y: 3.25, w: cw - 0.56, color: WARN });
  bullets(s, [
    "The datasets are in-lab and not public — 18 and 20 subjects, custom hardware.",
    "They are principally signal-processing work. Implementing them would have taught me DSP but written no PyTorch.",
    "They need a sealed in-ear microphone I do not have.",
  ], { x: M + cw + 0.68, y: 3.68, w: cw - 0.56, h: 1.8, size: 12.5 });
  foot(s, "Choosing the paper", 4);
  s.addNotes("Before OPERA, I want to explain what I didn't do. I started with RespEar and hEARt — the papers on your website. I read them properly and I'm glad I did. RespEar's central idea, that rather than trying to hear breathing you recover it from its fingerprint on things that are loud, is the most elegant thing I read all month. But the datasets are in-lab and not public. They're also principally signal-processing work, so implementing them would have taught me DSP but written no PyTorch.");
}

// 5 — Why OPERA
{
  const s = pres.addSlide();
  kicker(s, "Choosing what to work on");
  title(s, "So I chose OPERA — the one piece of the lab's\noutput I could actually run", 0.76, 27);
  numbered(s, [
    "Professor Mascolo is senior author, so it is genuinely the lab's work rather than something adjacent to it.",
    "The checkpoints are released, so no pretraining is required — 404 hours was never within reach.",
    "Linear probing is the paper's own protocol, and a frozen encoder costs almost nothing to evaluate.",
    "ICBHI is public, small, and has one property that made a later experiment possible at all.",
  ], { x: M, y: 2.5, w: 11.4, gap: 0.62 });
  card(s, { x: M, y: 5.15, w: 11.4, h: 1.35, tone: "accent" });
  para(s, "There was also a simpler reason. Your posting describes the goal as \"the world's best foundation model for turning sound into health insights.\" OPERA seemed like the closest public thing to that direction, so it felt like the right place to try to be useful.",
    { x: M + 0.3, y: 5.38, w: 10.8, h: 0.95, size: 13.5, color: INK });
  foot(s, "Choosing the paper", 5);
  s.addNotes("Professor Mascolo is senior author, so it's genuinely the lab's work. The checkpoints are released, so no pretraining is needed. Linear probing is the paper's own protocol. And there was a simpler reason — your job posting describes the goal as building the world's best foundation model for turning sound into health insights. OPERA seemed like the closest public thing to that.");
}

// 6 — The problem
{
  const s = pres.addSlide();
  kicker(s, "The theory · 1 of 4");
  title(s, "The problem OPERA sets out to solve");
  const cw = (W - 0.45) / 2;
  cardHead(s, "Labels are the bottleneck", { x: M, y: 2.05, w: cw });
  para(s, "Training a COPD detector traditionally needs lung recordings a clinician has already diagnosed — recruitment, ethics, spirometry. It is slow and expensive, so respiratory datasets stay tiny. ICBHI has 920 recordings.\n\nEvery group then trains a bespoke model on its own small set, reports a number, and nothing transfers.",
    { x: M, y: 2.5, w: cw, h: 2.4, size: 13.5 });
  card(s, { x: M + cw + 0.45, y: 2.0, w: cw, h: 3.5, tone: "accent" });
  cardHead(s, "The move", { x: M + cw + 0.73, y: 2.28, w: cw - 0.56 });
  para(s, "Separate learning what breathing sounds like from learning what disease sounds like.\n\nThe first needs no labels at all, and there are hundreds of thousands of unlabelled coughs and breaths in the world. Do it once, expensively. Everyone else reuses it.\n\nThat reusable encoder is the foundation model.",
    { x: M + cw + 0.73, y: 2.72, w: cw - 0.56, h: 2.6, size: 13, color: INK });
  foot(s, "Theory", 6);
  s.addNotes("The problem OPERA addresses is that labels are the bottleneck. Respiratory datasets stay small — ICBHI has 920 recordings. You can't train a modern network on that. The move the paper makes is to separate learning what breathing sounds like from learning what disease sounds like. The first needs no labels at all. You do that once, expensively, and everyone else reuses it.");
}

// 7 — Two strategies
{
  const s = pres.addSlide();
  kicker(s, "The theory · 2 of 4");
  title(s, "Two ways to learn without labels");
  const cw = (W - 0.4) / 2;
  card(s, { x: M, y: 1.95, w: cw, h: 2.3 });
  cardHead(s, "Contrastive — OPERA-CT, CE", { x: M + 0.28, y: 2.18, w: cw - 0.56 });
  para(s, "Two crops from the same spectrogram are a positive pair; crops from other recordings are negatives. The model picks the partner from a lineup.\n\nLearns what makes recordings different from each other — discriminative features.",
    { x: M + 0.28, y: 2.6, w: cw - 0.56, h: 1.5, size: 12.5 });
  card(s, { x: M + cw + 0.4, y: 1.95, w: cw, h: 2.3 });
  cardHead(s, "Generative — OPERA-GT", { x: M + cw + 0.68, y: 2.18, w: cw - 0.56 });
  para(s, "Chop into patches, hide 70%, reconstruct what was behind the mask.\n\nLearns what respiratory sound looks like in general — descriptive features, because you cannot redraw what you only vaguely understand.",
    { x: M + cw + 0.68, y: 2.6, w: cw - 0.56, h: 1.5, size: 12.5 });
  card(s, { x: M, y: 4.5, w: W, h: 1.75, tone: "good" });
  para(s, "The paper's results show a near-perfect double dissociation: contrastive wins classification (MRR 0.694), generative wins regression (0.655), and the two almost invert between task groups. I found that the most satisfying thing in the paper — it has a clean theoretical explanation and it generalises well beyond audio.",
    { x: M + 0.32, y: 4.78, w: W - 0.64, h: 1.25, size: 13.5, color: INK });
  foot(s, "Theory", 7);
  s.addNotes("Contrastive learning asks what makes recordings different, so it keeps discriminative features. Masked reconstruction asks what respiratory sound looks like in general, so it keeps descriptive detail. And the paper's results show almost exactly that — contrastive wins classification, generative wins regression, and the two nearly invert. I found that the most satisfying thing in the paper.");
}

// 8 — Linear probing
{
  const s = pres.addSlide();
  kicker(s, "The theory · 3 of 4");
  title(s, "Linear probing — and why it is deliberately weak");
  flow(s, ["frozen encoder", "embeddings", { t: "one linear layer" }, "AUROC"],
    { x: M, y: 2.0, w: 11.4 });
  para(s, "A single fully-connected layer can barely do anything by itself. That is the point: if it performs well, the credit belongs to the representation, not the classifier. It is also the honest test of the foundation-model promise — can someone with 200 labelled examples and no GPU get a good result by reusing your encoder?",
    { x: M, y: 3.0, w: 11.4, h: 1.15, size: 14 });
  card(s, { x: M, y: 4.35, w: 11.4, h: 1.7 });
  para(s, "19 tasks across 10 datasets — 12 classification scored by AUROC, 7 lung-function regression scored by MAE. I worked on exactly one: T7, COPD detection from ICBHI lung sounds. Which is also one of the three where the paper reports OPERA being beaten by a general-audio baseline. That seemed a more interesting place to look than one where everything already works.",
    { x: M + 0.32, y: 4.62, w: 10.75, h: 1.25, size: 13.5, color: INK });
  foot(s, "Theory", 8);
  s.addNotes("Freeze the encoder, push labelled data through, train exactly one fully-connected layer. Deliberately hobbling yourself is the point — if it performs well, the credit belongs to the representation. The benchmark has nineteen tasks. I worked on one, T7, which is also one of the three where the paper reports OPERA being beaten by a general-audio baseline.");
}

// 9 — Three encoders
{
  const s = pres.addSlide();
  kicker(s, "The theory · 4 of 4");
  title(s, "The three released encoders");
  table(s, [
    ["Model", "Architecture", "Params", "Objective", "Could I rebuild it?"],
    ["OPERA-CT", "HTSAT transformer, from CLAP", "32.4M", "Contrastive", "Not in a week — needs CLAP"],
    ["OPERA-CE", "EfficientNet-B0", "4.97M", "Contrastive", "Needs MBConv reimplementation"],
    ["OPERA-GT", "MAE Vision Transformer", "21M", "Generative", "Yes — a standard ViT, pure PyTorch"],
  ], { x: M, y: 1.95, w: 11.4, colW: [1.7, 3.3, 1.1, 1.7, 3.6], hiRow: 3 });
  card(s, { x: M, y: 4.3, w: 11.4, h: 1.85, tone: "warn" });
  cardHead(s, "A consequence I should flag early", { x: M + 0.32, y: 4.55, w: 10.75, color: WARN });
  para(s, "GT is the generative model. My expectation was that the contrastive models would be the ones most likely to encode nuisance variables, because their positive pairs come from the same recording. So I ended up testing the encoder least likely to show the effect I later went looking for.",
    { x: M + 0.32, y: 4.98, w: 10.75, h: 1.0, size: 13.5, color: INK });
  foot(s, "Theory", 9);
  s.addNotes("I had to choose one encoder. CT is an HTSAT transformer from CLAP — not a one-week rebuild. CE is EfficientNet. GT is a standard Vision Transformer I could write in pure PyTorch. I want to flag a consequence that cuts against me: GT is the generative model, and my own reasoning was that the contrastive models would be most likely to encode nuisance variables. So I tested the encoder least likely to show the effect I later went looking for.");
}

// 10 — Mine vs yours
{
  const s = pres.addSlide();
  kicker(s, "Implementation · 1 of 5");
  title(s, "What is mine and what is yours");
  const cw = (W - 0.4) / 2;
  card(s, { x: M, y: 2.0, w: cw, h: 2.3 });
  cardHead(s, "Yours", { x: M + 0.28, y: 2.25, w: cw - 0.56, color: MUTED });
  bullets(s, [
    "The pretrained OPERA-GT weights, from huggingface.co/evelyn0414/OPERA",
    "The ICBHI 2017 dataset (Rocha et al.), via the public Kaggle mirror",
  ], { x: M + 0.28, y: 2.68, w: cw - 0.56, h: 1.4, size: 13 });
  card(s, { x: M + cw + 0.4, y: 2.0, w: cw, h: 2.3, tone: "accent" });
  cardHead(s, "Mine", { x: M + cw + 0.68, y: 2.25, w: cw - 0.56 });
  bullets(s, [
    "The encoder architecture, written from the checkpoint's tensor shapes",
    "The entire audio front end, including a hand-written mel filterbank",
    "Feature extraction, probing, evaluation, profiling",
    "The experimental design and all analysis",
  ], { x: M + cw + 0.68, y: 2.68, w: cw - 0.56, h: 1.5, size: 13, color: INK });
  para(s, "I deliberately did not use the repository's conda environment. That cost me something important, which I come back to on slide 16.",
    { x: M, y: 4.7, w: 11.4, h: 0.6, size: 14 });
  foot(s, "Implementation", 10);
  s.addNotes("Yours: the pretrained weights and the ICBHI dataset. Mine: the encoder architecture, the entire audio front end including a hand-written mel filterbank, the feature extraction, the probing, the profiling, and the experimental design. I deliberately didn't use the repository's conda environment — that decision cost me something important, and I'll come back to it.");
}

// 11 — Architecture recovery
{
  const s = pres.addSlide();
  kicker(s, "Implementation · 2 of 5");
  title(s, "Recovering the architecture from the weights");
  para(s, "There was no architecture file, so I read it off the tensor shapes.",
    { x: M, y: 1.82, w: 11.4, h: 0.32, size: 14 });
  table(s, [
    ["Checkpoint tensor", "Shape", "What it implies"],
    ["patch_embed.proj.weight", "(384, 1, 4, 4)", "Conv2d 1 → 384, non-overlapping 4×4 patches"],
    ["pos_embed", "(1, 1025, 384)", "1024 patches + one CLS token"],
    ["blocks.0.attn.qkv.weight", "(1152, 384)", "1152 = 3 × 384 → fused QKV projection"],
    ["blocks.0.mlp.fc1.weight", "(1536, 384)", "MLP ratio 4"],
    ["blocks.11.*", "—", "Depth 12"],
  ], { x: M, y: 2.28, w: 11.4, colW: [3.6, 2.4, 5.4] });
  para(s, "Depth 12, dim 384, MLP ratio 4 is ViT-Small, whose convention is 6 heads. That head count is the one number I inferred rather than read, and I have marked it unverified everywhere it appears.",
    { x: M, y: 5.55, w: 11.4, h: 0.8, size: 13.5 });
  foot(s, "Implementation", 11);
  s.addNotes("There was no architecture file, so I read the architecture off the tensor shapes. The patch embedding weight tells you it's a convolution with non-overlapping four-by-four patches. The positional embedding has 1025 entries — 1024 patches plus a CLS token. The QKV weight is 1152 by 384, and 1152 is three times 384, so it's a fused projection. Depth twelve, dimension 384, MLP ratio four — that's ViT-Small, whose convention is six heads. That head count is the one number I inferred rather than read.");
}

// 12 — Input shape
{
  const s = pres.addSlide();
  kicker(s, "Implementation · 3 of 5");
  title(s, "The input shape is derived, not chosen");
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 1.95, w: 11.4, h: 1.95, rectRadius: 0.09,
    fill: { color: "0E1D24" }, line: { color: "0E1D24", width: 1 },
  });
  s.addText([
    { text: "pos_embed = 1025 = 1024 patches + 1 CLS", options: { breakLine: true, color: "C9DEE3" } },
    { text: "patch size = 4 × 4", options: { breakLine: true, color: "C9DEE3" } },
    { text: "paper states 64 mel bins", options: { breakLine: true, color: "C9DEE3" } },
    { text: "→  (64 / 4) × (n_frames / 4) = 1024", options: { breakLine: true, color: "6FD9C0" } },
    { text: "→  n_frames = 256  →  8.192 s clips", options: { color: "6FD9C0" } },
  ], { x: M + 0.35, y: 2.15, w: 10.7, h: 1.6, fontFace: "Courier New", fontSize: 14,
       margin: 0, lineSpacingMultiple: 1.32 });
  card(s, { x: M, y: 4.15, w: 11.4, h: 2.0, tone: "warn" });
  cardHead(s, "An honest gap", { x: M + 0.32, y: 4.4, w: 10.75, color: WARN });
  para(s, "A 128 × 128 input also yields 1024 patches. Both load, both run, neither raises an error. I chose 64 × 256 because the paper states 64 mel bins — which makes it much the likelier of the two — but I have not verified it, and if I am wrong the embeddings would be quietly degraded rather than obviously broken. It is the first thing I would check with you.",
    { x: M + 0.32, y: 4.82, w: 10.75, h: 1.2, size: 13.5, color: INK });
  foot(s, "Implementation", 12);
  s.addNotes("The input shape I didn't choose — I derived it. The positional embedding tells you there are 1024 patches, patches are four by four, and the paper states 64 mel bins. That gives 256 frames, or clips of 8.192 seconds. But there's an honest gap. A 128 by 128 input also gives 1024 patches. Both load, both run, neither errors. I picked 64 by 256 because the paper states 64 mel bins, but I haven't verified it. That's the first thing I'd want to check with you.");
}

// 13 — Audio front end
{
  const s = pres.addSlide();
  kicker(s, "Implementation · 4 of 5");
  title(s, "The audio front end, written by hand");
  flow(s, ["wav", "resample 16 kHz", "centre clip", "STFT 1024/512",
    { t: "64 mel triangles" }, "log", { t: "standardise" }],
    { x: M, y: 1.95, w: 11.4 });
  const cw = (W - 0.4) / 2;
  card(s, { x: M, y: 2.95, w: cw, h: 2.5 });
  cardHead(s, "Why no torchaudio", { x: M + 0.28, y: 3.2, w: cw - 0.56 });
  para(s, "Its defaults differ subtly from the paper's stated parameters — mel scale variant, normalisation, padding. Depending on them silently would risk a preprocessing mismatch that produces plausible-looking, meaningless embeddings. Writing the filterbank is five lines.",
    { x: M + 0.28, y: 3.62, w: cw - 0.56, h: 1.7, size: 12.5 });
  card(s, { x: M + cw + 0.4, y: 2.95, w: cw, h: 2.5, tone: "accent" });
  cardHead(s, "The line that matters most", { x: M + cw + 0.68, y: 3.2, w: cw - 0.56 });
  para(s, "Per-example standardisation. Log compression turns gain into an additive offset, so subtracting the mean removes recording level entirely. ICBHI's four devices have very different sensitivities — without this, the model would see equipment rather than pathology.",
    { x: M + cw + 0.68, y: 3.62, w: cw - 0.56, h: 1.7, size: 12.5, color: INK });
  foot(s, "Implementation", 13);
  s.addNotes("I wrote the audio front end by hand rather than using torchaudio, because torchaudio's defaults differ subtly from the paper's stated parameters, and depending on them silently seemed like a good way to end up with embeddings that look fine and mean nothing. The line that matters most is the per-example standardisation. Log compression turns a multiplicative gain into an additive offset, so subtracting the mean removes recording level entirely. Without it, the model would look at equipment rather than pathology.");
}

// 14 — Verification
{
  const s = pres.addSlide();
  kicker(s, "Implementation · 5 of 5");
  title(s, "How I know the reimplementation is correct");
  const sw = (W - 0.5) / 3;
  stat(s, { x: M, y: 2.1, w: sw, big: "21,694,848", cap: "parameters in my encoder\nthe paper states 21M", color: GOOD });
  stat(s, { x: M + sw + 0.25, y: 2.1, w: sw, big: "0", cap: "missing keys on a\nstrict=True load", color: GOOD });
  stat(s, { x: M + (sw + 0.25) * 2, y: 2.1, w: sw, big: "+0.0148", cap: "same-patient minus different-patient\ncosine — sanity check passes", color: GOOD });
  para(s, "A wrong architecture would almost certainly either fail to load or give a different parameter count. I also load strictly and raise on any missing parameter, because a silent partial load would leave layers randomly initialised and produce embeddings that look entirely plausible and mean nothing. That was the failure mode I was most afraid of, so it seemed worth converting into an immediate exception.",
    { x: M, y: 4.35, w: 11.4, h: 1.5, size: 14 });
  foot(s, "Verification", 14);
  s.addNotes("How do I know the reimplementation is right? My encoder has 21,694,848 parameters. The paper states 21 million. A wrong architecture would almost certainly either fail to load or give a different count. It loads with strict mode on and zero missing keys — I made the loader raise if any parameter is missing, because a silent partial load would leave layers randomly initialised and produce embeddings that look plausible and mean nothing. And the sanity check passes.");
}

// 15 — The data
{
  const s = pres.addSlide();
  kicker(s, "The data");
  title(s, "ICBHI, and one property that shaped everything");
  const cw = (W - 0.45) / 2;
  table(s, [
    ["Recordings", "920"],
    ["Patients", "126"],
    ["Duration", "5.49 h"],
    ["COPD patients", "64 / 126  ·  51%"],
    ["COPD recordings", "793 / 920  ·  86%"],
  ], { x: M, y: 1.95, w: cw, colW: [3.0, 2.85], headRow: false, hiRow: 4 });
  para(s, "COPD patients contribute ~12 recordings each; everyone else about two. So accuracy is meaningless — a constant predictor scores 86% — and a recording-wise split almost guarantees the same patient on both sides.",
    { x: M, y: 4.35, w: cw, h: 1.2, size: 13 });
  card(s, { x: M + cw + 0.45, y: 1.9, w: cw, h: 3.85, tone: "warn" });
  cardHead(s, "The thing I nearly missed", { x: M + cw + 0.73, y: 2.12, w: cw - 0.56, color: WARN });
  table(s, [
    ["Device", "Native rate"],
    ["AKGC417L", "100% @ 44.1 kHz"],
    ["LittC2SE", "100% @ 44.1 kHz"],
    ["Litt3200", "100% @ 4 kHz"],
    ["Meditron", "mixed"],
  ], { x: M + cw + 0.73, y: 2.55, w: cw - 0.56, colW: [2.5, 3.0], hiRow: 3 });
  para(s, "Resample 4 kHz to 16 kHz and it has zero energy above 2 kHz. Litt3200 is identifiable from that alone — nothing to do with the representation.",
    { x: M + cw + 0.73, y: 4.85, w: cw - 0.56, h: 0.75, size: 12.5, color: INK });
  foot(s, "The data", 15);
  s.addNotes("Two things about ICBHI shaped everything. First, about half the patients have COPD but 86 percent of the recordings do — COPD patients contribute roughly twelve recordings each. So accuracy is meaningless and a random split almost guarantees the same patient on both sides. Second, and I nearly missed this: I checked native sample rate by device, and the Litt3200 recordings are all at four kilohertz. Resample that to sixteen and it has zero energy above two kilohertz. It's identifiable from that alone.");
}

// 16 — Experiment A
{
  const s = pres.addSlide();
  kicker(s, "Experiment A");
  title(s, "COPD detection — and why this is not a reproduction");
  table(s, [
    ["Protocol", "AUROC", "Balanced accuracy"],
    ["Subject-wise, 8 resampled splits", "0.874 ± 0.024", "0.763 ± 0.034"],
    ["Random split (patient-leaky)", "0.940 ± 0.018", "0.822 ± 0.035"],
    ["Shuffled-label control", "0.492", "0.533"],
  ], { x: M, y: 1.95, w: 11.4, colW: [5.4, 3.0, 3.0], hiRow: 1 });
  card(s, { x: M, y: 3.95, w: 11.4, h: 2.15, tone: "warn" });
  cardHead(s, "I want to be very clear about this", { x: M + 0.32, y: 4.2, w: 10.75, color: WARN });
  para(s, "I have not compared this to the published T7 number, and I have deliberately not quoted one. My clip selection, pooling and split construction all differ. Until I have read the benchmark definition line by line, my number and yours are not measuring the same thing — calling this a reproduction would be overclaiming. It was my mistake not to read that definition first; the two goals were never in conflict.",
    { x: M + 0.32, y: 4.62, w: 10.75, h: 1.3, size: 13.5, color: INK });
  foot(s, "Experiment A", 16);
  s.addNotes("With subject-wise splits the probe gets AUROC 0.874. With a leaky random split, 0.940. The shuffled control sits at 0.492, which tells me the harness isn't leaking. Now — I want to be very clear. I have not compared this to your published T7 number, and I've deliberately not quoted one. My clip selection, pooling and split construction all differ from yours. Calling this a reproduction would be overclaiming. I should have read your T7 definition first so I could have matched it — that was my mistake, and the two goals were never actually in conflict.");
}

// 17 — C1 question
{
  const s = pres.addSlide();
  kicker(s, "Experiment C1 · the question");
  title(s, "What else, besides disease, do the embeddings carry?");
  card(s, { x: M, y: 1.95, w: 11.4, h: 1.35, tone: "accent" });
  para(s, "In a contrastive objective built from two crops of the same recording, the device, the subject and the session are all constant within that recording — so they are perfectly predictive of a positive pair.",
    { x: M + 0.35, y: 2.2, w: 10.7, h: 0.9, size: 14.5, color: INK, bold: true });
  para(s, "That observation prompted the experiment. A representation encoding nothing but \"which recording is this\" would achieve zero contrastive loss. So I pointed the same linear probe at variables that should not matter clinically — device, chest location, patient identity.",
    { x: M, y: 3.55, w: 11.4, h: 1.0, size: 14 });
  card(s, { x: M, y: 4.7, w: 11.4, h: 1.5 });
  para(s, "ICBHI was recorded with four different stethoscopes, which is the only reason this was possible on public data at all. And because sample rate is confounded with device, I ran it twice — naively, and on a bandwidth-matched subset. The gap between those two is the actual measurement.",
    { x: M + 0.32, y: 4.95, w: 10.75, h: 1.05, size: 13.5, color: INK });
  foot(s, "Experiment C1", 17);
  s.addNotes("If your positive pairs are two crops of the same recording, then device, subject and session are constant within that recording — perfectly predictive of a positive pair. A representation encoding nothing except which recording this is would achieve zero contrastive loss. That made me curious what else the embeddings carry. Because sample rate is confounded with device, I had to run it twice — naively, and on a bandwidth-matched subset. The gap between those two is the actual measurement.");
}

// 18 — C1 result
{
  const s = pres.addSlide();
  kicker(s, "Experiment C1 · the result");
  title(s, "Device appears to be encoded about twice as\nstrongly as the disease", 0.76, 27);
  table(s, [
    ["Linear probe target", "Classes", "Chance", "Balanced acc.", "Lift of headroom"],
    ["Device — naive", "4", "0.250", "0.861 ± 0.045", "0.815"],
    ["Device — bandwidth-matched", "3", "0.333", "0.810 ± 0.033", "0.715"],
    ["COPD — same subset", "2", "0.500", "0.759 ± 0.051", "0.518"],
    ["Chest location", "7", "0.143", "0.341 ± 0.018", "0.231"],
    ["Device — shuffled control", "4", "0.250", "0.264", "0.019"],
  ], { x: M, y: 2.35, w: 11.4, colW: [4.0, 1.4, 1.5, 2.5, 2.0], hiRow: 2 });
  card(s, { x: M, y: 5.25, w: 11.4, h: 1.3, tone: "good" });
  para(s, "I expected the bandwidth control to explain most of this. It did not — the signal barely moved. That was genuinely surprising to me, and it is why I think the effect is worth mentioning rather than something I should quietly drop.",
    { x: M + 0.32, y: 5.5, w: 10.75, h: 0.9, size: 13.5, color: INK });
  foot(s, "Experiment C1", 18);
  s.addNotes("The naive device probe gets 86 percent balanced accuracy on a four-class problem where chance is 25. Then the control — dropping every low-sample-rate file leaves three devices sharing a bandwidth, and device is still recoverable at 81 percent against a chance of 33. Normalising for the different chance levels, device reaches about 71 percent of the available headroom, COPD about 52. So device appears roughly twice as decodable as the disease. I expected the bandwidth control to explain most of that. It didn't. That was genuinely surprising to me.");
}

// 19 — C1 caveats
{
  const s = pres.addSlide();
  kicker(s, "Experiment C1 · the caveats");
  title(s, "What this does not show");
  const cw = (W - 0.4) / 2;
  card(s, { x: M, y: 2.0, w: cw, h: 2.5, tone: "warn" });
  cardHead(s, "It is not evidence of shortcut learning", { x: M + 0.28, y: 2.25, w: cw - 0.56, color: WARN });
  para(s, "Linear decodability is a necessary condition, not a sufficient one. Showing device information is present is not showing the COPD classifier uses it. Establishing that would need a cross-device generalisation test — train on some devices, hold others out — which I have not run.",
    { x: M + 0.28, y: 2.72, w: cw - 0.56, h: 1.6, size: 12.5, color: INK });
  card(s, { x: M + cw + 0.4, y: 2.0, w: cw, h: 2.5, tone: "warn" });
  cardHead(s, "Device is confounded with site", { x: M + cw + 0.68, y: 2.25, w: cw - 0.56, color: WARN });
  para(s, "In ICBHI, different clinical sites used different stethoscopes. So some of what my probe reads may be site or patient population rather than microphone. I cannot separate those with this dataset, and that limits how far the observation generalises.",
    { x: M + cw + 0.68, y: 2.72, w: cw - 0.56, h: 1.6, size: 12.5, color: INK });
  para(s, "I am raising this as a question rather than a finding. It seemed relevant given that the product has to work across many different earbuds — but you will know far better than I do whether it is a real concern or something you have already accounted for.",
    { x: M, y: 4.85, w: 11.4, h: 1.1, size: 14 });
  foot(s, "Experiment C1", 19);
  s.addNotes("I want to be careful about what that does and doesn't show. It is not evidence of shortcut learning. Linear decodability is necessary but not sufficient. Showing device information is present is not showing the classifier uses it. And there's a confound I can't remove — different clinical sites used different stethoscopes, so some of what my probe reads might be site rather than microphone. I'm raising this as a question rather than a finding. You'll know far better than I do whether it's a real concern.");
}

// 20 — C2
{
  const s = pres.addSlide();
  kicker(s, "Experiment C2");
  title(s, "What does the encoder cost to run?");
  const sw = (W - 0.5) / 3;
  stat(s, { x: M, y: 1.95, w: sw, big: "125.3 ms", cap: "median latency\nbatch 1, CPU, 200 iterations" });
  stat(s, { x: M + sw + 0.25, y: 1.95, w: sw, big: "82.8 MB", cap: "fp32 on disk\n21.7M parameters" });
  stat(s, { x: M + (sw + 0.25) * 2, y: 1.95, w: sw, big: "11.3 ms", cap: "inter-quartile range\nmedian and IQR, not mean and SD" });
  para(s, "I measured this because I could not find it reported, and for anything running continuously on a wearable it seemed like the number that decides what is possible. Batch size 1 because a wearable classifies one clip at a time; warm-up discarded; host machine recorded, since a latency figure without its hardware is not really a result.",
    { x: M, y: 4.1, w: 11.4, h: 1.15, size: 13.5 });
  card(s, { x: M, y: 5.35, w: 11.4, h: 1.25, tone: "warn" });
  para(s, "Two honest limitations. This is Apple Silicon, not embedded hardware — an upper bound on a fast CPU, not a wearable estimate. And INT8 quantisation failed on my machine, so I have no quantised number. I recorded the failure in the output rather than omitting it.",
    { x: M + 0.32, y: 5.58, w: 10.75, h: 0.85, size: 13, color: INK });
  foot(s, "Experiment C2", 20);
  s.addNotes("125 milliseconds median at batch size one on CPU, 83 megabytes in fp32. I used batch size one because a wearable classifies one clip at a time, discarded warm-up iterations, and report median and IQR rather than mean and standard deviation because latency distributions are right-skewed. Two limitations — this is Apple Silicon, not embedded hardware. And INT8 quantisation actually failed on my machine, so I have no quantised number. I recorded that rather than leaving it out.");
}

// 21 — C3
{
  const s = pres.addSlide();
  kicker(s, "Experiment C3");
  title(s, "How does it compare to a small model\ntrained directly on the task?", 0.76, 27);
  table(s, [
    ["Model", "Parameters", "AUROC, subject-wise"],
    ["OPERA-GT + linear probe", "21,694,848", "0.874 ± 0.024"],
    ["A small CNN, trained from scratch", "60,530", "0.969 best · 0.951 mean of last 5"],
  ], { x: M, y: 2.4, w: 11.4, colW: [4.8, 2.8, 3.8], hiRow: 2 });
  card(s, { x: M, y: 4.0, w: 11.4, h: 2.35, tone: "warn" });
  cardHead(s, "I do not think this means the foundation model does not work", { x: M + 0.32, y: 4.24, w: 10.75, color: WARN });
  bullets(s, [
    "Not like-for-like — the probe is linear on frozen features by protocol; the CNN trains end to end. Fine-tuning the encoder would be the fair fight, and I have not run it.",
    "The CNN figure is a single split with best-epoch selection, which is optimistic. The probe figure is a mean over eight resampled splits.",
    "The CNN may be exploiting the same device structure my earlier probe found — possibly more effectively than the frozen features do.",
  ], { x: M + 0.32, y: 4.68, w: 10.75, h: 1.5, size: 12.5, color: INK });
  foot(s, "Experiment C3", 21);
  s.addNotes("I trained a small CNN from scratch — sixty thousand parameters against the encoder's twenty-one point seven million. It reached 0.969 against the probe's 0.874. I don't think that should be read as the foundation model not working. It isn't like-for-like — the probe is linear on frozen features by protocol, the CNN trains end to end. The CNN figure is a single split with best-epoch selection. And the CNN may be exploiting the same device structure my earlier probe found. The most I'd claim is that on this task at this scale, under linear probing, the pretraining didn't pay for itself.");
}

// 22 — The mistake
{
  const s = pres.addSlide();
  kicker(s, "A mistake I made");
  title(s, "My first run reported zero variance on every probe");
  para(s, "It looked like remarkable stability. It was nothing of the sort — logistic regression fitted on fixed data is deterministic, so varying the classifier's random seed varies nothing at all. My error bars were describing solver noise, which is zero.",
    { x: M, y: 1.9, w: 11.4, h: 1.0, size: 14.5 });
  card(s, { x: M, y: 3.0, w: 11.4, h: 1.35, tone: "warn" });
  para(s, "The uncomfortable part is that I had written a note to myself, a few days earlier, observing that reporting variance across probe seeds rather than across data splits would understate uncertainty — and then implemented exactly that. I only caught it because the zeros were conspicuous.",
    { x: M + 0.32, y: 3.24, w: 10.75, h: 0.95, size: 13.5, color: INK });
  table(s, [
    ["Patient-leakage inflation", "One split", "Eight resampled splits"],
    ["AUROC difference", "+0.004", "+0.066"],
  ], { x: M, y: 4.55, w: 11.4, colW: [5.0, 3.2, 3.2], hiRow: 1 });
  para(s, "A sixteen-fold difference. The single split had landed somewhere the leakage effect was invisible. I am including this because it changed my conclusions more than any of my results did.",
    { x: M, y: 5.85, w: 11.4, h: 0.8, size: 13.5 });
  foot(s, "What went wrong", 22);
  s.addNotes("My first full run reported zero variance. Not small — exactly zero, on every probe. It looked like stability. It was nothing of the sort. Logistic regression on fixed data is deterministic, so varying the classifier's random seed varies nothing. The uncomfortable part is that I'd written a note to myself a few days earlier observing exactly this flaw, and then implemented it. When I fixed it, the estimate of how much patient leakage inflates the result went from plus 0.004 to plus 0.066. A sixteen-fold difference. If I'd reported from one split, I'd have concluded patient leakage doesn't matter here. It does.");
}

// 23 — Limitations
{
  const s = pres.addSlide();
  kicker(s, "Honest limitations");
  title(s, "What this work does not support");
  const cw = (W - 0.5) / 2;
  bullets(s, [
    "One encoder of three — and the one my own reasoning said was least likely to show the device effect.",
    "One task of nineteen.",
    "Not benchmarked against your numbers. The task definition is unmatched.",
    "Input orientation unverified — 128 × 128 is also consistent with the checkpoint.",
  ], { x: M, y: 2.0, w: cw, h: 2.6, size: 13.5 });
  bullets(s, [
    "Head count inferred from ViT-Small convention, not read from anywhere.",
    "One 8.2 s clip per recording, centred. Recordings run to 86 s, so most audio is discarded.",
    "No fine-tuning comparison.",
    "Latency measured on a laptop, not on anything resembling a wearable.",
  ], { x: M + cw + 0.5, y: 2.0, w: cw, h: 2.6, size: 13.5 });
  card(s, { x: M, y: 4.85, w: 11.4, h: 1.5 });
  para(s, "The honest label for all of this: a verified reimplementation of the OPERA-GT encoder, and an independent evaluation of it on one dataset — not a replication of your benchmark. I have tried to say that everywhere it appears, including in the code.",
    { x: M + 0.32, y: 5.12, w: 10.75, h: 1.05, size: 13.5, color: INK });
  foot(s, "Limitations", 23);
  s.addNotes("Let me put the limitations in one place, because I'd rather state them than have them found. One encoder of three — and the one least likely to show the device effect. One task of nineteen. Not benchmarked against your numbers. Input orientation unverified. Head count inferred. One eight-second clip per recording when recordings run to eighty-six seconds. No fine-tuning comparison. Latency measured on a laptop. The honest label is: a verified reimplementation of the encoder and an independent evaluation of it — not a replication of your benchmark.");
}

// 24 — Next steps
{
  const s = pres.addSlide();
  kicker(s, "Where I would go next");
  title(s, "If I were carrying this further");
  numbered(s, [
    "Match your T7 definition and re-run. A few hours, since the features are cached — and the prerequisite for any comparison at all.",
    "Cross-device generalisation. Train on some stethoscopes, test on held-out ones. This turns \"device is decodable\" into \"device causes failure\" — the claim that would matter.",
    "Probe OPERA-CT. My reasoning predicts the contrastive model encodes device more strongly than the generative one. GT already scores high.",
    "Profile OPERA-CE. At ~4M parameters, if it is much faster at comparable accuracy the practical recommendation might invert.",
    "Give the CNN baseline a fair fight — same eight splits, validation-based early stopping.",
  ], { x: M, y: 2.05, w: 11.4, gap: 0.84, size: 13.5 });
  foot(s, "Next steps", 24);
  s.addNotes("If I were carrying this further, in order: match your T7 definition and re-run — that's the prerequisite for any comparison, and it's a few hours because the features are cached. Then the cross-device generalisation test, which is what would turn device is decodable into device causes failure. Then probe OPERA-CT, because my reasoning predicts the contrastive model should encode device more strongly. Then profile CE. And give the CNN baseline a fair fight.");
}

// 25 — Close
{
  const s = darkSlide();
  s.addText("THANK YOU", { x: M, y: 1.6, w: W, h: 0.32, fontFace: BODY, fontSize: 12,
    bold: true, color: SEA, charSpacing: 1.8, margin: 0 });
  s.addText("Thank you for making the weights\nand the code public", {
    x: M, y: 2.05, w: 10.8, h: 1.5, fontFace: HEAD, fontSize: 33, bold: true,
    color: WHITE, margin: 0, lineSpacingMultiple: 1.08,
  });
  s.addText("None of this would have been possible otherwise. Being able to take a NeurIPS model, rebuild its encoder, and test my own understanding against it in a week is a direct consequence of that decision, and I am grateful for it.",
    { x: M, y: 3.6, w: 10.4, h: 0.95, fontFace: BODY, fontSize: 14.5, color: "B7CDD3",
      margin: 0, lineSpacingMultiple: 1.2 });
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 4.75, w: 11.4, h: 1.35, rectRadius: 0.09,
    fill: { color: "12414D" }, line: { color: "1B5A69", width: 1 },
  });
  s.addText("I am very aware that I am a week into a field you have spent a decade in, and that some of what I have said here is probably naive or already well understood. I would genuinely welcome being told where I have gone wrong — I would learn more from that than from anything in these slides.",
    { x: M + 0.34, y: 4.97, w: 10.7, h: 0.95, fontFace: BODY, fontSize: 13.5,
      color: "DCEAED", margin: 0, lineSpacingMultiple: 1.16 });
  s.addText("Rohan Sakeri  ·  code, documentation and full results available on request", {
    x: M, y: 6.35, w: W, h: 0.3, fontFace: BODY, fontSize: 11.5, color: "6F8B94", margin: 0 });
  s.addNotes("Last slide, and I mostly want to say thank you. None of this would have been possible if you hadn't released the weights and the code. Being able to take a NeurIPS model, rebuild its encoder and test my own understanding against it inside a week is a direct consequence of that decision, and I'm genuinely grateful for it. I'm very aware that I'm a week into something you've spent a decade on, and that some of what I've said is probably naive or already well understood. I'd welcome being told where. Thank you very much for watching.");
}

const OUT = "/Users/rohansakeri/my-portfolio/auryx-respiratory-ml/presentation/"
  + (NO_TABLES ? "OPERA-GT-study-Rohan-Sakeri-CANVA.pptx" : "OPERA-GT-study-Rohan-Sakeri.pptx");
pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
