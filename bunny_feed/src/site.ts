// A small public page, served from the edge worker at `/`, reporting what this
// feed's filter measurably does to the posts it shows.
//
// Everything here is static: the numbers are transcribed from the analysis
// reports in the repo root (REPORT_emotion_vs_loc.json / .md and
// REPORT_perspective_threshold_audit.md / REPORT_perspective_de_validity.json)
// rather than computed live. That is deliberate — the page must not touch the
// database or an upstream API, so it stays a cacheable constant that cannot add
// latency or cost to a feed request, and the PullZone in front of this worker is
// free to cache it hard.
//
// When a retrain moves the vector, these numbers go stale together with the
// threshold in feeds.ts. Re-run score_emotions_loc.py + analyze_emotion_shift.py
// and update DATA below in the same change.

// --- Measurements ----------------------------------------------------------

const STUDY = {
  corpus: 5000,          // teacher-annotated posts the deployed vector trained on
  pool: 4367,            // of those, passing the minWords 6 length gate
  admitted: 1858,        // of the pool, also scoring >= 0.88
  keepRate: 0.425,
  threshold: 0.88,
  emotionModel: 'tabularisai/multilingual-emotion-classification',
  updated: '2026-09-10',
};

type EmotionRow = {
  name: string; pool: number; admitted: number; lift: number; rho: number;
};

// prevalence at the model's tau = 0.5, in percent of posts
const EMOTIONS: EmotionRow[] = [
  { name: 'gratitude', pool: 1.05, admitted: 2.10, lift: 1.99, rho: +0.332 },
  { name: 'joy', pool: 2.45, admitted: 4.52, lift: 1.85, rho: +0.290 },
  { name: 'love', pool: 2.89, admitted: 4.95, lift: 1.72, rho: +0.292 },
  { name: 'frustration', pool: 1.03, admitted: 1.29, lift: 1.25, rho: -0.098 },
  { name: 'neutral', pool: 54.20, admitted: 59.15, lift: 1.09, rho: +0.088 },
  { name: 'surprise', pool: 4.90, admitted: 4.04, lift: 0.82, rho: -0.111 },
  { name: 'contempt', pool: 6.73, admitted: 5.33, lift: 0.79, rho: -0.096 },
  { name: 'sadness', pool: 3.50, admitted: 2.37, lift: 0.68, rho: -0.082 },
  { name: 'anger', pool: 9.75, admitted: 6.51, lift: 0.67, rho: -0.194 },
  { name: 'disgust', pool: 3.62, admitted: 1.99, lift: 0.55, rho: -0.263 },
  { name: 'fear', pool: 6.07, admitted: 3.18, lift: 0.52, rho: -0.245 },
];

// frustration is the one row that does not clear p < 0.05
const NOT_SIGNIFICANT = new Set(['frustration']);

type DecileRow = { d: number; anger: number; fear: number; joy: number; love: number };

const DECILES: DecileRow[] = [
  { d: 1, anger: 11.0, fear: 12.4, joy: 0.7, love: 0.7 },
  { d: 2, anger: 12.1, fear: 10.3, joy: 0.5, love: 0.7 },
  { d: 3, anger: 14.9, fear: 8.2, joy: 0.2, love: 1.6 },
  { d: 4, anger: 12.8, fear: 8.2, joy: 1.6, love: 1.6 },
  { d: 5, anger: 11.4, fear: 4.6, joy: 1.1, love: 1.6 },
  { d: 6, anger: 10.3, fear: 4.8, joy: 1.8, love: 1.8 },
  { d: 7, anger: 10.1, fear: 6.4, joy: 1.1, love: 2.5 },
  { d: 8, anger: 8.5, fear: 2.1, joy: 3.2, love: 1.8 },
  { d: 9, anger: 4.4, fear: 2.8, joy: 4.6, love: 5.7 },
  { d: 10, anger: 2.1, fear: 0.9, joy: 9.6, love: 10.8 },
];

// Perspective x LoC audit, 17,581 unique texts (REPORT_perspective_threshold_audit.md)
type CutRow = { cut: string; keep: number; tox: number; toxHot: number; spamLift: number | null };
const CUTOFFS: CutRow[] = [
  { cut: '0.55', keep: 100.0, tox: 0.1032, toxHot: 3.81, spamLift: 1.00 },
  { cut: '0.80', keep: 89.3, tox: 0.0981, toxHot: 3.60, spamLift: null },
  { cut: '0.85', keep: 69.3, tox: 0.0823, toxHot: 2.49, spamLift: null },
  { cut: '0.88', keep: 47.1, tox: 0.0687, toxHot: 1.44, spamLift: 1.17 },
  { cut: '0.90', keep: 29.2, tox: 0.0607, toxHot: 0.99, spamLift: 1.30 },
  { cut: '0.92', keep: 14.2, tox: 0.0563, toxHot: 0.92, spamLift: 1.54 },
  { cut: '0.95', keep: 1.2, tox: 0.0415, toxHot: 0.00, spamLift: 1.80 },
];

// Bridging attributes vs the LoC score, at the 0.88 gate
const BRIDGING = [
  { name: 'RESPECT', lift: 1.17, rho: +0.492 },
  { name: 'AFFINITY', lift: 1.19, rho: +0.343 },
  { name: 'PERSONAL_STORY', lift: 1.15, rho: +0.308 },
  { name: 'COMPASSION', lift: 1.08, rho: +0.151 },
  { name: 'REASONING', lift: 1.10, rho: +0.120 },
  { name: 'CURIOSITY', lift: 1.05, rho: +0.101 },
  { name: 'NUANCE', lift: 1.06, rho: +0.090 },
];

// German validity screen, 800 German + 800 English posts
// (REPORT_perspective_de_validity.json)
const VALIDITY = {
  nDe: 800,
  nEn: 800,
  keep: ['TOXICITY', 'INSULT', 'PROFANITY', 'FLIRTATION', 'OBSCENE', 'SPAM',
    'UNSUBSTANTIAL', 'AFFINITY', 'COMPASSION', 'CURIOSITY', 'PERSONAL_STORY',
    'REASONING', 'RESPECT'],
  lowPower: ['SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'THREAT', 'SEXUALLY_EXPLICIT',
    'INFLAMMATORY', 'ATTACK_ON_AUTHOR', 'ATTACK_ON_COMMENTER', 'NUANCE'],
  drop: ['INCOHERENT', 'LIKELY_TO_REJECT'],
};

// --- Tiny SVG chart helpers ------------------------------------------------
//
// Hand-rolled rather than pulled from a library: the whole page has to bundle
// into the same edge script as the feed, and these are three fixed charts.

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function fmt(n: number, digits = 2): string {
  return n.toFixed(digits);
}

/**
 * Diverging bars for prevalence ratios, centred on 1.0 (no change).
 * Widths are log2(lift), so "half as common" and "twice as common" are mirror
 * images instead of the ratio scale's squashed left half.
 */
function liftChart(rows: EmotionRow[]): string {
  const W = 720, rowH = 30, padTop = 26, padL = 108, padR = 78;
  const H = padTop + rows.length * rowH + 8;
  const mid = padL + (W - padL - padR) / 2;
  const half = (W - padL - padR) / 2;
  const span = 1.05; // log2 units either side of centre

  const parts: string[] = [];
  for (const t of [0.5, 0.7, 1, 1.4, 2]) {
    const x = mid + (Math.log2(t) / span) * half;
    parts.push(
      `<line class="grid" x1="${fmt(x, 1)}" y1="${padTop - 8}" x2="${fmt(x, 1)}" y2="${H - 8}"/>`,
      `<text class="tick" x="${fmt(x, 1)}" y="${padTop - 13}" text-anchor="middle">${t}x</text>`,
    );
  }
  rows.forEach((r, i) => {
    const y = padTop + i * rowH;
    const w = (Math.log2(r.lift) / span) * half;
    const x = w >= 0 ? mid : mid + w;
    const cls = r.lift >= 1 ? 'up' : 'down';
    const faded = NOT_SIGNIFICANT.has(r.name) ? ' faded' : '';
    parts.push(
      `<text class="lbl" x="${padL - 12}" y="${y + 15}" text-anchor="end">${esc(r.name)}</text>`,
      `<rect class="bar ${cls}${faded}" x="${fmt(x, 1)}" y="${y + 5}" width="${fmt(Math.abs(w), 1)}" height="15" rx="2"/>`,
      `<text class="val" x="${W - padR + 10}" y="${y + 16}">${fmt(r.lift)}x</text>`,
    );
  });
  parts.push(`<line class="axis" x1="${fmt(mid, 1)}" y1="${padTop - 8}" x2="${fmt(mid, 1)}" y2="${H - 8}"/>`);

  return `<figure class="chart"><svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Prevalence ratio of each emotion after the filter versus before"
    preserveAspectRatio="xMidYMid meet">
    <title>How often each emotion appears after filtering, relative to before</title>
    ${parts.join('\n    ')}
  </svg>
  <figcaption>Left of the line, the filter shows you less of it; right, more.
  Grey bar = not statistically distinguishable from no change.</figcaption>
  </figure>`;
}

/** Four emotion series across LoC-score deciles. */
function decileChart(rows: DecileRow[]): string {
  const W = 720, H = 300, padL = 46, padR = 96, padT = 18, padB = 40;
  const maxY = 16;
  const x = (i: number) => padL + (i / (rows.length - 1)) * (W - padL - padR);
  const y = (v: number) => H - padB - (v / maxY) * (H - padT - padB);

  const series: { key: keyof DecileRow; cls: string }[] = [
    { key: 'fear', cls: 's-fear' }, { key: 'anger', cls: 's-anger' },
    { key: 'love', cls: 's-love' }, { key: 'joy', cls: 's-joy' },
  ];

  const parts: string[] = [];
  for (let v = 0; v <= maxY; v += 4) {
    parts.push(
      `<line class="grid" x1="${padL}" y1="${fmt(y(v), 1)}" x2="${W - padR}" y2="${fmt(y(v), 1)}"/>`,
      `<text class="tick" x="${padL - 8}" y="${fmt(y(v) + 4, 1)}" text-anchor="end">${v}%</text>`,
    );
  }
  for (const s of series) {
    const d = rows.map((r, i) => `${i ? 'L' : 'M'}${fmt(x(i), 1)} ${fmt(y(r[s.key] as number), 1)}`).join(' ');
    parts.push(`<path class="line ${s.cls}" d="${d}"/>`);
    rows.forEach((r, i) =>
      parts.push(`<circle class="dot ${s.cls}" cx="${fmt(x(i), 1)}" cy="${fmt(y(r[s.key] as number), 1)}" r="3"/>`));
    const last = rows[rows.length - 1][s.key] as number;
    parts.push(`<text class="lgd ${s.cls}" x="${W - padR + 10}" y="${fmt(y(last) + 4, 1)}">${s.key}</text>`);
  }
  parts.push(
    `<line class="axis" x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}"/>`,
    `<text class="tick" x="${padL}" y="${H - padB + 20}">lowest scores</text>`,
    `<text class="tick" x="${W - padR}" y="${H - padB + 20}" text-anchor="end">highest</text>`,
    `<text class="tick" x="${(padL + W - padR) / 2}" y="${H - padB + 34}" text-anchor="middle">posts ranked by locus-of-control score, in tenths</text>`,
  );

  return `<figure class="chart"><svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Emotion prevalence across deciles of the locus-of-control score"
    preserveAspectRatio="xMidYMid meet">
    <title>Emotion prevalence rises and falls smoothly across the score range</title>
    ${parts.join('\n    ')}
  </svg>
  <figcaption>The threshold could have been placed anywhere along this axis; the
  effect is a slope, not a cliff.</figcaption></figure>`;
}

/** Toxicity falling and spam rising as the gate tightens. */
function cutoffChart(rows: CutRow[]): string {
  const W = 720, H = 260, padL = 52, padR = 58, padT = 22, padB = 58;
  const pts = rows.map((r, i) => ({ ...r, i }));
  const x = (i: number) => padL + (i / (rows.length - 1)) * (W - padL - padR);
  const yT = (v: number) => H - padB - (v / 4.0) * (H - padT - padB);      // tox >= 0.5, %
  const yS = (v: number) => H - padB - ((v - 0.9) / 1.1) * (H - padT - padB); // spam lift

  const parts: string[] = [];
  for (let v = 0; v <= 4; v += 1) {
    parts.push(
      `<line class="grid" x1="${padL}" y1="${fmt(yT(v), 1)}" x2="${W - padR}" y2="${fmt(yT(v), 1)}"/>`,
      `<text class="tick s-anger" x="${padL - 8}" y="${fmt(yT(v) + 4, 1)}" text-anchor="end">${v}%</text>`,
    );
  }
  for (const v of [1.0, 1.5, 2.0]) {
    parts.push(`<text class="tick s-spam" x="${W - padR + 8}" y="${fmt(yS(v) + 4, 1)}">${fmt(v, 1)}x</text>`);
  }
  const tox = pts.map(p => `${p.i ? 'L' : 'M'}${fmt(x(p.i), 1)} ${fmt(yT(p.toxHot), 1)}`).join(' ');
  parts.push(`<path class="line s-anger" d="${tox}"/>`);
  pts.forEach(p => parts.push(`<circle class="dot s-anger" cx="${fmt(x(p.i), 1)}" cy="${fmt(yT(p.toxHot), 1)}" r="3"/>`));

  const spam = pts.filter(p => p.spamLift !== null);
  const sd = spam.map((p, k) => `${k ? 'L' : 'M'}${fmt(x(p.i), 1)} ${fmt(yS(p.spamLift as number), 1)}`).join(' ');
  parts.push(`<path class="line s-spam dashed" d="${sd}"/>`);
  spam.forEach(p => parts.push(`<circle class="dot s-spam" cx="${fmt(x(p.i), 1)}" cy="${fmt(yS(p.spamLift as number), 1)}" r="3"/>`));

  pts.forEach(p => parts.push(
    `<text class="tick" x="${fmt(x(p.i), 1)}" y="${H - padB + 18}" text-anchor="middle">${p.cut}</text>`,
    `<text class="tick dim" x="${fmt(x(p.i), 1)}" y="${H - padB + 32}" text-anchor="middle">${fmt(p.keep, 0)}%</text>`,
  ));
  const live = pts[3];
  parts.push(
    `<line class="marker" x1="${fmt(x(live.i), 1)}" y1="${padT}" x2="${fmt(x(live.i), 1)}" y2="${H - padB}"/>`,
    `<text class="tick strong" x="${fmt(x(live.i), 1)}" y="${padT - 6}" text-anchor="middle">live setting</text>`,
    `<line class="axis" x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}"/>`,
    `<text class="lgd s-anger" x="${fmt(x(1) + 8, 1)}" y="${fmt(yT(pts[1].toxHot) - 8, 1)}">rated toxic</text>`,
    `<text class="lgd s-spam" x="${fmt(x(4) + 6, 1)}" y="${fmt(yS(pts[4].spamLift as number) + 18, 1)}">spam lift</text>`,
    `<text class="tick" x="${(padL + W - padR) / 2}" y="${H - padB + 48}" text-anchor="middle">score gate, and the share of the corpus it keeps</text>`,
  );

  return `<figure class="chart"><svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="As the score threshold rises, toxicity falls and spam rises"
    preserveAspectRatio="xMidYMid meet">
    <title>Tightening the filter trades toxicity down for spam up</title>
    ${parts.join('\n    ')}
  </svg>
  <figcaption>Solid: share of kept posts Perspective rates toxic. Dashed: how much
  more spam-like the kept posts are than the corpus average.</figcaption></figure>`;
}

function chips(names: string[], cls: string): string {
  return `<p class="chips">${names.map(n => `<span class="chip ${cls}">${esc(n)}</span>`).join(' ')}</p>`;
}

// --- Page ------------------------------------------------------------------

const CSS = `
:root{color-scheme:light dark;
--bg:#fbfaf8;--fg:#1a1a19;--dim:#6b6a66;--rule:#e3e0d9;--card:#fff;
--up:#2f7d63;--down:#b4472e;--spam:#8a6d1f;--accent:#2f5d8a;--mark:#c9c4b8}
@media (prefers-color-scheme:dark){:root{
--bg:#14140f;--fg:#eceae4;--dim:#9b988f;--rule:#2c2c26;--card:#1c1c17;
--up:#6fc3a2;--down:#e0866a;--spam:#d0b25c;--accent:#7fb0dd;--mark:#3d3c34}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
-webkit-text-size-adjust:100%}
.wrap{max-width:52rem;margin:0 auto;padding:3rem 1.25rem 5rem}
header{border-bottom:1px solid var(--rule);padding-bottom:1.75rem;margin-bottom:2.5rem}
h1{font-size:clamp(1.7rem,4.5vw,2.4rem);line-height:1.15;margin:0 0 .6rem;letter-spacing:-.02em}
h2{font-size:1.32rem;margin:3.25rem 0 .5rem;letter-spacing:-.01em;line-height:1.25}
h3{font-size:1.02rem;margin:2rem 0 .4rem;letter-spacing:.01em}
p{margin:.85rem 0}
a{color:var(--accent)}
.lede{font-size:1.1rem;color:var(--dim);margin:0}
.kicker{text-transform:uppercase;letter-spacing:.09em;font-size:.72rem;color:var(--dim);margin:0 0 .75rem}
.dim{color:var(--dim)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(8.5rem,1fr));gap:.75rem;margin:1.75rem 0}
.stat{background:var(--card);border:1px solid var(--rule);border-radius:.55rem;padding:.85rem .9rem}
.stat b{display:block;font-size:1.5rem;line-height:1.1;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.stat span{display:block;font-size:.78rem;color:var(--dim);margin-top:.3rem;line-height:1.35}
figure.chart{margin:1.75rem 0;background:var(--card);border:1px solid var(--rule);
border-radius:.55rem;padding:1.1rem .9rem .75rem}
figure.chart svg{width:100%;height:auto;display:block;overflow:visible}
figcaption{font-size:.82rem;color:var(--dim);margin-top:.85rem;line-height:1.5}
.grid{stroke:var(--rule);stroke-width:1}
.axis{stroke:var(--dim);stroke-width:1}
.marker{stroke:var(--mark);stroke-width:1;stroke-dasharray:3 3}
text{font:12px ui-sans-serif,system-ui,sans-serif;fill:var(--fg)}
.tick{font-size:11px;fill:var(--dim)}
.tick.strong{fill:var(--fg);font-weight:600}
.tick.dim{opacity:.7}
.lbl{font-size:12.5px;fill:var(--fg)}
.val{font-size:12px;fill:var(--dim);font-variant-numeric:tabular-nums}
.bar.up{fill:var(--up)}.bar.down{fill:var(--down)}.bar.faded{fill:var(--dim);opacity:.45}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.dashed{stroke-dasharray:5 4}
.lgd{font-size:11.5px;font-weight:600}
/* Series colour is carried by stroke only. The fill:none on .line above must
   not be overridden here, or every line chart fills into a solid blob. Dots and
   legend labels opt back into a fill explicitly. */
.s-fear{stroke:var(--down)}
.s-anger{stroke:var(--down);opacity:.62}
.s-joy{stroke:var(--up)}
.s-love{stroke:var(--up);opacity:.62}
.s-spam{stroke:var(--spam)}
circle.s-fear{fill:var(--down)}
circle.s-anger{fill:var(--down)}
circle.s-joy{fill:var(--up)}
circle.s-love{fill:var(--up)}
circle.s-spam{fill:var(--spam)}
text.s-fear{fill:var(--down);stroke:none}
text.s-anger{fill:var(--down);stroke:none}
text.s-joy{fill:var(--up);stroke:none}
text.s-love{fill:var(--up);stroke:none}
text.s-spam{fill:var(--spam);stroke:none}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:1.5rem 0}
table{border-collapse:collapse;width:100%;font-size:.88rem;min-width:34rem}
th,td{padding:.5rem .7rem;text-align:right;border-bottom:1px solid var(--rule);
font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
thead th{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);font-weight:600}
tbody tr:last-child td{border-bottom:none}
tr.hi td{background:color-mix(in srgb,var(--accent) 8%,transparent);font-weight:600}
.note{border-left:3px solid var(--mark);padding:.15rem 0 .15rem 1.1rem;margin:1.75rem 0;color:var(--dim)}
.note b{color:var(--fg)}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin:.75rem 0}
.chip{font-size:.72rem;letter-spacing:.03em;padding:.24rem .55rem;border-radius:1rem;
border:1px solid var(--rule);background:var(--card);color:var(--dim);white-space:nowrap}
.chip.keep{border-color:var(--up);color:var(--up)}
.chip.weak{opacity:.75}
.chip.drop{border-color:var(--down);color:var(--down);text-decoration:line-through}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:.5rem;margin:1.75rem 0}
.step{flex:1 1 8rem;background:var(--card);border:1px solid var(--rule);border-radius:.5rem;padding:.7rem .8rem}
.step b{display:block;font-size:.8rem}
.step span{font-size:.75rem;color:var(--dim);line-height:1.4;display:block;margin-top:.2rem}
.step.gate{border-color:var(--accent)}
footer{margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--rule);
font-size:.83rem;color:var(--dim)}
code{font:.86em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--card);
border:1px solid var(--rule);border-radius:.25rem;padding:.05rem .3rem}
`;

export function renderSite(): string {
  const emotionRows = EMOTIONS.map(r => {
    const hi = ['fear', 'anger', 'gratitude', 'joy'].includes(r.name) ? ' class="hi"' : '';
    return `<tr${hi}><td>${esc(r.name)}</td><td>${fmt(r.pool)}%</td>` +
      `<td>${fmt(r.admitted)}%</td><td>${fmt(r.lift)}x</td>` +
      `<td>${r.rho >= 0 ? '+' : '−'}${fmt(Math.abs(r.rho), 3)}</td></tr>`;
  }).join('\n');

  const bridgingRows = BRIDGING.map(b =>
    `<tr><td>${esc(b.name)}</td><td>${fmt(b.lift)}x</td><td>+${fmt(b.rho, 3)}</td></tr>`).join('\n');

  const cutRows = CUTOFFS.map(c =>
    `<tr${c.cut === '0.88' ? ' class="hi"' : ''}><td>${c.cut}</td><td>${fmt(c.keep, 1)}%</td>` +
    `<td>${fmt(c.tox, 4)}</td><td>${fmt(c.toxHot, 2)}%</td>` +
    `<td>${c.spamLift === null ? '—' : fmt(c.spamLift) + 'x'}</td></tr>`).join('\n');

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>What this feed filters — Feed Forward</title>
<meta name="description" content="Measurements of what a locus-of-control ranking vector does to the emotional and civility profile of a Bluesky feed.">
<meta name="robots" content="index,follow">
<meta property="og:title" content="What this feed filters">
<meta property="og:description" content="Measurements of what a locus-of-control ranking vector does to the posts it shows you.">
<meta property="og:type" content="article">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='14'>&#x2197;</text></svg>">
<style>${CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <p class="kicker">Feed Forward &middot; measurement notes</p>
  <h1>What this filter actually does to what you see</h1>
  <p class="lede">This feed ranks Bluesky posts on <strong>locus of control</strong> &mdash;
  roughly, whether a post treats the world as something you can act on or something
  that happens to you. That is an editorial choice, and editorial choices have side
  effects. This page reports the side effects we have measured, including the ones
  that came out wrong.</p>
</header>

<h2>How a post reaches you</h2>
<div class="flow">
  <div class="step"><b>1. Candidates</b><span>A few hundred recent posts arrive from the candidate generator.</span></div>
  <div class="step"><b>2. Length gate</b><span>Anything under six words is dropped &mdash; too short to read a position off.</span></div>
  <div class="step"><b>3. Score</b><span>Each post is embedded and scored against one trained vector.</span></div>
  <div class="step gate"><b>4. Gate</b><span>Below ${STUDY.threshold}, a post is discarded. This is the only thing the vector decides.</span></div>
  <div class="step"><b>5. Order</b><span>Survivors are ordered mostly by their locus-of-control rank, blended with a separate engagement model.</span></div>
</div>
<p>Because step 4 is the vector's whole editorial act, every measurement below
holds the length gate fixed and varies only the threshold: <b>${STUDY.pool.toLocaleString('en')}</b>
posts go in, <b>${STUDY.admitted.toLocaleString('en')}</b> come out.</p>

<h2>1. It makes the feed calmer, and that is not a neutral act</h2>
<p>We ran an independent emotion classifier over the ${STUDY.corpus.toLocaleString('en')}
posts the vector was trained on, then compared the emotional mix of the posts the
gate admits against the pool it drew them from. The classifier never saw the
locus-of-control score, and the score never saw the emotion labels.</p>

<div class="stats">
  <div class="stat"><b>0.52x</b><span>as much fear</span></div>
  <div class="stat"><b>0.67x</b><span>as much anger</span></div>
  <div class="stat"><b>1.85x</b><span>as much joy</span></div>
  <div class="stat"><b>1.99x</b><span>as much gratitude</span></div>
</div>

${liftChart(EMOTIONS)}

<p>Aggregated, posts carrying any negative emotion fall from 28.7% to 19.7%, and
any positive emotion rises from 6.1% to 11.0%. The ratio of negative to positive
goes from <b>4.7&#8239;:&#8239;1 to 1.8&#8239;:&#8239;1</b>. The feed does not become happy.
It becomes about two and a half times less lopsided.</p>

<div class="scroll">
<table>
<thead><tr><th>emotion</th><th>before</th><th>after</th><th>ratio</th><th>correlation</th></tr></thead>
<tbody>
${emotionRows}
</tbody>
</table>
</div>
<p class="dim" style="font-size:.83rem">Share of posts carrying each emotion,
before and after the gate. Every row except <em>frustration</em> is statistically
significant. Correlation is Spearman's rho between the score and the emotion.</p>

<h3>The effect is a slope, not a cliff</h3>
<p>Splitting the pool into tenths by score, fear falls almost monotonically from
12.4% at the bottom to 0.9% at the top &mdash; a fourteenfold swing. Anger holds
steady until the top two tenths and then collapses. Wherever the threshold had
been placed, it would have moved these numbers in the same direction.</p>

${decileChart(DECILES)}

<h3>But it is not simply a mood filter</h3>
<p>The obvious suspicion is that "locus of control" has quietly become "positive
vibes". We tested it: a model given all eleven emotion probabilities can recover
only <b>13%</b> of the variance in the locus-of-control score. The axis is
emotionally tilted, not emotionally defined.</p>
<p>The cleanest evidence sits inside the data. Among posts the labeller marked as
<em>external</em> &mdash; the helpless-framing class this feed is built to filter
out &mdash; the gate barely touches anger at all (18.0% before, 17.8% after). It
is not removing angry posts. It is removing external ones, and only 23% of those
survive. Fear inside that same class <em>does</em> drop, from 11.5% to 7.4%: the
filter is sharper on frightened helplessness than on angry helplessness.</p>

<div class="note">
<p><b>Where it overreaches.</b> Anger is bimodal on this axis. Helpless posts are
the angriest (18.0%), but agency posts are second (13.2%) &mdash; roughly three
times angrier than neutral chatter. Indignant "get up and do something" writing
is real, and it is angry.</p>
<p>The gate does not respect that. Among posts the labeller called agency-bearing,
the angry ones average a score of 0.864 and the calm ones 0.894 &mdash; and the
live threshold sits at 0.88, right between them. So <b>41% of angry agency posts
get through, against 72% of calm ones</b>. That is a preference for tone that the
underlying idea never asked for, and it is an artifact of where the threshold
landed rather than a property of the vector. It is on the list to fix. (34 posts
in that cell, so read the exact figure loosely.)</p>
</div>

<h2>2. The civility screening, and the filter that was not filtering</h2>
<p>Before the emotion work, we rated ${(17581).toLocaleString('en')} posts with
Google's <a href="https://perspectiveapi.com/" rel="noopener">Perspective API</a>
&mdash; an independent classifier for toxicity, insult, spam and related
attributes &mdash; to see what the vector was selecting for. The first finding
was embarrassing: <b>the gate was rejecting nothing at all.</b></p>
<p>A retrain had shifted the score distribution upward and the threshold constant
had not moved with it. The lowest score anywhere in the corpus was 0.595, against
a gate of 0.55. Zero of ${(17581).toLocaleString('en')} posts, and zero of 993 live
candidates, were being filtered. Whatever character the feed had was coming from
the ranking model, not from the vector. The gate now sits at ${STUDY.threshold},
where it keeps about half the corpus.</p>

${cutoffChart(CUTOFFS)}

<div class="scroll">
<table>
<thead><tr><th>gate</th><th>kept</th><th>mean toxicity</th><th>rated toxic</th><th>spam lift</th></tr></thead>
<tbody>
${cutRows}
</tbody>
</table>
</div>

<h3>Tightening the filter trades toxicity for spam</h3>
<p>Toxicity falls steadily as the gate rises &mdash; and so, unexpectedly, does
the feed's resistance to marketing copy. Spam is the <em>strongest single
correlate</em> of the locus-of-control score (+0.219). Promotional writing is
first-person, confident and goal-directed, which is precisely what an agency
vector rewards. Push the gate to 0.95 and 71% of survivors are rated spam; the
top of the range fills with loan offers and jewellery ads scoring above 0.94.
This is the main reason the threshold is not higher.</p>

<h3>What it aligns with best is not civility but bridging</h3>
<p>Perspective also scores seven <em>bridging</em> attributes &mdash; qualities
that make a conversation work rather than qualities that make it polite. The
vector's alignment with these is stronger (+0.369 composite) than its
relationship with either toxicity (&minus;0.266) or spam (+0.219). Of the posts
the gate keeps, 59.6% are above the corpus average on bridging, against 46.1% of
all posts.</p>

<div class="scroll">
<table>
<thead><tr><th>attribute</th><th>lift at the gate</th><th>correlation</th></tr></thead>
<tbody>
${bridgingRows}
</tbody>
</table>
</div>
<p>One attribute resists: <b>curiosity</b>. Selecting on locus of control gets you
a feed that is more respectful and more personal, and no more inquisitive. Some of
the rest of the lift is the spam artifact again &mdash; marketing copy scores well
on respect and reasoning too.</p>

<h2>3. Does any of this work in German?</h2>
<p>Roughly one post in eight here is German, and Perspective's coverage outside
English is thinner than its documentation suggests. We screened every attribute
on ${VALIDITY.nDe} German and ${VALIDITY.nEn} English posts before trusting any of
it.</p>
<p>Two things fell out. First, the <code>languages</code> parameter is close to
decorative: scoring the same German text as German and as English agrees at
r&nbsp;=&nbsp;1.000, and the <code>_EXPERIMENTAL</code> attribute variants are
byte-identical to the production ones. Second, and more usefully, two attributes
<b>score the language rather than the content</b>:</p>

${chips(VALIDITY.drop, 'drop')}

<p>On German text, <code>INCOHERENT</code> rates <b>100%</b> of posts positive
&mdash; 96% of them above 0.9 &mdash; while agreeing with its English behaviour
only 17% of the time. <code>LIKELY_TO_REJECT</code> does the same. Fed German,
these two effectively answer "yes, this is German". Anyone moderating a
multilingual feed on them would be filtering by language while believing they
were filtering by quality. Both are excluded from our work.</p>

<p>Thirteen attributes survived with enough signal to measure something:</p>
${chips(VALIDITY.keep, 'keep')}
<p class="dim">Another eight are valid but fire too rarely on German to support a
conclusion &mdash; threats and identity attacks are genuinely scarce in this
sample, which is good news and bad statistics:</p>
${chips(VALIDITY.lowPower, 'weak')}

<h2>What we are not claiming</h2>
<p><b>These classifiers are proxies.</b> Perspective measures perceived rudeness,
not harm. The emotion model is trained on synthetic multilingual text and has no
published human evaluation; 14% of posts fire no emotion label at all. The
relative shifts above are large and consistent enough to survive a fair amount of
measurement noise, but the absolute percentages should not be quoted.</p>
<p><b>The measured corpus is not the live feed.</b> These posts were sampled from
the firehose for annotation. The live candidate pool is popularity- and
network-weighted and will have a different baseline.</p>
<p><b>Nothing here is adversarial.</b> A threshold chosen on this data will not
hold against text written to game it &mdash; and the spam finding shows the vector
is already gameable by accident.</p>
<p><b>Every number is tied to one vector at one moment.</b> Retraining moves the
score distribution underneath the threshold, which is exactly how the gate came to
be rejecting nothing. These measurements are re-run on each retrain; the date
below says when.</p>

<footer>
<p>Emotion measurements: <code>${esc(STUDY.emotionModel)}</code> at its default
decision threshold, over the ${STUDY.corpus.toLocaleString('en')}-post training
corpus, scored with the deployed vector exactly as the edge worker scores it.
Civility measurements: Perspective API over ${(17581).toLocaleString('en')} unique
texts. Reports and analysis scripts live in the project repository.</p>
<p>Last measured ${STUDY.updated}. <a href="/xrpc/app.bsky.feed.getFeedSkeleton?feed=feed-forward">Feed endpoint</a></p>
</footer>

</div>
</body>
</html>`;
}

/**
 * The page as a cacheable response. It is a pure constant, so the PullZone in
 * front of this worker may cache it as long as it likes — unlike the feed
 * skeleton, which is per-reader and must never be cached at the edge.
 */
export function siteResponse(): Response {
  return new Response(renderSite(), {
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'public, max-age=600, s-maxage=3600',
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer',
    },
  });
}
