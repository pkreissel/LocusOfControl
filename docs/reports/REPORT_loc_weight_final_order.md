# LoC weight in the final order (`locWeight`, 12f16e7)

2026-09-29. Does blending the LoC rank into the final order (deployed at
`locWeight: 0.65`) put better posts at the top of feed-forward, and what does it
cost in engagement?

## Setup

- **Pools.** 10 signed-in GE pools rebuilt offline with the feed's own modules
  (`research/feed_order/dump_pools.ts`): same GE body, minWords, MiniLM
  shortlist, German bypass. Readers: the owner's account plus nine public
  accounts (EN tech/politics/media, two German-leaning). No seen-exclusion, no
  score cache. Every pool fills the 150-post cap.
- **Scores.** Jina on local weights × `vector.json` (`score_pools.py`), then
  gate, `fuseByRank`, cut and the production `blendLocAndEngagement` at
  w ∈ {0, .25, .5, .65, .8, 1} (`order_eval.ts`). Selection is identical across
  weights, so every difference below is ordering alone. w=0 is the order before
  12f16e7.
- **Judge.** decider-2b + the fine-tuned LoRA (`research/data/laya_loc/decider_short`,
  rubric short; gold AP 0.769, AUC 0.945 vs Gemini). Re-checked on 30 gold posts
  before use: Spearman 0.96 against its published scores, 28/30 same class.
  1385 unique posts, 0 errors.
- **Metrics** (`metrics.py`): judge P(internal), internal/external share and
  nDCG (gain = P(internal)) over the top 10 and 25; `ge@k` = mean engagement rank
  of the top k (1 = best). Paired over pools vs w=0, bootstrap over pools.

## Result

Mean over 10 pools. The whole 150-post list is 30.5% internal whatever the weight.

| w | int@10 | ext@10 | int@25 | ext@25 | nDCG@25 | ge@25 |
|---|---|---|---|---|---|---|
| 0 (before) | 0.340 | 0.290 | 0.380 | 0.244 | 0.458 | 0.927 |
| 0.25 | 0.460 | 0.100 | 0.408 | 0.180 | 0.507 | 0.901 |
| 0.5 | 0.510 | 0.090 | 0.440 | 0.120 | 0.546 | 0.850 |
| **0.65 (deployed)** | **0.540** | **0.070** | **0.476** | **0.104** | **0.577** | **0.806** |
| 0.8 | 0.580 | 0.040 | 0.516 | 0.076 | 0.614 | 0.728 |
| 1 | 0.610 | 0.020 | 0.548 | 0.064 | 0.643 | 0.631 |

w=0.65 vs w=0, paired:

    int@25   +0.096 [+0.036, +0.160]   8 pools better / 1 tied / 1 worse
    ext@25   -0.140 [-0.180, -0.104]   10 / 0 / 0
    nDCG@25  +0.119 [+0.093, +0.144]   10 / 0 / 0
    ge@25    -0.121 [-0.136, -0.104]   costs engagement in every pool

Page 1 goes from roughly one external post in four to one in ten, and the internal
share rises by about ten points. The cost curve is smooth up to 0.8, then steep:
w=1 halves the drop in external posts per unit of engagement lost (ge@10 0.58).

## Caveat: the judge is not fully independent of the order

The MiniLM half of `fuseByRank` is the union vector, trained partly on net20k
labelled by this same fine-tuned decider. Ranking the same lists by the Jina
vector alone (teacher: TypeSafe) through the same blend:

| w | int@25 | ext@25 |
|---|---|---|
| 0 | 0.380 | 0.244 |
| 0.65 | 0.388 | 0.112 |
| 1 | 0.428 | 0.080 |

The **external suppression holds** without the shared-label path (0.244 → 0.112).
The **internal gain mostly does not** (+0.008 against +0.096 fused). That is
consistent with either (a) contamination or (b) the known MiniLM strength on the
internal pole (62% internal in its top 50 against Jina's 42%,
REPORT_ge_embeddings.md), and this setup cannot tell them apart. A Gemini- or
TypeSafe-judged sample of the top 25 would.

## Limits

- Ten pools, one day, public accounts rather than real readers. No seen-exclusion,
  so these are first-visit lists.
- `ge@k` is a proxy: the engagement *rank* given up, not measured clicks or likes.
- `germanFirst` not applied (no reader language).

Reproduce: `research/feed_order/{dump_pools.ts, score_pools.py, order_eval.ts, metrics.py}`,
outputs in `research/data/feed_order/`.
