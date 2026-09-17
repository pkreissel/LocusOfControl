# MiniLM vs Jina on German content

Measured 2026-09-11. Script: `eval_minilm_vs_jina_by_lang.py`. Machine-readable
output: `REPORT_minilm_vs_jina_by_lang.json`. All CIs are 4,000-sample percentile
bootstraps; the two spaces are always compared **paired**, on the same rows, with
the same folds, the same target and the same hyperparameters (T = 5.0, λ = 10.0 —
the values the sweep selected for both).

**Short answer: MiniLM's advantage is English-only. On German it is close to a
coin flip, and Jina is the space that holds up.** `REPORT_ge_embeddings.md` found
the two spaces statistically indistinguishable on the val set and MiniLM slightly
ahead out-of-fold. That corpus is 68% English tagged; the aggregate hid the split.

## The best-powered measurement: out-of-fold CV

5-fold OOF Ridge over the full 2822-row teacher corpus (2500 teacher + 322
interaction rows), band target, 385 band-positives. 345 rows carry a declared
`de` tag and 51 of those are band-positive — roughly 5× the German positives the
gold val set has.

| group | n | pos | Jina AUC | MiniLM AUC | Jina − MiniLM | p |
| :--- | ---: | ---: | :--- | :--- | ---: | ---: |
| all | 2822 | 385 | 0.687 [0.659, 0.715] | **0.712** [0.684, 0.739] | −0.025 | 0.11 |
| English | 1414 | 185 | 0.634 [0.593, 0.675] | **0.723** [0.683, 0.762] | −0.089 | **0.00** |
| German | 345 | 51 | **0.674** [0.594, 0.753] | 0.588 [0.492, 0.677] | +0.085 | 0.14 |
| German (incl. detected) | 626 | 108 | **0.674** [0.616, 0.729] | 0.627 [0.564, 0.692] | +0.047 | 0.19 |

Neither per-language gap clears significance on its own, but the contrast between
them does — and that is the quantity in question:

**(Jina − MiniLM on German) − (Jina − MiniLM on English) = +0.174, 95% CI
[+0.057, +0.292], p = 0.004.**

So the two languages are not being served by the same ranker twice. MiniLM gains
~0.09 AUC over Jina on English and gives back ~0.09 on German. Jina, by contrast,
scores German and English within 0.04 of each other.

Three secondary readouts point the same way, on the same rows:

| group | ρ vs graded target | | external-leak AUC (lower is better) | |
| :--- | ---: | ---: | ---: | ---: |
| | Jina | MiniLM | Jina | MiniLM |
| English | 0.302 | **0.423** | 0.318 | **0.258** |
| German | **0.363** | 0.174 | **0.279** | 0.396 |

On German, MiniLM not only ranks internal posts worse — it ranks *external* posts
higher (0.396 vs Jina's 0.279), on an axis whose whole point is to push external
down. That is the same failure mode §3 of `REPORT_ge_embeddings.md` flagged for
MiniLM alone (6× the external leak), concentrated in German.

## It is the space, not the training mix

The obvious hypothesis is that a probe trained on a 68%-English corpus simply
does not transfer. Train on one language, test on the other (`ridge λ=10`, the
shipped head; scored on German rows out-of-fold in every case):

| trained on | Jina, on German | MiniLM, on German |
| :--- | ---: | ---: |
| English rows only | 0.652 | 0.546 |
| all rows (the shipped mix) | **0.674** | **0.588** |
| German rows only | 0.585 | 0.511 |

German rows do help both spaces a little — MiniLM 0.546 → 0.588 — but neither
space can be trained *into* German: with 51 German band-positives, a German-only
fit is worse than an English-only one in both. Jina reaches 0.674 on German from a
corpus that is two-thirds English; MiniLM reaches 0.588 from the same corpus. That
is a property of the spaces, not of the mix.

**Upweighting German makes it strictly worse.** Sweeping the German rows' sample
weight in the Ridge fit (`eval_german_weighting.py`,
`REPORT_german_weighting.json`) is monotone in the wrong direction:

| w(de) | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM, German AUC | **0.588** | 0.582 | 0.578 | 0.561 | 0.536 | 0.501 | 0.461 |
| MiniLM, German recall @20% | **29.4%** | 29.4% | 25.5% | 25.5% | 21.6% | 15.7% | 13.7% |
| Jina, German AUC | **0.674** | 0.666 | 0.660 | 0.650 | 0.633 | 0.617 | 0.611 |

Both spaces decline, so this is not a MiniLM quirk — it is 51 positives being
asked to speak for a 384- or 1024-dimensional fit. The weight that works is 1.

Nor does training for recall instead of rank rescue it. `REPORT_ge_embeddings.md`
§4 tried the recall-shaped variants on the *aggregate* (bigger heads, logistic on
the binarised band, low regularisation) and found they cost more than they bought;
run per language and German-upweighted, the best MiniLM German AUC any of them
reaches is **0.589** — the unweighted baseline — and it gets there with an
external-leak AUC of 0.549, i.e. by ranking German external posts *above* internal
ones. Low-λ Ridge is worse still (0.510–0.531). There is no German-data fix here:
the axis is not linearly present in the MiniLM space for German text.

That is the expected result from the model cards. `all-MiniLM-L12-v2` is trained
on English pairs only; `jina-embeddings-v5-text-small` is multilingual. The
finding here is the size of the penalty on this axis, not its existence.

Two corroborating signals:

- **The two rankings barely relate on German.** Spearman between the Jina and
  MiniLM scores: English 0.51, German **0.21** (gold set: 0.57 vs 0.24). Fusion
  has less to fuse exactly where one of the inputs is unreliable.
- **German text sits in a tighter clump in MiniLM.** Mean pairwise cosine within
  language, on the val set: MiniLM en 0.079, de **0.127**; Jina en 0.763, de
  0.680. MiniLM spreads English out and collapses German together — the geometry
  of a model that encodes "this is German" more strongly than what the German
  says.

**Length is not the confound.** German and English rows are the same size (median
120 vs 116 chars, p90 280 vs 286 — well inside MiniLM's 128-token window). On a
length-matched subset (the German p10–p90 band): Jina de 0.601 / en 0.634,
MiniLM de 0.522 / en 0.736. The gap survives intact.

## Measured on the group the feed actually routes

Everything above splits by the post's declared `langs` tag. The shipped feed has
no such tag (GE does not send one) and splits by `looksGerman()` instead, so the
question that decides whether the routing is worth anything is how the two spaces
compare **on the posts that detector flags** — its false positives included, its
misses excluded. Flags come from the shipped implementation itself
(`bunny_feed/dump_lang_flags.ts`), not a Python copy that could drift from it.

It flags 638 of the 2822 corpus rows (22.6%) and 75 of the 500 gold posts (15%).
Out-of-fold:

| group | n | pos | Jina | MiniLM | Jina − MiniLM | p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed` (flagged German) | 638 | 106 | **0.691** | 0.618 | +0.072 [−0.003, +0.147] | 0.06 |
| `kept` (everything else) | 2184 | 279 | 0.684 | **0.741** | −0.057 [−0.090, −0.025] | 0.00 |

The detector's group is the better-powered version of the same finding — 638 rows
against the declared tag's 345, and the gap nearly clears significance on its own.
Its false positives do not dilute the effect; by the two readouts that are not
AUC, the routed group is the *worst* MiniLM group in the whole analysis:

| group | ρ vs graded target | external-leak AUC ↓ |
| :--- | ---: | ---: |
| | Jina / MiniLM | Jina / MiniLM |
| `routed` | 0.355 / **0.147** | 0.307 / **0.440** |
| declared `de` | 0.363 / 0.174 | 0.279 / 0.396 |
| `kept` | 0.346 / 0.443 | 0.305 / 0.251 |

On the gold set the same holds on the contrast the feed acts on: internal-vs-
external over the 42 routed posts is Jina 0.819, MiniLM 0.689, fusion 0.833.

## The rule, end to end — and what the cap does

Budgets scaled from the live config (shortlist 150, bypassLimit 50, of a ~706-post
pool). This corpus is 22.6% flagged against the live pool's 1.4%, so it is the
stress case: the one where the cap binds and has to choose.

| arm | paid | band-pos recall | routed positives | English positives |
| :--- | ---: | ---: | ---: | ---: |
| global MiniLM cut (today) | 600 | 44.7% | 31.1% | 46.5% |
| routed, bypass in pool order | 800 | 47.3% | 27.4% | 54.1% |
| **routed, bypass in MiniLM order** | 800 | **52.7%** | **47.2%** | 53.5% |
| routed, budget-matched | 600 | 46.2% | 47.2% | 44.3% |
| Jina on everything (unaffordable) | 600 | 41.6% | 50.0% | 32.4% |

Two things fall out, and the second one changed the implementation.

**Taking the bypass in GE's pool order is wrong.** It gets 27.4% of the routed
positives — *worse than the global MiniLM cut it replaces* (31.1%), which keeps
only 14.9% of routed rows to do it. Ordering the same detour by MiniLM score
instead gets 47.2%, at identical cost.

That looks like a contradiction — ranking by the scorer the group was routed
around — and it is not. What the measurement kills is comparing a German score
against an English one: on a shared scale German is depressed, which is why a
global cut keeps 15.1% of German against 21.4% of English. *Within* the group
that bias is common to every candidate and cancels, and the residual ranking is
still worth something: AUC 0.618, against the 0.500 an arbitrary order earns. The
invalid operation is the cross-language cut, not the score.

**The budget-matched row is the honest headline.** At exactly today's paid cost,
the routing rule takes routed-positive recall from 31.1% to 47.2% — most of the
way to what paying Jina for the entire pool would get (50.0%) — while overall
band-positive recall goes up rather than down (46.2% vs 44.7%). On the live pool
the cap never binds at all (10 flagged of 708), so the feed simply spends 160
Jina slots instead of 150 and the question does not arise.

## Is 47% good? What bounds it

The routed group is 638 posts of which 106 (16.6%) are band-positive, and the
budget at a binding cap buys 31.3% of the group. **An oracle would therefore get
100%** — the budget is not the constraint at that setting. What the 47.2% measures
is ranking quality inside the group, and the relevant comparison is not the oracle
but what can be bought:

| ordering of the detour | 25 slots | 50 | 75 | 100 | 125 | 160 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM score (shipped) | 31.1% | 47.2% | 59.4% | 71.7% | 81.1% | 100% |
| Jina score (unreachable) | 33.0% | 54.7% | 69.8% | 84.0% | 91.5% | 100% |
| oracle | 94.3% | 100% | 100% | 100% | 100% | 100% |

(Slots in live terms; the Jina row is unreachable by construction — it needs the
paid score to decide what to pay for.)

So at the shipped budget the free ordering reaches **86% of what the paid ranker
itself could do with the same slots**. The remaining 53 points to the oracle are
not recoverable by any ranker available here: Jina's within-group AUC is 0.691, and
nothing in §"It is the space" gets a free scorer past 0.589 on German. Recall at
this budget is ranker-bound, and the ranker is exhausted.

That leaves one lever that does move it, and it is a straight token trade: the
budget. 100 slots buys 71.7%, 160 buys the whole group. Which is why the budget is
no longer a fixed 50.

## The budget is share-matched, not fixed

A fixed 50 is right at exactly one pool composition. Measured across the three
this deployment actually draws:

| pool | German | a fixed 50 pays for | share-matched pays for | (the rest gets) |
| :--- | ---: | ---: | ---: | ---: |
| live, generic DID | 1.4% | 100% of them | 21.5% | 21.5% |
| labelled corpus | 22.6% | 31.3% | 27.5% | 27.5% |
| one reader's own likes | 71% | 10.0% | 73.3% → capped | 73.3% |

At the low end a fixed cap is harmlessly generous; at the high end it reinstates
exactly the under-service this detour exists to remove, arriving through a
different door. `bypassBudget()` in feeds.ts gives the detour the same fraction of
its own group that the shortlist takes of the rest, floored at 50 (a handful of
German posts costs a rounding error, so buy them all) and ceilinged at the
shortlist (the paid stage doubles at worst rather than growing with the pool).

Past roughly half a pool being unreadable to the free scorer, the honest move is
not a bigger detour but no prefilter at all — the single-stage path `preFellBack`
already takes when hydration stops arriving. Nothing above 22.6% German is
measured here, because no labelled pool goes there; `pre_bypass` /
`pre_bypass_dropped` in feed_logs are what would say whether it needs to be.

## What it does to the planned prefilter

`REPORT_ge_embeddings.md` §5 recommends scoring 100% of candidates with MiniLM
and paying Jina only for the top ~20%. A single global cut over a mixed-language
pool penalises German twice — fewer German posts are kept, and the ones kept are
the wrong ones:

| cut | space | share of German kept | German positives recalled | (English positives) |
| :--- | :--- | ---: | ---: | ---: |
| top 20% | **MiniLM** | 15.1% | **29.4%** | 45.4% |
| top 20% | Jina | 23.2% | 43.1% | 30.3% |
| top 20% | fusion | 17.4% | 35.3% | 42.7% |
| top 10% | **MiniLM** | 5.2% | **13.7%** | 30.8% |
| top 10% | Jina | 11.9% | 21.6% | 15.7% |
| top 10% | fusion | 6.4% | 15.7% | 22.2% |

At the recommended 20% keep, the MiniLM prefilter drops 70% of German internal
posts before Jina ever sees them — and Jina, on its own scores, would have kept
43%. The loss happens at the stage the architecture calls "free and lossless".

## Caveats

- The OOF truth is the **LM Studio teacher's** band label, not gold. It measures
  fidelity to the teacher, not absolute correctness. What makes that acceptable
  here: `REPORT_vector_teacher_by_lang.md` found the 27B teacher is *not* the weak
  link on German (hard-label accuracy 0.778 / kappa 0.605 on German against 0.687
  / 0.507 on English), so German teacher labels are not noise that would depress
  both spaces — and in any case Jina reproduces them at 0.674.
- The gold-standard val set cannot settle this: 72 declared German posts, **9**
  internal. It is consistent in direction on the internal-vs-external contrast
  the feed acts on (Jina 0.753, MiniLM 0.650, fusion 0.765) and null everywhere
  else, exactly as its power predicts.
- `de+det` mixes langdetect output into the group and is noisier than the declared
  tag; it moves every number toward the middle without changing a sign.

## What to do

1. **Do not ship the global MiniLM prefilter as a single pool-wide cut.** Route
   non-English candidates around it, straight to Jina.

   **Where the language comes from — not from GE.** Checked live 2026-09-11
   (`num_candidates: 200`, `hydrate_embeddings: true`): a candidate carries
   `at_uri, content, minilm_l12_embedding, score, generator_name, author_did,
   author_username, contains_images, contains_video, image_count, video_count,
   external_uri, like_count`. **There is no language field**, and the feed never
   fetches the AT record (it works purely off GE candidates + Jina), so the post's
   own `langs` is not available anywhere in the current path.

   Two ways to get it without a new network call:

   - **A stopword ratio over GE's `content`** (~20 lines, no dependency, runs in
     the edge runtime). Against the declared tags on the 1759 de/en corpus rows:
     accuracy 0.955, German recall 0.858, precision 0.908, and only **2.1%** of
     English rows falsely flagged. This is the recommended one.
   - **The free embedding itself.** Language is linearly decodable in MiniLM —
     logistic 5-fold CV AUC **0.966**, and a plain dot product (the machinery
     `minilm.ts` already runs) gets 0.960. But at 95% German recall it also drags
     14–24% of English candidates onto the paid path, which is worse than the text
     heuristic on both axes. No reason to prefer it when we have the text.

   **Cost of the routing:** on a live GE pool for the default example DID, the
   stopword detector flagged **1.0%** of candidates (2 of 200, both genuinely
   German), so the Jina saving is essentially untouched. That pool is not a German
   user's pool, though — the training corpus, drawn from a German-leaning network,
   flags 22.7%. Budget the paid stage against the user, not against the average.
2. Failing that, cut **per language quantile** rather than globally — it fixes the
   keep-rate column but not the recall column, since MiniLM's German ranking is
   near-chance whatever threshold is applied to it.
3. Fusion is not a repair. On German it lands at 0.675 — between the two, which is
   what averaging a good ranker with a near-random one does.
4. If a free German prefilter is wanted later, the fix is a multilingual free
   encoder — not more German rows, not heavier German rows, and not a recall-shaped
   head, all three of which are measured above and none of which clear 0.589. GE's
   hydrated field is MiniLM-L12 and cannot be changed from here.
