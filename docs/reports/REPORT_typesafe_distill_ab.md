# Does a better teacher make a better vector? TypeSafe vs bonsai, same posts

Measured 2026-09-17. Controlled A/B: **identical 2500 training posts, identical Jina
embeddings, identical Ridge sweep and selection metric — only the teacher's labels
change.** Validation is the 500-post Gemini gold set, held out of training by
`load_train_rows()` as it always is.

Code: `build_typesafe_teacher.py` (adapter). Arms:
`internal_vector_ab_base.npy` (bonsai), `_ab_ts.npy` (TypeSafe, flat band),
`_ab_tsg.npy` (TypeSafe, band intensity = 6 + 3·P(internal)).

Row alignment was verified before training: the embedding caches are positional, and
the adapter reproduces `load_train_rows()`'s filter and order exactly (2500 rows,
order identical to the bonsai file). The baseline arm reproduces the shipped vector's
published 0.763 within selection noise.

## Result: the teacher is better, the vector barely moves

AUC of internal-vs-rest on the gold 500, paired bootstrap (5000 resamples, same
resamples across arms) against the bonsai baseline trained in the same run.

| model | AUC (v1 labels) | Δ vs baseline | AUC (v2 labels) | Δ vs baseline |
|---|---|---|---|---|
| shipped `internal_vector.npy` | 0.763 | — | 0.772 | — |
| A/B baseline (bonsai teacher) | 0.775 | — | 0.794 | — |
| A/B TypeSafe, flat band | 0.793 | +0.018 [−0.004, +0.040] p=0.105 | 0.808 | +0.014 [−0.009, +0.038] p=0.231 |
| **A/B TypeSafe, graded band** | **0.808** | **+0.033 [+0.010, +0.057] p=0.007** | **0.820** | **+0.026 [+0.001, +0.050] p=0.040** |
| **TypeSafe direct (no vector)** | **0.942** | **+0.166 [+0.109, +0.226]** | **0.952** | **+0.158 [+0.097, +0.225]** |

The teacher swap is real but small: **+0.03**. The same labels, used directly instead
of distilled, are worth **+0.17**. Distillation discards 0.133 [0.081, 0.191] of AUC —
four times the entire benefit of changing teacher.

Using P(internal) as the intensity inside the 6–9 positive band (rather than the flat
6.0 the bonsai rows fall back to) is worth +0.015 [+0.006, +0.025] on its own, and is
free. Do not use TypeSafe's Score primitive for this; P(internal) comes from Choice,
the strong primitive.

## Why: the linear probe, not the teacher, is the ceiling

5-fold out-of-fold Ridge on the 2500, each teacher predicting *its own* labels:

| target | positives | OOF AUC |
|---|---|---|
| bonsai teacher | 298/2500 | 0.677 |
| TypeSafe teacher | 266/2500 | 0.683 |

A linear probe in Jina space reproduces the *cleaner* teacher no better than the noisy
one — 0.683 vs 0.677. The labels improved; the hypothesis class did not. That is the
whole story of the +0.03.

## Learning curve: more annotation helps, slowly

Mean over 12 subsample draws, evaluated on the gold 500 (v1 labels):

| n_train | TypeSafe | bonsai |
|---|---|---|
| 250 | 0.709 | 0.700 |
| 500 | 0.725 | 0.705 |
| 1000 | 0.747 | 0.727 |
| 1500 | 0.755 | 0.735 |
| 2000 | 0.763 | 0.741 |
| 2500 | 0.768 | 0.744 |

Still rising at 2500, and TypeSafe's margin widens with volume (+0.009 at n=250 to
+0.024 at n=2500) — a better teacher is worth more the more of it you buy. But the
slope is ~+0.005 per 500 rows and flattening. At `$0.028/1k` even 50k rows is $1.40,
so buy them; just do not expect them to close a 0.13 gap that is architectural.

## What this changes

1. **Adopt TypeSafe as the teacher** — it is strictly better at equal cost, and the
   graded-band target makes the difference significant. But book the gain as +0.03,
   not as the +0.18 the label-level comparison suggested.
2. **The distillation step is now the main loss in the system.** `post_scores` cache
   warming with TypeSafe directly (`bunny_feed/push_score_cache.ts`,
   `local-jina-weights-match-the-api`) is not a nice-to-have — it is worth 5× what
   retraining the vector is worth, at $0.09/day. The vector remains the fallback for
   uncached posts.
3. **`loc-threshold-tracks-the-vector` applies.** Every arm here is a new vector with a
   new score distribution; the 0.88 gate is meaningless for them until recalibrated.

## Caveats

* The validation labels are Gemini's, and Gemini is the odd one out on 11.6% of posts
  where all three labellers are compared. A TypeSafe-trained vector is scored wrong on
  exactly the posts where Gemini is wrong, so +0.03 is a lower bound.
* The sweep selects on the validation set (`val_ndcg_graded`), so absolute AUCs are
  optimistic. This is identical across arms, so the paired deltas are unaffected.
* The corpus annotated here is the **bonsai** corpus — the training set of the
  deployed vector, not of the newer muse-glimmer teacher
  (`dataset_annotated_pilot_v4.jsonl`, 1500 rows). The comparison answers "better than
  what ships", not "better than the current teacher". The glimmer arm is $0.04 and
  `embeddings_pilot_v4_jina.npz` already exists.

---

# Part 2: the same test against the *current* teacher (muse-glimmer, 5000 rows)

The corpus above is the bonsai one — the deployed vector's training set, but not the
newest teacher. `dataset_annotated_lmstudio_soft.jsonl` (5500 rows, v3 Label/Level
prompt, `annotate_lmstudio_soft_labels.py` default model `meta/muse-glimmer`) is the
current teacher of record, and `embeddings_lmstudio_soft_v3_jina.npz` (5000 rows =
5500 − the 500 gold) aligns with it. TypeSafe scored all 5500 in 146s, 0 errors,
$0.27.

Note glimmer is a *harder* baseline than bonsai in one respect and a weaker one in
another: twice the data, but only 268 internal rows in 5000 (5.4%, vs bonsai's 11.9%).
TypeSafe labels the same posts 10.0% internal and 48.8% external — glimmer under-fires
both poles, exactly as the bonsai teacher did.

## Tuned arms (full sweep, selection on the val set)

| model | AUC v1 | ΔAUC vs glimmer | AP | ΔAP | P@10 |
|---|---|---|---|---|---|
| shipped (bonsai 2.5k) | 0.763 | — | 0.368 | — | 0.50 |
| glimmer baseline (5k) | 0.786 | — | 0.436 | — | 0.50 |
| glimmer + TypeSafe flat | 0.790 | +0.004 [−0.024, +0.031] | 0.477 | +0.041 [−0.018, +0.097] | 0.80 |
| **glimmer + TypeSafe graded** | 0.803 | +0.016 [−0.012, +0.045] | **0.523** | **+0.087 [+0.021, +0.143] p=0.007** | **0.90** |
| TypeSafe direct | 0.942 | +0.155 [+0.100, +0.216] | 0.788 | +0.352 | 1.00 |

**Against the current teacher the AUC gain disappears, but the head of the ranking
improves sharply.** P@10 goes 0.50 → 0.90 and average precision +0.087. For a feed
that consumes the top of a ranking and never sees the tail, AP and P@10 are the
operative metrics and AUC is the misleading one. (On the v2 label set the AP gain is
+0.068, p=0.096 — same direction, not significant.)

## Matched-n, fixed alpha (no selection noise)

The tuned arms above each pick hyperparameters on the 500-post val set, which injects
selection noise comparable to the effect being measured. Refitting at fixed alpha=1.0
over 12 subsample draws per point isolates the teacher:

| n | AUC glimmer | AUC TypeSafe | AP glimmer | AP TypeSafe |
|---|---|---|---|---|
| 500 | 0.695 | 0.730 | 0.311 | 0.345 |
| 1000 | 0.710 | 0.751 | 0.322 | 0.401 |
| 2000 | 0.742 | 0.779 | 0.363 | 0.461 |
| 3000 | 0.753 | 0.784 | 0.394 | 0.486 |
| 4000 | 0.757 | 0.790 | 0.402 | 0.510 |
| 5000 | 0.760 | 0.791 | 0.409 | 0.515 |

TypeSafe wins at every sample size, by +0.031 AUC and +0.106 AP at n=5000, and the AP
margin *widens* with data. The flat ΔAUC in the tuned table is selection noise, not an
absence of effect.

## And here the probe capacity does move

| target | positives | 5-fold OOF AUC |
|---|---|---|
| glimmer | 268/5000 | 0.658 |
| **TypeSafe** | 498/5000 | **0.723** |

Unlike the bonsai corpus (0.677 vs 0.683 — no difference), the TypeSafe target on this
corpus is substantially more learnable, +0.065. Glimmer's 5.4% internal rate is sparse
enough that the probe has little to fit; TypeSafe's 10% gives it a real signal. So the
"the probe is the ceiling" conclusion from Part 1 is corpus-specific: against a teacher
this conservative, better labels *do* buy representable structure.

## Revised conclusion

* **Retrain on TypeSafe, graded band, on the glimmer corpus.** +0.087 AP and P@10
  0.50 → 0.90 is a large feed-visible gain even though AUC barely moves.
* **The distillation gap remains the dominant loss**: TypeSafe direct is 0.942/0.788
  against the best distilled 0.803/0.523. Cache-warming `post_scores` is still worth
  several times what any retrain is worth.
* **Judge this axis on AP/P@10, not AUC.** The two metrics disagree in sign here, and
  the one that matches what the feed does is AP.

---

# Part 3: the two vectors want different objectives

Deploying both vectors selected on `val_ndcg_graded` was a mistake caught after
the fact. The Jina vector is the quality **gate** and is rightly judged on the
head of the ranking. The MiniLM vector is a **prefilter**: its only job is to
hand a shortlist to the paid stage, so anything it drops is lost for good, while
its ordering within the shortlist is rescored and fused anyway. Those are
different objectives and they moved in opposite directions on this retrain.

## The production cut, measured not assumed

Recall has to be evaluated at the n where the cut actually happens. Over five
production requests (2026-09-17) the `X-Prefilter` header reported `hyd=754-755`
hydrated candidates, of which 11-12 were German and bypassed the stage, and
`kept=161-162` = `shortlist` 150 + the bypassed. So the cut is **150 of ~743
non-German candidates, 20.2%** — now `PREFILTER_CUT` in
`train_internal_vector.py`, to be re-derived whenever `shortlist`,
`numCandidates` or `minWords` changes.

## What the retrain did to recall

Recall, at that cut, of the posts the paid Jina stage would keep (non-German):

| MiniLM | recall@cut | recall@30% | Spearman vs Jina | fused AUC | fused AP |
|---|---|---|---|---|---|
| LIVE (bonsai labels) | **0.375** | 0.510 | 0.544 | 0.816 | 0.509 |
| NEW (TypeSafe labels) | 0.349 | 0.464 | 0.495 | **0.829** | **0.543** |
| distilled from the Jina score | **0.375** | **0.531** | 0.566 | 0.806 | 0.499 |

Three findings:

1. **The TypeSafe labels cost 2.6 points of prefilter recall** (paired
   −0.026 [−0.052, +0.015]) while improving precision@25 from 0.56 to 0.68.
   Selecting on recall instead of nDCG picks the *same* vector — the labels are
   the binding constraint, not the hyperparameters.
2. **Distilling MiniLM from the Jina score maximises recall and destroys
   fusion.** It is the best prefilter in isolation and drags fused AP to 0.499,
   *below* Jina alone at 0.523. The `fuse` design in feeds.ts depends on MiniLM
   being an independent view; making it a cheap Jina mimic removes the thing
   that made the second opinion worth having. Not shipped.
3. **End-to-end, the TypeSafe vector still wins.** Simulating the whole path at
   production geometry — 20.2% shortlist, German bypass, gate 0.86, fusion, show
   100 of ~162 survivors — the feed is 52% internal against 46%, +0.052
   [−0.035, +0.138], P(better) = 87%. The recall lost at the shortlist is more
   than repaid by the better ordering downstream.

So the deployed pair stays as it is. What changed is that the objectives are now
explicit: `LOC_SELECT_METRIC=val_precision@50` for Jina and `val_recall@cut` for
MiniLM, wired into run_all.sh, so the next retrain cannot improve the feed's top
while quietly shrinking what reaches it.

## Caveat

`val_recall@cut` in the trainer measures recall of the *validation label*, not of
gate-passers — the truer target needs the other space's vector and threshold, which
a single-space training run does not have. The gate-passer numbers above were
measured against the trained pair afterwards. Do that before deploying a prefilter.
