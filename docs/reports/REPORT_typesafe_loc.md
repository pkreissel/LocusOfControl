# TypeSafe System One (jev) on the Locus-of-Control axis

Measured 2026-09-17 against `dataset_validation_gemini.jsonl` (the 500 Gemini-labelled
held-out posts) and `dataset_annotated_bonsai_v2.jsonl` (the 2500-row local-teacher
corpus the deployed Jina vector was distilled from).

Code: `ts_questions.py` (the rubric as System One questions), `score_loc_typesafe.py`
(the client), `eval_typesafe_vs_jina.py`, `eval_typesafe_teacher_oof.py`.
Raw numbers: `REPORT_typesafe_vs_jina.json`, `REPORT_typesafe_teacher_oof.json`.

## What it is

`POST https://api.typesafe.ai/v1/systemone` takes a `state` (the post) and a map of
typed questions, and returns a probability distribution per question — no generated
text. Three primitives: **Choice** (pick a class, with per-class probabilities),
**Score** (position on 2–10 ordered levels, returned as a probability-weighted mean),
**Noul** (is this statement true, 0–1). Every question is evaluated in isolation
against the same state, so several can ride in one request.

The LoC rubric of `label_prompts.py` maps onto it directly: `DEFINITIONS_V3` becomes
Choice criteria, and v4's 0–9 band scale folds onto a 5-level Score.

## Headline: it beats the deployed vector by a wide margin

Gold set, AUC of P(internal) against the Gemini label, internal-vs-rest.
Jina is the shipped `internal_vector_bonsai_v2` probe, computed by the same code
path as `eval_minilm_vs_jina_by_lang.py` part A.

| group | n | int | TypeSafe Choice | Jina | difference |
|---|---|---|---|---|---|
| all | 500 | 81 | **0.942** [0.915, 0.964] | 0.763 [0.706, 0.816] | **+0.178** [+0.122, +0.240] p<0.001 |
| en | 278 | 51 | 0.928 [0.895, 0.957] | 0.763 [0.697, 0.829] | +0.165 [+0.098, +0.236] p=0.002 |
| de | 72 | 9 | 0.882 [0.687, 0.991] | 0.672 [0.426, 0.884] | +0.210 [−0.096, +0.530] p=0.168 |
| de+det | 83 | 11 | 0.901 [0.742, 0.992] | 0.712 [0.501, 0.887] | +0.189 [−0.057, +0.447] p=0.123 |
| routed(de) | 75 | 12 | 0.907 [0.769, 0.993] | 0.750 [0.544, 0.915] | +0.157 [−0.068, +0.392] p=0.172 |

`loc-gold-set-cannot-resolve-model-gains` said this set has an AUC CI of ±0.077 and
cannot separate models. That holds for the *marginal* CIs, and the paired bootstrap is
what gets around it: the same 500 posts scored by both systems, resampled together.
+0.178 [+0.122, +0.240] clears the noise floor by a wide margin. German on its own
still does not — 9 internal posts is 9 internal posts.

## The confound, and why it does not explain the result

The gold labels are Gemini's output. TypeSafe is a model handed roughly the rubric
Gemini was handed, so part of that margin could be two language models agreeing with
each other rather than either being right.

So the same comparison was run against a *different* labeller, arranged to favour
Jina as hard as possible: the reference is the local llama.cpp/bonsai teacher, and
Jina is a 5-fold out-of-fold Ridge fit **on that corpus's own teacher targets**.
Jina is predicting its own teacher; TypeSafe has never seen it.

| group | n | pos | TypeSafe Choice | Jina (OOF, own teacher) | difference |
|---|---|---|---|---|---|
| all | 2500 | 298 | **0.925** | 0.673 | **+0.252** [+0.220, +0.283] p<0.001 |
| en | 1414 | 185 | 0.923 | 0.631 | +0.292 [+0.249, +0.335] p<0.001 |
| de | 345 | 51 | **0.899** | 0.655 | **+0.244** [+0.168, +0.322] p<0.001 |
| de+det | 396 | 55 | 0.907 | 0.657 | +0.250 [+0.178, +0.330] p<0.001 |

The ordering survives, and on German it is now properly powered (51 positives, 6×
the gold set). The margin is not an artefact of who wrote the labels.

## German

This is the operationally important part, given `minilm-english-only-on-loc`.

* Gold set: de 0.882 vs en 0.928 — a 0.046 gap, inside the noise at n=72.
* Teacher corpus: de 0.899 vs en 0.923 — a 0.024 gap at n=345.
* The TypeSafe-minus-Jina advantage does **not** depend on language:
  interaction −0.048 [−0.139, +0.042] p=0.289 (teacher set), +0.045 [−0.276, +0.363]
  p=0.786 (gold set).
* 3-class agreement with the local teacher is flat across languages:
  de 0.788, en 0.786.

MiniLM's German deficit was −0.135 OOF (0.588 de vs 0.723 en, interaction p=0.004).
TypeSafe shows nothing of that shape. A multilingual scorer with no German deficit
removes the reason `looksGerman()` routing exists at all.

## Which primitive — the Score is the trap

The obvious mapping (v4's 0–9 scale onto an ordered Score) is the *worst* of the
three. Gold-set AUC, internal-vs-rest:

| question | tokens/post | $/1k posts | AUC all | en | de |
|---|---|---|---|---|---|
| **Choice** (3-way, P(internal)) | 670 | **$0.028** | **0.944** | 0.932 | 0.888 |
| Score (5 ordered levels) | 740 | $0.031 | 0.880 | 0.878 | 0.810 |
| Noul (single agency statement) | 348 | $0.015 | 0.890 | 0.877 | 0.816 |

The Score collapses: on clearly-neutral posts it returns probability 1.0 on the
middle level, so most of the corpus lands on one point with confidence 0.99 — the
same failure as `virtue-axis-band-clamp-kills-gradient`, arriving through a
different door. Choice keeps a usable gradient because P(internal) stays continuous
even when the argmax is "neutral".

Asking all three in one request costs 1164 tokens and changes no answer (choice-only
0.944 vs choice-in-combined-request 0.942 — isolation confirmed). **Ship Choice
alone.**

`jev-preview` is indistinguishable from `jev-latest`: +0.004 [−0.002, +0.015]
p=0.427, labels agree on 98.6% of posts.

## Calibration is usable

`confidence` is a real signal, not decoration. Gold-set 3-class agreement by
confidence band:

| gate | coverage | 3-class acc | AUC |
|---|---|---|---|
| none | 100% | 0.810 | 0.942 |
| ≥0.5 | 79% | 0.876 | 0.952 |
| ≥0.7 | 63% | 0.934 | 0.941 |
| ≥0.9 | 44% | 0.964 | 0.942 |

The 105 low-confidence posts alone agree only 56% of the time. A confidence gate
isolates the disagreements cleanly enough to route just those to a stronger check —
or to a human.

## Cost and throughput vs Jina

| | Jina vector | TypeSafe Choice |
|---|---|---|
| per post | 1 embedding, ~100 tok, batched 32/req | 1 request, 670 tok |
| marginal cost | ~0 (local weights match the API) | $0.028 / 1000 posts |
| latency | ~0.5s for a 100-post rerank | 0.62s p50, 0.73s p95 — **per post** |
| throughput | 8 parallel × 32/batch | 49 posts/s at 32 workers |
| training needed | teacher corpus + annotation + Ridge fit | none |

Measured over a 3000-post run: 50.3 posts/s, zero 429s, zero errors, p50 flat at
0.62s from 8 to 32 workers. The documented ceiling (1200 req/min = 20/s) was
exceeded by 2.5× without throttling, so treat 20/s as the contractual number and
50/s as what the service currently gives.

## Batch API for feed ranking — there isn't one

TypeSafe batches **questions per state**, not states. Many posts means many requests;
throughput comes from concurrency alone. That decides where it can sit:

* **Not on the live feed path.** The rank stage budget is `GE_RANK_TIMEOUT_MS` =
  6000ms over `MAX_RERANK_CANDIDATES` = 100 posts. At the fan-out the edge worker
  can sustain (`JINA_MAX_PARALLEL` = 8, and `bunny-edge-cache-concurrency` caps
  concurrent ops at ≤8 anyway), 100 posts is 13 rounds × 0.62s ≈ 7.8s. It blows the
  budget before the first candidate is ranked.
* **Yes as an offline warm of `post_scores`.** `bunny_feed/push_score_cache.ts`
  already exists for exactly this (`local-jina-weights-match-the-api`). At ~3.2k new
  posts/day — the figure behind the 4-day TTL in `prune-capacity-traffic-coupled` —
  that is 65 seconds of wall clock and **$0.09/day, ~$2.70/month**. The live request
  keeps reading a cached scalar and never calls TypeSafe.

The cache-warm shape also sidesteps the per-request rubric cost: there is no prompt
caching, so the 670-token rubric is re-billed every call regardless. At this volume
it does not matter.

## Recommendation

1. **Adopt it as the annotator, now.** It reproduces the local teacher at 0.925 AUC
   and Gemini at 0.942, at 50 posts/s and $0.028/1k. Against the current teacher that
   is a straight replacement: no 500/day quota
   (`gemini-free-tier-500-per-day`), no `>3 concurrent` context-overflow failures
   (`llamacpp-teacher-concurrency-limit`), ~60s for a 3000-post corpus.
2. **Re-annotate and retrain the vector against it.** The distilled vector stays the
   thing that runs per-request; a better teacher is the cheapest available lift, and
   `loc-threshold-tracks-the-vector` means recalibrating the 0.88 gate afterwards.
3. **Warm `post_scores` with TypeSafe Choice directly** for the German slice at
   minimum. There is no German deficit to route around, which makes the
   `looksGerman()` prefilter bypass reconsiderable.
4. **Do not put it on the live request path**, and do not use the Score primitive.

## Caveat

Every number above is agreement with a *model's* labels — Gemini's or the local
teacher's. Neither is ground truth for locus of control. The teacher cross-check
shows TypeSafe is not merely Gemini-shaped, but it does not establish that it is
right about the construct. That would need human-labelled data, which this project
does not have.
