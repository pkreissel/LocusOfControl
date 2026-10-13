# Open-weight System One models against `jev-latest`

**Question.** The LoC teacher is a paid TypeSafe call. Two open candidates claim
to do the same job locally. Can either replace it?

**Answer.** No, and neither is worth running at all: both are beaten by the free
Jina dot product already deployed at the edge.

Measured 2026-09-19 on the 500 Gemini-labelled gold posts
(`dataset_validation_gemini.jsonl`; 81 internal / 243 external / 176 neutral),
same harness, same groupings and same 4000-sample bootstrap as
`REPORT_typesafe_vs_jina.md`. Scripts: `research/evals/score_loc_laya.py`,
`score_loc_qwen_rlcd.py`, `eval_laya_vs_jev.py`, `eval_laya_floor.py`.

## internal-vs-rest, all 500 posts

Best primitive per system. AP and P@25 lead, per CLAUDE.md — the feed consumes
the head of the ranking and never sees the tail.

| system | primitive | AUC [95% CI] | AP | P@25 | ΔAUC vs `jev.choice` |
|---|---|---|---|---|---|
| **jev** (`jev-latest`, paid) | choice | **0.942** [0.914, 0.965] | **0.788** | **0.920** | — |
| jina (deployed vector) | dot product | 0.803 [0.741, 0.858] | 0.523 | 0.680 | −0.139 [−0.202, −0.081] p<0.001 |
| qwen_rlcd, Qwen2.5-7B backbone | score | 0.868 [0.816, 0.916] | 0.631 | 0.760 | −0.157 [−0.258, −0.048] p=0.005 |
| qwen_rlcd, Qwen3-8B backbone | choice | 0.837 [0.781, 0.888] | 0.615 | 0.840 | −0.173 [−0.275, −0.073] p=0.001 |
| qwen_rlcd, Qwen2.5-1.5B (engine default) | score | 0.743 [0.683, 0.802] | 0.439 | 0.600 | −0.349 [−0.446, −0.239] p<0.001 |
| laya (English, ModernBERT-large 421M) | noul | 0.685 [0.621, 0.747] | 0.340 | 0.440 | −0.257 [−0.321, −0.193] p<0.001 |
| laya-multilingual (mmBERT-base 322M) | noul | 0.622 [0.555, 0.690] | 0.245 | 0.360 | −0.319 [−0.392, −0.246] p<0.001 |

ΔAP, not ΔAUC: AP is the lead metric here, and the two have disagreed in sign on
a real retrain of this axis before. Every gap to `jev` is significant at n=500.
The ordering is the same on internal-vs-external, on the declared-English subset,
on the declared-German subset and on what the shipped `looksGerman()` detector
routes.

`jina` reads 0.803 here against 0.763 in `REPORT_typesafe_vs_jina.md`: that
report scored the archived bonsai-v2 vector, this one scores the deployed
TypeSafe-distilled vector (`internal_vector.npy`). Same posts, different probe.

## Laya's choice head is anti-correlated on this axis

`laya_ml.choice` scores AUC 0.374 [0.307, 0.446] — significantly *below* chance,
so its P(internal) ranks internal posts lower than the rest. Inverting it gives
0.626, still under the Jina vector. Its 3-class output shows why: it labels
404 of 500 posts `internal` where the gold set has 81, and the English
checkpoint collapses the other way, labelling 499 of 500 `neutral`.

| system | 3-class acc | macro F1 | F1 int | predicted mix (gold: 81/243/176) |
|---|---|---|---|---|
| jev | 0.810 | 0.762 | 0.614 | 46 int / 233 ext / 221 neu |
| qwen_rlcd | 0.578 | 0.392 | 0.023 | 7 / 404 / 89 |
| laya_en (compact) | 0.378 | 0.248 | 0.122 | 17 / 12 / 471 |
| laya_ml (compact) | 0.232 | 0.212 | 0.198 | 292 / 168 / 40 |

## It is not the rubric, and it is not this axis

Two controls, because "the axis is subjective" is the easy explanation and it is
wrong here.

**The rubric was given twice.** Laya packs instructions and options into
`head_max_len` (256 tokens on the multilingual checkpoint, 192 on the English
one) and hard-truncates every option to 48 tokens. `ts_questions.py` was written
for an API with no such cap and its options run 49–93 tokens, so fed verbatim the
score question leaves 16 tokens for its instructions and all five levels lose
their tails. `loc_questions_compact.py` is the same rubric compressed until
nothing clips (asserted by its `check()`). It moves the English checkpoint from
AUC 0.606 to 0.668 on choice and the multilingual one not at all
(0.374 → 0.384). Truncation was not the problem.

**The model cannot answer questions with objective answers.** From
`eval_laya_floor.py`, on the same 500 posts, ground truth taken from the corpus
rather than a labeller:

| | laya-multilingual | laya (English) |
|---|---|---|
| German vs English, acc (majority class 0.794) | 0.426 | 0.409 |
| contains `?`, AUC | 0.706 | 0.585 |
| contains a link, AUC | 0.790 | 0.913 |

Neither checkpoint can tell German from English at better than a coin flip, on a
task its own mmBERT / ModernBERT encoder does trivially, and neither reliably
detects a literal question mark. Whatever these decision heads are doing, they
are not conditioning on the question being asked, so the LoC numbers above are
not a measurement of the axis being hard.

## The backbone is the only lever that moved anything

`harshatheg/Qwen-2.5-1B-RLCD` hardcodes `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
as a module constant. It is not a trained part of the engine, and swapping it is
the one change in this whole benchmark that produced a real improvement.

| backbone | AUC | AP | P@25 | 3-class acc / macro F1 | p50 | weights |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B (default) | 0.743 | 0.439 | 0.600 | 0.578 / 0.392 | 0.31 s | 1.0 GB |
| Qwen2.5-7B | 0.868 | **0.631** | 0.760 | 0.694 / **0.658** | 1.50 s | 4.0 GB |
| Qwen3-8B | 0.837 | 0.615 | **0.840** | **0.738** / 0.615 | 1.99 s | 4.3 GB |

Best primitive per row; `score` for both Qwen2.5 rows, `choice` for Qwen3-8B.

**1.5B → 7B is real.** AP 0.439 → 0.631, and the 3-class output stops collapsing:
the 1.5B labels 404 of 500 posts `external` where gold has 243, the 7B predicts
144/261/95 against gold 81/243/176.

**7B → Qwen3-8B is not measurable here.** ΔAP = −0.017 [−0.128, +0.086] p=0.72.
Qwen3-8B leads on P@25 (0.840 against 0.760) and that is 21 hits against 19 out
of 25 — its bootstrap CI is [0.640, 0.960]. AUC and AP prefer the 7B, P@25
prefers Qwen3-8B, and nothing separates them. Qwen3-8B is the more even of the
two across languages (AUC 0.857 en / 0.762 de on choice, against the 7B's 0.879 /
0.750 on score), which is the only difference worth a second look.

**Neither clears the deployed Jina vector, as measured.** The point estimates say
they do — AP 0.631 and 0.615 against 0.523 — but paired:

    AP(qwen_7b.score)   - AP(jina.vector) = +0.108 [-0.017, +0.237] p=0.085
    AP(qwen3_8b.choice) - AP(jina.vector) = +0.092 [-0.036, +0.222] p=0.174

Both CIs straddle zero. This gold set cannot resolve the difference: 500 rows
with 81 positives put ±0.16 on P@25 and ±0.12 on a paired AP gap, which is the
same limit already recorded for sweep and epoch selection on this axis. A claim
that a 4 GB local model beats the shipped dot product needs a larger labelled
set, not a bigger backbone.

## Which backbones the engine can carry

Compatibility is decided by one line: the engine prefills at batch 1 and widens
the cache with `mx.repeat(layer.keys, M, axis=0)`, guarded by
`hasattr(c, "keys")`.

- **Works unchanged** — anything whose model class has no `make_cache` override,
  so `make_prompt_cache` returns a plain `KVCache` per layer: Qwen2.5 at every
  size, Qwen3 dense (0.6B–32B), and the MoE variants (`qwen2_moe`, `qwen3_moe`,
  `qwen3_5_moe`).
- **Silently wrong** — hybrid models. `mlx_lm/models/qwen3_5.py` builds
  `[ArraysCache(size=2) if l.is_linear else KVCache() ...]`, and with Qwen3.8-27B's
  `full_attention_interval: 4` that leaves 48 of 64 layers holding a recurrent
  state with no `.keys` to widen. The engine *skips* those layers rather than
  failing, so every field but the first is scored against a prefix it never saw.
  Same for `qwen3_next`.

`score_loc_qwen_rlcd.py` detects this from the cache types and switches to a
tiled prefill: tile the prompt to batch M and prefill at batch M, so the cache is
built at the right width whatever its type. That costs M× the prefill FLOPs —
exactly the saving the broadcast exists to make — so it is not a replacement for
the engine, only a way to measure a backbone the engine cannot carry. On a
plain-attention backbone the two paths agree to `max |Δp| = 0.0092` over 20 posts
× 3 fields (`QWEN_TILED_CHECK=1`), which is 4-bit kernel nondeterminism between
batch shapes rather than a difference in logic.

## What `Qwen-2.5-1B-RLCD` actually is

It ships no weights: `harshatheg/Qwen-2.5-1B-RLCD` is an inference engine over
stock `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. Its claim is latency and
schema validity — one prefill, one batched pass over the schema's fields,
softmax over each field's candidate first-tokens — and that claim holds; nothing
here contradicts it. What the numbers say is that the axis is mostly a
function of the backbone: at the engine's default 1.5B it ranks below the Jina
vector, and at 7–8B it draws level with it and with `jev.score`, all zero-shot
and never trained on the axis.

Two things to know if it is used elsewhere:

- **Candidates must differ in their first token.** Otherwise the engine drops
  into a multi-token disambiguation path that returns a *fabricated* flat
  distribution (`(1-p)/(k-1)` spread over the losers) instead of a calibrated
  one. That rules out the v4 band names — "Strong external" and "Mild external"
  share a token — which is why the levels here are digits with the legend in the
  context. `score_loc_qwen_rlcd.py` asserts `has_collisions` is empty.
- **Field descriptions are truncated to their first line** by the prefill
  catalog (`to_parallel_schema_str`). A multi-line rubric hung off a field is
  silently dropped; it has to go in the context.

## Cost and latency

Latency is one post at a time on an M2 Max, and is not comparable to `jev`'s,
which is a network round trip with 12-way concurrency available.

| system | p50 | p95 | USD / 1k posts | weights |
|---|---|---|---|---|
| jev | 0.624 s | 0.716 s | 0.049 | — (API) |
| qwen_rlcd, Qwen2.5-7B | 1.495 s | 2.440 s | 0 | 4.0 GB |
| qwen_rlcd, Qwen3-8B | 1.993 s | 2.142 s | 0 | 4.3 GB |
| laya-multilingual | 0.045 s | 0.058 s | 0 | 647 MB |
| laya (English) | 0.103 s | 0.126 s | 0 | 2.2 GB |
| qwen_rlcd, Qwen2.5-1.5B | 0.305 s | 5.578 s | 0 | 1.0 GB |

`qwen_rlcd`'s p95 is 18× its p50 because MLX holds freed buffers in its own
cache: an uncapped run allocated a fresh prompt cache per post, broadcast it
across the schema's fields, and was SIGKILLed by the OOM killer at post 230 of
500 on a 32 GB machine. `score_loc_qwen_rlcd.py` now caps the buffer cache
(`QWEN_MLX_CACHE_MB`, default 1024) and calls `mx.clear_cache()` every 25 posts,
which holds the process at 1.3 GB RSS / 1.6 GB MLX peak for the whole run and
leaves those reclaim pauses in the tail.

## Consequence

The teacher stays `jev-latest`. Nothing in `training/` or `bunny_feed/` changes.
The Qwen ladder is the only thread worth pulling, and pulling it needs a bigger
gold set before a bigger backbone: at n=500 / 81 positives the 7B and Qwen3-8B
are indistinguishable from each other and from the shipped vector, so a 14B or
32B run would produce a number this set cannot interpret.
The score files and both JSON reports are under `research/data/evals/` if a
future checkpoint is worth a rematch; the two scorers take a model id and a
corpus and will re-run against one unchanged.
