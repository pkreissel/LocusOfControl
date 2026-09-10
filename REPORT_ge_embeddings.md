# Are GE's hydrated embeddings enough to replace Jina?

Measured 2026-09-10 against a live candidate pool and the bonsai-v2 labelled corpus.
Short answer: **yes for the pre-filter, and the roles in the "recall then precision"
plan are backwards** — the MiniLM probe is not the weak recall stage, it is the equal
(slightly better) partner. What the two spaces buy together is an *ensemble*, not a
cascade. Keep Jina, but only on the top slice, and fuse the two rankings.

## 1. What `hydrate_embeddings` actually returns

`candidates/generate` with `hydrate_embeddings: true` adds `minilm_l12_embedding`:
base64 of 384 float32s, L2-normalised (‖v‖ = 1.0000).

It is reproducible **exactly** offline. Embedding the returned `content` locally with
`sentence-transformers/all-MiniLM-L12-v2`, mean-pooled, `max_length=128`:

| local truncation | cosine vs GE (n=300) | fraction > 0.99 |
|---|---|---|
| 128 tokens | mean 1.0000, min 0.9995 | 100% |
| 256 tokens | mean 0.9988, min 0.8265 | 99.0% |

So GE truncates at 128 tokens, and a probe can be trained in GE's exact space with no
API call at all. This is what makes the whole evaluation possible offline.

**Coverage is a non-issue.** Of 500 mix candidates, 20 (4.0%) carry no embedding — and
all 20 have *empty* `content`, i.e. exactly the rows `minWords: 6` already drops. After
the existing text gate: 364 candidates, **0** without an embedding.

**Truncation is a non-issue.** Median candidate is 31 MiniLM tokens, p90 = 73; only
0.8% exceed the 128-token window (0% exceed Jina's 2000 chars). The two models see
substantially the same text.

**Hydration is free.** Same request, `n=500`, back to back:

| | latency | payload |
|---|---|---|
| `hydrate_embeddings: false` | 3.31 s / 3.04 s | 0.24 MB |
| `hydrate_embeddings: true` | 2.19 s / 1.43 s | 1.22 MB |

No latency penalty (if anything the hydrated calls were faster; call-to-call noise
dominates). Cost is ~2 KB of base64 per candidate.

## 2. Quality: MiniLM vs Jina, same probe, same rows

`train_internal_vector.py` was run unmodified in both spaces over identical rows
(2500 teacher + 322 interaction), identical splits, identical sweep.

| | val AUC internal | AP | ext-leak AUC ↓ | P@25 | P@50 | graded nDCG |
|---|---|---|---|---|---|---|
| Jina v5 small (1024-d, deployed) | 0.794 | 0.339 | 0.286 | 0.44 | 0.38 | 0.887 |
| GE MiniLM-L12 (384-d, free) | **0.795** | **0.348** | **0.264** | 0.36 | 0.30 | **0.903** |
| rank fusion of the two | **0.826** | **0.366** | **0.245** | — | — | — |

Bootstrap on the AUC gap (500 val posts, 51 internal): `jina − minilm = −0.000`,
95% CI [−0.068, +0.068]. **Statistically indistinguishable.**

The validation set has only 51 positives and both probes were hyperparameter-selected
on it, so it was repeated as 5-fold **out-of-fold** CV over the full 2822-row teacher
corpus (385 band-positives, no selection leakage):

| space | AUC | AP | ρ vs teacher | recall@10% | @20% | @30% |
|---|---|---|---|---|---|---|
| Jina | 0.684 | 0.248 | 0.335 | 20.3% | 38.4% | 51.7% |
| GE MiniLM | **0.711** | **0.321** | **0.372** | **27.8%** | **43.9%** | 54.5% |
| fusion | **0.727** | 0.314 | **0.410** | 26.8% | 43.1% | **57.4%** |

`jina − minilm = −0.026`, 95% CI [−0.058, +0.004], P(jina better) = 0.045 — MiniLM is
*ahead*, marginally significantly. `fusion − jina = +0.043`, CI [+0.027, +0.059],
P(>0) = 1.000.

The two rankers agree only moderately — Spearman **0.46** on 2143 live candidates,
0.54 on the validation set, 0.46 out-of-fold. Their top-50s over the live pool share
**9 of 50** posts. Two roughly equally accurate raters that disagree this much are the
textbook case where fusion wins, and it does.

## 3. The decisive test: live posts, independent judge

The top-50 by each ranker over the same 2143-candidate live pool, plus 50 random
controls (148 unique posts), labelled by the project's own Gemini validation prompt
(`annotate_validation_v2.py`, v2 axis) — which is independent of the local LM Studio
teacher these vectors were distilled from.

| selection | internal | external ↓ | graded (int 2 / neu 1 / ext 0) |
|---|---|---|---|
| Jina top-50 (today's criterion) | 42% | **2%** | 1.40 |
| GE MiniLM top-50 | **62%** | 12% | 1.50 |
| **fusion top-50** | **62%** | 6% | **1.56** |
| random from pool | 20% | 30% | 0.90 |

Bootstrap: `minilm − jina = +20.1pp` internal, 95% CI [+0.0, +38.0], P(>0) = 0.974;
`fusion − jina` the same, with a third of the external leak.

Prefix precision — Jina is **not** sharper at the top, which was the premise worth
testing:

| | @10 | @20 | @30 | @50 |
|---|---|---|---|---|
| Jina | int 50% / ext 0% | 50% / 0% | 40% / 3% | 42% / 2% |
| MiniLM | 60% / 10% | 65% / 15% | 63% / 13% | 62% / 12% |
| **fusion** | **70% / 0%** | **65% / 5%** | **60% / 3%** | **62% / 6%** |

Fusion dominates at every depth: MiniLM's hit rate with Jina's doomer-suppression.
MiniLM alone finds more internal posts but leaks 6× more external ones — on an axis
whose whole point is "internal high, external low", that leak is the reason to keep
Jina in the loop, not precision-at-the-top.

## 4. What does not help

- **A nonlinear head on the free embeddings.** MLP (128) and MLP (256, 64) on the band
  target, both spaces, are *worse* than Ridge everywhere (MiniLM AUC 0.735 / 0.715 vs
  0.795). 2822 rows with 385 positives will not feed a bigger head. "Train GE for
  maximum recall" cannot be bought with model capacity here.
- **Logistic regression on the binarised band.** Better recall@20% (56.9%) but ext-leak
  AUC explodes to 0.486 — it ranks external posts high too. Ridge on the graded target
  stays the right head.
- **A recall-then-precision cascade aimed at reproducing today's feed.** Keeping the
  Jina top-30 inside MiniLM's top-K needs K = 1000 of 2143 (47% of the pool) for 90%
  recall. But that framing takes Jina's picks as ground truth, and §3 shows they are
  the *worse* set. This number is a measure of disagreement, not of MiniLM's blindness.

## 5. Recommended architecture

Score everything free, pay Jina only for the shortlist, rank by the ensemble:

1. GE `candidates/generate` with `hydrate_embeddings: true` (no extra latency).
2. Existing free gates (`minWords: 6`, ~75% survive) — unchanged, and they already
   remove every candidate that lacks an embedding.
3. Score **100%** of survivors with a MiniLM-space vector. Free: 20 ms for 4000
   candidates measured in Node (base64 → Float32Array → cosine, ~5 µs each).
4. Keep the top **~20%** (or a fixed top 150) and send only those to Jina.
5. Final order = **mean of the two normalised ranks**, not either score alone.

On the CV corpus that prefilter preserves **90%** of the full-fusion top-30 at 20% keep,
and precision@30 is flat from 15% keep upward — the shortlist is not where quality is
lost.

Cost: today ~364 candidates are embedded per uncached request (~18k Jina tokens at the
measured ~49 tokens/post). Step 4 cuts that to ~150 → **≈60% fewer Jina tokens**, while
§3 says quality goes up rather than down. Free scoring also makes `fanout` affordable
for the first time — in the 6-call pool collected here, `random_posts` supplied 14 of
the MiniLM top-30 and 17 of the fusion top-30, so a wider pool is now worth its GE
latency rather than its embedding bill.

### Before shipping

- **Train and export a MiniLM vector properly.** `internal_vector_minilm.npy` in the
  repo is from 9 Aug on the older corpus. The one used here was trained on bonsai-v2
  through the unmodified harness; re-run it as a first-class artifact and
  `export_vector_json.py` it (384-d) alongside the Jina one.
- **Recalibrate the gate — it does not transfer** (see the 0.55 → 0.88 history). The
  MiniLM probe's live distribution is median 0.897, max 0.989; the thresholds are
  0.9356 for top-20%, 0.9425 for top-15%, 0.9497 for top-10%.
- **Two vectors now score one post**, in two different spaces. `ALL_SCORERS` and the
  `post_scores` (uri, version) cache assume one embedding per post fanned out to many
  vectors; the MiniLM score is free to recompute and need not be cached at all, while
  the Jina score still should be.
- **`candidate_score` for GE's reranker** currently carries the Jina probability. If the
  final order becomes the fusion rank, decide deliberately what goes in that field —
  or move the LoC feed to `rankBy: 'score'`, since the fusion *is* the editorial
  criterion.
- **Payload at high fanout**: 2 KB/candidate of base64. Fanout 8 × 1000 ≈ 8 MB of JSON
  to parse at the edge. Fine at 500–1000; measure before going past that.

### Caveats

One pool, one day, 50 posts per arm — the ±20pp live gap has a 95% CI whose lower
bound touches 0. The direction is consistent across three independent evaluations
(val set, out-of-fold CV, live adjudication), which is why the recommendation is to
ship the prefilter; the fusion-vs-MiniLM-alone choice rests mainly on the external-leak
column and is worth re-measuring on a second pool.

---

# 6. Would synthetic counterexamples close the gap?

## 6.1 The premise checked first: Jina has *not* had that benefit

`dataset_annotated_bonsai_v2.jsonl` contains **0** rows matching `#synth`, and no
`internal_vector_bonsai_v2_aug.npy` was ever produced — `run_all.sh` step 7 was
skipped for this corpus. The deployed `internal_vector.npy` is byte-identical to
`internal_vector_bonsai_v2.npy`, trained on that synth-free file.

Both vectors in §2–§3 were trained on the identical 2822 synth-free rows. So
augmentation does not explain why Jina ranks worse — **neither space has been
augmented yet**, and the headroom is open to both.

(The 441 rows in `dataset_synthetic_counterexamples.jsonl` are from 18 Aug, mined
against a *v3 Jina* vector on the older `lmstudio_soft` corpus. They were used for the
v4 pilot vectors — `internal_vector_band_synth.npy` — never for bonsai-v2.)

## 6.2 Do counterexamples transfer between spaces? No — and that is the good news

Those 441 rows are a natural experiment: they were mined against **Jina's** failures.
Adding them to both spaces separates "synthetic data helps" from "synthetic data helps
the space it was mined against". Their sources are disjoint from the bonsai-v2 corpus
(0 of 441 overlap), so nothing leaks.

Against the **independent Gemini validation labels** (the measurement that counts):

| space | synth | AUC internal | AP | ext-leak ↓ | P@25 |
|---|---|---|---|---|---|
| Jina | none | 0.772 | 0.281 | 0.289 | 36% |
| Jina | W=1 | **0.792** | **0.348** | 0.328 | **48%** |
| MiniLM | none | 0.795 | 0.348 | **0.264** | 36% |
| MiniLM | W=0.5 | **0.803** | 0.357 | 0.300 | 32% |
| MiniLM | W=1 | 0.792 | 0.353 | 0.325 | 28% |

Bootstrapped (51 internal / 254 external):

| | AUC internal Δ | ext-leak Δ |
|---|---|---|
| Jina + Jina-targeted synth | **+0.020**, CI [+0.000, +0.040], P(>0) = 0.978 | +0.038, P(worse) = **1.000** |
| MiniLM + Jina-targeted synth | −0.003, CI [−0.041, +0.036], P(>0) = 0.453 | +0.061, P(worse) = **1.000** |

**Counterexamples pay off in the space they were mined against and nowhere else.**
Jina-targeted rows buy Jina a real +0.020 and buy MiniLM exactly nothing. That is the
strongest available evidence *for* the proposal: MiniLM-targeted rows are a different
set of rows, and MiniLM's own +0.02-ish is still unclaimed.

Two warnings the same table carries:

- **Both spaces get worse at suppressing external posts, with certainty**
  (P(worse) = 1.000, +0.038 and +0.061). The synth pool is skewed: mean target score
  3.43 vs 1.92 for real rows, 186 internal / 211 external / 44 neutral teacher labels
  against a real base rate near 13% positive. In §3 the external column is exactly what
  made fusion preferable to MiniLM alone (6% vs 12% leak) — running the generator
  as-is could spend the win it is meant to protect. Select on `val_ext_leak_auc` and
  `val_ndcg_graded`, not on AUC, and prefer the lower synth weights: **W = 0.5 was the
  only setting that improved MiniLM at all**, and W ≥ 1 made it worse.
- **Teacher-fidelity CV disagrees and should not be believed here.** Out-of-fold CV
  against the teacher's own band label says synth helps MiniLM *more* (+0.044 vs
  +0.033, both P(>0) = 1.000). But the synth rows carry that same teacher's labels, so
  that metric partly rewards learning the teacher's boundary rather than the axis.
  Where the two disagree, the independent Gemini labels win. (ρ vs teacher also falls
  monotonically as W rises — 0.372 → 0.272 for MiniLM — while AUC rises: the model is
  trading fidelity for boundary sharpness.)

## 6.3 Sharpening does not erode the ensemble

The obvious worry — that making the two spaces individually better makes them agree,
destroying the diversity fusion feeds on — does not happen. Augmenting both *lowers*
their correlation and raises the fused score:

| augmented | ρ(jina, minilm) | fusion AUC (CV) | fusion AUC (val) |
|---|---|---|---|
| neither | 0.463 | 0.727 | 0.821 |
| Jina only | 0.435 | 0.747 | — |
| both | **0.410** | **0.773** | **0.838** |

Each space is being sharpened on its *own* failures, which pushes them apart, not
together. The fusion architecture in §5 survives augmentation.

## 6.4 How to run it

`create_synthetic_counterexamples.py` is already space-agnostic — it reads
`LOC_EMB_CACHE` + `LOC_VECTOR`, ranks lures/misses by percentile disagreement with the
teacher, and never assumes a dimension. Two repo changes were needed to make the
MiniLM side reproducible, both in place:

- **`embed_minilm.py`** (new) — builds train/interaction/validation/synth caches in
  GE's exact space (all-MiniLM-L12-v2, mean-pooled, **128 tokens**). Needed because
  `train_internal_vector.py` falls back to the *Jina* model whenever a cache is missing.
- **`train_internal_vector.py`** — `CACHE_INTERACTIONS_EMBS` was the one cache path not
  env-configurable (`LOC_INT_EMB_CACHE` now), so the 322 interaction rows silently
  arrived as 1024-dim Jina vectors in any other space. The `1024-dim` in its final log
  line is now the actual dimension.

Staged and verified (reproduces val AUC 0.795 / ext-leak 0.264 from the project root):
`embeddings_bonsai_v2_minilm.npz`, `embeddings_interactions_minilm.npz`,
`embeddings_gemini_minilm_v2.npz`, `embeddings_synth_minilm.npz`,
`internal_vector_bonsai_v2_minilm.npy` (+ meta, results).

**Blocked on one thing**: the generator re-annotates each rewrite with the local
llama.cpp teacher at `LOC_API_BASE` (default `127.0.0.1:8080/v1`, model
`meta/muse-glimmer`). Nothing is serving it — port 8080 currently holds an unrelated
`node` process, and LM Studio is not up on 1234 either. Start the teacher, then:

```sh
# 1. mine MiniLM's own failures and rewrite them (200 Gemini calls; free tier is
#    500/day and 148 were used for the §3 adjudication)
LOC_PROMPT_VERSION=v2 \
LOC_TEACHER_FILE=dataset_annotated_bonsai_v2.jsonl \
LOC_EMB_CACHE=embeddings_bonsai_v2_minilm.npz \
LOC_VECTOR=internal_vector_bonsai_v2_minilm.npy \
LOC_SYNTH_OUT=dataset_synth_minilm.jsonl \
LOC_SYNTH_PER_SIDE=100 \
  uv run --with numpy --with openai --with google-genai --with python-dotenv --with tqdm \
  python create_synthetic_counterexamples.py

# 2. embed the counterexamples in the same space
LOC_EMBED_ONLY=synth:dataset_synth_minilm.jsonl LOC_EMB_CACHE=embeddings_synth_minilm_new.npz \
  uv run --with numpy --with torch --with transformers python embed_minilm.py

# 3. retrain on corpus + counterexamples (cache must be rebuilt over the concatenation,
#    row-for-row — the harness asserts on the count)
cat dataset_annotated_bonsai_v2.jsonl dataset_synth_minilm.jsonl > dataset_bonsai_v2_minilm_aug.jsonl
LOC_TEACHER_FILE=dataset_bonsai_v2_minilm_aug.jsonl LOC_EMB_CACHE=embeddings_bonsai_v2_minilm_aug.npz \
  uv run --with numpy --with torch --with transformers python embed_minilm.py
LOC_PROMPT_VERSION=v2 LOC_TARGET_MODE=band \
LOC_TEACHER_FILE=dataset_bonsai_v2_minilm_aug.jsonl \
LOC_VAL_LABELS_FILE=dataset_validation_gemini_v2.jsonl \
LOC_EMB_CACHE=embeddings_bonsai_v2_minilm_aug.npz \
LOC_INT_EMB_CACHE=embeddings_interactions_minilm.npz \
LOC_VAL_EMB_CACHE=embeddings_gemini_minilm_v2.npz \
LOC_VECTOR_OUT=internal_vector_bonsai_v2_minilm_aug.npy \
  uv run --with numpy --with scipy --with scikit-learn --with torch --with transformers \
  --with tqdm python train_internal_vector.py
```

The harness sweeps the synth weight (`SYNTH_WEIGHTS`) and selects on
`val_ndcg_graded`, which already prices in external leak — but check
`val_ext_leak_auc` against the 0.264 baseline by hand before shipping, because §6.2
says that is the number augmentation puts at risk.

Expected payoff, from the transfer experiment: **+0.02 AUC internal**, the same order
Jina got from its own targeted rows. That would put MiniLM at ~0.815 alone and fusion
near 0.85 — worth doing, and worth re-running the live adjudication in §3 afterwards
rather than trusting the val set's 51 positives.

---

# 7. Result: MiniLM-targeted counterexamples did not pay off

The §6.4 pipeline was run. **199 of 200 counterexamples** generated against
`internal_vector_bonsai_v2_minilm.npy` (100 lures + 100 misses; one row lost to a
persistent Gemini 503). Teacher agrees with the rewrite target on 168/199 (84%),
comparable to the Jina-targeted set. Artifacts: `dataset_synth_minilm.jsonl`,
`embeddings_synth_minilm_new.npz`, `internal_vector_bonsai_v2_minilm_aug.npy`.

**The predicted +0.02 AUC did not materialise.** On the independent Gemini labels:

| counterexamples | W | AUC internal | AP | ext-leak ↓ | P@25 |
|---|---|---|---|---|---|
| none | — | 0.795 | 0.348 | 0.264 | 36% |
| MiniLM-targeted | 0.5 | 0.796 | 0.345 | **0.258** | 36% |
| MiniLM-targeted | 1 | 0.795 | 0.338 | **0.256** | 36% |
| MiniLM-targeted | 2 | 0.791 | 0.336 | 0.256 | 36% |
| Jina-targeted (for contrast) | 0.5 | 0.803 | 0.357 | 0.300 | 32% |

Bootstrapped: AUC internal **+0.001**, CI [−0.007, +0.009] — flat. The only real
effect is a small *improvement* in external suppression: **−0.006**, CI [−0.012,
−0.001], P(worse) = 0.011. That is the opposite sign from the Jina-targeted set's
+0.036, so the warning in §6.2 did not bite — but neither did the payoff.

## 7.1 Why: the lures have no agent to make agentic

All **31** disagreements land on *neutral*; not one lands on the opposite pole. They
are lopsided (26 of 31 on the lure side), and they are the rewrites that changed their
source **least** — cosine to source 0.836 vs 0.748 for the ones that landed.

Reading them explains it. MiniLM's lures are overwhelmingly news and product
announcements carrying agentic *vocabulary* with no agent behind it:

| source | rewrite | teacher |
|---|---|---|
| "'Talk to Spotify' lets you create playlists … voice commands" | "…we can now use the new beta feature to curate our own vibes" | neutral |
| "Canva Code 2.0 adds visual editing, HTML imports…" | "…—time to finally streamline our workflow and build faster" | neutral |
| "The EU and Albania have provisionally closed the first negotiation chapters" | "We've officially closed the first negotiation chapters…" | neutral |

That *is* the vector's real failure mode — it keys on words like build, create,
control, milestone — but rewriting cannot fix it, because a Spotify press release has
no first person to give agency to. Gemini can only paint on a "we can now…" veneer,
which is exactly the stock-phrase padding the rewrite prompt forbids, and the teacher
correctly refuses to call it internal. Consistent with that, **the rewrite moved the
vector's score in the intended direction only 60% of the time** — barely above chance.

## 7.2 Hard-example upweighting doesn't work either

If the lures can't be rewritten, the cheap alternative is to weight them: those 193
posts already carry teacher labels, so it costs no Gemini and no teacher call.

| weighting | AUC internal | ext-leak ↓ | vs baseline |
|---|---|---|---|
| baseline | 0.795 | 0.264 | — |
| hard × 2 | 0.788 | **0.252** | AUC −0.007 (ns), leak −0.012 (P = 0.003) |
| hard × 3 | 0.776 | **0.247** | AUC −0.019, leak −0.017 |
| hard × 5 | 0.734 | 0.257 | AUC −0.062 |
| hard × 10 | 0.622 | 0.336 | AUC −0.172 |

The same trade as §7: a small external-leak gain, no internal-detection gain, and
sharp collapse when pushed. Two independent interventions reaching the same trade is
the informative part.

## 7.3 What this means

The §6.2 inference — "Jina-targeted rows bought Jina +0.020, so MiniLM's own +0.02 is
unclaimed" — **was wrong**, and the reason is visible in the starting points. Jina
began at 0.772 with cruder, more learnable errors; MiniLM began at 0.795, and its
top-100 residual disagreements are mostly teacher noise and genuinely ambiguous posts
rather than a systematic pattern a linear probe can absorb. Transfer *failure* between
spaces (§6.2) established that targeting matters; it did not establish that every
space has the same headroom, and MiniLM's is smaller.

**Recommendation unchanged from §5, with one simplification: ship the baseline
MiniLM vector.** `internal_vector_bonsai_v2_minilm.npy` (val AUC 0.795, ext-leak
0.264) is within noise of every augmented variant. If the external column is worth
protecting — and §3 says it is, since leak is the only place fusion beat MiniLM alone —
the augmented vector at W = 0.5 (0.796 / 0.258) is a defensible alternative at the cost
of an extra dataset dependency.

**Do not spend more Gemini quota on counterexamples for this axis in this space.** If
more MiniLM headroom is wanted, the binding constraint is real labelled data — 2822
rows with 385 positives — not synthetic rewrites.

---

# 8. When is counterexample generation worth it?

Two known outcomes to calibrate against: Jina (baseline val AUC 0.772) gained **+0.020**
from its targeted rows; MiniLM (baseline 0.795) gained **+0.001** from its own. Three
candidate pre-flight signals were tested against that. **All three failed.**

| candidate signal | Jina (paid off) | MiniLM (didn't) | discriminates? |
|---|---|---|---|
| free dry run: upweight the mined hard posts ×2 | ΔAUC −0.003 [−0.013, +0.008] | −0.007 [−0.018, +0.004] | **no** — predicted failure for both |
| agent pronouns in the mined lures | 54% | 47% | **no** (corpus base rate 31%) |[^1]

[^1]: Those two percentages are comparable to each other but not portable: they were
measured over lure sets mined by freshly refit probes, and the pre-flight in
`create_synthetic_counterexamples.py` scores the same MiniLM lure set at 34% because it
mines with the *saved* vector and uses a narrower pronoun inventory. The absolute number
moves with both; the conclusion — that it does not discriminate — does not.
| hard posts shared with the other space | 29% lures / 33% misses | same, symmetric | **no** |

And the obvious candidate — "did the rewrites land?" — is ruled out by the run itself:
the MiniLM batch landed **84%** (168/199), the same yield as the Jina batch, and still
produced nothing. **Rewrite quality was not the limiter; headroom was.**

So there is no validated cheap predictor. What survives is weaker but honest:

**1. Headroom is the only thing that separated the two cases.** The intervention paid
where the probe was underperforming (0.772) and not where it was already near what the
corpus supports (0.795). With n = 2 that is an observation, not a threshold — but it is
the right question to ask first: *is this probe visibly worse than a sibling probe on
the same data?* If nothing suggests it is, expect nothing.

**2. Look at 20 mined hard posts before spending the quota.** This is judgment, but the
mechanism behind it is measured (§7.1). Ask of each: *could this post plausibly have
been written the other way?*

- Personal narrative, first-person, a decision or a reaction → rewritable. Measured
  land rate on the lure side **85%** with an agent pronoun, **95%** on the miss side.
- Headline, product announcement, link-out, third-party report → **nothing to flip**.
  Land rate **68%**, and the failures come back labelled *neutral*: near-duplicates of
  their source that dilute rather than teach.

A lure set dominated by the second kind means the probe is making a *category* error
(scoring agentic vocabulary in agent-free text), and the fix is more real negatives of
that category, not rewrites of them.

**3. Never reuse another space's or axis's counterexamples.** §6.2: transfer is exactly
zero (−0.003, CI spanning 0) while the external-leak cost transfers in full.

**4. Check the external column, not just AUC.** Every counterexample run in this report
moved external suppression, twice for the worse and significantly (+0.036, +0.061,
P(worse) = 1.000). On an axis defined as "positive high, negative low", that can cost
more than the AUC gains.

**5. Order of operations.** The whole observed upside of counterexample generation was
+0.020 AUC for ~90 minutes and ~200 Gemini calls. Rank fusion of two embedding spaces
was worth **+0.03** (§2) for one extra API call on a shortlist. Counterexamples are the
last lever to reach for, not the first — after ensembling, and after more real labelled
data (this corpus is 2822 rows carrying 385 positives).

If the decision is still genuinely open after 1–4, the generator is resumable and its
cost is linear, so a ~50-row pilot costs ~25 minutes. Note the asymmetry when reading
one: a pilot that shows external leak degrading is a reliable *stop* signal, but a null
result at 50 rows is weak evidence against an effect that took 441 rows to reach +0.020.
