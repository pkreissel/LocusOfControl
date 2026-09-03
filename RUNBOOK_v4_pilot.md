# Overnight runbook — v4 graded-score pilot (agent instructions)

You are an autonomous agent running an overnight experiment. **Goal:** test whether a
genuinely graded teacher signal (`loc_score`, 0–9) lets the production **vector** rank
locus-of-control better than the current one-hot-derived target — cheaply, on a ~1500-post
pilot. Work the numbered phases in order. **Do not deploy anything, do not commit, do not
overwrite production artifacts.** Leave results + a report for the human to review.

Working dir: `/Volumes/Daten/Datenprojekte/LocusOfControl`. Everything runs locally
(local teacher LLM + local Jina embeddings). Code changes for this experiment are already
in place (`label_prompts.py` v4, `annotate_lmstudio_soft_labels.py` loc_score,
`train_internal_vector.py` graded target + `LOC_USE_LOC_SCORE` switch).

## Hard rules (safety)
- **Never** set `LOC_VECTOR_OUT=internal_vector.npy` and **never** run `export_vector_json.py`
  or touch `bunny_feed/src/vector.json`. Those are the deployed artifacts. Write only to new
  filenames (e.g. `internal_vector_v4.npy`).
- **Never** annotate into `dataset_annotated_lmstudio_soft.jsonl`. Use a new output file.
- If a precondition fails (Phase 0), **stop and write the report** — do not burn the night
  retrying a dead server.
- Reasonable oversight cadence: poll long-running jobs every ~10–20 min (don't busy-wait).
  Total wall time expected: annotation is the long pole; vector training is minutes.

## Baseline to beat (current deployed vector, from `results_internal_vector.json`)
Config `ridge_l20.1_T5.0_W2.0`:
- **val_auc_internal = 0.760**, **val_avg_precision = 0.452** (Gemini gold — the headline gate)
- val_precision@10/25/50 = 0.80 / 0.60 / 0.52 (treat as NOISY — small val set)
- val_spearman_vs_teacher = 0.077, dev_spearman_vs_teacher = 0.141

---

## Phase 0 — Preconditions (stop if any fail)
1. Teacher LLM is up and OpenAI-compatible with logprobs:
   ```
   curl -s http://localhost:8080/v1/models
   ```
   Must return JSON listing the model (default `meta/muse-glimmer`). If it returns HTML/an
   error, the right server is NOT running — **stop and report** (as of writing this, port 8080
   was serving a non-API error page).
2. Confirm `uv` works: `uv --version`.
3. Confirm gold + embedding caches exist: `dataset_validation_gemini.jsonl`,
   `embeddings_gemini_jina.npz`.

## Phase 1 — Build the 1500-post pilot subset
Deterministic sample from `dataset_raw.jsonl` (never from an annotated file):
```
uv run --with-none python3 - <<'PY'
import json, random
rows=[json.loads(l) for l in open("dataset_raw.jsonl") if l.strip()]
random.seed(42); random.shuffle(rows)
sub=rows[:1500]
with open("pilot_raw_1500.jsonl","w") as f:
    for r in sub: f.write(json.dumps(r)+"\n")
print("wrote", len(sub), "posts")
PY
```
(If `uv run --with-none` errors, use plain `python3` — this snippet has no deps.)

## Phase 2 — Smoke test the annotator (5 posts) before the full run
```
head -n 5 pilot_raw_1500.jsonl > pilot_smoke.jsonl
LOC_PROMPT_VERSION=v4 \
LOC_INPUT_FILE=pilot_smoke.jsonl \
LOC_OUTPUT_FILE=pilot_smoke_v4.jsonl \
LOC_TEACHER_MODEL=meta/muse-glimmer \
LOC_TOP_LOGPROBS=20 \
LOC_MAX_CONCURRENT=4 \
uv run --with openai --with numpy --with tqdm python annotate_lmstudio_soft_labels.py
```
Then verify every smoke row has a real `loc_score` and a spread digit distribution:
```
uv run --with numpy python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("pilot_smoke_v4.jsonl") if l.strip()]
ok=[r for r in rows if r.get("loc_score") is not None]
print(f"{len(ok)}/{len(rows)} rows have loc_score")
for r in ok:
    lp=r.get("loc_score_logprobs") or {}
    nz={k:round(v,2) for k,v in lp.items() if v>-20}
    print(r.get("loc_label"), "score=", round(r["loc_score"],2), "digits=", nz)
PY
```
**Gate:** loc_score present for all 5 AND at least some rows show >1 non-floor digit (real mass
on neighbors). If loc_score is None everywhere or every row is a single hard digit → the server
isn't returning digit logprobs or isn't following v4. Stop and report.

## Phase 3 — Full pilot annotation (background + monitor)
```
LOC_PROMPT_VERSION=v4 \
LOC_INPUT_FILE=pilot_raw_1500.jsonl \
LOC_OUTPUT_FILE=dataset_annotated_pilot_v4.jsonl \
LOC_TEACHER_MODEL=meta/muse-glimmer \
LOC_TOP_LOGPROBS=20 \
LOC_MAX_CONCURRENT=4 \
uv run --with openai --with numpy --with tqdm python annotate_lmstudio_soft_labels.py
```
Run in the background. The script appends and resumes by URI, so it's safe to restart if it
dies. Monitor progress: `wc -l dataset_annotated_pilot_v4.jsonl` (target ~1500; some posts may
be skipped on errors). Consider it done when the count plateaus and the process exits.
**If <80% of 1500 rows land with non-null loc_score**, investigate (server logprobs / model)
before training — a degenerate file will just reproduce the current problem.

## Phase 4 — Pre-train signal gate (cheap, no gold needed)
```
uv run --with numpy python3 - <<'PY'
import json, numpy as np
rows=[json.loads(l) for l in open("dataset_annotated_pilot_v4.jsonl") if l.strip()]
s=np.array([r["loc_score"] for r in rows if r.get("loc_score") is not None])
print(f"n_with_score={len(s)}  mean={s.mean():.2f}  std={s.std():.2f}  distinct(round1)={len(set(np.round(s,1)))}")
for lab in ["internal","external","neutral"]:
    v=np.array([r["loc_score"] for r in rows if r.get("loc_label")==lab and r.get("loc_score") is not None])
    if len(v): print(f"  {lab:8s} n={len(v):4d}  mean_loc_score={v.mean():.2f}")
PY
```
**Gate (all should hold):** std clearly > 0 (target ≳ 1.5), many distinct values, and the
ordering **mean(internal) > mean(neutral) > mean(external)** with a real gap. If the buckets
don't separate, the score isn't tracking LoC — record it and stop before training.

## Phase 5 — Train vectors (A = graded, B = legacy control) on the SAME file
Run A first (it builds the embedding cache); B reuses it (identical embeddings), isolating the
effect of the target from data size. **Both write to NEW filenames.**

A — graded loc_score (the hypothesis):
```
LOC_TEACHER_FILE=dataset_annotated_pilot_v4.jsonl \
LOC_EMB_CACHE=embeddings_pilot_v4_jina.npz \
LOC_VECTOR_OUT=internal_vector_v4.npy \
LOC_USE_LOC_SCORE=1 \
uv run --with numpy --with scipy --with scikit-learn --with torch --with transformers \
   --with tqdm --with einops python train_internal_vector.py 2>&1 | tee train_v4_A.log
```
B — legacy target on the same 1500 posts (control):
```
LOC_TEACHER_FILE=dataset_annotated_pilot_v4.jsonl \
LOC_EMB_CACHE=embeddings_pilot_v4_jina.npz \
LOC_VECTOR_OUT=internal_vector_ctrl.npy \
LOC_USE_LOC_SCORE=0 \
uv run --with numpy --with scipy --with scikit-learn --with torch --with transformers \
   --with tqdm --with einops python train_internal_vector.py 2>&1 | tee train_v4_B.log
```
Each prints a "Final model" block and writes `results_internal_vector_v4.json` /
`results_internal_vector_ctrl.json`. If embedding compute fails on GPU, it falls back to CPU;
that's fine (1500 embeddings). If `einops` isn't needed by the model it's harmless.

## Phase 6 — Decision
Compare **val_auc_internal** and **val_avg_precision** (Gemini gold — target-independent, the
only clean cross-run metric) for: baseline (0.760 / 0.452) vs B vs A.
- **Greenlight (graded signal works):** A ≥ baseline on AUC/AP **and** A > B. → Recommend
  scaling annotation to the full set and, optionally, running `sweep_jina_reranker.py`.
- **Data-size, not signal:** A ≈ B ≈ baseline → the target change didn't help at 1500;
  recommend investigating the prompt/scale (Phase 4 buckets) before spending more budget.
- **Regression:** A < baseline → do not proceed; capture the numbers and the Phase-4 stats.
Ignore P@10/P@25 swings for the decision (noisy small val set); mention them only as color.

## Phase 7 — (Optional, only if Phase 6 greenlights AND time remains) reranker
```
LOC_TEACHER_FILE=dataset_annotated_pilot_v4.jsonl \
LOC_OUTPUT_DIR=./loc-reranker-pilot-v4 \
uv run --with torch --with transformers --with datasets --with scikit-learn \
   --with accelerate --with einops python train_jina_reranker.py 2>&1 | tee train_reranker_v4.log
```
Note: the script header lists `--with flash_attn`; on this Mac flash-attn usually won't build —
omit it (done above). If the model hard-requires it and errors, skip the reranker and note it;
the vector is the priority and its result stands alone.

## Phase 8 — Report (always do this, even on early stop)
Write `REPORT_v4_pilot.md` containing:
- Which phase you reached and any precondition/stop reason.
- Phase-4 signal stats (std, distinct, per-label means).
- A table: metric × {baseline, B legacy@1500, A graded@1500} for val_auc_internal,
  val_avg_precision, dev_spearman_vs_teacher (+ P@k as noisy footnote).
- Your Phase-6 verdict and a one-line recommendation.
- List of new files created (so nothing is mistaken for a deployed artifact).
Do not commit, deploy, or overwrite `internal_vector.npy` / `vector.json`. Leave everything for
human review in the morning.
```
```
