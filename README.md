# Locus of Control — Bluesky Feed Classifier

Classifies Bluesky posts by the **Locus of Control** (LoC) they implicitly foster
in a reader — `internal` (agency, effort, learnable skill, constructive ownership),
`external` (helplessness, blame of distant powers, negative news without solutions),
or `neutral` (no control/agency framing) — and serves feeds ranked by those scores.

Two models are distilled from the same LLM teacher labels:

1. **Internal query vector** — a single 1024-dim vector in Jina embedding space;
   cosine similarity ranks posts by "internalness". Cheap enough to run anywhere,
   including a CDN edge worker.
2. **Transformer classifier** — xlm-roberta fine-tuned on soft labels (3-class).

Current numbers (500 Gemini-validated posts, 10.3% internal): vector AUC 0.787
(above its own 27B teacher at 0.751), transformer kappa 0.505 vs teacher 0.539.

## One-shot pipeline

```bash
./run_all.sh
```

runs everything: fetch network → stream posts → teacher labels → validation
labels → vector + transformer training → synthetic counterexamples → augmented
retrain → deployable artifacts. Every step is resumable — rerunning the script
skips finished work. Requirements:

- **LM Studio** running on `localhost:1234` with the teacher model loaded
  (default `ternary-bonsai-27b-mlx`).
- **`.env`**: `BSKY_HANDLE`/`BSKY_APP_PASSWORD` (data collection),
  `GAI_API` (Gemini: validation labels + counterexample rewrites; free tier is
  500 requests/day — the validation step alone needs ~500, so plan on separate
  days or a paid key).
- `HF_HOME` pointing at your HuggingFace cache (Jina embedding model).

`SKIP_SYNTH=1 ./run_all.sh` skips the counterexample stage.

## Reproducing with a different prompt

The construct definition lives entirely in **`label_prompts.py`** — every
annotation script imports from it, and the version is selected by env var, so a
new construct is a three-step change:

1. Add your definitions in `label_prompts.py` (copy the `DEFINITIONS_V2` block
   to `DEFINITIONS_V3`, build `TEACHER_PROMPT_V3` / `VALIDATION_PROMPT_V3`,
   extend `get_teacher_prompt()`/`get_validation_prompt()`). Keep the two rules
   that make the pipeline work: the teacher must end with `Label: X` where X is
   a **single letter** (single-token answers keep all classes visible in
   `top_logprobs` → real soft labels), and teacher + validator must use the
   **same definitions** (construct drift between them looks like model error
   and caps your ceiling — we learned this the hard way).
2. `export LOC_PROMPT_VERSION=v3 TAG=mymodel_v3`
3. `./run_all.sh`

All artifacts are tagged (`dataset_annotated_${TAG}.jsonl`,
`internal_vector_${TAG}.npy`, `loc-classifier-${TAG}/`), so versions coexist.

Key tuning knobs (env): `LOC_TEACHER_MODEL` (any LM Studio model; add a
brevity instruction to the prompt for reasoning models or they overthink),
`LOC_TEMPERATURE` (0 = stable labels; the pipeline falls back to 0.7 for posts
that deterministically fail), `LOC_TOP_LOGPROBS` (MLX engines cap at 10),
`LOC_SYNTH_PER_SIDE` (counterexamples per boundary side).

## Pipeline stages (scripts)

| Stage | Script | Notes |
|---|---|---|
| Fetch follow network | `1_fetch_network.py` | → `network_dids.json` |
| Stream posts | `2_stream_posts.py` | → `dataset_raw.jsonl` |
| Teacher soft labels | `annotate_lmstudio_soft_labels.py` | logprob-based soft labels via LM Studio; env-configurable (model, prompt version, output, sampling) |
| Validation set | `create_validation_set.py` + `annotate_validation_v2.py` | independent Gemini labels; validation URIs are excluded from all training |
| Vector training | `train_internal_vector.py` | binary soft-label logistic regression on Jina embeddings; sweeps softmax-T × L2, selects by held-out teacher-ranking fidelity; computes/caches embeddings itself; reports AUC/AP/P@k vs validation |
| Transformer training | `train_transformer_soft.py` | xlm-roberta, soft cross-entropy on temperature-scaled logprobs |
| Counterexamples | `create_synthetic_counterexamples.py` | mines posts the vector misranks vs the teacher, Gemini rewrites them to the opposite LoC framing (same topic/language/register), teacher labels the rewrites |
| Comparison | `compare_models_v2.py` | vector vs transformer vs teacher, ranking + classification |
| Deploy export | `export_vector_json.py` | vector + calibration → `bunny_feed/src/vector.json` |

Label schema per post: `soft_labels` `[P(internal), P(external), P(neutral)]`,
`raw_logprobs` (re-scalable with any temperature later), `loc_label` (argmax).

## Serving

**`backend/`** — FastAPI SSE server: streams a Bluesky feed and scores every
post with both models. `GET /api/posts?method=classifier|vector|both`.
Env: `LOC_CLASSIFIER_PATH` (default `../loc-classifier-bonsai-v2/final_model`),
`VECTOR_PROB_THRESHOLD`, Bluesky creds, `HF_HOME`.

```bash
cd backend
HF_HOME=/Volumes/Daten/huggingface uv run --with-requirements requirements.txt \
  uvicorn main:app --reload
```

**`frontend/`** — React app (`npm run dev`): toggle Classifier/Vector scoring
on the same feed items.

**`bunny_feed/`** — Bunny Edge Script implementing an ATProto
`getFeedSkeleton` feed generator. Embeds candidate posts via the hosted Jina
API and scores them with the exported vector (`src/vector.json`, written by
`export_vector_json.py` — never edit by hand). Needs `JINA_API_KEY` in the
edge environment. ⚠️ Training embeds with the *local* Jina model; verify
API-vs-local score parity on a few posts before trusting edge rankings.

## Repo layout

- `label_prompts.py` — all prompt versions, byte-exact history (v1 teacher and
  v1 validator differed; v2 unified them).
- `internal_vector.npy` / `internal_vector_meta.json` — deployed vector
  (committed; calibration: `p = sigmoid(scale·cos + bias)`).
- `dataset_*.jsonl`, `embeddings_*.npz`, `loc-classifier-*/` — regenerable,
  git-ignored.
- `backup/` — archived experiments and superseded scripts (git-ignored).
