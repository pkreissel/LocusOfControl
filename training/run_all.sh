#!/bin/zsh
# End-to-end pipeline: raw Bluesky posts -> TypeSafe teacher labels -> Gemini
# validation labels -> Jina + MiniLM vectors -> the two JSONs bunny_feed reads.
#
# Every step is resumable; rerunning skips completed work. Paths resolve against
# training/data/ inside the Python, so this script's cwd only has to be stable,
# not any particular directory.
#
# Deploy is NOT part of this script: commit bunny_feed/src/*.json and push to
# main. Recalibrate `threshold` in bunny_feed/src/feeds.ts first — it never
# survives a retrain (see CLAUDE.md).
set -e
cd "$(dirname "$0")"

: ${LOC_PROMPT_VERSION:=v2}          # which rubric in label_prompts.py
: ${TAG:=typesafe_g3g}               # artifact name tag
: ${HF_HOME:=/Volumes/Daten/huggingface}
export LOC_PROMPT_VERSION HF_HOME

RAW=data/dataset_raw.jsonl
SCORES=data/scores_loc_typesafe_${TAG}.jsonl
TEACHER=data/dataset_annotated_${TAG}.jsonl
VAL=data/dataset_validation_gemini.jsonl
PY=${PY:-../tmp_venv/bin/python}

echo "=== [1/7] Follow network (BSKY_HANDLE/BSKY_APP_PASSWORD in .env) ==="
[ -f data/network_dids.json ] || $PY 1_fetch_network.py

echo "=== [2/7] Stream posts ==="
[ -f $RAW ] || $PY 2_stream_posts.py

echo "=== [3/7] Teacher labels (TypeSafe; TYPESAFE_API_KEY in .env) ==="
# One request per post carries every question in ts_questions.py, because
# TypeSafe bills the state once per call and scores each question against it in
# isolation. Adding an axis here is close to free.
[ -f $SCORES ] || TYPESAFE_WORKERS=${TYPESAFE_WORKERS:-32} $PY score_loc_typesafe.py $RAW $SCORES
# Rewrites the scores into the teacher schema the trainer reads, preserving row
# order and count so the positional embedding caches stay aligned.
[ -f $TEACHER ] || $PY build_typesafe_teacher.py $RAW $SCORES $TEACHER graded

echo "=== [4/7] Validation labels (Gemini, GAI_API in .env; free tier 500/day) ==="
[ -f $VAL ] || $PY create_validation_set.py
[ -f data/dataset_validation_gemini_v2.jsonl ] || $PY annotate_validation_v2.py

echo "=== [5/7] MiniLM embedding caches ==="
# Must be pre-built: train_internal_vector.py computes *Jina* embeddings for any
# cache it finds missing, which would fill a MiniLM cache with 1024-dim vectors.
[ -f data/embeddings_lmstudio_soft_minilm.npz ] || LOC_TEACHER_FILE=$TEACHER $PY embed_minilm.py

echo "=== [6/7] Train both vectors ==="
# The two vectors are selected on DIFFERENT metrics because they do different
# jobs. Jina is the quality gate, judged on the head of the ranking. MiniLM is a
# prefilter whose only job is to not lose candidates before the paid stage,
# judged on recall into the shortlist at the production cut (see PREFILTER_CUT).
LOC_TEACHER_FILE=$TEACHER LOC_SELECT_METRIC=val_precision@50 $PY train_internal_vector.py

LOC_TEACHER_FILE=$TEACHER \
LOC_EMB_CACHE=data/embeddings_lmstudio_soft_minilm.npz \
LOC_INT_EMB_CACHE=data/embeddings_interactions_minilm.npz \
LOC_VAL_EMB_CACHE=data/embeddings_gemini_minilm_v2.npz \
LOC_VECTOR_OUT=data/internal_vector_minilm.npy \
LOC_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L12-v2 \
LOC_EMBEDDING_TASK=none \
LOC_SELECT_METRIC=val_recall@cut \
  $PY train_internal_vector.py

echo "=== [7/7] Export for the edge ==="
$PY export_vector_json.py
LOC_VECTOR=data/internal_vector_minilm.npy \
LOC_VECTOR_JSON_OUT=../bunny_feed/src/vector_minilm.json \
  $PY export_vector_json.py

echo
echo "Done. bunny_feed/src/vector.json and vector_minilm.json updated."
echo "NEXT: recalibrate `threshold` in bunny_feed/src/feeds.ts against the new"
echo "score distribution, then commit both and push to main to deploy."
