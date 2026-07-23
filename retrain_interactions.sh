#!/bin/bash
# Script to resolve user interactions and retrain the vector

set -e
cd "$(dirname "$0")"

: ${LOC_PROMPT_VERSION:=v2}
: ${TAG:=bonsai_v2}
: ${HF_HOME:=/Volumes/Daten/huggingface}
export LOC_PROMPT_VERSION HF_HOME

TEACHER_FILE=dataset_annotated_${TAG}.jsonl
VAL_FILE=dataset_validation_gemini_${LOC_PROMPT_VERSION}.jsonl
EMB_CACHE=embeddings_${TAG}_jina.npz
VECTOR=internal_vector_${TAG}.npy

PY="uv run --with numpy --with scipy --with scikit-learn --with torch --with transformers --with einops --with datasets --with accelerate --with openai --with google-genai --with python-dotenv --with tqdm python"

echo "=== [1/4] Fetching latest interactions from edge DB ==="
(cd bunny_feed && npx tsx dump_interactions.ts)

echo "=== [2/4] Resolving user interaction posts ==="
$PY resolve_interaction_posts.py

echo "=== [3/4] Retraining internal vector with interactions ==="
LOC_TEACHER_FILE=$TEACHER_FILE LOC_EMB_CACHE=$EMB_CACHE \
LOC_VAL_LABELS_FILE=$VAL_FILE LOC_VECTOR_OUT=$VECTOR \
  $PY train_internal_vector.py

echo "=== [4/4] Export deployable artifacts ==="
cp $VECTOR internal_vector.npy
cp ${VECTOR%.npy}_meta.json internal_vector_meta.json
$PY export_vector_json.py

echo "Done. Vector retrained on latest interactions."
