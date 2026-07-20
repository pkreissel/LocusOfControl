#!/bin/zsh
# End-to-end pipeline: raw Bluesky posts -> teacher labels -> validation labels
# -> internal vector + transformer -> synthetic counterexamples -> augmented
# retrain -> comparison -> deployable artifacts.
#
# Every step is resumable; rerunning skips completed work. Steps that need an
# external service state it. Configure via env (defaults reproduce the current
# bonsai-v2 setup); see README "Reproducing with a different prompt".
set -e
cd "$(dirname "$0")"

: ${LOC_PROMPT_VERSION:=v2}              # prompt version from label_prompts.py
: ${LOC_TEACHER_MODEL:=ternary-bonsai-27b-mlx}  # LM Studio model id
: ${LOC_TOP_LOGPROBS:=10}                # MLX engines cap at 10
: ${LOC_TEMPERATURE:=0}
: ${LOC_MAX_TOKENS:=1500}
: ${TAG:=bonsai_v2}                      # artifact name tag
: ${HF_HOME:=/Volumes/Daten/huggingface}
export LOC_PROMPT_VERSION LOC_TEACHER_MODEL LOC_TOP_LOGPROBS LOC_TEMPERATURE LOC_MAX_TOKENS HF_HOME

TEACHER_FILE=dataset_annotated_${TAG}.jsonl
VAL_FILE=dataset_validation_gemini_${LOC_PROMPT_VERSION}.jsonl
EMB_CACHE=embeddings_${TAG}_jina.npz
VECTOR=internal_vector_${TAG}.npy
SYNTH_FILE=dataset_synthetic_${TAG}.jsonl
AUG_FILE=dataset_annotated_${TAG}_aug.jsonl

PY="uv run --with numpy --with scipy --with scikit-learn --with torch --with transformers --with einops --with datasets --with accelerate --with openai --with google-genai --with python-dotenv --with tqdm python"

echo "=== [1/8] Fetch follow network (Bluesky creds in .env) ==="
[ -f network_dids.json ] || $PY 1_fetch_network.py

echo "=== [2/8] Stream posts ==="
[ -f dataset_raw.jsonl ] || $PY 2_stream_posts.py

echo "=== [3/8] Teacher soft labels (LM Studio must be running) ==="
for pass in 1 2 3; do
  LOC_OUTPUT_FILE=$TEACHER_FILE $PY annotate_lmstudio_soft_labels.py
done

echo "=== [4/8] Validation labels (Gemini, GAI_API in .env; 500 req/day free tier) ==="
[ -f dataset_validation_gemini.jsonl ] || $PY create_validation_set.py
for pass in 1 2; do
  LOC_VAL_OUTPUT=$VAL_FILE $PY annotate_validation_v2.py
done

echo "=== [5/8] Train internal vector ==="
LOC_TEACHER_FILE=$TEACHER_FILE LOC_EMB_CACHE=$EMB_CACHE \
LOC_VAL_LABELS_FILE=$VAL_FILE LOC_VECTOR_OUT=$VECTOR \
  $PY train_internal_vector.py

echo "=== [6/8] Fine-tune transformer ==="
LOC_TEACHER_FILE=$TEACHER_FILE LOC_VAL_LABELS_FILE=$VAL_FILE \
LOC_OUTPUT_DIR=./loc-classifier-${TAG} \
  $PY train_transformer_soft.py

echo "=== [7/8] Synthetic counterexamples + augmented retrain ==="
if [ "${SKIP_SYNTH:-0}" != "1" ]; then
  LOC_TEACHER_FILE=$TEACHER_FILE LOC_EMB_CACHE=$EMB_CACHE LOC_VECTOR=$VECTOR \
  LOC_SYNTH_OUT=$SYNTH_FILE $PY create_synthetic_counterexamples.py
  cat $TEACHER_FILE $SYNTH_FILE > $AUG_FILE
  LOC_TEACHER_FILE=$AUG_FILE LOC_EMB_CACHE=embeddings_${TAG}_aug_jina.npz \
  LOC_VAL_LABELS_FILE=$VAL_FILE LOC_VECTOR_OUT=internal_vector_${TAG}_aug.npy \
    $PY train_internal_vector.py
  VECTOR=internal_vector_${TAG}_aug.npy
fi

echo "=== [8/8] Export deployable artifacts ==="
cp $VECTOR internal_vector.npy
cp ${VECTOR%.npy}_meta.json internal_vector_meta.json
$PY export_vector_json.py
echo "Done. Serve: backend (LOC_CLASSIFIER_PATH=./loc-classifier-${TAG}/final_model),"
echo "vector: internal_vector.npy, bunny_feed/src/vector.json updated."
