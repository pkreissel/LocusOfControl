#!/bin/zsh
# End-to-end pipeline for the "assist" axis (request-for-help vs inert-question).
#
# Trains a vector model that ranks posts highly that ask for help or advice
# (situated, concrete, actionable requests) and filters out questions that
# lead nowhere (rhetorical, circular, engagement-bait, no practical resolution).
#
# Steps:
#   1. Annotate raw posts with the assist teacher prompt
#   2. Annotate the validation posts on the assist axis with Gemini
#   3. Train the Ridge/Rocchio vector against H/R/N soft labels
#   4. Optionally: generate synthetic counterexamples
#   5. Optionally: augment the teacher set and retrain
#   6. Export vector.json for bunny_feed
#
# Set env vars to configure. Defaults produce assist_v1 artifacts.
# Note: Steps 2 and 4 require the GAI_API env var for Gemini.
set -e
cd "$(dirname "$0")"

: ${LOC_PROMPT_VERSION:=assist}           # use the assist prompt
: ${TAG:=assist_v1}                      # artifact name tag
: ${LOC_API_BASE:=http://localhost:8080/v1} # llama.cpp server (or LM Studio / OpenRouter)
: ${LOC_TEACHER_MODEL:=muse-glimmer-30B-kquant-17gb.gguf}  # model for teacher soft labels
: ${LOC_TOP_LOGPROBS:=10}
: ${LOC_TEMPERATURE:=0}
: ${LOC_MAX_TOKENS:=1500}

export LOC_PROMPT_VERSION LOC_TEACHER_MODEL LOC_TOP_LOGPROBS LOC_TEMPERATURE LOC_MAX_TOKENS

TEACHER_FILE=dataset_annotated_${TAG}.jsonl
VECTOR=assist_vector_${TAG}.npy
EMB_CACHE=embeddings_${TAG}_jina.npz
VAL_FILE=dataset_validation_${TAG}.jsonl
VAL_EMB_CACHE=embeddings_validation_${TAG}_jina.npz
SYNTH_FILE=dataset_synthetic_${TAG}.jsonl
AUG_FILE=dataset_annotated_${TAG}_aug.jsonl

PY=(uv run --with numpy --with scipy --with scikit-learn --with torch --with transformers --with einops --with peft --with openai --with tqdm --with google-genai --with python-dotenv python)

echo "=== [assist 1/6] Annotate with assist prompt (llama.cpp server must be running) ==="
[ -f "$TEACHER_FILE" ] || LOC_OUTPUT_FILE="$TEACHER_FILE" "${PY[@]}" annotate_lmstudio_soft_labels.py

echo "=== [assist 2/6] Annotate validation posts on the assist axis (Gemini) ==="
# The LoC validation set (internal/external/neutral) says nothing about this axis,
# so the same 500 posts are relabelled helpful/inert/neutral. Resumable.
LOC_VAL_OUTPUT="$VAL_FILE" "${PY[@]}" annotate_validation_v2.py

echo "=== [assist 3/6] Train internal vector ==="
# Embed and train the Ridge vector against the assist axis
LOC_TEACHER_FILE="$TEACHER_FILE" \
  LOC_VECTOR_OUT="$VECTOR" \
  LOC_EMB_CACHE="$EMB_CACHE" \
  LOC_VAL_FILE="$VAL_FILE" \
  LOC_VAL_EMB_CACHE="$VAL_EMB_CACHE" \
  LOC_SELECT_METRIC=val_ndcg_graded \
  "${PY[@]}" train_internal_vector.py

echo "=== [assist 4/6] Create synthetic counterexamples (optional, set SKIP_SYNTH=1 to skip) ==="
: ${SKIP_SYNTH:=0}
if [ "$SKIP_SYNTH" = "0" ]; then
  LOC_TEACHER_FILE="$TEACHER_FILE" \
    LOC_SYNTH_OUT="$SYNTH_FILE" \
    SKIP_SYNTH=1 \
    "${PY[@]}" create_synthetic_counterexamples.py
fi

echo "=== [assist 5/6] Augment and retrain vector (optional, set SKIP_AUG=1 to skip) ==="
: ${SKIP_AUG:=0}
if [ "$SKIP_AUG" = "0" ] && [ "$SKIP_SYNTH" = "0" ]; then
  # cat teacher + synth, retrain
  cat "$TEACHER_FILE" "$SYNTH_FILE" > "$AUG_FILE"
  LOC_TEACHER_FILE="$AUG_FILE" \
    LOC_VECTOR_OUT="$VECTOR" \
    LOC_EMB_CACHE="${EMB_CACHE%.*}_aug.npz" \
    LOC_VAL_FILE="$VAL_FILE" \
    LOC_VAL_EMB_CACHE="$VAL_EMB_CACHE" \
    LOC_SELECT_METRIC=val_ndcg_graded \
    SYNTH_WEIGHTS="0.5,1.0,2.0" \
    "${PY[@]}" train_internal_vector.py
fi

echo "=== [assist 6/6] Export the feed's vector JSON for bunny_feed ==="
# NB the variable names matter: export_vector_json.py reads LOC_VECTOR /
# LOC_VECTOR_JSON_OUT. This step used to pass VECTOR_IN / VECTOR_OUT, which the
# script ignores - so it silently exported the default internal_vector.npy over
# bunny_feed/src/vector.json, i.e. clobbered the live locus-of-control feed with
# whichever vector happened to be lying around. The assist axis has its own file.
LOC_VECTOR="$VECTOR" \
  LOC_VECTOR_JSON_OUT="${LOC_VECTOR_JSON_OUT:-bunny_feed/src/vector_helpful.json}" \
  "${PY[@]}" export_vector_json.py

echo "\nDone. Artifacts:"
echo "  teacher annotations: $TEACHER_FILE"
echo "  validation labels:  $VAL_FILE"
echo "  vector:             $VECTOR + _meta.json"
echo "  edge-ready:        ${LOC_VECTOR_JSON_OUT:-bunny_feed/src/vector_helpful.json}"
if [ "$SKIP_SYNTH" = "0" ]; then
  echo "  synthetic:         $SYNTH_FILE"
fi
if [ "$SKIP_AUG" = "0" ] && [ "$SKIP_SYNTH" = "0" ]; then
  echo "  augmented:         $AUG_FILE"
fi
