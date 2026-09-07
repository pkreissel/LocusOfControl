#!/bin/zsh
# End-to-end pipeline for the "virtue" axis (intellectual virtue vs dogmatism).
#
# Trains a vector model that ranks posts highly which handle a contestable
# claim well - humility about one's own limits, confidence proportioned to the
# evidence, competing considerations weighed, claims scoped - and filters out
# dogmatism: blanket statements about groups, unearned certainty, contempt in
# place of argument.
#
# Two things differ from the other axes and are worth reading before a run:
#   * The confound here is REGISTER. "Nuanced-sounding" correlates with long,
#     essayistic, academic prose, and a Ridge vector over embeddings will
#     happily learn that shortcut instead of the epistemics. Step 4/5 (same
#     length, same voice, only the framing flipped) is therefore NOT optional
#     on this axis - run it and compare the two vectors.
#   * A random 500-post validation set is far too sparse in 'virtuous' to
#     select on (the same problem the assist axis hit). Select on
#     dev_auc_teacher and measure on a deliberately enriched slice.
#
# Steps:
#   1. Annotate raw posts with the virtue teacher prompt
#   2. Annotate the validation posts on the virtue axis with Gemini
#   3. Train the Ridge/Rocchio vector against V/D/N soft labels
#   4. Optionally: generate synthetic counterexamples
#   5. Optionally: augment the teacher set and retrain
#   6. Export vector.json (review before pointing the live feed at it)
#
# Set env vars to configure. Defaults produce virtue_v1 artifacts.
# Note: Steps 2 and 4 require the GAI_API env var for Gemini.
set -e
cd "$(dirname "$0")"

: ${LOC_PROMPT_VERSION:=virtue}           # use the virtue prompt
: ${TAG:=virtue_v1}                      # artifact name tag
: ${LOC_API_BASE:=http://localhost:8080/v1} # llama.cpp server (or LM Studio / OpenRouter)
: ${LOC_TEACHER_MODEL:=muse-glimmer-30B-kquant-17gb.gguf}  # model for teacher soft labels
: ${LOC_TOP_LOGPROBS:=10}
: ${LOC_TEMPERATURE:=0}
: ${LOC_MAX_TOKENS:=1500}

export LOC_PROMPT_VERSION LOC_TEACHER_MODEL LOC_TOP_LOGPROBS LOC_TEMPERATURE LOC_MAX_TOKENS

TEACHER_FILE=dataset_annotated_${TAG}.jsonl
VECTOR=virtue_vector_${TAG}.npy
EMB_CACHE=embeddings_${TAG}_jina.npz
VAL_FILE=dataset_validation_${TAG}.jsonl
VAL_EMB_CACHE=embeddings_validation_${TAG}_jina.npz
SYNTH_FILE=dataset_synthetic_${TAG}.jsonl
AUG_FILE=dataset_annotated_${TAG}_aug.jsonl

PY=(uv run --with numpy --with scipy --with scikit-learn --with torch --with transformers --with einops --with peft --with openai --with tqdm --with google-genai --with python-dotenv python)

echo "=== [virtue 1/6] Annotate with virtue prompt (llama.cpp server must be running) ==="
[ -f "$TEACHER_FILE" ] || LOC_OUTPUT_FILE="$TEACHER_FILE" "${PY[@]}" annotate_lmstudio_soft_labels.py

echo "=== [virtue 2/6] Annotate validation posts on the virtue axis (Gemini) ==="
# The LoC validation set (internal/external/neutral) says nothing about this axis,
# so the same 500 posts are relabelled virtuous/dogmatic/neutral. Resumable.
# Expect very few 'virtuous' in a random sample - treat this file as a sanity
# check on the definitions, not as the model-selection signal.
LOC_VAL_OUTPUT="$VAL_FILE" "${PY[@]}" annotate_validation_v2.py

echo "=== [virtue 3/6] Train internal vector ==="
# Embed and train the Ridge vector against the assist axis
LOC_TEACHER_FILE="$TEACHER_FILE" \
  LOC_VECTOR_OUT="$VECTOR" \
  LOC_EMB_CACHE="$EMB_CACHE" \
  LOC_VAL_FILE="$VAL_FILE" \
  LOC_VAL_EMB_CACHE="$VAL_EMB_CACHE" \
  LOC_SELECT_METRIC=${LOC_SELECT_METRIC:-dev_auc_teacher} \
  "${PY[@]}" train_internal_vector.py

echo "=== [virtue 4/6] Create synthetic counterexamples (do NOT skip on this axis - breaks the register shortcut) ==="
: ${SKIP_SYNTH:=0}
if [ "$SKIP_SYNTH" = "0" ]; then
  LOC_TEACHER_FILE="$TEACHER_FILE" \
    LOC_SYNTH_OUT="$SYNTH_FILE" \
    SKIP_SYNTH=1 \
    "${PY[@]}" create_synthetic_counterexamples.py
fi

echo "=== [virtue 5/6] Augment and retrain vector (optional, set SKIP_AUG=1 to skip) ==="
: ${SKIP_AUG:=0}
if [ "$SKIP_AUG" = "0" ] && [ "$SKIP_SYNTH" = "0" ]; then
  # cat teacher + synth, retrain
  cat "$TEACHER_FILE" "$SYNTH_FILE" > "$AUG_FILE"
  LOC_TEACHER_FILE="$AUG_FILE" \
    LOC_VECTOR_OUT="$VECTOR" \
    LOC_EMB_CACHE="${EMB_CACHE%.*}_aug.npz" \
    LOC_VAL_FILE="$VAL_FILE" \
    LOC_VAL_EMB_CACHE="$VAL_EMB_CACHE" \
    LOC_SELECT_METRIC=${LOC_SELECT_METRIC:-dev_auc_teacher} \
    SYNTH_WEIGHTS="0.5,1.0,2.0" \
    "${PY[@]}" train_internal_vector.py
fi

echo "=== [virtue 6/6] Export vector.json (review before pointing the live feed at it) ==="
VECTOR_IN="$VECTOR" \
  VECTOR_OUT=${VECTOR_JSON_OUT:-virtue_vector.json} \
  "${PY[@]}" export_vector_json.py

echo "\nDone. Artifacts:"
echo "  teacher annotations: $TEACHER_FILE"
echo "  validation labels:  $VAL_FILE"
echo "  vector:             $VECTOR + _meta.json"
echo "  edge-ready:        ${VECTOR_JSON_OUT:-virtue_vector.json}"
if [ "$SKIP_SYNTH" = "0" ]; then
  echo "  synthetic:         $SYNTH_FILE"
fi
if [ "$SKIP_AUG" = "0" ] && [ "$SKIP_SYNTH" = "0" ]; then
  echo "  augmented:         $AUG_FILE"
fi
