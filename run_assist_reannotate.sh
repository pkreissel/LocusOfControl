#!/bin/zsh
# Re-annotate on the fixed assist prompt (reader-answerability guideline +
# Score decoupled from Level). Two queues, both resumable: rerunning skips
# every URI already present in the output file, so it is safe to Ctrl-C.
#
#   1. old corpus requeue - the question-bearing and relay-bot posts from
#      dataset_annotated_assist_v1.jsonl, whose old labels predate both fixes
#   2. the question-filtered stream sample collected with LOC_TEXT_FILTER='\?'
#
# Concurrency is capped at 3 on purpose: llama.cpp splits its context across
# parallel slots, and at 4+ requests come back "Context size has been exceeded".
set -e
cd "$(dirname "$0")"

: ${LOC_PROMPT_VERSION:=assist}
: ${LOC_TEACHER_MODEL:=muse-glimmer-30B-kquant-17gb.gguf}
: ${LOC_API_BASE:=http://localhost:8080/v1}
: ${LOC_MAX_CONCURRENT:=3}
: ${LOC_MAX_TOKENS:=800}
: ${LOC_TEMPERATURE:=0}
: ${LOC_TOP_LOGPROBS:=10}
: ${LOC_REQUEST_TIMEOUT:=180}
: ${LOC_MAX_RETRIES:=3}
export LOC_PROMPT_VERSION LOC_TEACHER_MODEL LOC_API_BASE LOC_MAX_CONCURRENT \
       LOC_MAX_TOKENS LOC_TEMPERATURE LOC_TOP_LOGPROBS LOC_REQUEST_TIMEOUT LOC_MAX_RETRIES

PY=(uv run --with numpy --with openai --with tqdm python -u)

echo "=== [1/2] Re-annotate old corpus question + relay-bot posts ==="
LOC_INPUT_FILE=dataset_raw_old_requeue.jsonl \
LOC_OUTPUT_FILE=dataset_annotated_assist_requeue.jsonl \
  "${PY[@]}" annotate_lmstudio_soft_labels.py

echo "=== [2/2] Annotate the firehose help-request sample ==="
# Collected by collect_help_requests.sh. This replaced an earlier queue of
# network posts filtered on "?" alone: that pool ran ~3.3% helpful, this one
# ~22-28%, roughly 8x more positives per hour of teacher time.
LOC_INPUT_FILE=dataset_raw_help.jsonl \
LOC_OUTPUT_FILE=dataset_annotated_assist_help.jsonl \
  "${PY[@]}" annotate_lmstudio_soft_labels.py

echo "Done. Outputs: dataset_annotated_assist_requeue.jsonl, dataset_annotated_assist_help.jsonl"
