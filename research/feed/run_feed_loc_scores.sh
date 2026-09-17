#!/bin/zsh
# Cache a locus-of-control prediction for every post a live feed serves, locally,
# on the same 0-9 axis as the deployed vector score.
#
#   1. collect   - read the feed from the public AppView (no auth) into
#                  feed_posts_$TAG.jsonl
#   2. vector    - the deployed vector's own score for those posts, computed from
#                  local Jina weights instead of the Jina API, so a cached row is
#                  the number the feed would have assigned. scale*cos+bias is its
#                  Ridge prediction, which is what puts it on the teacher's axis.
#   3. reranker  - the fine-tuned jina-reranker-v3's raw score. A listwise loss
#                  fixes order, not magnitude, so it is on no scale of its own;
#                  step 5 projects it and the vector onto the shared band axis by
#                  the same fit, and ranks on the raw numbers either way.
#   4. teacher   - OPTIONAL (LOC_RUN_TEACHER=1): the v4 teacher LLM's graded 0-9
#                  loc_score. The only non-circular judge of the two above, since
#                  both were distilled from it, and the slow one: ~40s/post.
#                  Needs a server that returns logprobs — llama-server does,
#                  LM Studio's MLX path does NOT (it answers with empty logprobs
#                  and every row fails):
#                    llama-server -m /Volumes/Daten/LMstudio/lmstudio-community/\
#                      Muse-Glimmer-30B-GGUF/muse-glimmer-30B-kquant-17gb.gguf \
#                      --port 8080 -c 32768 -np 1
#   5. report    - joins whichever caches exist. Loads no model, so it is free to
#                  re-run as they fill.
#   6. push      - OPTIONAL (LOC_PUSH_PROD=1): write the vector scores into the
#                  LIVE feed's post_scores cache, so those posts arrive
#                  pre-scored and skip the Jina call. Verified faithful: local
#                  weights vs what production itself stored differ by a median
#                  6e-4 in probability, with no verdict flips at the 0.88 gate.
#                  Note the feed prunes created_at < now-4d, so a warmed cache
#                  decays within four days, and a retrain invalidates every
#                  pushed row at once by changing the version string.
#
# Every step is resumable and skips what its cache already holds, so a killed run
# costs one post. Re-running after the feed has moved on extends the sample.
set -e
cd "$(dirname "$0")"

: ${TAG:=silberskeets}
: ${LOC_FEED:=https://bsky.app/profile/sebastian.eurosky.social/feed/aaanl422clvee}
: ${LOC_FEED_MAX_POSTS:=600}
: ${HF_HOME:=/Volumes/Daten/huggingface}
# Gate this feed's vector scores are compared at (bunny_feed/src/feeds.ts).
: ${LOC_FEED_THRESHOLD:=0.88}
export HF_HOME LOC_FEED_THRESHOLD

# --- teacher (step 4 only) -------------------------------------------------
# v4 is the only LoC prompt that emits the graded 0-9 Score, which is what makes
# a teacher judgement comparable to a regression output at all.
: ${LOC_PROMPT_VERSION:=v4}
: ${LOC_TEACHER_MODEL:=muse-glimmer-30B-kquant-17gb.gguf}
: ${LOC_API_BASE:=http://localhost:8080/v1}
# The virtue axis measured this box as compute-bound at ~20s/post, with
# concurrency >1 costing prefix caching and latency for no throughput.
: ${LOC_MAX_CONCURRENT:=1}
: ${LOC_MAX_TOKENS:=800}
# Deterministic by construction: the cache is meant to be re-read, not re-sampled.
: ${LOC_TEMPERATURE:=0}
: ${LOC_TOP_LOGPROBS:=10}
: ${LOC_REQUEST_TIMEOUT:=300}
: ${LOC_MAX_RETRIES:=3}
export LOC_PROMPT_VERSION LOC_TEACHER_MODEL LOC_API_BASE LOC_MAX_CONCURRENT \
       LOC_MAX_TOKENS LOC_TEMPERATURE LOC_TOP_LOGPROBS LOC_REQUEST_TIMEOUT LOC_MAX_RETRIES

POSTS=feed_posts_${TAG}.jsonl
TEACHER_IN=feed_posts_${TAG}_scorable.jsonl
TEACHER_OUT=scores_loc_teacher_${TAG}.jsonl
VECTOR_OUT=scores_loc_vector_${TAG}.jsonl
RERANK_OUT=scores_loc_reranker_${TAG}.jsonl

PY_NET=(uv run --with python-dotenv python -u)
PY_LLM=(uv run --with openai --with numpy --with tqdm python -u)
PY_EMB=(uv run --with numpy --with torch --with transformers --with einops --with peft --with tqdm python -u)
PY_RR=(uv run --with numpy --with torch --with transformers --with einops --with peft --with tqdm python -u)
PY_REP=(uv run --with numpy --with scipy --with scikit-learn python -u)

echo "=== [1/6] Collect feed posts -> $POSTS ==="
"${PY_NET[@]}" collect_feed_posts.py --feed "$LOC_FEED" --out "$POSTS" \
  --max-posts "$LOC_FEED_MAX_POSTS"

echo "=== [2/6] Deployed vector score -> $VECTOR_OUT ==="
"${PY_EMB[@]}" score_loc_vector.py --posts "$POSTS" --out "$VECTOR_OUT"

echo "=== [3/6] Reranker score -> $RERANK_OUT ==="
"${PY_RR[@]}" score_loc_reranker.py --posts "$POSTS" --out "$RERANK_OUT"

if [ "${LOC_RUN_TEACHER:-0}" = "1" ]; then
  echo "=== [4/6] Teacher ($LOC_TEACHER_MODEL via $LOC_API_BASE) -> $TEACHER_OUT ==="
  # Only posts the feed would actually embed: the deployment drops anything under
  # minWords=6 before paying for an embedding, and a three-word post carries no
  # readable position on the axis. LOC_SCORE_ALL=1 annotates the short ones too.
  python3 - "$POSTS" "$TEACHER_IN" "${LOC_SCORE_ALL:-0}" <<'PY'
import json, sys
src, dst, keep_all = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
sel = [r for r in rows if (r.get("text") or "").strip() and (keep_all or r["enough_text"])]
with open(dst, "w", encoding="utf-8") as f:
    for r in sel:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(sel)} of {len(rows)} posts to score" + ("" if keep_all else " (feed-eligible only)"))
PY
  LOC_INPUT_FILE=$TEACHER_IN LOC_OUTPUT_FILE=$TEACHER_OUT \
    "${PY_LLM[@]}" annotate_lmstudio_soft_labels.py
else
  echo "=== [4/6] Teacher skipped (set LOC_RUN_TEACHER=1, and read the note above) ==="
fi

echo "=== [5/6] Join + report ==="
TEACHER_ARG=()
[ -f "$TEACHER_OUT" ] && TEACHER_ARG=(--teacher "$TEACHER_OUT")
"${PY_REP[@]}" report_loc_feed.py --posts "$POSTS" --vector "$VECTOR_OUT" \
  --reranker "$RERANK_OUT" "${TEACHER_ARG[@]}" --tag "$TAG" \
  --threshold "$LOC_FEED_THRESHOLD"

if [ "${LOC_PUSH_PROD:-0}" = "1" ]; then
  echo "=== [6/6] Warm the live score cache ==="
  # Dry run first, always: it prints what the write would change, and the commit
  # that follows is the same code path. Needs PROD_DATABASE_URL /
  # PROD_DATABASE_AUTH_TOKEN in bunny_feed/.env — BUNNY_DATABASE_URL is a
  # different, unused database.
  ( cd bunny_feed && npx tsx push_score_cache.ts "../$VECTOR_OUT" && \
    npx tsx push_score_cache.ts "../$VECTOR_OUT" --commit )
else
  echo "=== [6/6] Live score cache not touched (set LOC_PUSH_PROD=1 to warm it) ==="
fi
