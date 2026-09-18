#!/bin/zsh
# Score every post a live feed serves on the locus-of-control axis, locally,
# and compare the deployed vector against the teacher it was distilled from.
#
#   1. collect    - read the feed from the public AppView (no auth) into
#                   feed_posts_$TAG.jsonl. Resumable: re-running extends the
#                   sample with whatever the feed has added since.
#   2. eligible   - the subset the deployment would actually embed (minWords=6).
#                   Every later step works on exactly this set, so the report's
#                   columns all cover the same posts -- report_loc_feed.py drops
#                   its teacher section entirely if one row is missing a score.
#   3. vector     - the deployed vector's own score, computed from local Jina
#                   weights instead of the Jina API, so a cached row is the
#                   number the feed would have assigned. scale*cos+bias is its
#                   Ridge prediction, which is what puts it on the teacher's axis.
#   4. reranker   - OPTIONAL (LOC_RUN_RERANKER=1): the fine-tuned
#                   jina-reranker-v3's raw score. A listwise loss fixes order,
#                   not magnitude, so it is on no scale of its own; the report
#                   projects it and the vector onto the shared band axis by the
#                   same fit, and ranks on the raw numbers either way.
#                   Off by default because the model itself is in archive/: it
#                   was never deployed, and the teacher below is the judge the
#                   deployed vector is actually measured against.
#   5. teacher    - TypeSafe System One, the teacher the deployed vectors are
#                   distilled from (training/score_loc_typesafe.py). Runs by
#                   default: it is ~1s per post at 12 workers and $0.05 per 1000,
#                   against the ~40s per post of the local v4 GGUF teacher this
#                   step used to gate behind LOC_RUN_TEACHER. Billed per post and
#                   never bought twice: training/data/typesafe_ledger.jsonl is
#                   the record of every post ever scored, across every tag, and
#                   this step asks only for what is missing from it.
#   6. calibrate  - put the teacher's judgement on the deployed vector's
#                   probability scale, and report what each would admit at the
#                   live gate. Needed before any TypeSafe score can go near the
#                   feed's cache, which is read against a threshold measured on
#                   the vector's distribution.
#   7. report     - joins whichever caches exist. Loads no model, so it is free
#                   to re-run as they fill.
#   8. push       - OPTIONAL (LOC_PUSH_PROD=1): warm the LIVE feed's post_scores
#                   cache so these posts arrive pre-scored and skip the Jina
#                   call. LOC_PUSH_SOURCE picks what gets written:
#                     typesafe  (default) the calibrated teacher score from step
#                               6. Its SCALE is the vector's, so the gate still
#                               means what it means, but the ORDER is the
#                               teacher's: at the vector's own keep-rate it holds
#                               every internal post and leaks 0.5% of externals,
#                               against the vector's 67.8% and 28.1%. That is the
#                               reason it is the default, and the cost is that
#                               post_scores no longer holds what the worker would
#                               have computed -- only the score_source field in
#                               the pushed file records the difference.
#                     vector    the vector's own score -- byte-for-byte what the
#                               worker would have produced. A pure warm-up: it
#                               saves the Jina call and changes nothing about
#                               what the feed serves.
#                   Either way the map is fitted on THIS run's pool, so a pushed
#                   score is only comparable to others from the same run.
#                   Note the feed prunes created_at < now-4d, so a warmed cache
#                   decays within four days, and a retrain invalidates every
#                   pushed row at once by changing the version string.
#
# Every step is resumable and skips what its cache already holds, so a killed run
# costs one post.
set -e
cd "$(dirname "$0")"

# Data lives under research/data/ (gitignored), not next to the scripts.
DATA=../data/feed
mkdir -p "$DATA"

: ${TAG:=silberskeets}
: ${LOC_FEED:=https://bsky.app/profile/sebastian.eurosky.social/feed/aaanl422clvee}
: ${LOC_FEED_MAX_POSTS:=600}
: ${HF_HOME:=/Volumes/Daten/huggingface}
# Gate this feed's vector scores are compared at (bunny_feed/src/feeds.ts). It
# moved to 0.86 with the TypeSafe retrain; a threshold never survives a retrain,
# so this number is only ever copied from feeds.ts, never chosen here.
: ${LOC_FEED_THRESHOLD:=0.86}
# The deployed Jina vector and the gold artifacts the report's band calibration
# is fitted on. All of them live in training/data/ and none of them resolve from
# this directory, so they are named here rather than left to a default cwd.
: ${LOC_VECTOR:=../../training/data/internal_vector.npy}
: ${LOC_VAL_FILE:=../../training/data/dataset_validation_gemini.jsonl}
: ${LOC_VAL_EMB_CACHE:=../../training/data/embeddings_val_loc_jina.npz}
: ${LOC_TEACHER_FILE:=../../training/data/dataset_annotated_lmstudio_soft.jsonl}
export HF_HOME LOC_FEED_THRESHOLD LOC_VECTOR LOC_VAL_FILE LOC_VAL_EMB_CACHE \
       LOC_TEACHER_FILE

# --- teacher (step 5) ------------------------------------------------------
: ${TYPESAFE_MODEL:=jev-latest}
# 12 workers measured 15.5 posts/s wall, against a documented ceiling of 1200
# requests/min. There is ~1.3x left, not more.
: ${TYPESAFE_WORKERS:=12}
export TYPESAFE_MODEL TYPESAFE_WORKERS
if [ -z "$TYPESAFE_API_KEY" ] && [ -f ../../.env ]; then
  set -a; . ../../.env; set +a
fi

# How the teacher's three answers fold into one ranking (see step 6). `band`
# floors external the way the trainer's band target does, and that floor is the
# whole difference: without it the teacher leaks MORE external than the vector.
: ${LOC_RANK_KEY:=band}

PY_NET=(uv run --with python-dotenv python -u)
PY_TS=(python3 -u)
PY_EMB=(uv run --with numpy --with torch --with transformers --with einops --with peft --with tqdm python -u)
PY_RR=("${PY_EMB[@]}")
PY_REP=(uv run --with numpy --with scipy --with scikit-learn python -u)

POSTS=$DATA/feed_posts_${TAG}.jsonl
SCORABLE=$DATA/feed_posts_${TAG}_scorable.jsonl
EMB=$DATA/scores_loc_vector_${TAG}_emb.npz
RERANK_OUT=$DATA/scores_loc_reranker_${TAG}.jsonl
TEACHER_OUT=$DATA/scores_loc_typesafe_${TAG}.jsonl
PUSH_OUT=$DATA/scores_push_${TAG}.jsonl

echo "=== [1/8] Collect feed posts -> $POSTS ==="
"${PY_NET[@]}" collect_feed_posts.py --feed "$LOC_FEED" --out "$POSTS" \
  --max-posts "$LOC_FEED_MAX_POSTS"

echo "=== [2/8] Feed-eligible subset -> $SCORABLE ==="
# LOC_SCORE_ALL=1 keeps the short posts too. They are cheap to score and useless
# to read: a three-word post carries no position on this axis, and the deployment
# drops them before paying for an embedding.
python3 - "$POSTS" "$SCORABLE" "${LOC_SCORE_ALL:-0}" <<'PY'
import json, sys
src, dst, keep_all = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
sel = [r for r in rows if (r.get("text") or "").strip() and (keep_all or r["enough_text"])]
with open(dst, "w", encoding="utf-8") as f:
    for r in sel:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(sel)} of {len(rows)} posts to score" + ("" if keep_all else " (feed-eligible only)"))
PY

# The vector cache is stamped with the vector's own fingerprint, because
# report_loc_feed.py indexes by uri alone (last row wins) and reads the FIRST
# row's metadata for its calibration — so one file holding two retrains' rows
# would quietly report new scores through an old vector's calibration. The
# fingerprint is imported from the scorer, never restated, exactly as
# push_score_cache.ts imports it from the worker.
VER=$(uv run --with numpy python -c "
import json, numpy as np, score_loc_vector as S
v, meta, _ = S.load_vector()
print(S.fingerprint(list(np.load(S.VECTOR_FILE).astype(np.float32)) +
                    [float(meta['scale']), float(meta['bias'])]))" | tail -1)
VECTOR_OUT=$DATA/scores_loc_vector_${TAG}_${VER}.jsonl

echo "=== [3/8] Deployed vector score ($VER) -> $VECTOR_OUT ==="
"${PY_EMB[@]}" score_loc_vector.py --posts "$SCORABLE" --out "$VECTOR_OUT" \
  --emb-cache "$EMB"

# The reranker was archived with the rest of the superseded work, so its default
# path no longer resolves from anywhere. Named here rather than left to fail at
# model-load time, four minutes into a run.
: ${LOC_RERANKER_DIR:=../../archive/loc-reranker-jina-v3/final_model}
export LOC_RERANKER_DIR

echo "=== [4/8] Reranker score -> $RERANK_OUT ==="
if [ "${LOC_RUN_RERANKER:-0}" = "1" ]; then
  "${PY_RR[@]}" score_loc_reranker.py --posts "$SCORABLE" --out "$RERANK_OUT"
else
  echo "skipped (set LOC_RUN_RERANKER=1; the model lives in $LOC_RERANKER_DIR)"
fi

echo "=== [5/8] Teacher: TypeSafe $TYPESAFE_MODEL -> $TEACHER_OUT ==="
if [ -z "$TYPESAFE_API_KEY" ]; then
  echo "TYPESAFE_API_KEY unset and not found in ../../.env — skipping the teacher." >&2
else
  "${PY_TS[@]}" ../../training/score_loc_typesafe.py "$SCORABLE" "$TEACHER_OUT"
fi

echo "=== [6/8] Calibrate the teacher onto the vector's scale -> $PUSH_OUT ==="
if [ -f "$TEACHER_OUT" ]; then
  "${PY_REP[@]}" calibrate_typesafe_to_vector.py --posts "$SCORABLE" \
    --vector "$VECTOR_OUT" --typesafe "$TEACHER_OUT" \
    --threshold "$LOC_FEED_THRESHOLD" --key "$LOC_RANK_KEY" --out "$PUSH_OUT"
else
  echo "no teacher scores; skipped"
fi

echo "=== [7/8] Join + report ==="
TEACHER_ARG=()
[ -f "$TEACHER_OUT" ] && TEACHER_ARG=(--teacher "$TEACHER_OUT")
RERANK_ARG=()
[ -f "$RERANK_OUT" ] && RERANK_ARG=(--reranker "$RERANK_OUT")
"${PY_REP[@]}" report_loc_feed.py --posts "$SCORABLE" --vector "$VECTOR_OUT" \
  "${RERANK_ARG[@]}" "${TEACHER_ARG[@]}" --tag "$TAG" \
  --threshold "$LOC_FEED_THRESHOLD"

if [ "${LOC_PUSH_PROD:-0}" = "1" ]; then
  : ${LOC_PUSH_SOURCE:=typesafe}
  case "$LOC_PUSH_SOURCE" in
    vector)   SRC=$VECTOR_OUT ;;
    typesafe) SRC=$PUSH_OUT ;;
    *) echo "LOC_PUSH_SOURCE must be 'vector' or 'typesafe'" >&2; exit 1 ;;
  esac
  # Step 6 is skipped when the teacher has no scores, and the default source is
  # its output -- so say that, rather than letting npx fail on a missing path.
  if [ ! -f "$SRC" ]; then
    echo "LOC_PUSH_SOURCE=$LOC_PUSH_SOURCE but $SRC does not exist (step 6 ran?)" >&2
    exit 1
  fi
  echo "=== [8/8] Warm the live score cache from $LOC_PUSH_SOURCE ($SRC) ==="
  # Dry run first, always: it prints what the write would change, and the commit
  # that follows is the same code path. Needs PROD_DATABASE_URL /
  # PROD_DATABASE_AUTH_TOKEN in bunny_feed/.env — BUNNY_DATABASE_URL is a
  # different, unused database.
  ( cd ../../bunny_feed && npx tsx push_score_cache.ts "../research/data/feed/$(basename $SRC)" && \
    npx tsx push_score_cache.ts "../research/data/feed/$(basename $SRC)" --commit )
else
  echo "=== [8/8] Live score cache not touched (set LOC_PUSH_PROD=1 to warm it) ==="
fi
