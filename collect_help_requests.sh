#!/bin/zsh
# Collect likely help/advice requests from the OPEN firehose.
#
# Rationale (measured 2026-09-06): the follow network is a politics/news-heavy
# population where genuine help-seeking is rare - 0.58% of posts, and only 3.3%
# even among posts containing "?". Filtering the open firehose on "?" AND a
# help/advice cue instead ran ~28% genuine requests in a live probe, about 8x
# more positives per hour of teacher annotation.
#
# The engagement gate is deliberately OFF: a genuine niche request often sits
# unanswered, so requiring a like/reply selects for engagement-bait, which is
# the negative class here.
set -e
cd "$(dirname "$0")"

CUE='\b(help|advice|recommend|recommendations?|suggestions?|tips?|anyone know|anybody know|does anyone|how do i|how can i|what should i|looking for|struggling|weiß jemand|kennt (jemand|wer)|kann mir (wer|jemand)|suche)\b'

: ${LOC_RAW_OUTPUT:=dataset_raw_help.jsonl}
: ${LOC_TARGET_POSTS:=2000}
: ${LOC_TEXT_FILTER:="(?is)(?=.*\?)(?=.*$CUE)"}
: ${LOC_ENGAGEMENT:=none}
: ${LOC_REQUIRE_NETWORK:=0}
: ${LOC_BACKFILL_DAYS:=7}
export LOC_RAW_OUTPUT LOC_TARGET_POSTS LOC_TEXT_FILTER LOC_ENGAGEMENT \
       LOC_REQUIRE_NETWORK LOC_BACKFILL_DAYS

exec uv run --with websockets --with tqdm python -u 2_stream_posts.py
