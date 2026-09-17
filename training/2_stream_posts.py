import asyncio
import json
import websockets
from tqdm import tqdm
import os

# Data lives in training/data/, resolved against this file rather than the
# caller's cwd: run_all.sh, an editor and a cron job all invoke these scripts
# from different directories, and a bare relative default silently creates a
# second, empty corpus instead of failing. Env overrides still win, and a
# relative override still resolves against cwd -- deliberately, so a one-off
# experiment can point at a file next to it.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
def _d(name):
    return os.path.join(DATA, name)


import re
import time

# Connecting specifically to Jetstream to get app.bsky.feed.post and app.bsky.feed.like creations
JETSTREAM_URI = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post&wantedCollections=app.bsky.feed.like"

# Configure via env; defaults reproduce the original unfiltered collection.
TARGET_POST_COUNT = int(os.environ.get("LOC_TARGET_POSTS", "5000"))
OUTPUT_FILE = os.environ.get("LOC_RAW_OUTPUT", _d("dataset_raw.jsonl"))
# Regex a post's text must match to enter the buffer at all. Empty = keep all.
# For the assist axis: LOC_TEXT_FILTER='\\?' keeps only posts asking something.
TEXT_FILTER = os.environ.get("LOC_TEXT_FILTER", "")
# Which engagement makes a buffered post worth keeping: any | like | reply | none.
# "none" writes matching posts straight through (much faster, but no engagement signal).
ENGAGEMENT = os.environ.get("LOC_ENGAGEMENT", "any").lower()
# Restrict to the follow network (default) or take the whole firehose. The network
# is a small, topically narrow population; genuine help-seeking is rare in it, so a
# strict text filter over the open firehose can beat a loose one over the network.
REQUIRE_NETWORK = os.environ.get("LOC_REQUIRE_NETWORK", "1") != "0"
BACKFILL_DAYS = float(os.environ.get("LOC_BACKFILL_DAYS", "7"))

TEXT_RE = re.compile(TEXT_FILTER) if TEXT_FILTER else None
# Links are stripped before matching so a URL query string ("...?v=abc") does not
# read as a question mark.
URL_RE = re.compile(r"(?:https?://|www\.)\S+")


def dig(obj, *keys):
    """Walk nested dict keys, tolerating malformed records on the firehose.

    Posts occasionally carry a non-dict where a dict is expected (e.g. "reply": true),
    which otherwise raises and forces a needless 5s reconnect.
    """
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def matches_filter(text):
    if TEXT_RE is None:
        return True
    return TEXT_RE.search(URL_RE.sub(" ", text)) is not None
KEEP_ON_LIKE = ENGAGEMENT in ("any", "like")
KEEP_ON_REPLY = ENGAGEMENT in ("any", "reply")

async def stream_posts():
    if not os.path.exists(_d('network_dids.json')):
        print("network_dids.json not found. Run 1_fetch_network.py first.")
        return

    with open(_d('network_dids.json'), 'r') as f:
        network_dids = set(json.load(f))
    
    print(f"Loaded {len(network_dids)} DIDs from network.")
    print(f"Connecting to Jetstream: {JETSTREAM_URI}")
    print(f"Output: {OUTPUT_FILE} | target: {TARGET_POST_COUNT} | "
          f"text filter: {TEXT_FILTER or '(none)'} | engagement: {ENGAGEMENT} | "
          f"network-only: {REQUIRE_NETWORK}")
    
    current_cursor = int((time.time() - BACKFILL_DAYS * 24 * 3600) * 1_000_000) # microseconds
    
    # Check if we already have some collected posts to resume
    collected_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            last_line = None
            for line in f:
                collected_count += 1
                last_line = line
            if last_line:
                try:
                    last_data = json.loads(last_line)
                    if 'time_us' in last_data and last_data['time_us']:
                        current_cursor = last_data['time_us']
                except json.JSONDecodeError:
                    pass
        print(f"Found {collected_count} existing posts in {OUTPUT_FILE}. Resuming from cursor {current_cursor}...")
    else:
        print(f"Starting fresh. Backfilling from {BACKFILL_DAYS} days ago (cursor {current_cursor})...")

    if collected_count >= TARGET_POST_COUNT:
        print(f"Already reached the target of {TARGET_POST_COUNT} posts.")
        return

    post_buffer = {} # uri -> post_data

    # We open in append mode in case of restarts
    with open(OUTPUT_FILE, 'a') as f:
        with tqdm(initial=collected_count, total=TARGET_POST_COUNT, desc="Collecting Posts") as pbar:
            while collected_count < TARGET_POST_COUNT:
                try:
                    uri_with_cursor = f"{JETSTREAM_URI}&cursor={current_cursor}"
                    async with websockets.connect(uri_with_cursor) as websocket:
                        async for message in websocket:
                            data = json.loads(message)
                            
                            time_us = data.get('time_us')
                            if time_us:
                                current_cursor = time_us
                                
                            if data.get('kind') != 'commit':
                                continue
                                
                            commit = data.get('commit', {})
                            if not isinstance(commit, dict) or commit.get('operation') != 'create':
                                continue
                                
                            collection = commit.get('collection')
                            record = commit.get('record', {})
                            if not isinstance(record, dict):
                                continue
                            did = data.get('did')
                            
                            if collection == 'app.bsky.feed.post':
                                # 1. Is this a reply? If so, check if it replies to a buffered post
                                if 'reply' in record:
                                    parent_uri = dig(record, 'reply', 'parent', 'uri')
                                    if KEEP_ON_REPLY and parent_uri and parent_uri in post_buffer:
                                        # The parent post got a reply! Save it.
                                        f.write(json.dumps(post_buffer[parent_uri]) + '\n')
                                        f.flush()
                                        del post_buffer[parent_uri]
                                        collected_count += 1
                                        pbar.update(1)
                                        if collected_count >= TARGET_POST_COUNT:
                                            break
                                            
                                # 2. Process this post for the buffer (if it's a top-level post from our network)
                                if (not REQUIRE_NETWORK or did in network_dids) and 'reply' not in record:
                                    text = record.get('text', '').strip()
                                    # Removed length caps and language filters as requested.
                                    if len(text) > 0 and matches_filter(text):
                                        uri = f"at://{did}/app.bsky.feed.post/{commit.get('rkey')}"
                                        post = {
                                            "did": did,
                                            "uri": uri,
                                            "text": text,
                                            "createdAt": record.get('createdAt'),
                                            "langs": record.get('langs', []),
                                            "time_us": time_us
                                        }
                                        if ENGAGEMENT == "none":
                                            # No engagement gate: the text filter alone decides.
                                            f.write(json.dumps(post) + '\n')
                                            f.flush()
                                            collected_count += 1
                                            pbar.update(1)
                                            if collected_count >= TARGET_POST_COUNT:
                                                break
                                            continue
                                        post_buffer[uri] = post
                                        
                                        # Periodically clean up old posts from the buffer to prevent memory leaks
                                        # 10,000 items is a safe upper bound
                                        if len(post_buffer) > 10000:
                                            # Remove the oldest 10%
                                            oldest_uris = list(post_buffer.keys())[:1000]
                                            for old_uri in oldest_uris:
                                                del post_buffer[old_uri]

                            elif collection == 'app.bsky.feed.like':
                                # Check if it's a like on a buffered post
                                subject_uri = dig(record, 'subject', 'uri')
                                if KEEP_ON_LIKE and subject_uri and subject_uri in post_buffer:
                                    # The post got a like! Save it.
                                    f.write(json.dumps(post_buffer[subject_uri]) + '\n')
                                    f.flush()
                                    del post_buffer[subject_uri]
                                    collected_count += 1
                                    pbar.update(1)
                                    if collected_count >= TARGET_POST_COUNT:
                                        break
                                        
                except websockets.exceptions.ConnectionClosed:
                    print("\nConnection closed by server. Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                except Exception as e:
                    print(f"\nUnexpected error: {e}. Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                    
    print(f"\nSuccessfully collected {collected_count} posts in {OUTPUT_FILE}.")

if __name__ == '__main__':
    asyncio.run(stream_posts())
