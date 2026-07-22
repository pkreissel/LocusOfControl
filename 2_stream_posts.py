import asyncio
import json
import websockets
from tqdm import tqdm
import os
import time

# Connecting specifically to Jetstream to get app.bsky.feed.post creations
JETSTREAM_URI = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
TARGET_POST_COUNT = 5000
OUTPUT_FILE = "dataset_raw.jsonl"

async def stream_posts():
    if not os.path.exists('network_dids.json'):
        print("network_dids.json not found. Run 1_fetch_network.py first.")
        return

    with open('network_dids.json', 'r') as f:
        network_dids = set(json.load(f))
    
    print(f"Loaded {len(network_dids)} DIDs from network.")
    print(f"Connecting to Jetstream: {JETSTREAM_URI}")
    
    current_cursor = int((time.time() - 7 * 24 * 3600) * 1_000_000) # 7 days ago in microseconds
    
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
        print(f"Starting fresh. Backfilling from 7 days ago (cursor {current_cursor})...")

    if collected_count >= TARGET_POST_COUNT:
        print(f"Already reached the target of {TARGET_POST_COUNT} posts.")
        return

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
                            if commit.get('operation') != 'create':
                                continue
                                
                            did = data.get('did')
                            if did not in network_dids:
                                continue
                                
                            record = commit.get('record', {})
                            
                            # Skip replies to only collect top-level posts
                            if 'reply' in record:
                                continue

                            text = record.get('text', '').strip()
                            
                            # Filter out empty posts or very short ones (less than 15 chars might lack context for LoC)
                            if len(text) < 15:
                                continue
                                
                            post_data = {
                                "did": did,
                                "uri": f"at://{did}/app.bsky.feed.post/{commit.get('rkey')}",
                                "text": text,
                                "createdAt": record.get('createdAt'),
                                "langs": record.get('langs', []),
                                "time_us": time_us
                            }
                            
                            f.write(json.dumps(post_data) + '\n')
                            f.flush()
                            
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
                    
    print(f"\nSuccessfully collected {TARGET_POST_COUNT} posts in {OUTPUT_FILE}.")

if __name__ == '__main__':
    asyncio.run(stream_posts())
