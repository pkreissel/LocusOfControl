import os
import json
import logging
from dotenv import load_dotenv
from atproto import Client

logging.basicConfig(level=logging.INFO)

def main():
    load_dotenv()
    
    # Load interactions
    interactions_file = 'bunny_feed/interactions.json'
    if not os.path.exists(interactions_file):
        logging.error(f"{interactions_file} not found.")
        return

    with open(interactions_file, 'r', encoding='utf-8') as f:
        interactions = json.load(f)

    # Count events to see what we have
    events_count = {}
    for inter in interactions:
        evt = inter.get('event')
        events_count[evt] = events_count.get(evt, 0) + 1
    
    logging.info(f"Events found: {events_count}")

    # We want to map requestLess to external (label=0.0)
    # What about positive interactions?
    # For now, let's just collect all interactions that are not "seen" 
    # Or map specific ones: requestLess -> external, interactionLike -> internal
    
    target_interactions = []
    for inter in interactions:
        evt = inter.get('event')
        if evt == 'app.bsky.feed.defs#requestLess':
            label = 'external' # negative for this feed
            target_interactions.append((inter['item'], label))
        elif evt in ['app.bsky.feed.defs#interactionLike', 'app.bsky.feed.defs#interactionRepost', 'app.bsky.feed.defs#interactionReply', 'app.bsky.feed.defs#interactionQuote']:
            label = 'internal' # positive for this feed
            target_interactions.append((inter['item'], label))

    # Deduplicate by URI (take the most recent or strongest? let's just use a dict, last one wins or conflict resolution)
    # Actually, a single post might have a Like and a requestLess? Unlikely but possible.
    uri_to_label = {}
    for uri, label in target_interactions:
        uri_to_label[uri] = label

    uris_to_fetch = list(uri_to_label.keys())
    logging.info(f"Found {len(uris_to_fetch)} unique posts to fetch.")

    if not uris_to_fetch:
        logging.info("No posts to fetch.")
        return

    # Login to ATProto
    client = Client('https://eurosky.social')
    handle = os.environ.get('BLUESKY_HANDLE', '')
    password = os.environ.get('BLUESKY_PASSWORD', os.environ.get('BLUESKY_APP_PASSWORD', ''))
    client.login(handle, password)

    # Fetch in chunks of 25
    chunk_size = 25
    fetched_posts = {}
    
    for i in range(0, len(uris_to_fetch), chunk_size):
        chunk = uris_to_fetch[i:i+chunk_size]
        try:
            response = client.app.bsky.feed.get_posts({'uris': chunk})
            for p in response.posts:
                fetched_posts[p.uri] = p.record.text
        except Exception as e:
            logging.error(f"Error fetching chunk {i}: {e}")

    # Create dataset
    output_file = 'dataset_interactions.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for uri, label in uri_to_label.items():
            text = fetched_posts.get(uri)
            if text is not None:
                # Synthesize highly confident logprobs
                # e.g., if label == 'internal', internal gets 0.0, others get -10.0
                raw_logprobs = {"internal": -10.0, "external": -10.0, "neutral": -10.0}
                if label in raw_logprobs:
                    raw_logprobs[label] = 0.0
                    
                record = {
                    "uri": uri,
                    "text": text,
                    "loc_label": label,
                    "raw_logprobs": raw_logprobs
                }
                f.write(json.dumps(record) + '\n')
            else:
                logging.warning(f"Could not fetch text for URI: {uri}")

    logging.info(f"Saved {len(fetched_posts)} posts to {output_file}")

if __name__ == '__main__':
    main()
