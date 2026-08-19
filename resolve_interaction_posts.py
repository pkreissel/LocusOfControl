import os
import json
import logging
from dotenv import load_dotenv
from atproto import Client
from openai import OpenAI
import csv

# Import the teacher annotation logic
from create_synthetic_counterexamples import teacher_annotate

logging.basicConfig(level=logging.INFO)

API_BASE = os.environ.get("LOC_API_BASE", "http://127.0.0.1:8080/v1")
DISAGREEMENTS_LOG = "backend/disagreements_log.csv"

def main():
    load_dotenv()
    
    # Init LLM client
    llm_client = OpenAI(base_url=API_BASE, api_key="test-key")
    
    # Load interactions
    interactions_file = 'bunny_feed/interactions.json'
    if not os.path.exists(interactions_file):
        logging.error(f"{interactions_file} not found.")
        return

    with open(interactions_file, 'r', encoding='utf-8') as f:
        interactions = json.load(f)

    # We want to map requestLess to external (label=0.0)
    target_interactions = []
    for inter in interactions:
        evt = inter.get('event')
        if evt == 'app.bsky.feed.defs#requestLess':
            label = 'external' # negative for this feed
            target_interactions.append((inter['item'], label))
        elif evt in ['app.bsky.feed.defs#interactionLike', 'app.bsky.feed.defs#interactionRepost', 'app.bsky.feed.defs#interactionReply', 'app.bsky.feed.defs#interactionQuote']:
            label = 'internal' # positive for this feed
            target_interactions.append((inter['item'], label))

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

    # Prepare disagreements CSV
    os.makedirs(os.path.dirname(DISAGREEMENTS_LOG), exist_ok=True)
    file_exists = os.path.isfile(DISAGREEMENTS_LOG)
    
    output_file = 'dataset_interactions.jsonl'
    success_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f, \
         open(DISAGREEMENTS_LOG, 'a', encoding='utf-8', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['URI', 'User Interaction', 'Teacher Label', 'Text'])
            
        for uri, user_label in uri_to_label.items():
            text = fetched_posts.get(uri)
            if text is not None:
                try:
                    # Pure teacher annotation
                    lp, soft, teacher_label, lvl_lp, soft_lvl, loc_lvl = teacher_annotate(llm_client, text)
                    
                    if user_label != teacher_label:
                        logging.warning(f"DISAGREEMENT on {uri}: User said {user_label}, Teacher said {teacher_label}")
                        writer.writerow([uri, user_label, teacher_label, text])
                    
                    record = {
                        "uri": uri,
                        "text": text,
                        "loc_label": user_label, # Preserve the user's action for reference
                        "teacher_label": teacher_label,
                        "raw_logprobs": lp,
                        "soft_labels": soft,
                        "level_logprobs": lvl_lp,
                        "soft_levels": soft_lvl,
                        "loc_level": loc_lvl,
                        "is_interaction": True
                    }
                    f.write(json.dumps(record) + '\n')
                    success_count += 1
                except Exception as e:
                    logging.error(f"Failed to annotate {uri}: {e}")
            else:
                logging.warning(f"Could not fetch text for URI: {uri}")

    logging.info(f"Saved {success_count} posts with pure teacher annotations to {output_file}")

if __name__ == '__main__':
    main()
