import os
import json
import datetime
from datetime import timezone
import numpy as np
import torch
from transformers import AutoModel
from dotenv import load_dotenv
from atproto import Client

VECTOR_FILE = "internal_vector.npy"
META_FILE = "internal_vector_meta.json"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"

def main():
    load_dotenv()
    handle = os.environ.get('BLUESKY_HANDLE') or os.environ.get('BSKY_HANDLE')
    app_password = os.environ.get('BLUESKY_APP_PASSWORD') or os.environ.get('BSKY_APP_PASSWORD')
    pds_url = os.environ.get('BLUESKY_PDS_URL') or os.environ.get('BSKY_PDS_URL') or 'https://bsky.social'
    
    if not handle or not app_password:
        print("Missing credentials in .env")
        return
        
    print(f"Logging in as {handle}...")
    client = Client(base_url=pds_url)
    client.login(handle, app_password)
    
    did = client.me.did
    print(f"User DID: {did}")
    
    print("Fetching author feed...")
    # Get all posts from the author (including reposts)
    feed = client.app.bsky.feed.get_author_feed({'actor': did, 'limit': 100})
    
    now = datetime.datetime.now(timezone.utc)
    seven_days_ago = now - datetime.timedelta(days=7)
    
    target_posts = []
    
    for item in feed.feed:
        # Check if it's a repost
        reason = item.reason
        if reason and hasattr(reason, '$type') and reason['$type'] == 'app.bsky.feed.defs#reasonRepost':
            # or reason.__class__.__name__ == 'ReasonRepost'
            pass
        elif reason and type(reason).__name__ == 'ReasonRepost':
            pass
        else:
            continue
            
        repost_time_str = getattr(reason, 'indexed_at', None) or getattr(reason, 'indexedAt', None)
        if repost_time_str:
            repost_time = datetime.datetime.fromisoformat(repost_time_str.replace('Z', '+00:00'))
            if repost_time < seven_days_ago:
                continue
                
        post = item.post
        if post:
            record = post.record
            post_created_str = getattr(record, 'created_at', None) or getattr(record, 'createdAt', None)
            
            if post_created_str:
                try:
                    post_created = datetime.datetime.fromisoformat(post_created_str.replace('Z', '+00:00'))
                    if post_created < seven_days_ago:
                        continue
                except Exception:
                    pass
            
            text = getattr(record, 'text', '')
            if not text:
                continue
                
            target_posts.append({
                'uri': post.uri,
                'text': text,
                'author': post.author.handle,
                'repost_time': repost_time_str,
                'post_created': post_created_str
            })
            
    print(f"Found {len(target_posts)} reposts matching criteria.")
    if not target_posts:
        return
        
    print("Loading vector metadata...")
    with open(META_FILE, 'r') as f:
        meta = json.load(f)
    scale = meta['scale']
    bias = meta['bias']
    
    print("Loading internal vector...")
    internal_vector = np.load(VECTOR_FILE)
    
    print("Loading Jina model...")
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).eval()
    
    texts = [p['text'] for p in target_posts]
    
    print("Computing embeddings...")
    with torch.no_grad():
        embeddings = model.encode([t[:2000] for t in texts], task="classification")
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.float().cpu().numpy()
        elif isinstance(embeddings, list):
            embeddings = np.stack([
                e.float().cpu().numpy() if isinstance(e, torch.Tensor)
                else np.array(e, dtype=np.float32) for e in embeddings
            ])
            
    print("Scoring...")
    results = []
    for i, p in enumerate(target_posts):
        emb = embeddings[i]
        norm = np.linalg.norm(emb)
        if norm < 1e-12:
            norm = 1e-12
        cos = np.dot(emb / norm, internal_vector)
        prob = 1.0 / (1.0 + np.exp(-(scale * cos + bias)))
        p['score'] = float(prob)
        p['cosine'] = float(cos)
        results.append(p)
        
    results.sort(key=lambda x: x['score'], reverse=True)
    
    with open('reposts_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    for p in results:
        print(f"---")
        print(f"Score: {p['score']:.4f}")
        print(f"Author: {p['author']}")
        print(f"Text:\n{p['text']}")
        print(f"URI: {p['uri']}")

if __name__ == '__main__':
    main()
