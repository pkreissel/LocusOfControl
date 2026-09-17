import os
import json
import urllib.request
from dotenv import load_dotenv
import numpy as np
import torch
from transformers import AutoModel

VECTOR_FILE = "internal_vector.npy"
META_FILE = "internal_vector_meta.json"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"
VECTOR_PROB_THRESHOLD = 0.25

def main():
    load_dotenv('bunny_feed/.env')
    load_dotenv('.env') # fallback to root
    ge_api_key = os.environ.get('GREEN_EARTH_API_KEY')
    if not ge_api_key:
        print("Missing GREEN_EARTH_API_KEY in .env")
        return
        
    user_did = "did:plc:xo2uz55ugf2yguhjryahasuf"
    
    with open('reposts_analysis.json', 'r') as f:
        reposts = json.load(f)
    high_scoring_reposts = [p for p in reposts if p['score'] >= VECTOR_PROB_THRESHOLD]
    high_scoring_uris = {p['uri'] for p in high_scoring_reposts}
    
    print("Fetching candidates from GE without seen exclusions...")
    req = urllib.request.Request(
        'https://api.greenearth.social/candidates/generate',
        data=json.dumps({
            "generators": [
                { "name": "followed_users", "weight": 1.0 },
                { "name": "network_likes", "weight": 1.0 },
                { "name": "two_tower", "weight": 1.0 }
            ],
            "infill": "popularity",
            "user_did": user_did,
            "num_candidates": 1000
        }).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'X-API-Key': ge_api_key
        }
    )
    
    with urllib.request.urlopen(req) as response:
        cand_data = json.loads(response.read().decode('utf-8'))
        
    candidates = cand_data.get('candidates', [])
    print(f"Generated {len(candidates)} candidates.")
    
    print("Loading vector metadata and model...")
    with open(META_FILE, 'r') as f:
        meta = json.load(f)
    scale = meta['scale']
    bias = meta['bias']
    internal_vector = np.load(VECTOR_FILE)
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).to(device).eval()
    
    print(f"Embedding and scoring {len(candidates)} candidates...")
    texts = [c.get('content', '') for c in candidates]
    
    batch_size = 64
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [t[:2000] for t in texts[i:i+batch_size]]
            embeddings = model.encode(batch_texts, task="classification")
            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.float().cpu().numpy()
            elif isinstance(embeddings, list):
                embeddings = np.stack([
                    e.float().cpu().numpy() if isinstance(e, torch.Tensor)
                    else np.array(e, dtype=np.float32) for e in embeddings
                ])
            all_embeddings.append(embeddings)
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\\r")
    print()
    embeddings = np.vstack(all_embeddings)
            
    filtered_candidates = []
    cand_uris_set = set()
    for i, c in enumerate(candidates):
        emb = embeddings[i]
        norm = np.linalg.norm(emb)
        if norm < 1e-12: norm = 1e-12
        cos = np.dot(emb / norm, internal_vector)
        prob = 1.0 / (1.0 + np.exp(-(scale * cos + bias)))
        
        if prob >= VECTOR_PROB_THRESHOLD:
            c['score'] = float(prob)
            filtered_candidates.append(c)
            cand_uris_set.add(c['at_uri'])
            
    print(f"{len(filtered_candidates)} candidates survived the Jina > {VECTOR_PROB_THRESHOLD} filter.")
    
    # Inject our 15 high scoring reposts so they are guaranteed to be in the pool
    for p in high_scoring_reposts:
        if p['uri'] not in cand_uris_set:
            filtered_candidates.append({
                "at_uri": p['uri'],
                "score": p['score']
            })
            
    print(f"Total pool size for ranking: {len(filtered_candidates)}")
    
    def test_ranking(model_name):
        print(f"\n--- Ranking with {model_name} + candidate_score ---")
        req = urllib.request.Request(
            'https://api.greenearth.social/rank/predict',
            data=json.dumps({
                "candidates": [{"at_uri": c['at_uri'], "score": c['score']} for c in filtered_candidates],
                "models": [
                    { "name": model_name, "weight": 1.0 },
                    { "name": "candidate_score", "weight": 2.0 }
                ],
                "user_did": user_did
            }).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'X-API-Key': ge_api_key
            }
        )
        with urllib.request.urlopen(req) as response:
            rank_data = json.loads(response.read().decode('utf-8'))
            rankings = rank_data.get('rankings', [])
            
        print("Positions of the 15 high-scoring reposts:")
        results = []
        for i, r in enumerate(rankings):
            uri = r['at_uri']
            if uri in high_scoring_uris:
                orig = next(x for x in high_scoring_reposts if x['uri'] == uri)
                results.append((i+1, orig['score'], uri))
                
        results.sort(key=lambda x: x[0])
        for rank, score, uri in results:
            print(f"Rank #{rank} (out of {len(rankings)}): Jina Score={score:.4f} | {uri}")

    test_ranking("two_tower")
    test_ranking("heavy_ranker")

if __name__ == '__main__':
    main()
