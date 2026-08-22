import asyncio
import json
import os
import time
import traceback
import csv
import urllib.request
import urllib.parse
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from atproto import Client
from dotenv import load_dotenv

from vector_scorer import VectorScorer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model pipeline globally
MODEL_PATH = os.environ.get("LOC_CLASSIFIER_PATH", "../loc-classifier-bonsai-v2/final_model")
try:
    classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)
    print(f"Loaded classifier from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading classifier: {e}")
    classifier = None

# Distilled query-vector scorer (see train_internal_vector.py)
try:
    vector_scorer = VectorScorer("../internal_vector.npy", "../internal_vector_meta.json")
    print("Loaded vector scorer")
except Exception as e:
    print(f"Error loading vector scorer: {e}")
    vector_scorer = None

CLASSIFIER_THRESHOLD = 0.15
VECTOR_PROB_THRESHOLD = float(os.environ.get("VECTOR_PROB_THRESHOLD", "0.25"))

@app.get("/api/posts")
async def get_posts(method: str = "classifier"):
    if method not in ("classifier", "vector", "both"):
        method = "classifier"
    use_classifier = method in ("classifier", "both")
    use_vector = method in ("vector", "both")

    missing = None
    if use_classifier and not classifier:
        missing = "Classifier not loaded"
    elif use_vector and not vector_scorer:
        missing = "Vector scorer not loaded"
    if missing:
        async def mock_stream():
            yield f"data: {json.dumps({'error': missing})}\n\n"
        return StreamingResponse(mock_stream(), media_type="text/event-stream")

    async def event_generator():
        try:
            # Load env vars
            print("Event generator started! Loading env vars...")
            load_dotenv("../.env")
            handle = os.environ.get('BSKY_HANDLE') or os.environ.get('BLUESKY_HANDLE')
            password = os.environ.get('BSKY_APP_PASSWORD') or os.environ.get('BLUESKY_APP_PASSWORD')
            pds_url = os.environ.get('BSKY_PDS_URL', 'https://bsky.social')
            
            print(f"Loaded credentials: handle={handle is not None}, password={password is not None}")
            
            if not handle or not password:
                yield f"data: {{\"error\": \"Missing credentials\"}}\n\n"
                return

            client = Client(base_url=pds_url)
            print("Logging into Bluesky...")
            client.login(handle, password)
            print("Login successful!")
            
            # Green Earth integration
            ge_api_key = os.environ.get('GREEN_EARTH_API_KEY')
            if not ge_api_key:
                load_dotenv("../bunny_feed/.env")
                ge_api_key = os.environ.get('GREEN_EARTH_API_KEY')
                
            if not ge_api_key:
                yield f"data: {{\"error\": \"Missing GREEN_EARTH_API_KEY\"}}\n\n"
                return

            print("Fetching candidates from Green Earth...")
            gen_req = urllib.request.Request(
                'https://api.greenearth.social/candidates/generate',
                data=json.dumps({
                    "generators": [
                        { "name": "followed_users", "weight": 1.0 },
                        { "name": "network_likes", "weight": 1.0 },
                        { "name": "two_tower", "weight": 1.0 }
                    ],
                    "infill": "popularity",
                    "user_did": client.me.did,
                    "num_candidates": 200  # Limiting to 200 for acceptable latency during local scoring
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'X-API-Key': ge_api_key}
            )
            with urllib.request.urlopen(gen_req) as gen_res:
                cand_data = json.loads(gen_res.read().decode('utf-8'))
                
            candidates = cand_data.get('candidates', [])
            print(f"Got {len(candidates)} candidates.")
            
            texts = [c.get('content', '') for c in candidates]
            
            print(f"Scoring {len(texts)} candidates locally...")
            results = classifier(texts) if use_classifier else [None] * len(texts)
            vector_results = vector_scorer.score(texts) if use_vector else [None] * len(texts)
            
            filtered_candidates = []
            for i, c in enumerate(candidates):
                should_keep = False
                n_int = 0.0
                v_int = 0.0
                top_label = "unknown"
                external_score = 0.0
                
                if use_classifier and results[i]:
                    top_score = 0.0
                    for r in results[i]:
                        if r['score'] > top_score:
                            top_score = r['score']
                            top_label = r['label']
                        if r['label'] in ['internal', '0', 'LABEL_0']:
                            n_int = r['score']
                        elif r['label'] in ['external', '1', 'LABEL_1']:
                            external_score = r['score']
                    if n_int >= CLASSIFIER_THRESHOLD or external_score >= CLASSIFIER_THRESHOLD:
                        should_keep = True
                        
                if use_vector and vector_results[i]:
                    v_int = vector_results[i]['internalProb']
                    if v_int >= VECTOR_PROB_THRESHOLD:
                        should_keep = True
                        
                c['score'] = v_int if use_vector else n_int
                c['local_scores'] = {
                    'internalScore': n_int,
                    'externalScore': external_score,
                    'label': top_label,
                    'vectorInternalScore': v_int,
                    'vectorCosine': vector_results[i]['cosine'] if use_vector and vector_results[i] else 0.0,
                    'should_yield': should_keep
                }
                
                if should_keep:
                    filtered_candidates.append(c)

            print(f"Filtered down to {len(filtered_candidates)} candidates.")
            
            if filtered_candidates:
                print("Ranking via Green Earth...")
                rank_req = urllib.request.Request(
                    'https://api.greenearth.social/rank/predict',
                    data=json.dumps({
                        "candidates": filtered_candidates,
                        "models": [
                            { "name": "heavy_ranker", "weight": 1.0 }
                        ],
                        "user_did": client.me.did
                    }).encode('utf-8'),
                    headers={'Content-Type': 'application/json', 'X-API-Key': ge_api_key}
                )
                try:
                    with urllib.request.urlopen(rank_req) as rank_res:
                        rank_data = json.loads(rank_res.read().decode('utf-8'))
                        
                    rankings = rank_data.get('rankings', [])
                    
                    # Map the rank_score (heavy ranker score) back to candidates
                    rank_score_map = { r['at_uri']: r.get('rank_score', 0) for r in rankings }
                    for c in filtered_candidates:
                        c['heavy_ranker_score'] = rank_score_map.get(c['at_uri'], 0)
                        
                    if rankings:
                        ranked_uris = [r['at_uri'] for r in rankings]
                        filtered_candidates.sort(key=lambda x: ranked_uris.index(x['at_uri']) if x['at_uri'] in ranked_uris else 9999)
                except Exception as e:
                    print(f"Green Earth ranking failed: {e}")
                    for c in filtered_candidates:
                        c['heavy_ranker_score'] = 0
            
            print("Hydrating and yielding...")
            posts_processed = 0
            chunk_size = 25
            
            for i in range(0, len(filtered_candidates), chunk_size):
                chunk = filtered_candidates[i:i+chunk_size]
                uris = [c['at_uri'] for c in chunk]
                if not uris:
                    continue
                    
                response = client.app.bsky.feed.get_posts({'uris': uris})
                hydrated_map = {p.uri: p for p in response.posts}
                
                for c in chunk:
                    post = hydrated_map.get(c['at_uri'])
                    if not post:
                        continue
                        
                    scores = c['local_scores']
                    post_data = {
                        "did": post.author.did,
                        "handle": getattr(post.author, 'handle', None),
                        "displayName": getattr(post.author, 'display_name', None),
                        "avatar": getattr(post.author, 'avatar', None),
                        "uri": post.uri,
                        "text": getattr(post.record, 'text', ''),
                        "createdAt": getattr(post.record, 'created_at', ''),
                        "internalScore": scores['internalScore'],
                        "externalScore": scores['externalScore'],
                        "label": scores['label'],
                        "vectorCosine": scores['vectorCosine'],
                        "vectorInternalScore": scores['vectorInternalScore'],
                        "heavyRankerScore": c.get('heavy_ranker_score', 0)
                    }
                    
                    yield f"data: {json.dumps(post_data)}\n\n"
                    posts_processed += 1
                    
                    file_exists = os.path.isfile("disagreements_log.csv")
                    with open("disagreements_log.csv", "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["uri", "text", "nli_internal", "vector_internal", "disagreement_type"])
                        
                        n_int = post_data['internalScore']
                        v_int = post_data['vectorInternalScore']
                        
                        n_catch = n_int >= CLASSIFIER_THRESHOLD
                        v_catch = v_int >= VECTOR_PROB_THRESHOLD
                        
                        if n_catch and not v_catch:
                            dtype = "Vector_Miss"
                        elif not n_catch and v_catch:
                            dtype = "NLI_Miss"
                        elif n_catch and v_catch:
                            dtype = "Both_Agree"
                        else:
                            dtype = "None"
                            
                        writer.writerow([
                            post_data['uri'],
                            post_data['text'].replace('\n', ' '),
                            round(n_int, 4),
                            round(v_int, 4),
                            dtype
                        ])
                        
                await asyncio.sleep(0.1)
                
                if posts_processed >= 100:
                    break
                    
            yield "event: end\ndata: {}\n\n"
            
        except Exception as e:
            traceback.print_exc()
            error_msg = json.dumps({"error": str(e) or repr(e)})
            yield f"data: {error_msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
