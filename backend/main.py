import asyncio
import json
import os
import time
import traceback
import csv
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
            
            feed_uri = "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/with-friends"
            
            posts_processed = 0
            cursor = None
            
            while posts_processed < 100:
                print(f"Fetching discover batch. Cursor: {cursor}")
                response = client.app.bsky.feed.get_feed({'feed': feed_uri, 'limit': 100, 'cursor': cursor})
                if not response.feed:
                    print("No feed items returned. Breaking.")
                    break
                    
                cursor = response.cursor
                
                # Extract texts for batch processing
                batch = []
                for item in response.feed:
                    if hasattr(item.post.record, 'text'):
                        # get rkey from uri
                        parts = item.post.uri.split('/')
                        rkey = parts[-1]
                        
                        batch.append({
                            "did": item.post.author.did,
                            "handle": getattr(item.post.author, 'handle', None),
                            "displayName": getattr(item.post.author, 'display_name', None),
                            "avatar": getattr(item.post.author, 'avatar', None),
                            "uri": item.post.uri,
                            "text": item.post.record.text,
                            "createdAt": getattr(item.post.record, 'created_at', ''),
                        })
                
                if not batch:
                    continue
                    
                texts = [p['text'] for p in batch]
                results = classifier(texts) if use_classifier else [None] * len(batch)
                vector_results = vector_scorer.score(texts) if use_vector else [None] * len(batch)

                for i, res in enumerate(results):
                    posts_processed += 1
                    post_data = batch[i]
                    should_yield = False

                    if use_classifier:
                        internal_score = 0.0
                        external_score = 0.0
                        top_label = "unknown"
                        top_score = 0.0

                        for r in res:
                            if r['score'] > top_score:
                                top_score = r['score']
                                top_label = r['label']
                            if r['label'] in ['internal', '0', 'LABEL_0']:
                                internal_score = r['score']
                            elif r['label'] in ['external', '1', 'LABEL_1']:
                                external_score = r['score']

                        post_data['internalScore'] = internal_score
                        post_data['externalScore'] = external_score
                        post_data['label'] = top_label
                        # Yield if either score is relatively high
                        if internal_score >= CLASSIFIER_THRESHOLD or external_score >= CLASSIFIER_THRESHOLD:
                            should_yield = True

                    if use_vector:
                        vec = vector_results[i]
                        post_data['vectorCosine'] = vec['cosine']
                        post_data['vectorInternalScore'] = vec['internalProb']
                        if vec['internalProb'] >= VECTOR_PROB_THRESHOLD:
                            should_yield = True

                    if should_yield:
                        yield f"data: {json.dumps(post_data)}\n\n"
                        
                        file_exists = os.path.isfile("disagreements_log.csv")
                        with open("disagreements_log.csv", "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["uri", "text", "nli_internal", "vector_internal", "disagreement_type"])
                            
                            n_int = post_data.get('internalScore', 0)
                            v_int = post_data.get('vectorInternalScore', 0)
                            
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
                                post_data.get('uri', ''),
                                post_data.get('text', '').replace('\n', ' '),
                                round(n_int, 4),
                                round(v_int, 4),
                                dtype
                            ])
                        
                # Small sleep to yield to event loop and not hit rate limits too hard
                await asyncio.sleep(0.5)
                
                if posts_processed >= 100:
                    break
                    
            yield "event: end\ndata: {}\n\n"
            
        except Exception as e:
            traceback.print_exc()
            error_msg = json.dumps({"error": str(e) or repr(e)})
            yield f"data: {error_msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
