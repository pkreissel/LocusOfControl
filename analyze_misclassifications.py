import json
import numpy as np
import os
import torch
from transformers import AutoModel

def main():
    # Load the v3 vector
    vector_path = 'internal_vector.npy'
    if not os.path.exists(vector_path):
        print(f"Error: {vector_path} not found.")
        return
        
    v = np.load(vector_path)

    # Load interaction rows
    dataset_path = 'dataset_interactions.jsonl'
    rows = []
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 'external' means user clicked requestLess
                if item.get("loc_label") == "external":
                    rows.append(item)
    
    if not rows:
        print("No 'requestLess' (external) interactions found in the dataset.")
        return

    print(f"Loaded {len(rows)} requestLess (external) interactions.")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}...")
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v5-text-small", trust_remote_code=True).to(device).eval()

    print("Computing embeddings...")
    with torch.no_grad():
        texts = [r["text"][:2000] for r in rows]
        emb = model.encode(texts, task="classification")
        if isinstance(emb, torch.Tensor):
            emb = emb.float().cpu().numpy()
        elif isinstance(emb, list):
            emb = np.stack([e.float().cpu().numpy() if isinstance(e, torch.Tensor) else np.array(e, dtype=np.float32) for e in emb])
        emb = np.asarray(emb, dtype=np.float32)

    # normalize embeddings to unit length (as during training)
    emb = np.nan_to_num(emb)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    # Compute dot product scores
    scores = emb @ v

    print("\n" + "="*80)
    print("MISCLASSIFIED POSTS (High Internal Score for 'requestLess' items)")
    print("="*80)
    
    # Sort by descending score (highest = most wrongly classified as 'internal')
    sorted_indices = np.argsort(-scores)
    
    for i in sorted_indices:
        score = scores[i]
        # Only show posts that got a fairly high score (e.g. > 0.0) 
        # meaning the model thought they were "internal"
        if score > 0.0:
            print(f"Score: {score:.4f}")
            print(f"URI: {rows[i].get('uri')}")
            print(f"Text:\n{rows[i]['text']}")
            print("-" * 80)
            
if __name__ == '__main__':
    main()
