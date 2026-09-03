import json
import numpy as np

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    # Load vectors and meta
    int_vec = np.load("internal_vector.npy")
    ext_vec = np.load("external_vector.npy")
    
    with open("internal_vector_meta.json", "r") as f:
        int_meta = json.load(f)
    with open("external_vector_meta.json", "r") as f:
        ext_meta = json.load(f)

    # Load validation data
    val_data = np.load("embeddings_gemini_jina.npz", allow_pickle=True)
    X_val = val_data["X"]
    X_val = np.nan_to_num(X_val)
    X_val /= np.linalg.norm(X_val, axis=1, keepdims=True)

    val_rows = []
    VALID_LABELS = ["internal", "external", "neutral"]
    for item in load_jsonl("dataset_validation_gemini.jsonl"):
        if not item.get("text"):
            continue
        if item.get("loc_label") in VALID_LABELS:
            val_rows.append(item)
            
    assert len(val_rows) == len(X_val), "Row mismatch!"

    results = []
    for i, row in enumerate(val_rows):
        emb = X_val[i]
        
        # Internal score
        cos_int = np.dot(emb, int_vec)
        prob_int = 1.0 / (1.0 + np.exp(-(int_meta['scale'] * cos_int + int_meta['bias'])))
        
        # External score
        cos_ext = np.dot(emb, ext_vec)
        prob_ext = 1.0 / (1.0 + np.exp(-(ext_meta['scale'] * cos_ext + ext_meta['bias'])))
        
        # Criteria: internal > external
        if prob_int > prob_ext:
            results.append({
                'text': row['text'],
                'prob_int': prob_int,
                'prob_ext': prob_ext,
                'sum_score': prob_int + prob_ext,
                'loc_label': row.get('loc_label')
            })

    # Sort by sum_score descending
    results.sort(key=lambda x: x['sum_score'], reverse=True)

    print(f"Found {len(results)} posts matching the criteria.")
    for i, res in enumerate(results[:10]):
        print(f"\n--- Post {i+1} ---")
        print(f"Internal: {res['prob_int']:.4f} | External: {res['prob_ext']:.4f} | Label: {res['loc_label']}")
        print(res['text'])

if __name__ == "__main__":
    main()
