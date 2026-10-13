"""Embed dataset_raw_net20k.jsonl with jina-embeddings-v5-text-small
(task=clustering) and cache the result, keyed by uri.

Uses the clustering task adapter, not the classification one the deployed LoC
vector trains on -- a different embedding space, so this cache is not
interchangeable with embeddings_*_jina.npz elsewhere in training/data/.

Resumable: writes a single .npz at the end, but batches are embedded in order
so a re-run after a crash just starts over (20k posts embeds in one sitting on
MPS, no need for row-level resume here).

Run: ../../tmp_venv/bin/python embed_net20k_jina.py
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW_FILE = os.path.join(ROOT, "training", "data", "dataset_raw_net20k.jsonl")
DATA_DIR = os.path.join(ROOT, "research", "data", "topic_model")
OUT_FILE = os.path.join(DATA_DIR, "embeddings_net20k_jina_clustering.npz")

MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"
TASK = "clustering"
MAX_CHARS = 2000
BATCH_SIZE = int(os.environ.get("LOC_EMB_BATCH", "32"))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    rows = [r for r in load_jsonl(RAW_FILE) if r.get("text", "").strip()]
    print(f"{len(rows)} rows with non-empty text (of {sum(1 for _ in open(RAW_FILE))})")
    uris = [r["uri"] for r in rows]
    texts = [r["text"] for r in rows]

    import torch
    from tqdm import tqdm
    from transformers import AutoModel

    torch.manual_seed(42)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding {len(texts)} texts with {MODEL_NAME} (task={TASK}) on {device}...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device).eval()

    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = [t[:MAX_CHARS] for t in texts[i:i + BATCH_SIZE]]
            emb = model.encode(batch, task=TASK)
            if isinstance(emb, torch.Tensor):
                emb = emb.float().cpu().numpy()
            out.append(np.asarray(emb, dtype=np.float32))
    X = np.concatenate(out, axis=0)
    print(f"done in {time.time() - t0:.0f}s, X shape {X.shape}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(OUT_FILE, X=X, uris=np.array(uris, dtype=object))
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
