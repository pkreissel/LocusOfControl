import os
os.environ["HF_HOME"] = "./.cache" # use local cache to avoid polluting global

import torch
from transformers import AutoModel
import numpy as np

EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"

def get_emb():
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).eval()
    with torch.no_grad():
        emb = model.encode(["This is a test document to check reproducibility."] * 5, task="classification")
    return emb

emb1 = get_emb()
emb2 = get_emb()

diff = np.abs(emb1 - emb2).max()
print(f"Max diff without seed: {diff}")

torch.manual_seed(42)
emb3 = get_emb()
torch.manual_seed(42)
emb4 = get_emb()

diff2 = np.abs(emb3 - emb4).max()
print(f"Max diff with seed: {diff2}")
