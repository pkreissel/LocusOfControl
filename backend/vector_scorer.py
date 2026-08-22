"""Cosine scoring of posts against the distilled internal-LoC query vector.

Embeds texts exactly the way the vector was trained (same Jina model, same
task, same truncation, L2-normalized) and returns both the raw cosine score
and the calibrated probability sigmoid(scale * cos + bias) from
internal_vector_meta.json.
"""

import json

import numpy as np
import torch
from transformers import AutoModel

torch.manual_seed(42)
if hasattr(torch, "use_deterministic_algorithms"):
    torch.use_deterministic_algorithms(True, warn_only=True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class VectorScorer:
    def __init__(self, vector_path, meta_path):
        self.vector = np.load(vector_path)
        with open(meta_path) as f:
            meta = json.load(f)
        self.scale = meta["scale"]
        self.bias = meta["bias"]
        self.task = meta["embedding_task"]
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            meta["embedding_model"], torch_dtype=torch.float32, trust_remote_code=True
        ).to(self.device).eval()

    def score(self, texts):
        """Return a list of {'cosine': float, 'internalProb': float} per text."""
        with torch.no_grad():
            emb = self.model.encode([t[:2000] for t in texts], task=self.task)
        if isinstance(emb, torch.Tensor):
            emb = emb.float().cpu().numpy()
        elif isinstance(emb, list):
            emb = np.stack([
                e.float().cpu().numpy() if isinstance(e, torch.Tensor)
                else np.array(e, dtype=np.float32)
                for e in emb
            ])
        else:
            emb = np.array(emb, dtype=np.float32)
        emb = np.nan_to_num(emb)
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        cos = emb @ self.vector
        prob = 1.0 / (1.0 + np.exp(-(self.scale * cos + self.bias)))
        return [
            {"cosine": float(c), "internalProb": float(p)}
            for c, p in zip(cos, prob)
        ]
