"""Build MiniLM-L12 embedding caches in Green Earth's embedding space.

GE's /candidates/generate returns `minilm_l12_embedding` when called with
`hydrate_embeddings: true`: 384 float32s, L2-normalised, base64. Measured against
300 live candidates, that is all-MiniLM-L12-v2, mean-pooled, truncated at 128
tokens — cosine 1.0000 (min 0.9995) against the local model with those settings.
So a probe trained on caches built here scores live candidates for free at the
edge, with no embedding API call at all.

train_internal_vector.py computes embeddings with the *Jina* model whenever a
cache is missing, so a MiniLM cache must always be pre-built here. Row filtering
mirrors that script exactly, which is what its `len(rows) == len(X)` assertions
check.

Run (train + interactions + validation caches for the bonsai-v2 corpus):
  LOC_TEACHER_FILE=dataset_annotated_bonsai_v2.jsonl \
  LOC_EMB_CACHE=embeddings_bonsai_v2_minilm.npz \
  LOC_INT_EMB_CACHE=embeddings_interactions_minilm.npz \
  LOC_VAL_EMB_CACHE=embeddings_gemini_minilm_v2.npz \
  uv run --with numpy --with torch --with transformers --with tqdm python embed_minilm.py

Set LOC_EMBED_ONLY=synth:<file>.jsonl to embed a counterexample file instead
(no filtering beyond text/raw_logprobs, matching create_synthetic_counterexamples.py).
"""

import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import train_internal_vector as T

MODEL = "sentence-transformers/all-MiniLM-L12-v2"
# GE truncates here. Going longer silently produces a *different* vector for the
# 0.8% of candidates over the window, which would then not match what the edge sees.
MAX_LEN = 128

TRAIN_OUT = os.environ.get("LOC_EMB_CACHE", "embeddings_bonsai_v2_minilm.npz")
INT_OUT = os.environ.get("LOC_INT_EMB_CACHE", "embeddings_interactions_minilm.npz")
VAL_OUT = os.environ.get("LOC_VAL_EMB_CACHE", "embeddings_gemini_minilm_v2.npz")
EMBED_ONLY = os.environ.get("LOC_EMBED_ONLY", "")


def embed(texts, batch_size=64):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = tok(texts[i:i + batch_size], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(e, dim=1).numpy())
            print(f"\r  {min(i + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    return np.vstack(out).astype(np.float32)


def main():
    if EMBED_ONLY.startswith("synth:"):
        path = EMBED_ONLY.split(":", 1)[1]
        val_uris = {r["uri"] for r in T.load_jsonl(T.GEMINI_FILE)}
        rows = [r for r in T.load_jsonl(path)
                if r.get("text") and r.get("raw_logprobs")
                and r["uri"] not in val_uris and r.get("source_uri") not in val_uris]
        print(f"{path}: {len(rows)} counterexamples")
        np.savez_compressed(TRAIN_OUT, X=embed([r["text"] for r in rows]))
        print(f"Saved {TRAIN_OUT}")
        return

    rows = T.load_train_rows()
    print(f"teacher rows: {len(rows)} -> {TRAIN_OUT}")
    X = embed([r["text"] for r in rows])
    y = np.array([[T.get_p_positive(r, T.CACHE_TEMPERATURE),
                   T.get_expected_level(r, T.CACHE_TEMPERATURE)] for r in rows])
    np.savez_compressed(TRAIN_OUT, X=X, y=y)

    int_rows = T.load_interaction_rows()
    if int_rows:
        print(f"interaction rows: {len(int_rows)} -> {INT_OUT}")
        np.savez_compressed(INT_OUT, X=embed([r["text"] for r in int_rows]))

    val_rows = T.load_val_rows()
    print(f"validation rows: {len(val_rows)} -> {VAL_OUT}")
    np.savez_compressed(VAL_OUT, X=embed([r["text"] for r in val_rows]),
                        y=np.array([r["loc_label"] for r in val_rows]))
    print("Done. Train with LOC_EMB_CACHE / LOC_INT_EMB_CACHE / LOC_VAL_EMB_CACHE "
          "pointed at these files.")


if __name__ == "__main__":
    main()
