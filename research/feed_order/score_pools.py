"""Score the dumped paid pools locally: the deployed Jina vector and the judge.

1. Jina: jina-embeddings-v5-text-small on local weights, dotted with
   bunny_feed/src/vector.json exactly as vector.ts does (local weights match the
   API to ~6e-4 in probability). This is the feed's own paid score.
2. Judge: decider-2b + the fine-tuned LoRA (research/data/laya_loc/decider_short),
   via research/evals/score_loc_decider.py, rubric short. Gold AP 0.769 / AUC
   0.945 against Gemini -- level with the paid teacher, and not the deployed
   vector, so it can grade the vector's order.

Run from the repo root:
  HF_HOME=/Volumes/Daten/huggingface tmp_venv/bin/python research/feed_order/score_pools.py pools_<date>.jsonl
"""

import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "research", "data", "feed_order")
sys.path.insert(0, os.path.join(ROOT, "research", "feed"))
from score_loc_vector import compute_embeddings  # noqa: E402

ADAPTER = os.path.join(ROOT, "research", "data", "laya_loc", "decider_short")


def main():
    pools = os.path.join(DATA, sys.argv[1])
    stem = os.path.splitext(os.path.basename(pools))[0]
    rows = [json.loads(l) for l in open(pools, encoding="utf-8") if l.strip()]
    uniq = {}
    for r in rows:
        uniq.setdefault(r["uri"], r["text"])
    posts = os.path.join(DATA, f"{stem}_posts.jsonl")
    with open(posts, "w", encoding="utf-8") as f:
        for u, t in uniq.items():
            f.write(json.dumps({"uri": u, "text": t}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} pool rows, {len(uniq)} unique posts")

    jina_out = os.path.join(DATA, f"{stem}_jina.jsonl")
    if not os.path.exists(jina_out):
        vd = json.load(open(os.path.join(ROOT, "bunny_feed", "src", "vector.json")))
        v = np.asarray(vd["vector"], dtype=np.float32)
        uris, texts = list(uniq), list(uniq.values())
        X = compute_embeddings(texts, vd["embedding_model"], vd["embedding_task"])
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        cos = X @ v
        with open(jina_out, "w") as f:
            for u, c in zip(uris, cos):
                p = 1 / (1 + math.exp(-(vd["scale"] * float(c) + vd["bias"])))
                f.write(json.dumps({"uri": u, "jinaScore": p}) + "\n")
        print(f"wrote {jina_out}")

    judge_out = os.path.join(DATA, f"{stem}_judge.jsonl")
    env = dict(os.environ, LOC_DECIDER_RUBRIC="short", LOC_DECIDER_ADAPTER=ADAPTER,
               PYTHONPATH=os.path.join(ROOT, "training"))
    subprocess.run([sys.executable, os.path.join(ROOT, "research", "evals", "score_loc_decider.py"),
                    posts, judge_out], env=env, check=True)


if __name__ == "__main__":
    main()
