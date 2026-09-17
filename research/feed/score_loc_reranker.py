"""Score posts with the fine-tuned jina-reranker-v3 LoC ranker, and cache it.

Feeds each post through the same query/document pair the model was trained on
(train_jina_reranker.py builds the identical encoding), so the cached number is
the model's own score rather than an approximation of it.

What is cached is `rr_logit`, the regression head's raw output and nothing else.
Trained mostly by a listwise ranking loss, it is a *ranking* score: on the gold
validation set it spans -0.07 to 0.94, nowhere near the 0-9 locus-of-control
axis the vector predicts. Putting it on that axis is a projection fitted against
gold labels, which belongs with the comparison rather than in the cache —
report_loc_feed.py does it for this model and the vector alike, so the two are
mapped the same way and a refitted mapping does not invalidate a single row here.

Resumable: appends one line per post and skips URIs already cached for this
model directory.

Run:
  uv run --with numpy --with torch --with transformers --with einops \
     --with peft python score_loc_reranker.py \
     --posts feed_posts_<tag>.jsonl --out scores_loc_reranker_<tag>.jsonl
"""

import argparse
import json
import os

import numpy as np

MODEL_DIR = os.environ.get("LOC_RERANKER_DIR", "loc-reranker-jina-v3/final_model")
MAX_LEN = int(os.environ.get("LOC_MAX_LEN", "512"))
MAX_CHARS = 2000
BATCH_SIZE = int(os.environ.get("LOC_RERANK_BATCH", "8"))
# Byte-for-byte the query train_jina_reranker.py paired every document with.
QUERY = (
    "Does this text demonstrate an internal locus of control? "
    "Look for: a sense of personal agency, constructive solutions, proactive steps, "
    "civic engagement, or attributing outcomes to controllable factors like effort and strategy."
)
def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default="feed_posts.jsonl")
    ap.add_argument("--out", default="scores_loc_reranker.jsonl")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    posts = [r for r in load_jsonl(args.posts) if (r.get("text") or "").strip()]
    done = {r["uri"] for r in load_jsonl(args.out) if r.get("model_dir") == MODEL_DIR}
    todo = [r for r in posts if r["uri"] not in done]
    print(f"{len(posts)} posts, {len(done)} already scored by {MODEL_DIR}, {len(todo)} to go.")
    if not todo:
        return

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # float32 for the same reason training used it: mixed bfloat16/float32 ops
    # crash this model on MPS.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=1, trust_remote_code=True, torch_dtype=torch.float32,
    ).to(device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    print(f"Scoring on {device}, batch {BATCH_SIZE}, max_len {MAX_LEN}...")

    from tqdm import tqdm
    with open(args.out, "a", encoding="utf-8") as f, torch.no_grad():
        for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Reranking"):
            batch = todo[i:i + BATCH_SIZE]
            texts = [r["text"][:MAX_CHARS] for r in batch]
            enc = tok([QUERY] * len(texts), texts, truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            logits = model(**enc).logits.float().squeeze(-1).cpu().numpy().reshape(-1)
            for r, z in zip(batch, logits):
                f.write(json.dumps({
                    "uri": r["uri"],
                    "rr_logit": float(z),
                    "model_dir": MODEL_DIR,
                    "max_len": MAX_LEN,
                    "truncated": len(r["text"]) > MAX_CHARS,
                }) + "\n")
            f.flush()

    rows = [r for r in load_jsonl(args.out) if r.get("model_dir") == MODEL_DIR]
    z = np.array([r["rr_logit"] for r in rows])
    print(f"\n{args.out}: {len(rows)} posts scored.")
    print(f"  rr_logit: min {z.min():.3f} p25 {np.percentile(z, 25):.3f} "
          f"p50 {np.median(z):.3f} p75 {np.percentile(z, 75):.3f} max {z.max():.3f}")


if __name__ == "__main__":
    main()
