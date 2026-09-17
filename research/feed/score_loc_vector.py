"""Score posts with the deployed LoC vector, locally, and cache the result.

Reproduces exactly what bunny_feed does per candidate — embed the text with
jina-embeddings-v5-text-small (task=classification, first 2000 chars), L2
normalize, dot the unit vector, calibrate — but against the local model weights
instead of the Jina API, so a cached score here is the number the feed would
have assigned.

Three numbers are cached per post, because only one of them is comparable to the
teacher:

  vector_cos    raw cosine against the trained direction.
  vector_score  scale * cos + bias. The vector is a Ridge regression fitted to
                the teacher's target on the 0-9 locus-of-control axis, and since
                `scale` is ||w|| and the stored vector is w/||w||, this product
                IS that regression's prediction. This is the column to put next
                to the teacher's loc_score.
  vector_prob   sigmoid(vector_score) — what feeds.ts thresholds (LoC: 0.88).
                A squashed [0,9] prediction, not a probability of anything, and
                saturated over most of the axis: it is here because it is what
                the deployment compares, not because it is comparable.

Resumable: one appended line per post, URIs already present are skipped, and the
embeddings are cached in an .npz so re-scoring after a retrain costs no GPU time.

Run:
  uv run --with numpy --with torch --with transformers --with einops --with peft \
     --with tqdm python score_loc_vector.py --posts feed_posts_<tag>.jsonl \
     --out scores_loc_vector_<tag>.jsonl
"""

import argparse
import json
import os
import struct

import numpy as np

VECTOR_FILE = os.environ.get("LOC_VECTOR", "internal_vector.npy")
META_FILE = os.environ.get("LOC_VECTOR_META", "")
EMBEDDING_MODEL = os.environ.get("LOC_EMBEDDING_MODEL", "")
EMBEDDING_TASK = os.environ.get("LOC_EMBEDDING_TASK", "")
# Same truncation the deployed embedder applies (bunny_feed/src/jina.ts).
MAX_CHARS = 2000
BATCH_SIZE = int(os.environ.get("LOC_EMB_BATCH", "32"))
SEED = 42


def fingerprint(nums):
    """FNV-1a over each float32's bit pattern — the port of vector.ts's
    `fingerprint`, so a cached row can be matched against the version string the
    deployment keys post_scores by. Hash of [...vector, scale, bias]."""
    h = 0x811c9dc5
    for n in nums:
        for b in struct.pack("<f", np.float32(n)):
            h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def load_vector():
    v = np.load(VECTOR_FILE).astype(np.float32)
    meta_path = META_FILE or (VECTOR_FILE[:-4] if VECTOR_FILE.endswith(".npy")
                              else VECTOR_FILE) + "_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    # The vector is stored unit-norm by the trainer; renormalize defensively so
    # `scale * cos + bias` really is the Ridge prediction.
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, meta, meta_path


def compute_embeddings(texts, model_name, task, batch_size=BATCH_SIZE):
    """Identical to train_internal_vector.compute_embeddings, kept here so
    scoring does not drag in the training module (and its dataset paths)."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModel

    torch.manual_seed(SEED)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding {len(texts)} texts with {model_name} (task={task}) on {device}...")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            emb = model.encode([t[:MAX_CHARS] for t in texts[i:i + batch_size]], task=task)
            if isinstance(emb, torch.Tensor):
                emb = emb.float().cpu().numpy()
            elif isinstance(emb, list):
                emb = np.stack([
                    e.float().cpu().numpy() if isinstance(e, torch.Tensor)
                    else np.array(e, dtype=np.float32) for e in emb
                ])
            out.append(np.asarray(emb, dtype=np.float32))
    return np.vstack(out)


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
    ap.add_argument("--posts", default=os.environ.get("LOC_FEED_OUT", "feed_posts.jsonl"))
    ap.add_argument("--out", default=os.environ.get("LOC_VECTOR_SCORES", "scores_loc_vector.jsonl"))
    ap.add_argument("--emb-cache", default=os.environ.get("LOC_FEED_EMB_CACHE", ""))
    args = ap.parse_args()
    emb_cache = args.emb_cache or args.out.replace(".jsonl", "_emb.npz")

    v, meta, meta_path = load_vector()
    model_name = EMBEDDING_MODEL or meta.get("embedding_model")
    task = EMBEDDING_TASK or meta.get("embedding_task", "classification")
    scale, bias = float(meta["scale"]), float(meta["bias"])
    version = fingerprint(list(np.load(VECTOR_FILE).astype(np.float32)) + [scale, bias])
    print(f"Vector: {VECTOR_FILE} ({len(v)}-dim) + {meta_path}")
    print(f"  scale={scale:.4f} bias={bias:.4f} version={version} space={model_name}")

    posts = [r for r in load_jsonl(args.posts) if (r.get("text") or "").strip()]
    done = {r["uri"]: r for r in load_jsonl(args.out)
            if r.get("vector_version") == version}
    todo = [r for r in posts if r["uri"] not in done]
    print(f"{len(posts)} posts, {len(done)} already scored by this vector, {len(todo)} to go.")
    if not todo:
        return

    cached = {}
    if os.path.exists(emb_cache):
        z = np.load(emb_cache, allow_pickle=False)
        uris, X = list(z["uris"]), z["X"]
        if X.shape[1] == len(v):
            cached = {u: X[i] for i, u in enumerate(uris)}
            print(f"{len(cached)} embeddings reused from {emb_cache}.")
        else:
            print(f"Ignoring {emb_cache}: {X.shape[1]}-dim, vector is {len(v)}-dim.")

    missing = [r for r in todo if r["uri"] not in cached]
    if missing:
        X_new = compute_embeddings([r["text"] for r in missing], model_name, task)
        for r, x in zip(missing, X_new):
            cached[r["uri"]] = x
        uris = sorted(cached)
        np.savez_compressed(emb_cache, uris=np.array(uris),
                            X=np.stack([cached[u] for u in uris]))
        print(f"Wrote {len(uris)} embeddings to {emb_cache}.")

    with open(args.out, "a", encoding="utf-8") as f:
        for r in todo:
            x = cached[r["uri"]].astype(np.float32)
            n = float(np.linalg.norm(x))
            cos = float(x @ v / max(n, 1e-12))
            score = scale * cos + bias
            f.write(json.dumps({
                "uri": r["uri"],
                "vector_cos": cos,
                "vector_score": score,
                "vector_prob": float(1.0 / (1.0 + np.exp(-score))),
                "vector_version": version,
                "vector_file": VECTOR_FILE,
                "embedding_model": model_name,
                "embedding_task": task,
                "truncated": len(r["text"]) > MAX_CHARS,
            }) + "\n")
    rows = [r for r in load_jsonl(args.out) if r.get("vector_version") == version]
    s = np.array([r["vector_score"] for r in rows])
    print(f"\n{args.out}: {len(rows)} posts scored by version {version}.")
    print(f"  vector_score (0-9 axis): min {s.min():.2f} p50 {np.median(s):.2f} "
          f"max {s.max():.2f}")
    p = 1.0 / (1.0 + np.exp(-s))
    print(f"  vector_prob: p50 {np.median(p):.4f}, above deployed 0.88 gate: "
          f"{int((p >= 0.88).sum())}/{len(p)}")


if __name__ == "__main__":
    main()
