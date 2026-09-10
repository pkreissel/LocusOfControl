"""Score the LoC training corpus with a multilingual emotion classifier.

Runs tabularisai/multilingual-emotion-classification (XLM-R base, 11 emotions,
multi-label sigmoid heads) over the exact 5000 posts the deployed internal
vector was trained on, and pairs each post's emotion probabilities with the
locus-of-control probability the live edge worker would give it.

Row order and filtering mirror load_train_rows() in train_internal_vector.py, so
the cached Jina embeddings in embeddings_lmstudio_soft_v3_jina.npz line up
one-to-one with the jsonl rows. Scoring reproduces bunny_feed/src/vector.ts:
L2-normalize, dot the vector, sigmoid(scale * cos + bias).

Writes emotions_loc_corpus.npz (emotion probs + LoC scores + gate flags).

Run: tmp_venv/bin/python score_emotions_loc.py
"""

import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TEACHER_FILE = "dataset_annotated_lmstudio_soft.jsonl"
VAL_FILE = "dataset_validation_gemini.jsonl"
EMB_CACHE = "embeddings_lmstudio_soft_v3_jina.npz"
VECTOR_JSON = "bunny_feed/src/vector.json"
EMOTION_MODEL = "tabularisai/multilingual-emotion-classification"
OUT = "emotions_loc_corpus.npz"

# Live feed gate (bunny_feed/src/feeds.ts): probability >= 0.88, minWords 6.
THRESHOLD = 0.88
MIN_WORDS = 6

URL_RE = re.compile(r"(?:https?://|www\.)\S+")
WORDY_RE = re.compile(r"[^\W_]", re.UNICODE)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_train_rows():
    """Same filter as train_internal_vector.load_train_rows (LoC axis)."""
    val_uris = {item["uri"] for item in load_jsonl(VAL_FILE)}
    rows = []
    for item in load_jsonl(TEACHER_FILE):
        if item.get("uri") in val_uris:
            continue
        if not item.get("text"):
            continue
        lp = item.get("raw_logprobs")
        if not lp or "internal" not in lp:
            continue
        rows.append(item)
    return rows


def has_enough_text(text, min_words=MIN_WORDS):
    """Port of hasEnoughText() in bunny_feed/src/index.ts."""
    stripped = URL_RE.sub(" ", text or "")
    words = chars = 0
    for tok in stripped.split():
        if not WORDY_RE.search(tok):
            continue
        words += 1
        chars += sum(1 for ch in tok if WORDY_RE.match(ch))
    return words >= min_words or chars >= min_words * 5


def loc_scores(rows):
    data = np.load(EMB_CACHE, allow_pickle=True)
    X = np.nan_to_num(data["X"]).astype(np.float64)
    assert len(rows) == len(X), f"{len(rows)} rows vs {len(X)} embeddings"
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    with open(VECTOR_JSON) as f:
        vec = json.load(f)
    cos = X @ np.asarray(vec["vector"], dtype=np.float64)
    prob = 1.0 / (1.0 + np.exp(-(vec["scale"] * cos + vec["bias"])))
    return cos, prob


def emotion_scores(texts, batch_size=64):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    model.to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"{EMOTION_MODEL} on {device}: {len(labels)} labels {labels}")

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=256, return_tensors="pt").to(device)
            logits = model(**batch).logits
            # Multi-label: independent sigmoid heads, tau = 0.5 (per model card).
            out.append(torch.sigmoid(logits).float().cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(out), labels


def main():
    rows = load_train_rows()
    texts = [r["text"] for r in rows]
    print(f"{len(rows)} training rows")

    cos, prob = loc_scores(rows)
    long_enough = np.array([has_enough_text(t) for t in texts])
    print(f"LoC prob: mean {prob.mean():.3f}  >= {THRESHOLD}: {(prob >= THRESHOLD).sum()}"
          f"  minWords pass: {long_enough.sum()}"
          f"  both: {((prob >= THRESHOLD) & long_enough).sum()}")

    if os.path.exists(OUT):
        print(f"{OUT} exists, reusing emotion scores")
        cached = np.load(OUT, allow_pickle=True)
        E, labels = cached["E"], list(cached["labels"])
    else:
        E, labels = emotion_scores(texts)

    np.savez_compressed(
        OUT, E=E, labels=np.array(labels), cos=cos, prob=prob,
        long_enough=long_enough,
        loc_label=np.array([r.get("loc_label", "") for r in rows]),
        loc_level=np.array([r.get("loc_level") or 0 for r in rows]),
        uri=np.array([r["uri"] for r in rows]),
        lang=np.array([(r.get("langs") or [""])[0] for r in rows]),
    )
    print(f"Wrote {OUT}: E {E.shape}")


if __name__ == "__main__":
    main()
