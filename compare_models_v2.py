"""Three-way comparison on the 500 Gemini-v2-labeled validation posts.

Rankers (internal-vs-rest): old vector (tess v1), new vector (bonsai v2),
transformer p_internal, teacher p_internal.
Classifiers (3-class vs Gemini v2): teacher hard labels, transformer argmax.

Run: uv run --with numpy --with scikit-learn python compare_models_v2.py
"""

import json

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             f1_score, roc_auc_score)

LABELS = ["internal", "external", "neutral"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def p_internal(raw_logprobs, T=1.0):
    logits = np.array([raw_logprobs[l] for l in LABELS]) / T
    e = np.exp(logits - logits.max())
    return e[0] / e.sum()


def precision_at_k(scores, rel, k):
    return float(rel[np.argsort(-scores)[:k]].mean())


def rank_row(name, scores, rel):
    print(f"{name:28s} AUC={roc_auc_score(rel, scores):.3f} "
          f"AP={average_precision_score(rel, scores):.3f} "
          f"P@10={precision_at_k(scores, rel, 10):.2f} "
          f"P@25={precision_at_k(scores, rel, 25):.2f} "
          f"P@50={precision_at_k(scores, rel, 50):.2f}")


def clf_row(name, y_true, y_pred):
    print(f"{name:28s} acc={accuracy_score(y_true, y_pred):.3f} "
          f"kappa={cohen_kappa_score(y_true, y_pred):.3f} "
          f"macroF1={f1_score(y_true, y_pred, average='macro'):.3f} "
          f"internalF1={f1_score(y_true, y_pred, labels=[0], average='macro'):.3f}")


def main():
    gem = {r["uri"]: r["loc_label"] for r in load_jsonl("dataset_validation_gemini_v2.jsonl")}
    teacher = {r["uri"]: r for r in load_jsonl("dataset_annotated_bonsai_v2.jsonl")}

    # embedding rows are ordered like the v1 validation file
    val_rows = [r for r in load_jsonl("dataset_validation_gemini.jsonl")
                if r.get("text") and r.get("loc_label") in LABELS]
    X_val = np.nan_to_num(np.load("embeddings_gemini_jina.npz", allow_pickle=True)["X"])
    X_val /= np.linalg.norm(X_val, axis=1, keepdims=True)
    assert len(val_rows) == len(X_val)

    tp = np.load("loc-classifier-bonsai-v2/val_predictions.npz", allow_pickle=True)
    t_probs = {u: p for u, p in zip(tp["uris"], tp["probs"])}

    keep = [i for i, r in enumerate(val_rows)
            if r["uri"] in gem and r["uri"] in teacher and r["uri"] in t_probs]
    uris = [val_rows[i]["uri"] for i in keep]
    X = X_val[keep]
    y_true = np.array([LABEL_MAP[gem[u]] for u in uris])
    rel = (y_true == 0).astype(int)
    print(f"n={len(uris)} validation posts, {rel.sum()} internal ({rel.mean():.1%})\n")

    print("=== Ranking internal-vs-rest (Gemini v2 relevance) ===")
    rank_row("vector OLD (tess v1)", X @ np.load("internal_vector.npy"), rel)
    rank_row("vector NEW (bonsai v2)", X @ np.load("internal_vector_bonsai.npy"), rel)
    rank_row("transformer p_internal", np.array([t_probs[u][0] for u in uris]), rel)
    rank_row("teacher p_internal", np.array([p_internal(teacher[u]["raw_logprobs"]) for u in uris]), rel)

    print("\n=== 3-class classification vs Gemini v2 ===")
    clf_row("teacher hard (bonsai)", y_true,
            np.array([LABEL_MAP[teacher[u]["loc_label"]] for u in uris]))
    clf_row("transformer argmax", y_true,
            np.array([int(np.argmax(t_probs[u])) for u in uris]))


if __name__ == "__main__":
    main()
