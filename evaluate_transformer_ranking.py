"""Evaluate transformer soft predictions on ranking benchmarks.

Computes Spearman, NDCG, AUC, AP, and Precision@K against the Gemini validation labels
and the teacher's soft targets.

Run: uv run --with numpy --with scipy --with scikit-learn python evaluate_transformer_ranking.py
"""

import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
import os

VAL_FILE = "dataset_validation_gemini.jsonl"
VAL_LABELS_FILE = os.environ.get("LOC_VAL_LABELS_FILE", VAL_FILE)
TEACHER_FILE = "dataset_annotated_lmstudio_soft.jsonl"
TRANSFORMER_OUTPUT = os.environ.get("TRANSFORMER_OUTPUT", "./loc-classifier-bonsai-v2")

LABELS = ["internal", "external", "neutral"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def get_target_score(r, temperature=1.0):
    # Same logic as train_internal_vector.py
    raw_logprobs = r.get("raw_logprobs", {})
    logits = np.array([raw_logprobs.get(l, -20.0) for l in LABELS]) / temperature
    e = np.exp(logits - logits.max())
    p_int = float(e[0] / e.sum())
    
    level_logprobs = r.get("level_logprobs")
    if not level_logprobs:
        exp_lvl = 1.0
    else:
        logits = np.array([level_logprobs.get(str(i), -20.0) for i in (1, 2, 3)]) / temperature
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        exp_lvl = float(probs[0]*1 + probs[1]*2 + probs[2]*3)
        
    return p_int * exp_lvl

def precision_at_k(scores, is_relevant, k):
    top = np.argsort(-scores)[:k]
    return float(is_relevant[top].mean())

def main():
    if not os.path.exists(f"{TRANSFORMER_OUTPUT}/val_predictions.npz"):
        print(f"Error: {TRANSFORMER_OUTPUT}/val_predictions.npz not found.")
        return
        
    data = np.load(f"{TRANSFORMER_OUTPUT}/val_predictions.npz", allow_pickle=True)
    probs = data["probs"]
    uris = data["uris"]
    
    # Internal class probability is at index 0 (LABEL_MAP["internal"] == 0)
    s_pred = probs[:, 0]
    
    val_rows = [r for r in load_jsonl(VAL_LABELS_FILE) if r.get("text") and r.get("loc_label") in LABELS]
    # Filter just in case the order got weird, but it shouldn't
    uri_to_score = {uri: score for uri, score in zip(uris, s_pred)}
    
    # Create the sorted aligned arrays
    s_val = np.array([uri_to_score[r["uri"]] for r in val_rows if r["uri"] in uri_to_score])
    val_rows = [r for r in val_rows if r["uri"] in uri_to_score]
    y_val = np.array([LABEL_MAP[r["loc_label"]] for r in val_rows])
    val_internal = (y_val == LABEL_MAP["internal"]).astype(int)
    
    print(f"Loaded {len(s_val)} predictions.")
    
    # Teacher targets for correlation
    teacher_by_uri = {
        item["uri"]: item
        for item in load_jsonl(TEACHER_FILE)
        if item.get("raw_logprobs") and "internal" in item.get("raw_logprobs", {})
    }
    
    teacher_mask = np.array([r["uri"] in teacher_by_uri for r in val_rows])
    p_val_teacher = np.array([
        get_target_score(teacher_by_uri[r["uri"]], 1.0)
        if r["uri"] in teacher_by_uri else np.nan
        for r in val_rows
    ])
    
    m = {}
    m["val_auc_internal"] = float(roc_auc_score(val_internal, s_val))
    m["val_avg_precision"] = float(average_precision_score(val_internal, s_val))
    for k in (10, 25, 50):
        m[f"val_precision@{k}"] = precision_at_k(s_val, val_internal, k)
        
    if teacher_mask.sum() > 10:
        m["val_spearman_vs_teacher"] = float(
            spearmanr(s_val[teacher_mask], p_val_teacher[teacher_mask]).statistic
        )
        
    print("\n=== Transformer Ranking Validation ===")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
        
if __name__ == "__main__":
    main()
