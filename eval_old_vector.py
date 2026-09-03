import json
import numpy as np
import os
from sklearn.metrics import average_precision_score, roc_auc_score

def precision_at_k(scores, is_relevant, k):
    top = np.argsort(-scores)[:k]
    return float(is_relevant[top].mean())

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# Load old vector from git
os.system("git show aa2ece5:bunny_feed/src/vector.json > old_vector.json")
with open("old_vector.json") as f:
    data = json.load(f)
v = np.array(data["vector"])
v = v / np.linalg.norm(v)

# Load validation embeddings
val_data = np.load("embeddings_gemini_jina.npz", allow_pickle=True)
X_val, y_val = val_data["X"], val_data["y"]
X_val = np.nan_to_num(X_val)
X_val /= np.linalg.norm(X_val, axis=1, keepdims=True)

VALID_LABELS = ["internal", "external", "neutral"]
LABEL_MAP = {"internal": 0, "external": 1, "neutral": 2}
if isinstance(y_val[0], str):
    y_val = np.array([LABEL_MAP.get(l, 2) for l in y_val])

GEMINI_FILE = "dataset_validation_gemini.jsonl"
val_rows = []
for item in load_jsonl(GEMINI_FILE):
    if not item.get("text"): continue
    if item.get("loc_label") in VALID_LABELS:
        val_rows.append(item)

# Ensure same logic as train_internal_vector.py
val_internal = (y_val == LABEL_MAP["internal"]).astype(int)

s_val = X_val @ v

print("=== Old Vector Validation ===")
print(f"AUC: {roc_auc_score(val_internal, s_val):.4f}")
print(f"AP:  {average_precision_score(val_internal, s_val):.4f}")
print(f"P@10: {precision_at_k(s_val, val_internal, 10):.4f}")
print(f"P@25: {precision_at_k(s_val, val_internal, 25):.4f}")
print(f"P@50: {precision_at_k(s_val, val_internal, 50):.4f}")
