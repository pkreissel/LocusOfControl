"""Hyperparameter search for jina-reranker-v3 on a subset of data.

Sweeps the ranking-distillation loss (see train_jina_reranker.py:RankDistillTrainer)
over learning rate, listwise temperature, and the pairwise-term weight.

Run: uv run --with torch --with transformers --with datasets --with scikit-learn \
     --with accelerate --with einops python sweep_jina_reranker.py
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import random
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from scipy.stats import spearmanr
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL_NAME = "jinaai/jina-reranker-v3"
TEACHER_FILE = os.environ.get("LOC_TEACHER_FILE", "dataset_annotated_lmstudio_soft.jsonl")
VAL_FILE = "dataset_validation_gemini.jsonl"
VAL_LABELS_FILE = os.environ.get("LOC_VAL_LABELS_FILE", VAL_FILE)
REFERENCE_T = 1.0
STU_TAU = float(os.environ.get("LOC_STU_TAU", "1.0"))
PAIR_TAU = float(os.environ.get("LOC_PAIR_TAU", "1.0"))
LOSS_BETA = float(os.environ.get("LOC_LOSS_BETA", "0.1"))

LABELS = ["internal", "external", "neutral"]

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def get_expected_score(r, temperature=1.0):
    lp = r.get("loc_score_logprobs")
    if not lp:
        return None
    logits = np.array([lp.get(str(d), -20.0) for d in range(10)]) / temperature
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return float((probs * np.arange(10)).sum())

def get_target_score(r, temperature=1.0):
    # Prefer the graded 0-9 loc_score (v4); fall back to legacy P(int)*E[level] -> [0,9].
    s = get_expected_score(r, temperature)
    if s is not None:
        return s
    if r.get("loc_score") is not None:
        return float(r["loc_score"])

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

    return p_int * exp_lvl * 3.0  # [0,3] -> [0,9]


class RankDistillTrainer(Trainer):
    """Listwise ranking-distillation loss; see train_jina_reranker.py for details."""

    def __init__(self, *args, list_T=1.0, stu_tau=1.0, pair_tau=1.0,
                 alpha=0.5, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_T = list_T
        self.stu_tau = stu_tau
        self.pair_tau = pair_tau
        self.alpha = alpha
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r = inputs.pop("labels").float().view(-1)
        outputs = model(**inputs)
        z = outputs.logits.float().view(-1)
        n = z.shape[0]

        if n >= 2:
            p_t = F.softmax(r / self.list_T, dim=0)
            log_p_s = F.log_softmax(z / self.stu_tau, dim=0)
            loss_list = F.kl_div(log_p_s, p_t, reduction="sum")
        else:
            loss_list = z.new_zeros(())

        loss_pair = z.new_zeros(())
        if self.alpha > 0 and n >= 2:
            iu = torch.triu_indices(n, n, offset=1, device=z.device)
            s_ij = z[iu[0]] - z[iu[1]]
            t_ij = torch.sigmoid((r[iu[0]] - r[iu[1]]) / self.pair_tau)
            loss_pair = F.binary_cross_entropy_with_logits(s_ij, t_ij)

        loss_pt = z.new_zeros(())
        if self.beta > 0:
            loss_pt = F.smooth_l1_loss(z, r)

        loss = loss_list + self.alpha * loss_pair + self.beta * loss_pt
        return (loss, outputs) if return_outputs else loss

def precision_at_k(scores, is_relevant, k):
    top = np.argsort(-scores)[:k]
    return float(is_relevant[top].mean())

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.targets[i], dtype=torch.float32)
        return item

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    val_internal = labels
    auc = float(roc_auc_score(val_internal, predictions))
    ap = float(average_precision_score(val_internal, predictions))
    return {
        "auc": auc,
        "ap": ap,
        "precision@25": precision_at_k(predictions, val_internal, 25)
    }

def main():
    val_uris = {r["uri"] for r in load_jsonl(VAL_FILE)}
    
    train_rows = [r for r in load_jsonl(TEACHER_FILE)
                  if r.get("text") and r.get("raw_logprobs") and "internal" in r.get("raw_logprobs", {}) and r["uri"] not in val_uris]
    
    # Subsample 500 examples for the sweep
    random.seed(42)
    train_rows = random.sample(train_rows, 500)
    print(f"Subsampled {len(train_rows)} posts for hyperparameter sweep.")

    val_rows = [r for r in load_jsonl(VAL_LABELS_FILE) if r.get("text") and r.get("loc_label") in LABELS]
    print(f"Validating on {len(val_rows)} Gemini-labeled posts.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    query = (
        "Does this text demonstrate an internal locus of control? "
        "Look for: a sense of personal agency, constructive solutions, proactive steps, "
        "civic engagement, or attributing outcomes to controllable factors like effort and strategy."
    )
        
    train_texts = [r["text"][:2000] for r in train_rows]
    val_texts = [r["text"][:2000] for r in val_rows]
    
    enc_train = tokenizer([query] * len(train_texts), train_texts, truncation=True, padding=True, max_length=512)
    enc_val = tokenizer([query] * len(val_texts), val_texts, truncation=True, padding=True, max_length=512)
    
    y_train = [get_target_score(r, REFERENCE_T) for r in train_rows]
    y_val = [(1.0 if r["loc_label"] == "internal" else 0.0) for r in val_rows]

    learning_rates = [1e-5, 2e-5, 5e-5]
    list_temps = [0.5, 1.0]      # listwise (ListNet) teacher temperature
    pair_alphas = [0.0, 0.5]     # weight on the pairwise RankNet term (0 = listwise only)

    best_metrics = None
    best_hyperparams = None

    for lr in learning_rates:
        for list_T in list_temps:
            for alpha in pair_alphas:
                print(f"\n--- Testing LR: {lr}, list_T: {list_T}, alpha: {alpha} ---")
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=1, trust_remote_code=True, torch_dtype=torch.float32
                )
                if model.config.pad_token_id is None:
                    model.config.pad_token_id = tokenizer.pad_token_id

                args = TrainingArguments(
                    output_dir="./loc-sweep-tmp",
                    learning_rate=lr,
                    per_device_train_batch_size=int(os.environ.get("LOC_TRAIN_BATCH", "32")),
                    per_device_eval_batch_size=32,
                    num_train_epochs=3,
                    logging_strategy="no",
                    eval_strategy="epoch",
                    save_strategy="no",
                    report_to=[],
                )

                trainer = RankDistillTrainer(
                    model=model,
                    args=args,
                    train_dataset=SimpleDataset(enc_train, y_train),
                    eval_dataset=SimpleDataset(enc_val, y_val),
                    compute_metrics=compute_metrics,
                    list_T=list_T, stu_tau=STU_TAU, pair_tau=PAIR_TAU,
                    alpha=alpha, beta=LOSS_BETA,
                )

                trainer.train()

                preds = trainer.predict(SimpleDataset(enc_val, y_val))
                auc = preds.metrics["test_auc"]
                ap = preds.metrics["test_ap"]
                p25 = preds.metrics["test_precision@25"]

                print(f"Result -> AUC: {auc:.4f} | AP: {ap:.4f} | P@25: {p25:.4f}")

                if best_metrics is None or auc > best_metrics["auc"]:
                    best_metrics = {"auc": auc, "ap": ap, "p25": p25}
                    best_hyperparams = {"lr": lr, "list_T": list_T, "alpha": alpha}

    print("\n=== Best Hyperparameters ===")
    print(f"LR: {best_hyperparams['lr']}, list_T: {best_hyperparams['list_T']}, "
          f"alpha: {best_hyperparams['alpha']}")
    print(f"AUC: {best_metrics['auc']:.4f}, AP: {best_metrics['ap']:.4f}, P@25: {best_metrics['p25']:.4f}")

if __name__ == "__main__":
    main()
