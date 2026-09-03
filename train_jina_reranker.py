"""Fine-tune jina-reranker-v3 to RANK posts by locus of control.

Uses the SAME learnings as the production vector (train_internal_vector.py): a
listwise ranking-distillation loss (ListNet KL + soft-pairwise RankNet + Huber
anchor) over each batch, trained against the BAND target external(0) < neutral(2)
< internal(6-9). The band structure is what pushes external/doomer content to the
bottom while keeping internal detection; raw graded loc_score instead lets the
neutral bulk crowd the top. Evaluation/selection is on graded metrics (ndcg_graded,
ext_leak_auc) not just internal-AUC. Train on the aug (base+#synth) file.

Run: LOC_TEACHER_FILE=dataset_annotated_lmstudio_soft_aug.jsonl \
     uv run --with torch --with transformers --with datasets --with scikit-learn \
     --with accelerate --with einops --with peft python train_jina_reranker.py
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from scipy.stats import spearmanr
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL_NAME = "jinaai/jina-reranker-v3"
TEACHER_FILE = os.environ.get("LOC_TEACHER_FILE", "dataset_annotated_lmstudio_soft.jsonl")
VAL_FILE = "dataset_validation_gemini.jsonl"
VAL_LABELS_FILE = os.environ.get("LOC_VAL_LABELS_FILE", VAL_FILE)
OUTPUT_DIR = os.environ.get("LOC_OUTPUT_DIR", "./loc-reranker-jina-v3")
REFERENCE_T = 1.0

# Ranking-distillation loss hyperparameters (env-overridable for sweeps).
LIST_T = float(os.environ.get("LOC_LIST_T", "1.0"))    # teacher softmax temperature over the batch
STU_TAU = float(os.environ.get("LOC_STU_TAU", "1.0"))  # student softmax temperature
PAIR_TAU = float(os.environ.get("LOC_PAIR_TAU", "1.0")) # RankNet soft-preference temperature
LOSS_ALPHA = float(os.environ.get("LOC_LOSS_ALPHA", "0.5"))  # weight on pairwise RankNet term
LOSS_BETA = float(os.environ.get("LOC_LOSS_BETA", "0.1"))    # weight on pointwise Huber anchor

# Target mode (see train_internal_vector.py — same learning). band (default) ranks
# external(0) < neutral(NEU_BAND) < internal(6-9); this is what pushes external down
# and stops the neutral bulk from crowding the top. graded=raw loc_score; legacy=old.
TARGET_MODE = os.environ.get("LOC_TARGET_MODE", "band")
NEU_BAND = float(os.environ.get("LOC_NEU_BAND", "2.0"))

LABELS = ["internal", "external", "neutral"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def get_expected_score(r, temperature=1.0):
    """v4 graded 0-9 loc_score target: expected value over the teacher's digit
    distribution, re-softmaxed at `temperature`. None when the row lacks it."""
    lp = r.get("loc_score_logprobs")
    if not lp:
        return None
    logits = np.array([lp.get(str(d), -20.0) for d in range(10)]) / temperature
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return float((probs * np.arange(10)).sum())


def get_class(r, temperature=1.0):
    lp = r.get("raw_logprobs")
    if lp:
        a = np.array([lp.get(l, -20.0) for l in LABELS]) / temperature
        return LABELS[int(a.argmax())]
    return r.get("loc_label") if r.get("loc_label") in LABELS else "neutral"


def get_expected_level(r, temperature=1.0):
    lp = r.get("level_logprobs")
    if not lp:
        return 1.0
    logits = np.array([lp.get(str(i), -20.0) for i in (1, 2, 3)]) / temperature
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    return float(p[0]*1 + p[1]*2 + p[2]*3)


def get_internal_intensity(r, temperature=1.0):
    """0-9 intensity for an internal post: smooth v4 loc_score when available,
    else mapped from the 3-level concreteness rubric (1->6, 2->7.5, 3->9)."""
    s = get_expected_score(r, temperature)
    if s is not None:
        return s
    if r.get("loc_score") is not None:
        return float(r["loc_score"])
    return 6.0 + 1.5 * (get_expected_level(r, temperature) - 1.0)


def _legacy_target(r, temperature):
    lp = r.get("raw_logprobs", {})
    a = np.array([lp.get(l, -20.0) for l in LABELS]) / temperature
    e = np.exp(a - a.max()); p_int = float(e[0] / e.sum())
    return p_int * get_expected_level(r, temperature) * 3.0  # [0,3] -> [0,9]


def get_target_score(r, temperature=1.0):
    if TARGET_MODE == "graded":
        s = get_expected_score(r, temperature)
        if s is not None:
            return s
        if r.get("loc_score") is not None:
            return float(r["loc_score"])
        return _legacy_target(r, temperature)
    if TARGET_MODE == "legacy":
        return _legacy_target(r, temperature)
    # band (default): external -> 0, neutral -> NEU_BAND, internal -> intensity (6-9)
    c = get_class(r, temperature)
    if c == "internal":
        return get_internal_intensity(r, temperature)
    if c == "neutral":
        return NEU_BAND
    return 0.0

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
        # For regression (num_labels=1), HuggingFace expects labels to be floats
        item["labels"] = torch.tensor(self.targets[i], dtype=torch.float32)
        return item

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()

    # labels carry graded relevance: internal=2, neutral=1, external=0.
    rel = labels
    is_int = (rel == 2).astype(int)
    is_ext = (rel == 0).astype(int)

    m = {
        "auc": float(roc_auc_score(is_int, predictions)),        # internal detection
        "ap": float(average_precision_score(is_int, predictions)),
        "precision@25": precision_at_k(predictions, is_int, 25),
        "ndcg_graded": float(ndcg_score(rel[None, :], predictions[None, :])),
        "spearman_graded": float(spearmanr(predictions, rel).statistic),
    }
    if 0 < is_ext.sum() < len(is_ext):
        # external exclusion: LOWER is better (external pushed down)
        m["ext_leak_auc"] = float(roc_auc_score(is_ext, predictions))
        top = np.argsort(-predictions)[:25]
        m["ext_in_top25"] = int(is_ext[top].sum())
    return m

class RankDistillTrainer(Trainer):
    """Listwise ranking-distillation loss over each minibatch.

    Teacher relevances r (the graded loc_score) and student logits z:
      L_list = KL( softmax(r / LIST_T) || softmax(z / STU_TAU) )   (ListNet)
      L_pair = BCE( sigmoid(z_i - z_j), sigmoid((r_i - r_j)/PAIR_TAU) )  (soft RankNet)
      L_pt   = Huber(z, r)                                          (calibration anchor)
      L = L_list + alpha * L_pair + beta * L_pt

    The listwise term teaches the model to reproduce the teacher's ordering and,
    with a small LIST_T, to focus on the top of the batch (matching precision@k).
    """

    def __init__(self, *args, list_T=1.0, stu_tau=1.0, pair_tau=1.0,
                 alpha=0.5, beta=0.1, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_T = list_T
        self.stu_tau = stu_tau
        self.pair_tau = pair_tau
        self.alpha = alpha
        self.beta = beta
        self.sample_weights = sample_weights

    def _get_train_sampler(self, *args, **kwargs):
        # Class-balanced sampling so each batch reliably contains internal posts
        # (~6% of the data) — otherwise the listwise loss has nothing internal to
        # rank up and internal detection stays weak. See sample_weights in main().
        if self.sample_weights is not None:
            w = torch.as_tensor(self.sample_weights, dtype=torch.double)
            return torch.utils.data.WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r = inputs.pop("labels").float().view(-1)          # teacher relevances
        outputs = model(**inputs)
        z = outputs.logits.float().view(-1)                # student scores
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


def main():
    val_uris = {r["uri"] for r in load_jsonl(VAL_FILE)}
    
    # Load training data
    train_rows = [r for r in load_jsonl(TEACHER_FILE)
                  if r.get("text") and r.get("raw_logprobs") and "internal" in r.get("raw_logprobs", {}) and r["uri"] not in val_uris]
    
    print(f"Training on {len(train_rows)} posts -> {OUTPUT_DIR}")

    # Load validation data
    val_rows = [r for r in load_jsonl(VAL_LABELS_FILE)
                if r.get("text") and r.get("loc_label") in LABELS]
    print(f"Validating on {len(val_rows)} Gemini-labeled posts")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    query = (
        "Does this text demonstrate an internal locus of control? "
        "Look for: a sense of personal agency, constructive solutions, proactive steps, "
        "civic engagement, or attributing outcomes to controllable factors like effort and strategy."
    )
        
    max_len = int(os.environ.get("LOC_MAX_LEN", "512"))
    train_texts = [r["text"][:2000] for r in train_rows]
    val_texts = [r["text"][:2000] for r in val_rows]

    enc_train = tokenizer([query] * len(train_texts), train_texts, truncation=True, padding=True, max_length=max_len)
    enc_val = tokenizer([query] * len(val_texts), val_texts, truncation=True, padding=True, max_length=max_len)

    y_train = [get_target_score(r, REFERENCE_T) for r in train_rows]
    # Validation labels carry GRADED relevance (internal=2, neutral=1, external=0)
    # so compute_metrics can score detection + external-exclusion + graded ranking.
    _rel = {"internal": 2.0, "neutral": 1.0, "external": 0.0}
    y_val = [_rel[r["loc_label"]] for r in val_rows]

    # Class-balanced sampling weights to counter the ~6% internal imbalance.
    # weight = (N / count[class]) ** power; power in [0,1] (0=off, 1=full balance).
    # Default 0.5 (sqrt) oversamples internal ~3x without severe over-repetition.
    balance_power = float(os.environ.get("LOC_BALANCE_POWER", "0.5"))
    train_classes = [get_class(r, REFERENCE_T) for r in train_rows]
    from collections import Counter
    cnt = Counter(train_classes); N = len(train_classes)
    sample_weights = [(N / cnt[c]) ** balance_power for c in train_classes]
    print(f"class counts {dict(cnt)}; balance_power={balance_power}")

    # For jina-reranker-v3 on MPS, we explicitly use float32 to prevent bfloat16/float32 mixed ops crashes
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        # Batch 8 keeps jina-reranker-v3 (Qwen3) within MPS memory (~2s/step, ~1h total);
        # batch 32 overflows and thrashes to ~210s/step. Bigger batch = more in-batch
        # comparisons for the listwise loss, so raise LOC_TRAIN_BATCH only on a CUDA GPU.
        per_device_train_batch_size=int(os.environ.get("LOC_TRAIN_BATCH", "8")),
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=os.environ.get("LOC_SELECT_METRIC", "ndcg_graded"),
        greater_is_better=True,
        report_to=[],
    )

    trainer = RankDistillTrainer(
        model=model,
        args=args,
        train_dataset=SimpleDataset(enc_train, y_train),
        eval_dataset=SimpleDataset(enc_val, y_val),
        compute_metrics=compute_metrics,
        list_T=LIST_T, stu_tau=STU_TAU, pair_tau=PAIR_TAU,
        alpha=LOSS_ALPHA, beta=LOSS_BETA, sample_weights=sample_weights,
    )
    
    trainer.train()

    preds = trainer.predict(SimpleDataset(enc_val, y_val))
    predictions = preds.predictions.squeeze()
    
    print("\n=== Validation Results ===")
    for k, v in preds.metrics.items():
        print(f"{k}: {v:.4f}")

    trainer.save_model(OUTPUT_DIR + "/final_model")
    tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")
    
    np.savez_compressed(OUTPUT_DIR + "/val_predictions.npz",
                        probs=predictions, uris=np.array([r["uri"] for r in val_rows]))
    print(f"Saved model + val predictions to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
