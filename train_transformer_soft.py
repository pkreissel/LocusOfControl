"""Fine-tune xlm-roberta on teacher soft labels (temperature-scaled logprobs).

Trains on LOC_TEACHER_FILE (default: the bonsai v2 dataset), excluding all
validation URIs, with soft cross-entropy against softmax(raw_logprobs / T).
Evaluates on the Gemini validation labels (LOC_VAL_LABELS_FILE if set, else the
v1 validation file).

Env: LOC_TEACHER_FILE, LOC_VAL_LABELS_FILE, LOC_OUTPUT_DIR, LOC_SOFT_T (default 2.0)

Run: uv run --with torch --with transformers --with datasets --with scikit-learn \
     --with accelerate python train_transformer_soft.py
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL_NAME = "xlm-roberta-base"
TEACHER_FILE = os.environ.get("LOC_TEACHER_FILE", "dataset_annotated_bonsai_v2.jsonl")
VAL_FILE = "dataset_validation_gemini.jsonl"
VAL_LABELS_FILE = os.environ.get("LOC_VAL_LABELS_FILE", VAL_FILE)
OUTPUT_DIR = os.environ.get("LOC_OUTPUT_DIR", "./loc-classifier-bonsai-v2")
SOFT_T = float(os.environ.get("LOC_SOFT_T", "2.0"))

LABELS = ["internal", "external", "neutral"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def soft_target(raw_logprobs, T):
    logits = np.array([raw_logprobs[l] for l in LABELS]) / T
    e = np.exp(logits - logits.max())
    return (e / e.sum()).tolist()


FOCAL_GAMMA = float(os.environ.get("LOC_FOCAL_GAMMA", "2.0"))
FOCAL_ALPHA_INTERNAL = float(os.environ.get("LOC_FOCAL_ALPHA_INTERNAL", "2.0"))

class SoftTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        logits = outputs.logits
        p = torch.softmax(logits, dim=-1)
        log_p = torch.log_softmax(logits, dim=-1)
        
        # Alpha weighting for class imbalance: Internal (0), External (1), Neutral (2)
        alpha = torch.tensor([FOCAL_ALPHA_INTERNAL, 1.0, 0.5], device=logits.device)
        
        focal_term = alpha * (1 - p) ** FOCAL_GAMMA
        loss = -torch.sum(labels * focal_term * log_p, dim=-1).mean()
        
        return (loss, outputs) if return_outputs else loss


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


def main():
    val_uris = {r["uri"] for r in load_jsonl(VAL_FILE)}
    train_rows = [r for r in load_jsonl(TEACHER_FILE)
                  if r.get("text") and r.get("raw_logprobs") and r["uri"] not in val_uris]
    print(f"Training on {len(train_rows)} posts (T={SOFT_T}) -> {OUTPUT_DIR}")

    val_rows = [r for r in load_jsonl(VAL_LABELS_FILE)
                if r.get("text") and r.get("loc_label") in LABELS]
    print(f"Validating on {len(val_rows)} Gemini-labeled posts from {VAL_LABELS_FILE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc_train = tokenizer([r["text"][:2000] for r in train_rows],
                          truncation=True, padding=True, max_length=256)
    enc_val = tokenizer([r["text"][:2000] for r in val_rows],
                        truncation=True, padding=True, max_length=256)
    y_train = [soft_target(r["raw_logprobs"], SOFT_T) for r in train_rows]
    # one-hot for eval; SoftTrainer's loss is only used for training
    y_val = [np.eye(3)[LABEL_MAP[r["loc_label"]]].tolist() for r in val_rows]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3,
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL_MAP)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )
    trainer = SoftTrainer(model=model, args=args,
                          train_dataset=SimpleDataset(enc_train, y_train))
    trainer.train()

    preds = trainer.predict(SimpleDataset(enc_val, y_val))
    y_pred = preds.predictions.argmax(-1)
    y_true = np.array([LABEL_MAP[r["loc_label"]] for r in val_rows])
    print("\n=== Validation (Gemini labels) ===")
    print(f"accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"kappa:    {cohen_kappa_score(y_true, y_pred):.3f}")
    print(f"macro-F1: {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"internal-F1: {f1_score(y_true, y_pred, labels=[0], average='macro'):.3f}")

    trainer.save_model(OUTPUT_DIR + "/final_model")
    tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")
    # save raw probabilities for downstream ranking comparison
    probs = torch.softmax(torch.tensor(preds.predictions), dim=-1).numpy()
    np.savez_compressed(OUTPUT_DIR + "/val_predictions.npz",
                        probs=probs, uris=np.array([r["uri"] for r in val_rows]))
    print(f"Saved model + val predictions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
