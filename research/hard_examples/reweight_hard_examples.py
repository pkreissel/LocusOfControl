"""Upweight the corpus rows a vector gets most wrong, and see whether it helps.

The idea: the vector is a linear probe, and its errors against its own teacher
are concentrated -- on live traffic, teacher-external posts scored high enough
to be served (the leak). Weighting those rows up in the Ridge fit asks the
probe to spend its capacity there.

Procedure, set up so the choice cannot select noise on the gold set:

  1. Hold out DEV_FRAC of the corpus (teacher labels) as a dev set.
  2. On the rest, K-fold out-of-fold predictions with the baseline config.
     In-sample residuals would shrink exactly where the probe is overfitting,
     so hardness is only ever read out-of-fold.
  3. Hard rows: teacher-external predicted high (false positives), and
     teacher-internal predicted low (false negatives), as rank percentiles of
     the OOF prediction. Weight them 1 + alpha.
  4. Pick (alpha, quantile, direction) on the dev set by teacher AP(internal)
     minus AUC(external), both teacher-labelled.
  5. Refit the chosen config and the baseline on the whole corpus and write both
     vectors; eval_reweight.py reads the Gemini gold sets once.

Baseline configs are the ones the union retrain selected (results_*_union.json):
Jina ridge l2=0.1 T=3.0, MiniLM ridge l2=10 T=0.5. Interaction rows are appended
exactly as train_internal_vector.py does, weight 1, never counted as hard.

Run from anywhere:
  ../../tmp_venv/bin/python reweight_hard_examples.py jina
  ../../tmp_venv/bin/python reweight_hard_examples.py minilm
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN = os.path.join(ROOT, "training")
OUT = os.path.join(ROOT, "research", "data", "hard_examples")

SPACE = sys.argv[1] if len(sys.argv) > 1 else "jina"
CFG = {
    "jina": dict(cache="embeddings_union_jina.npz", int_cache="embeddings_interactions_jina.npz",
                 l2=0.1, T=3.0, model="jinaai/jina-embeddings-v5-text-small", task="classification"),
    "minilm": dict(cache="embeddings_union_minilm.npz", int_cache="embeddings_interactions_minilm.npz",
                   l2=10.0, T=0.5, model="sentence-transformers/all-MiniLM-L12-v2", task="none"),
}[SPACE]

# The trainer reads its configuration at import time.
os.environ["LOC_TEACHER_FILE"] = os.path.join(TRAIN, "data", "dataset_annotated_union_jev_decider.jsonl")
os.environ["LOC_EMB_CACHE"] = os.path.join(TRAIN, "data", CFG["cache"])
os.environ["LOC_INT_EMB_CACHE"] = os.path.join(TRAIN, "data", CFG["int_cache"])
sys.path.insert(0, TRAIN)
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
import train_internal_vector as T  # noqa: E402

K = 5
DEV_FRAC = 0.2
ALPHAS = [1.0, 3.0, 9.0]
QUANTILES = [0.10, 0.25]           # how far into the wrong tail counts as hard
DIRECTIONS = ["fp", "both"]       # fp: only teacher-external scored high


def load():
    rows = T.load_train_rows()
    d = np.load(T.CACHE_LMSTUDIO_EMBS, allow_pickle=True)
    X = np.nan_to_num(d["X"]).astype(np.float32)
    # Positional cache: a row filter change would silently misalign everything.
    assert len(rows) == len(X), f"{len(rows)} rows vs {len(X)} cached embeddings"
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    int_rows = T.load_interaction_rows()
    Xi = np.nan_to_num(np.load(T.CACHE_INTERACTIONS_EMBS, allow_pickle=True)["X"]).astype(np.float32)
    assert len(int_rows) == len(Xi), "interaction cache misaligned"
    Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
    return rows, X, int_rows, Xi


def fit(X, y, w):
    coef, b = T.train_ridge(X, y, CFG["l2"], w)
    return coef, b


def hard_weights(pred, cls, alpha, q, direction):
    """1 + alpha on rows in the wrong tail of the OOF prediction's rank."""
    r = rankdata(pred) / len(pred)
    hard = (cls == "external") & (r >= 1 - q)
    if direction == "both":
        hard |= (cls == "internal") & (r <= q)
    return 1.0 + alpha * hard, int(hard.sum())


def dev_score(v, X, cls):
    s = X @ v
    ap = average_precision_score(cls == "internal", s)
    leak = roc_auc_score(cls == "external", s)
    return ap, leak


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, X, int_rows, Xi = load()
    y = np.array([T.get_target_score(r, CFG["T"]) for r in rows])
    yi = np.array([T.get_target_score(r, CFG["T"]) for r in int_rows])
    cls = np.array([T.get_class(r, 1.0) for r in rows])
    print(f"{SPACE}: {len(rows)} corpus rows ({dict(zip(*np.unique(cls, return_counts=True)))}), "
          f"{len(int_rows)} interaction rows")

    rng = np.random.default_rng(T.SEED)
    perm = rng.permutation(len(rows))
    n_dev = int(len(rows) * DEV_FRAC)
    dev, tr = perm[:n_dev], perm[n_dev:]

    def oof(idx):
        pred = np.empty(len(idx))
        folds = np.array_split(rng.permutation(len(idx)), K)
        for f in folds:
            m = np.ones(len(idx), bool); m[f] = False
            coef, b = fit(np.vstack([X[idx[m]], Xi]), np.concatenate([y[idx[m]], yi]), None)
            pred[f] = X[idx[f]] @ coef + b
        return pred

    def train_eval(idx, w_corpus):
        coef, _ = fit(np.vstack([X[idx], Xi]), np.concatenate([y[idx], yi]),
                      np.concatenate([w_corpus, np.ones(len(Xi))]))
        return coef / np.linalg.norm(coef)

    pred_tr = oof(tr)
    base_v = train_eval(tr, np.ones(len(tr)))
    base = dev_score(base_v, X[dev], cls[dev])
    print(f"baseline dev: AP(int) {base[0]:.4f}  AUC(ext) {base[1]:.4f}")
    res = []
    for d in DIRECTIONS:
        for q in QUANTILES:
            for a in ALPHAS:
                w, n = hard_weights(pred_tr, cls[tr], a, q, d)
                ap, leak = dev_score(train_eval(tr, w), X[dev], cls[dev])
                res.append(dict(direction=d, q=q, alpha=a, n_hard=n, dev_ap=ap, dev_leak=leak,
                                objective=ap - leak))
                print(f"  {d:4} q={q:.2f} a={a:<4g} hard={n:5}  AP(int) {ap:.4f} ({ap - base[0]:+.4f})"
                      f"  AUC(ext) {leak:.4f} ({leak - base[1]:+.4f})")
    best = max(res, key=lambda r: r["objective"])
    print(f"chosen on dev: {best}")

    # Refit on the whole corpus: OOF over everything for the weights, then both vectors.
    pred_all = oof(np.arange(len(rows)))
    w_all, n_all = hard_weights(pred_all, cls, best["alpha"], best["q"], best["direction"])
    for name, w in [("base", np.ones(len(rows))), ("hard", w_all)]:
        coef, b = fit(np.vstack([X, Xi]), np.concatenate([y, yi]), np.concatenate([w, np.ones(len(Xi))]))
        stem = os.path.join(OUT, f"vector_{SPACE}_{name}")
        np.save(stem + ".npy", coef / np.linalg.norm(coef))
        with open(stem + "_meta.json", "w") as f:
            json.dump({"scale": float(np.linalg.norm(coef)), "bias": float(b), "temperature": CFG["T"],
                       "l2_reg": CFG["l2"], "embedding_model": CFG["model"], "embedding_task": CFG["task"],
                       "hard_examples": None if name == "base" else {**best, "n_hard_full": n_all}}, f, indent=2)
    # The hard rows themselves, to read what the probe was failing on.
    hard_idx = np.where(w_all > 1)[0]
    with open(os.path.join(OUT, f"hard_rows_{SPACE}.jsonl"), "w", encoding="utf-8") as f:
        for i in hard_idx[np.argsort(-np.abs(pred_all[hard_idx] - y[hard_idx]))]:
            f.write(json.dumps({"uri": rows[i]["uri"], "teacher_class": cls[i], "target": float(y[i]),
                                "oof_pred": float(pred_all[i]), "text": rows[i]["text"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, f"sweep_{SPACE}.json"), "w") as f:
        json.dump({"baseline_dev": {"ap": base[0], "leak": base[1]}, "sweep": res, "chosen": best}, f, indent=2)
    print(f"wrote vectors and {len(hard_idx)} hard rows to {OUT}")


if __name__ == "__main__":
    main()
