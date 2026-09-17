"""Can MiniLM's German deficit be trained away — by upweighting German, or by
training for recall rather than rank?

REPORT_minilm_vs_jina_by_lang.md showed the MiniLM probe ranks German near chance
while Jina holds 0.674. Two repairs are worth trying before concluding the space
cannot carry the axis in German, and neither had been run:

  weight   upweight the German rows in the Ridge fit (w = 1 .. 50), and the limit
           case of training on German rows alone. REPORT_ge_embeddings.md §4 tried
           bigger heads and a recall-shaped loss, but never touched the language mix.
  recall   logistic on the binarised band, and low regularisation — the two knobs
           §4 found "improve recall" on the aggregate. Whether that recall is
           German recall was never checked.

Folds, rows, target and evaluation are identical to eval_minilm_vs_jina_by_lang.py,
so the numbers drop straight into that report's tables. The Jina column is the bar
to clear: German OOF AUC 0.674, which is what the routing currently buys.

Run: LOC_TEACHER_FILE=dataset_annotated_bonsai_v2.jsonl tmp_venv/bin/python eval_german_weighting.py
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

os.environ.setdefault("LOC_TEACHER_FILE", "dataset_annotated_bonsai_v2.jsonl")
import train_internal_vector as T  # noqa: E402

TEMPERATURE, FOLDS, SEED = 5.0, 5, 20260911
SPACES = {
    "minilm": ("embeddings_bonsai_v2_minilm.npz", "embeddings_interactions_minilm.npz"),
    "jina": ("embeddings_bonsai_v2_jina.npz", "embeddings_interactions_jina.npz"),
}


def unit(X):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def oof_scores(X, target, fold, fit, weights=None, subset=None):
    """Out-of-fold scores. `subset` restricts TRAINING rows (scoring stays global)."""
    s = np.full(len(X), np.nan)
    for k in range(FOLDS):
        tr = fold != k
        if subset is not None:
            tr = tr & subset
        te = fold == k
        w = None if weights is None else weights[tr]
        v = fit(X[tr], target[tr], w)
        s[te] = X[te] @ (v / np.linalg.norm(v))
    return s


def ridge(alpha):
    def f(X, y, w):
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=w)
        return m.coef_
    return f


def logistic(C):
    def f(X, y, w):
        m = LogisticRegression(max_iter=3000, C=C).fit(X, (y >= T.POSITIVE_BAND[0]).astype(int),
                                                       sample_weight=w)
        return m.coef_[0]
    return f


def main():
    rows = T.load_train_rows() + T.load_interaction_rows()
    langs = np.array([(r.get("langs") or ["(none)"])[0].split("-")[0] for r in rows])
    target = np.array([T.get_target_score(r, TEMPERATURE) for r in rows])
    cls = np.array([T.get_class(r, TEMPERATURE) for r in rows])
    y = (target >= T.POSITIVE_BAND[0]).astype(int)
    de, en = langs == "de", langs == "en"
    fold = np.random.default_rng(SEED).permutation(np.arange(len(rows)) % FOLDS)
    print(f"{len(rows)} rows, {y.sum()} band-positives; de {de.sum()} ({y[de].sum()} pos), "
          f"en {en.sum()} ({y[en].sum()} pos)")

    X = {name: unit(np.vstack([np.load(a)["X"], np.load(b)["X"]]))
         for name, (a, b) in SPACES.items()}
    report = {"rows": len(rows), "folds": FOLDS, "temperature": TEMPERATURE, "runs": []}

    def evaluate(space, tag, s):
        ok = ~np.isnan(s)
        d, e = de & ok, en & ok
        ext = cls == "external"
        leak = (roc_auc_score(ext[d].astype(int), s[d])
                if ext[d].sum() >= 3 and (~ext[d]).sum() >= 3 else float("nan"))
        row = {"space": space, "run": tag,
               "de_auc": roc_auc_score(y[d], s[d]), "en_auc": roc_auc_score(y[e], s[e]),
               "de_n": int(d.sum()), "de_ext_leak": float(leak),
               # What the prefilter actually does: of the German positives, how many
               # land in the global top 20% of the pool by this score?
               "de_recall_at_20": float(
                   (s[d & (y == 1)] >= np.quantile(s[ok], 0.8)).mean())}
        print(f"  {space:<7}{tag:<28}de {row['de_auc']:.3f}  en {row['en_auc']:.3f}  "
              f"de-leak {leak:.3f}  de-recall@20% {row['de_recall_at_20']:.1%}")
        report["runs"].append(row)
        return row

    print("\n1. German sample weight in the Ridge fit (λ = 10, the shipped head)")
    for space in ("minilm", "jina"):
        for w in (1, 2, 3, 5, 10, 20, 50):
            weights = np.where(de, float(w), 1.0)
            evaluate(space, f"ridge w(de)={w}",
                     oof_scores(X[space], target, fold, ridge(10.0), weights))
        # The limit case: German rows are the only rows.
        evaluate(space, "ridge, German rows only",
                 oof_scores(X[space], target, fold, ridge(10.0), subset=de))
        evaluate(space, "ridge, English rows only",
                 oof_scores(X[space], target, fold, ridge(10.0), subset=en))

    print("\n2. Trained for recall rather than rank (λ / C swept), German-upweighted")
    for space in ("minilm", "jina"):
        for alpha in (0.01, 0.1, 1.0):
            evaluate(space, f"ridge λ={alpha} w(de)=10",
                     oof_scores(X[space], target, fold, ridge(alpha), np.where(de, 10.0, 1.0)))
        for C in (0.1, 1.0):
            evaluate(space, f"logistic C={C} w(de)=10",
                     oof_scores(X[space], target, fold, logistic(C), np.where(de, 10.0, 1.0)))

    with open("REPORT_german_weighting.json", "w") as f:
        json.dump(report, f, indent=2)
    best = max((r for r in report["runs"] if r["space"] == "minilm"), key=lambda r: r["de_auc"])
    jina_base = [r for r in report["runs"] if r["space"] == "jina" and r["run"] == "ridge w(de)=1"][0]
    print(f"\nbest MiniLM German AUC: {best['de_auc']:.3f} ({best['run']}), "
          f"English {best['en_auc']:.3f}")
    print(f"Jina, unweighted:       {jina_base['de_auc']:.3f} German / {jina_base['en_auc']:.3f} English")
    print("Wrote REPORT_german_weighting.json")


if __name__ == "__main__":
    main()
