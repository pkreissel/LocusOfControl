"""Is the locus-of-control vector doing anything a valence filter wouldn't?

Follow-up to analyze_emotion_shift.py. Three questions:

1. How much of the LoC score is explained by the emotion mix alone (ridge on the
   11 emotion probabilities, and a crude positive-minus-negative valence score)?
2. Does the vector strip anger even from posts the teacher itself called
   internal — i.e. does it disagree with its own training label on emotion?
3. Which emotions survive the gate *within* each teacher label?

Run: tmp_venv/bin/python analyze_emotion_valence.py
"""

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

THRESHOLD = 0.88
TAU = 0.5
POS = ["gratitude", "joy", "love"]
NEG = ["anger", "contempt", "disgust", "fear", "frustration", "sadness"]


def main():
    d = np.load("emotions_loc_corpus.npz", allow_pickle=True)
    E, labels = d["E"], [str(x) for x in d["labels"]]
    prob, long_enough = d["prob"], d["long_enough"]
    loc_label = np.array([str(x) for x in d["loc_label"]])
    idx = {l: i for i, l in enumerate(labels)}
    hot = E >= TAU
    pool = long_enough
    report = {}

    # 1. How much of the LoC axis is emotion?
    valence = E[:, [idx[p] for p in POS]].sum(1) - E[:, [idx[n] for n in NEG]].sum(1)
    rho_val, _ = spearmanr(prob[pool], valence[pool])
    logit = np.log(prob / (1 - prob))
    pred = cross_val_predict(Ridge(alpha=1.0), E[pool], logit[pool], cv=5)
    ss_res = ((logit[pool] - pred) ** 2).sum()
    ss_tot = ((logit[pool] - logit[pool].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    rho_ridge, _ = spearmanr(logit[pool], pred)
    print(f"LoC logit vs valence (pos-neg):      spearman rho = {rho_val:.3f}")
    print(f"LoC logit from 11 emotion probs:     5-fold R^2 = {r2:.3f}  "
          f"(spearman {rho_ridge:.3f})")
    report["valence_spearman"] = float(rho_val)
    report["ridge_cv_r2"] = float(r2)
    report["ridge_cv_spearman"] = float(rho_ridge)

    # 2 & 3. Gate behaviour inside each teacher label.
    print(f"\nwithin each teacher label: keep rate, and anger/fear before -> after the gate")
    print(f"{'label':<10}{'n_pool':>7}{'keep':>7}   " +
          "  ".join(f"{l[:5]:>13}" for l in ["anger", "fear", "joy", "love", "gratitude"]))
    per_label = []
    for name in ["internal", "neutral", "external"]:
        m = pool & (loc_label == name)
        a = m & (prob >= THRESHOLD)
        cells, row = [], {"loc_label": name, "n_pool": int(m.sum()),
                          "keep_rate": float(a.sum() / m.sum()), "emotions": {}}
        for l in ["anger", "fear", "joy", "love", "gratitude"]:
            before, after = hot[m, idx[l]].mean() * 100, hot[a, idx[l]].mean() * 100
            cells.append(f"{before:5.1f}->{after:5.1f}")
            row["emotions"][l] = {"pool_pct": float(before), "admitted_pct": float(after)}
        per_label.append(row)
        print(f"{name:<10}{m.sum():7d}{a.sum()/m.sum():7.1%}   " + "  ".join(f"{c:>13}" for c in cells))
    report["within_teacher_label"] = per_label

    # Mean LoC probability of angry vs non-angry internal posts.
    m = pool & (loc_label == "internal")
    angry = hot[:, idx["anger"]]
    print(f"\nteacher-internal posts: mean LoC prob "
          f"angry {prob[m & angry].mean():.3f} (n={int((m & angry).sum())})  "
          f"not angry {prob[m & ~angry].mean():.3f} (n={int((m & ~angry).sum())})")
    report["internal_angry_mean_prob"] = float(prob[m & angry].mean())
    report["internal_not_angry_mean_prob"] = float(prob[m & ~angry].mean())

    with open("REPORT_emotion_vs_loc_valence.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote REPORT_emotion_vs_loc_valence.json")


if __name__ == "__main__":
    main()
