"""How the locus-of-control vector reshapes the emotional mix of the feed.

Reads emotions_loc_corpus.npz (see score_emotions_loc.py) and compares the
emotion prevalence of the posts the deployed vector admits against the pool it
draws from. The vector's only editorial action in the live feed is the 0.88
probability gate (feeds.ts; ordering among survivors is handed to GE), so the
headline comparison holds the length gate fixed and varies only the vector:

    pool      = posts passing minWords 6      (feed without the vector)
    admitted  = pool AND LoC probability >= 0.88   (feed with the vector)

Reports prevalence, lift, a two-proportion z-test, decile dose-response, rank
correlation, and the teacher-label sanity check.

Run: tmp_venv/bin/python analyze_emotion_shift.py
"""

import json

import numpy as np
from scipy.stats import norm, spearmanr

THRESHOLD = 0.88
TAU = 0.5  # model card's multi-label decision threshold
NPZ = "emotions_loc_corpus.npz"


def two_prop_z(k1, n1, k2, n2):
    """z and two-sided p for prevalence(k1/n1) vs prevalence(k2/n2)."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


def main():
    d = np.load(NPZ, allow_pickle=True)
    E, labels = d["E"], [str(x) for x in d["labels"]]
    prob, cos = d["prob"], d["cos"]
    long_enough, loc_label = d["long_enough"], [str(x) for x in d["loc_label"]]
    lang = [str(x) for x in d["lang"]]
    n = len(prob)

    hot = E >= TAU
    pool = long_enough                      # feed without the vector
    admitted = long_enough & (prob >= THRESHOLD)   # feed with the vector
    rejected = long_enough & (prob < THRESHOLD)

    report = {
        "n_corpus": int(n),
        "n_pool_minwords": int(pool.sum()),
        "n_admitted": int(admitted.sum()),
        "keep_rate": float(admitted.sum() / pool.sum()),
        "threshold": THRESHOLD,
        "tau": TAU,
        "emotion_model": "tabularisai/multilingual-emotion-classification",
        "labels_per_post_pool": float(hot[pool].sum(1).mean()),
        "labels_per_post_admitted": float(hot[admitted].sum(1).mean()),
    }

    print(f"corpus {n}  pool(minWords>=6) {pool.sum()}  "
          f"admitted(p>={THRESHOLD}) {admitted.sum()}  "
          f"keep {admitted.sum() / pool.sum():.1%}")
    print(f"mean emotions fired/post: pool {hot[pool].sum(1).mean():.2f}  "
          f"admitted {hot[admitted].sum(1).mean():.2f}\n")

    print(f"{'emotion':<12} {'pool%':>7} {'admit%':>7} {'lift':>6} "
          f"{'rej%':>7} {'p':>9}  {'rho':>6}")
    print("-" * 62)
    rows = []
    for j, lab in enumerate(labels):
        k_pool, k_adm = int(hot[pool, j].sum()), int(hot[admitted, j].sum())
        k_rej = int(hot[rejected, j].sum())
        p_pool = k_pool / pool.sum()
        p_adm = k_adm / admitted.sum()
        p_rej = k_rej / rejected.sum()
        _, pval = two_prop_z(k_adm, int(admitted.sum()), k_rej, int(rejected.sum()))
        rho, _ = spearmanr(prob[pool], E[pool, j])
        lift = p_adm / p_pool if p_pool else float("nan")
        rows.append({"emotion": lab, "pool_pct": p_pool * 100,
                     "admitted_pct": p_adm * 100, "rejected_pct": p_rej * 100,
                     "lift": lift, "p_value": pval, "spearman_rho": float(rho)})
        star = "***" if pval < 1e-3 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"{lab:<12} {p_pool*100:7.2f} {p_adm*100:7.2f} {lift:6.2f} "
              f"{p_rej*100:7.2f} {pval:9.2e}{star:<3} {rho:6.3f}")
    report["emotions"] = rows

    # Dose-response: decile of LoC probability within the pool.
    print("\nprevalence by LoC-score decile (pool only)")
    order = np.argsort(prob[pool])
    idx_pool = np.flatnonzero(pool)[order]
    deciles = np.array_split(idx_pool, 10)
    head = f"{'decile':<8}" + "".join(f"{l[:6]:>8}" for l in labels)
    print(head)
    dec_rows = []
    for i, chunk in enumerate(deciles, 1):
        vals = [hot[chunk, j].mean() * 100 for j in range(len(labels))]
        dec_rows.append({"decile": i, "n": len(chunk),
                         "loc_prob_mid": float(np.median(prob[chunk])),
                         "pct": {l: v for l, v in zip(labels, vals)}})
        print(f"D{i:<7}" + "".join(f"{v:8.1f}" for v in vals))
    report["deciles"] = dec_rows

    # Sanity: does the emotion model track the teacher's own LoC labels?
    print("\nprevalence by teacher label (all rows)")
    lab_arr = np.array(loc_label)
    print(f"{'label':<10}{'n':>6}" + "".join(f"{l[:6]:>8}" for l in labels))
    tl_rows = []
    for name in ["internal", "neutral", "external"]:
        m = lab_arr == name
        vals = [hot[m, j].mean() * 100 for j in range(len(labels))]
        tl_rows.append({"loc_label": name, "n": int(m.sum()),
                        "pct": {l: v for l, v in zip(labels, vals)}})
        print(f"{name:<10}{m.sum():6d}" + "".join(f"{v:8.1f}" for v in vals))
    report["by_teacher_label"] = tl_rows

    # Language split — the corpus is firehose-mixed; the model is multilingual.
    print("\nkeep rate and top shifts by language")
    lang_arr = np.array(lang)
    lang_rows = []
    for code in ["en", "de", ""]:
        m = pool & (lang_arr == code)
        if m.sum() < 100:
            continue
        a = m & (prob >= THRESHOLD)
        name = code or "(none)"
        shifts = {l: float(hot[a, j].mean() * 100 - hot[m, j].mean() * 100)
                  for j, l in enumerate(labels)}
        lang_rows.append({"lang": name, "n_pool": int(m.sum()),
                          "keep_rate": float(a.sum() / m.sum()), "pp_shift": shifts})
        top = sorted(shifts.items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"{name:<8} n={m.sum():5d} keep={a.sum()/m.sum():5.1%}  " +
              "  ".join(f"{k} {v:+.1f}pp" for k, v in top))
    report["by_language"] = lang_rows

    with open("REPORT_emotion_vs_loc.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote REPORT_emotion_vs_loc.json")


if __name__ == "__main__":
    main()
