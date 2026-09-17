"""TypeSafe's System One (jev) against the deployed Jina vector, on the gold set.

Same 500 Gemini-labelled posts, same AUC-of-P(internal) target, same language
groupings and same bootstrap as eval_minilm_vs_jina_by_lang.py -- whose helpers
are imported rather than re-implemented, so the two reports are comparable line
for line. What changes is only where the score comes from: a 128-dim dot product
against a trained vector, or a remote rubric evaluation with no training at all.

Four scorers are compared. Three are TypeSafe's, from one request each
(score / choice / noul, see ts_questions.py); the fourth is the shipped Jina
probe. `fusion` is the rank-average of the best TypeSafe scorer and Jina, which
is how REPORT_ge_embeddings.md combined two spaces.

Run: tmp_venv/bin/python eval_typesafe_vs_jina.py
"""

import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import eval_minilm_vs_jina_by_lang as E

TS_FILE = "scores_loc_typesafe_gold.jsonl"
OUT_JSON = "REPORT_typesafe_vs_jina.json"
SEED = 20260911


def main():
    rng = np.random.default_rng(SEED)
    val = E.load_jsonl(E.VAL_FILE)
    rows = [r for r in val if r.get("text") and r.get("loc_label") in ("internal", "external", "neutral")]

    ts = {json.loads(l)["uri"]: json.loads(l) for l in open(TS_FILE, encoding="utf-8")}
    bad = [r["uri"] for r in rows if r["uri"] not in ts or "error" in ts[r["uri"]]]
    assert not bad, f"{len(bad)} posts unscored by TypeSafe"

    gold = np.array([r["loc_label"] for r in rows])
    langs = np.array([E.lang_of(r) for r in rows])
    detected = E.detected_langs(rows)
    flagged = E.detector_flags(rows, E.FLAGS_VAL)

    # --- the deployed Jina probe, byte-for-byte as part_a computes it ---
    d = np.load(E.JINA["val_emb"], allow_pickle=True)
    X = d["X"]
    assert len(X) == len(rows)
    v = np.load(E.JINA["vector"]).astype(np.float64)
    meta = json.load(open(E.JINA["meta"]))
    cos = E.unit(X) @ (v / np.linalg.norm(v))
    jina = 1.0 / (1.0 + np.exp(-(meta["scale"] * cos + meta["bias"])))

    # --- TypeSafe's three ---
    S = [ts[r["uri"]] for r in rows]
    scorers = {
        "ts_score": np.array([s["ts_score"] for s in S]),
        "ts_choice": np.array([s["ts_class_probs"].get("internal", 0.0) for s in S]),
        "ts_noul": np.array([s["ts_noul"] for s in S]),
        "jina": jina,
    }
    scorers["fusion"] = (E.rank_norm(scorers["ts_score"]) + E.rank_norm(jina)) / 2

    report = {"n": len(rows), "seed": SEED, "groups": []}

    # ---------------- headline: internal vs rest, and internal vs external ----
    for target, name in ((("internal",), "internal-vs-rest"), (("internal", "external"), "internal-vs-external")):
        print(f"\n=== {name} ===")
        print(f"{'':12}{'n':>5}{'pos':>5}  " + "".join(f"{k:>12}" for k in scorers))
        print("-" * (22 + 12 * len(scorers)))
        for gname, m in E.groups_for(langs, detected, flagged):
            mm = m.copy()
            if len(target) == 2:
                mm &= np.isin(gold, target)
            y = (gold[mm] == "internal").astype(int)
            if y.sum() < 3 or (1 - y).sum() < 3:
                continue
            aucs = {k: roc_auc_score(y, s[mm]) for k, s in scorers.items()}
            print(f"{gname:12}{mm.sum():5d}{y.sum():5d}  " + "".join(f"{aucs[k]:12.3f}" for k in scorers))
            row = {"target": name, "group": gname, "n": int(mm.sum()), "n_pos": int(y.sum()),
                   "auc": {k: float(a) for k, a in aucs.items()},
                   "ap": {k: float(average_precision_score(y, s[mm])) for k, s in scorers.items()}}
            if name == "internal-vs-rest":
                for k in ("ts_score", "jina"):
                    lo, hi = E.boot_auc(y, scorers[k][mm], rng)
                    row.setdefault("auc_ci", {})[k] = [lo, hi]
                obs, (lo, hi), p = E.boot_paired_auc_diff(y, scorers["ts_score"][mm], jina[mm], rng)
                row["ts_minus_jina"] = {"obs": obs, "ci": [lo, hi], "p": p}
            report["groups"].append(row)

    # ---------------- CIs and the paired test, printed ----------------------
    print("\n=== internal-vs-rest: TypeSafe score vs Jina, with uncertainty ===")
    print(f"{'':12}{'n':>5}{'int':>5}  {'ts AUC':>22}  {'jina AUC':>22}  {'ts-jina':>24}")
    print("-" * 96)
    for row in report["groups"]:
        if row["target"] != "internal-vs-rest" or "ts_minus_jina" not in row:
            continue
        t, j, dd = row["auc_ci"]["ts_score"], row["auc_ci"]["jina"], row["ts_minus_jina"]
        print(f"{row['group']:12}{row['n']:5d}{row['n_pos']:5d}  "
              f"{row['auc']['ts_score']:.3f} [{t[0]:.3f},{t[1]:.3f}]  "
              f"{row['auc']['jina']:.3f} [{j[0]:.3f},{j[1]:.3f}]  "
              f"{dd['obs']:+.3f} [{dd['ci'][0]:+.3f},{dd['ci'][1]:+.3f}] p={dd['p']:.3f}")

    # ---------------- does the gap differ by language? ----------------------
    for de_key, en_key in (("de", "en"), ("de+det", "en+det")):
        gmap = dict(E.groups_for(langs, detected, flagged))
        md, me = gmap[de_key], gmap[en_key]
        yd, ye = (gold[md] == "internal").astype(int), (gold[me] == "internal").astype(int)
        obs, (lo, hi), p = E.boot_interaction(yd, scorers["ts_score"][md], jina[md],
                                              ye, scorers["ts_score"][me], jina[me], rng)
        print(f"\ninteraction ({de_key} vs {en_key}): (ts-jina|de) - (ts-jina|en) = "
              f"{obs:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}   "
              f"n_de={md.sum()}({yd.sum()} int) n_en={me.sum()}({ye.sum()} int)")
        report.setdefault("interaction", {})[f"{de_key}_vs_{en_key}"] = {
            "obs": obs, "ci": [lo, hi], "p": p,
            "n_de": int(md.sum()), "n_de_int": int(yd.sum()),
            "n_en": int(me.sum()), "n_en_int": int(ye.sum())}

    # ---------------- the 3-way label, which the vector cannot produce ------
    print("\n=== 3-class agreement with the Gemini label (TypeSafe choice) ===")
    pred = np.array([s["ts_class"] for s in S])
    for gname, m in E.groups_for(langs, detected, flagged):
        if m.sum() < 20:
            continue
        acc = (pred[m] == gold[m]).mean()
        f1s = []
        for c in ("internal", "external", "neutral"):
            tp = ((pred[m] == c) & (gold[m] == c)).sum()
            p_ = tp / max((pred[m] == c).sum(), 1)
            r_ = tp / max((gold[m] == c).sum(), 1)
            f1s.append(2 * p_ * r_ / max(p_ + r_, 1e-9))
        print(f"{gname:12}{m.sum():5d}  acc={acc:.3f}  macroF1={np.mean(f1s):.3f}  "
              f"per-class F1 int/ext/neu = {f1s[0]:.3f}/{f1s[1]:.3f}/{f1s[2]:.3f}")
        report.setdefault("three_class", {})[gname] = {
            "n": int(m.sum()), "acc": float(acc), "macro_f1": float(np.mean(f1s)),
            "f1_internal": float(f1s[0]), "f1_external": float(f1s[1]), "f1_neutral": float(f1s[2])}

    # ---------------- cost / latency, as measured ---------------------------
    tok = sum(s["input_tokens"] for s in S)
    lat = sorted(s["latency_s"] for s in S)
    report["cost"] = {
        "input_tokens": int(tok), "usd_per_1k_posts": tok / len(S) * 1000 / 1e6 * 0.042,
        "latency_p50": lat[len(lat) // 2], "latency_p95": lat[int(len(lat) * 0.95)]}

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
