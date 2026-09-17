"""The cross-check the gold set cannot give: is TypeSafe good, or just Gemini-shaped?

REPORT_typesafe_vs_jina.json measures both systems against 500 Gemini labels.
TypeSafe is an LLM handed roughly the rubric Gemini was handed, so some of its
margin there is two language models agreeing with each other rather than either
one being right.

This is the adversarial version. The reference is the LOCAL teacher's label -
the llama.cpp/bonsai annotations the deployed Jina vector was distilled from -
and Jina is scored out-of-fold on the very corpus it was fit to. Jina is
predicting its own teacher; TypeSafe has never seen it. If the ordering holds
here it is not an artefact of who wrote the labels.

Run: LOC_TEACHER_FILE=dataset_annotated_bonsai_v2.jsonl \
     tmp_venv/bin/python eval_typesafe_teacher_oof.py
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ.setdefault("LOC_TEACHER_FILE", "dataset_annotated_bonsai_v2.jsonl")
import eval_minilm_vs_jina_by_lang as E  # noqa: E402
import train_internal_vector as T  # noqa: E402

TS_FILE = "scores_loc_typesafe_bonsai_v2.jsonl"
OUT_JSON = "REPORT_typesafe_teacher_oof.json"


def main():
    rng = np.random.default_rng(E.SEED)
    rows = T.load_train_rows()
    X = np.load(E.JINA["train_emb"], allow_pickle=True)["X"]
    assert len(X) == len(rows), f"{len(X)} embeddings vs {len(rows)} rows"
    X = E.unit(X)

    ts = {}
    for line in open(TS_FILE, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            ts[r["uri"]] = r
    keep = np.array([r["uri"] in ts for r in rows])
    print(f"{keep.sum()}/{len(rows)} teacher rows scored by TypeSafe")
    rows = [r for r, k in zip(rows, keep) if k]
    X = X[keep]

    langs = np.array([E.lang_of(r) for r in rows])
    detected = E.detected_langs(rows)
    flagged = E.detector_flags(rows, E.FLAGS_TRAIN)
    target = np.array([T.get_target_score(r, E.TEMPERATURE) for r in rows])
    cls = np.array([T.get_class(r, E.TEMPERATURE) for r in rows])
    y = (target >= T.POSITIVE_BAND[0]).astype(int)

    # Jina, out-of-fold, fit to this corpus's own teacher target.
    fold = np.random.default_rng(E.SEED).permutation(np.arange(len(rows)) % E.FOLDS)
    jina = np.zeros(len(rows))
    for k in range(E.FOLDS):
        tr, te = fold != k, fold == k
        m = Ridge(alpha=E.L2, fit_intercept=True).fit(X[tr], target[tr])
        jina[te] = X[te] @ (m.coef_ / np.linalg.norm(m.coef_))

    S = [ts[r["uri"]] for r in rows]
    scorers = {
        "ts_score": np.array([s["ts_score"] for s in S]),
        "ts_choice": np.array([s["ts_class_probs"].get("internal", 0.0) for s in S]),
        "ts_noul": np.array([s["ts_noul"] for s in S]),
        "jina_oof": jina,
    }
    scorers["fusion"] = (E.rank_norm(scorers["ts_choice"]) + E.rank_norm(jina)) / 2

    print(f"\n=== band-positive vs rest ({y.sum()} positives of {len(y)}); "
          f"Jina is out-of-fold on its OWN teacher ===")
    print(f"{'':12}{'n':>5}{'pos':>5}  " + "".join(f"{k:>11}" for k in scorers)
          + f"  {'ts_choice-jina':>26}")
    print("-" * (22 + 11 * len(scorers) + 28))
    report = {"n": len(rows), "groups": []}
    for name, m in E.groups_for(langs, detected, flagged):
        yy = y[m]
        if yy.sum() < 5 or (1 - yy).sum() < 5:
            continue
        aucs = {k: roc_auc_score(yy, s[m]) for k, s in scorers.items()}
        d, (lo, hi), p = E.boot_paired_auc_diff(yy, scorers["ts_choice"][m], jina[m], rng)
        print(f"{name:12}{m.sum():5d}{yy.sum():5d}  "
              + "".join(f"{aucs[k]:11.3f}" for k in scorers)
              + f"  {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
        report["groups"].append({
            "group": name, "n": int(m.sum()), "n_pos": int(yy.sum()),
            "auc": {k: float(a) for k, a in aucs.items()},
            "ap": {k: float(average_precision_score(yy, s[m])) for k, s in scorers.items()},
            "ts_choice_minus_jina": {"obs": d, "ci": [lo, hi], "p": p},
            "spearman_vs_graded_target": {
                k: float(spearmanr(s[m], target[m]).statistic) for k, s in scorers.items()},
        })

    # Does the TypeSafe-minus-Jina gap depend on the language?
    gmap = dict(E.groups_for(langs, detected, flagged))
    for de_key, en_key in (("de", "en"), ("de+det", "en+det")):
        md, me = gmap[de_key], gmap[en_key]
        obs, (lo, hi), p = E.boot_interaction(
            y[md], scorers["ts_choice"][md], jina[md],
            y[me], scorers["ts_choice"][me], jina[me], rng)
        print(f"\ninteraction ({de_key} vs {en_key}): (ts-jina|de) - (ts-jina|en) = "
              f"{obs:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}   "
              f"n_de={md.sum()}({y[md].sum()} pos) n_en={me.sum()}({y[me].sum()} pos)")
        report.setdefault("interaction", {})[f"{de_key}_vs_{en_key}"] = {
            "obs": obs, "ci": [lo, hi], "p": p,
            "n_de": int(md.sum()), "n_de_pos": int(y[md].sum()),
            "n_en": int(me.sum()), "n_en_pos": int(y[me].sum())}

    # Agreement with the teacher's own 3-way label, by language.
    print("\n=== 3-class agreement with the LOCAL teacher label ===")
    pred = np.array([s["ts_class"] for s in S])
    for name, m in E.groups_for(langs, detected, flagged):
        if m.sum() < 50:
            continue
        print(f"{name:12}{m.sum():5d}  acc={(pred[m] == cls[m]).mean():.3f}")
        report.setdefault("three_class_vs_teacher", {})[name] = {
            "n": int(m.sum()), "acc": float((pred[m] == cls[m]).mean())}

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
