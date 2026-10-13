"""Open-weight System One models against TypeSafe's `jev-latest`, on the gold set.

The feed's teacher is a paid API call. Two open candidates claim to do the same
job locally, so this asks the only question that matters before swapping one in:
on the same 500 Gemini-labelled posts, with the same rubric, does either of them
rank internal posts the way jev does?

  jev          TypeSafe System One `jev-latest`, from scores_loc_typesafe_gold.jsonl.
               Already paid for; the incumbent.
  laya_ml      convaiinnovations/laya-multilingual, mmBERT-base, 322M, local.
  laya_en      convaiinnovations/laya, ModernBERT-large, 421M, local. English-only
               by its own model card, and 56% of this set is tagged English.
  qwen_rlcd    harshatheg/Qwen-2.5-1B-RLCD's constrained decoding over stock
               Qwen2.5-1.5B-Instruct-4bit, local.
  decider      Mapika/decider-2b, Qwen3.5-2B-Base fine-tuned on 942k typed
               decision examples, read off the option-letter logits, local.
               See score_loc_decider.py for why it is asked on two rubrics.
  laya_ft_ml   laya-multilingual after finetune_laya_loc.py has distilled the
               jev teacher's own distributions into it. Trained on the 5000
               corpus posts that are NOT in this gold set; see that file.
  jina         the deployed vector, as a floor: it is a 1024-dim dot product and
               it is what any of these would have to beat to be worth the latency.

Laya is measured on two rubrics. `full` is ts_questions.py sent verbatim -- the
literal drop-in swap -- and `compact` is the same rubric rewritten to fit Laya's
48-token-per-option head budget, which `full` overruns. See
loc_questions_compact.py; reporting only one of the two would either understate
the model or overstate what a swap actually buys.

Metrics, groupings and bootstrap come from eval_minilm_vs_jina_by_lang.py so the
numbers line up with REPORT_typesafe_vs_jina.md line for line. AP and P@25 lead,
per CLAUDE.md: the feed consumes the head of a ranking and never sees the tail.

Run: HF_HOME=/Volumes/Daten/huggingface ../../tmp_venv/bin/python eval_laya_vs_jev.py
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN_DATA = os.path.join(ROOT, "training", "data")
OUT_DATA = os.path.join(ROOT, "research", "data", "evals")
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, HERE)
import eval_minilm_vs_jina_by_lang as E  # noqa: E402  (load_jsonl, groups_for, boot_*)

VAL_FILE = os.path.join(TRAIN_DATA, "dataset_validation_gemini.jsonl")
FLAGS_VAL = os.path.join(OUT_DATA, "lang_flags_gemini.json")
OUT_JSON = os.path.join(OUT_DATA, "REPORT_laya_vs_jev.json")
SEED = 20260919

# name -> score file. Every one of these writes the same four fields, which is
# what makes them comparable at all; see score_loc_laya.py.
SYSTEMS = [
    ("jev", os.path.join(TRAIN_DATA, "scores_loc_typesafe_gold.jsonl")),
    ("laya_ml_full", os.path.join(OUT_DATA, "scores_loc_laya_ml_full_gold.jsonl")),
    ("laya_ml_compact", os.path.join(OUT_DATA, "scores_loc_laya_ml_compact_gold.jsonl")),
    ("laya_en_full", os.path.join(OUT_DATA, "scores_loc_laya_en_full_gold.jsonl")),
    ("laya_en_compact", os.path.join(OUT_DATA, "scores_loc_laya_en_compact_gold.jsonl")),
    ("qwen_1.5b", os.path.join(OUT_DATA, "scores_loc_qwen_rlcd_gold.jsonl")),
    ("qwen_7b", os.path.join(OUT_DATA, "scores_loc_qwen_rlcd_7b_gold.jsonl")),
    ("qwen3_8b", os.path.join(OUT_DATA, "scores_loc_qwen_rlcd_qwen3_8b_gold.jsonl")),
    ("decider_short", os.path.join(OUT_DATA, "scores_loc_decider_short_gold.jsonl")),
    ("decider_compact", os.path.join(OUT_DATA, "scores_loc_decider_compact_gold.jsonl")),
    ("laya_ft_ml", os.path.join(OUT_DATA, "scores_loc_laya_ft_ml_compact_gold.jsonl")),
    ("decider_ft", os.path.join(OUT_DATA, "scores_loc_decider_ft_decider_short_gold.jsonl")),
]
PRIMITIVES = {
    "score": lambda s: s["ts_score"],
    "choice": lambda s: s["ts_class_probs"].get("internal", 0.0),
    "noul": lambda s: s["ts_noul"],
}
JINA = {
    "vector": os.path.join(TRAIN_DATA, "internal_vector.npy"),
    "meta": os.path.join(TRAIN_DATA, "internal_vector_meta.json"),
    "val_emb": os.path.join(TRAIN_DATA, "embeddings_val_loc_xy.npz"),
}


def jina_scores(rows):
    """The deployed probe's sigmoid, byte-for-byte as bunny_feed computes it.

    The cache is positional against train_internal_vector.load_val_rows(), which
    is the same filter this file applies, so the assert is the alignment check
    CLAUDE.md asks for rather than a formality.
    """
    d = np.load(JINA["val_emb"], allow_pickle=True)
    X = d["X"]
    assert len(X) == len(rows), f"val cache is {len(X)} rows, gold set is {len(rows)}"
    v = np.load(JINA["vector"]).astype(np.float64)
    meta = json.load(open(JINA["meta"]))
    cos = E.unit(X) @ (v / np.linalg.norm(v))
    return 1.0 / (1.0 + np.exp(-(meta["scale"] * cos + meta["bias"])))


def load_scorers(rows):
    """{label: score array} plus {system: rows} for the 3-class and cost tables."""
    uris = [r["uri"] for r in rows]
    scorers, raw = {}, {}
    for name, path in SYSTEMS:
        if not os.path.exists(path):
            print(f"  (skipping {name}: {os.path.basename(path)} not present)")
            continue
        by_uri = {r["uri"]: r for r in E.load_jsonl(path)}
        missing = [u for u in uris if u not in by_uri or "error" in by_uri[u]]
        if missing:
            print(f"  (skipping {name}: {len(missing)} posts unscored)")
            continue
        S = [by_uri[u] for u in uris]
        raw[name] = S
        for p, get in PRIMITIVES.items():
            scorers[f"{name}.{p}"] = np.array([get(s) for s in S], dtype=float)
    scorers["jina.vector"] = jina_scores(rows)
    return scorers, raw


def precision_at_k(y, s, k=25):
    return float(y[np.argsort(-s)[:k]].mean())


def boot_paired_ap_diff(y, s_a, s_b, rng, n=E.N_BOOT):
    """AP(a) - AP(b) on the same posts. AUC and AP have disagreed in sign on a
    real retrain of this axis (CLAUDE.md), and they disagree again across these
    backbones, so the lead metric gets its own paired test rather than being
    read off a point estimate."""
    obs = average_precision_score(y, s_a) - average_precision_score(y, s_b)
    idx, vals = np.arange(len(y)), []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            vals.append(average_precision_score(y[b], s_a[b]) - average_precision_score(y[b], s_b[b]))
    vals = np.array(vals)
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(obs), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), float(p)


def main():
    rng = np.random.default_rng(SEED)
    rows = [r for r in E.load_jsonl(VAL_FILE)
            if r.get("text") and r.get("loc_label") in ("internal", "external", "neutral")]
    gold = np.array([r["loc_label"] for r in rows])
    langs = np.array([E.lang_of(r) for r in rows])
    detected = E.detected_langs(rows)
    flagged = E.detector_flags(rows, FLAGS_VAL)

    scorers, raw = load_scorers(rows)
    report = {"n": len(rows), "seed": SEED, "systems": list(raw), "groups": []}
    print(f"\n{len(rows)} posts, {(gold == 'internal').sum()} internal, "
          f"{(gold == 'external').sum()} external, {(gold == 'neutral').sum()} neutral")

    # -------- headline: can anything here rank internal posts at the head? ----
    y_all = (gold == "internal").astype(int)
    ref = "jev.choice"
    print(f"\n=== internal-vs-rest, all {len(rows)} posts (paired against {ref}) ===")
    print(f"{'scorer':24}{'AUC':>7}{'95% CI':>18}{'AP':>8}{'P@25':>8}{'  dAP vs ' + ref:>30}")
    print("-" * 95)
    for k in sorted(scorers):
        s = scorers[k]
        lo, hi = E.boot_auc(y_all, s, rng)
        row = {"scorer": k, "auc": float(roc_auc_score(y_all, s)), "auc_ci": [lo, hi],
               "ap": float(average_precision_score(y_all, s)),
               "p_at_25": precision_at_k(y_all, s)}
        if k != ref and ref in scorers:
            obs, (dlo, dhi), p = boot_paired_ap_diff(y_all, s, scorers[ref], rng)
            row["vs_ref_ap"] = {"obs": obs, "ci": [dlo, dhi], "p": p}
            row["vs_ref"] = dict(zip(("obs", "ci", "p"),
                                     E.boot_paired_auc_diff(y_all, s, scorers[ref], rng)))
            tail = f"{obs:+.3f} [{dlo:+.3f},{dhi:+.3f}] p={p:.3f}"
        else:
            tail = ""
        print(f"{k:24}{row['auc']:7.3f}  [{lo:.3f},{hi:.3f}]{row['ap']:8.3f}{row['p_at_25']:8.3f}  {tail:>28}")
        report.setdefault("headline", []).append(row)

    # -------- the same, by language and by what the feed's detector routes ----
    for target, name in ((("internal",), "internal-vs-rest"), (("internal", "external"), "internal-vs-external")):
        print(f"\n=== {name}: AUC by group ===")
        keys = sorted(scorers)
        print(f"{'':12}{'n':>5}{'pos':>5}  " + "".join(f"{k.replace('_', '')[:11]:>12}" for k in keys))
        print("-" * (22 + 12 * len(keys)))
        for gname, m in E.groups_for(langs, detected, flagged):
            mm = m.copy()
            if len(target) == 2:
                mm &= np.isin(gold, target)
            y = (gold[mm] == "internal").astype(int)
            if y.sum() < 3 or (1 - y).sum() < 3:
                continue
            aucs = {k: float(roc_auc_score(y, scorers[k][mm])) for k in keys}
            aps = {k: float(average_precision_score(y, scorers[k][mm])) for k in keys}
            print(f"{gname:12}{mm.sum():5d}{y.sum():5d}  " + "".join(f"{aucs[k]:12.3f}" for k in keys))
            report["groups"].append({"target": name, "group": gname, "n": int(mm.sum()),
                                     "n_pos": int(y.sum()), "auc": aucs, "ap": aps})

    # -------- the 3-way label, which the vector cannot produce ---------------
    print("\n=== 3-class agreement with the Gemini label ===")
    print(f"{'system':18}{'acc':>7}{'macroF1':>9}{'F1 int':>9}{'F1 ext':>9}{'F1 neu':>9}   predicted mix")
    for name, S in raw.items():
        pred = np.array([s["ts_class"] for s in S])
        f1s = []
        for c in ("internal", "external", "neutral"):
            tp = ((pred == c) & (gold == c)).sum()
            p_ = tp / max((pred == c).sum(), 1)
            r_ = tp / max((gold == c).sum(), 1)
            f1s.append(2 * p_ * r_ / max(p_ + r_, 1e-9))
        mix = {c: int((pred == c).sum()) for c in ("internal", "external", "neutral")}
        print(f"{name:18}{(pred == gold).mean():7.3f}{np.mean(f1s):9.3f}"
              f"{f1s[0]:9.3f}{f1s[1]:9.3f}{f1s[2]:9.3f}   {mix}")
        report.setdefault("three_class", {})[name] = {
            "acc": float((pred == gold).mean()), "macro_f1": float(np.mean(f1s)),
            "f1_internal": float(f1s[0]), "f1_external": float(f1s[1]),
            "f1_neutral": float(f1s[2]), "predicted_mix": mix,
            "gold_mix": {c: int((gold == c).sum()) for c in ("internal", "external", "neutral")}}

    # -------- what each one costs to run -------------------------------------
    print("\n=== cost / latency, as measured on this run ===")
    for name, S in raw.items():
        lat = sorted(s["latency_s"] for s in S if s.get("latency_s") is not None)
        tok = sum(s.get("input_tokens") or 0 for s in S)
        usd = tok / len(S) * 1000 / 1e6 * 0.042 if name == "jev" else 0.0
        print(f"{name:18} p50={lat[len(lat) // 2]:6.3f}s  p95={lat[int(len(lat) * .95)]:6.3f}s  "
              f"usd/1k posts={usd:.4f}" + ("  (local weights, no per-call cost)" if usd == 0 else ""))
        report.setdefault("cost", {})[name] = {
            "latency_p50": lat[len(lat) // 2], "latency_p95": lat[int(len(lat) * .95)],
            "input_tokens": int(tok), "usd_per_1k_posts": usd}

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
