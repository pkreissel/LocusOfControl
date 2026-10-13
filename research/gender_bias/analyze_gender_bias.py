"""Does the LoC feed treat authors differently by (inferred) gender?

Scores the net20k corpus with the deployed vector (bunny_feed/src/vector.json)
from the aligned union Jina cache, applies the production gate, and compares
groups on:

  teacher_int  share the decider teacher labels internal   -- what the feed is taught
  pass         share clearing the gate                     -- what the feed does
  pass|int     gate recall on teacher-internal posts       -- equal opportunity
  pass|~int    gate pass on teacher-external/neutral posts -- false-positive rate
  top5         share inside the corpus's top 5% by score

pass minus teacher_int separates bias the vector adds from bias it inherits.
Authors post many times, so CIs are a bootstrap over authors, not posts.

This measures the vector gate only. GE's ranker and the per-reader language
ordering sit after it and are not modelled.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
os.environ.setdefault("LOC_TEACHER_FILE", str(ROOT / "training/data/dataset_annotated_union_jev_decider.jsonl"))
import train_internal_vector as t  # noqa: E402

CORPUS = ROOT / "training/data/dataset_annotated_decider_net20k.jsonl"
CACHE = ROOT / "training/data/embeddings_union_jina.npz"
VECTOR = ROOT / "bunny_feed/src/vector.json"
GENDER = ROOT / "research/data/gender_bias/author_gender.jsonl"
GATE = float(os.environ.get("GB_GATE", 0.86))
GROUPS = ["female", "male", "nonbinary", "unknown"]
B = 2000


def load():
    # The cache is positional over load_train_rows() of the union file; the 20k
    # rows are picked out of it by URI, never by assumed offset.
    rows = t.load_train_rows()
    X = np.load(CACHE)["X"]
    assert len(rows) == len(X)
    # (uri, text): one URI appears twice in the union with an edited text.
    pos = {(r["uri"], r["text"]): i for i, r in enumerate(rows)}
    corpus = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    idx = np.array([pos[(r["uri"], r["text"])] for r in corpus])
    v = json.load(open(VECTOR))
    E = X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)
    prob = 1 / (1 + np.exp(-(v["scale"] * (E @ np.array(v["vector"])) + v["bias"])))
    gender = {g["did"]: g["gender"] for g in map(json.loads, open(GENDER, encoding="utf-8"))}
    return corpus, prob, np.array([gender.get(r["did"], "unknown") for r in corpus])


def metrics(m):
    out = {"posts": int(m["sel"].sum()), "authors": len(set(m["did"][m["sel"]]))}
    s = m["sel"]
    out["teacher_int"] = m["int"][s].mean()
    out["pass"] = m["pass"][s].mean()
    out["pass|int"] = m["pass"][s & m["int"]].mean() if (s & m["int"]).any() else np.nan
    out["pass|~int"] = m["pass"][s & ~m["int"]].mean()
    out["top5"] = m["top5"][s].mean()
    out["mean_p"] = m["prob"][s].mean()
    return out


def boot_ci(base, sel, key, rng):
    """Author-clustered bootstrap CI for one metric."""
    dids = np.unique(base["did"][sel])
    by = {d: np.flatnonzero(sel & (base["did"] == d)) for d in dids}
    vals = []
    for _ in range(B):
        ix = np.concatenate([by[d] for d in rng.choice(dids, len(dids))])
        sub = {k: v[ix] for k, v in base.items()}
        sub["sel"] = np.ones(len(ix), bool)
        vals.append(metrics(sub)[key])
    return np.nanpercentile(vals, [2.5, 97.5])


def report(corpus, prob, gender, lang_filter, title, rng):
    lang = np.array([(r["langs"] or ["none"])[0] for r in corpus])
    keep = np.ones(len(corpus), bool) if lang_filter is None else lang == lang_filter
    base = {
        "did": np.array([r["did"] for r in corpus])[keep],
        "int": np.array([r["loc_label"] == "internal" for r in corpus])[keep],
        "prob": prob[keep],
        "pass": prob[keep] >= GATE,
        "top5": prob[keep] >= np.quantile(prob[keep], 0.95),
    }
    g = gender[keep]
    print(f"\n## {title}  (n={keep.sum()}, gate {GATE}, overall pass {base['pass'].mean():.1%})")
    print(f"{'group':10s} {'posts':>6s} {'auth':>5s} {'teach_int':>10s} {'pass':>18s} {'pass|int':>18s} {'pass|~int':>9s} {'top5':>6s}")
    res = {}
    for grp in GROUPS:
        sel = g == grp
        if sel.sum() < 30:
            continue
        m = metrics({**base, "sel": sel})
        lo, hi = boot_ci(base, sel, "pass", rng)
        lo2, hi2 = boot_ci(base, sel, "pass|int", rng)
        res[grp] = (m, (lo, hi))
        print(f"{grp:10s} {m['posts']:6d} {m['authors']:5d} {m['teacher_int']:10.1%} "
              f"{m['pass']:6.1%} [{lo:5.1%},{hi:5.1%}] {m['pass|int']:6.1%} [{lo2:5.1%},{hi2:5.1%}] "
              f"{m['pass|~int']:9.1%} {m['top5']:6.1%}")
    # Female-vs-male difference with a joint author bootstrap.
    fm = (g == "female") | (g == "male")
    dids_f = np.unique(base["did"][g == "female"]); dids_m = np.unique(base["did"][g == "male"])
    idx = {d: np.flatnonzero(base["did"] == d) for d in np.concatenate([dids_f, dids_m])}
    diffs = {"pass": [], "teacher_int": [], "pass|int": []}
    for _ in range(B):
        f = np.concatenate([idx[d] for d in rng.choice(dids_f, len(dids_f))])
        mm = np.concatenate([idx[d] for d in rng.choice(dids_m, len(dids_m))])
        for k in diffs:
            a = metrics({**{kk: v[f] for kk, v in base.items()}, "sel": np.ones(len(f), bool)})[k]
            b = metrics({**{kk: v[mm] for kk, v in base.items()}, "sel": np.ones(len(mm), bool)})[k]
            diffs[k].append(a - b)
    for k, v in diffs.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        print(f"  female - male  {k:12s} {np.nanmean(v):+6.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
    # Within-author-type: does the vector add a gap once the teacher label is fixed?
    within = []
    for lab in (True, False):
        s = fm & (base["int"] == lab)
        if s.sum():
            within.append(f"teacher {'internal' if lab else 'other'}: "
                          f"F {base['pass'][s & (g == 'female')].mean():.1%} vs M {base['pass'][s & (g == 'male')].mean():.1%}")
    print("  " + " | ".join(within))


def main():
    rng = np.random.default_rng(0)
    corpus, prob, gender = load()
    report(corpus, prob, gender, None, "all languages", rng)
    report(corpus, prob, gender, "en", "English", rng)
    report(corpus, prob, gender, "de", "German", rng)


if __name__ == "__main__":
    main()
