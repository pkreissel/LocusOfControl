"""Grade each final ordering from order_eval.ts with the judge's labels.

The judge is decider-2b + the fine-tuned LoRA (see score_pools.py). Per pool and
locWeight, over the top k posts a reader actually sees:

  P(int)@k   mean judge P(internal) -- the graded version of precision
  int@k      share the judge classes internal
  ext@k      share it classes external (the leak the feed exists to suppress)
  nDCG@k     gain = P(internal), against the ideal order of the same list
  ge@k       mean engagement rank (1 best, 0 worst) -- what the weight costs

Selection is fixed across weights, so every difference here is ordering alone.
Paired over pools against w=0 (the order before 12f16e7), with a bootstrap over
pools for the CI.

Run: tmp_venv/bin/python research/feed_order/metrics.py pools_<date>
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "feed_order")
KS = (10, 25)
BASE_W, DEPLOYED_W = 0.0, 0.65


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def ndcg(gains, k):
    g = np.asarray(gains, dtype=float)
    disc = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(g)[::-1][:k]
    return float((g[:k] * disc[:len(g[:k])]).sum() / max((ideal * disc[:len(ideal)]).sum(), 1e-12))


def main():
    stem = sys.argv[1]
    judge = {r["uri"]: r for r in load(os.path.join(DATA, f"{stem}_judge.jsonl")) if "error" not in r}
    orders = load(os.path.join(DATA, f"{stem}_orders.jsonl"))

    res = {}  # (pool, w) -> metrics
    for o in orders:
        uris = [u for u in o["uris"] if u in judge]
        ge = [g for u, g in zip(o["uris"], o["geRank"]) if u in judge]
        pint = [judge[u]["ts_class_probs"]["internal"] for u in uris]
        cls = [judge[u]["ts_class"] for u in uris]
        m = {"n": len(uris)}
        for k in KS:
            m[f"P(int)@{k}"] = float(np.mean(pint[:k]))
            m[f"int@{k}"] = float(np.mean([c == "internal" for c in cls[:k]]))
            m[f"ext@{k}"] = float(np.mean([c == "external" for c in cls[:k]]))
            m[f"nDCG@{k}"] = ndcg(pint, k)
            m[f"ge@{k}"] = float(np.mean(ge[:k]))
        m["P(int)all"] = float(np.mean(pint))
        m["int_all"] = float(np.mean([c == "internal" for c in cls]))
        res[(o["pool"], o["weight"])] = m

    pools = sorted({p for p, _ in res})
    weights = sorted({w for _, w in res})
    cols = [f"{m}@{k}" for k in KS for m in ("P(int)", "int", "ext", "nDCG", "ge")]

    print(f"{len(pools)} pools, judge = decider_ft; lists n = "
          + ", ".join(str(res[(p, 0.0)]["n"]) for p in pools))
    print(f"list-wide (same for every weight): judge internal share "
          f"{np.mean([res[(p, 0.0)]['int_all'] for p in pools]):.3f}, "
          f"P(int) {np.mean([res[(p, 0.0)]['P(int)all'] for p in pools]):.3f}\n")

    print("mean over pools")
    print(f"{'w':>5} " + " ".join(f"{c:>10}" for c in cols))
    for w in weights:
        print(f"{w:>5} " + " ".join(f"{np.mean([res[(p, w)][c] for p in pools]):>10.3f}" for c in cols))

    rng = np.random.default_rng(0)
    print(f"\npaired vs w={BASE_W} (bootstrap over pools, 10k), wins = pools improved/tied/worse")
    for w in weights:
        if w == BASE_W:
            continue
        print(f"w={w}")
        for c in cols:
            d = np.array([res[(p, w)][c] - res[(p, BASE_W)][c] for p in pools])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            good = -1 if c.startswith("ext") else 1
            wins = (int((good * d > 1e-9).sum()), int((abs(d) <= 1e-9).sum()), int((good * d < -1e-9).sum()))
            print(f"   {c:>10} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  {wins[0]}/{wins[1]}/{wins[2]}")

    print(f"\nper pool at @25: w={BASE_W} -> w={DEPLOYED_W}")
    for p in pools:
        a, b = res[(p, BASE_W)], res[(p, DEPLOYED_W)]
        print(f"  {p:28} int@25 {a['int@25']:.2f} -> {b['int@25']:.2f}   "
              f"ext@25 {a['ext@25']:.2f} -> {b['ext@25']:.2f}   ge@25 {a['ge@25']:.2f} -> {b['ge@25']:.2f}")

    with open(os.path.join(DATA, f"{stem}_metrics.json"), "w") as f:
        json.dump({f"{p}|{w}": m for (p, w), m in res.items()}, f, indent=1)


if __name__ == "__main__":
    main()
