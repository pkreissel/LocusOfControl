"""Put TypeSafe's LoC judgement on the deployed vector's probability scale.

The feed gates on one number: `post_scores.score`, read against the per-feed
threshold in feeds.ts (LoC: 0.86). That number is the vector's calibrated
sigmoid, and the threshold was measured against ITS distribution. A TypeSafe
probability is a different distribution entirely -- P(internal) is 0.00 for a
quarter of a live feed and its p95 is 0.85 -- so writing one into that column
raw would be read by a gate that means something else, and the same 0.86 that
keeps ~44% of vector-scored posts would keep ~5% of TypeSafe-scored ones.

So the map here is a monotone quantile match: rank the pool by TypeSafe, rank it
by the vector, and give each post the vector probability sitting at its own
TypeSafe percentile. Order comes from TypeSafe; the SCALE, and therefore every
threshold expressed on it, stays the vector's. By construction the keep-rate at
any cut is identical -- what changes is which posts are on which side of it.

Ranking key is `band`, the same ordering train_internal_vector.py's band target
encodes: a post TypeSafe calls external is floored below everything else, and the
rest rank by P(internal) with the graded Score breaking ties (the API rounds its
probabilities to two decimals, which ties ~30% of a live pool at 0.00).

That floor is the whole key, and leaving it out is not a small difference.
Ranking by P(internal) alone puts an unsure external (P(int) 0.30, P(ext) 0.55)
above a confident neutral, and at the vector's own keep-rate on 1001 silberskeets
posts it leaked 43.9% of the externals -- worse than the vector's 29.2%. With the
floor: 0.8%, at 100% of the internals kept. Measured with --key:

  key                         internal kept   external leaked
  band (default)                     100.0%              0.8%
  P(internal) - P(external)          100.0%              3.4%
  ts_score (graded axis)              95.6%              9.4%
  agency noul                         98.8%             24.8%
  P(internal), ts_score              100.0%             43.9%
  deployed vector                     67.5%             29.2%

Read the vector's row as the only independent one. Every other row ranks TypeSafe
by TypeSafe's own labels, which measures self-consistency, not accuracy -- the
same trap as validating a distilled vector against its own teacher. The reason to
prefer `band` is the axis, not the table: the feed exists to exclude external, and
that is the key that does it.

The map is fitted on ONE pool and is only valid for it -- a percentile describes
the pool it was measured on, and this deployment draws others (see feeds.ts).
Emitted rows are marked `score_source: typesafe_calibrated` so a pushed cache row
is never mistaken for one the worker computed itself.

Run:
  python calibrate_typesafe_to_vector.py --vector ../data/feed/scores_loc_vector_<tag>_<ver>.jsonl \
      --typesafe ../data/feed/scores_loc_typesafe_<tag>.jsonl --posts ../data/feed/feed_posts_<tag>.jsonl \
      --threshold 0.86 [--out ../data/feed/scores_push_<tag>.jsonl]
"""

import argparse
import json
from collections import Counter

import numpy as np


# Ascending, so the largest key gets the highest vector probability. `band` is the
# default and the only one that floors external; the rest exist to reproduce the
# comparison in the docstring, not to be deployed.
KEYS = {
    # external floored below everything, then P(internal), Score breaking ties.
    "band": lambda pi, pe, s, n: (0 if pe > max(pi, 1.0 - pi - pe) else 1, pi, s),
    "margin": lambda pi, pe, s, n: (pi - pe, s),
    "score": lambda pi, pe, s, n: (s, pi),
    "noul": lambda pi, pe, s, n: (n, pi),
    "internal": lambda pi, pe, s, n: (pi, s),
}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def index(path):
    return {r["uri"]: r for r in load(path)} if path else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--typesafe", required=True)
    ap.add_argument("--threshold", type=float, default=0.86)
    ap.add_argument("--key", default="band", choices=sorted(KEYS),
                    help="how TypeSafe's answers are folded into one ranking")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    posts, vec = index(args.posts), index(args.vector)
    ts = {r["uri"]: r for r in load(args.typesafe) if "error" not in r}
    uris = [u for u in posts if u in vec and u in ts]
    print(f"{len(posts)} posts | vector {len(vec)} | typesafe {len(ts)} | "
          f"both {len(uris)}")
    if not uris:
        raise SystemExit("No overlap to calibrate on.")

    version = vec[uris[0]]["vector_version"]
    if any(vec[u]["vector_version"] != version for u in uris):
        raise SystemExit("Vector scores mix versions; re-run score_loc_vector.py.")

    p_vec = np.array([vec[u]["vector_prob"] for u in uris])
    p_int = np.array([ts[u]["ts_class_probs"]["internal"] for u in uris])
    s_ts = np.array([ts[u]["ts_score"] for u in uris])
    p_ext = np.array([ts[u]["ts_class_probs"].get("external", 0.0) for u in uris])
    noul = np.array([ts[u]["ts_noul"] for u in uris])
    cls = [ts[u]["ts_class"] for u in uris]

    order = sorted(range(len(uris)), key=lambda i: KEYS[args.key](p_int[i], p_ext[i], s_ts[i], noul[i]))
    # Quantile match: i-th smallest by TypeSafe gets the i-th smallest vector prob.
    ladder = np.sort(p_vec)
    cal = np.empty(len(uris))
    for rank, i in enumerate(order):
        cal[i] = ladder[rank]

    t = args.threshold
    print(f"\nGate {t} on {len(uris)} posts")
    print(f"  vector                  keeps {int((p_vec >= t).sum()):4d}  "
          f"({(p_vec >= t).mean()*100:5.1f}%)")
    print(f"  typesafe raw P(int)     keeps {int((p_int >= t).sum()):4d}  "
          f"({(p_int >= t).mean()*100:5.1f}%)   <- what an uncalibrated push would do")
    print(f"  typesafe calibrated     keeps {int((cal >= t).sum()):4d}  "
          f"({(cal >= t).mean()*100:5.1f}%)")

    # What each gate actually admits, judged by TypeSafe's own class.
    print("\nComposition of what the gate admits (TypeSafe loc_class):")
    for name, s in (("vector", p_vec), ("typesafe calibrated", cal)):
        kept = [cls[i] for i in range(len(uris)) if s[i] >= t]
        c = Counter(kept)
        n = max(len(kept), 1)
        print(f"  {name:20s} n={len(kept):4d}  "
              f"internal {c['internal']:4d} ({c['internal']/n*100:4.1f}%)  "
              f"neutral {c['neutral']:4d} ({c['neutral']/n*100:4.1f}%)  "
              f"external {c['external']:4d} ({c['external']/n*100:4.1f}%)")
    n_int = sum(1 for c in cls if c == "internal")
    n_ext = sum(1 for c in cls if c == "external")
    for name, s in (("vector", p_vec), ("typesafe calibrated", cal)):
        keep = s >= t
        ri = sum(1 for i in range(len(uris)) if keep[i] and cls[i] == "internal") / max(n_int, 1)
        re = sum(1 for i in range(len(uris)) if keep[i] and cls[i] == "external") / max(n_ext, 1)
        print(f"  {name:20s} internal kept {ri*100:5.1f}%   external leaked {re*100:5.1f}%")

    # How far the ordering moves, which is the whole point of the swap.
    def ranks(x):
        o = np.argsort(np.argsort(x))
        return o.astype(float)
    rv, rc = ranks(p_vec), ranks(cal)
    rho = float(np.corrcoef(rv, rc)[0, 1])
    print(f"\nSpearman(vector, typesafe) = {rho:.3f}")
    for k in (25, 100, 484):
        k = min(k, len(uris))
        top_v = {uris[i] for i in np.argsort(-p_vec)[:k]}
        top_c = {uris[i] for i in np.argsort(-cal)[:k]}
        print(f"  top {k:3d}: {len(top_v & top_c):3d} shared "
              f"(Jaccard {len(top_v & top_c)/len(top_v | top_c):.2f})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for i, u in enumerate(uris):
                f.write(json.dumps({
                    "uri": u,
                    "vector_prob": float(cal[i]),
                    "vector_version": version,
                    "score_source": "typesafe_calibrated",
                    "ts_class": cls[i],
                    "ts_p_internal": float(p_int[i]),
                    "vector_prob_own": float(p_vec[i]),
                }, ensure_ascii=False) + "\n")
        print(f"\n{args.out}: {len(uris)} calibrated scores, filed under version {version}")


if __name__ == "__main__":
    main()
