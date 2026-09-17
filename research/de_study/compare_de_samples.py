"""Compare the within-author findings between the primary and root-only samples.

The point of the comparison is NOT to count how many tests cross 0.05 in each. The
two samples have different n and different multiple-comparison burdens, so a result
can hold its effect size exactly and still change significance status -- which is
what happens here. What matters is whether the *point estimate* reproduces.

Three verdicts:

  reproduces  same sign, and the two estimates are within 0.02 of each other. The
              association is present in root posts as well as replies.
  reply-driven  the primary estimate is much larger, or the sign flips. The
              association lives in replies, whose engagement is driven by thread
              context this study does not measure -- so it is the weakest kind.
  root-driven   the reverse: stronger among root posts than in the mixed sample.

Samples overlap by construction (a root post can be in both), so this is a
consistency check on a shared window, not independent replication.
"""
import argparse
import json
import sys

CLOSE = 0.02


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return {(r["attribute"], r["outcome"]): r for r in json.load(f)["results"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="REPORT_de_engagement_findings.json")
    ap.add_argument("--root", default="REPORT_de_engagement_findings_root.json")
    ap.add_argument("--out", default="REPORT_de_sample_comparison.md")
    ap.add_argument("--q", type=float, default=0.05)
    args = ap.parse_args()

    P, R = load(args.primary), load(args.root)
    keys = sorted(set(P) & set(R))
    if not keys:
        print("no overlapping tests", file=sys.stderr)
        return 1

    rows = []
    for k in keys:
        p, r = P[k], R[k]
        rp, rr = p.get("within_author_r"), r.get("within_author_r")
        if rp is None or rr is None:
            continue
        qp, qr = p.get("within_author_q"), r.get("within_author_q")
        sig_any = (qp is not None and qp < args.q) or (qr is not None and qr < args.q)
        if not sig_any:
            continue
        if rp * rr <= 0:
            verdict = "sign flips"
        elif abs(rp - rr) <= CLOSE:
            verdict = "reproduces"
        elif abs(rp) > abs(rr):
            verdict = "reply-driven"
        else:
            verdict = "root-driven"
        rows.append({"attribute": k[0], "outcome": k[1], "r_primary": rp, "q_primary": qp,
                     "r_root": rr, "q_root": qr, "delta": round(rr - rp, 4),
                     "verdict": verdict, "flag": p.get("verdict")})

    order = {"reproduces": 0, "root-driven": 1, "reply-driven": 2, "sign flips": 3}
    rows.sort(key=lambda x: (order[x["verdict"]], -abs(x["r_primary"])))

    L = ["# Primary vs root-only: do the associations reproduce?", "",
         "Within-author estimates from the two samples, for every attribute x outcome "
         f"significant at FDR < {args.q} in **either**. Both are the `early` snapshot "
         "(counts not yet settled).", "",
         "> Significance counts are not comparable across the two samples -- different "
         "n, different FDR burden. A result can hold its effect size exactly and still "
         "cross the threshold in one and not the other. Compare the estimates.", "",
         "| attribute | outcome | r (primary) | q | r (root) | q | delta | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        L.append(f"| {x['attribute']} | {x['outcome']} | {x['r_primary']:+.4f} | "
                 f"{x['q_primary']} | {x['r_root']:+.4f} | {x['q_root']} | "
                 f"{x['delta']:+.4f} | {x['verdict']} |")

    counts = {}
    for x in rows:
        counts[x["verdict"]] = counts.get(x["verdict"], 0) + 1
    L += ["", "## Summary", ""]
    for v in ("reproduces", "root-driven", "reply-driven", "sign flips"):
        if counts.get(v):
            L.append(f"- **{v}**: {counts[v]}")
    L += ["", "Associations marked `reply-driven` or `sign flips` should not be "
          "reported as content effects. A reply's engagement is largely a function of "
          "the thread it sits in -- the parent's reach and the conversation's size -- "
          "and neither is in this table, so those estimates absorb thread context that "
          "cannot be conditioned away.", "",
          "Associations marked `reproduces` hold in both the mixed population and "
          "among root posts, where exposure is better measured. Those are the "
          "defensible ones."]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:9] + [f"| ... {len(rows)} rows ..." ]))
    print(f"\nsummary: {counts}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
