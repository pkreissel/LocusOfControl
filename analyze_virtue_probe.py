"""Diagnose whether a virtue-axis annotation run carries usable ranking signal.

The virtue axis cannot rely on its positive class: 'virtuous' posts are under 1%
of a live sample. The ranking signal therefore has to come from the graded 0-9
Score spread ACROSS the neutral mass - a neutral post that hedges honestly must
outrank a neutral post that asserts flatly. This script measures whether that
gradient exists, and shows the top-scoring neutrals so the ordering can be
judged by eye rather than trusted.

Two failure modes it is built to catch:
  1. Degenerate scores - the teacher collapsing every neutral onto one digit
     (what the hard Score band clamp did before it was removed).
  2. The register confound - the score ranking merely length or vocabulary.
     Reported as the Spearman correlation of score with text length; if that is
     high, the vector will learn prose style instead of epistemics.

  LOC_PROBE_FILE  annotated jsonl to analyse
"""
import collections
import json
import os

import numpy as np

FILE = os.environ.get("LOC_PROBE_FILE", "dataset_annotated_virtue_v1.jsonl")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def main():
    rows = [json.loads(l) for l in open(FILE, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("loc_score") is not None]
    print(f"{len(rows)} annotated posts from {FILE}\n")

    by_pool = collections.Counter(r.get("pool", "?") for r in rows)
    print("label distribution")
    for pool in sorted(by_pool):
        sub = [r for r in rows if r.get("pool", "?") == pool]
        c = collections.Counter(r["loc_label"] for r in sub)
        parts = "  ".join(f"{k} {c[k]:3d} ({c[k]/len(sub)*100:4.1f}%)"
                          for k in ("virtuous", "dogmatic", "neutral"))
        print(f"  {pool:9s} n={len(sub):3d}   {parts}")

    neu = [r for r in rows if r["loc_label"] == "neutral"]
    if not neu:
        print("\nno neutral posts - nothing to check")
        return

    sc = np.array([r["loc_score"] for r in neu])
    print(f"\nscore gradient inside the neutral mass (n={len(neu)})")
    print(f"  distinct scores (1dp): {len(set(np.round(sc, 1)))}   "
          f"min {sc.min():.2f}  median {np.median(sc):.2f}  max {sc.max():.2f}  sd {sc.std():.2f}")
    hist = collections.Counter(int(round(s)) for s in sc)
    print("  histogram: " + "  ".join(f"{d}:{hist.get(d,0)}" for d in range(10)))
    if len(set(np.round(sc, 1))) <= 3:
        print("  WARNING: degenerate - the teacher is not discriminating inside neutral.")

    lens = np.array([len(r["text"]) for r in neu])
    rho = spearman(sc, lens)
    print(f"\n  score vs text length, Spearman rho = {rho:+.2f}"
          f"{'   WARNING: score is largely tracking length' if rho > 0.5 else ''}")

    order = np.argsort(-sc)
    print("\ntop-scoring neutrals - are these genuinely the more careful posts?")
    for i in order[:10]:
        r = neu[i]
        print(f"  [{r['loc_score']:.2f}] ({len(r['text']):4d} ch) "
              f"{r['text'][:180].replace(chr(10), ' / ')}")
    print("\nbottom-scoring neutrals, for contrast")
    for i in order[-5:]:
        r = neu[i]
        print(f"  [{r['loc_score']:.2f}] ({len(r['text']):4d} ch) "
              f"{r['text'][:180].replace(chr(10), ' / ')}")


if __name__ == "__main__":
    main()
