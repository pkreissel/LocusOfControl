"""Do Perspective scores differ by hour of day, and does engagement track them?

Reported in **Europe/Berlin local time**, not UTC: the population is German-language
posters, so 03:00 local is the meaningful "middle of the night", and the study
window covers each local hour exactly once.

Inference is clustered by author throughout. The test for "does this attribute vary
by hour at all" is a joint Wald test on 23 hour dummies with a cluster-robust
covariance matrix -- not Kruskal-Wallis or a plain ANOVA, both of which would treat
a prolific author's 40 posts as 40 independent observations and badly understate
the p-values.

IMPORTANT LIMITATION, stated here because it is easy to forget: the sample spans a
single 24-hour window, so *hour of day* and *that specific hour on that specific
day* are perfectly confounded. A spike in hostility at 20:00 local could be a
diurnal rhythm or one news story that evening; this design cannot tell them apart.
Separating them needs several days of windows. Read the profile as descriptive.
"""
import argparse
import csv
import datetime as dt
import json
import math
import sys
from collections import Counter
from zoneinfo import ZoneInfo

import numpy as np
from scipy import stats

BERLIN = ZoneInfo("Europe/Berlin")
OUTCOMES = ["like_count", "repost_count", "reply_count", "quote_count", "total_engagement"]

# Pre-specified, NOT chosen by looking at the data. Peak-vs-trough is a maximum over
# 24 noisy hourly means -- with only ~35 posts in the quietest hour, the winner is
# biased high and the loser low, so that gap overstates any real swing. These two
# blocks are fixed in advance and each pools several hundred posts.
NIGHT = set(range(0, 6))     # 00:00-05:59 local
DAY = set(range(9, 18))      # 09:00-17:59 local

# Composition varies by hour: night posts are shorter and likelier to be replies,
# and Perspective scores move with both. The adjusted test asks whether the hour
# effect survives holding post shape constant.
COMPOSITION = ["text_len", "is_reply", "n_images", "has_video", "has_external",
               "n_links", "n_mentions", "n_tags"]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(rows, col):
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = r.get(col)
        if v not in (None, "", "None"):
            try:
                out[i] = float(v)
            except ValueError:
                pass
    return out


def local_hours(rows):
    hrs = np.full(len(rows), -1, dtype=int)
    for i, r in enumerate(rows):
        s = r.get("created_utc")
        if not s:
            continue
        try:
            t = dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        hrs[i] = t.astimezone(BERLIN).hour
    return hrs


def cluster_cov(y, X, groups):
    """OLS with CR1 cluster-robust covariance. Returns (beta, V)."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((k, k))
    uniq = np.unique(groups)
    for g in uniq:
        m = groups == g
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    G = len(uniq)
    if G <= k:
        return beta, None
    scale = G / (G - 1) * (n - 1) / (n - k)
    return beta, XtX_inv @ meat @ XtX_inv * scale


def hour_wald(y, hrs, groups, Z=None):
    """Joint test that the 24 hourly means are not all equal, clustered by author.
    Hour 0 is the reference; the Wald statistic covers the other 23 dummies.
    Passing Z adds composition controls, so the test becomes "does hour matter
    beyond the fact that posts made at different hours look different"."""
    present = sorted(set(int(h) for h in hrs if h >= 0))
    if len(present) < 3:
        return float("nan"), float("nan")
    others = present[1:]
    cols = [np.ones(len(y))] + [(hrs == h).astype(float) for h in others]
    if Z is not None and Z.size:
        live = [j for j in range(Z.shape[1]) if np.nanstd(Z[:, j]) > 1e-9]
        if live:
            cols.append(Z[:, live])
    X = np.column_stack(cols)
    beta, V = cluster_cov(y, X, groups)
    if V is None:
        return float("nan"), float("nan")
    idx = np.arange(1, 1 + len(others))
    b = beta[idx]
    Vb = V[np.ix_(idx, idx)]
    try:
        W = float(b @ np.linalg.pinv(Vb) @ b)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    df = len(others)
    return W, float(stats.chi2.sf(W, df))


def block_contrast(x, hrs, groups, Z=None):
    """Night-vs-day difference with cluster-robust inference, optionally holding
    post composition constant. Returns the difference on x's own scale."""
    sel = np.array([int(h) in NIGHT or int(h) in DAY for h in hrs])
    if sel.sum() < 100:
        return None
    xs, gs = x[sel], groups[sel]
    night = np.array([1.0 if int(h) in NIGHT else 0.0 for h in hrs[sel]])
    if night.sum() < 20 or (len(night) - night.sum()) < 20:
        return None
    cols = [np.ones(len(xs)), night]
    if Z is not None and Z.size:
        Zs = Z[sel]
        live = [j for j in range(Zs.shape[1]) if np.nanstd(Zs[:, j]) > 1e-9]
        if live:
            cols.append(Zs[:, live])
    X = np.column_stack(cols)
    beta, V = cluster_cov(xs, X, gs)
    if V is None:
        return None
    se = math.sqrt(max(V[1, 1], 0))
    if se == 0:
        return None
    t = beta[1] / se
    G = len(np.unique(gs))
    return {
        "n_night": int(night.sum()), "n_day": int(len(night) - night.sum()),
        "diff": round(float(beta[1]), 5), "se": round(se, 5),
        "t": round(float(t), 2), "p": float(2 * stats.t.sf(abs(t), df=G - 1)),
    }


def cluster_se_of_mean(x, groups):
    """SE of a mean when observations cluster by author."""
    n = len(x)
    if n < 2:
        return float("nan")
    mu = x.mean()
    dev = x - mu
    tot = 0.0
    uniq = np.unique(groups)
    for g in uniq:
        tot += dev[groups == g].sum() ** 2
    G = len(uniq)
    if G < 2:
        return float("nan")
    return math.sqrt(tot * G / (G - 1)) / n


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = np.minimum.accumulate((ranked * m / (np.arange(m) + 1))[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    q[ok] = res
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="de_study_table.csv")
    ap.add_argument("--validity", default="REPORT_perspective_de_validity.json")
    ap.add_argument("--out", default="REPORT_de_diurnal.json")
    ap.add_argument("--md-out", default="REPORT_de_diurnal.md")
    ap.add_argument("--label", default="primary")
    args = ap.parse_args()

    rows = load(args.table)
    if not rows:
        print("empty table", file=sys.stderr)
        return 1
    hrs = local_hours(rows)
    dids = np.array([r["did"] for r in rows])
    ok = hrs >= 0
    rows = [r for r, k in zip(rows, ok) if k]
    hrs, dids = hrs[ok], dids[ok]
    print(f"{len(rows)} rows with a resolvable local hour")

    flags = {}
    try:
        with open(args.validity) as f:
            rep = json.load(f)
        flags = {f"p_{a.lower()}": rep["attributes"][a]["verdict"] for a in rep["attributes"]}
    except (OSError, KeyError, json.JSONDecodeError):
        pass

    attrs = [c for c in rows[0] if c.startswith("p_")]
    counts = Counter(int(h) for h in hrs)
    volume = {h: counts.get(h, 0) for h in range(24)}

    Zc = np.column_stack([num(rows, c) for c in COMPOSITION])
    for j in range(Zc.shape[1]):
        col = Zc[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = np.nanmedian(col) if not np.all(np.isnan(col)) else 0.0

    results = []
    for a in attrs:
        x = num(rows, a)
        m = ~np.isnan(x)
        if m.sum() < 200:
            continue
        xs, hs, gs = x[m], hrs[m], dids[m]
        by_hour = {}
        for h in range(24):
            sel = hs == h
            if sel.sum() >= 5:
                by_hour[h] = {
                    "n": int(sel.sum()),
                    "mean": round(float(xs[sel].mean()), 5),
                    "se": round(float(cluster_se_of_mean(xs[sel], gs[sel])), 5),
                    "median": round(float(np.median(xs[sel])), 5),
                }
        if len(by_hour) < 3:
            continue
        # Rank-transform before the Wald test: these scores are bounded and skewed,
        # so a test on the raw scale is driven by a handful of extreme posts.
        rk = stats.rankdata(xs)
        Zm = Zc[m]
        W, p = hour_wald(rk, hs, gs)
        W_adj, p_adj = hour_wald(rk, hs, gs, Z=Zm)
        means = {h: v["mean"] for h, v in by_hour.items()}
        peak = max(means, key=means.get)
        trough = min(means, key=means.get)
        results.append({
            "attribute": a[2:], "verdict": flags.get(a, "unknown"),
            "n": int(m.sum()), "wald_chi2": None if W != W else round(W, 2),
            "p_hour_effect": p,
            "wald_chi2_adjusted": None if W_adj != W_adj else round(W_adj, 2),
            "p_hour_effect_adjusted": p_adj,
            "night_vs_day_raw": block_contrast(xs, hs, gs),
            "night_vs_day_adjusted": block_contrast(xs, hs, gs, Z=Zm),
            "peak_n": by_hour[peak]["n"], "trough_n": by_hour[trough]["n"],
            "peak_hour_local": peak, "peak_mean": means[peak],
            "trough_hour_local": trough, "trough_mean": means[trough],
            "peak_minus_trough": round(means[peak] - means[trough], 5),
            "relative_swing": (round((means[peak] - means[trough]) / means[trough], 3)
                               if means[trough] > 1e-9 else None),
            "by_hour": by_hour,
        })

    for r, q in zip(results, bh_fdr([r["p_hour_effect"] for r in results])):
        r["q_hour_effect"] = None if q != q else round(float(q), 6)
    for r, q in zip(results, bh_fdr([r["p_hour_effect_adjusted"] for r in results])):
        r["q_hour_effect_adjusted"] = None if q != q else round(float(q), 6)
    for r, q in zip(results, bh_fdr([(r["night_vs_day_adjusted"] or {}).get("p", float("nan"))
                                     for r in results])):
        r["q_night_vs_day_adjusted"] = None if q != q else round(float(q), 6)

    eng = []
    for oc in OUTCOMES:
        y = num(rows, oc)
        m = ~np.isnan(y)
        ys, hs, gs = y[m], hrs[m], dids[m]
        W, p = hour_wald(stats.rankdata(ys), hs, gs)
        by_hour = {h: {"n": int((hs == h).sum()),
                       "mean": round(float(ys[hs == h].mean()), 3),
                       "median": float(np.median(ys[hs == h]))}
                   for h in range(24) if (hs == h).sum() >= 5}
        means = {h: v["mean"] for h, v in by_hour.items()}
        eng.append({"outcome": oc, "wald_chi2": None if W != W else round(W, 2),
                    "p_hour_effect": p,
                    "peak_hour_local": max(means, key=means.get) if means else None,
                    "trough_hour_local": min(means, key=means.get) if means else None,
                    "by_hour": by_hour})

    report = {"label": args.label, "n": len(rows), "timezone": "Europe/Berlin",
              "volume_by_local_hour": volume, "attributes": results, "engagement": eng}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    sig = [r for r in results if r["q_hour_effect"] is not None and r["q_hour_effect"] < 0.05
           and r["verdict"] in ("keep", "low_power")]
    sig.sort(key=lambda r: -abs(r["peak_minus_trough"]))

    L = [f"# Diurnal profile of Perspective scores ({args.label} sample)", "",
         f"n = {len(rows)}. Hours are **Europe/Berlin local time**. SEs and tests are "
         "clustered by author.", "",
         "> **Single-window caveat.** The sample spans one 24-hour window, so hour of "
         "day is perfectly confounded with that particular hour on that particular "
         "day. A peak at 20:00 could be a nightly rhythm or one evening's news cycle "
         "— this design cannot separate them. Treat the profile as descriptive.", "",
         "## Posting volume by local hour", "",
         "| hour | posts |", "|---|---|"]
    for h in range(24):
        L.append(f"| {h:02d} | {volume.get(h, 0)} |")

    nd = [r for r in results
          if r.get("night_vs_day_adjusted")
          and r.get("q_night_vs_day_adjusted") is not None
          and r["q_night_vs_day_adjusted"] < 0.05
          and r["verdict"] in ("keep", "low_power")]
    nd.sort(key=lambda r: -abs(r["night_vs_day_adjusted"]["diff"]))

    L += ["", "## Night vs day, adjusted for post composition", "",
          f"Pre-specified blocks: night = {min(NIGHT):02d}:00-{max(NIGHT):02d}:59, "
          f"day = {min(DAY):02d}:00-{max(DAY):02d}:59 local. Positive = higher at "
          "night. Adjusted for text length, reply status, images, video, external "
          "links, and link/mention/tag counts, because night posts are shorter and "
          "likelier to be replies, and Perspective scores move with both.", "",
          "**This table, not the peak/trough one below, is the result to quote.** "
          "It is a fixed contrast decided in advance, so it is not inflated by "
          "picking the most extreme of 24 noisy hourly means.", "",
          "| attribute | night-day diff | SE | q | n night | n day | flag |",
          "|---|---|---|---|---|---|---|"]
    for r in nd:
        c = r["night_vs_day_adjusted"]
        L.append(f"| {r['attribute']} | {c['diff']:+.4f} | {c['se']:.4f} | "
                 f"{r['q_night_vs_day_adjusted']} | {c['n_night']} | {c['n_day']} | "
                 f"{r['verdict']} |")
    if not nd:
        L.append("| _none_ — no attribute shifts significantly between night and day "
                 "once composition is held constant_ | | | | | | |")

    L += ["", "### All attributes, with minimum detectable effect", "",
          "No significant shift is only informative alongside what the design could "
          "have caught. MDE is the smallest night-day difference this sample would "
          "detect at 80% power (2.8 x SE), on the attribute's own 0-1 scale.", "",
          "| attribute | raw diff | adjusted diff | SE | MDE | q | flag |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -abs((r.get("night_vs_day_adjusted") or {}).get("diff", 0))):
        if r["verdict"] == "drop":
            continue
        raw = r.get("night_vs_day_raw") or {}
        adj = r.get("night_vs_day_adjusted") or {}
        if not adj:
            continue
        L.append(f"| {r['attribute']} | {raw.get('diff', float('nan')):+.4f} | "
                 f"{adj['diff']:+.4f} | {adj['se']:.4f} | {2.8 * adj['se']:.4f} | "
                 f"{r.get('q_night_vs_day_adjusted')} | {r['verdict']} |")

    L += ["", "## Joint hour test: raw vs composition-adjusted", "",
          "An attribute significant raw but not adjusted is tracking *what kind of "
          "post* people make at that hour, not a shift in tone.", "",
          "> **Treat these p-values as weak evidence.** This is a 23-parameter Wald "
          "test with a cluster-robust covariance matrix, and the sparsest hour holds "
          "only a few dozen posts. That combination is known to over-reject. Where "
          "this test and the pre-specified night-vs-day contrast disagree, believe "
          "the contrast.", "",
          "| attribute | q (raw) | q (adjusted) | survives |", "|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r["q_hour_effect_adjusted"] is None,
                                            r["q_hour_effect_adjusted"] or 1)):
        if r["verdict"] not in ("keep", "low_power"):
            continue
        qa = r["q_hour_effect_adjusted"]
        surv = "yes" if (qa is not None and qa < 0.05) else "no"
        L.append(f"| {r['attribute']} | {r['q_hour_effect']} | {qa} | {surv} |")

    L += ["", "## Peak and trough hours (selection-biased -- context only)", "",
          "The quietest hours carry only a few dozen posts, and peak/trough is the "
          "max and min over 24 such estimates, so this swing is an upper bound on "
          "any real diurnal variation. Shown for orientation, not for quoting.", "",
          "| attribute | peak h | peak | n | trough h | trough | n | swing | q | flag |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sig:
        L.append(f"| {r['attribute']} | {r['peak_hour_local']:02d} | {r['peak_mean']:.4f} | "
                 f"{r['peak_n']} | {r['trough_hour_local']:02d} | {r['trough_mean']:.4f} | "
                 f"{r['trough_n']} | {r['peak_minus_trough']:+.4f} | "
                 f"{r['q_hour_effect']} | {r['verdict']} |")
    if not sig:
        L.append("| _none_ | | | | | | | | | |")

    L += ["", "## Engagement by local hour", "",
          "| outcome | peak h | trough h | chi2 | p |", "|---|---|---|---|---|"]
    for e in eng:
        L.append(f"| {e['outcome']} | {e['peak_hour_local']} | {e['trough_hour_local']} | "
                 f"{e['wald_chi2']} | {e['p_hour_effect']:.3g} |")

    L += ["", "## Why this matters for the engagement analysis", "",
          "If both a score and engagement vary by hour, then posting time is a "
          "confounder of the score-engagement association: an attribute could appear "
          "to predict likes purely because it is more common at hours when people "
          "are awake. `analyze_de_engagement.py` conditions on hour (as sin/cos in "
          "the control block, or as 23 dummies with `--hour-dummies`), and the "
          "within-author model conditions on it again inside each author. Compare "
          "those runs to see how much of an association is timing rather than content."]
    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    n_adj = sum(1 for r in results if (r["q_hour_effect_adjusted"] or 1) < 0.05)
    print(f"{len(results)} attributes tested; {len(sig)} with a raw hour effect at "
          f"FDR<0.05, {n_adj} surviving composition adjustment; "
          f"{len(nd)} with a night-vs-day shift at FDR<0.05")
    print(f"-> {args.out}\n-> {args.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
