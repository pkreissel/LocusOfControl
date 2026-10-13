"""Time-series structure of fusion, outgroup hostility and LoC in the German sample.

analyze_fusion_time.py reports means and a linear trend. A linear trend is the
wrong summary if a series moves in steps or spikes, so this script asks:

  1. Composition. The frame is "declares de", and in 2023-07/08 most of it was
     not German first (multi-tag clients); posts also grow from ~75 to ~105
     characters. Series are restricted to first-language-German posts from
     2023-09 on, and also shown adjusted for length, reply and embed via a
     weighted regression with month fixed effects.
  2. Level shifts. Piecewise-constant fits with 0..3 breaks, chosen by BIC on
     the monthly means weighted by their sampling variance; compared with a
     linear trend.
  3. Spikes. Months whose residual from the chosen model exceeds |z| = 3.
  4. Seasonality. Calendar-month effects on the break-model residuals (three
     years at most, so only a recurring month is worth reading).
  5. Autocorrelation of the residuals (is month-to-month persistence left?).
  6. Weekly zoom on the spike months, to put dates on them.
  7. Fusion targets around the spikes: which group carries a fusion spike.

Run: /usr/bin/python3 research/time_sample/timeseries_patterns.py
"""

import json
import math
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data" / "time_sample"
REPORT = ROOT / "docs" / "reports"
START = "2023-09"

MEASURES = {
    "fusion": lambda r: r["fusion_score"],
    "fusion>=2.5": lambda r: float(r["fusion_score"] >= 2.5),
    "we": lambda r: r["fusion_we"],
    "outgroup": lambda r: r["outgroup"],
    "outgroup>=3.5": lambda r: float(r["outgroup"] >= 3.5),
    "loc_ext": lambda r: r["ts_class_probs"]["external"],
    "loc_int": lambda r: r["ts_class_probs"]["internal"],
}
PERSP = {"p_toxicity": "TOXICITY", "p_identity_attack": "IDENTITY_ATTACK",
         "p_insult": "INSULT", "p_threat": "THREAT"}


def load():
    sc = {}
    for line in open(DATA / "scores_typesafe.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "outgroup" in r:
            sc[r["uri"]] = r
    persp = {}
    pf = DATA / "perspective_de.jsonl"
    if pf.exists():
        for line in open(pf, encoding="utf-8"):
            r = json.loads(line)
            if r.get("scores"):
                persp[r["uri"]] = r["scores"]
    rows = []
    for line in open(DATA / "time_sample_de.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if p["uri"] not in sc or not p.get("weight") or p["month"] < START:
            continue
        if not (p["langs"] or [""])[0].lower().startswith("de"):
            continue
        r = {**p, **sc[p["uri"]]}
        if p["uri"] in persp:
            r["persp"] = persp[p["uri"]]
        rows.append(r)
    return rows


def monthly(rows, f):
    by = defaultdict(list)
    for r in rows:
        by[r["month"]].append(r)
    months = sorted(by)
    mean, se = [], []
    for m in months:
        rs = by[m]
        w = np.array([r["weight"] for r in rs])
        y = np.array([f(r) for r in rs], dtype=float)
        mu = (w * y).sum() / w.sum()
        mean.append(mu)
        se.append(math.sqrt(((w / w.sum()) ** 2 * (y - mu) ** 2).sum()))
    return months, np.array(mean), np.array(se)


def adjusted(rows, f, months):
    """Month effects from weighted OLS of the post-level measure on month
    dummies + log length (quadratic) + reply + embed, evaluated at the pooled
    covariate means. Shows what the series does with composition held fixed."""
    idx = {m: i for i, m in enumerate(months)}
    n, k = len(rows), len(months)
    ll = np.log1p(np.array([r["text_len"] for r in rows], dtype=float))
    X = np.zeros((n, k + 4))
    for i, r in enumerate(rows):
        X[i, idx[r["month"]]] = 1
    X[:, k] = ll - ll.mean()
    X[:, k + 1] = (ll - ll.mean()) ** 2
    X[:, k + 2] = [float(r["is_reply"]) for r in rows]
    X[:, k + 3] = [float(bool(r.get("embed_type"))) for r in rows]
    for j in (k + 2, k + 3):
        X[:, j] -= X[:, j].mean()
    y = np.array([f(r) for r in rows], dtype=float)
    w = np.array([r["weight"] for r in rows])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta[:k], beta[k:]


def fit_breaks(y, se, max_breaks=3, min_seg=3):
    """Best piecewise-constant fit for each number of breaks (WLS on monthly
    means with known variances), BIC = chi2 + p log n."""
    n = len(y)
    w = 1 / se ** 2

    def seg_cost(a, b):
        ww, yy = w[a:b], y[a:b]
        mu = (ww * yy).sum() / ww.sum()
        return (ww * (yy - mu) ** 2).sum(), mu

    best = {}
    for nb in range(max_breaks + 1):
        top = None
        for cuts in combinations(range(min_seg, n - min_seg + 1), nb):
            bounds = (0, *cuts, n)
            if any(b - a < min_seg for a, b in zip(bounds, bounds[1:])):
                continue
            chi2 = sum(seg_cost(a, b)[0] for a, b in zip(bounds, bounds[1:]))
            if top is None or chi2 < top[0]:
                top = (chi2, cuts)
        chi2, cuts = top
        bounds = (0, *cuts, n)
        fitted = np.concatenate([[seg_cost(a, b)[1]] * (b - a) for a, b in zip(bounds, bounds[1:])])
        p = 2 * nb + 1  # each break: a location and a level
        best[nb] = {"chi2": chi2, "cuts": cuts, "fitted": fitted, "bic": chi2 + p * math.log(n)}
    # Linear trend for comparison.
    x = np.arange(n)
    W = np.diag(w)
    A = np.vstack([np.ones(n), x]).T
    coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    lin = A @ coef
    lin_chi2 = (w * (y - lin) ** 2).sum()
    return best, {"chi2": lin_chi2, "bic": lin_chi2 + 2 * math.log(n), "fitted": lin,
                  "slope_per_year": coef[1] * 12}


def weekly(rows, f, months):
    by = defaultdict(list)
    for r in rows:
        if r["month"] in months:
            d = datetime.fromisoformat(r["created_at"][:10])  # some carry 7-digit fractions
            y, wk, _ = d.isocalendar()
            by[f"{y}-W{wk:02d}"].append(r)
    out = []
    for k in sorted(by):
        rs = by[k]
        w = np.array([r["weight"] for r in rs])
        v = np.array([f(r) for r in rs], dtype=float)
        mu = (w * v).sum() / w.sum()
        se = math.sqrt(((w / w.sum()) ** 2 * (v - mu) ** 2).sum())
        out.append((k, len(rs), mu, se))
    return out


def main():
    rows = load()
    has_p = sum("persp" in r for r in rows)
    measures = dict(MEASURES)
    if has_p == len(rows):
        for k, a in PERSP.items():
            measures[k] = (lambda a: lambda r: r["persp"][a])(a)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"{len(rows)} first-language-German posts, {START} on "
        f"({has_p} with Perspective{'' if has_p == len(rows) else ' -- Perspective skipped'})\n")

    results = {}
    for name, f in measures.items():
        months, y, se = monthly(rows, f)
        adj, cov = adjusted(rows, f, months)
        best, lin = fit_breaks(y, se)
        nb = min(best, key=lambda k: best[k]["bic"])
        chosen = best[nb]
        resid = (y - chosen["fitted"]) / se
        r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
        # Same break search on the composition-adjusted series, same SEs.
        abest, alin = fit_breaks(adj, se)
        anb = min(abest, key=lambda k: abest[k]["bic"])
        results[name] = dict(months=months, y=y, se=se, adj=adj, best=best, nb=nb,
                             lin=lin, resid=resid, abest=abest, anb=anb)

        say(f"## {name}")
        say(f"- mean {np.average(y, weights=1/se**2):.4f}; typical monthly SE {np.median(se):.4f}")
        say(f"- linear: slope {lin['slope_per_year']:+.4f}/yr, BIC {lin['bic']:.1f}; "
            f"flat: BIC {best[0]['bic']:.1f}")
        segs = []
        bounds = (0, *chosen["cuts"], len(y))
        for a, b in zip(bounds, bounds[1:]):
            segs.append(f"{months[a]}..{months[b-1]} = {chosen['fitted'][a]:.4f}")
        say(f"- best step model: {nb} break(s), BIC {chosen['bic']:.1f}: " + "; ".join(segs))
        abounds = (0, *abest[anb]["cuts"], len(y))
        say(f"- composition-adjusted: {anb} break(s): " + "; ".join(
            f"{months[a]}..{months[b-1]} = {abest[anb]['fitted'][a]:.4f}"
            for a, b in zip(abounds, abounds[1:])))
        say(f"- covariates (log len, log len^2, reply, embed): "
            + ", ".join(f"{c:+.4f}" for c in cov))
        # The monthly SEs miss some real month-to-month variance (residual
        # chi2/df ~1.5), so each spike also gets z deflated by that overdispersion.
        od = math.sqrt(max(1.0, float((resid ** 2).sum()) / (len(y) - 2 * nb - 1)))
        spikes = [(months[i], y[i], resid[i]) for i in range(len(y)) if abs(resid[i]) >= 3]
        say("- spikes |z|>=3 vs step model: " + (", ".join(
            f"{m} ({v:.4f}, z={z:+.1f}, overdispersion-adjusted {z/od:+.1f})"
            for m, v, z in spikes) or "none"))
        say(f"- residual lag-1 autocorrelation {r1:+.2f}; "
            f"residual chi2/df {float((resid**2).sum())/(len(y)-2*nb-1):.2f}")
        cal = defaultdict(list)
        for m, z in zip(months, resid):
            cal[m[5:]].append(z)
        hot = sorted(((np.mean(v), k, len(v)) for k, v in cal.items() if len(v) >= 3), reverse=True)
        say("- calendar months, mean residual z (>=3 years): " + ", ".join(
            f"{k}:{v:+.1f}" for v, k, n in hot[:3]) + " ... " + ", ".join(
            f"{k}:{v:+.1f}" for v, k, n in hot[-2:]))
        say()

    # Weekly zoom on the outgroup / fusion spike months.
    for name in ("outgroup", "fusion"):
        r = results[name]
        spike_months = [m for m, z in zip(r["months"], r["resid"]) if abs(z) >= 2.5]
        if not spike_months:
            continue
        zoom = set()
        for m in spike_months:
            i = r["months"].index(m)
            zoom |= set(r["months"][max(0, i - 1): i + 2])
        say(f"## Weekly {name} around {', '.join(spike_months)}\n")
        say("| ISO week | n | mean | SE |")
        say("|---|---|---|---|")
        for k, n, mu, s in weekly(rows, measures[name], zoom):
            say(f"| {k} | {n} | {mu:.3f} | {s:.3f} |")
        say()

    # Fusion targets in the fusion spike months vs the rest.
    r = results["fusion"]
    spike_months = {m for m, z in zip(r["months"], r["resid"]) if z >= 2.5}
    if spike_months:
        say(f"## Fusion targets (fusion >= 1.5): spike months {sorted(spike_months)} vs rest\n")
        def share(rs):
            tot = sum(x["weight"] for x in rs)
            c = defaultdict(float)
            for x in rs:
                c[x["fusion_target"]] += x["weight"]
            return {k: v / tot for k, v in c.items()}, len(rs)
        hi = [x for x in rows if x["fusion_score"] >= 1.5 and x["month"] in spike_months]
        lo = [x for x in rows if x["fusion_score"] >= 1.5 and x["month"] not in spike_months]
        sh, nh = share(hi)
        sl, nl = share(lo)
        say(f"| target | spike (n={nh}) | rest (n={nl}) |")
        say("|---|---|---|")
        for t in sorted(set(sh) | set(sl), key=lambda t: -sh.get(t, 0)):
            say(f"| {t} | {100*sh.get(t,0):.1f}% | {100*sl.get(t,0):.1f}% |")
        say()

    # Monthly cross-correlation of the de-meaned series.
    say("## Monthly correlations between series (raw monthly means)\n")
    names = list(results)
    say("| | " + " | ".join(names) + " |")
    say("|---|" + "---|" * len(names))
    for a in names:
        say(f"| {a} | " + " | ".join(
            f"{np.corrcoef(results[a]['y'], results[b]['y'])[0,1]:+.2f}" for b in names) + " |")
    say()

    (DATA / "timeseries_patterns.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure: raw, adjusted, step model, spikes.
    show = [n for n in ("fusion", "fusion>=2.5", "outgroup", "outgroup>=3.5", "loc_ext",
                        "loc_int", "p_toxicity", "p_identity_attack") if n in results]
    rows_n = math.ceil(len(show) / 2)
    fig, axes = plt.subplots(rows_n, 2, figsize=(12, 2.9 * rows_n), sharex=True)
    for ax, name in zip(axes.flat, show):
        r = results[name]
        x = np.arange(len(r["months"]))
        ax.fill_between(x, r["y"] - 1.96 * r["se"], r["y"] + 1.96 * r["se"],
                        color="#2b6cb0", alpha=0.15, lw=0)
        ax.plot(x, r["y"], color="#2b6cb0", lw=1.5, label="raw")
        ax.plot(x, r["adj"], color="#718096", lw=1, ls="--", label="length/reply/embed-adjusted")
        ax.step(x, r["best"][r["nb"]]["fitted"], where="mid", color="#c05621", lw=1.4,
                label=f"step model ({r['nb']} breaks)")
        for i, z in enumerate(r["resid"]):
            if abs(z) >= 3:
                ax.plot(i, r["y"][i], "o", color="#c53030", ms=5)
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=7)
    months = results["fusion"]["months"]
    ticks = [i for i, m in enumerate(months) if m.endswith(("-01", "-07"))]
    for ax in axes[-1]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([months[i] for i in ticks], rotation=45, fontsize=8)
    for ax in list(axes.flat)[len(show):]:
        ax.axis("off")
    fig.suptitle("German Bluesky time sample: monthly means (95% CI), step model, |z|>=3 spikes",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORT / "fusion_timeseries_de.png", dpi=130)
    print(f"-> {DATA / 'timeseries_patterns.md'}, {REPORT / 'fusion_timeseries_de.png'}")


if __name__ == "__main__":
    main()
