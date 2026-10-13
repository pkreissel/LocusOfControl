"""What gets engaged with vs what gets posted: like-weighted series, German sample.

The post-weighted mean (draw weight only) estimates what a random post looks like.
Weighting each post additionally by its likes estimates what a random *like* lands
on -- the share of the month's likes going to fused, hostile, external... content.

  likes     draw weight x like_count. The interpretable estimand, but one viral post
            can carry a month (Kish n_eff 18-111 per month vs ~570 post-weighted),
            so it is reported by quarter.
  log1p     draw weight x log(1 + likes). A milder tilt toward engagement, n_eff ~300
            per month; not a share of anything, only a robustness check.
  premium   likes-weighted / post-weighted mean: > 1 means the content gets more
            than its share of likes. Paired bootstrap, so the CI is of the ratio.

Likes are mostly reach -- follower count -- so a premium can come from hostile
*authors* having bigger audiences rather than hostile *posts* being rewarded. The
within-author regression (log1p likes on the measure, author and month fixed
effects, authors with >= 2 posts) separates the two.

like_count is a snapshot at collection (2026-10-09); posts from the last days of
2026-09 are slightly under-counted (4.8 vs 5.7 mean likes).

Run: /usr/bin/python3 research/time_sample/like_weighted.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data" / "time_sample"
REPORT = ROOT / "docs" / "reports"
START = "2023-09"
B = 1000
rng = np.random.default_rng(17)

MEASURES = {
    "fusion": lambda r: r["fusion_score"],
    "fusion>=2.5": lambda r: float(r["fusion_score"] >= 2.5),
    "we": lambda r: r["fusion_we"],
    "outgroup": lambda r: r["outgroup"],
    "outgroup>=3.5": lambda r: float(r["outgroup"] >= 3.5),
    "loc_ext": lambda r: r["ts_class_probs"]["external"],
    "loc_int": lambda r: r["ts_class_probs"]["internal"],
    "p_toxicity": lambda r: r["persp"]["TOXICITY"],
    "p_insult": lambda r: r["persp"]["INSULT"],
    "p_identity_attack": lambda r: r["persp"]["IDENTITY_ATTACK"],
}


def load():
    sc = {}
    for line in open(DATA / "scores_typesafe.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "outgroup" in r:
            sc[r["uri"]] = r
    pp = {}
    for line in open(DATA / "perspective_de.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("scores"):
            pp[r["uri"]] = r["scores"]
    rows = []
    for line in open(DATA / "time_sample_de.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if (p["uri"] in sc and p["uri"] in pp and p.get("weight") and p["month"] >= START
                and (p["langs"] or [""])[0].lower().startswith("de")):
            rows.append({**p, **sc[p["uri"]], "persp": pp[p["uri"]],
                         "likes": p.get("like_count") or 0})
    return rows


def quarter(m):
    y, mo = m.split("-")
    return f"{y}Q{(int(mo) - 1) // 3 + 1}"


def estimates(rs, f):
    """Post-, log1p- and like-weighted means and the likes/post premium, each
    with a bootstrap 95% CI from one shared set of resamples."""
    w = np.array([r["weight"] for r in rs])
    lk = np.array([r["likes"] for r in rs], dtype=float)
    y = np.array([f(r) for r in rs], dtype=float)
    wl, wg = w * lk, w * np.log1p(lk)

    def stat(idx):
        ww, wwl, wwg, yy = w[idx], wl[idx], wg[idx], y[idx]
        post = (ww * yy).sum(-1) / ww.sum(-1)
        like = (wwl * yy).sum(-1) / wwl.sum(-1)
        logw = (wwg * yy).sum(-1) / wwg.sum(-1)
        return post, logw, like, like / post

    point = stat(np.arange(len(y)))
    boot = stat(rng.integers(0, len(y), size=(B, len(y))))
    out = {}
    for k, p, b in zip(("post", "log1p", "likes", "premium"), point, boot):
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        out[k] = (float(p), float(lo), float(hi))
    out["n_eff_likes"] = float(wl.sum() ** 2 / (wl ** 2).sum())
    return out


def within_author(rows, f):
    """OLS of log1p(likes) on the standardised measure after demeaning both by
    author and by month (two-way FE by alternating projections), authors with
    >= 2 posts. Cluster-robust SE by author."""
    by = defaultdict(list)
    for r in rows:
        by[r["did"]].append(r)
    rs = [r for d, v in by.items() if len(v) >= 2 for r in v]
    a_idx = {d: i for i, d in enumerate(sorted({r["did"] for r in rs}))}
    m_idx = {m: i for i, m in enumerate(sorted({r["month"] for r in rs}))}
    A = np.array([a_idx[r["did"]] for r in rs])
    M = np.array([m_idx[r["month"]] for r in rs])
    y = np.log1p(np.array([r["likes"] for r in rs], dtype=float))
    x = np.array([f(r) for r in rs], dtype=float)
    x = (x - x.mean()) / x.std()

    def demean(v):
        v = v.copy()
        for _ in range(30):
            v -= (np.bincount(A, v) / np.bincount(A))[A]
            v -= (np.bincount(M, v) / np.bincount(M))[M]
        return v

    yd, xd = demean(y), demean(x)
    beta = (xd * yd).sum() / (xd * xd).sum()
    u = yd - beta * xd
    g = np.bincount(A, xd * u)
    se = np.sqrt((g ** 2).sum()) / (xd * xd).sum()
    # Same, controlling for reply, log length (quadratic) and embed: replies and
    # one-word posts score 0 on every measure and get few likes, which alone
    # makes any content measure look rewarded.
    ll = np.log1p(np.array([r["text_len"] for r in rs], dtype=float))
    C = np.column_stack([[float(r["is_reply"]) for r in rs], ll, ll ** 2,
                         [float(bool(r.get("embed_type"))) for r in rs]])
    Cd = np.column_stack([demean(c) for c in C.T])
    Z = np.column_stack([xd, Cd])
    coef, *_ = np.linalg.lstsq(Z, yd, rcond=None)
    u2 = yd - Z @ coef
    bread = np.linalg.inv(Z.T @ Z)
    G = np.zeros((len(a_idx), Z.shape[1]))
    np.add.at(G, A, Z * u2[:, None])
    se_c = float(np.sqrt((bread @ (G.T @ G) @ bread)[0, 0]))
    beta_c = float(coef[0])
    # Pooled (between + within) slope for contrast, same standardised x, month FE only.
    yp = y - (np.bincount(M, y) / np.bincount(M))[M]
    xp = x - (np.bincount(M, x) / np.bincount(M))[M]
    pooled = (xp * yp).sum() / (xp * xp).sum()
    return beta, se, pooled, beta_c, se_c, len(rs), len(a_idx)


def main():
    rows = load()
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"{len(rows)} first-language-German posts {START}..; "
        f"{sum(r['likes'] for r in rows):,} likes\n")

    say("## Pooled over the whole period\n")
    say("| measure | post-weighted | like-weighted | premium (likes / post) [95% CI] |")
    say("|---|---|---|---|")
    for name, f in MEASURES.items():
        e = estimates(rows, f)
        star = " *" if e["premium"][1] > 1 or e["premium"][2] < 1 else ""
        say(f"| {name} | {e['post'][0]:.4f} | {e['likes'][0]:.4f} | "
            f"{e['premium'][0]:.2f} [{e['premium'][1]:.2f}, {e['premium'][2]:.2f}]{star} |")
    say()

    say("## Is it the post or the author? log1p(likes) on the standardised measure\n")
    say("Pooled = month FE only (between + within authors); within = author and month FE, "
        "authors with >= 2 posts, author-clustered SE. Units: log-likes per SD.\n")
    say("| measure | pooled | within-author [95% CI] | within-author + reply/length/embed [95% CI] | posts / authors |")
    say("|---|---|---|---|---|")
    for name, f in MEASURES.items():
        b, se, pooled, bc, sec, n, na = within_author(rows, f)
        star = " *" if abs(b) > 1.96 * se else ""
        starc = " *" if abs(bc) > 1.96 * sec else ""
        say(f"| {name} | {pooled:+.3f} | {b:+.3f} [{b-1.96*se:+.3f}, {b+1.96*se:+.3f}]{star} | "
            f"{bc:+.3f} [{bc-1.96*sec:+.3f}, {bc+1.96*sec:+.3f}]{starc} | {n} / {na} |")
    say()

    say("## Premium by period (pooled, paired bootstrap): did the reward change?\n")
    periods = {"2023-09..2024-12": lambda m: m < "2025-01",
               "2025-01..2025-12": lambda m: "2025-01" <= m < "2026-01",
               "2026-01..2026-09": lambda m: m >= "2026-01"}
    say("| measure | " + " | ".join(periods) + " | late minus early [95% CI] |")
    say("|---|" + "---|" * (len(periods) + 1))
    for name in ("fusion", "fusion>=2.5", "outgroup", "outgroup>=3.5", "loc_ext", "loc_int",
                 "p_toxicity", "p_identity_attack"):
        f = MEASURES[name]
        cells, boots = [], []
        for sel in periods.values():
            rs = [r for r in rows if sel(r["month"])]
            w = np.array([r["weight"] for r in rs])
            wl = w * np.array([r["likes"] for r in rs], dtype=float)
            y = np.array([f(r) for r in rs], dtype=float)
            idx = rng.integers(0, len(y), size=(B, len(y)))
            prem = ((wl[idx] * y[idx]).sum(1) / wl[idx].sum(1)) / ((w[idx] * y[idx]).sum(1) / w[idx].sum(1))
            point = ((wl * y).sum() / wl.sum()) / ((w * y).sum() / w.sum())
            lo, hi = np.percentile(prem, [2.5, 97.5])
            cells.append(f"{point:.2f} [{lo:.2f}, {hi:.2f}]")
            boots.append((point, prem))
        d = boots[-1][0] - boots[0][0]
        dlo, dhi = np.percentile(boots[-1][1] - boots[0][1], [2.5, 97.5])
        star = " *" if dlo > 0 or dhi < 0 else ""
        say(f"| {name} | " + " | ".join(cells) + f" | {d:+.2f} [{dlo:+.2f}, {dhi:+.2f}]{star} |")
    say()

    by_q = defaultdict(list)
    for r in rows:
        by_q[quarter(r["month"])].append(r)
    quarters = sorted(by_q)
    series = {}
    say("## By quarter: post-weighted vs like-weighted, premium [95% CI]\n")
    for name in ("fusion", "fusion>=2.5", "outgroup", "outgroup>=3.5", "loc_ext", "p_toxicity"):
        f = MEASURES[name]
        say(f"### {name}\n")
        say("| quarter | n | n_eff (likes) | post | log1p | likes | premium |")
        say("|---|---|---|---|---|---|---|")
        series[name] = []
        for q in quarters:
            e = estimates(by_q[q], f)
            series[name].append(e)
            say(f"| {q} | {len(by_q[q])} | {e['n_eff_likes']:.0f} | {e['post'][0]:.4f} | "
                f"{e['log1p'][0]:.4f} | {e['likes'][0]:.4f} [{e['likes'][1]:.4f}, {e['likes'][2]:.4f}] | "
                f"{e['premium'][0]:.2f} [{e['premium'][1]:.2f}, {e['premium'][2]:.2f}] |")
        say()

    (DATA / "like_weighted.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    x = np.arange(len(quarters))
    for ax, name in zip(axes.flat, series):
        es = series[name]
        for key, col, lab in (("post", "#2b6cb0", "per post"), ("likes", "#c05621", "per like"),
                              ("log1p", "#718096", "log(1+likes)")):
            y = [e[key][0] for e in es]
            if key != "log1p":
                ax.fill_between(x, [e[key][1] for e in es], [e[key][2] for e in es],
                                color=col, alpha=0.15, lw=0)
            ax.plot(x, y, color=col, lw=1.6 if key != "log1p" else 1,
                    ls="-" if key != "log1p" else "--", label=lab)
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(quarters, rotation=45, fontsize=8)
    fig.suptitle("German Bluesky time sample: what is posted vs what is liked (quarterly, 95% CI)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORT / "fusion_like_weighted_de.png", dpi=130)
    print(f"-> {DATA / 'like_weighted.md'}, {REPORT / 'fusion_like_weighted_de.png'}")


if __name__ == "__main__":
    main()
