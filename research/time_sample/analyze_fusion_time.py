"""Identity fusion, outgroup hostility and LoC over time in the German time sample.

Joins scores_typesafe.jsonl (score_time_sample.py) onto time_sample_de.jsonl
(collect_time_sample.py) and reports weighted monthly estimates with bootstrap
CIs. Every estimate uses the draw weight, which corrects the length bias of the
second-based draw (see collect_time_sample.py); months are equal-n strata, so
nothing is pooled across months without saying so.

Measures (all jev-latest, rubric fusion_questions.VERSION):
  fusion        0-9 fusion score (probability-weighted level)
  fusion>=2.5   share at 'shared activity' or above, i.e. past a passing 'we'
                and a bare membership label -- the part level-1 noise
                (greeting 'we', sarcastic 'uns') cannot move
  outgroup      0-8 hostility score; >=2.5 blame or worse, >=3.5 insult or worse
  joint         fusion>=2.5 AND outgroup>=2.5, the risk-literature combination
  loc_int/ext   LoC P(internal), P(external) from the 3-way choice

Perspective (research/perspective/score_perspective_de.py --no-bridging, minus
INCOHERENT and LIKELY_TO_REJECT, which score "not English" on German) joins as
p_<attribute>; it is both a trend in its own right and an outside check on the
outgroup rubric, which should track IDENTITY_ATTACK and INSULT.

Robustness: first declared language German only; replies vs top-level posts
(the reply share moves over time); dropping low-confidence fusion answers.

Run: /usr/bin/python3 research/time_sample/analyze_fusion_time.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data" / "time_sample"
REPORT = ROOT / "docs" / "reports"
B = 400  # bootstrap reps
rng = random.Random(11)

MEASURES = {
    "fusion": lambda r: r["fusion_score"],
    "fusion>=1.5": lambda r: float(r["fusion_score"] >= 1.5),
    "fusion>=2.5": lambda r: float(r["fusion_score"] >= 2.5),
    "fusion>=3.5": lambda r: float(r["fusion_score"] >= 3.5),
    "we": lambda r: r["fusion_we"],
    "outgroup": lambda r: r["outgroup"],
    "outgroup>=2.5": lambda r: float(r["outgroup"] >= 2.5),
    "outgroup>=3.5": lambda r: float(r["outgroup"] >= 3.5),
    "outgroup>=5.5": lambda r: float(r["outgroup"] >= 5.5),
    "joint": lambda r: float(r["fusion_score"] >= 2.5 and r["outgroup"] >= 2.5),
    "loc_int": lambda r: r["ts_class_probs"]["internal"],
    "loc_ext": lambda r: r["ts_class_probs"]["external"],
}
PERSPECTIVE = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY",
               "THREAT", "INFLAMMATORY", "ATTACK_ON_AUTHOR", "ATTACK_ON_COMMENTER",
               "UNSUBSTANTIAL", "SPAM", "FLIRTATION", "SEXUALLY_EXPLICIT", "OBSCENE"]
for _a in PERSPECTIVE:
    MEASURES[f"p_{_a.lower()}"] = (lambda a: lambda r: r["persp"][a])(_a)


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
            if r.get("scores") and all(r["scores"].get(a) is not None for a in PERSPECTIVE):
                persp[r["uri"]] = r["scores"]
    rows, no_persp = [], 0
    for line in open(DATA / "time_sample_de.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if p["uri"] in sc and p.get("weight"):
            if p["uri"] not in persp:
                no_persp += 1
                continue
            rows.append({**p, **sc[p["uri"]], "persp": persp[p["uri"]]})
    print(f"{no_persp} jev-scored posts dropped for lacking a Perspective score")
    return rows


def wmean(rs, f):
    w = sum(r["weight"] for r in rs)
    return sum(r["weight"] * f(r) for r in rs) / w if w else float("nan")


def boot_ci(rs, f):
    est = []
    for _ in range(B):
        s = [rs[rng.randrange(len(rs))] for _ in rs]
        est.append(wmean(s, f))
    est.sort()
    return est[int(0.025 * B)], est[int(0.975 * B)]


def monthly(rows, f, ci=True):
    by = defaultdict(list)
    for r in rows:
        by[r["month"]].append(r)
    out = {}
    for m in sorted(by):
        rs = by[m]
        out[m] = (wmean(rs, f), *(boot_ci(rs, f) if ci else (None, None)), len(rs))
    return out


def trend(rows, f):
    """Weighted OLS slope per year of the post-level measure on time, with a
    bootstrap CI that resamples posts within each month (the strata)."""
    by = defaultdict(list)
    for r in rows:
        by[r["month"]].append(r)
    months = sorted(by)
    idx = {m: i / 12 for i, m in enumerate(months)}

    def slope(rs):
        w = [r["weight"] for r in rs]
        x = [idx[r["month"]] for r in rs]
        y = [f(r) for r in rs]
        sw = sum(w)
        mx = sum(a * b for a, b in zip(w, x)) / sw
        my = sum(a * b for a, b in zip(w, y)) / sw
        sxx = sum(a * (b - mx) ** 2 for a, b in zip(w, x))
        sxy = sum(a * (b - mx) * (c - my) for a, b, c in zip(w, x, y))
        return sxy / sxx

    est = slope(rows)
    bs = []
    for _ in range(200):
        s = []
        for m in months:
            rs = by[m]
            s.extend(rs[rng.randrange(len(rs))] for _ in rs)
        bs.append(slope(s))
    bs.sort()
    return est, bs[5], bs[194]


def period(m):
    y, mo = m.split("-")
    return f"{y}H{1 if int(mo) <= 6 else 2}"


def main():
    rows = load()
    print(f"{len(rows)} scored German-sample posts with a weight, "
          f"{len({r['month'] for r in rows})} months")
    first_de = [r for r in rows if (r["langs"] or [""])[0].lower().startswith("de")]
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- level distributions
    say("## Level distributions (weighted share)\n")
    for name, key in (("fusion", "fusion_score"), ("outgroup", "outgroup")):
        c = Counter()
        for r in rows:
            c[round(r[key])] += r["weight"]
        tot = sum(c.values())
        say(f"{name}: " + "  ".join(f"{k}:{100*v/tot:.1f}%" for k, v in sorted(c.items())))
    say()

    # ---- trends
    say("## Linear trend per year (weighted OLS, 90% bootstrap CI) and first/last 6 months\n")
    say("| measure | overall mean | slope / year | 90% CI | first 6 mo | last 6 mo |")
    say("|---|---|---|---|---|---|")
    months = sorted({r["month"] for r in rows})
    head = [r for r in rows if r["month"] in months[:6]]
    tail = [r for r in rows if r["month"] in months[-6:]]
    trends = {}
    for name, f in MEASURES.items():
        s, lo, hi = trend(rows, f)
        trends[name] = (s, lo, hi)
        say(f"| {name} | {wmean(rows, f):.4f} | {s:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{wmean(head, f):.4f} | {wmean(tail, f):.4f} |")
    say()

    # ---- robustness
    say("## Robustness: slope / year under restrictions\n")
    say("| measure | all | first lang de | top-level only | replies only | fusion conf >= 0.5 |")
    say("|---|---|---|---|---|---|")
    subsets = {
        "first lang de": first_de,
        "top-level": [r for r in rows if not r["is_reply"]],
        "replies": [r for r in rows if r["is_reply"]],
        "conf": [r for r in rows if r["fusion_score_conf"] >= 0.5],
    }
    for name in ("fusion", "fusion>=2.5", "outgroup", "outgroup>=3.5", "joint", "loc_int",
                 "p_toxicity", "p_identity_attack"):
        f = MEASURES[name]
        cells = [f"{trends[name][0]:+.4f}"]
        for rs in subsets.values():
            s, lo, hi = trend(rs, f)
            cells.append(f"{s:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        say(f"| {name} | " + " | ".join(cells) + " |")
    say()
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["month"]].append(r)
    say("Composition drift: reply share " + ", ".join(
        f"{m}: {wmean(by_m[m], lambda r: float(r['is_reply'])):.2f}"
        for m in (months[0], months[len(months) // 2], months[-1])) +
        "; first-lang-de share " + ", ".join(
        f"{m}: {wmean(by_m[m], lambda r: float((r['langs'] or [''])[0].lower().startswith('de'))):.2f}"
        for m in (months[0], months[len(months) // 2], months[-1])))
    say()

    # ---- fusion targets by half-year
    say("## Fusion target among posts with fusion >= 1.5, by half-year (weighted %)\n")
    targets = ["family_friends", "local_online", "fandom_sports", "political", "nation_ethnic",
               "identity_group", "work_profession", "religion", "leader", "none"]
    halves = sorted({period(m) for m in months})
    say("| half | n | " + " | ".join(targets) + " |")
    say("|---|---|" + "---|" * len(targets))
    tshare = {}
    for h in halves:
        rs = [r for r in rows if period(r["month"]) == h and r["fusion_score"] >= 1.5]
        tot = sum(r["weight"] for r in rs)
        sh = {t: sum(r["weight"] for r in rs if r["fusion_target"] == t) / tot for t in targets}
        tshare[h] = sh
        say(f"| {h} | {len(rs)} | " + " | ".join(f"{100*sh[t]:.0f}" for t in targets) + " |")
    say()

    # ---- correlations
    def corr(a, b):
        w = [r["weight"] for r in rows]
        x = [a(r) for r in rows]
        y = [b(r) for r in rows]
        sw = sum(w)
        mx = sum(p * q for p, q in zip(w, x)) / sw
        my = sum(p * q for p, q in zip(w, y)) / sw
        cxy = sum(p * (q - mx) * (s - my) for p, q, s in zip(w, x, y))
        cxx = sum(p * (q - mx) ** 2 for p, q in zip(w, x))
        cyy = sum(p * (s - my) ** 2 for p, s in zip(w, y))
        return cxy / (cxx * cyy) ** 0.5
    say("## Post-level correlations (weighted Pearson)\n")
    for a, b in (("fusion", "outgroup"), ("fusion", "we"), ("fusion", "loc_int"),
                 ("outgroup", "loc_ext"), ("outgroup", "loc_int"),
                 ("outgroup", "p_identity_attack"), ("outgroup", "p_insult"),
                 ("outgroup", "p_toxicity"), ("outgroup", "p_inflammatory"),
                 ("fusion", "p_identity_attack"), ("fusion", "p_toxicity"),
                 ("joint", "p_threat")):
        say(f"- r({a}, {b}) = {corr(MEASURES[a], MEASURES[b]):+.3f}")
    say()

    (DATA / "fusion_time_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- figure
    panels = [
        ("Identity fusion (0-9), mean", "fusion"),
        ("Fusion >= 2.5 (past bare 'we' / label), share", "fusion>=2.5"),
        ("Outgroup hostility (0-8), mean", "outgroup"),
        ("Outgroup >= 3.5 (insult or worse), share", "outgroup>=3.5"),
        ("Joint: fusion >= 2.5 and outgroup >= 2.5, share", "joint"),
        ("LoC: P(internal) and P(external)", "loc"),
        ("Perspective TOXICITY, mean", "p_toxicity"),
        ("Perspective IDENTITY_ATTACK, mean", "p_identity_attack"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    xs = list(range(len(months)))
    for ax, (title, key) in zip(axes.flat, panels):
        keys = ["loc_int", "loc_ext"] if key == "loc" else [key]
        for k, col in zip(keys, ["#2b6cb0", "#c05621"]):
            mo = monthly(rows, MEASURES[k])
            y = [mo[m][0] for m in months]
            lo = [mo[m][1] for m in months]
            hi = [mo[m][2] for m in months]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(xs, y, color=col, lw=1.6, label=k if key == "loc" else None)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        if key == "loc":
            ax.legend(frameon=False, fontsize=8)
    ticks = [i for i, m in enumerate(months) if m.endswith(("-01", "-07"))]
    for ax in axes[-1]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([months[i] for i in ticks], rotation=45, fontsize=8)
    fig.suptitle("German Bluesky posts, time sample (weighted monthly means, 95% bootstrap CI)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORT / "fusion_time_de.png", dpi=130)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = [0.0] * len(halves)
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(targets[:-1]):
        v = [tshare[h][t] for h in halves]
        ax.bar(halves, v, bottom=bottom, color=cmap(i), label=t)
        bottom = [a + b for a, b in zip(bottom, v)]
    ax.set_title("Fusion target among posts with fusion >= 1.5 (weighted share)", loc="left",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORT / "fusion_targets_de.png", dpi=130)
    print(f"-> {DATA / 'fusion_time_summary.md'}, {REPORT / 'fusion_time_de.png'}, "
          f"{REPORT / 'fusion_targets_de.png'}")


if __name__ == "__main__":
    main()
