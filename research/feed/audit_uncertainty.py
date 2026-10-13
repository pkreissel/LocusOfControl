"""Feed comparison that keeps the decider's uncertainty instead of taking argmax.

Each post gets a tier from its argmax class and that class's probability:
  confident  P >= 0.7   (on the gold set, 90-94% of these agree with the label)
  leaning    P <  0.7   (agreement falls to ~55-70%; below 0.5 it is a coin flip)
plus 'neutral' for argmax neutral. Reports, per feed: the confident share, the
confident+leaning share (the range a feed could land in), and the mean
probability, with Wilson intervals on the confident shares.

Reads the caches audit_feeds.py wrote. Writes docs/reports/feed_audit_uncertainty.{md,png}.
Usage: audit_uncertainty.py [feed ...]   (feeds to chart; default the five below)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "research", "data", "feed", "audit")
CONF = 0.7
CHART = sys.argv[1:] or ["feed-forward", "mysky-your-feed", "discover", "for-you", "eurosky-fu"]
feeds = json.load(open(f"{D}/feeds.json"))
sc = {}
for l in open(f"{D}/scores_decider_ft.jsonl"):
    r = json.loads(l)
    if "error" not in r:
        sc[r["uri"]] = r

def posts(name):
    return sorted((json.loads(l) for l in open(f"{D}/posts_{name}.jsonl")), key=lambda p: p["draw_rank"])

def tier(s):
    c = s["ts_class"]
    if c == "neutral":
        return "neutral"
    return f"{c}_{'conf' if s['ts_class_probs'][c] >= CONF else 'lean'}"

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def stats(name, k=None):
    ps = [p for p in (posts(name)[:k] if k else posts(name)) if p["uri"] in sc]
    t = [tier(sc[p["uri"]]) for p in ps]
    n = len(t)
    return n, {x: t.count(x) / n for x in ("internal_conf", "internal_lean", "neutral", "external_lean", "external_conf")}, t

rows = []
for name in feeds:
    n, sh, t = stats(name)
    if not n:
        continue
    pe = np.mean([sc[p["uri"]]["ts_class_probs"]["external"] for p in posts(name) if p["uri"] in sc])
    pi = np.mean([sc[p["uri"]]["ts_class_probs"]["internal"] for p in posts(name) if p["uri"] in sc])
    ec, ic = t.count("external_conf"), t.count("internal_conf")
    rows.append((name, n, sh, wilson(ec, n), wilson(ic, n), pe, pi))
rows.sort(key=lambda r: r[2]["external_conf"])

md = ["| feed | n | confident external | + leaning = up to | confident internal | + leaning = up to | mean P(ext) | mean P(int) |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
for name, n, sh, eci, ici, pe, pi in rows:
    md.append(f"| {name} | {n} | {sh['external_conf']:.0%} [{eci[0]:.0%}–{eci[1]:.0%}] "
              f"| {sh['external_conf'] + sh['external_lean']:.0%} "
              f"| {sh['internal_conf']:.0%} [{ici[0]:.0%}–{ici[1]:.0%}] "
              f"| {sh['internal_conf'] + sh['internal_lean']:.0%} | {pe:.0%} | {pi:.0%} |")
table = "\n".join(md)
print(table)

COL = {"external_conf": "#b5530f", "external_lean": "#f0b27a", "neutral": "#d4d6da",
       "internal_lean": "#8fb8e6", "internal_conf": "#1f5fa6"}
LAB = {"external_conf": "external, confident (P≥0.7)", "external_lean": "external, leaning",
       "neutral": "neutral", "internal_lean": "internal, leaning", "internal_conf": "internal, confident (P≥0.7)"}
DEPTHS = [("Top 10", 10), ("Top 25", 25), ("Top 50", 50), ("All served", None)]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
for ax, (title, k) in zip(axes, DEPTHS):
    for i, name in enumerate(CHART):
        n, sh, _ = stats(name, k)
        left = 0
        for c in COL:
            v = sh[c]
            ax.barh(i, v, left=left, color=COL[c], label=LAB[c] if i == 0 and ax is axes[0] else None)
            if v >= 0.08:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if c.endswith("conf") else "#222")
            left += v
        ax.text(1.01, i, f"n={n}", va="center", fontsize=7, color="#666")
    ax.set_title(title); ax.set_xlim(0, 1.0); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
    ax.set_yticks(range(len(CHART))); ax.set_yticklabels(CHART)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].invert_yaxis()
fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9)
fig.suptitle("Locus of control by feed, split by the decider's confidence (signed-in reads)")
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(os.path.join(ROOT, "docs", "reports", "feed_audit_uncertainty.png"), dpi=150)
with open(os.path.join(ROOT, "docs", "reports", "feed_audit_uncertainty.md"), "w") as f:
    f.write("# Feed audit with uncertainty\n\nSorted by confident-external share. 'Confident' = the decider put "
            "P>=0.7 on its class (on the gold set 90-94% of those match Gemini's label; 0.5-0.7 matches ~55-70%). "
            "'Up to' adds the leaning calls, so each feed's true share lies between the two. Brackets: 95% Wilson on the confident share.\n\n"
            + table + "\n")
print("done")
