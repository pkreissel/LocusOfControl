"""Stacked internal/external/neutral shares for chosen feeds at the top-K and overall.
Reads the audit caches (audit_feeds.py); writes docs/reports/feed_audit_top.png.
Usage: plot_feed_audit.py feed-forward mysky-your-feed discover for-you eurosky-fu"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "research", "data", "feed", "audit")
feeds = sys.argv[1:] or ["feed-forward", "mysky-your-feed", "discover", "for-you", "eurosky-fu"]
sc = {}
for l in open(f"{D}/scores_decider_ft.jsonl"):
    r = json.loads(l); sc[r["uri"]] = r["ts_class"]
DEPTHS = [("Top 10", 10), ("Top 25", 25), ("Top 50", 50), ("All served", None)]
COL = {"internal": "#2a6fbb", "external": "#d9822b", "neutral": "#c9ccd1"}

def shares(name, k):
    ps = sorted((json.loads(l) for l in open(f"{D}/posts_{name}.jsonl")), key=lambda p: p["draw_rank"])
    cls = [sc[p["uri"]] for p in (ps[:k] if k else ps) if p["uri"] in sc]
    return [cls.count(c) / len(cls) for c in COL], len(cls)

fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
print("| feed | depth | n | internal | external | neutral |\n|---|---|---:|---:|---:|---:|")
for ax, (title, k) in zip(axes, DEPTHS):
    for i, name in enumerate(feeds):
        s, n = shares(name, k)
        left = 0
        for c, v in zip(COL, s):
            ax.barh(i, v, left=left, color=COL[c], label=c if i == 0 and ax is axes[0] else None)
            if v >= 0.07:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if c != "neutral" else "#333")
            left += v
        print(f"| {name} | {title} | {n} | {s[0]:.0%} | {s[1]:.0%} | {s[2]:.0%} |")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
    ax.set_yticks(range(len(feeds))); ax.set_yticklabels(feeds)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].invert_yaxis()  # shared y: invert once, not per panel
fig.legend(loc="lower center", ncol=3, frameon=False)
fig.suptitle("Locus of control in the posts each feed serves first (decider-2b, signed-in reads)")
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
out = os.path.join(ROOT, "docs", "reports", "feed_audit_top.png")
fig.savefig(out, dpi=150); print("\nwrote", out)
