"""One sharepic per account, for the ten with the most effective reach, plus the
ranking picture that orders them.

The corpus picture (make_sharepic.py) answers "what happens in German Bluesky".
It cannot answer "and who is doing it", because a stacked bar over 547 posts
hides which account wrote them. These do: the same two bars -- share of posts,
share of interactions -- drawn once per account, so the pictures are comparable
by construction and a reader can hold two of them side by side.

Ordered by handlungsfähig share, most agency first. Not by `loc_mean`: the Score
axis puts neutral at 2.0, so an account posting nothing but cartoons outscores
one arguing for agency a third of the time. REPORT_de_top_accounts_loc.md makes
that argument at length; this script simply obeys it. The reach rank each
account was selected on is printed alongside, because the gap between the two
ranks is the finding -- the two accounts that lead on agency sit 8th and 9th on
reach.

Both windows are the matured one, posts aged 2-90 days. The ten were selected on
their matured posts, and charting every post would put posts in the picture that
the selection was not allowed to see.

Every number comes from `by_class_matured` in the report; nothing is typed in.
Regenerate the report and the pictures follow:
  DE_TOP_LIMIT=10 python3 report_de_top_loc.py

Run, from anywhere:
  uv run --with matplotlib python make_account_sharepics.py
  SHAREPIC_LANG=en uv run --with matplotlib python make_account_sharepics.py
  SHAREPIC_REPORT=REPORT_de_top_loc.json \
      uv run --with matplotlib python make_account_sharepics.py   # all 37
  SHAREPIC_REPORT=REPORT_de_top_loc_top10_by_internal_share.json \
      uv run --with matplotlib python make_account_sharepics.py   # the axis ten

The directory name carries the selection the report made, so the ten by reach
and the ten by internal share cannot land on top of each other.

Writes, into research/data/de_study/account_sharepics_<selection>_<lang>/:
  00_ranking.png              the ten in order, one row each
  01_<handle>.png .. 10_...   one account each, numbered by that order
and an .alt.txt beside every one of them.
"""
import json
import os

from matplotlib.patches import FancyBboxPatch

from sharepic_common import (ALT_MAX, COLORS, INK, M, MUTED, ORDER, W, H, fit,
                             fmt_int, fmt_pct, legend_row, new_figure, px_wide,
                             save, set_lang, stacked)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "de_study")

REPORT = os.environ.get("SHAREPIC_REPORT", "REPORT_de_top_loc_top10.json")
REPORT = REPORT if os.path.isabs(REPORT) or os.path.exists(REPORT) else os.path.join(
    DATA, os.path.basename(REPORT))
LANG = os.environ.get("SHAREPIC_LANG", "de")
set_lang(LANG)
# The directory carries the selection, not just the language: the ten by reach
# and the ten by internal share are different tens, and one of them landing on
# top of the other is the failure CLAUDE.md names.
SCOPE = (json.load(open(REPORT)).get("scope") or {}).get("kind", "all") \
    if os.path.exists(REPORT) else "all"
OUTDIR = os.environ.get("SHAREPIC_OUTDIR",
                        os.path.join(DATA, f"account_sharepics_{SCOPE}_{LANG}"))

TEXT = {
    "de": {
        "kicker": "Selbstbestimmung im deutschen Bluesky",
        "of": "von {n}",
        "sub": "{followers} Follower · {posts} Posts aus 2–90 Tagen · Median {median} Interaktionen pro Post (Reichweite: Platz {reach} von {n})",
        "bar_posts": "Posts",
        "bar_inter": "Interaktionen",
        "labels": {"external": "fremdbestimmt", "neutral": "neutral",
                   "internal": "selbstbestimmt"},
        "median": "Median pro Post",
        "basis": "Eingestuft danach, wie ein Post auf Lesende wirkt – nicht danach, ob er recht hat.",
        "claim": "{int_p} der Posts selbstbestimmt – sie holen {int_i} der Interaktionen.",
        "claim_none": "Kein einziger als selbstbestimmt eingestufter Post.",
        "foot": "Stand {date} · Interaktionen = Likes + Reposts + Antworten + Zitate · Einstufung: TypeSafe System One ({model})",
        "alt": ("Balkendiagramm zu {name} ({handle}), Platz {rank} von {n} nach dem Anteil "
                "selbstbestimmter Posts. Zwei gleich breite Balken über dieselben {posts} "
                "Posts aus 2 bis 90 Tagen. Oberer Balken, Anteil der Posts: {ext_p} Prozent "
                "fremdbestimmt, {neu_p} Prozent neutral, {int_p} Prozent selbstbestimmt. "
                "Unterer Balken, Anteil der {inter} Interaktionen: {ext_i} Prozent "
                "fremdbestimmt, {neu_i} Prozent neutral, {int_i} Prozent selbstbestimmt. "
                "Median pro Post: {ext_m} fremdbestimmt, {neu_m} neutral, {int_m} "
                "selbstbestimmt."),
        # the ranking picture
        "rank_title": "Wer schreibt selbstbestimmt?",
        "rank_sub": "{n} deutschsprachige Bluesky-Accounts, sortiert nach dem Anteil selbstbestimmter Posts. {posts} Posts aus 2–90 Tagen.",
        "rank_sub_median_interactions": "Die {n} deutschsprachigen Bluesky-Accounts mit der größten Reichweite, sortiert nach dem Anteil selbstbestimmter Posts. {posts} Posts aus 2–90 Tagen.",
        "rank_sub_internal_share": "Die {n} deutschsprachigen Bluesky-Accounts mit dem größten Anteil selbstbestimmter Posts – unter denen, die groß genug für ein Ranking sind. {posts} Posts aus 2–90 Tagen.",
        "rank_col": "Anteil selbstbestimmt",
        "rank_alt": ("Ranking. {n} deutschsprachige Bluesky-Accounts{how}, je ein Balken "
                     "über alle ihre {posts} Posts aus 2 bis 90 Tagen, aufgeteilt in "
                     "fremdbestimmt, neutral und selbstbestimmt, sortiert nach dem Anteil "
                     "selbstbestimmter Posts: {list}."),
        "how": "",
        "how_median_interactions": " mit der größten Reichweite",
        "how_internal_share": " mit dem größten Anteil selbstbestimmter Posts",
        "rank_item": "{name} {pct}",
    },
    "en": {
        "kicker": "Self-determination on German-speaking Bluesky",
        "of": "of {n}",
        "sub": "{followers} followers · {posts} posts aged 2–90 days · median {median} interactions per post (reach: rank {reach} of {n})",
        "bar_posts": "posts",
        "bar_inter": "interactions",
        "labels": {"external": "externally framed", "neutral": "neutral",
                   "internal": "self-determined"},
        "median": "median per post",
        "basis": "Classified by how a post leaves the reader – not by whether it is right.",
        "claim": "{int_p} of the posts are self-determined – they take {int_i} of the interactions.",
        "claim_none": "Not one post classified as self-determined.",
        "foot": "As of {date} · interactions = likes + reposts + replies + quotes · classifier: TypeSafe System One ({model}) · one snapshot, not a causal claim",
        "alt": ("Bar chart for {name} ({handle}), rank {rank} of {n} by share of "
                "self-determined posts. Two equal-width bars covering the same {posts} posts "
                "aged 2 to 90 days. Top bar, share of posts: {ext_p} percent externally "
                "framed, {neu_p} percent neutral, {int_p} percent self-determined. Bottom "
                "bar, share of the {inter} interactions: {ext_i} percent externally "
                "framed, {neu_i} percent neutral, {int_i} percent self-determined. Median "
                "per post: {ext_m} externally framed, {neu_m} neutral, {int_m} "
                "self-determined."),
        "rank_title": "Who writes self-determined posts?",
        "rank_sub": "{n} German-speaking Bluesky accounts, ordered by the share of their posts that are self-determined. {posts} posts aged 2–90 days.",
        "rank_sub_median_interactions": "The {n} German-speaking Bluesky accounts with the most reach, ordered by the share of their posts that are self-determined. {posts} posts aged 2–90 days.",
        "rank_sub_internal_share": "The {n} German-speaking Bluesky accounts with the largest share of self-determined posts – among those big enough to rank at all. {posts} posts aged 2–90 days.",
        "rank_col": "share self-determined",
        "rank_alt": ("Ranking. {n} German-speaking Bluesky accounts{how}, one bar each "
                     "over all {posts} of their posts aged 2 to 90 days, split into "
                     "externally framed, neutral and self-determined, ordered by the share "
                     "of self-determined posts: {list}."),
        "how": "",
        "how_median_interactions": " with the most reach",
        "how_internal_share": " with the largest share of self-determined posts",
        "rank_item": "{name} {pct}",
    },
}


def scoped(t, key, kind):
    """The selection-specific string if the language has one, else the plain
    one. "the ten with the most reach" and "the ten with the most agency" are
    different claims over different accounts, and a picture that carries the
    wrong one is wrong even when every number on it is right."""
    return t.get(f"{key}_{kind.split('_by_', 1)[-1]}", t[key])


def shares(row, key):
    """post_share / interaction_share per class, with the classes an account
    never posted filled in as zero. A missing key here would otherwise read as
    a missing bar rather than as the absence it is."""
    by = row["by_class_matured"]["by_class"]
    return {c: by.get(c, {}).get(key, 0.0) for c in ORDER}


def counts(row, key):
    by = row["by_class_matured"]["by_class"]
    return {c: by.get(c, {}).get(key, 0) for c in ORDER}


def medians(row):
    by = row["by_class_matured"]["by_class"]
    return {c: (int(by[c]["median_per_post"]) if c in by else None) for c in ORDER}


def badge(fig, rank, n, t):
    """The locus rank, top right. Drawn as a chip rather than folded into the
    title, so that ten pictures stack into a ranking at a glance."""
    x, y, w, h = 1 - M - 0.105, 0.845, 0.105, 0.105
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        transform=fig.transFigure, facecolor=COLORS["internal"], linewidth=0))
    fig.text(x + w / 2, y + h * 0.60, f"{rank}", fontsize=34, fontweight="bold",
             color="white", ha="center", va="center")
    fig.text(x + w / 2, y + h * 0.20, t["of"].format(n=n), fontsize=10.5,
             color="white", alpha=0.9, ha="center", va="center")


def account_pic(r, row, rank, n, reach_rank, out, t):
    ps, isr = shares(row, "post_share"), shares(row, "interaction_share")
    pn, inn = counts(row, "posts"), counts(row, "interactions")
    med = medians(row)
    tot = row["by_class_matured"]

    fig = new_figure()
    col = 1 - 2 * M
    badge(fig, rank, n, t)

    fit(fig, M, 0.945, t["kicker"], 15, col - 0.14, color=MUTED, va="top",
        fontweight="bold")
    fit(fig, M, 0.905, row["name"] or row["handle"], 34, col - 0.14,
        fontweight="bold", color=INK, va="top")
    fit(fig, M, 0.830, f"@{row['handle']}", 15, col - 0.14, color=INK, va="top",
        alpha=0.75)
    fit(fig, M, 0.784, t["sub"].format(
        followers=fmt_int(row["followers"]), posts=tot["n_posts"],
        median=fmt_int(row["interactions_median"]), reach=reach_rank, n=n),
        13, col, color=MUTED, va="top")
    # What the classifier claims, and what it does not. The labels are softer
    # than the finding, so the basis for them says so in the picture rather
    # than only in the report.
    fit(fig, M, 0.742, t["basis"], 11.5, col, color=MUTED, va="top", style="italic")

    ax = fig.add_axes([M, 0.315, col, 0.36])
    ax.set_axis_off()
    ax.set_xlim(-0.17, 1)
    ax.set_ylim(0, 1)
    for y, sh, cn, label in ((0.74, ps, pn, t["bar_posts"]),
                             (0.26, isr, inn, t["bar_inter"])):
        ax.text(-0.025, y, label, ha="right", va="center",
                fontsize=14, fontweight="bold", color=INK)
        # 0 digits and a floor on the label: a 2% segment is 20 px wide and a
        # percentage painted on it lands on its neighbours.
        stacked(ax, y, 0.40, sh, cn, label, min_pct_share=0.055)

    legend_row(fig, M, 0.2425, t["labels"], med, t["median"])

    claim = (t["claim"].format(int_p=fmt_pct(ps["internal"], 0),
                               int_i=fmt_pct(isr["internal"], 0))
             if pn["internal"] else t["claim_none"])
    fit(fig, M, 0.145, claim, 16, col, fontweight="bold", color=INK, va="top")
    fit(fig, M, 0.048, t["foot"].format(date=r["as_of"], model=r["model"]),
        10, col, color=MUTED, va="top")

    short = {"ext": "external", "neu": "neutral", "int": "internal"}
    alt = t["alt"].format(
        name=row["name"] or row["handle"], handle=row["handle"], rank=rank, n=n,
        posts=tot["n_posts"], inter=fmt_int(tot["total_interactions"]),
        **{f"{k}_p": f"{ps[c] * 100:.0f}" for k, c in short.items()},
        **{f"{k}_i": f"{isr[c] * 100:.0f}" for k, c in short.items()},
        **{f"{k}_m": (fmt_int(med[c]) if med[c] is not None else "0")
           for k, c in short.items()})
    return save(fig, out, alt)


def ranking_pic(r, rows, out, t, kind):
    """All ten in one picture, one full-width bar each. Sorted by the internal
    share, which is the right-most segment, so the ranking is readable as the
    ragged edge down the right of the chart without reading a single number."""
    n = len(rows)
    fig = new_figure()
    col = 1 - 2 * M
    posts = sum(row["by_class_matured"]["n_posts"] for row in rows)

    fit(fig, M, 0.955, t["rank_title"], 31, col, fontweight="bold", color=INK,
        va="top")
    fit(fig, M, 0.875, scoped(t, "rank_sub", kind).format(n=n, posts=fmt_int(posts)),
        14, col, color=MUTED, va="top")
    fit(fig, M, 0.833, t["basis"], 11.5, col, color=MUTED, va="top", style="italic")

    top, bottom = 0.785, 0.175
    ax = fig.add_axes([M, bottom, col, top - bottom])
    ax.set_axis_off()
    ax.set_xlim(-0.42, 1.09)                    # gutter for names, margin for %
    ax.set_ylim(0, 1)
    step = 1.0 / n
    for i, row in enumerate(rows):
        y = 1 - step * (i + 0.5)
        ps = shares(row, "post_share")
        ax.text(-0.42, y, f"{i + 1}", ha="left", va="center", fontsize=12,
                color=MUTED)
        name = ax.text(-0.02, y, row["name"] or row["handle"], ha="right",
                       va="center", fontsize=12.5, color=INK)
        # "Zentrum für Politische Schönheit" is three times the length of "taz".
        # Shrink it into the gutter instead of letting it run over the rank.
        size = 12.5
        while px_wide(ax, name) > 0.36 and size > 8:
            size -= 0.5
            name.set_fontsize(size)
        left = 0.0
        for cls in ORDER:
            w = ps[cls]
            if w <= 0:
                continue
            ax.add_patch(FancyBboxPatch(
                (left + 0.0012, y - step * 0.30), max(w - 0.0024, 0.001),
                step * 0.60, boxstyle="round,pad=0,rounding_size=0.004",
                linewidth=0, facecolor=COLORS[cls], clip_on=False))
            left += w
        ax.text(1.02, y, fmt_pct(ps["internal"], 0), ha="left", va="center",
                fontsize=12.5, fontweight="bold", color=INK)
    fig.text(1 - M, top + 0.012, t["rank_col"], ha="right", va="bottom",
             fontsize=10, color=MUTED)

    legend_row(fig, M, 0.115, t["labels"], {}, t["median"])
    fit(fig, M, 0.048, t["foot"].format(date=r["as_of"], model=r["model"]),
        10, col, color=MUTED, va="top")

    listing = ", ".join(
        t["rank_item"].format(name=row["name"] or row["handle"],
                              pct=fmt_pct(shares(row, "post_share")["internal"], 0))
        for row in rows)
    return save(fig, out, t["rank_alt"].format(n=n, posts=fmt_int(posts),
                                               how=scoped(t, "how", kind),
                                               list=listing))


def main():
    if not os.path.exists(REPORT):
        raise SystemExit(f"{REPORT}: no such report.\n"
                         f"  the top 10:  DE_TOP_LIMIT=10 python3 report_de_top_loc.py")
    r = json.load(open(REPORT))
    t = TEXT[LANG]
    # The report's rows arrive in reach order -- that is the selection -- so the
    # reach rank is their index, and the locus order is applied on top of it.
    rows = [row for row in r["accounts"] if row["by_class_matured"]["n_posts"]]
    reach_rank = {row["handle"]: i + 1 for i, row in enumerate(rows)}
    rows.sort(key=lambda row: -shares(row, "post_share")["internal"])
    n = len(rows)
    os.makedirs(OUTDIR, exist_ok=True)

    out = os.path.join(OUTDIR, "00_ranking.png")
    size, alt_path, long_alt = ranking_pic(
        r, rows, out, t, (r.get("scope") or {}).get("kind", ""))
    print(f"{out}  {W}x{H}  {size:,} bytes"
          + ("  WARNING: alt over the 2000-char field" if long_alt else ""))

    for i, row in enumerate(rows, 1):
        stem = f"{i:02d}_{row['handle'].replace('.', '-')}"
        out = os.path.join(OUTDIR, f"{stem}.png")
        size, alt_path, long_alt = account_pic(
            r, row, i, n, reach_rank[row["handle"]], out, t)
        print(f"{out}  {size:,} bytes  "
              f"int {shares(row, 'post_share')['internal']:.0%} of "
              f"{row['by_class_matured']['n_posts']} posts, reach rank "
              f"{reach_rank[row['handle']]}"
              + (f"  WARNING: alt over Bluesky's {ALT_MAX}-char field"
                 if long_alt else ""))


if __name__ == "__main__":
    main()
