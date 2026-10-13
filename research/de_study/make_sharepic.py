"""Render the external/neutral/internal interaction split as a Bluesky sharepic.

Every number is read from REPORT_de_top_loc.json -- nothing here is typed in, so
regenerating the report regenerates the image and the two cannot drift apart.

The picture makes one comparison, because a sharepic that makes two makes none:
the same corpus split by post count and by interactions, drawn to the same width.
The external segment grows from the first bar to the second, and that growth is
the whole finding.

The Bluesky constraints -- the 1 MB cap, the 1200x675 frame, the mandatory alt
text -- and the palette, the stacked bar and the measured text fitting all live
in sharepic_common.py, shared with make_account_sharepics.py.

The output name is derived from the report name, so a narrowed report
(`DE_TOP_LIMIT=10 python report_de_top_loc.py`) cannot quietly overwrite the
picture made from the wide one. When that report carries a `scope` block, the
accounts it selected but could not use are named under the bars: the subtitle
says "the 10 largest", and eight of them is not ten.

Run, from anywhere -- SHAREPIC_REPORT is a name, not a path:
  uv run --with matplotlib python make_sharepic.py
  SHAREPIC_LANG=en uv run --with matplotlib python make_sharepic.py
  SHAREPIC_REPORT=REPORT_de_top_loc_top10.json \
      uv run --with matplotlib python make_sharepic.py

The window follows the report: a report that selected its accounts on matured
posts is drawn over matured posts. SHAREPIC_TOTALS overrides that.
"""
import json
import os

from sharepic_common import (
    ALT_MAX,
    INK,
    MAX_BYTES,
    MUTED,
    ORDER,
    H,
    M,
    W,
    fit,
    fmt_int,
    legend_row,
    new_figure,
    save,
    set_lang,
    stacked,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "de_study")


def find_data(p):
    """Locate a data file: absolute as given, relative if it exists from the cwd,
    otherwise by name under research/data/de_study/. Every other script here
    resolves its data against its own location so it runs from any directory,
    and an env var that resolves against the cwd instead is the one path in the
    pipeline that breaks the moment you cd."""
    if os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(DATA, os.path.basename(p))


REPORT = find_data(os.environ.get("SHAREPIC_REPORT", "REPORT_de_top_loc.json"))
LANG = os.environ.get("SHAREPIC_LANG", "de")
set_lang(LANG)
STEM = os.path.splitext(os.path.basename(REPORT))[0].replace("REPORT_", "", 1)
OUT = os.environ.get("SHAREPIC_OUT", "")
OUT = OUT if os.path.isabs(OUT) or os.path.dirname(OUT) else os.path.join(
    DATA, OUT or f"sharepic_{STEM}_{LANG}.png")

TEXT = {
    "de": {
        "title": "Wem gehört die Aufmerksamkeit im deutschen Bluesky?",
        "sub1": "{posts} Posts der {n} reichweitenstärksten deutschsprachigen Accounts, {days} Tage.",
        "sub1_median_interactions": "{posts} Posts der {n} deutschsprachigen Accounts mit der größten Reichweite. Posts 2–90 Tage alt.",
        "sub1_internal_share": "{posts} Posts der {n} deutschsprachigen Accounts mit dem größten Anteil selbstbestimmter Posts. Posts 2–90 Tage alt.",
        "bar_posts": "Posts",
        "bar_inter": "Interaktionen",
        "labels": {"external": "ausweglos", "neutral": "neutral",
                   "internal": "selbstbestimmt"},
        "median": "Median pro Post",
        "sub2": "Eingestuft danach, wie ein Post auf Lesende wirkt – nicht danach, ob er recht hat.",
        "note": "Nicht enthalten: {who}.",
        "window": "{days} Tagen",
        "window_median_interactions": "einem Zeitraum von 2 bis 90 Tagen",
        "window_internal_share": "einem Zeitraum von 2 bis 90 Tagen",
        "why_none": "kein Post im Zeitraum",
        "why_young": "alle Posts jünger als 2 Tage",
        "foot": "Stand {date} · Interaktionen = Likes + Reposts + Antworten + Zitate · Einstufung: TypeSafe System One ({model})",
        "alt": ("Balkendiagramm. Zwei gleich breite Balken über dieselben {posts} Posts der "
                "{n} reichweitenstärksten deutschsprachigen Bluesky-Accounts aus {window}. "
                "Oberer Balken, Anteil der Posts: {ext_p} Prozent fremdbestimmt, {neu_p} Prozent "
                "neutral, {int_p} Prozent selbstbestimmt. Unterer Balken, Anteil der {inter} "
                "Interaktionen: {ext_i} Prozent fremdbestimmt, {neu_i} Prozent neutral, "
                "{int_i} Prozent selbstbestimmt. Fremdbestimmte Posts bekommen also einen "
                "deutlich größeren Anteil der Interaktionen, als ihr Anteil an den Posts "
                "erwarten ließe. Median pro Post: {ext_m} fremdbestimmt, {neu_m} neutral, "
                "{int_m} selbstbestimmt."),
    },
    "en": {
        "title": "Who gets the attention on German-speaking Bluesky?",
        "sub1": "{posts} posts from the {n} largest German-speaking accounts, {days} days.",
        "sub1_median_interactions": "{posts} posts from the {n} German-speaking accounts with the most interactions per post. Posts aged 2–90 days.",
        "sub1_internal_share": "{posts} posts from the {n} German-speaking accounts with the largest share of self-determined posts. Posts aged 2–90 days.",
        "sub2": "Classified by how a post leaves the reader – not by whether it is right.",
        "bar_posts": "posts",
        "bar_inter": "interactions",
        "labels": {"external": "externally framed", "neutral": "neutral",
                   "internal": "self-determined"},
        "median": "median per post",
        "kicker": "{ext_i}% of all interactions go to externally framed posts – from {ext_p}% of the posts.",
        "note": "Not included: {who}.",
        "window": "{days} days",
        "window_median_interactions": "a window of 2 to 90 days",
        "window_internal_share": "a window of 2 to 90 days",
        "why_none": "no post in the window",
        "why_young": "all posts younger than 2 days",
        "foot": "As of {date} · interactions = likes + reposts + replies + quotes · classifier: TypeSafe System One ({model}) · one snapshot, not a causal claim",
        "alt": ("Bar chart. Two equal-width bars covering the same {posts} posts from the "
                "{n} largest German-speaking Bluesky accounts over {window}. Top bar, "
                "share of posts: {ext_p} percent externally framed, {neu_p} percent neutral, "
                "{int_p} percent self-determined. Bottom bar, share of the {inter} "
                "interactions: {ext_i} percent externally framed, {neu_i} percent neutral, "
                "{int_i} percent self-determined. Externally framed posts take a far larger "
                "share of interactions than their share of posts would suggest. Median per "
                "post: {ext_m} externally framed, {neu_m} neutral, {int_m} self-determined."),
    },
}




def scope_note(r, t):
    """Name the selected accounts that contributed nothing. Built from the
    report's structured fields rather than its `reason` string, so the picture
    stays translatable and the report stays machine-readable."""
    sc = r.get("scope")
    if not sc or not sc["excluded"]:
        return None
    who = ", ".join(
        f"{e['handle']} ({t['why_none'] if e['posts_in_window'] == 0 else t['why_young']})"
        for e in sc["excluded"])
    return t["note"].format(who=who)






def main():
    if not os.path.exists(REPORT):
        raise SystemExit(
            f"{REPORT}: no such report.\n"
            f"  the wide one:    python3 report_de_top_loc.py\n"
            f"  the top 10:      DE_TOP_LIMIT=10 python3 report_de_top_loc.py")
    r = json.load(open(REPORT))
    kind = (r.get("scope") or {}).get("kind", "")
    # Accounts picked on their matured posts, charted over every post, would put
    # posts in the picture that the selection was not allowed to see.
    totals = os.environ.get("SHAREPIC_TOTALS",
                            "totals_matured" if "median_interactions" in kind
                            else "totals_all_posts")
    tot = r[totals]
    by = tot["by_class"]
    t = TEXT[LANG]
    # The headline counts the accounts that were *selected*, not the ones that
    # happened to survive -- `note` carries the difference.
    n_acc = r.get("n_accounts_selected", r["n_accounts"])
    note = scope_note(r, t)
    # How the accounts were chosen belongs in the subtitle: "the 10 largest" and
    # "the 10 most engaged-with" are different claims over different accounts.
    def scoped(key):
        """The scope-specific string if the language has one, else the plain one."""
        return t.get(f"{key}_{kind.split('_by_', 1)[-1]}", t[key])

    sub1 = scoped("sub1")

    post_share = {c: by[c]["post_share"] for c in ORDER}
    int_share = {c: by[c]["interaction_share"] for c in ORDER}
    post_n = {c: by[c]["posts"] for c in ORDER}
    int_n = {c: by[c]["interactions"] for c in ORDER}
    med = {c: int(by[c]["median_per_post"]) for c in ORDER}

    fig = new_figure()
    col = 1 - 2 * M

    fit(fig, M, 0.945, t["title"], 31, col, fontweight="bold", color=INK, va="top")
    fit(fig, M, 0.845, sub1.format(posts=fmt_int(tot["n_posts"]),
                                        n=n_acc, days=r["window_days"]),
        15, col, color=MUTED, va="top")
    # sub2 and kicker are optional: a language may carry its whole claim in the
    # title, and an absent key means "leave the line out", not a crash.
    if t.get("sub2"):
        fit(fig, M, 0.790, t["sub2"], 15, col, color=MUTED, va="top")

    # The bars get their own axes with a left gutter for the row labels, so a
    # label can never push into the bar or into the other row.
    ax = fig.add_axes([M, 0.315, col, 0.36])
    ax.set_axis_off()
    ax.set_xlim(-0.17, 1)
    ax.set_ylim(0, 1)
    for y, shares, counts, label in ((0.74, post_share, post_n, t["bar_posts"]),
                                     (0.26, int_share, int_n, t["bar_inter"])):
        ax.text(-0.025, y, label, ha="right", va="center",
                fontsize=14, fontweight="bold", color=INK)
        stacked(ax, y, 0.40, shares, counts, label)

    legend_row(fig, M, 0.2425, t["labels"], med, t["median"])

    if t.get("kicker"):
        fit(fig, M, 0.145, t["kicker"].format(
            ext_i=f"{int_share['external'] * 100:.0f}",
            ext_p=f"{post_share['external'] * 100:.0f}"),
            16, col, fontweight="bold", color=INK, va="top")
    if note:
        fit(fig, M, 0.092, note, 10, col, color=MUTED, va="top")
    fit(fig, M, 0.048, t["foot"].format(date=r["as_of"], model=r["model"]),
        10, col, color=MUTED, va="top")

    short = {"ext": "external", "neu": "neutral", "int": "internal"}
    alt = t["alt"].format(
        posts=fmt_int(tot["n_posts"]), n=n_acc,
        window=scoped("window").format(days=r["window_days"]),
        inter=fmt_int(tot["total_interactions"]),
        **{f"{k}_p": f"{post_share[c] * 100:.0f}" for k, c in short.items()},
        **{f"{k}_i": f"{int_share[c] * 100:.0f}" for k, c in short.items()},
        **{f"{k}_m": fmt_int(med[c]) for k, c in short.items()})
    if note:
        alt += " " + note
    size, alt_path, long_alt = save(fig, OUT, alt)

    print(f"{OUT}  {W}x{H}  {size:,} bytes ({size / MAX_BYTES:.0%} of the Bluesky cap)")
    print(f"{alt_path}  {len(alt)} chars"
          + (f"  WARNING: over Bluesky's {ALT_MAX}-char field" if long_alt else ""))


if __name__ == "__main__":
    main()
