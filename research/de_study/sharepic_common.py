"""The pieces every sharepic in this study shares: the palette, the stacked bar,
and the text fitting.

Two scripts draw pictures from REPORT_de_top_loc*.json -- make_sharepic.py draws
the corpus as one bar pair, make_account_sharepics.py draws one bar pair per
account -- and they have to look like the same study. A palette copied into two
files is a palette that will disagree with itself on the next edit.

Text is fitted by measurement, not by guessing point sizes: `fit()` shrinks a
string until it actually fits its column. German runs ~30% longer than English
for the same sentence, so a layout tuned by eye in one language silently
overflows in the other.

Bluesky constraints these honour:
  * 1,000,000 bytes per image, hard. `save()` checks rather than assumes.
  * 1200x675 (16:9), which renders uncropped and matches the link-card shape.
  * Alt text is not optional on a chart, so `save()` writes one beside the PNG.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

MAX_BYTES = 1_000_000
ALT_MAX = 2000                              # Bluesky's alt-text field limit

W, H, DPI = 1200, 675, 100
M = 0.055                                   # one margin, used everywhere

INK = "#16181c"
MUTED = "#6b7280"
BG = "#fbfaf8"
# Warm for external, cool for internal, neutral genuinely neutral. Deliberately
# not the red/amber/green ramp, which collapses under the red-green confusions.
COLORS = {
    "external": "#7a8fa3",
    "neutral":  "#b3aca0",
    "internal": "#ff9f43",
}
ORDER = ["external", "neutral", "internal"]

# Set once per run by the calling script. Number formatting is the only thing
# in here that is language-dependent, and threading a lang argument through
# every helper buys nothing over one module-level switch.
LANG = "de"


def set_lang(lang):
    global LANG
    LANG = lang


def fmt_int(n):
    """German uses '.' for thousands, English ','. Applied per number -- never as
    a blanket replace over a formatted sentence, which eats the real commas."""
    s = f"{int(n):,}"
    return s.replace(",", ".") if LANG == "de" else s


def fmt_pct(x, digits=1):
    s = f"{x * 100:.{digits}f}"
    return (s.replace(".", ",") + " %") if LANG == "de" else (s + "%")


def use_ui_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        ([n, "DejaVu Sans"] for n in ("Inter", "Source Sans 3", "Helvetica Neue",
                                      "Helvetica", "Arial", "Liberation Sans")
         if n in have), ["DejaVu Sans"])


def new_figure():
    use_ui_font()
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    fig.canvas.draw()                                   # a renderer for fit()
    return fig


def fit(fig, x, y, s, size, max_frac, **kw):
    """Place text, then shrink it until it really fits `max_frac` of the figure
    width. Measured against the actual renderer, so it holds for any font,
    any language."""
    t = fig.text(x, y, s, fontsize=size, **kw)
    r = fig.canvas.get_renderer()
    while size > 6:
        w = t.get_window_extent(renderer=r).width / (fig.get_figwidth() * fig.dpi)
        if w <= max_frac:
            break
        size -= 0.5
        t.set_fontsize(size)
    return t


def px_wide(ax, t):
    """Width of a text in axes data units, measured on the real renderer."""
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    lo, hi = ax.get_xlim()
    return bb.width / ax.get_window_extent().width * (hi - lo)


def caption(ax, cx, y, seg_w, count, unit, size=10.5):
    """The count under a percentage, sized to its own segment. The 15% segment is
    a fifth the width of the 57% one, so a caption that fits the wide one spills
    out of the narrow one -- drop the unit word first, shrink only if that is
    still not enough."""
    pad = seg_w * 0.9
    for s in (f"{fmt_int(count)} {unit}", fmt_int(count)):
        t = ax.text(cx, y, s, ha="center", va="center", color="white",
                    fontsize=size, alpha=0.85)
        cur = size
        while px_wide(ax, t) > pad and cur > 7.5:
            cur -= 0.5
            t.set_fontsize(cur)
        if px_wide(ax, t) <= pad:
            return t
        t.remove()
    return ax.text(cx, y, fmt_int(count), ha="center", va="center",
                   color="white", fontsize=7.5, alpha=0.85)


def stacked(ax, y, height, shares, counts, unit, pct_size=19, cap_size=10.5,
            min_pct_share=0.0):
    """One bar. Percentage inside each segment and the absolute count beneath it,
    both inside the segment, so nothing can collide with the other bar.

    An account can post no internal at all, and a zero-width segment with a
    '0 %' painted on it is worse than no segment: `min_pct_share` drops the
    label from anything too narrow to hold it, and the legend still carries
    the class."""
    left = 0.0
    for cls in ORDER:
        w = shares.get(cls, 0.0)
        if w <= 0:
            continue
        ax.add_patch(FancyBboxPatch(
            (left + 0.0012, y - height / 2), max(w - 0.0024, 0.001), height,
            boxstyle="round,pad=0,rounding_size=0.005",
            linewidth=0, facecolor=COLORS[cls], clip_on=False))
        cx = left + w / 2
        if w >= min_pct_share:
            ax.text(cx, y + height * 0.13, fmt_pct(w, 0 if min_pct_share else 1),
                    ha="center", va="center", color="white",
                    fontsize=pct_size, fontweight="bold")
            caption(ax, cx, y - height * 0.27, w, counts[cls], unit, cap_size)
        left += w


def save(fig, out, alt):
    """Write the PNG and its alt text, and refuse to hand the client a file
    Bluesky will reject."""
    fig.savefig(out, format="png", facecolor=BG)
    plt.close(fig)
    size = os.path.getsize(out)
    if size > MAX_BYTES:
        raise SystemExit(f"{out} is {size:,} bytes, over Bluesky's {MAX_BYTES:,} limit")
    alt_path = os.path.splitext(out)[0] + ".alt.txt"
    open(alt_path, "w").write(alt + "\n")
    return size, alt_path, len(alt) > ALT_MAX


def legend_row(fig, x0, y, labels, medians, median_word, step=0.25, size=13):
    """The class key, with each class's median under its name. Same swatch, same
    spacing in every picture in the study -- which is the point of it living
    here rather than in each script."""
    x = x0
    for cls in ORDER:
        fig.patches.append(plt.Rectangle((x, y - 0.0105), 0.017, 0.021,
                                         transform=fig.transFigure,
                                         facecolor=COLORS[cls], linewidth=0))
        fig.text(x + 0.025, y, labels[cls], fontsize=size, color=INK, va="center")
        if medians.get(cls) is not None:
            fig.text(x + 0.025, y - 0.0375, f"{fmt_int(medians[cls])} {median_word}",
                     fontsize=size - 2.5, color=MUTED, va="center")
        x += step
