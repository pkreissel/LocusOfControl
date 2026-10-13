"""Join the LoC scores for the largest German-speaking accounts to their engagement.

Two questions, and they need different windows:

  what each account posts   -- every scored post in the 90-day window, because a
                               rubric reading does not care how old a post is.
  what engagement follows   -- posts aged 2-90 days only. Counts are one snapshot
                               read at collection time, so a post from this
                               morning has not finished accumulating and would
                               drag its class down for no reason but its age.

Both are reported. The per-class totals are given on the full window as well,
since "how many interactions went to external posts" is a question about the
whole corpus and not only its matured part.

Rank accounts on `internal_share`, not on `loc_mean`: the Score axis puts neutral
at 2.0, so an account that posts nothing but small talk outscores one that argues
for agency half the time. loc_mean measures harmlessness, internal_share measures
the thing the feed is for.

`DE_TOP_LIMIT=10` narrows everything to N accounts and writes
REPORT_de_top_loc_top10.json instead. `DE_TOP_BY` picks which N:

  median_interactions  (default) the N accounts with the highest median
                       interactions on posts aged 2-90 days -- the like-for-like
                       measure, which is the only one that compares a newsroom
                       posting hourly with an account posting weekly.
  internal_share       the N accounts with the largest share of internal posts,
                       among those with enough matured posts to be ranked at
                       all. This is the axis ranking rather than the reach
                       ranking, and the two barely agree: Pia Lamberty leads it
                       at 67% internal and sits 27th of 37 by reach.
  followers            the N largest by follower count. Reach is not engagement,
                       and two of the ten largest have nothing to count: one has
                       not posted in 90 days, one posts so fast that no post of
                       its has matured. `scope.excluded` names them, because a
                       headline saying "the ten largest" over eight accounts'
                       posts is a claim about two accounts that were never
                       counted.

Each account row carries `by_class_matured`, its own three-way split over the
posts its median rests on. That is what the per-account sharepics are drawn
from, so the pictures cannot disagree with the table above them.

Inputs (research/data/de_study/): de_top_corpus_90d.jsonl, de_top50_accounts.json,
de_top50_posts.jsonl, scores_loc_typesafe_de_top90.jsonl.
Writes REPORT_de_top_loc.json.
"""
import collections
import datetime as dt
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "de_study")
ASOF = dt.datetime(2026, 9, 18, tzinfo=dt.timezone.utc)
MIN_POSTS = 10          # an account thinner than this is reported, never ranked
MATURE_DAYS = 2.0
TOP_N = int(os.environ.get("DE_TOP_LIMIT", "0") or 0)
TOP_BY = os.environ.get("DE_TOP_BY", "median_interactions")


def parse(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def load():
    posts = [json.loads(l) for l in open(os.path.join(DATA, "de_top_corpus_90d.jsonl"))]
    scores = {}
    for l in open(os.path.join(DATA, "scores_loc_typesafe_de_top90.jsonl")):
        r = json.loads(l)
        if "error" not in r:
            scores[r["uri"]] = r
    accounts = json.load(open(os.path.join(DATA, "de_top50_accounts.json")))
    for p in posts:
        p["age_days"] = (ASOF - parse(p["created_at"])).total_seconds() / 86400
        p["interactions"] = (p["like_count"] + p["repost_count"]
                             + p["reply_count"] + p["quote_count"])
    return posts, scores, accounts


def window_counts(handles):
    """Posts in the 90-day window, and posts matured within it, straight from the
    collection file -- the only place that still knows about accounts the scored
    corpus dropped."""
    out = collections.defaultdict(lambda: [0, 0])
    for line in open(os.path.join(DATA, "de_top50_posts.jsonl")):
        r = json.loads(line)
        if r["handle"] not in handles:
            continue
        age = (ASOF - parse(r["created_at"])).total_seconds() / 86400
        if age <= 90:
            out[r["handle"]][0] += 1
            if age >= MATURE_DAYS:
                out[r["handle"]][1] += 1
    return out


def subset(posts, scores, accounts, accmap):
    """Keep only TOP_N accounts, and account for any that keep nothing.

    Selecting on median interactions or on internal share means selecting on
    `rankable` rows only: an account with three matured posts has a median, and
    it means nothing -- and one with three posts, all internal, is a 100% that
    means less."""
    if TOP_BY in ("median_interactions", "internal_share"):
        ranked = [r for r in per_account(posts, scores, accmap) if r["rankable"]]
        if TOP_BY == "internal_share":
            # n_scored breaks ties, so the deeper corpus wins a shared share
            # rather than the file order deciding it.
            ranked.sort(key=lambda r: (-r["internal_share"], -r["n_scored"]))
        keep = [accmap[r["handle"]] for r in ranked[:TOP_N]]
    elif TOP_BY == "followers":
        keep = accounts[:TOP_N]           # the file is already follower-ordered
    else:
        raise SystemExit(f"DE_TOP_BY={TOP_BY!r}: expected median_interactions, "
                         f"internal_share or followers")
    handles = {a["handle"] for a in keep}
    kept = [p for p in posts if p["handle"] in handles]
    present = {p["handle"] for p in kept}
    counts = window_counts(handles)
    excluded = []
    for a in keep:
        if a["handle"] in present:
            continue
        n90, nmat = counts[a["handle"]]
        excluded.append({
            "handle": a["handle"], "name": a["name"], "followers": a["followers"],
            "posts_in_window": n90, "matured_posts": nmat,
            "reason": ("no post in the 90-day window" if n90 == 0 else
                       f"{nmat} posts aged {MATURE_DAYS:.0f}-90 days, under the "
                       f"{MIN_POSTS} an account needs to enter the scored corpus"),
        })
    return kept, {"kind": f"top{TOP_N}_by_{TOP_BY}", "selected": TOP_N,
                  "selected_handles": [a["handle"] for a in keep],
                  "with_posts": len(present), "excluded": excluded}


def class_totals(posts, scores, matured_only):
    """The external/neutral/internal split of a set of posts, by count and by
    interactions. Used for the corpus and, per account, for the picture each
    account gets -- the same function, so a per-account bar and the corpus bar
    can never be computed two different ways."""
    agg = collections.defaultdict(lambda: collections.Counter())
    med = collections.defaultdict(list)
    for p in posts:
        if p["uri"] not in scores:
            continue
        if matured_only and p["age_days"] < MATURE_DAYS:
            continue
        c = scores[p["uri"]]["ts_class"]
        agg[c].update(posts=1, interactions=p["interactions"], likes=p["like_count"],
                      reposts=p["repost_count"], replies=p["reply_count"],
                      quotes=p["quote_count"])
        med[c].append(p["interactions"])
    n = sum(a["posts"] for a in agg.values())
    tot = sum(a["interactions"] for a in agg.values())
    if not n:
        return {"n_posts": 0, "total_interactions": 0, "by_class": {}}
    out = {}
    for c, a in agg.items():
        out[c] = {**dict(a),
                  "post_share": round(a["posts"] / n, 4),
                  "interaction_share": round(a["interactions"] / tot, 4),
                  "mean_per_post": round(a["interactions"] / a["posts"], 1),
                  "median_per_post": statistics.median(med[c])}
    return {"n_posts": n, "total_interactions": tot, "by_class": out}


def per_account(posts, scores, accounts):
    byh = collections.defaultdict(list)
    for p in posts:
        if p["uri"] in scores:
            byh[p["handle"]].append(p)
    rows = []
    for h, ps in byh.items():
        s = [scores[p["uri"]]["ts_score"] for p in ps]
        cl = collections.Counter(scores[p["uri"]]["ts_class"] for p in ps)
        mature = [p for p in ps if p["age_days"] >= MATURE_DAYS]
        inter = [p["interactions"] for p in mature]
        n = len(ps)
        rows.append({
            "handle": h,
            "name": accounts.get(h, {}).get("name", ""),
            "followers": accounts.get(h, {}).get("followers"),
            "n_scored": n,
            "loc_mean": round(statistics.mean(s), 3),
            "loc_median": round(statistics.median(s), 3),
            "p_internal_mean": round(statistics.mean(
                scores[p["uri"]]["ts_class_probs"].get("internal", 0.0) for p in ps), 4),
            "internal_share": round(cl["internal"] / n, 4),
            "neutral_share": round(cl["neutral"] / n, 4),
            "external_share": round(cl["external"] / n, 4),
            "noul_mean": round(statistics.mean(scores[p["uri"]]["ts_noul"] for p in ps), 3),
            "n_matured": len(inter),
            "interactions_median": statistics.median(inter) if inter else None,
            "interactions_total": sum(inter),
            # Within-account: does this account's more-internal post beat its own
            # average? Immune to the reach confound that swamps the pooled figure.
            "spearman_loc_vs_interactions": (
                round(spearman([scores[p["uri"]]["ts_score"] for p in mature], inter), 3)
                if len(mature) >= 25 else None),
            "rankable": len(inter) >= MIN_POSTS,
            # The account's own split, over the matured window that its median
            # rests on. make_account_sharepics.py draws one picture per account
            # from this and nothing else.
            "by_class_matured": class_totals(ps, scores, matured_only=True),
        })
    rows.sort(key=lambda r: -(r["interactions_median"] or -1))
    return rows


def main():
    posts, scores, accounts = load()
    accmap = {a["handle"]: a for a in accounts}
    scope = None
    if TOP_N:
        posts, scope = subset(posts, scores, accounts, accmap)
    rows = per_account(posts, scores, accmap)
    mature = [p for p in posts if p["uri"] in scores and p["age_days"] >= MATURE_DAYS]
    within = [r["spearman_loc_vs_interactions"] for r in rows
              if r["spearman_loc_vs_interactions"] is not None]
    report = {
        "as_of": ASOF.date().isoformat(),
        "model": "jev-latest",
        "window_days": 90,
        "n_accounts": len(rows),
        "n_posts_scored": sum(1 for p in posts if p["uri"] in scores),
        "totals_all_posts": class_totals(posts, scores, matured_only=False),
        "totals_matured": class_totals(posts, scores, matured_only=True),
        "loc_vs_engagement": {
            "pooled_spearman": round(spearman([scores[p["uri"]]["ts_score"] for p in mature],
                                              [p["interactions"] for p in mature]), 3),
            "pooled_n": len(mature),
            "within_account_median": round(statistics.median(within), 3),
            "within_account_n": len(within),
            "within_account_negative": sum(1 for r in within if r < 0),
        },
        "accounts": rows,
    }
    if scope:
        report["scope"] = scope
        report["n_accounts_selected"] = scope["selected"]
    # The default keeps the bare _top10 name that the write-up already cites;
    # any other selection says which, because two different tens under one
    # filename is the mistake CLAUDE.md warns about.
    suffix = (f"_top{TOP_N}" + ("" if TOP_BY == "median_interactions" else f"_by_{TOP_BY}")
              if TOP_N else "")
    out = os.path.join(DATA, f"REPORT_de_top_loc{suffix}.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(rows)} accounts, {report['n_posts_scored']} posts -> {out}")
    for k in ("totals_all_posts", "totals_matured"):
        t = report[k]
        print(f"\n{k}: {t['n_posts']} posts, {t['total_interactions']:,} interactions")
        for c in ("external", "neutral", "internal"):
            b = t["by_class"][c]
            print(f"  {c:<9}{b['posts']:>6} posts ({b['post_share']:.1%})  "
                  f"{b['interactions']:>9,} interactions ({b['interaction_share']:.1%})  "
                  f"mean {b['mean_per_post']:>6.0f}  median {b['median_per_post']:>5.0f}")
    print(f"\nLoC vs engagement: {report['loc_vs_engagement']}")
    if scope:
        print(f"\nselected by {TOP_BY}:")
        for r in rows:
            print(f"  {r['handle']:<32} median {r['interactions_median']:>8.0f}  "
                  f"n_matured {r['n_matured']:>3}  int% {r['internal_share']:.1%}")
    for e in (scope or {}).get("excluded", []):
        print(f"excluded: {e['handle']} ({e['followers']:,} followers) -- {e['reason']}")


if __name__ == "__main__":
    main()
