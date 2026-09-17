"""Draw the study sample from the German census pool, and merge it into one table.

  draw   uniform random sample over a fixed 24h analysis window
  table  join pool + engagement snapshot + author covariates + Perspective scores

Why a 24h window: engagement varies strongly by time of day, so a sample that
covers only part of the diurnal cycle confounds any predictor correlated with
posting hour. The Jetstream retention window is ~35h, so the oldest 24h of it is
taken -- that maximises post age (and thus how settled the counts are) while
still covering every hour of the day exactly once.

Why an oversample: posts deleted before hydration cannot be counted, and deletion
is plausibly correlated with toxicity. The draw is made in random order and
oversized; the analysis set is the first N still-available rows in that order,
which stays a uniform random sample of surviving posts while making the attrition
visible instead of silent.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

ISO = "%Y-%m-%dT%H:%M:%SZ"


def read_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def cmd_draw(args):
    rows = list(read_jsonl(args.pool))
    if not rows:
        print(f"empty pool: {args.pool}", file=sys.stderr)
        return 1
    times = [r["time_us"] for r in rows if r.get("time_us")]
    t_start = min(times)
    t_end_full = max(times)
    win_end = t_start + int(args.window_hours * 3600 * 1e6)
    if win_end > t_end_full:
        win_end = t_end_full
        print(f"WARNING: pool spans only "
              f"{(t_end_full - t_start)/1e6/3600:.2f}h, less than the requested "
              f"{args.window_hours}h window", file=sys.stderr)

    in_window = [r for r in rows if r.get("time_us") and t_start <= r["time_us"] < win_end]

    eligible = []
    dropped = Counter()
    for r in in_window:
        if r.get("text_len", 0) < args.min_chars:
            dropped["too_short"] += 1
            continue
        if args.require_detected_de:
            if r.get("det_lang") != "de" or (r.get("det_conf") or 0) < args.min_det_conf:
                dropped["not_detected_german"] += 1
                continue
        elif args.drop_confident_other_lang:
            # Asymmetric on purpose. Requiring *confident German* would delete most
            # posts under ~20 chars, where the detector cannot commit either way --
            # and short posts are a real stratum with distinctive engagement, so
            # removing them trades a language bias for a length bias. Excluding only
            # posts the detector confidently calls another language removes the clear
            # English contamination while leaving the ambiguous short ones in, where
            # the declared `de` is the better evidence.
            dl, dc = r.get("det_lang"), (r.get("det_conf") or 0)
            if dl not in (None, "de") and dc >= args.other_lang_conf:
                dropped["confident_other_language"] += 1
                continue
        if args.exclude_replies and r.get("is_reply"):
            dropped["reply"] += 1
            continue
        eligible.append(r)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    draw = eligible[:args.n_draw]
    for i, r in enumerate(draw):
        r["draw_rank"] = i

    with open(args.out, "w", encoding="utf-8") as f:
        for r in draw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    authors = Counter(r["did"] for r in draw)
    meta = {
        "pool_rows": len(rows),
        "pool_window_hours": round((t_end_full - t_start) / 1e6 / 3600, 2),
        "analysis_window_start": time.strftime(ISO, time.gmtime(t_start / 1e6)),
        "analysis_window_end": time.strftime(ISO, time.gmtime(win_end / 1e6)),
        "analysis_window_hours": round((win_end - t_start) / 1e6 / 3600, 2),
        "in_window": len(in_window),
        "eligible": len(eligible),
        "excluded": dict(dropped),
        "drawn": len(draw),
        "sampling_fraction": round(len(draw) / max(len(eligible), 1), 4),
        "seed": args.seed,
        "unique_authors_in_draw": len(authors),
        "authors_with_multiple_posts": sum(1 for c in authors.values() if c > 1),
        "filters": {
            "min_chars": args.min_chars,
            "require_detected_de": args.require_detected_de,
            "min_det_conf": args.min_det_conf if args.require_detected_de else None,
            "drop_confident_other_lang": args.drop_confident_other_lang,
            "other_lang_conf": args.other_lang_conf if args.drop_confident_other_lang else None,
            "exclude_replies": args.exclude_replies,
        },
        "detected_language_mix": None,
        "reply_share_in_draw": round(sum(1 for r in draw if r.get("is_reply")) / max(len(draw), 1), 4),
    }
    lang_mix = Counter(r.get("det_lang") for r in draw)
    meta["detected_language_mix"] = {str(k): v for k, v in lang_mix.most_common(10)}
    meta["short_post_share"] = round(
        sum(1 for r in draw if r.get("text_len", 0) < 20) / max(len(draw), 1), 4)

    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


def parse_iso(s):
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        import datetime as dt
        return dt.datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def cmd_table(args):
    sample = {r["uri"]: r for r in read_jsonl(args.sample)}
    if not sample:
        print(f"empty sample: {args.sample}", file=sys.stderr)
        return 1

    # Latest snapshot with the requested label wins (a re-run overwrites a partial pass).
    eng = {}
    for r in read_jsonl(args.engagement):
        if args.snapshot and r.get("snapshot") != args.snapshot:
            continue
        prev = eng.get(r["uri"])
        if prev is None or r.get("measured_at", 0) >= prev.get("measured_at", 0):
            eng[r["uri"]] = r

    authors = {}
    for r in read_jsonl(args.authors):
        if args.snapshot and r.get("snapshot") != args.snapshot:
            continue
        prev = authors.get(r["did"])
        if prev is None or r.get("measured_at", 0) >= prev.get("measured_at", 0):
            authors[r["did"]] = r

    persp = {}
    for r in read_jsonl(args.perspective):
        if r.get("scores"):
            persp[r["uri"]] = r["scores"]

    # Census-derived author activity: how many posts this author made in the whole
    # 35h window. A cheap, unconfounded activity measure the API does not expose.
    pool_author_posts = Counter()
    for r in read_jsonl(args.pool):
        if r.get("did"):
            pool_author_posts[r["did"]] += 1

    from score_perspective_de import ATTRS
    keep_attrs = ATTRS
    if args.validity_report and os.path.exists(args.validity_report):
        with open(args.validity_report) as f:
            rep = json.load(f)
        keep_attrs = rep.get("keep", ATTRS)
        print(f"validity report: keeping {len(keep_attrs)} of {len(ATTRS)} attributes")

    ordered = sorted(sample.values(), key=lambda r: r.get("draw_rank", 0))
    rows = []
    stats = Counter()
    for r in ordered:
        uri = r["uri"]
        e = eng.get(uri)
        if e is None:
            stats["no_engagement_record"] += 1
            continue
        if not e.get("available"):
            stats["deleted_or_unavailable"] += 1
            continue
        p = persp.get(uri)
        if p is None:
            stats["no_perspective"] += 1
            if args.require_perspective:
                continue
            p = {}
        a = authors.get(r["did"]) or {}

        created_s = r["time_us"] / 1e6
        measured_s = e.get("measured_at") or 0
        age_h = (measured_s - created_s) / 3600 if measured_s else None
        acct_created = parse_iso(a.get("account_created_at"))

        likes = e.get("like_count") or 0
        reposts = e.get("repost_count") or 0
        replies = e.get("reply_count") or 0
        quotes = e.get("quote_count") or 0
        total = likes + reposts + replies + quotes
        followers = a.get("followers_count")

        row = {
            "uri": uri,
            "did": r["did"],
            "draw_rank": r.get("draw_rank"),
            "text": r["text"],
            "text_len": r.get("text_len"),
            "langs": ",".join(r.get("langs") or []),
            "det_lang": r.get("det_lang"),
            "det_conf": r.get("det_conf"),
            # Firehose ingest time, not the client-supplied createdAt, which is spoofable.
            "created_utc": time.strftime(ISO, time.gmtime(created_s)),
            "hour_utc": int(time.strftime("%H", time.gmtime(created_s))),
            "weekday": int(time.strftime("%w", time.gmtime(created_s))),
            "measured_utc": time.strftime(ISO, time.gmtime(measured_s)) if measured_s else None,
            "age_hours_at_measure": round(age_h, 3) if age_h is not None else None,
            "snapshot": e.get("snapshot"),
            "is_reply": int(bool(r.get("is_reply"))),
            "is_self_reply": int(bool(r.get("is_self_reply"))),
            "n_images": r.get("n_images", 0),
            "has_video": int(bool(r.get("has_video"))),
            "has_external": int(bool(r.get("has_external"))),
            "has_quote": int(bool(r.get("has_quote"))),
            "n_links": r.get("n_links", 0),
            "n_mentions": r.get("n_mentions", 0),
            "n_tags": r.get("n_tags", 0),
            "like_count": likes,
            "repost_count": reposts,
            "reply_count": replies,
            "quote_count": quotes,
            "total_engagement": total,
            "log1p_likes": round(math.log1p(likes), 5),
            "log1p_reposts": round(math.log1p(reposts), 5),
            "log1p_replies": round(math.log1p(replies), 5),
            "log1p_quotes": round(math.log1p(quotes), 5),
            "log1p_total": round(math.log1p(total), 5),
            "author_handle": a.get("handle") or e.get("author_handle"),
            "author_followers": followers,
            "author_follows": a.get("follows_count"),
            "author_posts_total": a.get("posts_count"),
            "author_account_age_days": round((created_s - acct_created) / 86400, 1) if acct_created else None,
            "author_has_avatar": int(bool(a.get("has_avatar"))) if a else None,
            "author_desc_len": a.get("description_len"),
            "author_posts_in_window": pool_author_posts.get(r["did"], 0),
            "log1p_followers": round(math.log1p(followers), 5) if followers is not None else None,
            # Engagement per follower: the crude exposure-normalised outcome. Prefer
            # modelling counts with log1p_followers as a covariate; this is for eyeballing.
            "engagement_per_follower": round(total / (followers + 1), 6) if followers is not None else None,
            "post_labels": ",".join(e.get("post_labels") or []),
            "author_labels": ",".join(e.get("author_labels") or []),
        }
        for at in keep_attrs:
            row[f"p_{at.lower()}"] = p.get(at)
        rows.append(row)
        stats["kept"] += 1
        if args.n_final and stats["kept"] >= args.n_final:
            break

    if not rows:
        print("no rows survived the merge", file=sys.stderr)
        return 1

    cols = list(rows[0].keys())
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    authors_in = Counter(r["did"] for r in rows)
    summary = {
        "rows": len(rows),
        "attrition": dict(stats),
        "unique_authors": len(authors_in),
        "authors_with_multiple_posts": sum(1 for c in authors_in.values() if c > 1),
        "rows_in_multi_post_authors": sum(c for c in authors_in.values() if c > 1),
        "reply_share": round(sum(r["is_reply"] for r in rows) / len(rows), 4),
        "zero_engagement_share": round(sum(1 for r in rows if r["total_engagement"] == 0) / len(rows), 4),
        "median_likes": sorted(r["like_count"] for r in rows)[len(rows) // 2],
        "mean_likes": round(sum(r["like_count"] for r in rows) / len(rows), 3),
        "age_hours_range": [
            min(r["age_hours_at_measure"] for r in rows if r["age_hours_at_measure"] is not None),
            max(r["age_hours_at_measure"] for r in rows if r["age_hours_at_measure"] is not None),
        ],
        "perspective_attributes": len(keep_attrs),
        "outputs": [args.out, args.out_jsonl],
    }
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draw")
    d.add_argument("--pool", default="de_pool_raw.jsonl")
    d.add_argument("--out", default="de_study_sample.jsonl")
    d.add_argument("--meta-out", default="de_study_sample_meta.json")
    d.add_argument("--n-draw", type=int, default=11000,
                   help="oversample; the analysis set is the first N surviving rows")
    d.add_argument("--window-hours", type=float, default=24.0)
    d.add_argument("--seed", type=int, default=20260908)
    d.add_argument("--min-chars", type=int, default=1)
    d.add_argument("--require-detected-de", action="store_true",
                   help="strict: keep only posts the detector confidently calls German "
                        "(removes ~34%% of the frame and nearly all short posts)")
    d.add_argument("--min-det-conf", type=float, default=0.70)
    d.add_argument("--drop-confident-other-lang", action="store_true", default=True,
                   help="default frame rule: drop posts the detector confidently "
                        "assigns to a language other than German")
    d.add_argument("--no-drop-confident-other-lang", dest="drop_confident_other_lang",
                   action="store_false",
                   help="keep the raw declared-de frame, contamination and all")
    d.add_argument("--other-lang-conf", type=float, default=0.70)
    d.add_argument("--exclude-replies", action="store_true")
    d.set_defaults(func=cmd_draw)

    t = sub.add_parser("table")
    t.add_argument("--sample", default="de_study_sample.jsonl")
    t.add_argument("--pool", default="de_pool_raw.jsonl")
    t.add_argument("--engagement", default="de_engagement_snapshots.jsonl")
    t.add_argument("--authors", default="de_authors_snapshots.jsonl")
    t.add_argument("--perspective", default="de_perspective.jsonl")
    t.add_argument("--validity-report", default="REPORT_perspective_de_validity.json")
    t.add_argument("--snapshot", default=None, help="snapshot label to use")
    t.add_argument("--n-final", type=int, default=10000)
    t.add_argument("--require-perspective", action="store_true", default=True)
    t.add_argument("--out", default="de_study_table.csv")
    t.add_argument("--out-jsonl", default="de_study_table.jsonl")
    t.add_argument("--summary-out", default="de_study_table_summary.json")
    t.set_defaults(func=cmd_table)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
