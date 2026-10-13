"""Last N original posts of the largest German-speaking accounts, with engagement.

The account list is the follower ranking produced by the German-account survey:
search + two rounds of follow-graph snowball, each candidate language-checked
against its own recent posts. It is an input here, not something this script
derives -- see DE_TOP_ACCOUNTS.

Engagement is read from the AppView at collection time, so it is a single
snapshot: a post from last week has had its full life to accumulate likes, a
post from this morning has not. Rank accounts on medians, not sums, unless the
posting cadences being compared are similar.

"Post" here means an original top-level post: replies are excluded by the feed
filter, reposts by the author check. Quote posts count as originals.

Env:
  DE_TOP_ACCOUNTS  account list json  (default de_top50_accounts.json)
  DE_TOP_OUT       posts jsonl out    (default de_top50_posts.jsonl)
  DE_TOP_SUMMARY   per-account json   (default de_top50_summary.json)
  DE_TOP_N         posts per account  (default 100)
"""
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "de_study")
API = "https://public.api.bsky.app/xrpc/"

ACCOUNTS = os.environ.get("DE_TOP_ACCOUNTS", os.path.join(DATA, "de_top50_accounts.json"))
OUT = os.environ.get("DE_TOP_OUT", os.path.join(DATA, "de_top50_posts.jsonl"))
SUMMARY = os.environ.get("DE_TOP_SUMMARY", os.path.join(DATA, "de_top50_summary.json"))
N = int(os.environ.get("DE_TOP_N", "100"))


def get(method, params):
    url = API + method + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:            # the AppView is generous but not unlimited
                time.sleep(5 * (attempt + 1))
                continue
            if e.code in (400, 404):     # deactivated or renamed account
                return None
            time.sleep(1.5)
        except Exception:
            time.sleep(1.5)
    return None


def author_posts(handle, n):
    """Up to n original posts, newest first. Pages until n are collected or the
    feed runs out -- reposts and replies do not count toward n."""
    out, cursor, measured_at = [], None, time.time()
    while len(out) < n:
        params = {"actor": handle, "limit": 100, "filter": "posts_no_replies"}
        if cursor:
            params["cursor"] = cursor
        d = get("app.bsky.feed.getAuthorFeed", params)
        if d is None:
            break
        feed = d.get("feed", [])
        for item in feed:
            if item.get("reason"):                          # a repost, not their post
                continue
            p = item.get("post", {})
            if p.get("author", {}).get("handle") != handle:
                continue
            rec = p.get("record", {})
            embed = (p.get("embed") or {}).get("$type", "")
            out.append({
                "uri": p.get("uri"),
                "cid": p.get("cid"),
                "rkey": (p.get("uri") or "").rsplit("/", 1)[-1],
                "did": p.get("author", {}).get("did"),
                "handle": handle,
                "created_at": rec.get("createdAt"),
                "indexed_at": p.get("indexedAt"),
                "text": rec.get("text", ""),
                "text_len": len(rec.get("text", "")),
                "langs": rec.get("langs") or [],
                "embed_type": embed.split("#")[0].replace("app.bsky.embed.", "") or None,
                "has_quote": "record" in embed,
                "n_links": sum(1 for f in (rec.get("facets") or [])
                               for ft in f.get("features", [])
                               if ft.get("$type", "").endswith("#link")),
                "like_count": p.get("likeCount", 0),
                "repost_count": p.get("repostCount", 0),
                "reply_count": p.get("replyCount", 0),
                "quote_count": p.get("quoteCount", 0),
                "measured_at": measured_at,
            })
            if len(out) >= n:
                break
        cursor = d.get("cursor")
        if not cursor or not feed:
            break
    return out


def main():
    accounts = json.load(open(ACCOUNTS))
    rows, summary = [], []
    for i, a in enumerate(accounts, 1):
        h = a["handle"]
        posts = author_posts(h, N)
        rows.extend(posts)
        inter = [p["like_count"] + p["repost_count"] + p["reply_count"] + p["quote_count"]
                 for p in posts]
        span = None
        if len(posts) > 1 and posts[0]["created_at"] and posts[-1]["created_at"]:
            span = (posts[0]["created_at"][:10], posts[-1]["created_at"][:10])
        summary.append({
            "handle": h,
            "name": a.get("name", ""),
            "followers": a.get("followers"),
            "n_posts": len(posts),
            "span": span,
            "interactions_total": sum(inter),
            "interactions_median": statistics.median(inter) if inter else 0,
            "interactions_mean": round(statistics.mean(inter), 1) if inter else 0,
            "interactions_max": max(inter) if inter else 0,
            "likes": sum(p["like_count"] for p in posts),
            "reposts": sum(p["repost_count"] for p in posts),
            "replies": sum(p["reply_count"] for p in posts),
            "quotes": sum(p["quote_count"] for p in posts),
        })
        print(f"{i:>3}/{len(accounts)} {h:<36}{len(posts):>4} posts "
              f"{sum(inter):>8,} interactions", flush=True)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(summary, open(SUMMARY, "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(rows)} posts -> {OUT}\n{len(summary)} accounts -> {SUMMARY}")


if __name__ == "__main__":
    sys.exit(main())
