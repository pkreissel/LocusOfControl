"""Engagement counts + author covariates for the German post pool.

Counts come from the AppView's aggregates, not from the firehose: counting like
events on Jetstream would truncate every post's tally by where it sits in the
36h window, which is exactly the bias the study cannot afford.

  app.bsky.feed.getPosts     25 uris/req  -> likeCount, repostCount, replyCount,
                                             quoteCount, indexedAt
  app.bsky.actor.getProfiles 25 dids/req  -> followersCount, followsCount,
                                             postsCount, account age

Both are readable unauthenticated on public.api.bsky.app.

Written as SNAPSHOTS, appended, one row per (uri, measured_at). Engagement is
strongly age-dependent, so a single pass over posts aged 0-36h confounds the
outcome with position in the window. Run once now for the early curve, then again
once every post is >=48h old for settled counts; the analysis uses the settled pass
and can use the pair to check that ranking is stable.

A post missing from the response has been deleted or its author deactivated. That
is recorded rather than dropped -- deletion is plausibly correlated with toxicity,
so silent attrition would bias the very estimate under study.

Usage:
  python hydrate_engagement.py [--in de_pool_raw.jsonl] [--label settled]
"""
import argparse
import json
import os
import sys
import time

import requests

BASE = "https://public.api.bsky.app/xrpc"
CHUNK = 25


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class Api:
    def __init__(self, qps=8.0):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "loc-de-engagement-study/1.0"
        self.min_gap = 1.0 / qps
        self.last = 0.0

    def get(self, path, params):
        delay = 2.0
        for attempt in range(6):
            gap = self.min_gap - (time.time() - self.last)
            if gap > 0:
                time.sleep(gap)
            self.last = time.time()
            try:
                r = self.s.get(f"{BASE}/{path}", params=params, timeout=30)
            except requests.RequestException:
                time.sleep(delay); delay *= 2
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                reset = r.headers.get("ratelimit-reset")
                wait = delay
                if reset:
                    try:
                        wait = max(delay, float(reset) - time.time() + 1)
                    except ValueError:
                        pass
                print(f"  429, waiting {wait:.0f}s", flush=True)
                time.sleep(min(wait, 300)); delay *= 2
                continue
            if r.status_code >= 500:
                time.sleep(delay); delay *= 2
                continue
            print(f"  {r.status_code}: {r.text[:200]}", flush=True)
            return None
        return None


def read_pool(path):
    uris, dids = [], []
    seen_u, seen_d = set(), set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            u, d = row.get("uri"), row.get("did")
            if u and u not in seen_u:
                seen_u.add(u); uris.append(u)
            if d and d not in seen_d:
                seen_d.add(d); dids.append(d)
    return uris, dids


def hydrate_posts(api, uris, out_path, label):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("snapshot") == label:
                    done.add(row["uri"])
    todo = [u for u in uris if u not in done]
    print(f"posts: {len(uris)} total, {len(done)} already in snapshot '{label}', "
          f"{len(todo)} to fetch", flush=True)

    found = missing = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for i, batch in enumerate(chunks(todo, CHUNK)):
            data = api.get("app.bsky.feed.getPosts", {"uris": batch})
            now = time.time()
            got = {}
            if data:
                for p in data.get("posts") or []:
                    got[p["uri"]] = p
            for uri in batch:
                p = got.get(uri)
                if p is None:
                    out.write(json.dumps({
                        "uri": uri, "snapshot": label, "measured_at": now,
                        "available": False}) + "\n")
                    missing += 1
                    continue
                out.write(json.dumps({
                    "uri": uri, "snapshot": label, "measured_at": now,
                    "available": True,
                    "like_count": p.get("likeCount", 0),
                    "repost_count": p.get("repostCount", 0),
                    "reply_count": p.get("replyCount", 0),
                    "quote_count": p.get("quoteCount", 0),
                    "indexed_at": p.get("indexedAt"),
                    "author_handle": (p.get("author") or {}).get("handle"),
                    "author_labels": [l.get("val") for l in (p.get("author") or {}).get("labels") or []],
                    "post_labels": [l.get("val") for l in p.get("labels") or []],
                }, ensure_ascii=False) + "\n")
                found += 1
            if i % 20 == 0:
                out.flush()
                el = time.time() - t0
                n = found + missing
                print(f"  {n}/{len(todo)} found={found} missing={missing} "
                      f"({n/max(el,1):.0f}/s)", flush=True)
    print(f"posts done: {found} available, {missing} deleted/unavailable", flush=True)


def hydrate_authors(api, dids, out_path, label):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("snapshot") == label:
                    done.add(row["did"])
    todo = [d for d in dids if d not in done]
    print(f"authors: {len(dids)} unique, {len(todo)} to fetch", flush=True)

    n = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, batch in enumerate(chunks(todo, CHUNK)):
            data = api.get("app.bsky.actor.getProfiles", {"actors": batch})
            now = time.time()
            got = {}
            if data:
                for p in data.get("profiles") or []:
                    got[p["did"]] = p
            for did in batch:
                p = got.get(did)
                if p is None:
                    out.write(json.dumps({"did": did, "snapshot": label,
                                          "measured_at": now, "available": False}) + "\n")
                    continue
                out.write(json.dumps({
                    "did": did, "snapshot": label, "measured_at": now, "available": True,
                    "handle": p.get("handle"),
                    "followers_count": p.get("followersCount"),
                    "follows_count": p.get("followsCount"),
                    "posts_count": p.get("postsCount"),
                    "account_created_at": p.get("createdAt"),
                    "has_avatar": bool(p.get("avatar")),
                    "description_len": len(p.get("description") or ""),
                    "labels": [l.get("val") for l in p.get("labels") or []],
                }, ensure_ascii=False) + "\n")
                n += 1
            if i % 20 == 0:
                out.flush()
                print(f"  authors {n}/{len(todo)}", flush=True)
    print(f"authors done: {n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="de_pool_raw.jsonl")
    ap.add_argument("--posts-out", default="de_engagement_snapshots.jsonl")
    ap.add_argument("--authors-out", default="de_authors_snapshots.jsonl")
    ap.add_argument("--label", default=None,
                    help="snapshot name, e.g. 'early' or 'settled' (default: UTC timestamp)")
    ap.add_argument("--qps", type=float, default=8.0)
    ap.add_argument("--skip-authors", action="store_true")
    args = ap.parse_args()

    label = args.label or time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    uris, dids = read_pool(args.inp)
    api = Api(qps=args.qps)
    hydrate_posts(api, uris, args.posts_out, label)
    if not args.skip_authors:
        hydrate_authors(api, dids, args.authors_out, label)
    print(f"snapshot label: {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
