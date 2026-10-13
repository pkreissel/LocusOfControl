"""Time-based random sample of Bluesky posts over the platform's lifetime.

Design: stratify the research window by calendar month and draw N uniform
random instants per month. For each instant t, find the newest second at or
before t that holds a genuine post, enumerate every post in that second, and
pick one uniformly (app.bsky.feed.searchPosts, q="*", sort=latest, until=t).
Each draw is one post; per-month N is fixed, so every month carries the same
power no matter how small the platform was then.

Why seconds, not "the newest post": the search index files posts under
createdAt truncated to the second, filters `until` at that granularity
(inclusive), and returns posts inside a second in arbitrary order. At peak a
page of 100 covers 0.2 s, and bots fire on hh:mm:00.000, so below one second
"newest" is not identifiable.

The draw is length-biased. A post in second s is selected with probability

    p_i = (s - previous occupied second) / n_tied / month_length

so each row records `gap_s` and `n_tied`, and `weight` is n_tied / gap_s
normalised to mean 1 per month. From mid-2023 on nearly every second is
occupied, gap_s is 1, and the weight is the size of the second's cluster.

Backdated imports (Twitter archives etc.) are indexed under their claimed
createdAt: in March 2023 two thirds of the newest hits were posts indexed in
2024-2026. A genuine post is indexed within seconds of its createdAt; anything
more than LAG_TOL_S apart was not on Bluesky at t and is skipped, walking
further back until a genuine post turns up.

What this cannot correct: deleted and taken-down posts drop out at hydration,
so the sample is of posts that still exist at collection time.

Resumable: each draw has a stable id (month + index, instants from a fixed seed),
and ids already in the output are skipped.

Run (auth from bunny_feed/.env: BLUESKY_RESEARCH_HANDLE + BLUESKY_RESEARCH_APP_PASSWORD):
  python research/time_sample/collect_time_sample.py --per-month 1000
  python research/time_sample/collect_time_sample.py --per-month 1000 --lang de
  python research/time_sample/collect_time_sample.py --weights   # recompute weight column
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data" / "time_sample"
ENTRYWAY = "https://bsky.social"
PAGE = 100
# A genuine post is indexed within seconds of its createdAt, either side of it.
# An hour absorbs client clock skew and queued offline posts.
LAG_TOL_S = float(os.environ.get("TS_LAG_TOL_S", "3600"))
# Pages to walk back from t before giving up on a draw. Early 2023 (imports
# outnumber genuine posts) and bot-heavy hh:mm:00 seconds need the most.
MAX_PAGES = int(os.environ.get("TS_MAX_PAGES", "30"))


def parse_ts(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def iso(d):
    return d.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- client

class Client:
    def __init__(self):
        env = {}
        for line in (ROOT / "bunny_feed" / ".env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("\"'")
        env.update({k: v for k, v in os.environ.items() if k.startswith("BLUESKY_RESEARCH")})
        handle = env["BLUESKY_RESEARCH_HANDLE"]
        password = env.get("BLUESKY_RESEARCH_APP_PASSWORD") or env["BLUESKY_RESEARCH_PASSWORD"]
        s = self._post(f"{ENTRYWAY}/xrpc/com.atproto.server.createSession",
                       {"identifier": handle, "password": password})
        self._set_session(s)
        svc = [x for x in (s.get("didDoc") or {}).get("service", []) if x.get("id") == "#atproto_pds"]
        # Requests go through the account's own PDS, which proxies app.bsky.* to
        # the AppView with the session attached. searchPosts needs auth.
        self.pds = svc[0]["serviceEndpoint"] if svc else ENTRYWAY
        self.lock = threading.Lock()
        self.pause_until = 0.0

    def _set_session(self, s):
        self.access, self.refresh = s["accessJwt"], s["refreshJwt"]

    @staticmethod
    def _post(url, body=None, auth=None):
        req = urllib.request.Request(url, data=json.dumps(body or {}).encode() if body else b"",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": auth} if auth else {})},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def _refresh(self, stale):
        with self.lock:
            if self.access != stale:  # another thread already refreshed
                return
            self._set_session(self._post(f"{self.pds}/xrpc/com.atproto.server.refreshSession",
                                         auth="Bearer " + self.refresh))

    def get(self, path, params, tries=6):
        url = f"{self.pds}/xrpc/{path}?" + urllib.parse.urlencode(params)
        for attempt in range(tries):
            wait = self.pause_until - time.time()
            if wait > 0:
                time.sleep(wait)
            token = self.access
            try:
                req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
                with urllib.request.urlopen(req, timeout=30) as r:
                    remaining = int(r.headers.get("ratelimit-remaining", "1000"))
                    if remaining < 50:  # 3000 per 5 min; coast to the window reset
                        self.pause_until = float(r.headers.get("ratelimit-reset", time.time() + 60))
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read()[:300].decode("utf-8", "replace")
                if e.code == 401 or (e.code == 400 and "ExpiredToken" in body):
                    self._refresh(token)
                    continue
                if e.code == 429:
                    self.pause_until = float(e.headers.get("ratelimit-reset", time.time() + 60))
                    continue
                if e.code < 500 or attempt + 1 == tries:
                    raise RuntimeError(f"{path} {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                if attempt + 1 == tries:
                    raise
            time.sleep(2 * 2 ** attempt)


# ---------------------------------------------------------------- design

def month_starts(start, end):
    d = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while d < end:
        nxt = (d + timedelta(days=32)).replace(day=1)
        yield max(d, start), min(nxt, end), d.strftime("%Y-%m")
        d = nxt


def plan_draws(start, end, per_month, seed):
    """Instants are a function of (seed, month, index) only, so raising
    --per-month later extends each month's draws without moving existing ones."""
    draws = []
    for lo, hi, month in month_starts(start, end):
        rng = random.Random(f"{seed}:{month}")
        span_ms = int((hi - lo).total_seconds() * 1000)
        for i in range(per_month):
            t = lo + timedelta(milliseconds=rng.randrange(span_ms))
            draws.append({"draw_id": f"{month}-{i:05d}", "month": month, "t": iso(t)})
    return draws


def post_times(p):
    """(second the search index files the post under, indexing lag in s).
    The index sorts and filters on createdAt truncated to the second; order
    inside a second is arbitrary."""
    created, indexed = parse_ts(p["record"].get("createdAt")), parse_ts(p.get("indexedAt"))
    if not created or not indexed:
        return None, None
    return created.replace(microsecond=0), (indexed - created).total_seconds()


def collect_one(client, draw, lang=None):
    """Newest second at or before t holding a genuine post; every genuine post in
    it enumerated (paging past the arbitrary in-second order), one picked
    uniformly. Selection probability: gap_s / n_tied, with gap_s the distance in
    seconds to the previous occupied second."""
    t = parse_ts(draw["t"])
    t_sec = t.replace(microsecond=0)
    params = {"q": "*", "sort": "latest", "limit": PAGE,
              "until": t_sec.strftime("%Y-%m-%dT%H:%M:%SZ")}
    if lang:
        # Matches the post's declared langs (multi-language tags included), not
        # detected language: English text from a German-locale client counts.
        params["lang"] = lang
    sel_sec, members, prev, skipped, pages = None, {}, None, Counter(), 0
    cursor = None
    while pages < MAX_PAGES and prev is None:
        if cursor:
            params["cursor"] = cursor
        page = client.get("app.bsky.feed.searchPosts", params)
        pages += 1
        for p in page.get("posts", []):
            sec, lag = post_times(p)
            if sec is None:
                skipped["unparseable"] += 1
            elif sec > t_sec:
                skipped["after_t"] += 1
            elif abs(lag) > LAG_TOL_S:
                skipped["backdated"] += 1
            elif sel_sec is None or sec == sel_sec:
                sel_sec = sec
                members[p["uri"]] = (p, lag)  # keyed: pages can repeat a post
            elif sec < sel_sec:
                prev = sec
                break
            else:
                # A newer occupied second after the chosen one: a stray older
                # post led the page (seen 4x in 44k draws). The newer second is
                # the true candidate; start over from it.
                skipped["out_of_order"] += 1
                sel_sec, members = sec, {p["uri"]: (p, lag)}
        cursor = page.get("cursor")
        if not cursor or not page.get("posts"):
            break
    if sel_sec is None:
        return {**draw, "status": "no_post", "pages": pages, "skipped": dict(skipped)}

    # Order inside the second is the index's, not time: draw uniformly, seeded
    # by the draw so a rerun picks the same post.
    sel, sel_lag = members[random.Random(draw["draw_id"]).choice(sorted(members))]
    rec = sel["record"]
    reply = rec.get("reply") or {}
    text = rec.get("text") or ""
    return {
        **draw,
        "lang_filter": lang,
        # truncated: the page cap ran out inside the second, so not every
        # member was seen and the in-second draw is not uniform. Rare; flagged.
        "status": "ok" if prev else ("truncated" if pages >= MAX_PAGES else "no_prev"),
        "uri": sel["uri"],
        "cid": sel.get("cid"),
        "did": sel["author"]["did"],
        "handle": sel["author"].get("handle"),
        "text": text,
        "text_len": len(text),
        "langs": rec.get("langs") or [],
        "created_at": rec.get("createdAt"),
        "indexed_at": sel.get("indexedAt"),
        "second": iso(sel_sec),
        "index_lag_s": sel_lag,
        "prev_second": iso(prev) if prev else None,
        "gap_s": (sel_sec - prev).total_seconds() if prev else None,
        "n_tied": len(members),
        "dist_to_t_s": (t - sel_sec).total_seconds(),
        "is_reply": bool(reply),
        "reply_root": (reply.get("root") or {}).get("uri"),
        "embed_type": (rec.get("embed") or {}).get("$type"),
        "like_count": sel.get("likeCount"),
        "repost_count": sel.get("repostCount"),
        "reply_count": sel.get("replyCount"),
        "quote_count": sel.get("quoteCount"),
        "pages": pages,
        "skipped": dict(skipped),
        "collected_at": iso(datetime.now(timezone.utc)),
    }


# ---------------------------------------------------------------- weights + report

def add_weights(rows):
    """Inverse selection probability, normalised to mean 1 within each month.
    A draw whose gap is 0 (several posts on one timestamp, no distinct
    predecessor in reach) gets no weight rather than an infinite one."""
    by_month = defaultdict(list)
    for r in rows:
        r["weight"] = None
        if r.get("status") == "ok" and r["gap_s"] > 0:
            r["weight"] = r["n_tied"] / r["gap_s"]
            by_month[r["month"]].append(r)
    for rs in by_month.values():
        m = sum(r["weight"] for r in rs) / len(rs)
        for r in rs:
            r["weight"] /= m
    return rows


def load(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def report(rows):
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["month"]].append(r)
    print(f"{'month':8} {'draws':>6} {'ok':>6} {'uniq':>6} {'dup%':>5} {'med_gap_s':>9} "
          f"{'backdated_skipped':>17} {'deff':>5}")
    for m in sorted(by_month):
        rs = by_month[m]
        ok = [r for r in rs if r.get("uri")]
        uniq = len({r["uri"] for r in ok})
        gaps = sorted(r["gap_s"] for r in ok if r.get("gap_s") is not None)
        bd = sum(r.get("skipped", {}).get("backdated", 0) for r in rs)
        w = [r["weight"] for r in rs if r.get("weight")]
        # Kish design effect of the weights: n_eff = n / deff.
        deff = len(w) * sum(x * x for x in w) / sum(w) ** 2 if w else float("nan")
        print(f"{m:8} {len(rs):6} {len(ok):6} {uniq:6} {100 * (1 - uniq / max(len(ok), 1)):5.1f} "
              f"{gaps[len(gaps) // 2] if gaps else float('nan'):9.2f} {bd:17} {deff:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-02-01", help="window start (UTC, inclusive)")
    ap.add_argument("--end", default=None,
                    help="window end (UTC, exclusive); default: start of the current month")
    ap.add_argument("--per-month", type=int, default=1000)
    ap.add_argument("--seed", default="bsky-time-sample-v1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lang", default=None,
                    help="restrict to posts declaring this language (e.g. de); "
                         "the frame becomes seconds holding such a post")
    ap.add_argument("--out", default=None,
                    help="default: research/data/time_sample/time_sample[_<lang>].jsonl")
    ap.add_argument("--weights", action="store_true", help="only recompute weights + report")
    args = ap.parse_args()

    out = Path(args.out or DATA / f"time_sample{'_' + args.lang if args.lang else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    start = parse_ts(args.start + ("" if "T" in args.start else "T00:00:00Z"))
    now = datetime.now(timezone.utc)
    end = parse_ts(args.end + ("" if "T" in args.end else "T00:00:00Z")) if args.end else \
        now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if not args.weights:
        draws = plan_draws(start, end, args.per_month, args.seed)
        done = {r["draw_id"] for r in load(out)}
        todo = [d for d in draws if d["draw_id"] not in done]
        print(f"{len(draws)} draws over {len(draws) // max(args.per_month, 1)} months; "
              f"{len(done)} done, {len(todo)} to go", flush=True)
        client = Client()
        t0, n, failed = time.time(), 0, 0
        with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(collect_one, client, d, args.lang): d for d in todo}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as e:  # leave the draw undone; a rerun retries it
                    failed += 1
                    print(f"  {futs[fut]['draw_id']} failed: {e}", file=sys.stderr, flush=True)
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if n % 500 == 0:
                    f.flush()
                    rate = n / (time.time() - t0)
                    print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - n) / rate / 60:.0f} min",
                          flush=True)
        print(f"collected {n}, failed {failed}")

    rows = add_weights(load(out))
    with open(out.with_suffix(".tmp"), "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["draw_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.with_suffix(".tmp").replace(out)
    report(rows)


if __name__ == "__main__":
    main()
