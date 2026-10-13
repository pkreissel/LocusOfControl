"""Collect the posts a published Bluesky feed is currently serving.

Reads app.bsky.feed.getFeed from the public AppView — no auth, since a feed
generator's output is public — and writes one JSONL row per post in the shape
annotate_lmstudio_soft_labels.py already eats (`uri` + `text`), plus the feed
metadata a later comparison needs (engagement counts, langs, repost reason).

Resumable: rows are appended and URIs already in the output file are skipped, so
re-running extends the sample rather than re-fetching it. A feed's cursor walks
backwards in time, so each run picks up whatever the feed has added since.

Run:
  uv run --with python-dotenv python collect_feed_posts.py \
      --feed https://bsky.app/profile/<handle>/feed/<rkey> --max-posts 600
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

APPVIEW = os.environ.get("LOC_APPVIEW", "https://public.api.bsky.app")
# getFeed's ceiling per page. The AppView returns 99-100 rows at this limit.
PAGE_LIMIT = 100
# Same length gate the deployed feed applies before it embeds anything
# (bunny_feed/src/index.ts hasEnoughText, FeedConfig.minWords = 6): URLs
# stripped, a token counts as a word only if it holds a letter or digit, with a
# character escape hatch for scripts that do not delimit words with spaces.
MIN_WORDS = int(os.environ.get("LOC_MIN_WORDS", "6"))
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
WORDY_RE = re.compile(r"[^\W_]", re.UNICODE)


def has_enough_text(text, min_words=MIN_WORDS):
    stripped = URL_RE.sub(" ", text or "")
    words = chars = 0
    for tok in stripped.split():
        if not WORDY_RE.search(tok):
            continue
        words += 1
        chars += sum(1 for ch in tok if WORDY_RE.match(ch))
    return words >= min_words or chars >= min_words * 5


BASE, AUTH = APPVIEW, None


def login():
    """LOC_AUTH=1: read the feed as the account in .env (BLUESKY_HANDLE,
    BLUESKY_APP_PASSWORD, BSKY_PDS_URL), so personalised feeds answer the way they
    do for a signed-in reader. Requests then go through the user's PDS, which
    proxies app.bsky.* to the AppView with the session attached."""
    global BASE, AUTH
    from pathlib import Path
    if os.environ.get("LOC_AUTH_JWT"):  # a caller that collects many feeds logs in once
        AUTH, BASE = "Bearer " + os.environ["LOC_AUTH_JWT"], os.environ["LOC_AUTH_PDS"]
        return
    env = {}
    for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("\"'")
    pds = env.get("BSKY_PDS_URL", "https://bsky.social").rstrip("/")
    req = urllib.request.Request(
        f"{pds}/xrpc/com.atproto.server.createSession",
        data=json.dumps({"identifier": env["BLUESKY_HANDLE"],
                         "password": env["BLUESKY_APP_PASSWORD"]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        AUTH = "Bearer " + json.loads(r.read())["accessJwt"]
    BASE = pds


def get(path, params, tries=4):
    url = f"{BASE}/xrpc/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Authorization": AUTH} if AUTH else {})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt + 1 == tries:
                raise
            print(f"  {path} failed ({type(e).__name__}), retrying", file=sys.stderr)
            time.sleep(2 * (2 ** attempt))


def resolve_feed_uri(feed):
    """Accept an at:// URI or a bsky.app feed URL, return the at:// URI.

    getFeed wants a DID in the authority position; a handle there is answered
    with an InternalServerError, so resolve it here.
    """
    m = re.match(r"https?://bsky\.app/profile/([^/]+)/feed/([^/?#]+)", feed)
    if m:
        actor, rkey = m.group(1), m.group(2)
    else:
        m = re.match(r"at://([^/]+)/app\.bsky\.feed\.generator/([^/?#]+)", feed)
        if not m:
            raise SystemExit(f"Not a feed URL or at:// generator URI: {feed}")
        actor, rkey = m.group(1), m.group(2)
    did = actor
    if not actor.startswith("did:"):
        did = get("com.atproto.identity.resolveHandle", {"handle": actor})["did"]
    return f"at://{did}/app.bsky.feed.generator/{rkey}"


def row_from_item(item, draw_rank):
    post = item["post"]
    rec = post.get("record") or {}
    text = rec.get("text") or ""
    reason = item.get("reason") or {}
    reply = rec.get("reply") or {}
    return {
        "uri": post["uri"],
        "cid": post.get("cid"),
        "did": post["author"]["did"],
        "handle": post["author"].get("handle"),
        "rkey": post["uri"].rsplit("/", 1)[-1],
        "text": text,
        "text_len": len(text),
        "langs": rec.get("langs") or [],
        "created_at": rec.get("createdAt"),
        "indexed_at": post.get("indexedAt"),
        "like_count": post.get("likeCount"),
        "repost_count": post.get("repostCount"),
        "reply_count": post.get("replyCount"),
        "quote_count": post.get("quoteCount"),
        "is_reply": bool(reply),
        "embed_type": (rec.get("embed") or {}).get("$type"),
        # The feed can surface a post as someone's repost; the text scored is
        # still the original post's.
        "via_repost": reason.get("$type", "").endswith("reasonRepost"),
        # Position in the feed as served, so "what the feed ranked first" stays
        # answerable after the cursor has moved on.
        "draw_rank": draw_rank,
        "enough_text": has_enough_text(text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default=os.environ.get("LOC_FEED"),
                    help="bsky.app feed URL or at:// generator URI")
    ap.add_argument("--out", default=os.environ.get("LOC_FEED_OUT", "feed_posts.jsonl"))
    ap.add_argument("--max-posts", type=int,
                    default=int(os.environ.get("LOC_FEED_MAX_POSTS", "600")))
    args = ap.parse_args()
    if not args.feed:
        raise SystemExit("--feed (or LOC_FEED) is required")

    if os.environ.get("LOC_AUTH") == "1":
        login()
    feed_uri = resolve_feed_uri(args.feed)
    meta = get("app.bsky.feed.getFeedGenerator", {"feed": feed_uri}).get("view", {})
    print(f"Feed: {meta.get('displayName')!r} ({feed_uri})")
    print(f"  service: {meta.get('did')}  likes: {meta.get('likeCount')}")

    seen = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.add(json.loads(line)["uri"])
        print(f"{len(seen)} posts already in {args.out}; appending new ones.")

    cursor, fetched, new, rank = None, 0, 0, len(seen)
    with open(args.out, "a", encoding="utf-8") as out:
        while new < args.max_posts:
            params = {"feed": feed_uri, "limit": PAGE_LIMIT}
            if cursor:
                params["cursor"] = cursor
            page = get("app.bsky.feed.getFeed", params)
            items = page.get("feed", [])
            cursor = page.get("cursor")
            fetched += len(items)
            for item in items:
                uri = item["post"]["uri"]
                if uri in seen:
                    continue
                seen.add(uri)
                out.write(json.dumps(row_from_item(item, rank), ensure_ascii=False) + "\n")
                rank += 1
                new += 1
                if new >= args.max_posts:
                    break
            out.flush()
            print(f"  fetched {fetched}, new {new}" + ("" if cursor else " (end of feed)"))
            if not cursor or not items:
                break
            time.sleep(0.2)

    short = 0
    with open(args.out, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    short = sum(1 for r in rows if not r["enough_text"])
    print(f"\n{args.out}: {len(rows)} posts total (+{new} this run), "
          f"{short} below minWords={MIN_WORDS} (the feed would not embed those).")


if __name__ == "__main__":
    main()
