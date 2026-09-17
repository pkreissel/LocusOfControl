"""Census of German-language Bluesky posts over the Jetstream backfill window.

Sampling frame: every app.bsky.feed.post *create* whose record declares a `de*`
language, seen on Jetstream between the earliest retained event and the live edge.
No engagement gating -- unlike 2_stream_posts.py, nothing is filtered on whether a
post was liked or replied to, because that is the outcome under study.

Jetstream retains ~36h; a cursor further back silently clamps to it. Replay runs
~60x realtime, so a full window costs ~35 min and ~4 GB of traffic.

Env:
  DE_POOL_OUT     output jsonl              (default de_pool_raw.jsonl)
  DE_POOL_STATS   run stats json            (default de_pool_stats.json)
  DE_BACKFILL_H   hours to request          (default 40, clamps to retention)
  DE_AUDIT_EVERY  1-in-N non-de posts to language-detect for frame coverage (default 200)
"""
import asyncio
import json
import os
import signal
import sys
import time

import websockets

JETSTREAM = "wss://jetstream2.us-east.bsky.network/subscribe"
OUT = os.environ.get("DE_POOL_OUT", "de_pool_raw.jsonl")
STATS_FILE = os.environ.get("DE_POOL_STATS", "de_pool_stats.json")
BACKFILL_H = float(os.environ.get("DE_BACKFILL_H", "40"))
AUDIT_EVERY = int(os.environ.get("DE_AUDIT_EVERY", "200"))

_detector = None


def detector():
    """Lingua restricted to the languages that actually show up on the German-adjacent
    firehose. A restricted set is both faster and more accurate on short posts than
    the full 75-language model."""
    global _detector
    if _detector is None:
        from lingua import Language, LanguageDetectorBuilder
        langs = [Language.GERMAN, Language.ENGLISH, Language.DUTCH, Language.FRENCH,
                 Language.SPANISH, Language.ITALIAN, Language.PORTUGUESE, Language.POLISH,
                 Language.TURKISH, Language.SWEDISH, Language.DANISH, Language.CZECH,
                 Language.RUSSIAN, Language.JAPANESE]
        _detector = LanguageDetectorBuilder.from_languages(*langs).build()
    return _detector


def detect(text):
    vals = detector().compute_language_confidence_values(text)
    if not vals:
        return None, 0.0
    top = vals[0]
    return top.language.iso_code_639_1.name.lower(), round(top.value, 4)


def is_german(langs):
    return any(str(l).lower().split("-")[0] == "de" for l in langs or [])


def embed_features(record):
    """Media/link attachments, which are strong engagement confounders."""
    embed = record.get("embed") or {}
    etype = embed.get("$type") if isinstance(embed, dict) else None
    feats = {"embed_type": etype, "n_images": 0, "has_video": False,
             "has_external": False, "has_quote": False}
    stack = [embed] if isinstance(embed, dict) else []
    while stack:
        e = stack.pop()
        if not isinstance(e, dict):
            continue
        t = e.get("$type") or ""
        if t.startswith("app.bsky.embed.images"):
            feats["n_images"] += len(e.get("images") or [])
        elif t.startswith("app.bsky.embed.video"):
            feats["has_video"] = True
        elif t.startswith("app.bsky.embed.external"):
            feats["has_external"] = True
        elif t.startswith("app.bsky.embed.record"):
            feats["has_quote"] = True
        for key in ("media", "record"):
            if isinstance(e.get(key), dict):
                stack.append(e[key])
    return feats


def facet_features(record):
    n_links = n_mentions = n_tags = 0
    for f in record.get("facets") or []:
        if not isinstance(f, dict):
            continue
        for feat in f.get("features") or []:
            t = (feat or {}).get("$type") or ""
            if t.endswith("#link"):
                n_links += 1
            elif t.endswith("#mention"):
                n_mentions += 1
            elif t.endswith("#tag"):
                n_tags += 1
    return {"n_links": n_links, "n_mentions": n_mentions, "n_tags": n_tags}


def load_resume():
    """Resume from the newest event already written, so a restart neither
    duplicates nor skips."""
    cursor, seen, kept = None, set(), 0
    if not os.path.exists(OUT):
        return cursor, seen, kept
    with open(OUT, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            kept += 1
            seen.add(row["uri"])
            if row.get("time_us") and (cursor is None or row["time_us"] > cursor):
                cursor = row["time_us"]
    return cursor, seen, kept


class Stats:
    def __init__(self):
        self.events = 0
        self.post_creates = 0
        self.kept = 0
        self.empty_text = 0
        self.audit_n = 0          # non-de posts language-detected
        self.audit_de = 0         # of those, detected German (conf >= .70)
        self.first_time_us = None
        self.last_time_us = None

    def as_dict(self, elapsed):
        d = dict(self.__dict__)
        d["elapsed_s"] = round(elapsed, 1)
        if self.first_time_us and self.last_time_us:
            d["window_start_iso"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.first_time_us / 1e6))
            d["window_end_iso"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.last_time_us / 1e6))
            d["window_hours"] = round((self.last_time_us - self.first_time_us) / 1e6 / 3600, 2)
        # Share of German posts the declared-`de` frame misses: detected-German
        # posts among those NOT declaring de, scaled back up by the audit rate.
        if self.audit_n:
            d["audit_de_rate_among_non_de"] = round(self.audit_de / self.audit_n, 5)
            d["est_missed_german_posts"] = int(self.audit_de * AUDIT_EVERY)
        return d


async def run():
    resume_cursor, seen, kept0 = load_resume()
    if resume_cursor:
        cursor = resume_cursor
        print(f"Resuming: {kept0} rows in {OUT}, cursor {cursor}", flush=True)
    else:
        cursor = int((time.time() - BACKFILL_H * 3600) * 1_000_000)
        print(f"Fresh start: requesting {BACKFILL_H}h backfill (clamps to retention)", flush=True)

    st = Stats()
    st.kept = kept0
    t0 = time.time()
    stop = {"flag": False}

    def on_sigint(*_):
        stop["flag"] = True
        print("\nStopping after current message...", flush=True)
    signal.signal(signal.SIGINT, on_sigint)

    out = open(OUT, "a", encoding="utf-8")
    last_report = time.time()

    while not stop["flag"]:
        url = (f"{JETSTREAM}?wantedCollections=app.bsky.feed.post&cursor={cursor}")
        try:
            async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
                async for msg in ws:
                    if stop["flag"]:
                        break
                    st.events += 1
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    tu = data.get("time_us")
                    if tu:
                        cursor = tu
                        st.last_time_us = tu
                        if st.first_time_us is None:
                            st.first_time_us = tu

                    if data.get("kind") != "commit":
                        continue
                    commit = data.get("commit")
                    if not isinstance(commit, dict) or commit.get("operation") != "create":
                        continue
                    record = commit.get("record")
                    if not isinstance(record, dict):
                        continue
                    st.post_creates += 1

                    text = (record.get("text") or "").strip()
                    langs = record.get("langs") or []
                    if not is_german(langs):
                        # Frame-coverage audit on a 1-in-N subsample of the rest.
                        if AUDIT_EVERY and st.post_creates % AUDIT_EVERY == 0 and len(text) >= 20:
                            st.audit_n += 1
                            lang, conf = detect(text)
                            if lang == "de" and conf >= 0.70:
                                st.audit_de += 1
                        continue
                    if not text:
                        st.empty_text += 1
                        continue

                    did = data.get("did")
                    rkey = commit.get("rkey")
                    uri = f"at://{did}/app.bsky.feed.post/{rkey}"
                    if uri in seen:
                        continue
                    seen.add(uri)

                    reply = record.get("reply") if isinstance(record.get("reply"), dict) else None
                    root_uri = ((reply or {}).get("root") or {}).get("uri")
                    parent_uri = ((reply or {}).get("parent") or {}).get("uri")
                    det_lang, det_conf = detect(text) if len(text) >= 8 else (None, 0.0)

                    row = {
                        "uri": uri,
                        "did": did,
                        "rkey": rkey,
                        "cid": commit.get("cid"),
                        "text": text,
                        "text_len": len(text),
                        "langs": langs,
                        "n_langs": len(langs),
                        "det_lang": det_lang,
                        "det_conf": det_conf,
                        # createdAt is client-supplied and occasionally wrong; time_us is
                        # the firehose ingest timestamp and is what age should be based on.
                        "created_at": record.get("createdAt"),
                        "time_us": tu,
                        "is_reply": reply is not None,
                        "reply_root_uri": root_uri,
                        "reply_parent_uri": parent_uri,
                        "is_self_reply": bool(root_uri and root_uri.split("/")[2] == did) if root_uri else False,
                    }
                    row.update(embed_features(record))
                    row.update(facet_features(record))
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    st.kept += 1

                    if st.kept % 200 == 0:
                        out.flush()

                    now = time.time()
                    if now - last_report > 10:
                        last_report = now
                        age_h = (now * 1e6 - tu) / 1e6 / 3600
                        print(f"kept={st.kept} posts | events={st.events} | "
                              f"edge age={age_h:.2f}h | {st.events/(now-t0):.0f} ev/s",
                              flush=True)
                        with open(STATS_FILE, "w") as sf:
                            json.dump(st.as_dict(now - t0), sf, indent=2)
                        # Caught up with live: the backfill window is fully covered.
                        if age_h < 0.02:
                            print("Reached the live edge -- window complete.", flush=True)
                            stop["flag"] = True
                            break
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed; reconnecting in 5s...", flush=True)
            await asyncio.sleep(5)
        except Exception as e:  # noqa: BLE001 - the firehose serves malformed records
            print(f"Error: {type(e).__name__}: {e}; reconnecting in 5s...", flush=True)
            await asyncio.sleep(5)

    out.flush()
    out.close()
    stats = st.as_dict(time.time() - t0)
    with open(STATS_FILE, "w") as sf:
        json.dump(stats, sf, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
