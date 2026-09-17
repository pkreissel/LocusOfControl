"""Perspective API scoring for the German engagement study.

23 attributes in one request per text: the 6 core ones, the 10 that only resolve
under `languages=["en"]`, and the 7 bridging `*_EXPERIMENTAL` ones.

Language handling follows the measured behaviour of the API (see
memory/perspective-api-language-and-attribute-quirks.md): `languages` is a hint to
one multilingual model, not a model selector -- de vs en gave r=1.000 on the core
six over 17.5k texts -- so German text is submitted as `en`, which is the only way
to get the other 17 attributes at all.

Resumable and append-only: re-running skips URIs already in the output, so it can
run alongside the collector and be re-invoked as the pool grows.

Usage:
  python score_perspective_de.py [--follow] [--in FILE] [--out FILE]
"""
import argparse
import json
import os
import queue
import random
import sys
import threading
import time

import requests
from dotenv import load_dotenv

URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

CORE = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"]
EN_ONLY = ["SEXUALLY_EXPLICIT", "FLIRTATION", "OBSCENE", "SPAM", "INCOHERENT",
           "INFLAMMATORY", "LIKELY_TO_REJECT", "UNSUBSTANTIAL", "ATTACK_ON_AUTHOR",
           "ATTACK_ON_COMMENTER"]
BRIDGING = ["AFFINITY_EXPERIMENTAL", "COMPASSION_EXPERIMENTAL", "CURIOSITY_EXPERIMENTAL",
            "NUANCE_EXPERIMENTAL", "PERSONAL_STORY_EXPERIMENTAL", "REASONING_EXPERIMENTAL",
            "RESPECT_EXPERIMENTAL"]
ATTRS = CORE + EN_ONLY + BRIDGING

MAX_BYTES = 20000  # API limit is 20480 bytes; Bluesky posts never approach it


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uri"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def score_one(session, key, text):
    body = {
        "comment": {"text": text[:MAX_BYTES]},
        "languages": ["en"],
        "requestedAttributes": {a: {} for a in ATTRS},
        "doNotStore": True,
    }
    delay = 1.0
    for attempt in range(6):
        try:
            r = session.post(URL, params={"key": key}, json=body, timeout=45)
        except requests.RequestException as e:
            if attempt == 5:
                return None, f"net:{type(e).__name__}"
            time.sleep(delay + random.random()); delay *= 2
            continue
        if r.status_code == 200:
            scores = r.json().get("attributeScores", {})
            return ({a: scores.get(a, {}).get("summaryScore", {}).get("value")
                     for a in ATTRS}, None)
        if r.status_code in (429, 500, 503):
            if attempt == 5:
                return None, f"http:{r.status_code}"
            time.sleep(delay + random.random()); delay *= 2
            continue
        # 400 = unscoreable text (empty after normalisation, unsupported script...).
        return None, f"http:{r.status_code}:{r.text[:160]}"
    return None, "exhausted"


def worker(q, out_lock, out_f, session, key, stats):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            return
        uri, text = item
        scores, err = score_one(session, key, text)
        row = {"uri": uri, "scored_at": time.time()}
        if scores:
            row["scores"] = scores
            stats["ok"] += 1
        else:
            row["error"] = err
            stats["err"] += 1
        with out_lock:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
        q.task_done()


def pending(in_path, done):
    """Rows in the pool not yet scored. Skips texts too short to carry signal."""
    items = []
    if not os.path.exists(in_path):
        return items
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uri, text = row.get("uri"), (row.get("text") or "").strip()
            if uri and text and uri not in done:
                items.append((uri, text))
                done.add(uri)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="de_pool_raw.jsonl")
    ap.add_argument("--out", dest="out", default="de_perspective.jsonl")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--follow", action="store_true",
                    help="keep polling the input for new rows (run alongside the collector)")
    ap.add_argument("--idle-exit", type=int, default=900,
                    help="in --follow mode, stop after this many seconds with no new rows")
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("PERSPECTIVE_API_KEY")
    if not key:
        print("PERSPECTIVE_API_KEY not set", file=sys.stderr)
        return 1

    done = load_done(args.out)
    print(f"{len(done)} already scored in {args.out}", flush=True)

    session = requests.Session()
    q = queue.Queue(maxsize=args.concurrency * 4)
    out_lock = threading.Lock()
    stats = {"ok": 0, "err": 0}
    out_f = open(args.out, "a", encoding="utf-8")
    threads = [threading.Thread(target=worker,
                                args=(q, out_lock, out_f, session, key, stats), daemon=True)
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()

    t0 = time.time()
    last_new = time.time()
    total = 0
    try:
        while True:
            batch = pending(args.inp, done)
            if batch:
                last_new = time.time()
                for item in batch:
                    q.put(item)
                    total += 1
                    if total % 250 == 0:
                        el = time.time() - t0
                        print(f"queued={total} ok={stats['ok']} err={stats['err']} "
                              f"{stats['ok']/max(el,1):.1f}/s", flush=True)
            if not args.follow:
                break
            q.join()
            if time.time() - last_new > args.idle_exit:
                print("No new rows within idle window -- done.", flush=True)
                break
            time.sleep(20)
        q.join()
    finally:
        for _ in threads:
            q.put(None)
        q.join()
        out_f.close()

    el = time.time() - t0
    print(f"Scored {stats['ok']} ok, {stats['err']} errors in {el/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
