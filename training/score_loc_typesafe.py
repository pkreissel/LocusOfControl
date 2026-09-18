"""Score a JSONL corpus on the LoC axis with TypeSafe's System One API.

The three questions of ts_questions.py ride in one request per post, because
TypeSafe bills the state once per call and evaluates each question in isolation
against it. There is no multi-state batch endpoint, so throughput comes from
concurrency alone; the documented ceiling is 1200 requests/min and 250k
input tokens/s, and at ~1.2k tokens a post the request count binds first.

Every call is billed, so no post is ever asked twice. `data/typesafe_ledger.jsonl`
is the record of what has been paid for: one row per post, holding the full
answer, appended the instant it arrives and shared by every corpus, every feed
and every tag. A run asks only for posts that neither the ledger nor an existing
output already covers, and a post first scored under some other tag is served
from the ledger rather than bought again. Two things follow from appending per
row rather than writing at the end: a killed run keeps every answer it paid for,
and a second process pointed at the same ledger cannot silently drop the first
one's rows.

The ledger is the only cross-run record, and it lives under training/data/ which
is gitignored — but it is rebuildable from the score files it was fed:

    python score_loc_typesafe.py --rebuild-ledger data/scores_loc_typesafe_*.jsonl

which merges them in, keeping whatever the ledger already holds.

A ledger hit is served whatever model wrote it, because not paying twice is the
point; the model that answered is recorded per row and the breakdown is printed
on every run, so a pool mixing two models says so. TYPESAFE_RESCORE_ON_MODEL_CHANGE=1
asks again when the recorded model is not the current one.

Run: TYPESAFE_API_KEY=... tmp_venv/bin/python score_loc_typesafe.py \
         dataset_validation_gemini.jsonl scores_loc_typesafe_gold.jsonl
"""

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import ts_questions

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
WORKERS = int(os.environ.get("TYPESAFE_WORKERS", "12"))
# Read lazily, not at import: --rebuild-ledger asks the API for nothing and must
# work on a machine that has no key.
KEY = os.environ.get("TYPESAFE_API_KEY", "")
QUESTIONS = ts_questions.questions()

# Resolved against this file, not the cwd: the feed pipeline runs this scorer
# from research/feed/ and must land on the same ledger as a training run.
LEDGER = os.environ.get("TYPESAFE_LEDGER") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "typesafe_ledger.jsonl")
RESCORE_ON_MODEL_CHANGE = os.environ.get("TYPESAFE_RESCORE_ON_MODEL_CHANGE") == "1"

# The fields an answer must carry to be worth caching. A row missing any of them
# cannot serve a later run and so is not allowed to mark a post as paid for.
ANSWER_FIELDS = ("ts_score", "ts_class", "ts_class_probs", "ts_noul")

_lock = threading.Lock()
_ledger_lock = threading.Lock()
_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0, "errors": 0}


def text_key(text):
    """Key on the billed unit. The API bills the `state`, so two posts carrying
    the same text are one purchase even though they are two URIs."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def ledger_append(row):
    """Record a paid-for answer before the caller can do anything that might
    fail. Appended and fsynced under a lock, so a kill mid-run loses nothing."""
    with _ledger_lock:
        os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def load_ledger():
    by_uri, by_text = {}, {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if any(k not in r for k in ANSWER_FIELDS):
                continue
            by_uri[r["uri"]] = r
            if r.get("text_key"):
                by_text[r["text_key"]] = r
    return by_uri, by_text


def ask(text, attempts=5):
    body = json.dumps({"state": text, "model": MODEL, "questions": QUESTIONS}).encode()
    for i in range(attempts):
        req = urllib.request.Request(
            URL, data=body,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        try:
            t0 = time.time()
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            dt = time.time() - t0
            with _lock:
                _stats["calls"] += 1
                _stats["input_tokens"] += resp["usage"]["input_tokens"]
                _stats["output_tokens"] += resp["usage"]["output_tokens"]
            return resp, dt
        except urllib.error.HTTPError as e:
            # 429 rate limit and 529 overload are the documented backoff cases;
            # 401/422 are bugs in the request and must not be retried.
            if e.code not in (429, 529) or i == attempts - 1:
                with _lock:
                    _stats["errors"] += 1
                return {"error": f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"}, 0.0
            with _lock:
                _stats["retries"] += 1
            time.sleep(2 ** i)
        except Exception as e:  # connection reset, timeout
            if i == attempts - 1:
                with _lock:
                    _stats["errors"] += 1
                return {"error": repr(e)}, 0.0
            with _lock:
                _stats["retries"] += 1
            time.sleep(2 ** i)


def score_row(row):
    resp, dt = ask(row["text"])
    out = {"uri": row["uri"], "latency_s": round(dt, 3)}
    if "error" in resp:
        # Not ledgered: nothing was answered, so nothing is cached and the next
        # run asks again. An HTTP error is also the one case that is not billed.
        return {**out, "error": resp["error"]}
    a = resp["answers"]
    out = {
        **out,
        "ts_score": a["loc_score"]["score"],
        "ts_score_probs": a["loc_score"]["probabilities"],
        "ts_score_conf": a["loc_score"]["confidence"],
        "ts_class": a["loc_class"]["choice"],
        "ts_class_probs": a["loc_class"]["probabilities"],
        "ts_class_conf": a["loc_class"]["confidence"],
        "ts_noul": a["agency"]["noul"],
        "input_tokens": resp["usage"]["input_tokens"],
        "model": MODEL,
        "text_key": text_key(row["text"]),
    }
    ledger_append(out)
    return out


def describe(by_uri):
    models = Counter(r.get("model") or "unrecorded" for r in by_uri.values())
    return ", ".join(f"{m} {n}" for m, n in models.most_common())


def rebuild_ledger(files):
    """Fold existing score files into the ledger. Merges, never truncates: the
    ledger is the record of spend and a rebuild that could lose rows would cost
    money to undo."""
    by_uri, _ = load_ledger()
    print(f"ledger holds {len(by_uri)} posts before merge")
    added = 0
    for path in files:
        new = 0
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if any(k not in r for k in ANSWER_FIELDS) or r["uri"] in by_uri:
                continue
            # These rows predate the ledger, so the model that answered them was
            # never recorded. Left absent rather than guessed at.
            r.setdefault("model", None)
            by_uri[r["uri"]] = r
            ledger_append(r)
            new += 1
        added += new
        print(f"  {os.path.basename(path)}: +{new}")
    print(f"ledger now holds {len(by_uri)} posts (+{added}) -> {LEDGER}")
    print(f"  by model: {describe(by_uri)}")


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--rebuild-ledger":
        return rebuild_ledger(argv[1:])

    src, dst = argv[0], argv[1]
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("text")]

    by_uri, by_text = load_ledger()
    print(f"ledger: {len(by_uri)} posts already paid for ({describe(by_uri) or 'empty'})",
          flush=True)

    # An existing output is authoritative for its own tag; the ledger covers
    # everything scored anywhere. Either one means this post is not bought again.
    done = {}
    if os.path.exists(dst):
        for line in open(dst, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done[r["uri"]] = r
    in_output = len(done)

    todo, from_ledger, cross_model, unrecorded = [], 0, 0, 0
    for r in rows:
        if r["uri"] in done:
            continue
        hit = by_uri.get(r["uri"]) or by_text.get(text_key(r["text"]))
        if hit is not None and hit.get("model") is None:
            # Predates the model record. Unknown is not the same as different, so
            # these are never re-bought -- guessing they differ would re-buy the
            # whole backfilled ledger on a flag that means something narrower.
            unrecorded += 1
        elif hit is not None and hit["model"] != MODEL:
            if RESCORE_ON_MODEL_CHANGE:
                todo.append(r)
                continue
            cross_model += 1
        if hit is not None:
            # The hit may have been bought under another URI with the same text.
            done[r["uri"]] = {**hit, "uri": r["uri"]}
            from_ledger += 1
            continue
        todo.append(r)

    print(f"{len(rows)} posts -> {dst}", flush=True)
    print(f"  {in_output} already in the output, {from_ledger} served from the ledger, "
          f"{len(todo)} to ask  ({WORKERS} workers, model={MODEL})", flush=True)
    if cross_model:
        print(f"  {cross_model} of the ledger hits were answered by a model other than "
              f"{MODEL} (set TYPESAFE_RESCORE_ON_MODEL_CHANGE=1 to buy them again)", flush=True)
    if unrecorded:
        print(f"  {unrecorded} of the ledger hits predate the model record; served as-is",
              flush=True)

    # Two posts carrying the same text are one purchase, so only the first is
    # asked and its answer is copied onto the rest. The ledger lookup above only
    # catches repeats across runs; this catches them within one, which is where a
    # feed carrying a reposted line would otherwise pay twice.
    unique, siblings, first = [], {}, {}
    for r in todo:
        k = text_key(r["text"])
        if k in first:
            siblings.setdefault(k, []).append(r)
        else:
            first[k] = r
            unique.append(r)
    if len(unique) != len(todo):
        print(f"  {len(todo) - len(unique)} of those repeat another's text and are "
              f"copied, not asked", flush=True)

    if unique and not KEY:
        raise SystemExit("TYPESAFE_API_KEY unset; refusing to score.")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(score_row, unique)) if unique else []
    wall = time.time() - t0

    answered = {text_key(r["text"]): res
                for r, res in zip(unique, results) if "error" not in res}
    for k, rs in siblings.items():
        if k in answered:
            for r in rs:
                copy = {**answered[k], "uri": r["uri"]}
                ledger_append(copy)
                results.append(copy)
        else:
            results.extend({"uri": r["uri"], "error": "text twin failed"} for r in rs)

    out = {**done, **{r["uri"]: r for r in results}}
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            if r["uri"] in out:
                f.write(json.dumps(out[r["uri"]], ensure_ascii=False) + "\n")

    if not todo:
        print("nothing to ask; the ledger and the output already cover every post.")
        return
    asked = len(unique)

    cost = _stats["input_tokens"] / 1e6 * 0.042  # output tokens are free
    print(f"wall {wall:.1f}s  {asked/wall:.1f} posts/s  "
          f"retries={_stats['retries']} errors={_stats['errors']}")
    print(f"input_tokens={_stats['input_tokens']:,}  output_tokens={_stats['output_tokens']:,}  "
          f"cost=${cost:.4f}  (${cost/max(asked, 1)*1000:.4f} / 1000 posts)")
    lat = sorted(r["latency_s"] for r in results if r.get("latency_s"))
    if lat:
        print(f"latency p50={lat[len(lat)//2]:.2f}s p95={lat[int(len(lat)*.95)]:.2f}s max={lat[-1]:.2f}s")
    print(f"ledger now holds {len(by_uri) + len([r for r in results if 'error' not in r])} posts")


if __name__ == "__main__":
    main()
