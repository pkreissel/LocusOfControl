"""Score a JSONL corpus on the LoC axis with TypeSafe's System One API.

The three questions of ts_questions.py ride in one request per post, because
TypeSafe bills the state once per call and evaluates each question in isolation
against it. There is no multi-state batch endpoint, so throughput comes from
concurrency alone; the documented ceiling is 1200 requests/min and 250k
input tokens/s, and at ~1.2k tokens a post the request count binds first.

Run: TYPESAFE_API_KEY=... tmp_venv/bin/python score_loc_typesafe.py \
         dataset_validation_gemini.jsonl scores_loc_typesafe_gold.jsonl
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import ts_questions

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
WORKERS = int(os.environ.get("TYPESAFE_WORKERS", "12"))
KEY = os.environ["TYPESAFE_API_KEY"]
QUESTIONS = ts_questions.questions()

_lock = threading.Lock()
_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0, "errors": 0}


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
        return {**out, "error": resp["error"]}
    a = resp["answers"]
    return {
        **out,
        "ts_score": a["loc_score"]["score"],
        "ts_score_probs": a["loc_score"]["probabilities"],
        "ts_score_conf": a["loc_score"]["confidence"],
        "ts_class": a["loc_class"]["choice"],
        "ts_class_probs": a["loc_class"]["probabilities"],
        "ts_class_conf": a["loc_class"]["confidence"],
        "ts_noul": a["agency"]["noul"],
        "input_tokens": resp["usage"]["input_tokens"],
    }


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("text")]
    print(f"{len(rows)} posts -> {dst}  ({WORKERS} workers, model={MODEL})", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(score_row, rows))
    wall = time.time() - t0

    with open(dst, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cost = _stats["input_tokens"] / 1e6 * 0.042  # output tokens are free
    print(f"wall {wall:.1f}s  {len(rows)/wall:.1f} posts/s  "
          f"retries={_stats['retries']} errors={_stats['errors']}")
    print(f"input_tokens={_stats['input_tokens']:,}  output_tokens={_stats['output_tokens']:,}  "
          f"cost=${cost:.4f}  (${cost/len(rows)*1000:.4f} / 1000 posts)")
    lat = sorted(r["latency_s"] for r in results if r.get("latency_s"))
    if lat:
        print(f"latency p50={lat[len(lat)//2]:.2f}s p95={lat[int(len(lat)*.95)]:.2f}s max={lat[-1]:.2f}s")


if __name__ == "__main__":
    main()
