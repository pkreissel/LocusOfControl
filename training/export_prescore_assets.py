"""Export what the edge prescore job needs to ask TypeSafe and to calibrate its answer.

bunny_feed/src/prescore.ts scores a live feed's new posts with TypeSafe from the
edge, on a cron, and pushes them into the feed's score cache. It cannot import
this repository's Python, so the two things it must share with the offline
pipeline are exported here as JSON and committed next to it:

  ts_questions.json         the exact `questions` payload score_loc_typesafe.py
                            sends. An axis IS its prompt: an edge copy written
                            by hand would drift from ts_questions.py silently,
                            and every answer bought against it would then be a
                            different axis filed in the same ledger.

  prescore_calibration.json the quantile map calibrate_typesafe_to_vector.py
                            fits, frozen. That script fits it on the pool it is
                            given; the edge job sees 30 posts a call and cannot
                            fit anything, so it maps each post against this
                            fixed reference pool instead. The `band` key (external
                            floored, then P(internal), then Score) and the ladder
                            of the deployed vector's probabilities are stored
                            as-is; prescore.ts places a new post among the
                            reference keys and reads the ladder at that rank.

A threshold never survives a retrain, and neither does this: the ladder is the
vector's own probability distribution. The file records the vector version it
was fitted against, and the edge job refuses to push calibrated scores under any
other. Re-export after every retrain, from a pool scored by the new vector.

Run (from training/):
  python3 export_prescore_assets.py \
      --vector ../research/data/feed/scores_loc_vector_silberskeets_9d04026b.jsonl \
      --typesafe ../research/data/feed/scores_loc_typesafe_silberskeets.jsonl \
      --tag silberskeets
"""

import argparse
import hashlib
import json
import os
import time

import ts_questions

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "bunny_feed", "src")


def band_key(r):
    """calibrate_typesafe_to_vector.KEYS['band'], restated because prescore.ts has
    to restate it too: external floored below everything, then P(internal), then
    the graded Score breaking ties. test_prescore.ts checks the two agree."""
    p = r["ts_class_probs"]
    pi, pe = p.get("internal", 0.0), p.get("external", 0.0)
    floor = 0 if pe > max(pi, 1.0 - pi - pe) else 1
    return [floor, pi, r["ts_score"]]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True)
    ap.add_argument("--typesafe", required=True)
    ap.add_argument("--tag", required=True, help="the pool the map is fitted on")
    args = ap.parse_args()

    questions = ts_questions.questions()
    blob = json.dumps(questions, ensure_ascii=False, sort_keys=True)
    q_out = {
        "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
        "questions": questions,
    }
    with open(os.path.join(OUT_DIR, "ts_questions.json"), "w", encoding="utf-8") as f:
        json.dump(q_out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"ts_questions.json: {len(questions)} questions, sha {q_out['sha256']}")

    vec = {r["uri"]: r for r in load(args.vector)}
    ts = {r["uri"]: r for r in load(args.typesafe) if "error" not in r}
    uris = [u for u in vec if u in ts]
    versions = {vec[u]["vector_version"] for u in uris}
    if len(versions) != 1:
        raise SystemExit(f"Vector scores mix versions {sorted(versions)}; re-score first.")

    keys = sorted(band_key(ts[u]) for u in uris)
    ladder = sorted(float(vec[u]["vector_prob"]) for u in uris)
    cal = {
        "vector_version": versions.pop(),
        "key": "band",
        "fitted_on": args.tag,
        "n": len(uris),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Probabilities arrive from the API at two decimals; Score is rounded to
        # four only to keep the file small, which cannot reorder anything.
        "keys": [[k[0], round(k[1], 4), round(k[2], 4)] for k in keys],
        "ladder": [round(p, 6) for p in ladder],
    }
    with open(os.path.join(OUT_DIR, "prescore_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, separators=(",", ":"))
        f.write("\n")
    print(f"prescore_calibration.json: {cal['n']} reference posts from {args.tag}, "
          f"vector {cal['vector_version']}")


if __name__ == "__main__":
    main()
