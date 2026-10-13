"""Score the time samples with jev on identity fusion, with the LoC axis riding along.

TypeSafe bills the state once per call and evaluates each question in isolation,
so the fusion questions (fusion_questions.py) and the three LoC questions
(training/ts_questions.py) go out in one request per post. The answer is written
to the shared ledger (training/data/typesafe_ledger.jsonl) with both sets of
fields, so the LoC part serves every later run and nothing is bought twice. A
post the ledger already holds with LoC but without fusion is asked the fusion
questions alone.

Inputs are both collect_time_sample.py outputs; posts in both are one purchase.
Output is one row per unique uri:
  research/data/time_sample/scores_typesafe.jsonl

Run (TYPESAFE_API_KEY from .env):
  python research/time_sample/score_time_sample.py --limit 200   # pilot, stratified by month
  python research/time_sample/score_time_sample.py
  python research/time_sample/score_time_sample.py --de-only    # the German sample alone
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fusion_questions  # noqa: E402
import score_loc_typesafe as sl  # noqa: E402
import ts_questions  # noqa: E402

DATA = ROOT / "research" / "data" / "time_sample"
SOURCES = [DATA / "time_sample.jsonl", DATA / "time_sample_de.jsonl"]
OUT = DATA / "scores_typesafe.jsonl"

FUSION_Q = fusion_questions.questions()
BOTH_Q = {**ts_questions.questions(), **FUSION_Q}
FUSION_FIELDS = ("fusion_score", "fusion_target", "fusion_target_probs", "fusion_we", "outgroup")


def has_fusion(row):
    return (row is not None and all(k in row for k in FUSION_FIELDS)
            and row.get("fusion_version") == fusion_questions.VERSION)


def load_env():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    sl.KEY = os.environ.get("TYPESAFE_API_KEY", "")


def fusion_fields(a):
    return {
        "fusion_score": a["fusion_score"]["score"],
        "fusion_score_probs": a["fusion_score"]["probabilities"],
        "fusion_score_conf": a["fusion_score"]["confidence"],
        "fusion_target": a["fusion_target"]["choice"],
        "fusion_target_probs": a["fusion_target"]["probabilities"],
        "fusion_we": a["fusion_we"]["noul"],
        "outgroup": a["outgroup"]["score"],
        "outgroup_probs": a["outgroup"]["probabilities"],
        "outgroup_conf": a["outgroup"]["confidence"],
        "fusion_version": fusion_questions.VERSION,
    }


def loc_fields(a):
    return {
        "ts_score": a["loc_score"]["score"],
        "ts_score_probs": a["loc_score"]["probabilities"],
        "ts_score_conf": a["loc_score"]["confidence"],
        "ts_class": a["loc_class"]["choice"],
        "ts_class_probs": a["loc_class"]["probabilities"],
        "ts_class_conf": a["loc_class"]["confidence"],
        "ts_noul": a["agency"]["noul"],
    }


def score(row, prior):
    """Ask what is missing for this post. `prior` is a ledger row holding LoC and
    perhaps fusion answers from an older rubric version, or None. Returns the merged ledger row, or an error row."""
    sl.QUESTIONS = FUSION_Q if prior else BOTH_Q
    resp, dt = sl.ask(row["text"])
    if "error" in resp:
        return {"uri": row["uri"], "error": resp["error"]}
    a = resp["answers"]
    out = dict(prior) if prior else {}
    out.update({"uri": row["uri"], "latency_s": round(dt, 3)})
    if not prior:
        out.update(loc_fields(a))
        out["input_tokens"] = resp["usage"]["input_tokens"]
    out.update(fusion_fields(a))
    out["fusion_input_tokens"] = resp["usage"]["input_tokens"]
    out["model"] = out.get("model") or sl.MODEL
    out["fusion_model"] = sl.MODEL
    out["text_key"] = sl.text_key(row["text"])
    sl.ledger_append(out)
    return out


def load_posts(sources=SOURCES):
    posts = {}
    for src in sources:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok" and (r.get("text") or "").strip():
                posts.setdefault(r["uri"], r)
    return list(posts.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="pilot: this many posts, stratified by month")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--de-only", action="store_true", help="score time_sample_de.jsonl only")
    args = ap.parse_args()
    load_env()

    posts = load_posts(SOURCES[1:] if args.de_only else SOURCES)
    if args.limit:
        by_month = defaultdict(list)
        for p in posts:
            by_month[p["month"]].append(p)
        rng = random.Random(args.seed)
        per = max(1, args.limit // len(by_month))
        posts = [p for m in sorted(by_month) for p in rng.sample(by_month[m], per)]

    by_uri, by_text = sl.load_ledger()
    done = {}
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            if "error" not in r and has_fusion(r):
                done[r["uri"]] = r

    todo, served = [], 0
    for p in posts:
        if p["uri"] in done:
            continue
        hit = by_uri.get(p["uri"]) or by_text.get(sl.text_key(p["text"]))
        if has_fusion(hit):
            done[p["uri"]] = {**hit, "uri": p["uri"]}
            served += 1
            continue
        todo.append((p, hit))

    # One purchase per distinct text; twins get a copy.
    first, twins = {}, defaultdict(list)
    for p, hit in todo:
        k = sl.text_key(p["text"])
        if k in first:
            twins[k].append(p)
        else:
            first[k] = (p, hit)
    unique = list(first.values())
    print(f"{len(posts)} posts: {len(done) - served} in output, {served} from ledger, "
          f"{len(todo)} to score ({len(unique)} distinct texts, "
          f"{sum(1 for _, h in unique if h)} fusion-only), model={sl.MODEL}", flush=True)
    if unique and not sl.KEY:
        raise SystemExit("TYPESAFE_API_KEY unset; refusing to score.")

    t0, n, errors = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=sl.WORKERS) as pool:
        futs = {pool.submit(score, p, hit): p for p, hit in unique}
        for f in as_completed(futs):
            res = f.result()
            n += 1
            if "error" in res:
                errors += 1
                continue
            done[res["uri"]] = res
            for twin in twins.get(res["text_key"], []):
                copy = {**res, "uri": twin["uri"]}
                sl.ledger_append(copy)
                done[twin["uri"]] = copy
            if n % 1000 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(unique)}  {n/el:.1f}/s  errors={errors}  "
                      f"${sl._stats['input_tokens']/1e6*0.042:.2f}", flush=True)
                # Checkpoint: the ledger already holds every answer; this keeps the
                # output current for analysis of a run still in progress.
                write_out(done)
    write_out(done)

    wall = time.time() - t0
    print(f"done: {n} asked in {wall:.0f}s, errors={errors}, "
          f"input_tokens={sl._stats['input_tokens']:,} "
          f"cost=${sl._stats['input_tokens']/1e6*0.042:.2f}; {len(done)} rows -> {OUT}")


def write_out(done):
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)


if __name__ == "__main__":
    main()
