"""Score feed candidates with jev on in-group norm signalling (ingroup_questions.py).

The LoC questions ride along in the same request for any post the shared ledger
(training/data/typesafe_ledger.jsonl) does not already hold with LoC, so the
answer serves later LoC runs too; a post it already holds is asked the in-group
questions alone. Merged rows are appended to the ledger, as in
research/time_sample/score_time_sample.py. Run bunny_feed/sync_typesafe_ledger.ts
first, so posts the edge prescore job bought are not bought again.

Run (TYPESAFE_API_KEY from .env):
  python research/ingroup_norm/score_ingroup.py <candidates.jsonl> [--limit N]
Output: research/data/ingroup_norm/scores_<candidates stem>.jsonl
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingroup_left_questions  # noqa: E402
import ingroup_questions  # noqa: E402
import score_loc_typesafe as sl  # noqa: E402
import ts_questions  # noqa: E402

# --rubric picks the module: `generic` lets jev infer the writer's side, `left`
# fixes the in-group as the feed's readers (left, anti-Trump, anti-AfD). They write
# disjoint fields, so both can live in one ledger row.
RUBRICS = {"generic": ingroup_questions, "left": ingroup_left_questions}
iq = ingroup_questions
IG_Q = BOTH_Q = None


def use_rubric(name):
    global iq, IG_Q, BOTH_Q
    iq = RUBRICS[name]
    IG_Q = iq.questions()
    BOTH_Q = {**ts_questions.questions(), **IG_Q}


def has_ingroup(row):
    return row is not None and row.get(iq.VERSION_FIELD) == iq.VERSION


def has_loc(row):
    return row is not None and all(k in row for k in sl.ANSWER_FIELDS)


def load_env():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    sl.KEY = os.environ.get("TYPESAFE_API_KEY", "")


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
    """Ask what is missing. `prior` is a ledger row holding LoC (and perhaps other
    studies' fields), or None."""
    with_loc = not has_loc(prior)
    resp, dt = sl.ask(row["text"], questions=BOTH_Q if with_loc else IG_Q)
    if "error" in resp:
        return {"uri": row["uri"], "error": resp["error"]}
    a = resp["answers"]
    out = dict(prior) if prior else {}
    out.update({"uri": row["uri"], "latency_s": round(dt, 3)})
    if with_loc:
        out.update(loc_fields(a))
        out["input_tokens"] = resp["usage"]["input_tokens"]
        out["model"] = sl.MODEL
    out.update(iq.fields(a))
    out[iq.VERSION_FIELD.replace("version", "input_tokens")] = resp["usage"]["input_tokens"]
    out[iq.VERSION_FIELD.replace("version", "model")] = sl.MODEL
    out["text_key"] = sl.text_key(row["text"])
    sl.ledger_append(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--limit", type=int, help="pilot: the first N posts")
    ap.add_argument("--rubric", choices=sorted(RUBRICS), default="generic")
    args = ap.parse_args()
    load_env()
    use_rubric(args.rubric)

    src = Path(args.src)
    suffix = "" if args.rubric == "generic" else "_" + args.rubric
    out_path = src.with_name("scores_" + src.stem + suffix + ".jsonl")
    posts = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    posts = [p for p in posts if (p.get("text") or "").strip()][: args.limit]

    by_uri, by_text = sl.load_ledger()
    done = {}
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            if "error" not in r and has_ingroup(r):
                done[r["uri"]] = r

    todo, served = [], 0
    for p in posts:
        if p["uri"] in done:
            continue
        hit = by_uri.get(p["uri"]) or by_text.get(sl.text_key(p["text"]))
        if has_ingroup(hit):
            done[p["uri"]] = {**hit, "uri": p["uri"]}
            served += 1
            continue
        todo.append((p, hit))

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
          f"{sum(1 for _, h in unique if has_loc(h))} with LoC already paid), model={sl.MODEL}",
          flush=True)
    if unique and not sl.KEY:
        raise SystemExit("TYPESAFE_API_KEY unset; refusing to score.")

    meta = {p["uri"]: {k: p.get(k) for k in ("pool", "generator_name", "text")} for p in posts}
    t0, errors = time.time(), 0
    with ThreadPoolExecutor(max_workers=sl.WORKERS) as pool:
        futs = [pool.submit(score, p, hit) for p, hit in unique]
        for n, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if "error" in res:
                errors += 1
                print("  error:", res["error"][:200], flush=True)
                continue
            done[res["uri"]] = res
            for twin in twins.get(res["text_key"], []):
                copy = {**res, "uri": twin["uri"]}
                sl.ledger_append(copy)
                done[twin["uri"]] = copy
            if n % 100 == 0:
                print(f"  {n}/{len(unique)}  {n / (time.time() - t0):.1f}/s  errors={errors}  "
                      f"${sl._stats['input_tokens'] / 1e6 * 0.042:.3f}", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in posts:
            if p["uri"] in done:
                f.write(json.dumps({**done[p["uri"]], **meta[p["uri"]]}, ensure_ascii=False) + "\n")
    print(f"done: {len(unique)} asked in {time.time() - t0:.0f}s, errors={errors}, "
          f"input_tokens={sl._stats['input_tokens']:,} "
          f"cost=${sl._stats['input_tokens'] / 1e6 * 0.042:.3f}; {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()
