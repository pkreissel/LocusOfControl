"""Join the cached locus-of-control predictions for one feed and compare them.

Reads whichever caches exist — there is no model loading and no network here, so
this step is free and re-runnable as the caches fill:

  --vector    the deployed vector's score (score_loc_vector.py). On the teacher's
              0-9 axis by construction: `scale * cos + bias` is the Ridge
              prediction the vector was fitted to produce.
  --reranker  the fine-tuned jina-reranker-v3 score (score_loc_reranker.py),
              read from its `rr_band` column, which is the same axis by isotonic
              calibration against the gold labels.
  --teacher   the teacher's graded score — optional, and the only non-circular
              judge here, since both students were distilled from it. Two schemas
              are read: TypeSafe rows (score_loc_typesafe.py, the current teacher)
              and the superseded local LLM's (annotate_lmstudio_soft_labels.py,
              LOC_PROMPT_VERSION=v4). Which one produced a number belongs in any
              report of it.

With a teacher column present, each student is scored against it (rank
correlation, internal-detection AUC, precision@k). Without one, the students can
only be compared to each other: where they agree, and — more useful — which
posts they disagree about most, since that is where a teacher run would buy the
most information.

Run:
  uv run --with numpy --with scipy --with scikit-learn python report_loc_feed.py \
      --posts feed_posts_<tag>.jsonl --vector scores_loc_vector_<tag>.jsonl \
      --reranker scores_loc_reranker_<tag>.jsonl --tag <tag>
"""

import argparse
import csv
import json
import os

import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

# train_internal_vector lives in training/, which is not this script's package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "training"))

os.environ.setdefault("LOC_PROMPT_VERSION", "v4")
INTERNAL_CUT = 5.5  # the trainer's split on [0,9]: internal >= 6


def load_jsonl(path):
    rows = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def index(path, key="uri"):
    return {r[key]: r for r in load_jsonl(path)}


# --- putting both students on one readable scale ---------------------------
#
# `vector_score` is genuinely the Ridge prediction on the teacher's 0-9 axis,
# but Ridge at l2=10 shrinks hard: over the 500 gold rows it spans 0.69 to 3.10
# and predicts >=6 (the internal band) for none of them, with class means
# 2.22 / 1.99 / 1.70 for internal / neutral / external. The reranker's head is on
# no scale at all (a listwise loss fixes order, not magnitude). So the two are
# projected onto the band axis the same way: least squares against the gold
# labels (external 0, neutral 2, internal 6) on the same 500 rows.
#
# Least squares and not isotonic, which was tried first: with three distinct
# target levels, isotonic returns a step function — 4 distinct values over 573
# feed posts, p5 through p95 all identical. That is fine as a magnitude and fatal
# as an ordering, and ranking by it put the reranker's Spearman at 0.098 against
# a true 0.186, purely through arbitrary tie-breaks. A positive-slope linear map
# is strictly monotone, so calibrated and raw agree on every ranking question.
# Ranking metrics below are computed on raw scores regardless.
BAND = {"external": 0.0, "neutral": 2.0, "internal": 6.0}
VAL_FILE = os.environ.get("LOC_VAL_FILE", "dataset_validation_gemini.jsonl")
VAL_EMB = os.environ.get("LOC_VAL_EMB_CACHE", "embeddings_val_loc_jina.npz")
RERANK_VAL_PRED = os.environ.get("LOC_RERANKER_VAL_PRED",
                                 "loc-reranker-jina-v3/val_predictions.npz")
TEACHER_CORPUS = os.environ.get("LOC_TEACHER_FILE", "dataset_annotated_lmstudio_soft.jsonl")


def gold_rows():
    """The gold rows in the order embeddings_val_loc_jina.npz was built in
    (eval_vector_teacher_by_lang.py owns that convention)."""
    seen = {r["uri"] for r in load_jsonl(TEACHER_CORPUS)}
    return [r for r in load_jsonl(VAL_FILE)
            if r.get("text") and r["uri"] in seen and r.get("loc_label") in BAND]


def fit_linear(x, y, name):
    """score -> band axis, least squares, clipped to [0,9] when applied.

    A negative slope would mean the model orders the gold set backwards; refuse
    rather than silently flip every number."""
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 0:
        print(f"Skipping {name} calibration: slope {slope:.4f} is not positive.")
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"{name} calibration on {len(x)} gold rows: band ~ {slope:.3f}*score "
          f"{intercept:+.3f} (r={r:.3f})")
    return lambda s: float(np.clip(slope * s + intercept, 0.0, 9.0))


def vector_calibration(vector_file):
    """Fitted on the gold set. None if the val embedding cache is missing or
    does not match this vector."""
    rows = gold_rows()
    if not rows or not os.path.exists(VAL_EMB):
        return None
    X = np.load(VAL_EMB)["X"]
    v = np.load(vector_file)
    if len(X) != len(rows) or X.shape[1] != len(v):
        print(f"Skipping vector calibration: {VAL_EMB} is {X.shape}, "
              f"need ({len(rows)}, {len(v)}).")
        return None
    meta = json.load(open(vector_file[:-4] + "_meta.json"))
    X = np.nan_to_num(X).astype(np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    s = X @ v * meta["scale"] + meta["bias"]
    return fit_linear(s, np.array([BAND[r["loc_label"]] for r in rows]), "vector")


def reranker_calibration():
    """Fitted on the reranker's own cached gold predictions, which must come from
    the same run as the scored model or the mapping is a lie."""
    rows = gold_rows()
    if not rows or not os.path.exists(RERANK_VAL_PRED):
        return None
    z = np.load(RERANK_VAL_PRED, allow_pickle=True)
    pred = {u: float(p) for u, p in zip(z["uris"], z["probs"])}
    rows = [r for r in rows if r["uri"] in pred]
    if len(rows) < 50:
        print(f"Skipping reranker calibration: only {len(rows)} gold rows covered.")
        return None
    return fit_linear(np.array([pred[r["uri"]] for r in rows]),
                      np.array([BAND[r["loc_label"]] for r in rows]), "reranker")


def precision_at_k(s, flag, k):
    k = min(k, len(s))
    return float(flag[np.argsort(-s)[:k]].mean()) if k else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default="feed_posts.jsonl")
    ap.add_argument("--vector", default="scores_loc_vector.jsonl")
    ap.add_argument("--reranker", default="")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--tag", default=os.environ.get("TAG", "feed"))
    ap.add_argument("--out-dir", default="",
                    help="where the csv/json/md land; defaults beside --posts, "
                         "which is research/data/feed/ for every real run")
    ap.add_argument("--threshold", type=float,
                    default=float(os.environ.get("LOC_FEED_THRESHOLD", "0.88")),
                    help="deployed probability gate for this feed (feeds.ts)")
    args = ap.parse_args()

    posts = index(args.posts)
    vector = index(args.vector)
    reranker = index(args.reranker)
    teacher = index(args.teacher)
    print(f"{len(posts)} posts | vector {len(vector)} | reranker {len(reranker)} | "
          f"teacher {len(teacher)}")
    if not vector:
        raise SystemExit("No vector scores cached yet.")

    # The label temperature the vector was calibrated at, so a teacher column is
    # read the same way the trainer read it.
    vfile = next(iter(vector.values())).get("vector_file", "internal_vector.npy")
    meta = json.load(open(vfile[:-4] + "_meta.json" if vfile.endswith(".npy") else vfile))
    temperature = meta.get("temperature", 1.0)
    vcal = vector_calibration(vfile)
    rcal = reranker_calibration() if reranker else None
    # Only the local-LLM schema needs the trainer's accessors; a TypeSafe file is
    # read directly below, so a run with one does not depend on training/ at all.
    T = None
    if teacher and any("ts_class" not in r for r in teacher.values()):
        import train_internal_vector as T

    rows = []
    for uri, p in posts.items():
        v, rr, t = vector.get(uri), reranker.get(uri), teacher.get(uri)
        if not v:
            continue
        row = {
            "uri": uri,
            "handle": p.get("handle"),
            "created_at": p.get("created_at"),
            "draw_rank": p.get("draw_rank"),
            "like_count": p.get("like_count"),
            "langs": ",".join(p.get("langs") or []),
            "text_len": p.get("text_len"),
            "enough_text": p.get("enough_text"),
            "vector_score": float(v["vector_score"]),
            "vector_cos": float(v["vector_cos"]),
            "vector_prob": float(v["vector_prob"]),
            "vector_version": v.get("vector_version"),
        }
        if vcal is not None:
            row["vector_band"] = vcal(row["vector_score"])
        if rr:
            row["rr_logit"] = float(rr["rr_logit"])
            if rcal is not None:
                row["rr_band"] = rcal(row["rr_logit"])
        if t and "ts_class" in t:
            # TypeSafe carries its own schema (score_loc_typesafe.py), mapped here
            # onto the same 0-9 axis rather than round-tripped through the training
            # adapter. ts_score is a mean over the five levels of
            # ts_questions.LOC_LEVELS, and those are v4's 0-9 bands taken in pairs,
            # so *2 recovers the band and +0.5 lands mid-band, not on its floor.
            p_int = float(t["ts_class_probs"].get("internal", 0.0))
            row["teacher_score"] = 2.0 * float(t["ts_score"]) + 0.5
            row["teacher_label"] = t["ts_class"]
            # The band geometry build_typesafe_teacher.py hands the trainer:
            # external floored, neutral at the neutral point, internal graded by
            # P(internal) across the 6-9 band. Kept identical on purpose — a report
            # that scored the teacher differently from the trainer would be
            # measuring the adapter.
            row["teacher_band"] = (6.0 + 3.0 * p_int if t["ts_class"] == "internal"
                                   else 0.0 if t["ts_class"] == "external" else 2.0)
            row["teacher_level"] = None
        elif t:
            s = T.get_expected_score(t, temperature)
            row["teacher_score"] = float(s if s is not None else t.get("loc_score"))
            row["teacher_band"] = float(T.get_target_score(t, temperature))
            row["teacher_label"] = T.get_class(t, temperature)
            row["teacher_level"] = t.get("loc_level")
        row["text"] = (p.get("text") or "").replace("\n", " ")
        rows.append(row)
    rows.sort(key=lambda r: -r["vector_score"])
    print(f"{len(rows)} posts in the joined table.")

    # --- predictors on the shared 0-9 axis ---------------------------------
    # Raw model outputs: every ranking number below is computed on these, so no
    # calibration choice can move a rank.
    calibrated = all("vector_band" in r for r in rows)
    preds = {"vector": np.array([r["vector_score"] for r in rows])}
    if reranker and all("rr_logit" in r for r in rows):
        preds["reranker"] = np.array([r["rr_logit"] for r in rows])
    elif reranker:
        print(f"Reranker covers {sum('rr_logit' in r for r in rows)}/{len(rows)} rows; "
              "excluded from the comparison until it covers all of them.")
    enough = np.array([bool(r["enough_text"]) for r in rows])
    vp = np.array([r["vector_prob"] for r in rows])

    report = {
        "tag": args.tag,
        "n": len(rows),
        "n_feed_eligible": int(enough.sum()),
        "vector_version": rows[0]["vector_version"],
        "label_temperature": temperature,
        "distributions": {
            name: {"mean": float(s.mean()),
                   "percentiles": {str(q): float(np.percentile(s, q))
                                   for q in (5, 25, 50, 75, 95)}}
            for name, s in preds.items()
        },
    }
    report["calibrated"] = calibrated
    for col in ("vector_band", "rr_band"):
        vals = np.array([r[col] for r in rows if col in r])
        if len(vals) == len(rows):
            report["distributions"][col] = {
                "mean": float(vals.mean()),
                "percentiles": {str(q): float(np.percentile(vals, q))
                                for q in (5, 25, 50, 75, 95)},
            }
    report["distributions"]["vector_prob"] = {
        "mean": float(vp.mean()),
        "percentiles": {str(q): float(np.percentile(vp, q)) for q in (5, 25, 50, 75, 95)},
    }

    # --- against the teacher, when there is one ---------------------------
    if all("teacher_score" in r for r in rows) and teacher:
        ts = np.array([r["teacher_score"] for r in rows])
        tb = np.array([r["teacher_band"] for r in rows])
        yi = (ts >= INTERNAL_CUT).astype(int)
        ext = np.array([r.get("teacher_label") == "external" for r in rows]).astype(int)
        report["teacher"] = {
            "n": len(ts), "n_internal": int(yi.sum()),
            "mean_score": float(ts.mean()),
            "label_counts": {l: int(sum(1 for r in rows if r.get("teacher_label") == l))
                             for l in ("internal", "external", "neutral")},
        }
        report["vs_teacher"] = {}
        for name, s in preds.items():
            m = {
                "spearman_vs_loc_score": float(spearmanr(s, ts).statistic),
                "pearson_vs_loc_score": float(pearsonr(s, ts).statistic),
                "spearman_vs_band_target": float(spearmanr(s, tb).statistic),
                "ndcg_band_target": float(ndcg_score(tb[None, :], s[None, :])),
                "mean_abs_err_vs_loc_score": float(np.abs(s - ts).mean()),
            }
            if 0 < yi.sum() < len(yi):
                m["auc_internal"] = float(roc_auc_score(yi, s))
                m["avg_precision_internal"] = float(average_precision_score(yi, s))
            if 0 < ext.sum() < len(ext):
                m["ext_leak_auc"] = float(roc_auc_score(ext, s))
            for k in (10, 25, 50):
                m[f"precision@{k}"] = precision_at_k(s, yi, k)
            report["vs_teacher"][name] = m

    # --- students against each other --------------------------------------
    if len(preds) > 1:
        a, b = preds["vector"], preds["reranker"]
        # Equal-volume comparison: the 0.88 gate fixes how many posts the vector
        # admits, so take the reranker's top-N for the same N. Comparing two
        # rankings at different lengths says nothing.
        n_keep = int((vp >= args.threshold).sum())
        top_a = set(np.argsort(-a)[:n_keep])
        top_b = set(np.argsort(-b)[:n_keep])
        report["students"] = {
            "spearman": float(spearmanr(a, b).statistic),
            "pearson": float(pearsonr(a, b).statistic),
            "gate_keeps": n_keep,
            "gate_share": float(n_keep / len(rows)),
            "top_n_overlap": len(top_a & top_b),
            "top_n_jaccard": len(top_a & top_b) / max(len(top_a | top_b), 1),
        }
        ra = np.argsort(np.argsort(-a))
        rb = np.argsort(np.argsort(-b))
        for i, r in enumerate(rows):
            r["rank_vector"], r["rank_reranker"] = int(ra[i]) + 1, int(rb[i]) + 1
            r["rank_gap"] = int(rb[i]) - int(ra[i])

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.posts))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"loc_scores_{args.tag}.csv")
    cols = [c for c in rows[0] if c != "text"] + ["text"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json_path = os.path.join(out_dir, f"REPORT_loc_feed_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- markdown ---------------------------------------------------------
    def table(sorted_rows, n, title, cols_):
        out = [f"### {title}\n", "| " + " | ".join(cols_ + ["text"]) + " |",
               "|" + "---|" * (len(cols_) + 1)]
        for r in sorted_rows[:n]:
            vals = [f"{r[c]:.2f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols_]
            out.append("| " + " | ".join(vals) + " | " + r["text"][:80].replace("|", "¦") + " |")
        return "\n".join(out) + "\n"

    md = [f"# Locus of control on `{args.tag}`\n",
          f"{len(rows)} posts, {int(enough.sum())} of them past the feed's own "
          f"minWords gate. Every score below is cached; this report recomputes nothing.\n",
          f"- vector: version `{rows[0]['vector_version']}`, `scale*cos+bias` — its Ridge "
          f"prediction on the teacher's 0-9 axis"
          + (" — and `vector_band`, that score projected onto the gold labels' band axis, "
             "which is how it becomes comparable to the reranker at all (Ridge shrinkage "
             "keeps the raw prediction inside [0.7, 3.3])" if calibrated else ""),
          ]
    if "reranker" in preds:
        md.append("- reranker: `loc-reranker-jina-v3`, raw `rr_logit` (a ranking score, on no "
                  "scale of its own) plus `rr_band`, projected onto the same band axis by the "
                  "same least-squares fit on the same gold rows")
    if report.get("teacher"):
        if any("ts_class" in r for r in teacher.values()):
            md.append(f"- teacher: TypeSafe System One (`{os.environ.get('TYPESAFE_MODEL', 'jev-latest')}`), "
                      "`ts_score` mapped onto the 0-9 axis (2*level + 0.5) and `ts_class` as the label. "
                      "This is the teacher the deployed vector was distilled from, so read the numbers "
                      "below as distillation fidelity, not as independent accuracy")
        else:
            md.append(f"- teacher: local v4 LLM, graded loc_score at label temperature {temperature}")
    md += ["\n## Distributions\n",
           "Raw model outputs first — those are what every ranking number here is computed "
           "on — then the same scores on the shared band axis.\n",
           "| | p5 | p25 | p50 | p75 | p95 | mean |", "|---|---|---|---|---|---|---|"]
    for name, d in report["distributions"].items():
        q = d["percentiles"]
        md.append(f"| {name} | " + " | ".join(f"{q[str(x)]:.3f}" for x in (5, 25, 50, 75, 95))
                  + f" | {d['mean']:.3f} |")

    if report.get("vs_teacher"):
        t = report["teacher"]
        md += [f"\n## Against the teacher ({t['n_internal']} of {t['n']} internal)\n",
               "| predictor | Spearman (loc_score) | Spearman (band) | AUC internal | "
               "AP | P@25 | ext-leak AUC |", "|---|---|---|---|---|---|---|"]
        for name, m in report["vs_teacher"].items():
            md.append(f"| {name} | {m['spearman_vs_loc_score']:.3f} | "
                      f"{m['spearman_vs_band_target']:.3f} | "
                      f"{m.get('auc_internal', float('nan')):.3f} | "
                      f"{m.get('avg_precision_internal', float('nan')):.3f} | "
                      f"{m['precision@25']:.2f} | {m.get('ext_leak_auc', float('nan')):.3f} |")

    if report.get("students"):
        s = report["students"]
        md += [f"\n## Vector vs reranker\n",
               f"- Spearman {s['spearman']:.3f} over all {len(rows)} posts",
               f"- the deployed {args.threshold} gate admits {s['gate_keeps']} posts "
               f"({s['gate_share']*100:.0f}%); at that same length the two rankings share "
               f"{s['top_n_overlap']} of them (Jaccard {s['top_n_jaccard']:.2f})",
               "\nWhere they disagree most is where a teacher run would be worth the "
               "hours — these are the posts the two models cannot both be right about.\n"]
        gap = sorted(rows, key=lambda r: -r["rank_gap"])
        cols_ = ["rank_vector", "rank_reranker", "vector_score", "rr_logit"]
        md.append(table(gap, 10, "Vector ranks far above the reranker", cols_))
        md.append(table(gap[::-1], 10, "Reranker ranks far above the vector", cols_))

    cols_ = ["vector_score", "vector_prob"] + (["vector_band"] if calibrated else [])
    cols_ += ["rr_logit"] + (["rr_band"] if any("rr_band" in r for r in rows) else []) \
        if "reranker" in preds else []
    cols_ += ["teacher_score"] if report.get("teacher") else []
    md.append(table(rows, 15, "Top 15 by vector score", cols_))
    if "reranker" in preds:
        md.append(table(sorted(rows, key=lambda r: -r["rr_logit"]), 15,
                        "Top 15 by reranker score", cols_))

    md_path = os.path.join(out_dir, f"REPORT_loc_feed_{args.tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nWrote {csv_path}, {json_path}, {md_path}")
    if report.get("students"):
        s = report["students"]
        print(f"  vector vs reranker: spearman {s['spearman']:.3f}, "
              f"top-{s['gate_keeps']} overlap {s['top_n_overlap']}")


if __name__ == "__main__":
    main()
