"""Rewrite TypeSafe System One scores into the teacher-annotation schema that
train_internal_vector.py reads, so the same Ridge pipeline can be run against a
TypeSafe teacher without touching the training code.

Row ORDER and row COUNT are preserved from the source corpus: the embedding
caches (embeddings_*_jina.npz) are positional, keyed to the order
load_train_rows() produces, so an adapter that reorders or drops rows silently
misaligns every target with its vector.

Two intensity modes, because the band target needs a 6-9 score for positives:
  flat   (default) omit loc_score -> get_positive_intensity() falls back to 6.0,
         exactly the geometry the bonsai teacher gets (it has no loc_score
         either). This is the controlled A/B: only the labels differ.
  graded loc_score = 6 + 3*P(internal), so the internal band keeps a gradient.
         Derived from the Choice primitive, not from TypeSafe's Score - the
         Score is the weak one (0.880 vs 0.944 AUC, REPORT_typesafe_loc.md).

Run: tmp_venv/bin/python build_typesafe_teacher.py SRC.jsonl TS_SCORES.jsonl OUT.jsonl [flat|graded]
"""

import json
import math
import sys

LABELS = ("internal", "external", "neutral")
FLOOR = 1e-6  # log(0) is not a logprob


def main():
    src, ts_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "flat"
    assert mode in ("flat", "graded"), mode

    ts = {}
    for line in open(ts_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "ts_class" in r:
                ts[r["uri"]] = r

    kept = missing = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            t = ts.get(row["uri"])
            if t is None:
                # Keep the row so the positional embedding cache stays aligned,
                # but strip raw_logprobs: load_train_rows() drops rows without
                # the positive label, and so did the cache builder.
                missing += 1
                row.pop("raw_logprobs", None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            p = {k: float(t["ts_class_probs"].get(k, 0.0)) for k in LABELS}
            new = {k: row[k] for k in ("did", "uri", "text", "createdAt", "langs", "time_us") if k in row}
            new["soft_labels"] = [p["internal"], p["external"], p["neutral"]]
            new["raw_logprobs"] = {k: math.log(max(p[k], FLOOR)) for k in LABELS}
            new["loc_label"] = t["ts_class"]
            new["ts_class_conf"] = t["ts_class_conf"]
            if mode == "graded" and t["ts_class"] == "internal":
                new["loc_score"] = 6.0 + 3.0 * p["internal"]
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{out}: {kept} rows from TypeSafe, {missing} passed through unscored (mode={mode})")


if __name__ == "__main__":
    main()
