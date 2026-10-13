"""Score a JSONL corpus on the LoC axis with a local Laya System One checkpoint.

Laya is the open-weight analogue of the TypeSafe `jev-latest` endpoint this
repository already pays for: the same three typed primitives (choice / score /
noul), the same "one state, N questions, one forward pass" contract, the same
answer shape. So `ts_questions.questions()` is handed to it verbatim and the
answers are unpacked into exactly the fields score_loc_typesafe.py writes --
ts_score / ts_class / ts_class_probs / ts_noul -- which is what lets
eval_laya_vs_jev.py treat the two scorers as interchangeable inputs.

No ledger and no spend: the weights are local, so a post may be asked twice.
The output file is still appended per row, because a 500-post MPS run is long
enough to be worth not repeating after a Ctrl-C.

Run: HF_HOME=/Volumes/Daten/huggingface PYTHONPATH=../../training \
     LAYA_MODEL=convaiinnovations/laya-multilingual LAYA_RUBRIC=full \
     ../../tmp_venv/bin/python score_loc_laya.py \
         dataset_validation_gemini.jsonl scores_loc_laya_multilingual_gold.jsonl
"""

import json
import os
import sys
import time

# Inputs live in training/data/ and outputs in research/data/evals/, both
# resolved against this file rather than the cwd -- the same rule the training
# scripts follow, and for the same reason: a bare relative default silently
# creates a second, empty file instead of failing.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN_DATA = os.path.join(ROOT, "training", "data")
OUT_DATA = os.path.join(ROOT, "research", "data", "evals")

sys.path.insert(0, os.path.join(ROOT, "training"))
import ts_questions  # noqa: E402
import loc_questions_compact  # noqa: E402

MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya-multilingual")
DEVICE = os.environ.get("LAYA_DEVICE")  # None -> cuda / mps / cpu, in that order
# `full` is ts_questions.py verbatim -- what a drop-in swap of the TypeSafe call
# would send, and what Laya's 48-token option cap then clips. `compact` is the
# same rubric rewritten to fit that cap. See loc_questions_compact.py.
RUBRIC = os.environ.get("LAYA_RUBRIC", "full")
QUESTIONS = (ts_questions if RUBRIC == "full" else loc_questions_compact).questions()


def resolve(path, default_dir):
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(default_dir, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = resolve(sys.argv[1], TRAIN_DATA)
    dst = resolve(sys.argv[2], OUT_DATA)

    rows = [r for r in load_jsonl(src) if r.get("text")]
    done = {r["uri"] for r in load_jsonl(dst)} if os.path.exists(dst) else set()
    todo = [r for r in rows if r["uri"] not in done]
    print(f"{src}: {len(rows)} posts, {len(done)} already scored, {len(todo)} to do")
    if not todo:
        return

    import laya
    t0 = time.time()
    agent = laya.load(MODEL, device=DEVICE)
    print(f"loaded {MODEL} on {agent.device} in {time.time() - t0:.1f}s")
    print(f"rubric={RUBRIC}, head_max_len={agent.cfg.get('head_max_len')}:")
    loc_questions_compact.check(agent, QUESTIONS)

    # The first call pays for kernel compilation and graph warm-up; timing it
    # with the rest would put ~2.5s of one-off cost into the p50.
    agent.predict(todo[0]["text"], QUESTIONS)

    t_run = time.time()
    with open(dst, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            t = time.time()
            resp = agent.predict(row["text"], QUESTIONS)
            dt = time.time() - t
            a = resp["answers"]
            f.write(json.dumps({
                "uri": row["uri"],
                "latency_s": round(dt, 3),
                "ts_score": a["loc_score"]["score"],
                "ts_score_probs": a["loc_score"]["probabilities"],
                "ts_score_conf": a["loc_score"]["confidence"],
                "ts_class": a["loc_class"]["choice"],
                "ts_class_probs": a["loc_class"]["probabilities"],
                "ts_class_conf": a["loc_class"]["confidence"],
                "ts_noul": a["agency"]["noul"],
                "input_tokens": resp["usage"]["input_tokens"],
                "model": MODEL,
                "rubric": RUBRIC,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t_run
                print(f"\r  {i}/{len(todo)}  {el / i:.2f}s/post  eta {(len(todo) - i) * el / i / 60:.1f}min",
                      end="", flush=True)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
