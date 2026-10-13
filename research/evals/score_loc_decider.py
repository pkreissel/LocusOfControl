"""Score a JSONL corpus on the LoC axis with Mapika/decider-2b.

Third open System One candidate, and the first one that is neither an encoder
with a decision head (laya) nor an inference trick over a stock chat model
(qwen_rlcd). decider-2b is Qwen3.5-2B-Base fully fine-tuned on 942k typed
decision examples, and it answers by reading the logits of the option letters at
one position -- one forward pass, no generation, a distribution rather than a
token. Same contract as the other two, so it writes the same four fields
(ts_score / ts_class / ts_class_probs / ts_noul) and eval_laya_vs_jev.py can
treat it as another column.

TWO RUBRICS, for the reason the Laya report needed two. The model card asks for
"short option phrases": the training distribution puts the semantics in the
question and a bare label in the option. ts_questions.py does the opposite --
its options are 49-93 token rubric paragraphs, because TypeSafe was built to
read them. Feeding those verbatim is the drop-in swap and it is worth measuring,
but judging the model only on that measures the mismatch, not the model.

  LOC_DECIDER_RUBRIC=short    labels as options, rubric in the question. What
                              the model was trained to see.
  LOC_DECIDER_RUBRIC=compact  loc_questions_compact.py's "label: description"
                              options, as Laya received them.
  LOC_DECIDER_RUBRIC=full     ts_questions.py verbatim. The literal swap.

The ordinal question is asked as digits 0-4 with the legend in the question, not
as band names: the letter readout needs candidates that differ, and it is the
same reason score_loc_qwen_rlcd.py could not use the v4 band names.

No ledger and no spend -- local weights, so a post may be asked twice. Output is
appended per row anyway, because a 500-post MPS run is long enough to be worth
not repeating after a Ctrl-C.

Run: HF_HOME=/Volumes/Daten/huggingface PYTHONPATH=../../training \
     LOC_DECIDER_RUBRIC=short ../../tmp_venv/bin/python score_loc_decider.py \
         dataset_validation_gemini.jsonl scores_loc_decider_short_gold.jsonl
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN_DATA = os.path.join(ROOT, "training", "data")
OUT_DATA = os.path.join(ROOT, "research", "data", "evals")

sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, HERE)
import loc_questions_compact as C  # noqa: E402
import ts_questions  # noqa: E402

MODEL = os.environ.get("LOC_DECIDER_MODEL", "Mapika/decider-2b")
RUBRIC = os.environ.get("LOC_DECIDER_RUBRIC", "short")
DEVICE = os.environ.get("LOC_DECIDER_DEVICE")
LIMIT = int(os.environ.get("LOC_DECIDER_LIMIT", "0"))
ADAPTER = os.environ.get("LOC_DECIDER_ADAPTER")  # LoRA dir from finetune_decider_loc.py
REVISION = os.environ.get("LOC_DECIDER_REVISION")  # pin the base weights; refs/main has moved once
LETTERS = "ABCDEFGHIJ"


def loc_questions(rubric):
    """[(qid, kind, question_text, [option strings])] in the order they are asked.

    kind is the primitive the answer is unpacked as, which is what keeps the
    output file interchangeable with the other scorers'.
    """
    if rubric in ("compact", "full"):
        src = C if rubric == "compact" else ts_questions
        qs = src.questions()
        levels = qs["loc_score"]["criteria"]
        return [
            ("loc_class", "choice", qs["loc_class"]["instructions"],
             [f"{k}: {v}" for k, v in qs["loc_class"]["criteria"].items()],
             list(qs["loc_class"]["criteria"])),
            ("loc_score", "score", qs["loc_score"]["instructions"],
             [f"level {i}: {c}" for i, c in enumerate(levels)], None),
            ("agency", "noul", qs["agency"]["instructions"] + " Is that true of this post?",
             ["no", "yes"], None),
        ]

    # short: the model card's shape -- bare labels, semantics in the question.
    cls = C.CLASS_CRITERIA
    class_q = (C.questions()["loc_class"]["instructions"] + "\n"
               + "\n".join(f"- {k} = {v}" for k, v in cls.items()))
    score_q = (C.INSTRUCTIONS + " Answer with the level that fits best.\n"
               + "\n".join(f"- {i} = {c}" for i, c in enumerate(C.LEVELS)))
    return [
        ("loc_class", "choice", class_q, list(cls), list(cls)),
        ("loc_score", "score", score_q, [str(i) for i in range(len(C.LEVELS))], None),
        ("agency", "noul", C.AGENCY_INSTRUCTIONS + " Is that true of this post?", ["no", "yes"], None),
    ]


def prompt_for(text, question, options):
    """The model card's format verbatim, ending on the open paren the letter
    follows. Anything else and the readout position is not the one it was
    trained to answer at."""
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return (f"Context:\n{text}\n\nQuestion: {question}\nOptions:\n{opts}\nAnswer: (")


def resolve(path, default_dir):
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(default_dir, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = resolve(sys.argv[1], TRAIN_DATA)
    dst = resolve(sys.argv[2], OUT_DATA)
    rows = [r for r in load_jsonl(src) if r.get("text")]
    done = {r["uri"] for r in load_jsonl(dst)} if os.path.exists(dst) else set()
    todo = [r for r in rows if r["uri"] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"{src}: {len(rows)} posts, {len(done)} already scored, {len(todo)} to do")
    if not todo:
        return

    if DEVICE:
        device = torch.device(DEVICE)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"  # the readout is the last position, so pad the front
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, revision=REVISION)
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER)
        model = model.merge_and_unload()  # fold LoRA in, so scoring costs what the base costs
    model = model.to(device).eval()
    # Mapika/decider-2b's weights were replaced under refs/main mid-project, which
    # silently split one eval across two base models. Record what was actually
    # loaded, and pin REVISION if you need a run to be reproducible.
    base_rev = getattr(model.config, "_commit_hash", None) or "unknown"
    print(f"loaded {MODEL}@{base_rev[:12]}{' + ' + ADAPTER if ADAPTER else ''} on {device} "
          f"({dtype}) in {time.time() - t0:.1f}s")

    questions = loc_questions(RUBRIC)
    print(f"rubric={RUBRIC}")
    for qid, kind, q, opts, _ in questions:
        print(f"  {qid:10} {kind:6} {len(opts)} options, "
              f"question {len(tok(q)['input_ids'])} tok, "
              f"options {[len(tok(o)['input_ids']) for o in opts]} tok")

    # Candidates must differ in their first token or the readout is comparing a
    # letter against itself; single capitals are one token each, but assert it.
    letter_ids = [tok.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(len(LETTERS))]
    assert len(set(letter_ids)) == len(letter_ids), f"option letters collide: {letter_ids}"

    @torch.no_grad()
    def ask(text):
        """All three questions in one padded batch, one forward pass."""
        prompts = [prompt_for(text, q, opts) for _, _, q, opts, _ in questions]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        # The three prompts differ by ~120 tokens, and a raw forward numbers
        # positions 0..L-1 regardless of padding -- which with left padding shifts
        # the short prompt's RoPE positions by the pad width. Deriving them from
        # the mask instead costs nothing and halves the disagreement with the
        # unpadded single-prompt answer (max |dp| 0.030 -> 0.015 over 36 fields).
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        # Only the last position is ever read, so ask for only that one. A full
        # forward materialises [3, L, 248320] and discards 99.8% of it: 21%
        # slower and 9.7 GB instead of 4.4 GB, for bit-identical answers
        # (verified 40 posts x 3 questions, max |dp| 0, no argmax flips).
        # finetune_decider_loc.py does the same, for the same reason.
        logits = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()
        out = {}
        for r, (qid, kind, _, opts, keys) in enumerate(questions):
            ids = letter_ids[:len(opts)]
            p = torch.softmax(logits[r, ids], -1).cpu().numpy()
            out[qid] = (kind, p, keys)
        return out, int(enc["attention_mask"].sum())

    ask(todo[0]["text"])  # warm-up: first call pays for kernel compilation

    t_run = time.time()
    with open(dst, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            t = time.time()
            try:
                ans, ntok = ask(row["text"])
            except Exception as e:  # keep the run alive; the eval skips error rows
                f.write(json.dumps({"uri": row["uri"], "error": str(e)[:300]}) + "\n")
                f.flush()
                continue
            dt = time.time() - t
            _, cls_p, cls_keys = ans["loc_class"]
            _, sc_p, _ = ans["loc_score"]
            _, nl_p, _ = ans["agency"]
            f.write(json.dumps({
                "uri": row["uri"],
                "latency_s": round(dt, 3),
                "ts_score": round(float(sum(j * v for j, v in enumerate(sc_p))), 4),
                "ts_score_probs": {str(j): round(float(v), 4) for j, v in enumerate(sc_p)},
                "ts_score_conf": round(float(max(sc_p)), 4),
                "ts_class": cls_keys[int(cls_p.argmax())],
                "ts_class_probs": {k: round(float(v), 4) for k, v in zip(cls_keys, cls_p)},
                "ts_class_conf": round(float(max(cls_p)), 4),
                "ts_noul": round(float(nl_p[1]), 4),
                "input_tokens": ntok,
                "model": MODEL,
                "base_revision": base_rev,
                "adapter": ADAPTER,
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
