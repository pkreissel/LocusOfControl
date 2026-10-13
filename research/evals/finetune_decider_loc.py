"""LoRA-distil the `jev-latest` LoC teacher into Mapika/decider-2b.

decider-2b is the strongest open candidate measured zero-shot (AP 0.589 against
the deployed Jina vector's 0.523, REPORT_laya_finetune.md) and the only one that
was never trained on this axis at all. It answers by reading the option-letter
logits at a single position, so fine-tuning it means training exactly that
readout -- and nothing else.

SAME DATA, SAME SPLIT, SAME TARGETS AS THE LAYA RUN. `build_pairs`, `split` and
`teacher_target` are imported from finetune_laya_loc.py rather than reimplemented,
so both candidates see identical rows with identical soft targets and a
difference in the result is a difference in model. That also inherits the guard
that matters: all 500 Gemini gold posts are inside the 5500-post corpus and are
dropped before the split, with an assert.

Soft targets throughout -- ts_class_probs / ts_score_probs / ts_noul -- under
`laya.common.proper_reward`, the same strictly proper scoring rule (log +
spherical, plus a ranked-probability term on the ordinal question). Reusing that
loss across two very different architectures is deliberate: it is the one thing
held fixed between them.

THE LOSS IS RESTRICTED TO THE CANDIDATE LETTERS. Inference computes
`softmax(logits[-1][letter_ids])` over K letters, so that is what is trained.
Full-vocabulary cross-entropy on the correct letter would be a different
objective -- 248k vocabulary, most of the gradient spent suppressing tokens the
readout never looks at -- and would not match how the answer is actually read.

WHY LoRA. 1.9B parameters under AdamW is ~30 GB of optimiser state before any
activations; it does not fit, and 4500 posts against 1.9B parameters is the wrong
capacity anyway. r=16 on `all-linear` is ~1% trainable and the rank cap is itself
the regulariser. `all-linear` rather than named projections because the model is
hybrid -- 18 linear-attention blocks and 6 full-attention blocks, whose modules
are not named alike -- so a hand-written target list would silently skip most of
the network.

LETTER PERMUTATION is the augmentation that matters here. decider's training mix
gives (A) a prior, and nothing else in this setup would remove it. Applied to
`choice` only: permuting the ordinal question would destroy the reading that both
the RPS term and the runtime's `sum(i * p_i)` depend on.

LEFT PADDING NEEDS EXPLICIT POSITIONS. The readout is the last token, so prompts
are left-padded; a raw forward then numbers positions 0..L-1 regardless of the
padding and shifts every short prompt's RoPE positions by the pad width. Position
ids are derived from the attention mask instead, matching score_loc_decider.py.

MEMORY. Same guards as the Laya trainer, for the same reason: on unified memory
an over-large batch pages the machine instead of failing. The process is capped,
the worst-case batch is measured before training starts, and sequence length is
quantised so the MPS allocator is not asked for a new buffer size every step.

Run:
    HF_HOME=/Volumes/Daten/huggingface PYTHONPATH=../../training \
    ../../tmp_venv/bin/python finetune_decider_loc.py

    DEC_FT_MEM_PROBE_ONLY=1   measure the worst batch and the step rate, train nothing
"""

import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_DATA = os.path.join(ROOT, "research", "data", "laya_loc")

sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, HERE)

import finetune_laya_loc as L          # noqa: E402  data, split, targets, EMA, memory guards
import score_loc_decider as D          # noqa: E402  prompt format and question set


def env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


MODEL = env("DEC_FT_MODEL", "Mapika/decider-2b")
RUBRIC = env("DEC_FT_RUBRIC", "short")
RUN = env("DEC_FT_RUN", "decider_short")
OUT_DIR = env("DEC_FT_OUT", os.path.join(OUT_DATA, RUN))
SEED = env("DEC_FT_SEED", 20260919, int)

EPOCHS = env("DEC_FT_EPOCHS", 2, int)
BATCH = env("DEC_FT_BATCH", 4, int)
ACCUM = env("DEC_FT_ACCUM", 4, int)
LR = env("DEC_FT_LR", 1e-4, float)
WD = env("DEC_FT_WD", 0.01, float)
WARMUP = env("DEC_FT_WARMUP", 0.1, float)
CLIP = env("DEC_FT_CLIP", 1.0, float)
LORA_R = env("DEC_FT_LORA_R", 16, int)
LORA_ALPHA = env("DEC_FT_LORA_ALPHA", 32, int)
LORA_DROPOUT = env("DEC_FT_LORA_DROPOUT", 0.05, float)
EMA_DECAY = env("DEC_FT_EMA", 0.999, float)
SHUFFLE_OPTS = env("DEC_FT_SHUFFLE", 1, int)
EVALS_PER_EPOCH = env("DEC_FT_EVALS_PER_EPOCH", 4, int)
PATIENCE = env("DEC_FT_PATIENCE", 3, int)
SELECT = env("DEC_FT_SELECT", "loss")
MAX_LEN = env("DEC_FT_MAX_LEN", 768, int)
PAD_MULTIPLE = env("DEC_FT_PAD_MULTIPLE", 64, int)
GRAD_CKPT = env("DEC_FT_GRAD_CKPT", 1, int)
MEM_FRACTION = env("DEC_FT_MEM_FRACTION", 0.6, float)
MEM_PROBE_STEPS = env("DEC_FT_MEM_PROBE", 3, int)
MEM_PROBE_ONLY = env("DEC_FT_MEM_PROBE_ONLY", 0, int)
EMPTY_EVERY = env("DEC_FT_EMPTY_CACHE", 50, int)

QUESTIONS = D.loc_questions(RUBRIC)          # [(qid, kind, question, options, keys)]
QKIND = {q[0]: q[1] for q in QUESTIONS}


def as_laya_question(kind, options, keys):
    """The shape finetune_laya_loc.teacher_target expects, so the soft targets are
    computed by the same code for both models rather than by two copies of it."""
    if kind == "choice":
        return {"type": "choice", "criteria": {k: None for k in keys}}
    if kind == "score":
        return {"type": "score", "criteria": list(options)}
    return {"type": "noul"}


def make_items(tok, pairs, index, rng=None):
    """One item per (post, question): token ids, the letter count, the soft target."""
    items = []
    for i in index:
        text, s, cls = pairs[i]
        for qid, kind, question, options, keys in QUESTIONS:
            tgt = L.teacher_target(qid, as_laya_question(kind, options, keys), s)
            k = len(tgt)
            order = list(range(k))
            if rng is not None and SHUFFLE_OPTS and kind == "choice":
                rng.shuffle(order)
            opts = [options[o] for o in order]
            ids = tok(D.prompt_for(text, question, opts), add_special_tokens=True)["input_ids"]
            if len(ids) > MAX_LEN:
                # Truncate the post, never the tail: the prompt must still end on
                # "Answer: (" or the readout position is not the trained one.
                head = tok(D.prompt_for("", question, opts), add_special_tokens=True)["input_ids"]
                keep = MAX_LEN - len(head)
                body = tok(text, add_special_tokens=False)["input_ids"][:max(0, keep)]
                ids = tok(D.prompt_for(tok.decode(body), question, opts),
                          add_special_tokens=True)["input_ids"][:MAX_LEN]
            items.append({"ids": ids, "k": k, "qid": qid, "kind": kind,
                          "target": [tgt[o] for o in order], "post": i, "cls": cls})
    return items


def collate(items, pad_id, kmax_letters):
    """Left-padded batch plus mask-derived positions; targets padded to kmax."""
    n = len(items)
    L_ = max(len(it["ids"]) for it in items)
    if PAD_MULTIPLE > 0:
        L_ = -(-L_ // PAD_MULTIPLE) * PAD_MULTIPLE
    ids = torch.full((n, L_), pad_id, dtype=torch.long)
    att = torch.zeros((n, L_), dtype=torch.long)
    target = torch.zeros((n, kmax_letters), dtype=torch.float32)
    mask = torch.zeros((n, kmax_letters), dtype=torch.bool)
    qtype = torch.zeros(n, dtype=torch.long)
    for i, it in enumerate(items):
        m = len(it["ids"])
        ids[i, L_ - m:] = torch.tensor(it["ids"])   # left pad: last column is the readout
        att[i, L_ - m:] = 1
        k = it["k"]
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
        mask[i, :k] = True
        qtype[i] = {"choice": 0, "score": 1, "noul": 2}[it["kind"]]
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": ids, "attention_mask": att, "position_ids": pos,
            "target": target, "marker_mask": mask, "qtype": qtype}


def letter_logits(model, batch, device, letter_ids):
    # logits_to_keep=1 runs lm_head on the final position only. The answer is read
    # from that position and nowhere else, so computing the other ~450 is pure
    # waste -- and with a 248k vocabulary it is expensive waste: the full tensor is
    # [4, 448, 248320], 1.78 GB in fp32, which is what put this over the memory cap
    # before activations were even counted.
    out = model(input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                position_ids=batch["position_ids"].to(device),
                logits_to_keep=1)
    z = out.logits[:, -1, :].float()
    k = batch["marker_mask"].shape[1]            # letters actually in play, not all ten
    sel = z[:, letter_ids[:k]]                   # [n, kmax]
    return sel.masked_fill(~batch["marker_mask"].to(device), -1e4)


def loss_of(logits, batch, device, weights=None):
    from laya.common import proper_reward
    q = torch.softmax(logits, -1)
    r = proper_reward(q, batch["target"].to(device), batch["qtype"].to(device),
                      batch["marker_mask"].to(device))
    if weights is not None:
        return -(r * weights.to(device)).mean()
    return -r.mean()


@torch.no_grad()
def evaluate(model, items, tok, device, pairs, letter_ids, kmax):
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    tot, n, p_int, p_score, y = 0.0, 0, [], [], []
    keys = [q for q in QUESTIONS if q[0] == "loc_class"][0][4]
    for b in L.batches(items, BATCH):
        batch = collate(b, tok.pad_token_id, kmax)
        lg = letter_logits(model, batch, device, letter_ids)
        tot += float(loss_of(lg, batch, device)) * len(b)
        n += len(b)
        p = torch.softmax(lg, -1).float().cpu().numpy()
        for j, it in enumerate(b):
            k = it["k"]
            if it["qid"] == "loc_class":
                p_int.append(float(p[j, :k][keys.index("internal")]))
                y.append(1 if pairs[it["post"]][2] == "internal" else 0)
            elif it["qid"] == "loc_score":
                p_score.append(float((np.arange(k) * p[j, :k]).sum()))
    y = np.array(y)
    out = {"loss": tot / max(1, n)}
    if y.sum() >= 3:
        out["ap_choice"] = float(average_precision_score(y, p_int))
        out["auc_choice"] = float(roc_auc_score(y, p_int))
        out["ap_score"] = float(average_precision_score(y, p_score))
    return out


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = random.Random(SEED)

    pairs = L.build_pairs()
    tr_idx, dv_idx = L.split(pairs, rng)
    print(f"split: {len(tr_idx)} train / {len(dv_idx)} dev posts "
          f"(train {Counter(pairs[i][2] for i in tr_idx)}, dev {Counter(pairs[i][2] for i in dv_idx)})")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device)
    print(f"loaded {MODEL} on {device} ({dtype}) in {time.time() - t0:.1f}s")

    if GRAD_CKPT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print("  gradient checkpointing enabled")
    peft_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, peft_cfg)
    # The base stays bf16 -- 1.9B params in fp32 would not fit -- but the adapter
    # trains in fp32. bf16 carries ~3 decimal digits, and an Adam step at lr 1e-4
    # on a bf16 weight rounds away to nothing. peft casts the input to the adapter
    # dtype and the output back, so mixing the two is supported.
    for _n, _p in model.named_parameters():
        if _p.requires_grad and _p.dtype != torch.float32:
            _p.data = _p.data.float()
    model.print_trainable_parameters()
    print(f"  adapter dtype {next(p.dtype for p in model.parameters() if p.requires_grad)}, "
          f"base dtype {dtype}")

    letter_ids = [tok.encode(D.LETTERS[i], add_special_tokens=False)[0] for i in range(10)]
    assert len(set(letter_ids)) == len(letter_ids), "option letters collide"
    kmax = max(len(q[3]) for q in QUESTIONS)

    tr_items = make_items(tok, pairs, tr_idx, rng=rng)
    dv_items = make_items(tok, pairs, dv_idx)
    w = L.class_weights(pairs, tr_idx)
    lens = [len(it["ids"]) for it in tr_items]
    print(f"{len(tr_items)} train / {len(dv_items)} dev items; "
          f"prompt tokens p50/p95/max {int(np.percentile(lens,50))}/"
          f"{int(np.percentile(lens,95))}/{max(lens)}; class weights "
          + ", ".join(f"{c}={v:.2f}" for c, v in sorted(w.items())))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD, betas=(0.9, 0.98), eps=1e-6)
    steps_per_epoch = math.ceil(len(tr_items) / BATCH / ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    eval_every = max(1, steps_per_epoch // max(1, EVALS_PER_EPOCH))

    # ---- memory and speed, before anything is trained -----------------------
    # memory_cap reads the module-level fraction of whichever trainer defined it,
    # so point it at this run's setting before asking for the cap.
    L.MEM_FRACTION = MEM_FRACTION
    cap = L.memory_cap(device) if device.type != "cpu" else None
    if cap is not None:
        L.empty_cache(device)
        worst = sorted(tr_items, key=lambda it: -len(it["ids"]))[:BATCH]
        model.train()
        t = time.time()
        try:
            for _ in range(max(1, MEM_PROBE_STEPS)):
                batch = collate(worst, tok.pad_token_id, kmax)
                wts = torch.tensor([w[it["cls"]] for it in worst], dtype=torch.float32)
                loss_of(letter_logits(model, batch, device, letter_ids), batch, device, wts).backward()
                torch.nn.utils.clip_grad_norm_(params, CLIP)
                opt.step(); opt.zero_grad(set_to_none=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            opt.zero_grad(set_to_none=True); L.empty_cache(device)
            raise SystemExit(
                f"\nWILL NOT FIT: worst-case batch ({BATCH} x "
                f"{collate(worst, tok.pad_token_id, kmax)['input_ids'].shape[1]} tokens) "
                f"exceeds the {cap / 2**30:.1f} GB cap. Nothing was trained.\n"
                f"Reduce DEC_FT_BATCH (raise DEC_FT_ACCUM to match), lower DEC_FT_MAX_LEN, "
                f"or raise DEC_FT_MEM_FRACTION if the machine is idle.")
        dt = (time.time() - t) / max(1, MEM_PROBE_STEPS)
        peak = L.peak_memory(device)
        # The worst batch bounds memory but overstates the clock: it is the longest
        # prompts in the corpus, and the median is half that. Time a median-length
        # batch too, because the epoch estimate is what decides whether to run.
        mid = sorted(tr_items, key=lambda it: len(it["ids"]))
        mid = mid[len(mid) // 2: len(mid) // 2 + BATCH]
        t = time.time()
        for _ in range(max(1, MEM_PROBE_STEPS)):
            batch = collate(mid, tok.pad_token_id, kmax)
            wts = torch.tensor([w[it["cls"]] for it in mid], dtype=torch.float32)
            loss_of(letter_logits(model, batch, device, letter_ids), batch, device, wts).backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step(); opt.zero_grad(set_to_none=True)
        dt_mid = (time.time() - t) / max(1, MEM_PROBE_STEPS)
        print(f"  worst-case batch: {BATCH} x {collate(worst, tok.pad_token_id, kmax)['input_ids'].shape[1]}"
              f" tokens -> peak {peak / 2**30:.2f} GB of {cap / 2**30:.1f} GB cap ({peak / cap:.0%})")
        print(f"  {dt:.2f}s worst / {dt_mid:.2f}s median micro-batch -> "
              f"~{dt_mid * len(tr_items) / BATCH / 60:.0f} min/epoch, "
              f"~{dt_mid * len(tr_items) / BATCH * EPOCHS / 3600:.1f} h for {EPOCHS} epochs")
        if peak > 0.85 * cap:
            raise SystemExit("ABORTED: too little headroom; see the settings above.")
        # roll the probe back -- LoRA weights only, so zeroing them restores the
        # base model exactly without reloading 3.8 GB from disk
        for n_, p_ in model.named_parameters():
            if p_.requires_grad and "lora_B" in n_:
                torch.nn.init.zeros_(p_)
            elif p_.requires_grad and "lora_A" in n_:
                torch.nn.init.kaiming_uniform_(p_, a=math.sqrt(5))
        opt.state.clear()
        L.empty_cache(device)
        if MEM_PROBE_ONLY:
            print("\nprobe only, nothing trained")
            return

    ema = L.EMA(model, EMA_DECAY) if EMA_DECAY > 0 else None
    base_lr = LR
    print(f"{EPOCHS} epochs x {steps_per_epoch} steps, eval every {eval_every}, "
          f"select on dev {SELECT}, patience {PATIENCE}")
    base = evaluate(model, dv_items, tok, device, pairs, letter_ids, kmax)
    print("  dev @0: " + "  ".join(f"{k}={v:.4f}" for k, v in base.items()) + "   (zero-shot)")

    best = {"score": None, "state": None, "step": 0, "metrics": base}
    bad, step, stop = 0, 0, False
    for epoch in range(EPOCHS):
        if stop:
            break
        model.train()
        t_ep, seen, run = time.time(), 0, 0.0
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(L.batches(tr_items, BATCH, rng)):
            batch = collate(b, tok.pad_token_id, kmax)
            wts = torch.tensor([w[it["cls"]] for it in b], dtype=torch.float32)
            loss = loss_of(letter_logits(model, batch, device, letter_ids), batch, device, wts) / ACCUM
            loss.backward()
            run += float(loss.detach()) * ACCUM; seen += 1
            if (bi + 1) % ACCUM:
                continue
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            for g in opt.param_groups:
                g["lr"] = base_lr * L.lr_lambda(step, total_steps)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if ema:
                ema.update(model)
            if EMPTY_EVERY and step % EMPTY_EVERY == 0:
                L.empty_cache(device)

            if step % eval_every == 0 or step == total_steps:
                backup = ema.swap_in(model) if ema else None
                m = evaluate(model, dv_items, tok, device, pairs, letter_ids, kmax)
                if ema:
                    L.EMA.swap_out(model, backup)
                model.train()
                sc = -m["loss"] if SELECT == "loss" else m.get("ap_choice", 0.0)
                flag = ""
                if best["score"] is None or sc > best["score"]:
                    src = ema.shadow if ema else {k: v for k, v in model.state_dict().items()
                                                  if "lora" in k}
                    best = {"score": sc, "step": step, "metrics": m,
                            "state": {k: v.detach().to("cpu").clone() for k, v in src.items()}}
                    bad, flag = 0, "  *"
                else:
                    bad += 1
                print(f"  e{epoch} s{step:5d}  train={run / max(1, seen):.4f}  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
                      + L.mem_note(device) + flag, flush=True)
                run, seen = 0.0, 0
                if bad >= PATIENCE:
                    print(f"  no dev improvement in {PATIENCE} evals, stopping")
                    stop = True
                    break
        print(f"  epoch {epoch} in {(time.time() - t_ep) / 60:.1f} min")

    if best["state"] is not None:
        sd = model.state_dict()
        model.load_state_dict({k: v.to(dtype=sd[k].dtype) for k, v in best["state"].items()},
                              strict=False)
    print(f"best dev at step {best['step']}: "
          + "  ".join(f"{k}={v:.4f}" for k, v in best["metrics"].items()))

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "loc_finetune.json"), "w") as f:
        json.dump({"base_model": MODEL, "rubric": RUBRIC, "seed": SEED,
                   "n_train_posts": len(tr_idx), "n_dev_posts": len(dv_idx),
                   "best_step": best["step"], "dev_metrics": best["metrics"],
                   "zero_shot_dev": base,
                   "hparams": {"epochs": EPOCHS, "batch": BATCH, "accum": ACCUM, "lr": LR,
                               "wd": WD, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                               "lora_dropout": LORA_DROPOUT, "ema": EMA_DECAY,
                               "shuffle_choice_options": bool(SHUFFLE_OPTS),
                               "max_len": MAX_LEN, "select": SELECT}}, f, indent=2)
    print(f"wrote {OUT_DIR}")
    print(f"\nscore the gold set with it:\n"
          f"  LOC_DECIDER_ADAPTER={OUT_DIR} LOC_DECIDER_RUBRIC={RUBRIC} \\\n"
          f"    ../../tmp_venv/bin/python score_loc_decider.py \\\n"
          f"      dataset_validation_gemini.jsonl scores_loc_decider_ft_{RUN}_gold.jsonl")


if __name__ == "__main__":
    main()
