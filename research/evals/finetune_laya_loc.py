"""Distil the `jev-latest` LoC teacher into a Laya System One checkpoint.

Zero-shot, both Laya checkpoints rank this axis below the deployed Jina dot
product (REPORT_laya_vs_jev.md). `eval_laya_floor.py` showed that is not the
axis being hard: neither checkpoint can tell German from English or spot a
literal question mark, so the decision heads are barely conditioning on the
question they are asked. That is a trained-behaviour problem rather than an
architecture problem, which is what makes it worth training.

WHAT IT DISTILS -- soft targets, not the argmax label. The teacher's own
distributions, for all three questions of one request:

    loc_class  ts_class_probs  -> one marker per class, in `criteria` key order
    loc_score  ts_score_probs  -> five ordered level markers
    agency     ts_noul         -> [1 - p, p]

That is exactly the shape Laya's own objective wants. `laya.common.proper_reward`
is a strictly proper scoring rule -- log score plus spherical score, plus a
ranked-probability term on ordinal `score` questions -- defined over soft target
distributions. Using the library's loss rather than a hand-rolled cross-entropy
is what keeps the checkpoint's calibration meaningful, and it is the only loss
here that knows level 0 sits further from level 4 than level 3 does. A hard
argmax would also throw away most of what was paid for: the teacher separates
"external 0.99" from "external 0.51", and the second is the interesting post.

WHAT IT MUST NOT DO. The 500 Gemini-labelled gold posts are a SUBSET of the
5500-post TypeSafe corpus -- all 500 of them, 498 by text as well as by URI.
Train on the corpus as it ships and every number `eval_laya_vs_jev.py` reports
afterwards is memorisation. They are dropped before the split and the run aborts
if any survive it. Model selection uses a held-out slice of the *teacher's*
labels; the Gemini set is touched once, by the eval, after training is over.
Per CLAUDE.md those are two different measurements and this file only ever makes
the first: dev numbers here are distillation fidelity, not axis accuracy.

REGULARISATION. 4500 posts against a 322M encoder, so the defaults are cautious
and every one of them is an env var:

  frozen embeddings + bottom encoder layers   LAYA_FT_FREEZE_BELOW (default 25%)
  layer-wise LR decay toward the input        LAYA_FT_LLRD          0.9
  decoupled weight decay, off norms/biases    LAYA_FT_WD            0.01
  dropout, encoder and head                   LAYA_FT_DROPOUT_{ENC,HEAD}
  label smoothing on the soft target          LAYA_FT_SMOOTH        0.02
  option-order shuffling (choice only, below) LAYA_FT_SHUFFLE       1
  weight averaging (EMA) of the eval weights  LAYA_FT_EMA           0.999
  cosine decay with warmup, grad clipping     LAYA_FT_WARMUP, _CLIP
  early stopping on dev, best weights kept    LAYA_FT_PATIENCE      3

MEMORY. On Apple Silicon the weights live in unified memory and MPS allocations do
not show up in RSS, so `ps` reports a tenth of a GB for a run holding several. Each
eval prints allocated/driver GB instead. The levers, cheapest first:

  LAYA_FT_EMPTY_CACHE  50   steps between allocator releases (0 disables)
  LAYA_FT_GRAD_CKPT    0    recompute encoder activations; ~30% slower, and the
                            one lever that changes no hyperparameter
  LAYA_FT_BATCH        16   with LAYA_FT_ACCUM to hold the effective batch fixed
  LAYA_FT_FREEZE_BELOW -1   freezing more layers drops their grads and Adam states

Activations dominate at batch 16, so GRAD_CKPT first, then batch+accum. Dropping
the batch alone without raising ACCUM changes the optimisation, not just the
footprint.

The run will not start without measuring this. `memory_cap` caps the process at
LAYA_FT_MEM_FRACTION (0.5) of the GPU working set, so an over-large batch raises a
RuntimeError instead of paging the machine into a state where it has to be killed
from outside; `memory_probe` then runs the heaviest batch length bucketing can
produce -- the BATCH longest items -- for a few real steps, so Adam's moments and
the EMA shadow are resident while it measures, and aborts with the settings to
change if the peak is within 15% of the cap. Its steps are rolled back before
training.

Option-order shuffling applies to `choice` only. Shuffling a `score` question
would destroy the ordinal reading both the RPS term and the runtime's
`sum(i * p_i)` depend on, and `noul`'s marker 1 is `true` by convention
throughout the runtime. Choice is where a positional prior can hide, and it is
the primitive the deployed P(internal) would come from.

OUTPUT. A directory `laya.Agent` loads unmodified -- same `rl_agent_config.json`,
`model.safetensors`, `encoder/`, `tokenizer/` layout as the checkpoint it
started from -- so `score_loc_laya.py LAYA_MODEL=<dir>` scores with it and
`eval_laya_vs_jev.py` compares it, both unchanged. Per-qtype temperatures are
refitted on dev and written into the config, because the runtime applies them
and a fine-tune invalidates whatever the checkpoint shipped with.

Run (MPS, ~25 min/epoch for the multilingual checkpoint):

    HF_HOME=/Volumes/Daten/huggingface PYTHONPATH=../../training \
    ../../tmp_venv/bin/python finetune_laya_loc.py

    LAYA_FT_PREFLIGHT=1   checks alignment, clipping and the split, trains nothing
    LAYA_FT_FREEZE_ENCODER=1  heads only: ~20x faster, and the cheap first answer
"""

import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN_DATA = os.path.join(ROOT, "training", "data")
OUT_DATA = os.path.join(ROOT, "research", "data", "laya_loc")

sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, HERE)

import loc_questions_compact  # noqa: E402
import ts_questions  # noqa: E402

# ---- inputs -------------------------------------------------------------------
# The corpus carries the text; the score file carries the teacher's distributions.
# They are written by the same pass and share a row order, which is asserted
# rather than assumed -- the positional-cache failure in CLAUDE.md is this same
# mistake one file over.
CORPUS = os.path.join(TRAIN_DATA, "dataset_annotated_typesafe_g3.jsonl")
TEACHER = os.path.join(TRAIN_DATA, "scores_loc_typesafe_glimmer_v3.jsonl")
HELD_OUT = os.path.join(TRAIN_DATA, "dataset_validation_gemini.jsonl")

def env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)

MODEL = env("LAYA_MODEL", "convaiinnovations/laya-multilingual")
DEVICE = env("LAYA_DEVICE", None)
RUBRIC = env("LAYA_RUBRIC", "compact")
RUN = env("LAYA_FT_RUN", "ml_compact")
OUT_DIR = env("LAYA_FT_OUT", os.path.join(OUT_DATA, RUN))
SEED = env("LAYA_FT_SEED", 20260919, int)

EPOCHS = env("LAYA_FT_EPOCHS", 4, int)
BATCH = env("LAYA_FT_BATCH", 16, int)
ACCUM = env("LAYA_FT_ACCUM", 1, int)
ENC_LR = env("LAYA_FT_ENC_LR", 2e-5, float)
HEAD_LR = env("LAYA_FT_HEAD_LR", 1e-4, float)
LLRD = env("LAYA_FT_LLRD", 0.9, float)
WD = env("LAYA_FT_WD", 0.01, float)
WARMUP = env("LAYA_FT_WARMUP", 0.1, float)
CLIP = env("LAYA_FT_CLIP", 1.0, float)
SMOOTH = env("LAYA_FT_SMOOTH", 0.02, float)
TARGET_TEMP = env("LAYA_FT_TARGET_TEMP", 1.0, float)
DROPOUT_ENC = env("LAYA_FT_DROPOUT_ENC", 0.1, float)
DROPOUT_HEAD = env("LAYA_FT_DROPOUT_HEAD", 0.1, float)
EMA_DECAY = env("LAYA_FT_EMA", 0.999, float)
SHUFFLE_OPTS = env("LAYA_FT_SHUFFLE", 1, int)
CLASS_WEIGHT = env("LAYA_FT_CLASS_WEIGHT", "sqrt")  # sqrt | inverse | none
FREEZE_ENCODER = env("LAYA_FT_FREEZE_ENCODER", 0, int)
FREEZE_BELOW = env("LAYA_FT_FREEZE_BELOW", -1, int)  # -1 -> bottom quarter
DEV_FRAC = env("LAYA_FT_DEV_FRAC", 0.1, float)
EVALS_PER_EPOCH = env("LAYA_FT_EVALS_PER_EPOCH", 2, int)
PATIENCE = env("LAYA_FT_PATIENCE", 3, int)
SELECT = env("LAYA_FT_SELECT", "loss")  # loss | ap
PREFLIGHT = env("LAYA_FT_PREFLIGHT", 0, int)
GRAD_CKPT = env("LAYA_FT_GRAD_CKPT", 0, int)        # recompute activations instead of storing them
MEM_FRACTION = env("LAYA_FT_MEM_FRACTION", 0.5, float)  # share of the GPU working set this run may hold
MEM_PROBE_STEPS = env("LAYA_FT_MEM_PROBE", 3, int)      # worst-case steps to measure before training; 0 skips
MEM_PROBE_ONLY = env("LAYA_FT_MEM_PROBE_ONLY", 0, int)  # measure and exit, training nothing
PAD_MULTIPLE = env("LAYA_FT_PAD_MULTIPLE", 32, int)     # quantise sequence length; 0 pads to the batch max
EMPTY_EVERY = env("LAYA_FT_EMPTY_CACHE", 50, int)   # steps between allocator releases; 0 disables

QUESTIONS = (ts_questions if RUBRIC == "full" else loc_questions_compact).questions()
QORDER = list(QUESTIONS)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- targets ------------------------------------------------------------------
def teacher_target(qid, q, s):
    """The teacher's distribution over this question's options, in marker order.

    Marker j is option j of `render_options(q)`, which is `criteria` key order for
    choice, level order for score, and [false, true] for noul. Built off the
    question definition rather than a hardcoded list so a rubric edit cannot
    silently transpose two classes -- the one failure here that would still train.
    """
    if q["type"] == "choice":
        p = [float(s["ts_class_probs"].get(k, 0.0)) for k in q["criteria"]]
    elif q["type"] == "score":
        p = [float(s["ts_score_probs"].get(str(i), 0.0)) for i in range(len(q["criteria"]))]
    else:
        n = float(s["ts_noul"])
        p = [1.0 - n, n]
    p = np.asarray(p, dtype=np.float64)
    if p.sum() <= 0:
        raise ValueError(f"{qid}: teacher distribution is all zeros")
    p /= p.sum()
    if TARGET_TEMP != 1.0:  # sharpen (<1) or soften (>1) the teacher
        p = p ** (1.0 / TARGET_TEMP)
        p /= p.sum()
    if SMOOTH > 0:  # keeps the log term off a saturated 0.99/0.00 target
        p = (1.0 - SMOOTH) * p + SMOOTH / len(p)
    return p


def build_pairs():
    """[(text, teacher_row, class)] for every corpus post that is not held out."""
    corpus, teacher = load_jsonl(CORPUS), load_jsonl(TEACHER)
    assert len(corpus) == len(teacher), f"{len(corpus)} corpus rows vs {len(teacher)} teacher rows"
    assert [r["uri"] for r in corpus] == [r["uri"] for r in teacher], \
        "corpus and teacher score file are not in the same row order"

    held = {r["uri"] for r in load_jsonl(HELD_OUT)}
    held_text = {r["text"] for r in load_jsonl(HELD_OUT) if r.get("text")}
    pairs, dropped = [], 0
    for c, s in zip(corpus, teacher):
        if not c.get("text"):
            continue
        if c["uri"] in held or c["text"] in held_text:
            dropped += 1
            continue
        pairs.append((c["text"], s, s["ts_class"]))

    # The gold set is wholly contained in this corpus, so this is the guard that
    # keeps the eval honest rather than a formality.
    assert dropped >= len(held), f"only {dropped} of {len(held)} held-out posts were dropped"
    leaked = [t for t, _, _ in pairs if t in held_text]
    assert not leaked, f"{len(leaked)} held-out texts survived the filter"
    print(f"corpus {len(corpus)} rows - {dropped} held out = {len(pairs)} trainable")
    return pairs


def split(pairs, rng):
    """Stratified by the teacher's class: internal is 10% of the corpus and a dev
    slice that sampled it away would early-stop on the two classes nobody ranks."""
    by_class = {}
    for i, (_, _, c) in enumerate(pairs):
        by_class.setdefault(c, []).append(i)
    tr, dv = [], []
    for c, idx in sorted(by_class.items()):
        idx = list(idx)
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * DEV_FRAC)))
        dv += idx[:n]
        tr += idx[n:]
    rng.shuffle(tr)
    rng.shuffle(dv)
    return tr, dv


# ---- items --------------------------------------------------------------------
def make_items(tok, pairs, index, cfg, rng=None):
    """One item per (post, question). `rng` non-None permutes choice options."""
    from laya.common import QTYPES, build_sequence

    max_len, head_max = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    items = []
    for i in index:
        text, s, cls = pairs[i]
        for qid in QORDER:
            q = QUESTIONS[qid]
            tgt = teacher_target(qid, q, s)
            k = len(tgt)
            order = list(range(k))
            if rng is not None and SHUFFLE_OPTS and q["type"] == "choice":
                rng.shuffle(order)
            seq, markers = build_sequence(
                tok, text, {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")},
                max_len, head_max, option_order=order)
            if len(markers) != k:
                raise ValueError(f"{qid}: {len(markers)} markers for {k} options; head_max_len={head_max}")
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]],
                          "target": [tgt[o] for o in order], "post": i, "cls": cls, "qid": qid})
    return items


def class_weights(pairs, index):
    if CLASS_WEIGHT == "none":
        return {c: 1.0 for c in {p[2] for p in pairs}}
    n = Counter(pairs[i][2] for i in index)
    total = sum(n.values())
    w = {c: (total / (len(n) * v)) ** (0.5 if CLASS_WEIGHT == "sqrt" else 1.0) for c, v in n.items()}
    mean = sum(w[c] * v for c, v in n.items()) / total  # normalise so the LR still means what it did
    return {c: v / mean for c, v in w.items()}


def batches(items, size, shuffle_rng=None, bucket=32):
    """Length-bucketed when shuffling: shuffle, then sort within windows of
    `bucket` batches and emit the batches in random order. Sequences here run
    140-340 tokens, so a randomly mixed batch pads almost every row up to its
    longest -- and activation memory is charged on the padded length, not the real
    one. Windowed sorting keeps the sampling close to random while cutting that."""
    idx = list(range(len(items)))
    if shuffle_rng is None:
        for i in range(0, len(idx), size):
            yield [items[j] for j in idx[i:i + size]]
        return
    shuffle_rng.shuffle(idx)
    out = []
    for w in range(0, len(idx), size * bucket):
        win = sorted(idx[w:w + size * bucket], key=lambda j: len(items[j]["ids"]))
        out += [win[i:i + size] for i in range(0, len(win), size)]
    shuffle_rng.shuffle(out)
    for b in out:
        yield [items[j] for j in b]


# ---- model plumbing -----------------------------------------------------------
def layer_index(name):
    """Encoder depth of a parameter, or None. Covers `layers.N.` (ModernBERT,
    mmBERT) and `encoder.layer.N.` (BERT-style)."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def set_dropout(module, p):
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p


def param_groups(model, n_layers):
    """Layer-wise LR decay: the closer to the input, the less it moves. Norms and
    biases are excluded from weight decay as usual; `act_head` is frozen because
    nothing here supplies an escalation target and AdamW would otherwise decay a
    head that never sees a gradient."""
    model.act_head.requires_grad_(False)

    freeze_below = FREEZE_BELOW if FREEZE_BELOW >= 0 else max(0, n_layers // 4)
    groups, frozen = {}, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_enc = name.startswith("encoder.")
        li = layer_index(name) if is_enc else None
        if FREEZE_ENCODER and is_enc:
            p.requires_grad_(False); frozen += p.numel(); continue
        if is_enc and (li is None and "embed" in name.lower() or (li is not None and li < freeze_below)):
            p.requires_grad_(False); frozen += p.numel(); continue
        if is_enc:
            depth = li if li is not None else n_layers  # post-encoder norms ride with the top
            lr = ENC_LR * (LLRD ** (n_layers - depth))
        else:
            lr = HEAD_LR
        decay = WD if (p.dim() > 1 and "norm" not in name.lower()) else 0.0
        groups.setdefault((lr, decay), []).append(p)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable {trainable / 1e6:.1f}M, frozen {frozen / 1e6:.1f}M "
          f"(encoder layers < {freeze_below if not FREEZE_ENCODER else n_layers} held)")
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


def memory_cap(device):
    """Cap this process's GPU allocation, so overflow is a Python exception.

    Unified memory means an over-large batch does not fail, it pages -- the machine
    stops responding and the run has to be killed from outside. A per-process cap
    turns that into a RuntimeError at the allocation that would have crossed the
    line, which is recoverable and reports where it happened. It is set below what
    Metal reports as its working set, because the display server and everything
    else on the desktop draw from the same pool.
    """
    if device.type == "mps":
        total = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(MEM_FRACTION)
    elif device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, device)
    else:
        return None
    cap = total * MEM_FRACTION
    print(f"  GPU working set {total / 2**30:.1f} GB, this run capped at "
          f"{cap / 2**30:.1f} GB ({MEM_FRACTION:.0%})")
    return cap


def peak_memory(device):
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return 0


def memory_probe(model, items, tok, device, weights, opt, cap, steps):
    """Run the heaviest batch the real loop can produce, and refuse to go on if it
    does not fit with room to spare.

    The heaviest batch is not a random one: length bucketing deliberately groups the
    longest sequences together, so the worst case is the `BATCH` longest items, and
    activation cost is charged on that padded length. Optimiser state is allocated
    lazily on the first `step()`, so this takes several -- one step would measure a
    model without its Adam moments and under-report by 2x the trainable size.
    """
    if steps <= 0 or cap is None:
        return None
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["ids"]))
    worst = [items[i] for i in order[:BATCH]]
    seq = max(len(it["ids"]) for it in worst)
    empty_cache(device)
    model.train()
    try:
        for _ in range(steps):
            batch = collate(worst, tok.pad_token_id)
            wts = torch.tensor([weights[it["cls"]] for it in worst], dtype=torch.float32)
            loss = loss_of(forward_batch(model, batch, device), batch, device, wts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        # The cap did its job: this is the crash that would otherwise have been the
        # machine paging itself to a standstill. Report it as a setting to change
        # rather than as a stack trace out of the attention kernel.
        opt.zero_grad(set_to_none=True)
        empty_cache(device)
        raise SystemExit(
            f"\nWILL NOT FIT: the worst-case batch ({BATCH} x {seq} tokens) "
            f"exceeds the {cap / 2**30:.1f} GB cap.\n"
            f"Nothing was trained and the machine is untouched. Reduce it with:\n"
            f"  LAYA_FT_GRAD_CKPT=1        recompute activations (~30% slower)\n"
            f"  LAYA_FT_BATCH={max(2, BATCH // 2)} LAYA_FT_ACCUM={ACCUM * 2}   "
            f"half the batch, same effective batch\n"
            f"  LAYA_FT_FREEZE_BELOW=<n>   freeze more layers, fewer Adam states\n"
            f"or raise the ceiling with LAYA_FT_MEM_FRACTION if the machine is idle.")
    peak = peak_memory(device)
    print(f"  worst-case batch: {BATCH} x {seq} tokens -> peak {peak / 2**30:.2f} GB "
          f"of {cap / 2**30:.1f} GB cap ({peak / cap:.0%})")
    if peak > 0.85 * cap:
        raise SystemExit(
            f"\nABORTED before training: the worst-case batch peaks at "
            f"{peak / 2**30:.2f} GB against a {cap / 2**30:.1f} GB cap.\n"
            f"Training would page the machine rather than fail. Reduce it with:\n"
            f"  LAYA_FT_GRAD_CKPT=1        recompute activations (~30% slower)\n"
            f"  LAYA_FT_BATCH={max(2, BATCH // 2)} LAYA_FT_ACCUM={ACCUM * 2}   "
            f"half the batch, same effective batch\n"
            f"  LAYA_FT_FREEZE_BELOW=<n>   freeze more layers, fewer Adam states\n"
            f"or raise the ceiling with LAYA_FT_MEM_FRACTION if the machine is idle.")
    return peak


def collate(items, pad_id):
    """collate_items, with the sequence length rounded up to PAD_MULTIPLE.

    MPS pools free buffers by size and does not hand them back -- empty_cache()
    leaves the reserved pool where it is. So every distinct padded length a run
    produces permanently costs another pool. Padding to the batch maximum makes
    almost every batch a new length: the first attempt at this run grew ~1 GB per
    200 steps and was heading for 21 GB, with live tensors flat at 3 GB the whole
    time. Rounding to a multiple of 32 leaves ~7 lengths for a 140-340 token corpus,
    so the pool saturates in the first few hundred steps and then stays put.

    The padding is masked, so this changes arithmetic not at all -- only the number
    of distinct buffer sizes the allocator is asked for.
    """
    from laya.common import collate_items

    b = collate_items([items], pad_id)
    if PAD_MULTIPLE <= 0:
        return b
    n, L = b["input_ids"].shape
    L2 = -(-L // PAD_MULTIPLE) * PAD_MULTIPLE
    if L2 > L:
        pad = torch.full((n, L2 - L), pad_id, dtype=b["input_ids"].dtype)
        b["input_ids"] = torch.cat([b["input_ids"], pad], 1)
        b["attention_mask"] = torch.cat(
            [b["attention_mask"], torch.zeros((n, L2 - L), dtype=b["attention_mask"].dtype)], 1)
    return b


def empty_cache(device):
    """MLX is not the only allocator that holds freed buffers; MPS does too, and an
    uncapped run grows until the OOM killer arrives. Cheap enough to call often."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def mem_note(device):
    if device.type == "mps":
        return f"  mem={torch.mps.current_allocated_memory() / 2**30:.2f}/{torch.mps.driver_allocated_memory() / 2**30:.2f}GB"
    if device.type == "cuda":
        return f"  mem={torch.cuda.memory_allocated() / 2**30:.2f}/{torch.cuda.max_memory_allocated() / 2**30:.2f}GB"
    return ""


def lr_lambda(step, total):
    warm = max(1, int(total * WARMUP))
    if step < warm:
        return step / warm
    t = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


class EMA:
    """Weight averaging over the trajectory. On a 4500-post fine-tune the averaged
    weights are reliably a little better than the last step's, and nearly free.

    Only the parameters that actually move are averaged. Shadowing the whole
    state_dict would copy mmBERT's ~200M-row embedding matrix -- frozen here, so
    its average is itself -- and that copy was the single largest avoidable
    allocation in the run. Restoring a partial shadow is still correct precisely
    because the omitted tensors cannot have changed."""

    def __init__(self, model, decay):
        self.decay = decay
        train = {n for n, p in model.named_parameters() if p.requires_grad}
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if k in train and v.dtype.is_floating_point}

    def update(self, model):
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def swap_in(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({k: v.to(dtype=backup[k].dtype) for k, v in self.shadow.items()}, strict=False)
        return backup

    @staticmethod
    def swap_out(model, backup):
        model.load_state_dict(backup, strict=False)


def forward_batch(model, batch, device):
    return model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                 batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                 batch["qtype"].to(device))[0]


def loss_of(logits, batch, device, weights=None):
    from laya.common import proper_reward
    mask = batch["marker_mask"].to(device)
    target = batch["target"].to(device)
    q = torch.softmax(logits, -1)
    r = proper_reward(q, target, batch["qtype"].to(device), mask)
    if weights is not None:
        return -(r * weights.to(device)).mean()
    return -r.mean()


@torch.no_grad()
def evaluate(model, items, tok, device, pairs):
    """Dev loss under the same proper scoring rule, plus the ranking metric the
    feed actually consumes -- but against the TEACHER's internal/not call, not a
    human's. This is distillation fidelity and nothing more."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    tot, n = 0.0, 0
    p_int, p_score, y = [], [], []
    for b in batches(items, BATCH):
        batch = collate(b, tok.pad_token_id)
        logits = forward_batch(model, batch, device)
        tot += float(loss_of(logits, batch, device)) * len(b)
        n += len(b)
        p = torch.softmax(logits, -1).float().cpu().numpy()
        for j, it in enumerate(b):
            k = len(it["markers"])
            if it["qid"] == "loc_class":
                keys = list(QUESTIONS["loc_class"]["criteria"])
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


@torch.no_grad()
def fit_temperature(model, items, tok, device):
    """One scalar per qtype, refitted against the teacher's soft distribution.

    The runtime divides by these before softmaxing, so the shipped values describe
    the head that shipped, not this one. Fitted on dev, by grid over log T --
    three parameters, 1500 items, nothing subtler is warranted."""
    model.eval()
    per_type = {0: [], 1: [], 2: []}
    for b in batches(items, BATCH):
        batch = collate(b, tok.pad_token_id)
        logits = forward_batch(model, batch, device).float().cpu()
        for j, it in enumerate(b):
            k = len(it["markers"])
            per_type[it["qtype"]].append((logits[j, :k], torch.tensor(it["target"][:k])))

    temps = [1.0, 1.0, 1.0]
    for qt, rows in per_type.items():
        if not rows:
            continue
        best, best_nll = 1.0, float("inf")
        for T in np.exp(np.linspace(math.log(0.25), math.log(4.0), 61)):
            nll = 0.0
            for z, t in rows:
                lp = torch.log_softmax(z / float(T), -1)
                nll -= float((t * lp).sum())
            if nll < best_nll:
                best_nll, best = nll, float(T)
        temps[qt] = round(best, 4)
    return temps


def restore_pristine(model, src_dir):
    """Undo the probe's optimiser steps by reloading the checkpoint it started from.

    The probe has to take real steps -- Adam allocates its moments lazily, so a
    forward/backward alone measures a smaller model than the one that will train.
    Those steps are on the worst-case batch only, which is not a sample anyone wants
    in the weights, so they are rolled back rather than kept.
    """
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(src_dir, "model.safetensors")), strict=True)


def save_checkpoint(model, cfg, src_dir, out_dir, extra):
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    for sub in ("encoder", "tokenizer"):
        s, d = os.path.join(src_dir, sub), os.path.join(out_dir, sub)
        if os.path.exists(s):
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(s, d)  # follows the HF cache's blob symlinks
    sd = {k: v.detach().to("cpu", torch.float32).contiguous().clone()
          for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    cfg = dict(cfg)
    cfg["temperature_by_options"] = {}  # a stale bucket would shadow the refit scalars
    cfg["loc_finetune"] = extra
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {out_dir}")


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = random.Random(SEED)

    pairs = build_pairs()
    tr_idx, dv_idx = split(pairs, rng)
    print(f"split: {len(tr_idx)} train / {len(dv_idx)} dev posts "
          f"(train {Counter(pairs[i][2] for i in tr_idx)}, dev {Counter(pairs[i][2] for i in dv_idx)})")

    import laya
    from huggingface_hub import snapshot_download
    from laya.common import collate_items

    src_dir = MODEL if os.path.exists(MODEL) else snapshot_download(MODEL, token=os.environ.get("HF_TOKEN"))
    t0 = time.time()
    agent = laya.load(src_dir, device=DEVICE)
    model, tok, cfg, device = agent.model, agent.tok, agent.cfg, agent.device
    print(f"loaded {MODEL} on {device} in {time.time() - t0:.1f}s "
          f"(max_len={cfg.get('max_len')}, head_max_len={cfg.get('head_max_len')})")

    print(f"rubric={RUBRIC}:")
    if not loc_questions_compact.check(agent, QUESTIONS):
        print("  WARNING: options or instructions are being clipped; the model is "
              "being trained on a truncated rubric")

    tr_items = make_items(tok, pairs, tr_idx, cfg, rng=rng)
    dv_items = make_items(tok, pairs, dv_idx, cfg)
    w = class_weights(pairs, tr_idx)
    print(f"{len(tr_items)} train items / {len(dv_items)} dev items; class weights "
          + ", ".join(f"{c}={v:.2f}" for c, v in sorted(w.items())))

    if PREFLIGHT:
        b = collate_items([tr_items[:4]], tok.pad_token_id)
        print("\npreflight sample:")
        for it in tr_items[:3]:
            print(f"  {it['qid']:10} qtype={it['qtype']} markers={len(it['markers'])} "
                  f"target={[round(t, 3) for t in it['target']]} len={len(it['ids'])}")
        print(f"  collated: ids{tuple(b['input_ids'].shape)} target{tuple(b['target'].shape)}")
        print(f"  seq len p50/p95/max: " + "/".join(
            str(int(np.percentile([len(i['ids']) for i in tr_items], p))) for p in (50, 95, 100)))
        print("\npreflight only, nothing trained")
        return

    n_layers = int(getattr(model.encoder.config, "num_hidden_layers", 12))
    if GRAD_CKPT:
        # Trades ~30% throughput for most of the activation memory. The encoder is
        # 22 layers deep and activations dominate at batch 16; this is the lever to
        # pull before dropping the batch size, because it changes no hyperparameter.
        try:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print("  gradient checkpointing enabled on the encoder")
        except Exception as e:
            print(f"  gradient checkpointing unavailable ({e}); continuing without")
    set_dropout(model.encoder, DROPOUT_ENC)
    for m in (model.head, model.scorer):
        if m is not None:
            set_dropout(m, DROPOUT_HEAD)
    groups = param_groups(model, n_layers)
    opt = torch.optim.AdamW(groups, lr=HEAD_LR, betas=(0.9, 0.98), eps=1e-6)
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = math.ceil(len(tr_items) / BATCH / ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    eval_every = max(1, steps_per_epoch // max(1, EVALS_PER_EPOCH))

    # Everything that will ever be resident must exist before the probe measures:
    # weights, grads, Adam moments and the EMA shadow, under the worst batch the
    # loop can build. Then the probe's steps are rolled back and training starts
    # from the checkpoint, not from three steps of the longest posts in the corpus.
    cap = memory_cap(device)
    ema = EMA(model, EMA_DECAY) if EMA_DECAY > 0 else None
    probe_peak = memory_probe(model, tr_items, tok, device, w, opt, cap, MEM_PROBE_STEPS)
    if MEM_PROBE_ONLY:
        print(f"\nprobe only: worst-case step fits in "
              f"{probe_peak / 2**30:.2f} GB of a {cap / 2**30:.1f} GB cap, nothing trained")
        return
    restore_pristine(model, src_dir)
    opt.state.clear()
    if ema is not None:
        ema = None
        empty_cache(device)
        ema = EMA(model, EMA_DECAY)
    empty_cache(device)
    print(f"{EPOCHS} epochs x {steps_per_epoch} steps, eval every {eval_every}, "
          f"select on dev {SELECT}, patience {PATIENCE}")

    base = evaluate(model, dv_items, tok, device, pairs)
    print("  dev @0: " + "  ".join(f"{k}={v:.4f}" for k, v in base.items()) + "   (zero-shot)")

    best = {"score": None, "state": None, "step": 0, "metrics": base}
    bad, step = 0, 0
    stop = False
    for epoch in range(EPOCHS):
        if stop:
            break
        model.train()
        t_ep, seen, run = time.time(), 0, 0.0
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(batches(tr_items, BATCH, rng)):
            batch = collate(b, tok.pad_token_id)
            wts = torch.tensor([w[it["cls"]] for it in b], dtype=torch.float32)
            loss = loss_of(forward_batch(model, batch, device), batch, device, wts) / ACCUM
            loss.backward()
            run += float(loss.detach()) * ACCUM; seen += 1
            if (bi + 1) % ACCUM:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], CLIP)
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * lr_lambda(step, total_steps)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if ema:
                ema.update(model)
            if EMPTY_EVERY and step % EMPTY_EVERY == 0:
                empty_cache(device)
                if cap is not None and peak_memory(device) > 0.8 * cap:
                    # The probe bounds one step; this bounds the run. Allocator
                    # growth across many distinct padded shapes is what actually
                    # killed the first attempt, and it does not show up in a probe.
                    held = peak_memory(device)
                    raise SystemExit(
                        f"\nABORTED at step {step}: holding {held / 2**30:.2f} GB, "
                        f"over 80% of the {cap / 2**30:.1f} GB cap, after a cache "
                        f"release.\nThis is allocator growth, not one big batch. "
                        f"Re-run with LAYA_FT_EMPTY_CACHE={max(5, EMPTY_EVERY // 5)} "
                        f"or LAYA_FT_GRAD_CKPT=1.")

            if step % eval_every == 0 or step == total_steps:
                backup = ema.swap_in(model) if ema else None
                m = evaluate(model, dv_items, tok, device, pairs)
                if ema:
                    EMA.swap_out(model, backup)
                model.train()
                score = -m["loss"] if SELECT == "loss" else m.get("ap_choice", 0.0)
                flag = ""
                if best["score"] is None or score > best["score"]:
                    best = {"score": score, "step": step, "metrics": m,
                            "state": {k: v.detach().to("cpu").clone()
                                      for k, v in (ema.shadow if ema else model.state_dict()).items()}}
                    bad, flag = 0, "  *"
                else:
                    bad += 1
                print(f"  e{epoch} s{step:5d}  train={run / max(1, seen):.4f}  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
                      + mem_note(device) + flag, flush=True)
                run, seen = 0.0, 0
                if bad >= PATIENCE:
                    print(f"  no dev improvement in {PATIENCE} evals, stopping")
                    stop = True
                    break
        print(f"  epoch {epoch} in {(time.time() - t_ep) / 60:.1f} min")

    if best["state"] is not None:
        model.load_state_dict({k: v.to(dtype=dict(model.state_dict())[k].dtype)
                               for k, v in best["state"].items()}, strict=False)
    print(f"best dev at step {best['step']}: "
          + "  ".join(f"{k}={v:.4f}" for k, v in best["metrics"].items()))

    temps = fit_temperature(model, dv_items, tok, device)
    print(f"refit temperatures (choice/score/noul): {temps}")
    cfg = dict(cfg); cfg["temperature"] = temps

    save_checkpoint(model, cfg, src_dir, OUT_DIR, {
        "base_model": MODEL, "rubric": RUBRIC, "teacher": os.path.basename(TEACHER),
        "corpus": os.path.basename(CORPUS), "held_out": os.path.basename(HELD_OUT),
        "n_train_posts": len(tr_idx), "n_dev_posts": len(dv_idx), "seed": SEED,
        "best_step": best["step"], "dev_metrics": best["metrics"], "zero_shot_dev": base,
        "hparams": {"epochs": EPOCHS, "batch": BATCH, "enc_lr": ENC_LR, "head_lr": HEAD_LR,
                    "llrd": LLRD, "wd": WD, "smooth": SMOOTH, "target_temp": TARGET_TEMP,
                    "dropout_enc": DROPOUT_ENC, "dropout_head": DROPOUT_HEAD, "ema": EMA_DECAY,
                    "shuffle_choice_options": bool(SHUFFLE_OPTS), "class_weight": CLASS_WEIGHT,
                    "freeze_encoder": bool(FREEZE_ENCODER), "freeze_below": FREEZE_BELOW,
                    "select": SELECT, "warmup": WARMUP, "clip": CLIP},
    })
    print(f"\nscore the gold set with it:\n"
          f"  LAYA_MODEL={OUT_DIR} LAYA_RUBRIC={RUBRIC} \\\n"
          f"    ../../tmp_venv/bin/python score_loc_laya.py \\\n"
          f"      dataset_validation_gemini.jsonl scores_loc_laya_ft_{RUN}_gold.jsonl")


if __name__ == "__main__":
    main()
