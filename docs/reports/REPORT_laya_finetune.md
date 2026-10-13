# Fine-tuning an open System One model on the LoC axis

**Question.** `REPORT_laya_vs_jev.md` measured four open System One candidates
zero-shot and none was worth running: all were beaten by the free Jina dot
product already deployed at the edge. But that report also showed *why* Laya
failed, and the reason was fixable — `eval_laya_floor.py` found neither
checkpoint could tell German from English or spot a literal question mark, so
the decision heads were barely conditioning on the question at all. That is a
trained-behaviour problem, not an architecture problem. So: does training it fix
it?

**Answer.** Yes, and by a wide margin — AP +0.372 on the choice primitive,
p<0.001 — but it lands exactly level with the deployed vector rather than above
it. What it buys is not rank quality. It is a usable three-way label, a 12x
faster call than the paid teacher, and German.

Measured 2026-09-19 on the same 500 Gemini-labelled gold posts
(81 internal / 243 external / 176 neutral), same harness, same 4000-sample
bootstrap. Scripts: `research/evals/finetune_laya_loc.py`,
`score_loc_decider.py`, `eval_laya_vs_jev.py`.

## The gold set was inside the training corpus

Before any of the numbers below: all 500 posts of `dataset_validation_gemini.jsonl`
are present in `dataset_annotated_typesafe_g3.jsonl` — by URI, and 498 of them by
text as well. The 5500-post corpus is a superset of the gold set.

Distilling from the corpus as it ships and then reporting on the gold set would
have measured memorisation and produced a spectacular, meaningless number.
`build_pairs()` drops them before the split and asserts that it did; 5000 posts
remain, 4500 train / 500 dev, stratified on the teacher's class because internal
is only 10% of the corpus.

Model selection never touches the gold set either. The dev slice is scored by the
*teacher*, so every dev number in the training log is distillation fidelity;
the gold set is read once, by the eval, after training is over. That is the
distinction CLAUDE.md asks for, and it is the reason the dev AP (0.541) and the
gold AP (0.524) are not the same measurement even though they are close.

## What was distilled, and with which loss

Soft targets, not labels. The teacher answers three questions per post and its
full distributions are kept, because they are what was paid for — it separates
"external 0.99" from "external 0.51" and the second is the interesting post.

    loc_class   ts_class_probs   -> one marker per class, in criteria key order
    loc_score   ts_score_probs   -> five ordered level markers
    agency      ts_noul          -> [1-p, p]

The objective is `laya.common.proper_reward`, the library's own training loss: a
strictly proper scoring rule (log + spherical, plus a ranked-probability term on
ordinal `score` questions) defined over soft targets. Using Laya's own loss
rather than a hand-rolled cross-entropy keeps the checkpoint's calibration
meaningful, and it is the only loss here that knows level 0 sits further from
level 4 than level 3 does.

Regularisation, because it is 4500 posts against a 322M encoder: frozen
embeddings and bottom quarter of the encoder, layer-wise LR decay 0.9, weight
decay off norms and biases, dropout, label smoothing 0.02 on the soft target,
EMA weight averaging, cosine schedule with warmup, gradient clipping, early
stopping on dev.

Option-order shuffling is applied to `choice` only. Shuffling a `score` question
would destroy the ordinal reading both the RPS term and the runtime's
`sum(i * p_i)` depend on, and `noul`'s marker 1 is `true` by convention
throughout the runtime. Choice is where a positional prior can hide and it is
the primitive a deployed P(internal) would come from.

`act_head` is frozen: nothing here supplies an escalation target, and left in the
optimiser AdamW would have decayed a head that never receives a gradient.

## The encoder is where the axis lives

The first run froze the encoder and trained only the 14.8M-parameter head. Dev
loss fell — 0.946 to 0.828 — and the ranking did not move at all: AP 0.076 to
0.089 against a 0.10 base rate, AUC below chance throughout. It learned the
marginal class distribution and nothing else.

That is a clean null and it settles what `eval_laya_floor.py` could only suggest.
The marker-position representations coming out of the frozen encoder do not carry
the post-to-option relationship. Whatever the shipped decision head is doing, the
information it would need is not there to be read. Unfreezing the encoder is what
makes the axis learnable, and everything below uses it.

## Results on the gold set

Best primitive per system. AP and P@25 lead, per CLAUDE.md.

| system | primitive | AUC [95% CI] | AP | P@25 |
|---|---|---|---|---|
| **jev** (`jev-latest`, paid) | choice | **0.942** [0.913, 0.964] | **0.788** | **0.920** |
| qwen_7b (Qwen2.5-7B, 4.0 GB) | score | 0.868 [0.816, 0.916] | 0.631 | 0.760 |
| qwen3_8b (4.3 GB) | choice | 0.837 [0.783, 0.888] | 0.615 | 0.840 |
| **decider-2b** (Qwen3.5-2B-Base) | choice | 0.861 [0.816, 0.902] | 0.589 | 0.720 |
| **laya-ft** (mmBERT-base, 322M) | noul | 0.793 [0.732, 0.847] | 0.524 | 0.680 |
| jina (deployed vector) | dot product | 0.803 [0.742, 0.855] | 0.523 | 0.680 |
| laya-ft | choice | 0.830 [0.778, 0.877] | 0.504 | 0.600 |
| laya-ml (same checkpoint, zero-shot) | noul | 0.622 [0.552, 0.690] | 0.245 | 0.360 |
| laya-ml, zero-shot | choice | 0.384 [0.314, 0.458] | 0.132 | 0.120 |

**The fine-tune is large and unambiguous.** Paired against the same checkpoint's
own zero-shot output, on the same posts:

    AP(ft.choice) - AP(zero-shot.choice) = +0.372 [+0.261, +0.486] p<0.001
    AP(ft.score)  - AP(zero-shot.score)  = +0.345 [+0.243, +0.452] p<0.001
    AP(ft.noul)   - AP(zero-shot.noul)   = +0.278 [+0.184, +0.362] p<0.001

The choice head was *anti*-correlated before training — AUC 0.384, ranking
internal posts below the rest — and is 0.830 after. It had to invert through
chance to get there, which is visible in the training log around step 600.

**It draws exactly level with the deployed vector, and that is the honest read.**

    AP(ft.noul)   - AP(jina.vector) = +0.001 [-0.104, +0.109] p=0.962
    AP(ft.choice) - AP(jina.vector) = -0.019 [-0.136, +0.106] p=0.810

Not "slightly better". Level, with a CI that would have shown a difference of
0.1 AP had there been one. Anyone reporting the +0.001 as an improvement is
reporting noise.

**It remains well below the paid teacher**: ΔAP = −0.264 [−0.366, −0.163]
p<0.001. Distillation recovered a large part of the gap and did not close it.

## What it buys that the vector cannot

Three things, none of which is rank quality.

**A three-way label.** The vector is a dot product and produces one scalar; the
axis has three classes. Against the Gemini label:

| system | 3-class acc | macro F1 | F1 int | predicted mix (gold 81/243/176) |
|---|---|---|---|---|
| jev | 0.810 | 0.762 | 0.614 | 46 / 233 / 221 |
| **laya-ft** | **0.708** | **0.624** | 0.370 | **38 / 238 / 224** |
| decider-2b | 0.506 | 0.502 | 0.540 | 145 / 81 / 274 |
| laya-ml zero-shot | 0.232 | 0.212 | 0.198 | 292 / 168 / 40 |

The zero-shot checkpoint called 292 of 500 posts internal where gold has 81. The
fine-tuned one predicts 38/238/224 against the teacher's own 46/233/221 — it has
learned the teacher's class proportions almost exactly, which is what
distillation on soft targets is supposed to do and what a hard-label fit would
not have given.

**German.** AUC on the declared-German subset and on what the shipped
`looksGerman()` detector actually routes:

| | de (n=72, 9 pos) | routed(de) (n=75, 12 pos) |
|---|---|---|
| jev.choice | 0.882 | 0.907 |
| decider-2b.choice | 0.859 | 0.864 |
| **laya-ft** (best primitive) | **0.824** | **0.870** |
| jina.vector | 0.631 | 0.706 |
| laya-ml zero-shot.choice | 0.198 | 0.130 |

The deployed vector's German weakness is already recorded and is the reason this
matters: on the posts the feed's own detector routes as German, the fine-tuned
322M model reads 0.870 against the vector's 0.706. With 9–12 positives these are
wide intervals and this is a direction to test, not a number to deploy on.

**Latency.** 0.050 s/post against the teacher's 0.624 s network round trip,
decider-2b's 0.796 s and Qwen2.5-7B's 1.495 s. 647 MB of weights, no per-call
cost.

## decider-2b, zero-shot

`Mapika/decider-2b` is Qwen3.5-2B-Base fully fine-tuned on 942k typed decision
examples, answered by reading the option-letter logits at one position. It is the
strongest open candidate measured here without any training at all: AP 0.589
against the Jina vector's 0.523 and both Laya checkpoints' 0.245/0.340.

Two findings worth carrying:

**The rubric barely matters to it** — AP 0.589 packing the rubric into the
options, 0.575 with bare labels and the rubric in the question. That is the
opposite of Laya, where the 48-token option cap was the whole story. decider
takes letters as options, so the truncation failure mode does not exist for it
and `ts_questions.py` can be sent nearly verbatim.

**It still does not clear the deployed vector, as measured**: ΔAP = +0.066
[−0.052, +0.181] p=0.267. Same verdict this gold set has now returned for the
Qwen ladder and for the fine-tuned Laya. At n=500 with 81 positives it cannot
resolve a 0.1 AP difference, and three separate systems have now landed inside
that band.

Its hybrid architecture (18 linear-attention + 6 full-attention blocks) is the
one this repository already flagged as silently breaking the MLX engine's
cache-widening trick. That hazard does not apply here: the readout is a single
forward pass with no cache reuse.

## Training decider-2b: the plan

Not run. The design, so it can be:

- **Same data, same split, same loss.** Import `build_pairs`/`split`/
  `teacher_target` from `finetune_laya_loc.py`, so both candidates see identical
  rows and identical soft targets and a difference in result is a difference in
  model.
- **Loss restricted to the candidate letters.** Inference reads
  `softmax(logits[-1][letter_ids])`; train that same K-way distribution with
  `proper_reward`. Full-vocabulary cross-entropy is the wrong objective — 248k
  vocab, spent suppressing tokens the readout never looks at.
- **LoRA, not a full fine-tune.** 1.9B params under AdamW is ~30 GB of optimiser
  state before activations, and it is the wrong capacity for 4500 posts anyway.
  r=16, alpha=32, dropout 0.05, `target_modules="all-linear"` — the blocks are
  heterogeneous, so naming projections by hand would silently skip most of the
  model. The rank cap is the regulariser.
- **Letter permutation** is the augmentation that matters: decider's training mix
  gives (A) a prior. Same exception as above — not on the ordinal question.
- **Decide in advance.** Zero-shot it is already +0.066 AP over the vector with a
  ±0.12 CI, so a modest gain lands in exactly the band this set cannot read. The
  primary target should be the German subset, where the gap is widest; and if it
  looks good, the next spend is more labelled gold, not more model.

## Running it without killing the machine

The first full run reached 29 GB of unified memory and wedged the Mac — 0.02 GB
free, 7.5 GB of 8 GB swap, the process `stuck` and only killable from outside.
Two causes, and the obvious one was second:

**Activations dominate.** The encoder forward alone is 4.20 GB at batch 8 x 338
tokens; Laya's own 2-layer head is 0.02 GB. At batch 16 that exceeds 8.7 GB on
its own. The static footprint — weights, grads, Adam moments, EMA — is only
2.7 GB, and quoting it is how the first run was mistakenly judged safe.

**Padded shape churn.** MPS pools free buffers by size and `empty_cache()` does
not hand them back. Padding each batch to its own maximum produced 173 distinct
lengths over three epochs, each permanently costing another pool.

`finetune_laya_loc.py` now refuses to start without measuring:

- `memory_cap` — `set_per_process_memory_fraction`, so an overflow is a
  `RuntimeError` at the offending allocation rather than the machine paging until
  it has to be killed from outside. This is what makes the failure survivable.
- `memory_probe` — runs the heaviest batch length bucketing can produce for a few
  *real* steps (Adam allocates moments lazily; one forward/backward under-reports
  by 2x the trainable size), then rolls those steps back by reloading the
  checkpoint. Aborts with the setting to change, not a stack trace.
- a runtime growth guard — the probe bounds one step, not a run. Every 50 steps
  it releases the cache and *then* checks; still above 80% of the cap means
  growth, and it says so.
- `LAYA_FT_PAD_MULTIPLE=32` — 173 distinct shapes become 10. Padding is masked,
  so arithmetic is unchanged; only the number of distinct buffer sizes changes.
- `LAYA_FT_GRAD_CKPT=1` — worst-case step 8.7 GB+ to 5.69 GB, and the only lever
  that changes no hyperparameter.
- EMA over trainable parameters only — the shipped code averaged mmBERT's frozen
  196.6M-row embedding matrix, whose average is itself. −0.83 GB.
- length-bucketed batching — padded/real tokens 1.369 to 1.013.

The settled run holds 6.7–7.7 GB, bounded, at 15.8 min/epoch: checkpointing's
~30% cost comes out at ~10% because the bucketing saving offsets most of it.

## Consequence

The teacher stays `jev-latest`; nothing in `training/` or `bunny_feed/` changes
on this evidence. What changed is which threads are worth pulling.

Distillation into a 322M multilingual encoder is now a live option rather than a
dead one: it reaches the deployed vector's AP at 0.05 s/post, produces the
three-way label the vector cannot, and reads German substantially better than the
vector does. It is not better at ranking, so it does not displace anything on
rank quality alone.

Best dev score was at the *final* step of four epochs — it had not converged, and
more epochs are the cheapest untried improvement.

The binding constraint is unchanged and now demonstrated three times over: at
n=500 with 81 positives this gold set cannot separate the fine-tuned Laya, the
Qwen ladder and decider-2b from the shipped dot product or from each other. A
larger labelled set is the prerequisite for any of these becoming a decision.

Checkpoint: `research/data/laya_loc/ml_compact/` (loads with `laya.Agent`
unmodified; per-qtype temperatures refit on dev to 1.203 / 1.097 / 1.047).
Scores and the JSON report are under `research/data/evals/`.

## decider-2b fine-tuned: the first system that clears the shipped vector

Run of 2026-09-19/20, `research/evals/finetune_decider_loc.py`, rubric `short`,
2 epochs x 844 steps, batch 4 x accum 4, LoRA `all-linear` fp32 over the bf16
base. 456.5 min on MPS, peak memory 4.0/5.8 GB against a 15 GB cap. Adapter at
`research/data/laya_loc/decider_short/`.

Seven evals, every one a new best, so early stopping never fired and the selected
checkpoint is the **final** one (step 1477) — the same non-convergence Laya
showed. On the teacher's dev slice `ap_choice` ran 0.5211 (zero-shot) → 0.5768 →
0.7186 → 0.7885 → 0.8145 → 0.8360 → 0.8435 → 0.8503. Those are TypeSafe-vs-
TypeSafe and measure distillation fidelity only; everything below is the Gemini
gold set, read once, after training.

### It lands on the paid teacher

| system | primitive | AUC [95% CI] | AP | P@25 |
|---|---|---|---|---|
| **jev** (`jev-latest`, paid) | choice | 0.942 [0.914, 0.964] | **0.788** | 0.920 |
| **decider_ft** (2B, local) | choice | **0.945** [0.921, 0.965] | 0.769 | 0.920 |
| qwen_7b | score | 0.868 [0.814, 0.915] | 0.631 | 0.760 |
| decider_short (its own zero-shot) | choice | 0.856 [0.808, 0.899] | 0.575 | 0.760 |
| jina (deployed vector) | dot product | 0.803 [0.742, 0.858] | 0.523 | 0.680 |
| laya_ft_ml | noul | 0.793 [0.736, 0.850] | 0.524 | 0.680 |

Paired bootstraps on the same 500 posts, AP as the lead metric:

    vs decider_short.choice  +0.194 [+0.110, +0.277] p<0.001   training worked
    vs jina.vector           +0.246 [+0.136, +0.352] p<0.001   clears what ships
    vs qwen_7b.score         +0.138 [+0.025, +0.251] p=0.017   beats 7B at 2B
    vs laya_ft_ml.noul       +0.245 [+0.148, +0.341] p<0.001
    vs jev.choice            -0.019 [-0.096, +0.054] p=0.663   not separable

**Read the last line as a null, not as parity.** The interval admits a true
deficit of up to ~0.1 AP; n=500 with 81 positives cannot exclude it. What is
established is the direction of everything else: this is the first system in this
programme to clear the deployed dot product significantly, and it does so by a
margin four times the width that defeated fine-tuned Laya, the Qwen ladder and
decider zero-shot.

### The three-class output also tracks jev

| system | acc | macroF1 | F1 int | predicted mix (gold: 81/243/176) |
|---|---|---|---|---|
| jev | 0.810 | 0.762 | 0.614 | 46 int / 233 ext / 221 neu |
| decider_ft | 0.798 | 0.758 | **0.632** | 52 / 227 / 221 |
| qwen_7b | 0.694 | 0.658 | 0.569 | 144 / 261 / 95 |
| decider_short | 0.528 | 0.523 | 0.532 | 141 / 86 / 273 |

Zero-shot decider over-fired `internal` at 141 against a gold 81; fine-tuning
moved the mix to 52, within six posts of jev's own. F1 on the internal class is
marginally the best in the table. This matters beyond ranking: the vector emits
no class at all, so a usable three-way label was previously only purchasable.

### German, which was the point

AUC on the three German cuts — the deployed vector's weakest ground:

| cut | n / pos | decider_ft | jev | jina |
|---|---|---|---|---|
| declared `de` | 72 / 9 | 0.956 | 0.882 | 0.631 |
| `de` + detected | 83 / 11 | 0.965 | 0.901 | 0.682 |
| `looksGerman()` routed | 75 / 12 | 0.960 | 0.907 | 0.706 |

decider_ft is numerically ahead of the paid teacher on every German cut. Paired
AP says do not bank that: +0.195 p=0.201 declared, +0.170 p=0.166 with detection,
+0.111 p=0.272 routed — nine to twelve positives cannot resolve it. Against the
vector the same cuts are +0.445 p=0.004, +0.465 p=0.001 and +0.419 p<0.001, which
is resolvable and large.

### Cost

    jev         p50 0.624s   $0.0489 / 1k posts
    decider_ft  p50 0.905s   $0       (local weights)

1.5x the wall-clock per post, and the per-call cost goes to zero: ~4.0k posts/hour
on one machine, so the 5500-post corpus labels in about 1.4 h.

### Consequence, and what it supersedes

The closing paragraph of the previous section no longer holds as written. The
claim was that at n=500/81 positives this gold set cannot separate any open model
from the shipped dot product. decider-ft separates from it at p<0.001. The
constraint was real for the three systems it was written about, and it was not a
property of the gold set alone — it was a property of those systems.

What the gold set still cannot do is separate decider-ft from jev, and that is
now the question that matters, so more labels remain the next spend — but to
decide a **replacement**, not to rescue a null.

Against the second decision rule in `docs/runbooks/RESUME_decider_finetune.md`:
gold AP 0.769 clears the ~0.63 trigger by a wide margin, so decider-ft is the
local teacher candidate and the distill A/B is the next step. Note what it may
and may not be asked: it is a student of jev on those same 5500 posts, so
re-labelling that corpus with it measures nothing. Its value is on posts jev was
never paid for, and the German cuts above are where it is most likely to be real.
