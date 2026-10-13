# Resume: fine-tuning decider-2b on the LoC axis

**DONE — run completed 2026-09-20 05:00, results in `docs/reports/REPORT_laya_finetune.md`.**
Adapter at `research/data/laya_loc/decider_short/`, best at the final step (1477),
456.5 min. Gold AP 0.769 against jev's 0.788 (−0.019, p=0.663), +0.246 over the
deployed vector (p<0.001). Gold scored, `decider_ft` added to `SYSTEMS`, eval
re-run. The second decision rule below is triggered: the distill A/B is next.
The procedure is kept below because it is the recipe, not because anything is
pending.

State as of 2026-09-19, written before a machine restart. Nothing is mid-flight —
every process was killed cleanly and no partial checkpoint was written.

## Where things stand

**Done, and reported in `docs/reports/REPORT_laya_finetune.md`:**

- `research/evals/finetune_laya_loc.py` — Laya fine-tune, finished. Checkpoint at
  `research/data/laya_loc/ml_compact/`, scored on gold, in the eval.
  Headline: AP +0.372 vs its own zero-shot (p<0.001), lands level with the
  deployed Jina vector (+0.001, p=0.962), still below the paid teacher.
- `research/evals/score_loc_decider.py` — decider-2b scorer. Zero-shot gold scores
  already written for both rubrics:
  `research/data/evals/scores_loc_decider_{short,compact}_gold.jsonl`.
- `research/evals/eval_laya_vs_jev.py` — already lists `decider_short`,
  `decider_compact` and `laya_ft_ml` in `SYSTEMS`.

**Written, probed, NOT yet run:** `research/evals/finetune_decider_loc.py`.

## The one command to run

    cd research/evals
    HF_HOME=/Volumes/Daten/huggingface \
    DEC_FT_RUN=decider_short DEC_FT_RUBRIC=short \
    DEC_FT_EPOCHS=2 DEC_FT_BATCH=4 DEC_FT_ACCUM=4 DEC_FT_GRAD_CKPT=1 \
    DEC_FT_MEM_FRACTION=0.6 DEC_FT_EVALS_PER_EPOCH=4 DEC_FT_PATIENCE=3 \
    ../../tmp_venv/bin/python finetune_decider_loc.py

Run it detached and tee the log; it is long.

**Expect ~6.4 hours** (~3.2 h/epoch, 2 epochs), possibly less — early stopping is
on (4 evals/epoch, patience 3). Throughput is compute-bound on MPS and flat
across batch sizes: batch 4, 8 and 16 all land at 191–203 min/epoch, so the batch
was chosen for its smaller footprint, not for speed. Peak memory 5.65 GB against
a 15 GB cap.

Re-probed 2026-09-19 before the actual run: same memory (5.65 GB peak, matching
below) but a 4.88 s median micro-batch rather than ~3.6 s, projecting ~9.2 h for
2 epochs. Same config, so read this as machine load, not a changed setup — but
budget for the wider range.

Sanity-check first if you want, it takes about a minute and trains nothing:

    DEC_FT_MEM_PROBE_ONLY=1 ... ../../tmp_venv/bin/python finetune_decider_loc.py

It should print `worst-case batch: 4 x 448 tokens -> peak 5.65 GB of 15.0 GB cap
(38%)` and a median micro-batch around 3.6 s.

## Then, after it finishes

1. Score the gold set with the adapter (~7 min):

        LOC_DECIDER_ADAPTER=/Volumes/Daten/Datenprojekte/LocusOfControl/research/data/laya_loc/decider_short \
        LOC_DECIDER_RUBRIC=short HF_HOME=/Volumes/Daten/huggingface \
        ../../tmp_venv/bin/python score_loc_decider.py \
          dataset_validation_gemini.jsonl scores_loc_decider_ft_decider_short_gold.jsonl

   The trainer prints this command with the right paths when it finishes.

2. Add one line to `SYSTEMS` in `eval_laya_vs_jev.py`:

        ("decider_ft", os.path.join(OUT_DATA, "scores_loc_decider_ft_decider_short_gold.jsonl")),

3. Re-run the eval, then extend `REPORT_laya_finetune.md`:

        HF_HOME=/Volumes/Daten/huggingface ../../tmp_venv/bin/python eval_laya_vs_jev.py

## What to look at, and the decision rule set in advance

The comparisons that mean anything, all paired bootstraps on the same 500 posts:

- **vs its own zero-shot** (`decider_short.choice`, AP 0.575) — did training do
  anything at all. This is the analogue of Laya's +0.372.
- **vs `jina.vector`** (AP 0.523) — the bar. Zero-shot decider is already +0.066
  [-0.052, +0.181] p=0.267, i.e. inside the noise. A modest gain lands in exactly
  the band this gold set cannot read.
- **vs `jev.choice`** (AP 0.788) — the paid teacher, still the thing to beat.
- **the German subset** — this is the primary target, not the headline AP. The
  deployed vector reads 0.631 on declared-German and 0.706 on what
  `looksGerman()` routes; decider-2b zero-shot reads 0.859/0.864 and the
  fine-tuned Laya 0.824/0.870. German is where the deployed vector is weakest and
  where these models have shown the largest gap.

Three systems have now landed inside the unresolvable band against the shipped
dot product (fine-tuned Laya, the Qwen ladder, decider-2b zero-shot). If
decider-ft does too, the conclusion is not "try a bigger model" — it is that at
n=500 with 81 positives the gold set is the binding constraint, and the next
spend is more labels.

## The second decision rule: decider-ft as a local teacher

The rule above answers "is this a better ranker than what ships". That is not
the only question being asked of this run. The standing goal is to label posts
locally and stop paying `jev-latest` per post, and that goal is scored
differently — against the *student* a teacher produces, not against the teacher's
own gold AP.

**Why the bar is lower than it looks.** Nothing in production calls jev. The feed
scores with `internal_vector.npy`, a dot product, local and free. jev's only job
is teaching. And the student lands far below its teacher either way — AP 0.523
from a 0.788 teacher — so a local teacher does not have to match jev. It has to
be good enough that the distillation floor does not drop.

**That cost has already been measured once**, in `REPORT_typesafe_distill_ab.md`,
same corpus, teacher swapped:

| teacher | student vector AP |
|---|---|
| `meta/muse-glimmer` (local) | 0.436 |
| TypeSafe `jev-latest` (paid) | 0.523 (+0.087 [+0.021, +0.143] p=0.007) |

So dropping to a local teacher cost 0.087 AP with the local teacher of the day.
Every candidate on the current ladder is stronger than glimmer, and Qwen2.5-7B
(gold AP 0.631) already scores above the 0.523 student it would be training.

**The rule.** After the gold eval:

1. If decider-ft's gold AP clears ~0.63 — Qwen2.5-7B's level, at 2B parameters
   instead of 7B — it is the best local teacher candidate and the distill A/B is
   worth running.
2. The A/B is the decision, not the gold AP: label the corpus with decider-ft,
   train the vector on those targets, and paired-bootstrap that student against
   the deployed jev-taught one. Same harness that produced the table above.
3. Student within noise of 0.523 → the local teacher is viable; stop buying
   corpus labels. Student falling toward glimmer's 0.436 → the teacher gap
   propagates; keep jev for labelling.

**What the local teacher is actually for, so this is not run on the wrong posts.**
decider-ft is itself distilled from jev on those same 5500 posts, so its labels
carry no information about *that corpus* the vector does not already have.
Re-labelling the existing corpus with it measures nothing. Its value is on posts
jev was never paid for — corpus expansion — and it is worth only as much as it
generalises past the vector. German is where that is most likely to pay: decider
zero-shot reads 0.859/0.864 there against the deployed vector's 0.631/0.706.
Validation stays Gemini's gold set, distinct from both, per CLAUDE.md.

**Throughput is not the constraint, so do not optimise it.** Measured p50 per
post on the gold set: decider-2b 0.86 s (4.2k/hour), qwen 1.5B 0.30 s
(11.8k/hour), laya-ft 0.05 s (72k/hour). Labelling a 5500-post corpus with
decider is ~1.3 hours. Only live firehose scoring would need more, and the vector
already does that job.

**The cascade alternative, measured and not chosen.** Using a local model to
prefilter and forwarding only survivors to jev buys less than it sounds like —
the fraction of the pool you must still forward to keep a given share of
internal posts, from the existing gold score files:

| system | keep @95% recall | keep @90% recall |
|---|---|---|
| jev (ceiling) | 30.8% | 26.2% |
| decider-2b zero-shot | 64.2% | 47.6% |
| qwen 7B | 74.8% | 45.6% |

About 2x saved for 10% of internal posts lost. Recompute these before building
anything on them: they describe the gold pool, which is 16% internal, and the
live pool is not.

## Things that will bite you, all of them already paid for once

**The gold set is inside the training corpus.** All 500 posts of
`dataset_validation_gemini.jsonl` are in `dataset_annotated_typesafe_g3.jsonl`,
498 by text as well. `L.build_pairs()` drops them and asserts it did — do not
"optimise" that away. Any new corpus/validation pair deserves the same check.

**Dev numbers in the log are the teacher's, not gold.** They measure
distillation fidelity. The gold set is read once, by the eval, at the end.

**Do not run two training jobs at once, and do not trust `ps`.** MPS allocations
do not count toward RSS — `ps` reported 0.11 GB for a run holding 29 GB. Read
the `mem=` field the trainer prints at each eval instead.

**If it dies with `MPS backend out of memory`, that is the guard working**, not a
bug. The cap makes an overflow a Python exception instead of paging the machine
into a state that needs an external kill. Lower `DEC_FT_BATCH` (raise
`DEC_FT_ACCUM` to match so the effective batch is unchanged) or lower
`DEC_FT_MAX_LEN`.

**Memory should sit flat around 5.7–7 GB.** The trainer aborts by itself if it is
still above 80% of the cap right after a cache release, which is the signature of
allocator growth rather than one big batch.

## Why the trainer looks the way it does

Short version, so nothing here gets "simplified" back into a bug:

- `build_pairs` / `split` / `teacher_target` are imported from
  `finetune_laya_loc.py` on purpose, so both candidates train on identical rows
  with identical soft targets and a difference in result is a difference in model.
- **Soft targets, not labels** — the teacher's full distributions, under
  `laya.common.proper_reward` (log + spherical, plus a ranked-probability term on
  the ordinal question). Same loss for both models, deliberately.
- **Loss restricted to the candidate letters.** Inference reads
  `softmax(logits[-1][letter_ids])`; full-vocabulary cross-entropy would be a
  different objective over a 248k vocabulary.
- **`logits_to_keep=1`.** Only the last position is ever read. Without this the
  logits tensor is [4, 448, 248320] — 1.78 GB in fp32 — and the run does not fit.
- **Left padding needs explicit `position_ids`.** A raw forward numbers positions
  0..L-1 regardless of padding, shifting short prompts' RoPE positions by the pad
  width. Derived from the attention mask in both the trainer and the scorer.
- **LoRA fp32 over a bf16 base.** bf16 carries ~3 decimal digits and an Adam step
  at lr 1e-4 on a bf16 weight rounds away to nothing.
- **Letter permutation on `choice` only.** decider's training mix gives (A) a
  prior. Not on the ordinal question, where permuting would destroy the reading
  that the RPS term and `sum(i * p_i)` both depend on.
- **`target_modules="all-linear"`** because the model is hybrid — 18
  linear-attention and 6 full-attention blocks, not named alike — so a hand-written
  target list would silently skip most of the network.

## Optional, if the 6.4 hours is too much

- `DEC_FT_EPOCHS=1` — ~3.2 h. decider starts strong (AP 0.589) so one pass may be
  enough, but note the Laya run's best dev was at its *final* step, so
  undertraining is the more likely error.
- Train `loc_class` alone for ~2.2 h by cutting `QUESTIONS` down in
  `finetune_decider_loc.py`. Fastest route to the primitive that would actually
  ship, at the cost of the multi-task signal and of being able to report
  `score`/`noul` for the fine-tuned model.
