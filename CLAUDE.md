# Working in this repository

One feed is in production: **`feed-forward`**, the locus-of-control axis, served
from `bunny_feed/` on Bunny Edge. Everything tracked here exists to produce or
measure that feed. That is the organising rule, and the rest of this file is
mostly consequences of it.

## Layout

```
training/     the pipeline that produces the two deployed vectors
  data/       corpora, embedding caches, vectors, sweep results  (gitignored)
research/     measurement and studies that inform the pipeline but do not ship
  data/       their inputs and outputs                            (gitignored)
bunny_feed/   the edge worker. THIS is production.
local/        the local backend + frontend used for annotation and review
docs/         reports/ (findings) and runbooks/ (procedures)
archive/      superseded work, kept on disk, NOT tracked            (gitignored)
```

**Where new work goes.** If it produces the deployed vectors, `training/`. If it
measures or investigates, `research/<topic>/`. If it is prose, `docs/`. Nothing
new belongs in the repository root — the root holds `README.md`, `CLAUDE.md` and
configuration, and that is all it has ever needed to hold.

**Where old work goes.** `archive/`, not the trash. When something stops being
the current pipeline — a superseded teacher, a scrapped axis, an experiment that
answered its question — move it there in the same change that supersedes it. It
is untracked, so it costs nothing and survives locally. The failure mode this
prevents is the one that actually happened: three teachers' corpora sitting side
by side in the root with nothing marking which was current, and a full A/B run
against the wrong one.

## The axes are one pipeline, not several

`training/label_prompts.py` holds the rubric for every axis. An axis **is** its
prompt — the corpus collection, the annotator, the distillation and the export
are identical, parameterised by `LOC_PROMPT_VERSION`. Adding an axis means
adding a prompt, never forking the pipeline. Two axes have been tried and
retired this way (`assist`, `virtue`); both are in `archive/`.

TypeSafe evaluates each question in isolation against one `state`, and bills the
state once per call, so several axes can ride in a single request per post —
see `training/ts_questions.py`. Labelling a corpus on three axes costs barely
more than labelling it on one. Do that rather than three passes.

## Things that will bite you

**Embedding caches are positional.** `training/data/embeddings_*.npz` have no
keys — row *i* of `X` is row *i* of whatever `load_train_rows()` returned. A
change to the filter, the order, or the source file silently misaligns every
target with the wrong text and still trains. `build_typesafe_teacher.py`
preserves order and count for exactly this reason. Verify alignment before
trusting a number, not after.

**A threshold never survives a retrain.** It is a sigmoid probability, and each
retrain moves the distribution underneath it. The LoC gate has been 0.55 (which
rejected literally nothing), 0.88, and is now 0.86. Recalibrate against the pools
in `bunny_feed/src/feeds.ts` — corpus, live candidates, validation — and record
what you measured. Same for any percentile: it describes the pool it was measured
on, and this deployment draws two different ones.

**Validation labels are a model's output, not ground truth.** The gold set is
Gemini's. Distilling from a teacher and then validating against the same teacher
measures distillation fidelity and nothing else. Keep the training teacher and the
validation labeller distinct, and say which is which in any number you report.

**Every TypeSafe call is billed, so nothing is asked twice.**
`training/data/typesafe_ledger.jsonl` is the record of every post ever scored —
one row per post, holding the full answer, shared across every corpus, feed and
tag. `score_loc_typesafe.py` asks only for what is missing from it and appends
each answer the moment it arrives, so a killed run keeps what it paid for. It is
gitignored like the rest of `training/data/`, and rebuildable from the score
files with `--rebuild-ledger`. Before the ledger existed the same 500 gold posts
had been bought four times over.

**Judge this axis on AP and P@25, not AUC.** The feed consumes the top of a
ranking and never sees the tail. The two metrics have already disagreed in sign
on a real retrain.

## Running it

Scripts resolve `training/data/` against their own location, so they work from
any cwd. Defaults describe the *current* pipeline — `python
training/train_internal_vector.py` with no environment set reproduces the
deployed Jina vector. Env vars override for experiments (`LOC_TEACHER_FILE`,
`LOC_EMB_CACHE`, `LOC_VAL_EMB_CACHE`, `LOC_VECTOR_OUT`, …).

MiniLM caches must be pre-built by `training/embed_minilm.py`; the trainer
computes Jina embeddings for any cache it finds missing, which would silently
fill a MiniLM cache with 1024-dim Jina vectors.

## Deploying

Push to `main` touching `bunny_feed/**`. That is the deploy — `.github/workflows/deploy.yml`
migrates the DB and ships the bundle. There is no separate step and no staging,
so a vector export and its threshold recalibration belong in the same commit.
`bunny_feed/src/vector.prev.json` is the local rollback copy.
