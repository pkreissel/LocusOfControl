# Locus of Control — Bluesky Feed

Ranks Bluesky posts by the **locus of control** they implicitly foster in a
reader — `internal` (agency, effort, learnable skill, constructive ownership),
`external` (helplessness, blame of distant powers, negative news without
solutions), or `neutral` (no control/agency framing) — and serves a feed built
on those scores.

Live: **`feed-forward`**, served from `bunny_feed/` on Bunny Edge.

## How it works

A language model labels a corpus on the rubric in `training/label_prompts.py`.
Those labels are distilled into two small vectors that can be evaluated at the
edge for nothing:

| | space | role |
|---|---|---|
| `bunny_feed/src/vector_minilm.json` | all-MiniLM-L12-v2, 384-dim | free first pass over Green Earth's hydrated embeddings; ranks the candidate pool down to a shortlist |
| `bunny_feed/src/vector.json` | Jina v5 text-small, 1024-dim | paid second pass; the quality gate (`threshold` in `feeds.ts`) |

German candidates bypass the free stage — MiniLM is English-only on this axis —
and the two scores are rank-fused for the final order. The reasoning behind each
of those choices is documented inline in `bunny_feed/src/feeds.ts`, which is the
most useful single file in the repository.

Teacher: **TypeSafe System One** (`training/score_loc_typesafe.py`), which
replaced a local llama.cpp model in September 2026. It reproduces the previous
teacher's labels at 0.925 AUC, runs at ~40 posts/s with no quota, and — used
directly rather than distilled — scores the axis at 0.942 AUC against 0.803 for
the vector it trains. That gap is the largest known loss in the system; see
`docs/reports/REPORT_typesafe_distill_ab.md`.

## Layout

```
training/     the pipeline that produces the deployed vectors
research/     measurement and studies that do not ship
bunny_feed/   the edge worker — production
local/        local backend + frontend for annotation and review
docs/         reports (findings) and runbooks (procedures)
archive/      superseded work, kept on disk, untracked
```

`CLAUDE.md` states the rules that keep it that way, and is worth reading before
adding anything.

## Running it

```bash
training/run_all.sh                        # full pipeline, resumable
python training/train_internal_vector.py   # just the Jina vector, current defaults
python training/embed_minilm.py            # rebuild MiniLM caches (required before
                                           # training in that space)
```

Scripts resolve `training/data/` against their own location, so cwd does not
matter. Defaults describe the current pipeline; environment variables override
for experiments (`LOC_TEACHER_FILE`, `LOC_EMB_CACHE`, `LOC_VECTOR_OUT`, …).

**`.env`** needs `TYPESAFE_API_KEY` (teacher), `BSKY_HANDLE` / `BSKY_APP_PASSWORD`
(collection) and `GAI_API` (Gemini validation labels; free tier is 500/day and the
validation step needs ~500). `HF_HOME` should point at your HuggingFace cache.

## Deploying

Push to `main` touching `bunny_feed/**`. GitHub Actions migrates the database and
ships the bundle; there is no separate deploy step. A retrained vector and its
recalibrated threshold belong in the same commit — a threshold is a sigmoid
probability and never survives a retrain unchanged.
