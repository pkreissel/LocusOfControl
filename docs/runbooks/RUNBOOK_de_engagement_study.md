# German Bluesky: Perspective scores vs interaction counts

A sampling frame, a dataset, and an analysis path for the question *which
Perspective attribute predicts which interaction count on German posts*.

## Why not the existing collector

`2_stream_posts.py` buffers a post and writes it out **only once it receives a
like or a reply**. That conditions the sample on the outcome under study: posts
with zero engagement — the majority, and the interesting comparison group — never
enter the file, and the recorded engagement is "at least one", not a count. It is
the right tool for mining training examples and the wrong one here, so this study
uses a separate collector.

## Pipeline

```
collect_de_posts.py      census of de-declared posts over the Jetstream window
  -> de_pool_raw.jsonl

build_study_sample.py draw   uniform random draw from the oldest 24h
  -> de_study_sample.jsonl

score_perspective_de.py      23 Perspective attributes per post
  -> de_perspective.jsonl

hydrate_engagement.py        AppView counts + author covariates, as snapshots
  -> de_engagement_snapshots.jsonl, de_authors_snapshots.jsonl

validate_perspective_de.py   which attributes mean anything on German
  -> REPORT_perspective_de_validity.json

build_study_sample.py table  join everything into one analysis table
  -> de_study_table.csv / .jsonl

analyze_de_engagement.py     raw / partial / within-author associations
  -> REPORT_de_engagement_findings.md

analyze_de_diurnal.py        scores and engagement by local hour
  -> REPORT_de_diurnal.md

compare_de_samples.py        does each association reproduce in root posts?
  -> REPORT_de_sample_comparison.md
```

All scripts use `.venv_de` (the repo's `.venv` has a dangling interpreter symlink;
`.venv_de` was created fresh with `uv` and does not touch it).

## What was collected (run of 2026-09-08)

| | |
|---|---|
| Census | 169,022 `de`-declared posts over 34.7h, from 4,577,660 post creates |
| Analysis window | 2026-09-07 01:54 -> 2026-09-08 01:54 UTC (24.0h) |
| Posts in window | 112,172; eligible after the language rule 101,318 |
| Drawn | 11,000 (seed 20260908, sampling fraction 10.9%) |
| Hydrated | 10,684 available, 316 deleted (2.9%) |
| **Analysis table** | **10,000 rows**, 5,387 authors, 1,947 with >=2 posts (6,560 rows) |
| Perspective | 21 attributes retained of 23; 0 scoring errors |
| Snapshot | `early`, post ages 12.3-36.3h — **not settled** |

Composition: 67% replies, 84% detector-confirmed German, 16.8% posts under 20
characters. Engagement is heavily right-skewed — 27% zero likes, median 1,
p99 = 80, max = 1,762. `quote_count` is 97.6% zero and carries almost no power as
an outcome; `like_count` and `total_engagement` are the usable ones.

Note that `de_pool_stats.json` reports `window_start` from the resumed cursor of
the final collector process, not from the first row of the pool file; the true
pool span is in `de_study_sample_meta.json`.

## Sampling frame

**Population.** Every `app.bsky.feed.post` create whose record declares a `de*`
language, seen on Jetstream over a fixed 24-hour window.

**Why 24 hours exactly.** Engagement varies strongly by time of day, and so does
what people post about. A window that covers only part of the diurnal cycle
confounds any predictor correlated with posting hour. Twenty-four hours covers
every hour once.

**Why the *oldest* 24h.** Jetstream retains ~36h and a cursor beyond that silently
clamps, so the oldest 24h is the slice whose posts are furthest along their
engagement curve at hydration time.

**Known frame limitation.** `record.langs` is declared by the posting client — in
practice its UI language setting — not detected. So the frame both admits English
posts from users with a German-configured client and misses German posts from
users with an English one. Measured, not assumed: the collector's 1-in-200 audit
of *non*-`de` posts found 0.88% of them to be German, implying the frame **misses
roughly 16% of all German posts** — and misses them non-randomly, since those
authors are the ones running an English-configured client. The population this
sample represents is therefore *posts declared German*, not *all German posts*.

Both directions are instrumented:

- every row carries `det_lang` / `det_conf` from a language detector (lingua,
  restricted to 14 languages, which is more accurate than the full model on short
  text), so the sample can be restricted to detector-confirmed German post hoc;
- the collector language-detects a 1-in-200 subsample of the *non*-`de` posts and
  reports the estimated miss rate in `de_pool_stats.json`.

**Language rule applied to the draw.** Of raw `de`-declared posts, only 74% detect
as German and 13% detect as English — enough contamination for an attribute to
predict engagement purely through the language channel. The frame therefore
excludes posts the detector *confidently* assigns to another language
(conf >= 0.70), rather than *requiring* confident German. The asymmetry is
deliberate: requiring confident German would have removed 34% of the sample and
cut posts under 20 characters from 16% to 2.9%, trading a language bias for a
length bias. The rule as applied keeps 90% of the frame, cuts English to 5.3%, and
leaves the short-post stratum intact at 16.8%.

**Draw.** Uniform random, seed 20260908, oversampled to 11,000 for a 10,000-row
analysis set. The draw is stored in random order (`draw_rank`), and the analysis
set is the first 10,000 rows that survive hydration — which keeps it a uniform
random sample of *surviving* posts while making attrition visible. Posts deleted
before hydration are recorded as `deleted_or_unavailable` rather than silently
skipped, because deletion plausibly correlates with toxicity and dropping them
quietly would bias the very estimate under study. Observed deletion rate is ~3%.

## Interaction counts

Counts come from the AppView aggregates
(`app.bsky.feed.getPosts`, 25 uris/request, unauthenticated on
`public.api.bsky.app`): `likeCount`, `repostCount`, `replyCount`, `quoteCount`.

**Not from the firehose.** Counting like events on Jetstream would only capture
likes cast inside the retention window, truncating each post's tally by where it
sits in that window — a bias perfectly correlated with post age.

**Snapshots, not a single pass.** Engagement is strongly age-dependent, so counts
are written as labelled snapshots with a `measured_at`, one row per
(uri, snapshot). Two passes:

| snapshot | when | post ages | use |
|---|---|---|---|
| `early` | at collection | ~4-28h | growth-curve reference; rank-stability check |
| `settled` | >= 48h after the window ends | 48-72h | **the analysis snapshot** |

Retention is only ~36h, so there is no way to collect *and* measure settled counts
in one sitting — the second pass is a separate run of the same command. It must
cover **both** samples; posts common to the two are fetched once, because the
hydrator resumes on (uri, snapshot):

```
# after 2026-09-10 01:54 UTC, when every sampled post is >= 48h old
.venv_de/bin/python hydrate_engagement.py --in de_study_sample.jsonl      --label settled
.venv_de/bin/python hydrate_engagement.py --in de_study_sample_root.jsonl --label settled

.venv_de/bin/python build_study_sample.py table --snapshot settled \
    --sample de_study_sample.jsonl --out de_study_table.csv \
    --out-jsonl de_study_table.jsonl --summary-out de_study_table_summary.json
.venv_de/bin/python build_study_sample.py table --snapshot settled \
    --sample de_study_sample_root.jsonl --out de_study_table_root.csv \
    --out-jsonl de_study_table_root.jsonl --summary-out de_study_table_root_summary.json

.venv_de/bin/python analyze_de_engagement.py --table de_study_table.csv
.venv_de/bin/python analyze_de_engagement.py --table de_study_table_root.csv \
    --out REPORT_de_engagement_findings_root.json \
    --md-out REPORT_de_engagement_findings_root.md
```

## Two samples

| | primary | root-only |
|---|---|---|
| file | `de_study_sample.jsonl` | `de_study_sample_root.jsonl` |
| table | `de_study_table.csv` | `de_study_table_root.csv` |
| frame | all posts in the window | root posts only (`--exclude-replies`) |
| eligible | 101,318 | 33,398 |
| seed / fraction | 20260908 / 10.9% | 20260909 / 32.9% |
| composition | 67% replies, 16.8% short | 0% replies, 7.4% short |
| snapshot label | `early` | `early_root` |

**The primary sample is the population-representative one** and is what a claim
about "German Bluesky posts" should rest on. The root-only sample exists because a
reply's engagement is driven largely by the thread it sits in — the parent's reach
and the conversation's size are not in the table and cannot be conditioned on — so
root posts give a cleaner exposure model at the cost of describing only a third of
the population.

**The two samples are not independent.** They are drawn from overlapping
populations over the same window, and a root post can appear in both. Do not pool
them, and do not read agreement between them as independent replication — treat
the root-only result as a robustness check on the subset where exposure is better
measured.

## Covariates

Reach dominates raw engagement, so the table carries what is needed to condition
on it: `author_followers`, `author_follows`, `author_posts_total`,
`author_account_age_days`, and `author_posts_in_window` (how many posts the author
made in the whole 35h census — an activity measure the API does not expose).
Post-level: `is_reply`, `is_self_reply`, `n_images`, `has_video`, `has_external`,
`n_links`, `n_mentions`, `n_tags`, `text_len`, `hour_utc`, `weekday`, and
`age_hours_at_measure`.

`created_utc` is derived from the firehose ingest timestamp (`time_us`), not from
the record's `createdAt`, which is client-supplied and occasionally wrong.

## Perspective attributes on German text

23 attributes are requested in one call each. German text is submitted as
`languages=["en"]`: the parameter is a hint to one multilingual model rather than
a model selector, and 17 of the 23 attributes reject `de` with a 400 outright.
Verified on this corpus, not assumed — scoring the same German texts under `de`
and under `en` gives **r >= 0.9999 on all six core attributes**.

Returning a number is not the same as measuring the construct, so
`validate_perspective_de.py` screens every attribute and sorts it into three tiers:

- **drop — language artifact.** `INCOHERENT` and `LIKELY_TO_REJECT` saturate high
  on German (96% / 91% above .9) while overlapping comparable English text by only
  0.16 / 0.24. They are scoring *"this is not English"*. Excluded from the table.
- **low_power — valid but rare.** Attributes whose positive rate implies fewer than
  100 cases above 0.5 at n=10,000. A correct THREAT detector *should* be near zero
  on a random sample of German posts. These are kept and flagged, so a null result
  reads as "no power" rather than "no effect".
- **keep — usable.** Everything else.

Re-run the screen whenever the corpus changes; the verdicts are corpus-dependent.

## Analysis

`analyze_de_engagement.py` reports three columns per attribute x outcome:

1. **raw** Spearman — descriptive only, dominated by author reach.
2. **partial** — both sides residualised on the control block. The
   "same-sized account, same kind of post" comparison.
3. **within-author** — author fixed effects over authors with >= 2 posts in the
   sample. Removes *all* stable author-level confounding and asks whether an
   author's higher-scoring posts beat their own average. **This is the column to
   believe.** A 24h census yields many repeat authors, which is what makes it
   possible.

Standard errors are clustered by author (posts by one author are not independent).
p-values are FDR-corrected across the whole attribute x outcome grid — with ~20
attributes and 5 outcomes, uncorrected significance is meaningless.

## What this design does not establish

Associations, not causal effects. Nothing is randomised; reverse causation
(engagement shaping later posting behaviour) is not excluded; and the counts
reflect whatever the Bluesky ranking and moderation stack did to these posts,
which is itself a function of their content.
