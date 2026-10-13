# Gender bias in the LoC gate

**Question.** Does `feed-forward` treat authors differently by gender?

**Answer.** Slightly, and in favour of women. Posts by female-named authors clear
the deployed gate 5.4 points more often than posts by male-named authors
[+2.2, +8.5]. About half the gap is inherited from the teacher, and the vector
adds the rest. The gap is English-only: German shows none.

Measured 2026-09-24 on the net20k corpus
(`training/data/dataset_annotated_decider_net20k.jsonl`: 20,000 posts, 5,066
authors, decider-labelled 1,960 internal / 9,066 external / 8,974 neutral),
scored with the deployed vector (`bunny_feed/src/vector.json`) at the production
gate of 0.86. Scripts: `research/gender_bias/fetch_profiles.py`,
`infer_gender.py`, `analyze_gender_bias.py`. Output:
`research/data/gender_bias/analysis.txt`.

## Method

**Gender is inferred, not known.** Each author's public profile was fetched
from the AppView (`app.bsky.actor.getProfiles`; 5,058 of 5,066 available).
Gender is read from two signals, in order of precision:

1. pronouns stated in the bio or display name (she/her, sie/ihr, elle; he/him,
   er/ihm, il/lui; they/them and neopronouns → non-binary);
2. the first given name in the display name, looked up in `gender_guesser`
   (which covers German and most European names), skipping titles like "Dr".

The handle is not used because it is too often a nickname. Everything else is
`unknown`.

| label | source | authors |
|---|---|---|
| male | name | 1,936 |
| male | pronouns | 226 |
| female | name | 724 |
| female | pronouns | 170 |
| non-binary | pronouns | 75 |
| unknown | — | 1,935 (38%) |

A spot check of 36 name-based labels found none clearly wrong; "Riley" was the
only ambiguous name. The 16 pronoun-based labels checked were not fully
verified, because the matched pronoun often sat beyond the part of the bio
that was read. The unknown group is
mostly organisations, outlets and nicknames. A first name measures how an author
*presents*, not their gender; every number below should be read that way.

**Scores come from the cached embeddings.** `embeddings_union_jina.npz` is
positional over `load_train_rows()` of `dataset_annotated_union_jev_decider.jsonl`
(25,500 rows minus the 500 gold posts). Alignment was verified before use: six
rows re-embedded with the local Jina weights gave a diagonal cosine of 0.9999
against the cache. Rows are matched by `(uri, text)`, not by URI alone, because
one URI appears twice in the union file with an edited text. The probability is
`sigmoid(scale · cos + bias)`, the same as `vector.ts`.

**CIs are an author-clustered bootstrap** (2,000 resamples). Authors post many
times, so resampling posts would overstate precision.

## Results, all languages

| group | posts | authors | teacher internal | gate pass [95% CI] | pass, teacher-internal | pass, other | top 5% |
|---|---|---|---|---|---|---|---|
| female | 2,884 | 894 | 11.2% | **51.8%** [49.3, 54.2] | 83.2% [78.5, 88.0] | 47.9% | 5.6% |
| male | 7,238 | 2,162 | 8.8% | **46.4%** [44.4, 48.3] | 78.0% [74.5, 81.4] | 43.4% | 4.6% |
| non-binary | 274 | 75 | 14.6% | 51.8% [44.9, 59.0] | 92.5% [83.3, 100] | 44.9% | 8.0% |
| unknown | 9,604 | 1,935 | 10.0% | 46.7% [44.9, 48.4] | 83.4% [80.9, 86.0] | 42.6% | 5.0% |

Female minus male:

| metric | difference | 95% CI |
|---|---|---|
| gate pass | +5.4 pp | [+2.2, +8.5] |
| teacher internal | +2.3 pp | [+0.3, +4.7] |
| gate pass, teacher-internal posts | +5.2 pp | [−0.4, +11.2] |

The gate passes 47.4% of this pool, so it is a loose floor. What readers see
depends more on the head of the ranking. There, female-named authors hold 1.12×
their share of the top 5% and male-named authors 0.92×.

## Inherited or added?

The teacher labels women's posts internal more often, so some of the gate gap is
simply the vector reproducing its training targets. To isolate what the vector
adds, the score logit was regressed on the teacher's soft labels (P(internal),
P(external)), language (en, de) and a female indicator, over female- and
male-named authors:

| term | coefficient | 95% CI (author bootstrap) |
|---|---|---|
| female | +0.063 logit | [+0.022, +0.107] |

That is 0.07 SD of the score logit (SD 0.941; the gate sits at logit 1.815).
It is small but not zero, and it shows up on both sides of the teacher label:
83% vs 78% on teacher-internal posts, 48% vs 43% on the rest.

This regression does not prove the vector responds to gender as such. The
likelier mechanism is style: first-person, reflective writing that the vector
rewards but that the teacher's 3-class label does not fully capture, and that is
more common among female-named authors. Holding the soft labels fixed only
partly separates the two.

## By language

| language | posts | overall pass | F pass | M pass | F − M pass [95% CI] | F − M teacher internal |
|---|---|---|---|---|---|---|
| English | 11,516 | 48.6% | 53.4% | 46.5% | **+7.0 pp** [+3.3, +10.8] | +3.8 pp [+1.1, +6.8] |
| German | 3,885 | 50.8% | 50.3% | 48.8% | +1.5 pp [−4.7, +7.9] | −0.7 pp [−3.6, +2.1] |

In German, the teacher labels internal at the same rate for both groups
(8.8% vs 9.5%), and the gate follows it. The whole effect sits in English, where
both the teacher and the vector lean female.

## What this does not cover

- **Only the gate.** Green Earth's ranker and the per-reader language ordering
  come after the vector and are not modelled here.
- **38% of authors are unknown.** If the unknown group's real gender mix differs
  from the labelled groups', the population-level gap could be larger or smaller.
- **Non-binary is too small to read.** 75 authors; its CIs span everything.
- **The teacher is a model.** "Teacher internal" is the decider's output, not
  ground truth, so this does not show whether women's posts *are* more internal.
  It shows only that the pipeline treats them as such.

## Rerunning

After any retrain, with `tmp_venv` (`.venv`'s interpreter is gone):

```
tmp_venv/bin/python research/gender_bias/fetch_profiles.py      # resumable
tmp_venv/bin/python research/gender_bias/infer_gender.py
tmp_venv/bin/python research/gender_bias/analyze_gender_bias.py  # GB_GATE overrides 0.86
```

Profiles and inferred genders stay in `research/data/gender_bias/` (gitignored);
only aggregates belong in reports.
