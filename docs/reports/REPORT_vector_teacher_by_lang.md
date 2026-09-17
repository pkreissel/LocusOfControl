# German vs English: the vector, and the teacher it was distilled from

**Gold standard:** the 500 held-out Gemini-labelled posts (`dataset_validation_gemini.jsonl`)
— 81 internal, 243 external, 176 neutral.
**Teacher:** the LM Studio 27B labels for those same 500 posts, which exist because
`load_train_rows` excludes them from training but leaves them in the annotated file.
**Vector:** `bunny_feed/src/vector.json`, scored exactly as `vector.ts` scores it.
The committed val embedding cache holds 176 rows (built for another axis), so the
500 texts were re-embedded through the production path into
`embeddings_val_loc_jina.npz`.

Script: `eval_vector_teacher_by_lang.py`. Machine-readable output:
`REPORT_vector_teacher_by_lang.json`. All CIs are 4,000-sample percentile
bootstraps over posts.

## The short answer: German cannot be answered from this data

| language | n | internal | vector AUC | teacher AUC |
| :--- | ---: | ---: | :--- | :--- |
| English (declared) | 278 | 51 | 0.763 [0.696, 0.826] | 0.612 [0.499, 0.718] |
| German (declared) | 72 | **9** | 0.672 [0.422, 0.878] | 0.690 [0.415, 0.921] |
| all | 500 | 81 | 0.763 [0.706, 0.815] | 0.692 [0.613, 0.772] |

Nine German internal posts. The German confidence intervals are ±0.23 wide —
they contain "no better than a coin" and "very good" simultaneously. Every
direct German-vs-English contrast is null:

| contrast | de − en | 95% CI | p |
| :--- | ---: | :--- | ---: |
| vector AUC | −0.091 | [−0.342, +0.130] | 0.49 |
| teacher AUC | +0.078 | [−0.217, +0.354] | 0.57 |
| teacher label kappa | +0.097 | [−0.107, +0.282] | 0.32 |

Adding the untagged quarter of the set via langdetect (it resolves to 108
English, 11 German, 8 French — so "untagged" is mostly English) moves German to
83 posts and 11 internal, and changes nothing: all three contrasts stay null.

## What the data does support

**On English, the vector beats the teacher's label probability** — 0.763 vs
0.612, a gap of +0.151 with p = 0.01. Distillation did not merely preserve the
teacher on English; the student ranks internal posts better than the signal it
was trained from.

**But "the teacher" has two readouts, and they disagree.** The bare label
probability P(internal) is the weaker one. The *band target* — argmax class plus
graded intensity, the score actually handed to Ridge as the training signal —
is much stronger:

| language | vector | teacher P(internal) | teacher band target |
| :--- | ---: | ---: | ---: |
| English | 0.763 | 0.612 | 0.716 |
| German | 0.672 | 0.690 | **0.852** |
| all | 0.763 | 0.692 | 0.774 |

Against the band target the vector does not beat its teacher anywhere, and on
German it trails by 0.18 — the largest gap in the table. On the
internal-vs-external contrast the same pattern is sharper still: German band
target 0.926 against the vector's 0.753 (n = 36).

That is a hypothesis, not a finding, at these sample sizes. But it is the one
worth testing: it says the German signal is present in the teacher's output and
is being lost in the distillation, rather than being absent from the teacher.

**The teacher's hard labels are nominally better on German** — accuracy 0.778 /
kappa 0.605, against 0.687 / 0.507 on English. The kappa CIs overlap heavily
(de [0.421, 0.760], en [0.426, 0.589]) so this is not established, but it points
the same way: the 27B teacher is not the weak link on German.

**The vector does not reproduce its teacher's ranking, in any language.**

| language | Spearman, vector vs teacher | 95% CI |
| :--- | ---: | :--- |
| English | 0.182 | [0.077, 0.283] |
| German | 0.050 | [−0.185, 0.279] |
| all | 0.170 | [0.082, 0.251] |

This is the best-powered fact in the whole analysis and it reframes the rest.
The vector is not a compressed copy of the teacher — at rho 0.17 it is close to
an independent estimator that happens to agree with Gemini more often on
English. "The student beat the teacher" is accurate but understates it: they are
barely ranking the same axis.

## The practically relevant answer: the gate is language-neutral

For the live feed, what matters is not AUC but what the 0.88 threshold keeps:

| language | keeps overall | keeps gold-internal | keeps gold-external |
| :--- | ---: | ---: | ---: |
| English | 47.5% | 76.5% | 32.4% |
| German | 45.8% | 77.8% | 29.6% |
| all | 41.8% | 76.5% | 25.1% |

These rates rest on 9 German internal and 27 German external posts, so they are
soft — but German is if anything filtered slightly *more* cleanly than English
(a wider internal-to-external spread: 48pp vs 44pp). Nothing here suggests
German readers are getting a materially different filter.

## What would settle it

The blocker is 9 labelled German internal posts, not the models. German material
is not scarce anywhere else in this project: `de_pool_raw.jsonl` holds 169k
posts, `de_study_sample.jsonl` 11k, and the teacher corpus already carries 665
German posts with teacher labels (36 of them internal) that have no gold label.

Labelling ~400 German posts with the Gemini validation prompt — sampled to
enrich for internal, since a random German draw is 5% internal — would yield
roughly 40–60 German internal posts and shrink the German CI from ±0.23 to
around ±0.10. That is enough to tell a real German deficit from this noise, and
enough to test the band-target hypothesis above.

## Caveats

- The absolute numbers differ slightly from the README's training-time figures
  (0.787 vector / 0.751 teacher). The teacher figure depends entirely on which
  readout is called "the teacher" (0.692 / 0.713 / 0.774 for P(internal) /
  legacy P x level / band target), and the vector was re-embedded here rather
  than read from the training cache. Both models are measured identically on the
  same posts, so the comparison is internally consistent even where the
  absolutes are not comparable to the README.
- Language tags are self-reported by the posting client. langdetect agrees with
  the declared tag on 87% of the de/en posts that carry one, so the
  "+detected" rows are noisier than the declared ones, and are reported
  separately rather than merged.
- Gemini is the gold standard here, not ground truth. A German deficit in the
  gold labels themselves would be invisible to this analysis.

---

# Addendum: the full teacher set (5,000 posts, 595 German)

The gold set was the right place to start, because "how good is it" needs a
standard outside both models — the teacher cannot grade itself. But the teacher
set answers a different question that *is* answerable at German sample sizes,
and it should have been run alongside:

    gold set (500)      is the vector RIGHT?     accuracy vs Gemini
    teacher set (5,000) is the vector FAITHFUL?  agreement with its own teacher

It carries **31 German teacher-internal posts against the gold set's 9**, and
595 German posts overall. Script: `eval_teacher_fidelity_by_lang.py`.

Scoring training rows with the deployed vector is in-sample, so fidelity is
measured out-of-fold — 5-fold CV refitting the same Ridge recipe, each row
scored only by a vector that never saw it — with the deployed vector's
in-sample number printed beside it.

| language | n | teacher-internal | out-of-fold AUC | rho vs target | deployed, in-sample |
| :--- | ---: | ---: | :--- | :--- | ---: |
| English | 3,352 | 190 | 0.633 [0.595, 0.669] | 0.382 [0.354, 0.409] | 0.646 |
| German | 595 | 31 | 0.649 [0.553, 0.743] | 0.364 [0.295, 0.430] | 0.646 |
| untagged | 897 | 44 | 0.757 [0.685, 0.827] | 0.392 [0.336, 0.445] | 0.781 |
| all | 5,000 | 268 | 0.657 [0.624, 0.688] | 0.384 [0.360, 0.407] | 0.669 |

**German fidelity is indistinguishable from English fidelity.** AUC de − en
= +0.016, CI [−0.091, +0.118], p = 0.79. Spearman de − en = −0.019, CI
[−0.092, +0.052], p = 0.63. The deployed vector's own in-sample numbers agree
exactly: 0.646 on both languages.

## This retires the band-target hypothesis

The previous section speculated that the German signal is present in the teacher
and lost in the distillation, on the strength of a German band-target AUC of
0.852 against the vector's 0.672. If that were true, German fidelity would be
visibly worse than English fidelity. It is not — measured on 3.4x the positives,
the two are the same to within 0.02. **That 0.18 gap was noise from nine posts.**
The hypothesis is withdrawn, not merely unconfirmed.

## What the bigger sample does establish

**Fidelity is mediocre in every language.** Out-of-fold, the vector reproduces
its own teacher's internal label at AUC 0.657 and its graded target at rho 0.384.
Nothing about that is German-specific; the distillation is lossy across the
board. Taken with the gold-set result — the vector beating the teacher's label
probability on English — the picture is consistent: the vector is not a lossy
copy of the teacher, it is a smoother and partly independent estimator.

**In-sample contamination was small.** In-sample 0.669 vs out-of-fold 0.657, a
gap of 0.012, for a Ridge at alpha 10 on 1024 dimensions. Worth knowing for
future work: scoring training rows with the deployed vector is not the
methodological trap it looks like here.

## Limitation on the CV proxy

The refit does not exactly reproduce the deployed vector: best cosine **0.755**,
from band target + alpha 10 + interaction rows. That was the best of 90
configurations swept over target mode, temperature, L2 and data mix — and it
beats every other committed vector against the deployed one
(`internal_vector_band` 0.646, `previous_internal_vector` 0.615,
`internal_vector_v4` 0.510) — but the real training run evidently used a data
mix or a selection step this does not capture. So the out-of-fold column
describes a close relative of the deployed vector rather than the vector itself.
That is why the deployed vector's in-sample column is reported next to it; the
two routes give the same de-vs-en answer, which is the claim being made.

## Still out of reach

Whether the vector is *right* on German. Fidelity to a teacher says nothing
about correctness, and the German gold count is still 9. The labelling pass
proposed above remains the only way to close that.
