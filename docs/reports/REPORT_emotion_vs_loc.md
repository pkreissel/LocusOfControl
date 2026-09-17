# What the locus-of-control vector does to the emotional mix of the feed

**Corpus:** the 5,000 posts the deployed internal vector was trained on
(`dataset_annotated_lmstudio_soft.jsonl`, minus the 500 held out for Gemini
validation, minus empty texts — the exact filter in
`train_internal_vector.load_train_rows`, so the cached Jina embeddings in
`embeddings_lmstudio_soft_v3_jina.npz` line up row-for-row).

**LoC score:** reproduced from `bunny_feed/src/vector.json` exactly as
`bunny_feed/src/vector.ts` computes it — L2-normalize, dot the vector,
`sigmoid(6.542 * cos + 1.807)`.

**Emotion score:** `tabularisai/multilingual-emotion-classification`
(XLM-R base, 23 languages, 11 emotions, independent sigmoid heads, τ = 0.5).

**Gate:** the vector's only editorial act in the live feed is the 0.88
probability threshold (`feeds.ts`); ordering among survivors is handed to GE's
reranker. So the comparison holds the `minWords: 6` length gate fixed and varies
only the vector:

- **pool** — 4,367 posts passing `minWords 6` (the feed without the vector)
- **admitted** — 1,858 of those also scoring ≥ 0.88 (the feed with it)
- keep rate **42.5%**

Scripts: `score_emotions_loc.py`, `analyze_emotion_shift.py`,
`analyze_emotion_valence.py`. Machine-readable output in
`REPORT_emotion_vs_loc.json` and `REPORT_emotion_vs_loc_valence.json`.

## Headline

| emotion | pool % | admitted % | lift | rejected % | p | ρ (score vs emotion) |
|---|---:|---:|---:|---:|---:|---:|
| gratitude | 1.05 | 2.10 | **1.99** | 0.28 | 5.7e-09 | +0.332 |
| joy | 2.45 | 4.52 | **1.85** | 0.92 | 2.6e-14 | +0.290 |
| love | 2.89 | 4.95 | **1.72** | 1.36 | 2.2e-12 | +0.292 |
| frustration | 1.03 | 1.29 | 1.25 | 0.84 | 0.14 (n.s.) | −0.098 |
| neutral | 54.20 | 59.15 | 1.09 | 50.54 | 1.6e-08 | +0.088 |
| surprise | 4.90 | 4.04 | 0.82 | 5.54 | 0.023 | −0.111 |
| contempt | 6.73 | 5.33 | 0.79 | 7.77 | 1.4e-03 | −0.096 |
| sadness | 3.50 | 2.37 | 0.68 | 4.34 | 4.5e-04 | −0.082 |
| anger | 9.75 | 6.51 | **0.67** | 12.16 | 5.1e-10 | −0.194 |
| disgust | 3.62 | 1.99 | **0.55** | 4.82 | 7.3e-07 | −0.263 |
| fear | 6.07 | 3.18 | **0.52** | 8.21 | 5.6e-12 | −0.245 |

Aggregated: any negative emotion 28.7% → 19.7% (×0.69), any positive 6.1% →
11.0% (×1.79). The negative:positive ratio goes from **4.7:1 to 1.8:1** — the
gate does not make the feed happy, it makes it roughly two and a half times less
lopsided.

The dose-response is monotone across score deciles, which is what you'd want
from a threshold that could have been placed anywhere:

| decile | anger | fear | disgust | joy | love | gratitude |
|---|---:|---:|---:|---:|---:|---:|
| D1 (lowest LoC) | 11.0 | 12.4 | 5.3 | 0.7 | 0.7 | 0.2 |
| D5 | 11.4 | 4.6 | 4.1 | 1.1 | 1.6 | 0.0 |
| D8 | 8.5 | 2.1 | 0.7 | 3.2 | 1.8 | 2.1 |
| D10 (highest) | 2.1 | 0.9 | 2.8 | 9.6 | 10.8 | 3.9 |

Fear collapses earliest and hardest (12.4% → 0.9%, a 14× swing across the
range); anger only really breaks in the top two deciles.

## It is not just a sentiment filter

The obvious worry is that the LoC axis has quietly become valence. It hasn't,
or not mostly:

- LoC logit vs a crude positive-minus-negative valence score: **ρ = 0.329**
- Ridge on all 11 emotion probabilities, 5-fold CV: **R² = 0.131**

So ~13% of the variance in the LoC score is recoverable from the emotion mix.
The axis is emotionally *tilted*, not emotionally *defined* — and the tilt is
the direction you'd predict from the construct (helplessness co-occurs with
fear, agency with gratitude).

The clearest evidence it's tracking the construct rather than the tone: **within
the teacher's `external` class the gate barely touches anger** (18.0% → 17.8%).
It isn't filtering angry posts; it's filtering external posts, only 22.8% of
which survive. Fear inside that same class *does* drop (11.5% → 7.4%), so the
gate is sharper on frightened helplessness than on angry helplessness.

## The one place it disagrees with its own teacher

Anger is **bimodal on the LoC axis**. In the pool, external posts are the
angriest class (18.0%), internal are second (13.2%), and teacher-neutral posts
are the calm middle (4.5%). Agency talk is roughly three times angrier than
neutral chatter — much indignant "get up and do something" writing carries real
anger — even though helpless-framing posts are angrier still.

That matters because the corpus-level anger drop (9.75% -> 6.51%) is mostly a
composition effect, not a within-class one. Kitagawa decomposition of the
-3.24pp: **-2.04pp** comes from the gate shifting the class mix away from
external, **-1.21pp** from anger falling inside the classes. Both terms are
negative — anger falls at every level of the analysis. Nothing here shows the
vector admitting more anger.

What the vector does not inherit is the internal class's share of that anger.
Within the internal class the gate cuts anger 13.2% → 8.0%, and the mechanism is
visible in the raw scores:

| teacher-internal posts | n in pool | admitted | keep rate | mean LoC prob |
|---|---:|---:|---:|---:|
| angry | 34 | 14 | **41.2%** | 0.864 |
| not angry | 224 | 162 | **72.3%** | 0.894 |

Angry internal posts sit at 0.864 — just under the 0.88 gate — while calm ones
sit at 0.894, just over. The threshold happens to fall right between the two
groups, so it drops well over half of the angry agency posts the teacher
approved of, and keeps nearly three quarters of the calm ones. That is a
tone preference the LoC construct never asked for, and it is a threshold
artifact, not a property of the vector: at a lower gate the two groups would
separate far less.

Sample size is small (34 angry internal posts in the pool), so treat the exact
41% as indicative. But the direction is consistent with the D10 anger figure
(2.1%) and worth a deliberate decision rather than an accident.

## Composition of the admitted set

| | pool | admitted |
|---|---:|---:|
| internal | 258 (5.9%) | 176 (9.5%) |
| neutral | 2,577 (59.0%) | 1,333 (71.7%) |
| external | 1,532 (35.1%) | 349 (18.8%) |

The gate roughly halves external's share and raises internal's by 1.6×, but the
feed is still 72% teacher-neutral — the vector is a pre-filter, and the emotional
shift above is achieved mostly by removing external posts, not by promoting
internal ones.

## Caveats

1. **The emotion model is unvalidated on this data.** It's trained on synthetic
   multilingual text with no human-labeled evaluation, and it's CC-BY-NC-4.0.
   13.8% of posts fire no label at all and 54% fire `neutral`, so the tail
   emotions rest on a few dozen posts each. The relative shifts are large and
   consistent enough to survive a fair amount of measurement noise, but the
   absolute prevalences shouldn't be quoted.
2. **This is the training corpus, not the live candidate pool.** It's firehose
   sampled (66% en, 12% de, 17% no lang tag), which is the right population for
   "what does the vector do to posts", but GE's candidate mix is popularity- and
   network-weighted and will have a different emotional baseline.
3. **The measured keep rate is 42.5% here** against the 47% of corpus texts /
   34% of live candidates recorded in `feeds.ts`; the difference is the
   `minWords` gate being applied first, which is the correct order.
4. The per-language split is consistent (en keeps 43.9%, de 48.8%, both shifting
   neutral up and anger down), so this isn't an artifact of the English majority.

## What I'd do with this

- **Decide whether angry agency belongs in the feed.** If it does, 0.88 is
  cutting in the wrong place for that subgroup. Recalibrating against angry
  internal posts specifically — rather than against overall keep rate — is a
  small change with a clear editorial meaning.
- **Track fear rather than anger as the health metric.** It's the emotion the
  vector moves most cleanly and monotonically (ρ = −0.245, 14× across deciles),
  so it's the most sensitive canary for a retrain shifting the axis.
- **Re-run this after every retrain.** `score_emotions_loc.py` reuses the
  embedding cache, so the whole thing is a couple of minutes on MPS; the
  emotion profile is a cheap, independent read on whether a new vector still
  means what the old one meant.
