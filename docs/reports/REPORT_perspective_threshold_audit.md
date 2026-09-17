# Perspective × LoC Vector Audit — Threshold Recalibration

**Date:** 2026-09-06
**Outcome:** `VECTOR_PROB_THRESHOLD` raised 0.55 → 0.88 and deployed (`d8e7c51`).
**Trigger:** rating every local dataset with the Perspective API surfaced that the
production vector filter was rejecting nothing at all.

---

## 1. Headline findings

1. **The 0.55 gate was a no-op.** It rejected 0 of 17,581 unique dataset texts and
   0 of 993 live two-tower candidates. The feed's topical selectivity was coming
   entirely from the reranker's `RANK_SIM_WEIGHT` blend, not from the vector filter.
2. **The vector itself is sound.** AUC 0.764 internal-vs-rest and 0.829
   internal-vs-external on the held-out validation set, matching the 0.773 reported
   at training time in `4a9286e`. Only the operating point was wrong.
3. **Perspective's `languages` parameter is effectively a no-op** — `de` and `en`
   scoring agree at r = 1.000 — and the `*_EXPERIMENTAL` core attributes are
   byte-identical aliases of the production ones.
4. **Raising the threshold trades toxicity down for spam up.** SPAM is the strongest
   single correlate of the LoC score (Spearman +0.219); mean toxicity falls to 0.80×
   at a 0.85 cut while spam rises to 1.80× at 0.95.
5. **The vector's best alignment is with bridging, not civility.** The composite of
   Perspective's 7 bridging attributes correlates +0.369 with the LoC score —
   stronger than its relationship with either toxicity or spam.

---

## 2. Why 0.55 rejected nothing

The vector retrained in `4a9286e` (322 live-feed interactions) shifted the score
distribution upward; the threshold constant was never moved with it.

Observed cosine range across 17,581 texts: **−0.2172 … +0.2517**, median 0.0232.
Scoring is `p = sigmoid(scale·cos + bias)` with `scale = 6.542`, `bias = 1.807`:

```
p ≥ 0.55  ⟺  6.542·cos + 1.807 ≥ 0.2007  ⟺  cos ≥ −0.2456
```

The corpus minimum (−0.2172) never reaches the rejection region. Lowest LoC
probability observed anywhere: **0.595**. The stale comment at `index.ts:21`
("0.55 passes ~50%, 0.60 passes ~25%") described an earlier vector.

This is not an artifact of the annotation corpora — re-scoring 993 live two-tower
candidates from Green Earth gives p1 = 0.698, median 0.860, and a 100% pass rate
at 0.55.

---

## 3. Methodology

### 3.1 Corpus

Every `text` field across the 23 `dataset_*.jsonl` and `pilot_*.jsonl` files,
deduplicated by SHA-1: **47,908 rows → 17,581 unique texts** (raw and annotated
corpora overlap heavily). Language tags: 10,488 `en`, 2,394 `de`, 4,006 untagged,
~700 other.

### 3.2 LoC scoring — production path, reproduced exactly

- Model `jina-embeddings-v5-text-small`, `task=classification`, input truncated to
  2,000 chars — matching `bunny_feed/src/jina.ts`.
- Scored with `bunny_feed/src/vector.json` (`scale` 6.54238748550415, `bias`
  1.8073093891143799), L2-normalising the embedding, then
  `sigmoid(scale·cos + bias)` — matching `scoreEmbedding()` in `vector.ts`.
- **Verified** the API path reproduces the local model by recomputing cosines from
  the committed `embeddings_*_jina.npz` files: same scale, same distribution, same
  100% pass rate at 0.55. The finding is not an artifact of re-embedding.
- `vector.json` confirmed byte-equal to `internal_vector.npy`; file committed clean,
  no working-tree drift.

Script: `scratchpad/loc_score.py` (756k Jina tokens, ~2 min, resumable).

### 3.3 Perspective scoring

Three passes over the same 17,581 texts, one request per text, all attributes
batched into a single request, concurrency 10 (~40-50 req/s, well above the
documented 1 QPS default), zero errors in all three runs:

| Pass | `languages` | Attributes | Runtime |
| :--- | :--- | :--- | :--- |
| German | `["de"]` | 6 core | 6.9 min |
| English | `["en"]` | 22 (6 core + 6 `*_EXPERIMENTAL` + 10 en-only) | 7.1 min |
| Bridging | `["en"]` | 7 bridging | 6.1 min |

Attribute availability was probed empirically rather than read from the docs (the
docs page is a JS SPA that renders nothing to a fetcher):

- **German supports only** TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT,
  PROFANITY, THREAT (plus their `*_EXPERIMENTAL` twins).
- **English adds** SEXUALLY_EXPLICIT, FLIRTATION, OBSCENE, SPAM, INCOHERENT,
  INFLAMMATORY, LIKELY_TO_REJECT, UNSUBSTANTIAL, ATTACK_ON_AUTHOR,
  ATTACK_ON_COMMENTER.
- **Bridging = exactly 7**, all `*_EXPERIMENTAL`-suffixed, English-only: AFFINITY,
  COMPASSION, CURIOSITY, NUANCE, PERSONAL_STORY, REASONING, RESPECT. Unsuffixed
  names and the negative-side attributes (ALIENATION, FEARMONGERING, MORAL_OUTRAGE,
  SCAPEGOATING, GENERALIZATION) return "Unknown requested attribute".

Scripts: `scratchpad/perspective_rate.py`, `_en.py`, `_bridge.py` — all resumable
(keyed by text hash, a rerun skips completed ids).

### 3.4 Validation-set evaluation

`embeddings_gemini_jina.npz` — 500 held-out labelled posts (81 internal, 176
neutral, 243 external), embedded with the local model at training time, scored with
the live vector.

---

## 4. Results

### 4.1 Toxicity vs LoC cutoff (German scoring; English is identical)

| Cutoff | Kept | Keep % | Toxicity | Insult | Profanity | Identity | Threat | Tox ≥ 0.5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.55 *(old prod)* | 17,581 | 100.00% | 0.1032 | 0.0599 | 0.0560 | 0.0258 | 0.0201 | 3.81% |
| 0.80 | 15,707 | 89.34% | 0.0981 | 0.0566 | 0.0540 | 0.0242 | 0.0186 | 3.60% |
| 0.85 | 12,189 | 69.33% | 0.0823 | 0.0446 | 0.0456 | 0.0198 | 0.0161 | 2.49% |
| **0.88 *(new prod)*** | **8,280** | **47.10%** | **0.0687** | **0.0342** | **0.0391** | **0.0157** | **0.0142** | **1.44%** |
| 0.90 | 5,131 | 29.18% | 0.0607 | 0.0285 | 0.0364 | 0.0129 | 0.0132 | 0.99% |
| 0.92 | 2,499 | 14.21% | 0.0563 | 0.0248 | 0.0370 | 0.0118 | 0.0125 | 0.92% |
| 0.94 | 709 | 4.03% | 0.0445 | 0.0172 | 0.0293 | 0.0085 | 0.0127 | 0.00% |
| 0.95 | 214 | 1.22% | 0.0415 | 0.0159 | 0.0271 | 0.0082 | 0.0138 | 0.00% |

Baseline correlations with the LoC score (Spearman): TOXICITY −0.266, INSULT
−0.271, IDENTITY_ATTACK −0.250, SEVERE_TOXICITY −0.215, THREAT −0.158,
PROFANITY −0.147. All negative but weak; AUC for "low LoC predicts TOXICITY ≥ 0.5"
is 0.68. The dropped set is consistently 1.4–1.6× more toxic than the kept set from
0.80 upward.

**Decile note:** the relationship is monotonic across deciles 2–10, but decile 2
(LoC 0.797–0.829) is *more* toxic than decile 1 (0.595–0.797) — mean toxicity
0.163 vs 0.144, on five of six attributes. The very bottom of the score range is
not the worst content.

### 4.2 Validation-set separation — how the cutoff was chosen

| Cutoff | Internal kept | Neutral kept | External kept |
| :--- | ---: | ---: | ---: |
| 0.55 | 100.0% | 100.0% | 100.0% |
| 0.75 | 100.0% | 96.6% | 89.7% |
| 0.85 | 91.4% | 76.1% | 49.0% |
| 0.87 | 85.2% | 56.2% | 32.1% |
| **0.88** | **76.5%** | **49.4%** | **25.9%** |
| 0.89 | 65.4% | 38.6% | 16.9% |
| 0.90 | 54.3% | 27.8% | 11.5% |
| 0.92 | 29.6% | 12.5% | 3.7% |
| 0.95 | 1.2% | 0.6% | 0.0% |

**0.95 is not a strict filter, it is an off switch** — it discards 99% of internal
posts too.

### 4.3 Live candidate yield — the binding constraint

Measured on 993 live two-tower candidates; the feed generates 500 per request and
`MAX_RERANK_CANDIDATES = 100`.

| Cutoff | Live pass | ≈ survivors of 500 | Corpus pass |
| :--- | ---: | ---: | ---: |
| 0.85 | 57.8% | 289 | 69.3% |
| 0.87 | 42.2% | 210 | 55.4% |
| **0.88** | **34.4%** | **172** | **47.1%** |
| 0.89 | 27.3% | 136 | 38.0% |
| 0.90 | 20.8% | 104 | 29.2% |

**0.88 was chosen over 0.90** because 0.90 leaves ~104 survivors — right at the
rerank cap, with no headroom for a dip in candidate quality. 0.88 also restores the
"~50% pass" intent the old constant documented. Taking 0.90's sharper separation
requires raising `num_candidates` above 500 first.

### 4.4 German vs English, production vs experimental

| Comparison | Result |
| :--- | :--- |
| `de` vs `en`, 6 core attributes | r = 1.000; mean \|diff\| ≈ 0.0002; max \|diff\| ≤ 0.045; ~70% bit-identical |
| `TOXICITY` vs `TOXICITY_EXPERIMENTAL` (and all 5 others) | max \|diff\| = 0.000000; 100% identical |

The `languages` parameter is a hint to one multilingual model, not a model
selector. Practical consequence: **score German text as `en`** — the scores are the
same and you unlock the 10 English-only attributes.

This retires the language-confound caveat from the first pass. Restricted to the
2,397 German-tagged texts the pattern is unchanged (100% pass at 0.55; mean
toxicity 0.094 → 0.073 at a 0.85 cut).

### 4.5 The spam inversion

English-only attributes, lift vs baseline:

| Attribute | Baseline | @0.88 | @0.90 | @0.95 | Spearman vs LoC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| INFLAMMATORY | 0.2153 | 0.1843 | 0.79× | 0.55× | −0.136 |
| LIKELY_TO_REJECT | 0.6553 | 0.6269 | 0.96× | 0.95× | −0.115 |
| SEXUALLY_EXPLICIT | 0.0297 | 0.0263 | 0.89× | 0.68× | −0.048 |
| **SPAM** | **0.3699** | **0.4344** | **1.30×** | **1.80×** | **+0.219** |
| FLIRTATION | 0.3844 | 0.3856 | 1.03× | 1.14× | +0.001 |
| INCOHERENT | 0.6865 | 0.6959 | 1.04× | 1.09× | +0.011 |

At 0.88 the spam lift is already 1.17×. At 0.92, 54% of survivors score SPAM ≥ 0.5; at 0.95, 71.5%. Examples surviving a
0.92 cut: `spam=1.000 loc=0.947` ("Diamond Earrings in Leicester | …"),
`spam=1.000 loc=0.928` ("instant loan without kyc… RapidRupee"), and an escort ad
at `loc=0.951`.

**Why:** promotional copy is first-person, agentive, and goal-directed — exactly
what a locus-of-control vector rewards. This is a second argument against pushing
the threshold past 0.88.

### 4.6 Bridging attributes — the strongest alignment found

| Attribute | Baseline | @0.88 | Lift @0.88 | Lift @0.95 | Spearman vs LoC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| RESPECT | 0.5155 | 0.6011 | 1.17× | 1.46× | **+0.492** |
| AFFINITY | 0.3928 | 0.4668 | 1.19× | 1.65× | +0.343 |
| PERSONAL_STORY | 0.4224 | 0.4852 | 1.15× | 1.51× | +0.308 |
| COMPASSION | 0.3357 | 0.3619 | 1.08× | 1.50× | +0.151 |
| REASONING | 0.2329 | 0.2554 | 1.10× | 1.54× | +0.120 |
| CURIOSITY | 0.4210 | 0.4432 | 1.05× | 0.99× | +0.101 |
| NUANCE | 0.2047 | 0.2172 | 1.06× | 1.31× | +0.090 |

Composite (mean of all 7): baseline 0.3607, **Spearman +0.369** with the LoC score —
stronger than the vector's relationship with toxicity (−0.266) or spam (+0.219).
Of the 8,280 texts the 0.88 gate keeps, **59.6% are above the corpus mean bridging
score, vs 46.1% of all texts.**

Two qualifications:

- **Bridging is not inverted toxicity.** RESPECT correlates −0.358 with TOXICITY,
  but AFFINITY (+0.099) and COMPASSION (+0.095) correlate *positively* with it, and
  CURIOSITY is orthogonal (−0.009).
- **CURIOSITY is the one attribute the vector cannot lift** (1.05× at 0.88, 0.99× at
  0.95). A feed selecting on locus of control gets more respectful and more
  personal, but not more inquisitive.
- Part of the lift is the spam artifact: REASONING (+0.260) and RESPECT (+0.274)
  correlate with SPAM about as strongly as with anything else — marketing copy is
  polite and structured.

---

## 5. Deployment

- Branch `vector-threshold-recalibration`, commit `d8e7c51`, fast-forwarded to
  `main` and pushed 2026-09-06.
- GitHub Actions `deploy.yml` run `34047073324` — success.
- Verified against the **runtime domain** `audacitytohope-bji54.bunny.run` (not the
  PullZone, which CDN-caches and strips `X-` headers): HTTP 200, 30 posts, cursor
  present, `x-score-cache: h=161 m=339 jina=8`, `x-wait-until: y`,
  `x-page-path: fresh`; page 2 served from the frozen session in 0.1s.
- **Not directly observable from outside:** the survivor count. Both 0.55 and 0.88
  leave ≥100 survivors and the reranker caps at 100, so a full page looks identical
  either way. The green CI run is the evidence the constant is live.

---

## 6. Limitations

- **Perspective is a proxy.** It measures perceived rudeness, not the harms a feed
  actually cares about. Its German models are weaker than its English ones — though
  §4.4 shows that distinction may be moot.
- **The corpus is not the feed.** These datasets were sampled for locus-of-control
  annotation, not to represent the live candidate pool. The live-candidate
  cross-check covers the threshold finding (§2) but not the toxicity distribution.
- **Non-adversarial.** A cutoff chosen on this data will not hold against text
  written to game it.
- **Label-based, for the NSFW half.** Image content nobody has labelled is invisible
  to this method.
- **Single vector, single point in time.** Every number here is tied to
  `vector.json` as of `4a9286e`; a retrain invalidates the cutoff tables.

---

## 7. Open items

1. **Watch for a spam increase in the live feed.** This is the first deploy where
   the filter rejects anything, and §4.5 predicts the survivors skew promotional.
   Perspective's SPAM attribute, or the account-level `spam` labels already
   arriving through hydration, would work as a second gate.
2. **Make the threshold track the vector automatically.** `export_vector_json.py`
   should emit a calibrated percentile alongside scale/bias so a retrain moves the
   cutoff instead of silently disabling it. 0.88 is only correct until the next
   retrain.
3. **`internal_vector.npy` and `internal_vector_bonsai_v2.npy` are byte-identical**
   (same mtime, same meta) — the retrain wrote both. The `bonsai_v2` copy is not an
   older baseline to compare against; the genuinely distinct predecessors are
   `previous_internal_vector.npy` (cos +0.61 with live), `internal_vector_band.npy`
   (+0.65), `internal_vector_v4.npy` (+0.51), and `external_vector.npy` (−0.55,
   correctly anti-aligned).
4. **Separately: the NSFW gap in candidate generation.** `two_tower_empty_history`
   returns 29.5% adult-labelled content and `popularity` — used as the feed's
   `infill` on every request, and as the primary generator for logged-out users —
   returns 15.2%. Raising this threshold does not close that; nothing in the
   pipeline reads Bluesky labels. (`two_tower` itself, with a real user DID, was
   clean: 0 of 968.)

---

## 8. Artefacts

Analysis outputs live in the session scratchpad (not committed):

| File | Contents |
| :--- | :--- |
| `perspective_scores.jsonl` | 6 core attributes, `de`, 17,581 rows |
| `perspective_en.jsonl` | 22 attributes, `en`, 17,581 rows |
| `perspective_bridge.jsonl` | 7 bridging attributes, `en`, 17,581 rows |
| `loc_scores.jsonl` | cosine + probability per text, production vector |
| `text_index.json` | hash → text and hash → dataset membership |
| `analysis.txt`, `analysis_en.txt`, `analysis_bridge.txt` | full tables |
| `perspective_rate*.py`, `loc_score.py`, `analyze*.py` | resumable scoring and analysis scripts |

Committed change: `bunny_feed/src/index.ts` — `VECTOR_PROB_THRESHOLD` 0.55 → 0.88,
with the calibration recorded in the comment above the constant.
