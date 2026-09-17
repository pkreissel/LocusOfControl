# Diurnal profile of Perspective scores (root sample)

n = 10000. Hours are **Europe/Berlin local time**. SEs and tests are clustered by author.

> **Single-window caveat.** The sample spans one 24-hour window, so hour of day is perfectly confounded with that particular hour on that particular day. A peak at 20:00 could be a nightly rhythm or one evening's news cycle — this design cannot separate them. Treat the profile as descriptive.

## Posting volume by local hour

| hour | posts |
|---|---|
| 00 | 197 |
| 01 | 93 |
| 02 | 65 |
| 03 | 48 |
| 04 | 66 |
| 05 | 131 |
| 06 | 325 |
| 07 | 560 |
| 08 | 623 |
| 09 | 586 |
| 10 | 626 |
| 11 | 567 |
| 12 | 584 |
| 13 | 577 |
| 14 | 525 |
| 15 | 493 |
| 16 | 566 |
| 17 | 559 |
| 18 | 517 |
| 19 | 537 |
| 20 | 588 |
| 21 | 523 |
| 22 | 381 |
| 23 | 263 |

## Night vs day, adjusted for post composition

Pre-specified blocks: night = 00:00-05:59, day = 09:00-17:59 local. Positive = higher at night. Adjusted for text length, reply status, images, video, external links, and link/mention/tag counts, because night posts are shorter and likelier to be replies, and Perspective scores move with both.

**This table, not the peak/trough one below, is the result to quote.** It is a fixed contrast decided in advance, so it is not inflated by picking the most extreme of 24 noisy hourly means.

| attribute | night-day diff | SE | q | n night | n day | flag |
|---|---|---|---|---|---|---|
| unsubstantial | +0.0349 | 0.0107 | 0.012164 | 600 | 5083 | keep |
| affinity_experimental | +0.0312 | 0.0089 | 0.009676 | 600 | 5083 | keep |
| respect_experimental | +0.0209 | 0.0067 | 0.013563 | 600 | 5083 | keep |
| attack_on_author | +0.0068 | 0.0026 | 0.047114 | 600 | 5083 | low_power |

### All attributes, with minimum detectable effect

No significant shift is only informative alongside what the design could have caught. MDE is the smallest night-day difference this sample would detect at 80% power (2.8 x SE), on the attribute's own 0-1 scale.

| attribute | raw diff | adjusted diff | SE | MDE | q | flag |
|---|---|---|---|---|---|---|
| unsubstantial | +0.0952 | +0.0349 | 0.0107 | 0.0300 | 0.012164 | keep |
| affinity_experimental | +0.0170 | +0.0312 | 0.0089 | 0.0249 | 0.009676 | keep |
| respect_experimental | +0.0204 | +0.0209 | 0.0067 | 0.0189 | 0.013563 | keep |
| personal_story_experimental | +0.0053 | +0.0177 | 0.0073 | 0.0205 | 0.065819 | keep |
| compassion_experimental | -0.0049 | +0.0170 | 0.0080 | 0.0223 | 0.099241 | keep |
| profanity | +0.0138 | +0.0137 | 0.0073 | 0.0204 | 0.124997 | keep |
| attack_on_commenter | +0.0254 | +0.0113 | 0.0048 | 0.0135 | 0.066098 | low_power |
| sexually_explicit | +0.0132 | +0.0111 | 0.0056 | 0.0157 | 0.114086 | low_power |
| toxicity | +0.0040 | +0.0093 | 0.0079 | 0.0221 | 0.460401 | keep |
| curiosity_experimental | -0.0171 | -0.0082 | 0.0082 | 0.0231 | 0.511187 | keep |
| severe_toxicity | +0.0080 | +0.0073 | 0.0035 | 0.0098 | 0.099241 | low_power |
| attack_on_author | +0.0110 | +0.0068 | 0.0026 | 0.0073 | 0.047114 | low_power |
| obscene | +0.0254 | +0.0060 | 0.0077 | 0.0217 | 0.613274 | keep |
| identity_attack | +0.0024 | +0.0046 | 0.0041 | 0.0115 | 0.466626 | low_power |
| inflammatory | +0.0147 | +0.0044 | 0.0046 | 0.0129 | 0.511187 | low_power |
| flirtation | -0.0138 | +0.0023 | 0.0050 | 0.0140 | 0.790152 | keep |
| reasoning_experimental | -0.0243 | -0.0021 | 0.0032 | 0.0090 | 0.67188 | keep |
| insult | -0.0046 | +0.0012 | 0.0059 | 0.0164 | 0.934167 | keep |
| threat | +0.0004 | +0.0003 | 0.0014 | 0.0040 | 0.934167 | low_power |
| spam | -0.0091 | -0.0003 | 0.0121 | 0.0339 | 0.985205 | keep |
| nuance_experimental | -0.0160 | -0.0001 | 0.0035 | 0.0098 | 0.985205 | low_power |

## Joint hour test: raw vs composition-adjusted

An attribute significant raw but not adjusted is tracking *what kind of post* people make at that hour, not a shift in tone.

> **Treat these p-values as weak evidence.** This is a 23-parameter Wald test with a cluster-robust covariance matrix, and the sparsest hour holds only a few dozen posts. That combination is known to over-reject. Where this test and the pre-specified night-vs-day contrast disagree, believe the contrast.

| attribute | q (raw) | q (adjusted) | survives |
|---|---|---|---|
| severe_toxicity | 6e-06 | 1.4e-05 | yes |
| personal_story_experimental | 0.002033 | 1.4e-05 | yes |
| toxicity | 0.000452 | 0.000278 | yes |
| affinity_experimental | 0.00207 | 0.000278 | yes |
| respect_experimental | 5.6e-05 | 0.000278 | yes |
| insult | 0.000414 | 0.001056 | yes |
| identity_attack | 0.003999 | 0.001855 | yes |
| unsubstantial | 0.0 | 0.005181 | yes |
| reasoning_experimental | 0.0 | 0.005252 | yes |
| compassion_experimental | 0.020081 | 0.019891 | yes |
| nuance_experimental | 2e-06 | 0.031427 | yes |
| flirtation | 0.00039 | 0.107103 | no |
| obscene | 0.0 | 0.149874 | no |
| attack_on_author | 0.0 | 0.157421 | no |
| curiosity_experimental | 0.039093 | 0.200245 | no |
| attack_on_commenter | 2e-06 | 0.295457 | no |
| inflammatory | 0.000136 | 0.495439 | no |
| spam | 0.906085 | 0.91374 | no |
| profanity | 0.0 | 0.0 | yes |
| threat | 0.0 | 0.0 | yes |
| sexually_explicit | 0.0 | 0.0 | yes |

## Peak and trough hours (selection-biased -- context only)

The quietest hours carry only a few dozen posts, and peak/trough is the max and min over 24 such estimates, so this swing is an upper bound on any real diurnal variation. Shown for orientation, not for quoting.

| attribute | peak h | peak | n | trough h | trough | n | swing | q | flag |
|---|---|---|---|---|---|---|---|---|---|
| unsubstantial | 02 | 0.5174 | 65 | 08 | 0.2640 | 623 | +0.2534 | 0.0 | keep |
| obscene | 02 | 0.2095 | 65 | 08 | 0.0935 | 623 | +0.1160 | 0.0 | keep |
| affinity_experimental | 05 | 0.4406 | 131 | 03 | 0.3396 | 48 | +0.1010 | 0.00207 | keep |
| compassion_experimental | 05 | 0.3858 | 131 | 03 | 0.2923 | 48 | +0.0935 | 0.020081 | keep |
| flirtation | 03 | 0.4551 | 48 | 04 | 0.3879 | 66 | +0.0671 | 0.00039 | keep |
| attack_on_commenter | 02 | 0.1064 | 65 | 08 | 0.0427 | 623 | +0.0637 | 2e-06 | low_power |
| profanity | 02 | 0.1050 | 65 | 17 | 0.0438 | 559 | +0.0612 | 0.0 | keep |
| inflammatory | 02 | 0.1238 | 65 | 08 | 0.0651 | 623 | +0.0587 | 0.000136 | low_power |
| respect_experimental | 05 | 0.5878 | 131 | 22 | 0.5298 | 381 | +0.0580 | 5.6e-05 | keep |
| sexually_explicit | 02 | 0.0859 | 65 | 17 | 0.0293 | 559 | +0.0566 | 0.0 | low_power |
| reasoning_experimental | 09 | 0.1400 | 586 | 03 | 0.0845 | 48 | +0.0556 | 0.0 | keep |
| personal_story_experimental | 05 | 0.4389 | 131 | 03 | 0.3857 | 48 | +0.0532 | 0.002033 | keep |
| toxicity | 02 | 0.1236 | 65 | 01 | 0.0727 | 93 | +0.0509 | 0.000452 | keep |
| curiosity_experimental | 17 | 0.3358 | 559 | 23 | 0.2934 | 263 | +0.0423 | 0.039093 | keep |
| insult | 23 | 0.0849 | 263 | 01 | 0.0446 | 93 | +0.0403 | 0.000414 | keep |
| nuance_experimental | 09 | 0.1157 | 586 | 03 | 0.0769 | 48 | +0.0388 | 2e-06 | low_power |
| attack_on_author | 02 | 0.0466 | 65 | 08 | 0.0145 | 623 | +0.0321 | 0.0 | low_power |
| identity_attack | 05 | 0.0467 | 131 | 03 | 0.0168 | 48 | +0.0299 | 0.003999 | low_power |
| severe_toxicity | 03 | 0.0295 | 48 | 17 | 0.0068 | 559 | +0.0226 | 6e-06 | low_power |
| threat | 22 | 0.0234 | 381 | 03 | 0.0106 | 48 | +0.0128 | 0.0 | low_power |

## Engagement by local hour

| outcome | peak h | trough h | chi2 | p |
|---|---|---|---|---|
| like_count | 8 | 1 | 90.37 | 6.16e-10 |
| repost_count | 8 | 1 | 73.96 | 2.93e-07 |
| reply_count | 6 | 1 | 53.62 | 0.000301 |
| quote_count | 8 | 3 | 44.4 | 0.00471 |
| total_engagement | 8 | 1 | 86.57 | 2.66e-09 |

## Why this matters for the engagement analysis

If both a score and engagement vary by hour, then posting time is a confounder of the score-engagement association: an attribute could appear to predict likes purely because it is more common at hours when people are awake. `analyze_de_engagement.py` conditions on hour (as sin/cos in the control block, or as 23 dummies with `--hour-dummies`), and the within-author model conditions on it again inside each author. Compare those runs to see how much of an association is timing rather than content.
