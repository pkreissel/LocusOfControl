# Diurnal profile of Perspective scores (primary sample)

n = 10000. Hours are **Europe/Berlin local time**. SEs and tests are clustered by author.

> **Single-window caveat.** The sample spans one 24-hour window, so hour of day is perfectly confounded with that particular hour on that particular day. A peak at 20:00 could be a nightly rhythm or one evening's news cycle — this design cannot separate them. Treat the profile as descriptive.

## Posting volume by local hour

| hour | posts |
|---|---|
| 00 | 185 |
| 01 | 73 |
| 02 | 68 |
| 03 | 35 |
| 04 | 67 |
| 05 | 98 |
| 06 | 281 |
| 07 | 473 |
| 08 | 648 |
| 09 | 596 |
| 10 | 634 |
| 11 | 505 |
| 12 | 579 |
| 13 | 600 |
| 14 | 542 |
| 15 | 473 |
| 16 | 542 |
| 17 | 525 |
| 18 | 601 |
| 19 | 604 |
| 20 | 619 |
| 21 | 608 |
| 22 | 397 |
| 23 | 247 |

## Night vs day, adjusted for post composition

Pre-specified blocks: night = 00:00-05:59, day = 09:00-17:59 local. Positive = higher at night. Adjusted for text length, reply status, images, video, external links, and link/mention/tag counts, because night posts are shorter and likelier to be replies, and Perspective scores move with both.

**This table, not the peak/trough one below, is the result to quote.** It is a fixed contrast decided in advance, so it is not inflated by picking the most extreme of 24 noisy hourly means.

| attribute | night-day diff | SE | q | n night | n day | flag |
|---|---|---|---|---|---|---|
| _none_ — no attribute shifts significantly between night and day once composition is held constant_ | | | | | | |

### All attributes, with minimum detectable effect

No significant shift is only informative alongside what the design could have caught. MDE is the smallest night-day difference this sample would detect at 80% power (2.8 x SE), on the attribute's own 0-1 scale.

| attribute | raw diff | adjusted diff | SE | MDE | q | flag |
|---|---|---|---|---|---|---|
| unsubstantial | +0.0767 | +0.0295 | 0.0111 | 0.0310 | 0.16318 | keep |
| spam | +0.0151 | +0.0163 | 0.0123 | 0.0344 | 0.434417 | keep |
| personal_story_experimental | +0.0115 | +0.0149 | 0.0076 | 0.0213 | 0.265689 | keep |
| toxicity | +0.0055 | +0.0118 | 0.0067 | 0.0187 | 0.265689 | keep |
| affinity_experimental | +0.0065 | +0.0105 | 0.0095 | 0.0265 | 0.472098 | keep |
| profanity | +0.0089 | +0.0099 | 0.0059 | 0.0165 | 0.2786 | keep |
| attack_on_commenter | +0.0210 | +0.0095 | 0.0052 | 0.0145 | 0.265689 | low_power |
| insult | +0.0028 | +0.0089 | 0.0056 | 0.0155 | 0.282079 | keep |
| curiosity_experimental | -0.0119 | -0.0085 | 0.0086 | 0.0242 | 0.484943 | keep |
| compassion_experimental | -0.0053 | +0.0072 | 0.0083 | 0.0232 | 0.536439 | keep |
| identity_attack | +0.0036 | +0.0061 | 0.0033 | 0.0093 | 0.265689 | low_power |
| attack_on_author | +0.0093 | +0.0059 | 0.0024 | 0.0069 | 0.170788 | low_power |
| sexually_explicit | +0.0056 | +0.0046 | 0.0044 | 0.0122 | 0.472098 | low_power |
| inflammatory | +0.0117 | +0.0045 | 0.0043 | 0.0120 | 0.472098 | low_power |
| respect_experimental | +0.0082 | +0.0043 | 0.0067 | 0.0188 | 0.602672 | keep |
| nuance_experimental | -0.0065 | +0.0043 | 0.0036 | 0.0101 | 0.472098 | low_power |
| flirtation | -0.0163 | -0.0039 | 0.0050 | 0.0139 | 0.536439 | keep |
| threat | +0.0012 | +0.0015 | 0.0019 | 0.0053 | 0.536439 | low_power |
| severe_toxicity | +0.0008 | +0.0011 | 0.0021 | 0.0060 | 0.681328 | low_power |
| reasoning_experimental | -0.0161 | -0.0009 | 0.0032 | 0.0090 | 0.825301 | keep |
| obscene | +0.0153 | -0.0007 | 0.0079 | 0.0222 | 0.932442 | keep |

## Joint hour test: raw vs composition-adjusted

An attribute significant raw but not adjusted is tracking *what kind of post* people make at that hour, not a shift in tone.

> **Treat these p-values as weak evidence.** This is a 23-parameter Wald test with a cluster-robust covariance matrix, and the sparsest hour holds only a few dozen posts. That combination is known to over-reject. Where this test and the pre-specified night-vs-day contrast disagree, believe the contrast.

| attribute | q (raw) | q (adjusted) | survives |
|---|---|---|---|
| profanity | 1.6e-05 | 4.1e-05 | yes |
| sexually_explicit | 0.0 | 4.1e-05 | yes |
| threat | 0.000188 | 0.000222 | yes |
| affinity_experimental | 0.000188 | 0.000331 | yes |
| severe_toxicity | 0.002047 | 0.000506 | yes |
| toxicity | 0.009118 | 0.001482 | yes |
| identity_attack | 0.025347 | 0.005815 | yes |
| reasoning_experimental | 1e-06 | 0.005815 | yes |
| respect_experimental | 0.009118 | 0.006247 | yes |
| insult | 0.03431 | 0.017258 | yes |
| spam | 0.00127 | 0.017258 | yes |
| personal_story_experimental | 0.048345 | 0.023967 | yes |
| compassion_experimental | 0.029414 | 0.052044 | no |
| unsubstantial | 0.0 | 0.068717 | no |
| obscene | 1e-06 | 0.105283 | no |
| flirtation | 5.3e-05 | 0.182193 | no |
| attack_on_author | 0.0 | 0.463119 | no |
| attack_on_commenter | 3e-06 | 0.837506 | no |
| nuance_experimental | 0.386789 | 0.837506 | no |
| curiosity_experimental | 0.792195 | 0.84911 | no |
| inflammatory | 0.004483 | 0.931201 | no |

## Peak and trough hours (selection-biased -- context only)

The quietest hours carry only a few dozen posts, and peak/trough is the max and min over 24 such estimates, so this swing is an upper bound on any real diurnal variation. Shown for orientation, not for quoting.

| attribute | peak h | peak | n | trough h | trough | n | swing | q | flag |
|---|---|---|---|---|---|---|---|---|---|
| unsubstantial | 04 | 0.5368 | 67 | 09 | 0.3387 | 596 | +0.1981 | 0.0 | keep |
| spam | 03 | 0.4773 | 35 | 01 | 0.3292 | 73 | +0.1481 | 0.00127 | keep |
| compassion_experimental | 03 | 0.3868 | 35 | 02 | 0.3085 | 68 | +0.0782 | 0.029414 | keep |
| flirtation | 13 | 0.4311 | 600 | 04 | 0.3580 | 67 | +0.0732 | 5.3e-05 | keep |
| obscene | 06 | 0.1788 | 281 | 03 | 0.1060 | 35 | +0.0727 | 1e-06 | keep |
| affinity_experimental | 06 | 0.4302 | 281 | 02 | 0.3670 | 68 | +0.0631 | 0.000188 | keep |
| attack_on_commenter | 04 | 0.1104 | 67 | 11 | 0.0585 | 505 | +0.0519 | 3e-06 | low_power |
| personal_story_experimental | 03 | 0.4243 | 35 | 14 | 0.3755 | 542 | +0.0488 | 0.048345 | keep |
| respect_experimental | 06 | 0.5893 | 281 | 02 | 0.5480 | 68 | +0.0413 | 0.009118 | keep |
| sexually_explicit | 03 | 0.0624 | 35 | 05 | 0.0242 | 98 | +0.0383 | 0.0 | low_power |
| reasoning_experimental | 07 | 0.1160 | 473 | 04 | 0.0790 | 67 | +0.0371 | 1e-06 | keep |
| profanity | 01 | 0.0818 | 73 | 09 | 0.0471 | 596 | +0.0347 | 1.6e-05 | keep |
| toxicity | 01 | 0.1120 | 73 | 05 | 0.0804 | 98 | +0.0316 | 0.009118 | keep |
| insult | 01 | 0.0759 | 73 | 03 | 0.0474 | 35 | +0.0284 | 0.03431 | keep |
| inflammatory | 05 | 0.1012 | 98 | 03 | 0.0742 | 35 | +0.0270 | 0.004483 | low_power |
| attack_on_author | 04 | 0.0425 | 67 | 17 | 0.0178 | 525 | +0.0247 | 0.0 | low_power |
| identity_attack | 03 | 0.0380 | 35 | 00 | 0.0162 | 185 | +0.0218 | 0.025347 | low_power |
| severe_toxicity | 03 | 0.0193 | 35 | 07 | 0.0075 | 473 | +0.0117 | 0.002047 | low_power |
| threat | 00 | 0.0181 | 185 | 06 | 0.0103 | 281 | +0.0078 | 0.000188 | low_power |

## Engagement by local hour

| outcome | peak h | trough h | chi2 | p |
|---|---|---|---|---|
| like_count | 7 | 1 | 39.43 | 0.0178 |
| repost_count | 7 | 4 | 163.99 | 2.94e-23 |
| reply_count | 11 | 1 | 50.4 | 0.000817 |
| quote_count | 12 | 3 | 189.44 | 3.9e-28 |
| total_engagement | 7 | 1 | 60.09 | 3.71e-05 |

## Why this matters for the engagement analysis

If both a score and engagement vary by hour, then posting time is a confounder of the score-engagement association: an attribute could appear to predict likes purely because it is more common at hours when people are awake. `analyze_de_engagement.py` conditions on hour (as sin/cos in the control block, or as 23 dummies with `--hour-dummies`), and the within-author model conditions on it again inside each author. Compare those runs to see how much of an association is timing rather than content.
