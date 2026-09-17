# Primary vs root-only: do the associations reproduce?

Within-author estimates from the two samples, for every attribute x outcome significant at FDR < 0.05 in **either**. Both are the `early` snapshot (counts not yet settled).

> Significance counts are not comparable across the two samples -- different n, different FDR burden. A result can hold its effect size exactly and still cross the threshold in one and not the other. Compare the estimates.

| attribute | outcome | r (primary) | q | r (root) | q | delta | verdict |
|---|---|---|---|---|---|---|---|
| insult | like_count | +0.0554 | 0.00191 | +0.0584 | 0.06618 | +0.0030 | reproduces |
| nuance_experimental | like_count | -0.0529 | 0.00392 | -0.0539 | 0.06618 | -0.0010 | reproduces |
| toxicity | like_count | +0.0509 | 0.00481 | +0.0511 | 0.06618 | +0.0002 | reproduces |
| identity_attack | like_count | +0.0497 | 0.0074 | +0.0302 | 0.35098 | -0.0195 | reproduces |
| unsubstantial | total_engagement | -0.0488 | 0.00573 | -0.0298 | 0.35098 | +0.0190 | reproduces |
| insult | total_engagement | +0.0487 | 0.00698 | +0.0427 | 0.17481 | -0.0060 | reproduces |
| toxicity | total_engagement | +0.0442 | 0.01491 | +0.0352 | 0.25907 | -0.0090 | reproduces |
| reasoning_experimental | total_engagement | -0.0409 | 0.03011 | -0.0225 | 0.45613 | +0.0184 | reproduces |
| insult | reply_count | +0.0397 | 0.03591 | +0.0239 | 0.43379 | -0.0158 | reproduces |
| identity_attack | total_engagement | +0.0379 | 0.04152 | +0.0238 | 0.43736 | -0.0141 | reproduces |
| inflammatory | reply_count | -0.0379 | 0.04152 | -0.0185 | 0.57968 | +0.0194 | reproduces |
| curiosity_experimental | like_count | -0.0867 | 0.0 | -0.0490 | 0.06618 | +0.0377 | reply-driven |
| respect_experimental | total_engagement | -0.0804 | 1e-05 | -0.0165 | 0.66702 | +0.0639 | reply-driven |
| nuance_experimental | reply_count | -0.0790 | 1e-05 | -0.0470 | 0.06618 | +0.0320 | reply-driven |
| nuance_experimental | total_engagement | -0.0766 | 1e-05 | -0.0433 | 0.16065 | +0.0333 | reply-driven |
| affinity_experimental | reply_count | -0.0758 | 1e-05 | -0.0029 | 0.91419 | +0.0729 | reply-driven |
| reasoning_experimental | reply_count | -0.0757 | 1e-05 | -0.0370 | 0.24318 | +0.0387 | reply-driven |
| unsubstantial | reply_count | -0.0726 | 3e-05 | -0.0124 | 0.73076 | +0.0602 | reply-driven |
| curiosity_experimental | total_engagement | -0.0692 | 8e-05 | -0.0299 | 0.35098 | +0.0393 | reply-driven |
| respect_experimental | like_count | -0.0690 | 0.00018 | -0.0170 | 0.65526 | +0.0520 | reply-driven |
| attack_on_commenter | reply_count | -0.0585 | 0.00135 | -0.0060 | 0.84857 | +0.0525 | reply-driven |
| respect_experimental | reply_count | -0.0515 | 0.00481 | -0.0052 | 0.86384 | +0.0463 | reply-driven |
| obscene | reply_count | -0.0496 | 0.0074 | -0.0120 | 0.73076 | +0.0376 | reply-driven |
| severe_toxicity | like_count | +0.0465 | 0.01141 | +0.0249 | 0.43379 | -0.0216 | reply-driven |
| flirtation | reply_count | +0.0439 | 0.01838 | +0.0189 | 0.516 | -0.0250 | reply-driven |
| affinity_experimental | total_engagement | -0.0739 | 1e-05 | +0.0230 | 0.45613 | +0.0969 | sign flips |
| attack_on_author | reply_count | -0.0607 | 0.00045 | +0.0048 | 0.88051 | +0.0655 | sign flips |
| compassion_experimental | reply_count | -0.0588 | 0.00064 | +0.0133 | 0.72487 | +0.0721 | sign flips |
| affinity_experimental | like_count | -0.0471 | 0.00771 | +0.0271 | 0.43379 | +0.0742 | sign flips |
| compassion_experimental | total_engagement | -0.0385 | 0.03591 | +0.0366 | 0.24318 | +0.0751 | sign flips |
| personal_story_experimental | total_engagement | -0.0373 | 0.04152 | +0.0061 | 0.8209 | +0.0434 | sign flips |

## Summary

- **reproduces**: 11
- **reply-driven**: 14
- **sign flips**: 6

Associations marked `reply-driven` or `sign flips` should not be reported as content effects. A reply's engagement is largely a function of the thread it sits in -- the parent's reach and the conversation's size -- and neither is in this table, so those estimates absorb thread context that cannot be conditioned away.

Associations marked `reproduces` hold in both the mixed population and among root posts, where exposure is better measured. Those are the defensible ones.
