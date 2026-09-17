# German Bluesky: Perspective scores vs interaction counts

> **PROVISIONAL — snapshot `early`.** Counts were measured at 12-36h of post age, before engagement has settled, and the 2.9x spread in age is itself correlated with posting hour. Re-run against the `settled` snapshot (>=48h) before drawing conclusions.

n = 10000 posts. Estimates are associations, not causal effects.
SEs clustered by author; q-values are Benjamini-Hochberg across the attribute x outcome grid.

## Within-author associations passing FDR < 0.05

Author fixed effects: does an author's higher-scoring post beat their own average?

| attribute | outcome | within-author r | q | raw Spearman r | partial r | n authors | flag |
|---|---|---|---|---|---|---|---|
| curiosity_experimental | like_count | -0.0867 | 0.0 | -0.0647 | -0.0683 | 1947 | keep |
| respect_experimental | total_engagement | -0.0804 | 1e-05 | -0.0251 | -0.0299 | 1947 | keep |
| nuance_experimental | reply_count | -0.079 | 1e-05 | 0.0287 | -0.0691 | 1947 | low_power |
| nuance_experimental | total_engagement | -0.0766 | 1e-05 | 0.0004 | -0.0872 | 1947 | low_power |
| affinity_experimental | reply_count | -0.0758 | 1e-05 | -0.0121 | -0.0476 | 1947 | keep |
| reasoning_experimental | reply_count | -0.0757 | 1e-05 | 0.0627 | -0.0543 | 1947 | keep |
| affinity_experimental | total_engagement | -0.0739 | 1e-05 | 0.0101 | -0.0294 | 1947 | keep |
| unsubstantial | reply_count | -0.0726 | 3e-05 | -0.1666 | -0.0784 | 1947 | keep |
| curiosity_experimental | total_engagement | -0.0692 | 8e-05 | -0.0547 | -0.0612 | 1947 | keep |
| respect_experimental | like_count | -0.069 | 0.00018 | -0.0141 | -0.0279 | 1947 | keep |
| attack_on_author | reply_count | -0.0607 | 0.00045 | -0.1293 | -0.0562 | 1947 | low_power |
| compassion_experimental | reply_count | -0.0588 | 0.00064 | 0.0321 | -0.0349 | 1947 | keep |
| attack_on_commenter | reply_count | -0.0585 | 0.00135 | -0.139 | -0.0652 | 1947 | low_power |
| insult | like_count | 0.0554 | 0.00191 | 0.0852 | 0.0676 | 1947 | keep |
| nuance_experimental | like_count | -0.0529 | 0.00392 | -0.0224 | -0.0751 | 1947 | low_power |
| respect_experimental | reply_count | -0.0515 | 0.00481 | -0.0096 | -0.0108 | 1947 | keep |
| toxicity | like_count | 0.0509 | 0.00481 | 0.0805 | 0.0691 | 1947 | keep |
| identity_attack | like_count | 0.0497 | 0.0074 | 0.0596 | 0.053 | 1947 | low_power |
| obscene | reply_count | -0.0496 | 0.0074 | -0.1426 | -0.0614 | 1947 | keep |
| unsubstantial | total_engagement | -0.0488 | 0.00573 | -0.1392 | -0.0768 | 1947 | keep |
| insult | total_engagement | 0.0487 | 0.00698 | 0.0888 | 0.0498 | 1947 | keep |
| affinity_experimental | like_count | -0.0471 | 0.00771 | 0.0238 | -0.0103 | 1947 | keep |
| severe_toxicity | like_count | 0.0465 | 0.01141 | 0.0613 | 0.0635 | 1947 | low_power |
| toxicity | total_engagement | 0.0442 | 0.01491 | 0.0799 | 0.0511 | 1947 | keep |
| flirtation | reply_count | 0.0439 | 0.01838 | 0.0879 | 0.0222 | 1947 | keep |
| reasoning_experimental | total_engagement | -0.0409 | 0.03011 | 0.0508 | -0.0584 | 1947 | keep |
| insult | reply_count | 0.0397 | 0.03591 | 0.0391 | -0.018 | 1947 | keep |
| compassion_experimental | total_engagement | -0.0385 | 0.03591 | 0.046 | -0.0095 | 1947 | keep |
| identity_attack | total_engagement | 0.0379 | 0.04152 | 0.0613 | 0.0373 | 1947 | low_power |
| inflammatory | reply_count | -0.0379 | 0.04152 | -0.1139 | -0.0567 | 1947 | low_power |
| personal_story_experimental | total_engagement | -0.0373 | 0.04152 | 0.0729 | 0.0198 | 1947 | keep |

## Effect sizes

The surviving associations are small -- |r| below 0.1 throughout, i.e. under 1% of variance in engagement. They are detectable because n is large and the within-author design is efficient, not because content scores are a major driver of engagement. Author reach dominates, which is exactly why the raw column is so much larger for some attributes.

## Reading the three columns

A large raw Spearman r that collapses in the partial and within-author columns is an author-reach artifact, not a content effect. Only the within-author column survives author-level confounding.

`low_power` attributes have too few positive cases to detect an effect; a null there means no power, not no effect. `drop` attributes (INCOHERENT, LIKELY_TO_REJECT) score the language rather than the construct on German text and are excluded.
