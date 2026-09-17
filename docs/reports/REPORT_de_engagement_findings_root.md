# German Bluesky: Perspective scores vs interaction counts

> **PROVISIONAL — snapshot `early_root`.** Counts were measured at 13-37h of post age, before engagement has settled, and the 2.9x spread in age is itself correlated with posting hour. Re-run against the `settled` snapshot (>=48h) before drawing conclusions.

n = 10000 posts. Estimates are associations, not causal effects.
SEs clustered by author; q-values are Benjamini-Hochberg across the attribute x outcome grid.

## Within-author associations passing FDR < 0.05

Author fixed effects: does an author's higher-scoring post beat their own average?

| attribute | outcome | within-author r | q | raw Spearman r | partial r | n authors | flag |
|---|---|---|---|---|---|---|---|
| _none_ | | | | | | | |

## Effect sizes

The surviving associations are small -- |r| below 0.1 throughout, i.e. under 1% of variance in engagement. They are detectable because n is large and the within-author design is efficient, not because content scores are a major driver of engagement. Author reach dominates, which is exactly why the raw column is so much larger for some attributes.

## Reading the three columns

A large raw Spearman r that collapses in the partial and within-author columns is an author-reach artifact, not a content effect. Only the within-author column survives author-level confounding.

`low_power` attributes have too few positive cases to detect an effect; a null there means no power, not no effect. `drop` attributes (INCOHERENT, LIKELY_TO_REJECT) score the language rather than the construct on German text and are excluded.
