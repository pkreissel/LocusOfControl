# Feed audit: internal vs. external locus of control

Judge: fine-tuned decider-2b (`decider_ft`, short rubric), local weights. Posts are
what each feed returned to an **anonymous** `getFeed` call, restricted to posts the
deployment would embed (≥ 6 words). Classes are the argmax of the decider's
three-way distribution; brackets are 95% Wilson intervals. Sorted by mean
P(internal) − P(external).

| feed | lang | n | internal | external | neutral | mean P(int)−P(ext) | first-25 int/ext |
|---|---|---:|---:|---:|---:|---:|---:|
| game-dev | en | 196 | 25% [19%–32%] | 8% [5%–12%] | 67% | +0.202 | 24% / 0% |
| gardening | en | 89 | 22% [15%–32%] | 3% [1%–9%] | 74% | +0.198 | 12% / 0% |
| booksky | en | 193 | 16% [11%–21%] | 10% [7%–15%] | 74% | +0.062 | 8% / 4% |
| science | en | 189 | 14% [10%–19%] | 6% [4%–11%] | 80% | +0.059 | 28% / 0% |
| artists-trending | en | 182 | 5% [3%–9%] | 5% [3%–10%] | 90% | +0.050 | 0% / 16% |
| birds | en | 178 | 5% [3%–9%] | 3% [2%–7%] | 92% | +0.016 | 0% / 4% |
| astronomy | en | 189 | 6% [4%–11%] | 6% [3%–10%] | 88% | +0.006 | 8% / 4% |
| hot | en | 150 | 6% [3%–11%] | 13% [8%–19%] | 81% | -0.040 | 4% / 12% |
| feed-forward | en | 152 | 18% [13%–25%] | 42% [35%–50%] | 40% | -0.191 | 40% / 20% |
| mysky-your-feed | en | 189 | 17% [12%–23%] | 48% [41%–55%] | 35% | -0.248 | 20% / 36% |
| what-s-hot-classic | en | 149 | 13% [8%–19%] | 44% [37%–52%] | 43% | -0.286 | 12% / 28% |
| discover | en | 172 | 12% [8%–17%] | 44% [36%–51%] | 45% | -0.287 | 16% / 32% |
| blacksky | en | 137 | 12% [7%–18%] | 52% [44%–60%] | 36% | -0.295 | 16% / 36% |
| for-you | en | 183 | 15% [10%–21%] | 49% [42%–56%] | 36% | -0.303 | 16% / 32% |
| eurosky-fu | en | 181 | 15% [10%–21%] | 51% [44%–59%] | 34% | -0.309 | 28% / 36% |
| news | ? | 197 | 10% [6%–15%] | 52% [45%–59%] | 38% | -0.358 | 12% / 36% |
| onlyposts | de | 183 | 8% [5%–12%] | 61% [53%–67%] | 32% | -0.440 | 8% / 72% |
| not-cias-do-momento | ? | 193 | 6% [3%–10%] | 61% [54%–68%] | 33% | -0.456 | 8% / 60% |
| popular-with-friends | de | 189 | 12% [8%–17%] | 67% [60%–73%] | 21% | -0.468 | 12% / 72% |
| mutuals | de | 179 | 5% [3%–9%] | 70% [63%–77%] | 25% | -0.527 | 0% / 92% |

Corpus-wide across 2924 unique scored posts: 12% internal,
32% external, 56% neutral.

## Read this before quoting a number

- The judge is a distilled model (gold AP 0.769 vs. the paid teacher's 0.788), not
  ground truth. Compare feeds with each other; do not read the absolute shares as
  the prevalence of anything.
- Anonymous requests: personalised feeds show what a viewer with no graph gets.
- Samples are the newest N posts each feed returned at collection time and differ in
  language mix (column `lang` is the most common first-listed language), and the
  decider was validated on a mixed EN/DE gold set only.
- "first-25" is the first 25 posts in the order the feed served them, n = 25, so each
  step is 4 points: treat it as indicative.
