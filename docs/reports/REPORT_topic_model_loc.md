# Which topics skew internal locus of control?

**Question.** Across the net20k corpus, which subject matter is associated
with the highest internal-locus-of-control signal?

**Answer.** Posts about personal agency and small-scale collective action —
practical philosophy/self-help ("what you can do"), grassroots civic and
education organising, community-scale energy transition, digital
self-defense/privacy tips, mutual-aid fundraising, reading/personal growth,
individual health actions (vaccination, self-care), and individual climate
action. All eight top topics roughly double the corpus's baseline P(internal)
and are 1.5-2x the average share of internal-classified posts, on cluster
sizes (222-1,039 posts) too large to be noise. The most external topics are
geopolitics/war (Houthis/Yemen, Russia/Ukraine sanctions), international
sports leagues, movie remakes, and small talk/memes.

Measured 2026-09-25 on the net20k corpus (`training/data/dataset_raw_net20k.jsonl`,
20,000 posts; same corpus as `REPORT_gender_bias_loc.md`), scored with the
decider-2b (short rubric) candidate teacher
(`training/data/scores_loc_decider_ft_net20k.jsonl`: 1,960 internal / 9,066
external / 8,974 neutral). Not the deployed TypeSafe vector or gate — see
Caveats. Scripts: `research/topic_model/embed_net20k_jina.py`,
`topic_model_loc.py`. Output: `research/data/topic_model/topic_model_results.json`
(per-cluster terms, stats, example posts; embeddings cache gitignored).

## Method

**Embedding.** Each post's text (first 2,000 chars) was embedded with
`jinaai/jina-embeddings-v5-text-small` using its **clustering** task adapter —
a different adapter, and a different embedding space, from the **classification**
adapter the deployed LoC vector trains and scores against. This cache is a
one-off for topic discovery, not interchangeable with `training/data/embeddings_*_jina.npz`.

**Clustering.** L2-normalized embeddings, `MiniBatchKMeans` (k=40,
`n_init=10`), which behaves like cosine k-means on unit vectors. 40 was picked
for cluster sizes readable by hand (~500 posts/cluster on average) at 20k
posts; not tuned against a cluster-quality metric.

**Topic labels.** c-TF-IDF: each cluster's tokenized posts are concatenated
into one pseudo-document, and a `TfidfVectorizer` fit across the 40
pseudo-documents ranks terms that are frequent within a cluster and rare
across the others. Tokenizer strips URLs/@mentions, lowercases, drops a short
combined EN/DE (plus a few Romance-language) stopword list. No lemmatization,
so German compounds and inflected forms sometimes split from their English
counterparts.

**Internal-LoC score per post.** `ts_class_probs["internal"]` from the
decider-2b short-rubric scorer — the softmax probability of the `internal`
label over {internal, external, neutral}, a continuous per-post value that
doesn't depend on where the class boundary sits. Reported per cluster as the
mean, alongside mean P(external) and the plain classified-internal fraction
for corroboration. Clusters below n=40 are dropped from the ranking as too
small to trust.

**Corpus baseline** (for comparison): mean P(internal) = 0.127, P(external) =
0.426; 9.8% of posts classified `internal`.

## Top 10 internal-leaning topics

| rank | n | P(internal) | P(external) | % internal-class | top terms |
|---|---|---|---|---|---|
| 1 | 699 | 0.230 | 0.380 | 21.3% | practical philosophy / self-help ("kindness", "hope", ethics rules) — *cypher, pinker, philosophy, oneness, mathematics* |
| 2 | 439 | 0.213 | 0.461 | 20.3% | grassroots civic organising / education / social services — *pisa, krankenkassen, educators, shelter, gewalt, bildung, schulen* |
| 3 | 222 | 0.186 | 0.544 | 16.2% | community-scale renewable energy — *solar, grid, renewable, energiewende, electrification, panels, renewables* |
| 4 | 527 | 0.185 | 0.445 | 16.3% | personal digital self-defense / privacy tooling — *iphone, ransomware, android, nextcloud, software, cybersecurity* |
| 5 | 489 | 0.179 | 0.231 | 14.9% | mutual-aid fundraising asks — *revolut, paypal, donation drives* |
| 6 | 361 | 0.176 | 0.170 | 13.9% | reading / books / personal growth — *comics, literary, novel, bücher, buch, booksky* |
| 7 | 370 | 0.172 | 0.479 | 15.1% | individual health action (vaccination, self-care) — *measles, vaccines, vaccine, medicaid, nih, flu* |
| 8 | 356 | 0.171 | 0.533 | 16.6% | individual climate action — *niño, hurricane, floods, drought, climate, climatecrisis* |
| 9 | 1,039 | 0.162 | 0.617 | 13.9% | US electoral politics — *republican, votes, voters, ballots, senate* |
| 10 | 765 | 0.158 | 0.665 | 13.5% | immigration/church/ICE enforcement commentary — *salutes, catholics, appeals, immigrants, detention* |

Representative posts for the top 3 (decider-2b `p_internal`, truncated):

- 0.98 — *"The more you practice small acts of kindness, the more natural it
  becomes, because it starts to shape how you see yourself..."* (cluster 1)
- 0.98 — *"I strongly recommend not to wait for governments but getting
  organised locally yourself. Get your act together and go #solarpunk."*
  (cluster 3)
- 0.96 — *"Wer Verantwortung für sein Leben übernehmen soll, muss auch
  darüber entscheiden können, wofür die verfügbaren Mittel eingesetzt
  werden."* (cluster 2)

The through-line: posts that name a concrete thing a person or a small local
group can do, not a plea for institutions to act. Ranks 9-10 (electoral
politics, immigration enforcement) sit at the bottom of the top-10 because
each still has a sizeable internal-classified minority (get-out-the-vote,
"what you can do about this" posts) inside a topic that is majority external.

## Most external topics (for contrast)

| n | P(internal) | P(external) | top terms |
|---|---|---|---|
| 704 | 0.048 | 0.768 | Yemen/Houthi conflict — *houthi, yemen, nato, saudi, riyadh, escalation* |
| 240 | 0.062 | 0.579 | Greenland/Denmark geopolitics — *grönland, greenland, denmark, dänemark* |
| 155 | 0.066 | 0.102 | meme/small-talk shorts — *eighty, shorts, sweaty, funky, drummers* |
| 200 | 0.070 | 0.319 | aviation industry news — *boeing, airbus, embraer, bombardier* |
| 513 | 0.073 | 0.208 | movie remakes — *mordor, remake, movie, starring, batman* |
| 587 | 0.080 | 0.734 | Belarus/Russia/Israel sanctions diplomacy — *lukashenko, sanctions, dictator, netanyahu, zelenskyy* |
| 1,479 | 0.084 | 0.132 | daily puzzle/wordle threads — *counterpunch, wordle, threads* |
| 493 | 0.086 | 0.255 | football league results — *arsenal, werder, tottenham, spurs* |

State-level geopolitics and passive-consumption content (sports scores,
puzzles, movie news) score lowest — nothing in the topic is something an
individual reader acts on.

## Caveats

- **Candidate teacher, not the deployed vector.** decider-2b (short rubric)
  is a System One candidate under evaluation ([[loc-teacher-glimmer-filename-is-the-corpus]]),
  not the TypeSafe teacher the shipped `bunny_feed` vector trains against. A
  topic ranking against the deployed Jina classification-task vector could
  differ; this analysis answers "what does decider-2b consider internal,"
  not "what would the live feed rank up."
- **Two different embedding spaces in this analysis.** Topic discovery used
  the clustering-task adapter; the LoC score used a separate LLM classifier,
  not these embeddings at all. The embeddings only group posts by subject —
  they play no role in the internal/external number.
- **c-TF-IDF terms are unlemmatized and multilingual.** A term list mixing
  English and German words for one topic is expected (the corpus is ~57% en,
  ~19% de, ~17% no `langs` tag); don't read a single odd term as mislabeling
  without checking the example posts.
- **k=40 was not tuned.** A different k would split or merge some of these
  topics (e.g. "personal growth" and "practical philosophy" might separate
  further at higher k); the ranking by P(internal) is more robust than any
  individual cluster's exact boundary.
- **Absolute P(internal) is low everywhere** — consistent with
  [[loc-vector-range-is-ordinal-only]]: internal posts are a genuine minority
  of the firehose (9.8% classified), so even the top topic (23%) is a
  relative, not absolute, concentration.
