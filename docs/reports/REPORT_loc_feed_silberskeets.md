# Locus of control on `silberskeets`

573 posts, 488 of them past the feed's own minWords gate. Every score below is cached; this report recomputes nothing.

- vector: version `704abee7`, `scale*cos+bias` — its Ridge prediction on the teacher's 0-9 axis — and `vector_band`, that score projected onto the gold labels' band axis, which is how it becomes comparable to the reranker at all (Ridge shrinkage keeps the raw prediction inside [0.7, 3.3])
- reranker: `loc-reranker-jina-v3`, raw `rr_logit` (a ranking score, on no scale of its own) plus `rr_band`, projected onto the same band axis by the same least-squares fit on the same gold rows

## Distributions

Raw model outputs first — those are what every ranking number here is computed on — then the same scores on the shared band axis.

| | p5 | p25 | p50 | p75 | p95 | mean |
|---|---|---|---|---|---|---|
| vector | 1.250 | 1.725 | 2.051 | 2.361 | 2.765 | 2.039 |
| reranker | 0.033 | 0.048 | 0.062 | 0.081 | 0.137 | 0.073 |
| vector_band | 0.424 | 1.361 | 2.005 | 2.617 | 3.413 | 1.985 |
| rr_band | 1.987 | 2.109 | 2.229 | 2.389 | 2.853 | 2.323 |
| vector_prob | 0.777 | 0.849 | 0.886 | 0.914 | 0.941 | 0.877 |

## Vector vs reranker

- Spearman 0.186 over all 573 posts
- the deployed 0.88 gate admits 313 posts (55%); at that same length the two rankings share 191 of them (Jaccard 0.44)

Where they disagree most is where a teacher run would be worth the hours — these are the posts the two models cannot both be right about.

### Vector ranks far above the reranker

| rank_vector | rank_reranker | vector_score | rr_logit | text |
|---|---|---|---|---|
| 31 | 547 | 2.75 | 0.03 | Von der Ottilien Kirche hatten wir einen schönen Ausblick auf die Berge auf dene |
| 28 | 541 | 2.77 | 0.03 | Die HU Berlin besitzt nun ein umfassendes Archiv zur Geschichte von HIV und Aids |
| 53 | 548 | 2.64 | 0.03 | Dazu gehört der Schulterschluss mit dem ostdeutschen Dorfwähler, der irgendwie e |
| 11 | 498 | 2.93 | 0.04 | Bin aktuell wenig hier, weil ich ein paar Dinge sortiere, aber kann berichten, d |
| 27 | 511 | 2.77 | 0.04 | Swirling timber  Architecture by: Urba Folia 📍Jardins Mallet-Stevens, Croix, Fra |
| 87 | 571 | 2.54 | 0.02 | #Wochenendlektüre Danke, @gfelbermayr.bsky.social & @wifo.bsky.social |
| 3 | 468 | 3.11 | 0.04 | Gestern tollen Abend gehabt, heute netten Arbeitstag mit Fromage et Baguette mit |
| 23 | 477 | 2.79 | 0.04 | Kühe hören Musik am liebsten auf Viehnyl. ☝️ |
| 124 | 570 | 2.44 | 0.02 | Ich bin nächste Woche zu einer Party eingeladen, soll ich eine Packung Ferrero K |
| 104 | 532 | 2.48 | 0.04 | "Dann also Erbitterungswahlen" aber für das 21. Jahrhundert. Ein völliges Versch |

### Reranker ranks far above the vector

| rank_vector | rank_reranker | vector_score | rr_logit | text |
|---|---|---|---|---|
| 530 | 33 | 1.36 | 0.13 | Ich weiß es gibt vielen Leuten vieles aber es gibt mmn nichts dümmeres als Rauch |
| 521 | 29 | 1.41 | 0.14 | Ich bin mit beiden Füßen in Hundescheiße getreten i b m u |
| 500 | 9 | 1.51 | 0.21 | My little son |
| 570 | 89 | 0.90 | 0.09 | Oh scheiße habt ihr das mit den Flugzeugen in New York gehört |
| 532 | 65 | 1.35 | 0.11 | Hab mir die Radieschen von unten angesehen..... |
| 522 | 59 | 1.41 | 0.11 | tatsächlich ja, besonders giftig schiebt die Version für Beerdigungen |
| 471 | 16 | 1.61 | 0.17 | On this day in 2002 I got extremely fucked up on percocets at work and drove my  |
| 486 | 35 | 1.55 | 0.13 | Froschstudien |
| 478 | 30 | 1.60 | 0.14 | zum bus gejoggt und bus bekommen |
| 490 | 49 | 1.54 | 0.11 | ihr rückflug wurde gecancelt sie wollte uns nicht alleine lassen mein herz ist w |

### Top 15 by vector score

| vector_score | vector_prob | vector_band | rr_logit | rr_band | text |
|---|---|---|---|---|---|
| 3.26 | 0.96 | 4.38 | 0.21 | 3.46 | Hab endlich mein HP Tattoo Cover Up angefragt 🙌🏻✨ good riddance |
| 3.21 | 0.96 | 4.28 | 0.07 | 2.29 | Der Wald eilt nicht.  Und doch wächst er jeden Tag.   Vielleicht müssen auch wir |
| 3.11 | 0.96 | 4.09 | 0.04 | 2.07 | Gestern tollen Abend gehabt, heute netten Arbeitstag mit Fromage et Baguette mit |
| 3.10 | 0.96 | 4.07 | 0.06 | 2.20 | I had the pleasure of attending the grand opening of the 360° Panorama Konstanz  |
| 2.99 | 0.95 | 3.86 | 0.08 | 2.38 | Es werden in Zukunft ganz sicher einige AuDHS-Comics kommen, hilft mir auch selb |
| 2.95 | 0.95 | 3.77 | 0.09 | 2.48 | Soll euch ganz lieb grüßen |
| 2.95 | 0.95 | 3.77 | 0.07 | 2.31 | Schönen Abend  🚪  💫  #TürzumAbend #selbstfotografiertvAaxii ;) |
| 2.94 | 0.95 | 3.76 | 0.08 | 2.35 | Ein schönes und erholsames Wochenende🍀 |
| 2.94 | 0.95 | 3.75 | 0.08 | 2.35 | Herzlichen Glückwunsch, es ist nicht nur Autismus, sondern auch ADHS (bei mir) 🎉 |
| 2.93 | 0.95 | 3.74 | 0.07 | 2.27 | Ich rieche nach Sonne, See und Wind. Herrlich. |
| 2.93 | 0.95 | 3.74 | 0.04 | 2.05 | Bin aktuell wenig hier, weil ich ein paar Dinge sortiere, aber kann berichten, d |
| 2.89 | 0.95 | 3.66 | 0.08 | 2.36 | Awesome 💙🙏🏻 |
| 2.89 | 0.95 | 3.66 | 0.12 | 2.75 | Heute war ich froh um die Sitzheizung im Auto. 🥶🥶 |
| 2.89 | 0.95 | 3.65 | 0.08 | 2.42 | Enten am Strand 🦆🍀 |
| 2.88 | 0.95 | 3.64 | 0.08 | 2.34 | Vielen Dank ihr Spatzen! 💕 |

### Top 15 by reranker score

| vector_score | vector_prob | vector_band | rr_logit | rr_band | text |
|---|---|---|---|---|---|
| 2.33 | 0.91 | 2.55 | 0.96 | 9.00 | Im Fitnessstudio war eine ältere Dame auf dem Laufband, die ihre Handtasche umge |
| 2.10 | 0.89 | 2.10 | 0.67 | 7.30 | Pedaling to the demo, silence isn't an option. 2026's heatwaves and fires showed |
| 2.26 | 0.91 | 2.42 | 0.65 | 7.15 | Am kommenden Dienstag ist Freedom Day. Wir kaufen so viele Menschen wie möglich  |
| 2.53 | 0.93 | 2.96 | 0.56 | 6.42 | Wie ich mich grade in Frankreich auf das #arminia Spiel vorbereite |
| 2.70 | 0.94 | 3.28 | 0.33 | 4.43 | Vamos al gym un rato 💪 |
| 2.30 | 0.91 | 2.50 | 0.24 | 3.72 | Die AfD bedroht Kultur und Meinungsfreiheit. Unser Auftrag als Demokrat*innen is |
| 1.98 | 0.88 | 1.86 | 0.23 | 3.64 | Brazilian Jiu-Jitsu |
| 3.26 | 0.96 | 4.38 | 0.21 | 3.46 | Hab endlich mein HP Tattoo Cover Up angefragt 🙌🏻✨ good riddance |
| 1.51 | 0.82 | 0.94 | 0.21 | 3.46 | My little son |
| 1.91 | 0.87 | 1.73 | 0.21 | 3.44 | Weg damit @ihatenig.bsky.social blockempfehlung |
| 1.88 | 0.87 | 1.66 | 0.19 | 3.33 | Freitach |
| 2.16 | 0.90 | 2.22 | 0.18 | 3.25 | Soll ich schon kochen es gibt Udon mit gemüse und erdnuss🆘 |
| 1.82 | 0.86 | 1.54 | 0.18 | 3.18 | Selfcare Tag hat auch was, aber lowkey will ich immer noch wegfahren, weiß nur n |
| 1.89 | 0.87 | 1.69 | 0.18 | 3.17 | Tight |
| 2.16 | 0.90 | 2.23 | 0.17 | 3.16 | Sis & Bros, hauswirtschaftliche Tätigkeiten erfordern meine Aufmerksamkeit 🫡 ble |
