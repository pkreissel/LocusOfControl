# The largest German-speaking Bluesky accounts, on the locus-of-control axis

Measured 2026-09-18. 37 accounts, 2,691 posts from the 90 days to that date, scored by
TypeSafe System One (`jev-latest`) through `ts_questions.py` — the same three questions,
the same rubric, and the same ledger as the training teacher.

Code: `research/de_study/collect_top_accounts.py` (collection),
`research/de_study/report_de_top_loc.py` (the join),
`research/de_study/make_sharepic.py` (the published chart),
`research/de_study/make_account_sharepics.py` (the per-account charts, sharing
`sharepic_common.py` with it).
Raw numbers: `research/data/de_study/REPORT_de_top_loc.json`;
`REPORT_de_top_loc_top10.json` for the ten with the most reach
(`DE_TOP_LIMIT=10 python3 report_de_top_loc.py`); and
`REPORT_de_top_loc_top10_by_internal_share.json` for the ten with the most
agency (`DE_TOP_BY=internal_share` with it).
Posts and scores: `de_top50_posts.jsonl`, `de_top_corpus_90d.jsonl`,
`scores_loc_typesafe_de_top90.jsonl`.

Cost: 2,641 billable calls, $0.13, 0 errors. The ledger absorbed every answer, so any
re-run of this corpus is free.

## Headline: external posts take two thirds of all engagement

All 2,691 scored posts, interactions = likes + reposts + replies + quotes.

| class | posts | share of posts | interactions | share of interactions | likes | reposts | replies | quotes | mean/post | median/post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external | 1,523 | 56.6% | **938,187** | **65.8%** | 719,239 | 179,932 | 30,525 | 8,491 | 616 | 280 |
| neutral | 765 | 28.4% | **269,408** | **18.9%** | 225,139 | 32,235 | 10,250 | 1,784 | 352 | 225 |
| internal | 403 | 15.0% | **217,462** | **15.3%** | 165,757 | 44,085 | 5,840 | 1,780 | 540 | 233 |
| **all** | **2,691** | 100% | **1,425,057** | 100% | 1,110,135 | 256,252 | 46,615 | 12,055 | 530 | |

Restricting to posts aged 2–90 days, so that nothing is counted before it has finished
accumulating, changes the shares by less than a point:

| class | posts | share of posts | interactions | share of interactions | mean/post | median/post |
|---|---:|---:|---:|---:|---:|---:|
| external | 1,303 | 55.3% | **870,237** | **65.2%** | 668 | 336 |
| neutral | 686 | 29.1% | **256,798** | **19.2%** | 374 | 242.5 |
| internal | 366 | 15.5% | **207,542** | **15.6%** | 567 | 255.5 |
| **all** | **2,355** | 100% | **1,334,577** | 100% | 567 | |

Two separate things produce that 65.8%. External posts are **more numerous** (56.6% of
what these accounts publish) and they are **individually bigger** (median 280 against
233 and 225). Neither alone would do it; together they leave internal posts with 15.3%
of the engagement in German Bluesky's largest feeds.

The reposts column is the one exception worth naming. Internal posts earn 0.27 reposts
per like against external's 0.25 and neutral's 0.14 — agency-promoting posts are passed
on at a higher rate than they are liked, which is the one behaviour in this table that
favours them.

## The relationship holds inside individual accounts

Pooled across the matured posts, Spearman(`ts_score`, interactions) = **−0.231**
(n = 2,355). That figure on its own proves little: accounts differ in both reach and
tone, and the correlation could be entirely an artifact of which accounts post which way.

The within-account version removes that. For each account with ≥25 matured posts, does
*its own* more-internal post beat *its own* average?

- median within-account Spearman: **−0.249** (30 accounts)
- negative in **26 of 30** accounts

So the effect is not a composition artifact. The same author, posting to the same
followers, gets less engagement on the post that promotes agency. The four exceptions
are FragDenStaat (+0.04), Pia Lamberty (+0.05), Mimikama (+0.06) and Stefan Niggemeier
(+0.01) — all near zero, none a positive relationship.

This is the mechanism `feed-forward` exists to counteract, measured on the accounts that
dominate the pool it draws from. It also sets an expectation for the feed: ranking on
internal content means ranking *against* the engagement signal, so a candidate pool
assembled by engagement will be 15% internal before the vector ever sees it.

## Per account

`LoC` is mean `ts_score` on the 5-level Score axis: 0 = strong external, 2 = neutral,
4 = internal and concrete. `median` is median interactions on posts aged 2–90 days;
`n` is how many posts that median rests on. `r` is the within-account Spearman.

Ordered by engagement, so the table reads as the engagement ranking with the axis attached.

| account | handle | followers | median | n | LoC | int% | neu% | ext% | scored | r |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ricarda Lang | `ricardalang.eurosky.social` | 60,371 | 2,318 | 15 | 1.41 | 13% | 20% | 67% | 15 | — |
| Lorenz Meyer | `thereallorenzmeyer.bsky.social` | 46,252 | 1,856 | 51 | 1.65 | 19% | 13% | 67% | 52 | −0.24 |
| Der Volksverpetzer | `volksverpetzer.de` | 129,666 | 968 | 79 | 1.68 | 22% | 7% | 71% | 100 | −0.23 |
| Zentrum für Politische Schönheit | `politicalbeauty.bsky.social` | 51,764 | 923 | 53 | 1.38 | 12% | 4% | 84% | 56 | −0.33 |
| Christian Stöcker | `christianstoecker.de` | 33,943 | 798 | 76 | 1.45 | 8% | 10% | 82% | 78 | −0.29 |
| Florian Aigner | `florianaigner.at` | 49,480 | 762 | 96 | 1.78 | 11% | 46% | 43% | 100 | −0.41 |
| Marina Weisband | `afelia.bsky.social` | 76,566 | 697 | 31 | 1.86 | 25% | 19% | 56% | 36 | −0.58 |
| **FragDenStaat** | `fragdenstaat.de` | 34,949 | 671 | 37 | **2.71** | **59%** | 3% | 38% | 39 | +0.04 |
| Maja Göpel | `beyond-ideology.bsky.social` | 40,374 | 657 | 18 | 1.73 | 33% | 11% | 56% | 18 | — |
| Natascha Strobl | `nataschastrobl.bsky.social` | 70,108 | 635 | 91 | 1.71 | 15% | 31% | 54% | 100 | −0.31 |
| Stefan Rahmstorf | `rahmstorf.bsky.social` | 63,795 | 528 | 97 | 1.53 | 12% | 18% | 70% | 100 | −0.26 |
| Carlo Masala | `carlomasala1.bsky.social` | 73,634 | 519 | 46 | 1.60 | 9% | 48% | 43% | 46 | −0.38 |
| Ralph Ruthe | `ruthe.de` | 59,262 | 481 | 96 | 2.19 | 16% | 71% | 13% | 100 | −0.30 |
| Georg Restle | `georgrestle.bsky.social` | 44,225 | 470 | 24 | 1.48 | 12% | 8% | 79% | 24 | — |
| Jan Böhmermann | `janboehm.bsky.social` | 112,803 | 459 | 38 | 1.91 | 8% | 47% | 45% | 38 | −0.01 |
| Katharina Nocun | `kattascha.bsky.social` | 41,050 | 394 | 95 | 1.71 | 16% | 26% | 58% | 100 | −0.24 |
| Annika Brockschmidt | `ardenthistorian.bsky.social` | 77,770 | 359 | 97 | 1.73 | 5% | 40% | 55% | 98 | −0.37 |
| Der Postillon | `der-postillon.com` | 85,182 | 346 | 86 | 1.88 | 5% | 63% | 32% | 100 | −0.43 |
| Stephan Anpalagan | `stephananpalagan.bsky.social` | 33,628 | 344 | 18 | 1.64 | 6% | 33% | 61% | 18 | — |
| Mario Sixtus | `sixtus.net` | 39,248 | 299 | 86 | 1.54 | 11% | 30% | 59% | 100 | −0.12 |
| Margarete Stokowski | `margaowski.bsky.social` | 37,311 | 287 | 67 | 2.08 | 11% | 57% | 31% | 70 | −0.04 |
| Krieg und Freitag | `kriegundfreitag.bsky.social` | 41,166 | 263 | 88 | 1.83 | 18% | 47% | 36% | 90 | −0.06 |
| Armin Wolf | `arminwolf.at` | 71,344 | 250 | 96 | 1.76 | 4% | 47% | 49% | 100 | −0.34 |
| Ingrid Brodnig | `brodnig.bsky.social` | 40,841 | 195 | 33 | 1.96 | 9% | 37% | 54% | 35 | −0.40 |
| Ruprecht Polenz | `polenz.bsky.social` | 41,834 | 190 | 70 | 1.73 | 23% | 19% | 57% | 77 | −0.26 |
| CORRECTIV | `correctiv.org` | 70,301 | 148 | 95 | 1.97 | 22% | 9% | 69% | 100 | −0.44 |
| **Pia Lamberty** | `pialamberty.bsky.social` | 39,229 | 145 | 52 | **2.72** | **67%** | 8% | 25% | 52 | +0.05 |
| netzpolitik.org | `netzpolitik.org` | 37,557 | 139 | 92 | 1.71 | 17% | 6% | 77% | 100 | −0.11 |
| Jörg Kachelmann | `realkachelmann.bsky.social` | 52,761 | 127 | 90 | 1.62 | 3% | 33% | 64% | 99 | −0.16 |
| Florian Klenk | `klenkflorian.bsky.social` | 45,082 | 123 | 91 | 1.78 | 10% | 34% | 55% | 96 | −0.22 |
| Stefan Niggemeier | `niggi.bsky.social` | 34,516 | 99 | 74 | 1.74 | 8% | 28% | 64% | 75 | +0.01 |
| BÜNDNIS 90/DIE GRÜNEN | `gruene.de` | 40,534 | 59 | 11 | 2.28 | 9% | 82% | 9% | 11 | — |
| taz | `taz.de` | 61,662 | 47 | 17 | 1.76 | 10% | 22% | 68% | 100 | — |
| Übermedien | `uebermedien.de` | 36,527 | 47 | 97 | 1.81 | 8% | 17% | 75% | 100 | −0.09 |
| medico international | `medico2.bsky.social` | 84,249 | 35 | 65 | 2.02 | 22% | 9% | 69% | 68 | −0.48 |
| Mimikama | `mimikama.org` | 34,300 | 29 | 71 | 2.20 | 34% | 19% | 47% | 100 | +0.06 |
| Tagesschau Bot | `tagesschau.bsky.social` | 57,451 | 28 | 16 | 1.66 | 6% | 20% | 74% | 100 | — |

## Read `int%`, not `LoC`

The two columns disagree, and where they disagree `LoC` is the one that misleads.

`gruene.de` has the third-highest mean score in the table (2.28) on 9% internal content
and 82% neutral. `ruthe.de` (2.19) and Margarete Stokowski (2.08) are the same story:
71% and 57% neutral. The Score axis places neutral at 2.0, so an account that posts
small talk and cartoons outscores one that argues for agency a third of the time.
**Mean `ts_score` measures harmlessness. `int%` measures the thing the feed selects for.**

It is the account-level form of something already known about the vector, whose 0–9
prediction never leaves [0.7, 3.3]: an averaged position on an ordinal axis compresses
toward the neutral mass, and the tail is what the feed consumes. Judge accounts the way
CLAUDE.md says to judge the axis — on the top of the ranking, not on its mean.

On `int%` the field separates cleanly: **Pia Lamberty (67%)** and **FragDenStaat (59%)**
are in a class of their own, Mimikama (34%) and Maja Göpel (33%) follow, and no one else
clears 26%. That both leaders sit in the bottom third by engagement — 27th and 8th — is
the same finding as the section above, seen per account.

## The ten with the most effective reach

The published chart does not show all 37. It shows the first ten rows of the table
above — the ten with the highest median interactions on posts aged 2–90 days.

That ranking, not the follower ranking, is the one worth cutting on. A follower count
is a nominal audience: ten thousand dormant accounts following someone is not reach,
and the public API exposes no impressions, so median interactions per post is the
closest observable proxy for who actually sees a post. The two rankings disagree
sharply. Robert Habeck (87,179 followers) and Spiegel (75,121) are top-ten by
followers and appear nowhere here — Habeck did not post in the window at all, and
Spiegel posts fast enough that no post of its had matured. Ricarda Lang at 60,371
followers leads the field.

`ricardalang.eurosky.social`, `thereallorenzmeyer.bsky.social`, `volksverpetzer.de`,
`politicalbeauty.bsky.social`, `christianstoecker.de`, `florianaigner.at`,
`afelia.bsky.social`, `fragdenstaat.de`, `beyond-ideology.bsky.social`,
`nataschastrobl.bsky.social`.

Selection and measurement use the same window here — posts aged 2–90 days — because
selecting accounts on their matured posts and then charting every post would put posts
in the picture that the selection was not allowed to see.

| class | posts | share of posts | interactions | share of interactions | likes | reposts | replies | quotes | mean/post | median/post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external | 334 | 61.1% | **450,169** | **71.9%** | 344,430 | 88,069 | 13,888 | 3,782 | 1,348 | 1,042.5 |
| neutral | 108 | 19.7% | **64,434** | **10.3%** | 54,548 | 7,075 | 2,493 | 318 | 597 | 465 |
| internal | 105 | 19.2% | **111,832** | **17.8%** | 84,128 | 24,097 | 2,738 | 869 | 1,065 | 850 |
| **all** | **547** | 100% | **626,435** | 100% | 483,106 | 119,241 | 19,119 | 4,969 | 1,145 | |

### The selection is not circular

Picking accounts by engagement and then reporting how engagement splits by class invites
an obvious objection: that the selection itself favours accounts which post external.
It does not. Across the 37 rankable accounts, Spearman(median interactions,
`external_share`) = **+0.035** — nothing.

The selection leans mildly the other way. Spearman with `internal_share` is **+0.179**,
and the ten average 21.8% internal against 14.1% for the other 27. The accounts that
reach people effectively write *more* agency, not less. Post-weighting is not doing any
work either: 62.0% post-weighted external share against 61.8% unweighted.

So the external post share moving 55.3% → 61.1% between the two pools is who these
particular ten are, not a property of selecting on reach.

### What differs from the wide pool

**The gap is wider.** External takes 71.9% of interactions from 61.1% of posts, a
+10.8-point spread against +9.9 across all 37.

**Neutral is the loser here, not internal.** In the 37-account pool internal and
neutral sit almost on top of each other (medians 255.5 and 242.5), both far below
external's 336. Among these ten they separate: internal's median is **850** against
neutral's **465**, and internal takes 17.8% of interactions from 19.2% of posts —
close to break-even. What these ten are punished for is filler, not agency.

**The reposts-per-like ordering is sharper.** Internal 0.286, external 0.256, neutral
0.130. The one behaviour that favours agency-promoting posts is stronger among the
accounts with the most reach.

**The within-account relationship survives the cut.** Pooled Spearman
(`ts_score`, interactions) = **−0.305** (n = 547); within-account median **−0.300**,
negative in **7 of 8** accounts that clear the ≥25-post threshold.

Chart: `research/data/de_study/sharepic_de_top_loc_top10_{de,en}.png`, with alt text
beside each. The chart rounds the external median to 1,042.

## One picture per account, ranked by agency

The ten get a picture each: the same two bars — share of posts, share of
interactions — over that account's own matured posts, so any two of them can be
held side by side. `research/de_study/make_account_sharepics.py` draws them from
`by_class_matured` in the report and nothing else.

They are numbered by **internal post share, most agency first**, not by
`loc_mean` and not by reach. The section above gives the reason: `loc_mean`
measures harmlessness, `int%` measures what the feed selects for. Ordering on
the matured window rather than on all scored posts — so that the number ranked
on is the number drawn — changes nothing except a near-tie: Florian Aigner
(11.5%) edges past Zentrum für Politische Schönheit (11.3%).

`int% posts` and `int% inter` are that account's internal share of its matured
posts and of the interactions they drew; `ext% posts` and `ext% inter` the same
for external. The two pairs do not sum to 100 — neutral is the remainder, left
out because it is the class the feed has no opinion about. Reading a row means
comparing a column pair with itself: `ext% inter` above `ext% posts` is the gap
this study is about. `reach` is the account's rank in the selection, 1 = highest
median interactions.

| # | account | handle | int% posts | int% inter | ext% posts | ext% inter | n | median | reach | LoC | r |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | FragDenStaat | `fragdenstaat.de` | **59%** | 58% | 38% | 41% | 37 | 671 | 8 | 2.71 | +0.04 |
| 2 | Maja Göpel | `beyond-ideology.bsky.social` | 33% | 22% | 56% | 73% | 18 | 658 | 9 | 1.73 | — |
| 3 | Marina Weisband | `afelia.bsky.social` | 29% | 10% | 55% | 82% | 31 | 697 | 7 | 1.86 | −0.58 |
| 4 | Der Volksverpetzer | `volksverpetzer.de` | 25% | 26% | 66% | 71% | 79 | 968 | 3 | 1.68 | −0.23 |
| 5 | Lorenz Meyer | `thereallorenzmeyer.bsky.social` | 20% | 21% | 67% | 68% | 51 | 1,856 | 2 | 1.65 | −0.24 |
| 6 | Natascha Strobl | `nataschastrobl.bsky.social` | 15% | 20% | 52% | 64% | 91 | 635 | 10 | 1.71 | −0.31 |
| 7 | Ricarda Lang | `ricardalang.eurosky.social` | 13% | 11% | 67% | 83% | 15 | 2,318 | 1 | 1.41 | — |
| 8 | Florian Aigner | `florianaigner.at` | 11% | 12% | 44% | 62% | 96 | 762 | 6 | 1.78 | −0.41 |
| 9 | Zentrum für Politische Schönheit | `politicalbeauty.bsky.social` | 11% | 12% | 85% | 87% | 53 | 923 | 4 | 1.38 | −0.33 |
| 10 | Christian Stöcker | `christianstoecker.de` | 7% | 4% | 83% | 88% | 76 | 798 | 5 | 1.45 | −0.29 |

**With external in the table, the effect is unanimous.** `ext% inter` exceeds
`ext% posts` in **10 of 10** accounts — from +1 point (Lorenz Meyer, Zentrum für
Politische Schönheit) to +27 (Marina Weisband). It is the one column pair that
points the same way for every account in the set, and it does so even for the
five whose internal share holds up, because what external gains there it takes
from neutral rather than from internal.

**The two rankings pull against each other.** Spearman(internal share, median
interactions) across these ten = **−0.394**. The two accounts that lead on agency
are 8th and 9th on reach; the account with the most reach, Ricarda Lang, is 7th
on agency. Nothing here says agency costs reach — ten accounts cannot carry that
claim, and the within-account correlations in the previous section are the
evidence that does — but the ordering is the one the pooled figure predicts.

**The penalty is not uniform per account.** Five of the ten take at least as
large a share of interactions with their internal posts as those posts are of
their output, and Natascha Strobl takes 20% from 15%. The group-level shortfall
is carried by two accounts: Marina Weisband (29% of posts, 10% of interactions)
and Maja Göpel (33% → 22%). This is the same fact as internal sitting near
break-even group-wide (19.2% of posts, 17.8% of interactions), seen per account,
and it is why the per-account pictures are worth having: the corpus bar is an
average over ten accounts that do not behave alike.

**`loc_mean` and `int%` agree better here than across all 37** — Spearman
**+0.709** among these ten, against the disagreement documented above. These ten
post little neutral filler (19.7% against 29.1% across the 37), and neutral
filler is exactly what pushes the two apart.

**What the pictures call the classes.** `internal` and `external` are the
rubric's names and stay that way in this report and in the data. The published
charts say **selbstbestimmt / neutral / fremdbestimmt** (`self-determined /
neutral / externally framed`), under a line naming the basis: *eingestuft
danach, wie ein Post auf Lesende wirkt – nicht danach, ob er recht hat.* An
earlier pass labelled the poles *handlungsfähig / ohnmächtig*, and that was
wrong twice over: printed beside an account's name it reads as a verdict on the
account rather than on the post's effect, and it overstates the mild external
band, which the rubric fills with things like "reports of threatening events
with no solution offered" — an ordinary news post, not helplessness. The chart
labels must be as strong as the rubric and no stronger.

Pictures: `research/data/de_study/account_sharepics_top10_by_median_interactions_{de,en}/`,
alt text beside each. `00_ranking.png` is all ten in one frame, one bar per
account in this order; `01_…png` to `10_…png` are the accounts, numbered by it.
The directory names the selection because the next section makes a second ten.

## The other ten: selected on the axis instead of on reach

`DE_TOP_BY=internal_share DE_TOP_LIMIT=10 python3 report_de_top_loc.py` picks
the ten with the largest internal share instead of the ten with the most reach,
among the 37 with enough matured posts to rank at all. It answers the question
the reach selection cannot: who actually writes agency, whether or not anyone
is listening.

Five accounts are in both tens — FragDenStaat, Maja Göpel, Marina Weisband, Der
Volksverpetzer, Lorenz Meyer. The other five are invisible to a reach cut. Pia
Lamberty leads the whole field at 67% internal and is **27th of 37** by median
interactions; Mimikama is 3rd on the axis and 36th by reach.

| # | account | handle | int% posts | int% inter | ext% posts | ext% inter | n | median | LoC | r |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Pia Lamberty | `pialamberty.bsky.social` | **67%** | 59% | 25% | 40% | 52 | 146 | 2.72 | +0.05 |
| 2 | FragDenStaat | `fragdenstaat.de` | 59% | 58% | 38% | 41% | 37 | 671 | 2.71 | +0.04 |
| 3 | Mimikama | `mimikama.org` | 39% | 42% | 45% | 43% | 71 | 29 | 2.20 | +0.06 |
| 4 | Maja Göpel | `beyond-ideology.bsky.social` | 33% | 22% | 56% | 73% | 18 | 658 | 1.73 | — |
| 5 | Marina Weisband | `afelia.bsky.social` | 29% | 10% | 55% | 82% | 31 | 697 | 1.86 | −0.58 |
| 6 | Der Volksverpetzer | `volksverpetzer.de` | 25% | 26% | 66% | 71% | 79 | 968 | 1.68 | −0.23 |
| 7 | medico international | `medico2.bsky.social` | 23% | 15% | 71% | 82% | 65 | 35 | 2.02 | −0.48 |
| 8 | CORRECTIV | `correctiv.org` | 22% | 14% | 68% | 78% | 95 | 148 | 1.97 | −0.44 |
| 9 | Ruprecht Polenz | `polenz.bsky.social` | 21% | 15% | 59% | 70% | 70 | 190 | 1.73 | −0.26 |
| 10 | Lorenz Meyer | `thereallorenzmeyer.bsky.social` | 20% | 21% | 67% | 68% | 51 | 1,856 | 1.65 | −0.24 |

**External gains in nine of these ten too.** `ext% inter` above `ext% posts`
holds everywhere except Mimikama (43% of interactions from 45% of posts), the
single account in either table where external under-earns. Pia Lamberty's +15
points is the widest gap in a set selected for agency.

**They are a third of the size.** Median of the ten medians: **424** interactions
per post, against **780** for the reach ten. Selecting on agency selects, as a
side effect, for smaller accounts — which is the −0.394 from the previous
section seen from the other end.

**The gap between post share and interaction share survives the selection**, and
that is the point worth taking from this table. Over these 569 matured posts,
external takes **67.2%** of interactions from **56.9%** of posts; internal takes
**26.2%** from **31.8%**. These are the ten accounts in German Bluesky that write
the most agency, they write nearly a third of it, and it still under-earns.
Pooled Spearman(`ts_score`, interactions) = **−0.253** (n = 569), within-account
median **−0.238**, negative in **6 of 9**.

**The three axis leaders are the three exceptions to the within-account
correlation.** Lamberty (+0.05), FragDenStaat (+0.04) and Mimikama (+0.06) are
the only positive `r` values in the set. On shares they are merely less
punished, not rewarded: Mimikama takes 42% of its interactions with 39% of its
posts, FragDenStaat breaks even (58% from 59%), and Lamberty still gives up
eight points (59% from 67%). Three accounts are not evidence for a mechanism;
they are the place to look for one.

Pictures: `research/data/de_study/account_sharepics_top10_by_internal_share_{de,en}/`,
plus the corpus bar for this selection in
`sharepic_de_top_loc_top10_by_internal_share_{de,en}.png`. Raw numbers:
`REPORT_de_top_loc_top10_by_internal_share.json`.

## What this does not establish

**The scorer is a model, not ground truth.** These are `jev-latest` readings of a rubric,
and `REPORT_typesafe_loc.md` measures its agreement against a 500-post Gemini gold set,
not against human labels. Class shares are its class shares.

**Engagement is a single snapshot** read at collection time on 2026-09-18. The matured
table mitigates the age problem, it does not remove it.

**Three of the ten pictures rest on a handful of posts.** Ricarda Lang's bars
are 15 posts and two internal ones, Maja Göpel's 18 and six, Marina Weisband's
31 and nine. A single viral post moves those interaction bars by ten points.
Read positions 2, 3 and 7 as indicative; 4, 5, 6, 8, 9 and 10 rest on 51 to 96
posts each. In the axis ten, Maja Göpel (18) and Marina Weisband (31) are the
thin ones.

**The ten are ten.** Section 5 rests on 547 posts and eight accounts with enough
matured posts to carry a within-account correlation. Ricarda Lang contributes 15 posts
and Maja Göpel 18, so the shares there move more easily than the 37-account ones.

**Six accounts are thin.** Ricarda Lang (15 matured posts), gruene.de (11),
`tagesschau` (16), taz (17), Maja Göpel (18) and Stephan Anpalagan (18) carry medians
that a handful of posts could move. They are reported, not ranked, and carry no `r`.

**The 37 accounts are not German Bluesky.** They are the top 50 by followers, minus the
13 that could not supply ≥10 matured posts in the window — a set skewed hard toward
politics, journalism and media criticism. The 65.8% external share is a statement about
what the largest accounts publish, not about the language as a whole. The comparable
census figure lives in `REPORT_de_engagement_findings.md`.

**Correlation, not effect.** Nothing here randomises what an account posts. A −0.25
within-account rank correlation is consistent with an audience that rewards outrage and
equally with authors writing their internal posts on quieter days.
