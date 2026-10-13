"""Identity fusion as TypeSafe System One questions.

Identity fusion (Swann et al. 2009, 2012) is a visceral sense of oneness with a
group in which the personal and the social self stay salient and the border
between them turns porous. It is not group identification: in identification the
group eclipses the personal self and the tie is to the category; in fusion the
personal self stays agentic, other members feel like family, the group's fate is
felt as one's own, and the bond is felt as irrevocable. Fused people take
personal responsibility for the group's outcomes, and fusion -- not
identification -- predicts extreme self-sacrifice for the group.

Text markers come from the fusion-based violence-risk work (Ebner, Kavanagh &
Whitehouse 2024) and CLIFS (Wright, An & Ahn, EMNLP 2025): kinship terms for
fellow members, self described in group terms or the group in self terms,
group events narrated as personal events, pledges of personal sacrifice.

On a random sample of posts strong fusion is vanishingly rare, so the scale is
built to resolve the weak end: a passing 'we', a membership label, shared
activity, pride, defending the group, before anything resembling the fusion
items. The Score answer is a probability-weighted mean, but jev puts nearly all
mass on one level per post (confidence ~1.0 on a graded probe set), so the
continuous part is the monthly mean over many posts, and movement among levels
1-5 is what it will mostly measure -- that is intended.

  fusion_score   0-9 graded strength of fusion expressed by the writer.
  fusion_target  what the writer is fused / identified with, so a trend can be
                 attributed (fandom vs. nation vs. political camp ...).
  fusion_we      a single Noul: does the writer speak as part of a collective.
  outgroup       0-8 graded hostility toward another group. Kept out of the
                 fusion scale (one dimension per question, per the jev docs):
                 the violence-risk literature treats fusion and outgroup
                 threat/hostility as separate factors whose combination is
                 the dangerous case, so both are needed to see it.
"""

# Bumped whenever a rubric below changes; score_time_sample.py re-asks the
# fusion questions (never LoC) for ledger rows answered under another version.
VERSION = "fusion-v2"

FUSION_INSTRUCTIONS = (
    "Rate how strongly the WRITER of this social media post expresses identity "
    "fusion with a group: a felt oneness in which the writer's personal self and a "
    "group (a nation, people, political camp or movement, party, leader, religion, "
    "team or fandom, profession, local or online community, minority or identity "
    "group, family or close circle) merge, so that the group's fate is felt as the "
    "writer's own. Judge only what the writer says about THEIR OWN bond. Merely "
    "talking ABOUT a group, reporting news about it, or criticising one is not "
    "fusion. Hostility toward other groups is rated separately; rate only the "
    "writer's bond with their own group. Posts may be in any language; judge them on the same scale "
    "without translating."
)

# Ten levels, the API maximum. The docs: each level is judged alone, without its
# neighbours or its number, so every description is a self-contained situation.
# Seven of the ten sit below the fusion items proper, because that is where a
# random sample of posts lives.
FUSION_LEVELS = [
    "The post is about the writer's own life, objects, information, jokes, or "
    "other people, with no group the writer counts themselves part of. Talking "
    "about or criticising a group from the outside belongs here.",

    "A collective 'we', 'us' or 'our' appears only in passing and factually - "
    "'our city got snow', 'we have an election Sunday' - with no feeling attached "
    "to the group.",

    "The writer states a group membership as a plain fact or credential - 'as a "
    "nurse', 'us Linux users', 'we Germans tend to' - without warmth or pride.",

    "The writer enjoys a shared group activity or in-group joke, greets or "
    "addresses fellow members ('morning, fellow birders'), or mildly sides with a "
    "group they belong to.",

    "The writer voices pride, warmth, loyalty or gratitude toward their group - "
    "'love this community', 'my people', 'proud to be one of them', cheering "
    "'we won!' for their team.",

    "The writer stands up for their own group when it is criticised, wronged or "
    "threatened - protective of 'us', feeling the group's treatment as unfair.",

    "Fellow members are called family, brothers or sisters, or the group's wins "
    "and injuries are told as the writer's own personal joy or pain.",

    "The writer defines who they are through the group - 'this is who I am', "
    "'it runs in my blood', 'I'll always be one of them', a bond declared "
    "permanent or unbreakable.",

    "The writer takes personal responsibility for the group's fate or pledges "
    "personal effort, risk or sacrifice for it - 'I'll do whatever it takes for "
    "us', 'I would fight for them'.",

    "The writer states oneness with the group or its leader - 'I am the "
    "movement', 'my country is me' - or readiness to die, kill or use violence "
    "for it.",
]

TARGET_CRITERIA = {
    "none": "The writer expresses no bond with any group they belong to.",
    "nation_ethnic": "A nation, country, region, people, ethnicity or language community.",
    "political": "A political camp, party, ideology, movement, cause or activist community, "
                 "including 'the left', 'the resistance', MAGA, antifascists, climate movement.",
    "leader": "A specific leader or public figure the writer identifies with personally.",
    "religion": "A religion, church, faith community or spiritual group.",
    "fandom_sports": "A sports team, club, fandom, band, game, artist or hobby community.",
    "work_profession": "A profession, employer, workplace, guild, discipline or field.",
    "identity_group": "A gender, sexuality, disability, age cohort, class, or minority "
                      "identity group (e.g. queer community, trans people, women, Black people).",
    "local_online": "A local community, city, neighbourhood, school, or an online community "
                    "or platform (e.g. 'Bluesky', 'this site', 'the fediverse').",
    "family_friends": "The writer's family, partner, friends or close personal circle.",
}

WE_INSTRUCTIONS = (
    "The writer speaks as a member of a collective they belong to - a 'we', 'us' "
    "or 'our people' that refers to a group, not just to the writer and one other "
    "person."
)


OUTGROUP_INSTRUCTIONS = (
    "Rate how much hostility the WRITER of this social media post expresses toward "
    "a group of people: a political camp or party and its supporters, a nation, "
    "ethnicity, religion, gender, sexuality, class, profession, generation, fandom, "
    "or any 'them'. A single person counts only when attacked as a representative of "
    "such a group. Criticism of an institution's decision, or of one politician's "
    "act, is not hostility toward a group. Posts may be in any language; judge them "
    "on the same scale without translating."
)

# Each level self-contained, judged alone (jev docs). Ordered by the ladder of
# the out-group animosity and dehumanisation literature: disagreement, contempt,
# blame, threat, dehumanisation, exclusion, violence.
OUTGROUP_LEVELS = [
    "No group of people is set against: the post is about the writer's life, "
    "things, or information, or mentions groups neutrally or approvingly.",

    "The writer disagrees with a group's position or behaviour in a civil, "
    "factual way, without belittling its members.",

    "The writer mocks, ridicules or rolls their eyes at a group - sarcasm, "
    "'typical X', jokes at the group's expense.",

    "The writer blames a group for problems or ascribes bad motives to it - "
    "'they only want to destroy', 'X are ruining this country'.",

    "The writer insults a group or its members with contemptuous labels - "
    "idiots, morons, Schwurbler, Covidioten, libtards.",

    "The writer portrays a group as a dangerous enemy or existential threat to "
    "us - 'they are coming for us', 'they will replace us', a war of us against them.",

    "The writer dehumanises a group - vermin, parasites, rats, animals, scum, "
    "disease, filth, subhuman.",

    "The writer calls for a group to be excluded, expelled, silenced, punished "
    "or stripped of rights - 'deport them all', 'ban them', 'lock them up'.",

    "The writer endorses, celebrates or calls for violence or death against a "
    "group or its members.",
]


def questions():
    return {
        "fusion_score": {
            "type": "score",
            "instructions": FUSION_INSTRUCTIONS,
            "criteria": FUSION_LEVELS,
        },
        "fusion_target": {
            "type": "choice",
            "instructions": (
                "Which group, if any, does the writer of this post express a personal "
                "bond with or count themselves part of? Choose the dominant one."
            ),
            "criteria": TARGET_CRITERIA,
        },
        "fusion_we": {
            "type": "noul",
            "instructions": WE_INSTRUCTIONS,
        },
        "outgroup": {
            "type": "score",
            "instructions": OUTGROUP_INSTRUCTIONS,
            "criteria": OUTGROUP_LEVELS,
        },
    }
