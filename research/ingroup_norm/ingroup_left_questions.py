"""The in-group norm questions with the in-group fixed: the political left.

ingroup_questions.py leaves 'own side' to jev, and jev cannot see it: a single
post rarely says which camp its writer is in, so `critic_target = own_side`
fired on outsiders attacking a party as often as on genuine self-correction
(REPORT_ingroup_norm_jev.md). The feed's readers are known, though -- mostly
left, anti-Trump, anti-AfD -- and You & Lee's mechanism runs through the
reader's perception of their OWN group's norm. So this variant names the sides
outright and asks every question from that reader's position: what does this
post tell a left reader that the left feels and does about the right?

Same levels as ingroup_questions.py (their 'own side' / 'other side' wording now
has a fixed referent), same questions, fields prefixed `left_` so both rubrics
can sit in one ledger row and be compared.
"""

import ingroup_questions as base

VERSION = "ingroup-left-v1"
VERSION_FIELD = "ingroup_left_version"
PREFIX = "left_"

SIDES = (
    "Context: this post is shown to readers on the political LEFT - progressive, "
    "anti-Trump, anti-AfD. Throughout, 'OWN SIDE' always means the left: progressives, "
    "liberals, Democrats, Greens, SPD, Linke, antifascists, left activists, and the "
    "progressive majority on Bluesky. 'OTHER SIDE' always means the right: Trump, MAGA, "
    "Republicans, AfD, conservatives and the Union when opposed from the left, right-wing "
    "media, and their voters and supporters. Apply these meanings whoever wrote the post. "
)

NORM_INSTRUCTIONS = SIDES + (
    "Rate the norm toward the right that this post models for a left reader: what it "
    "tells the left it should feel or do about the right and the people who vote for "
    "it. Low teaches the left hostility toward the right; high teaches the left "
    "restraint, fairness or goodwill toward people on the right. Criticising a policy, "
    "an institution or one politician's act is not hostility toward a group of people. "
    "Condemning the right's hatred, attacks, lies or lack of empathy, or showing "
    "solidarity with the left's victims, rallies the left against the right and is "
    "not restraint, however justified." + base.LANG
)

CORRECTION_INSTRUCTIONS = SIDES + (
    "Rate how strongly this post pushes back on hostility, contempt, generalisation, "
    "exaggeration or polarising language coming FROM THE LEFT toward the right or "
    "toward any other group. Criticising the right - its hostility, lies or attacks - "
    "is not this, however strongly put; only correcting the left counts." + base.LANG
)

HUMANIZE_INSTRUCTIONS = SIDES + (
    "Rate how far this post presents people on the right - Trump or AfD voters, "
    "conservatives, Republicans, rural or religious conservatives, people the left "
    "usually opposes - as reasonable, good-faith, individual people. Neutral news "
    "about them is not this; nor is praise of people on the left." + base.LANG
)

NORM_CORRECT_INSTRUCTIONS = SIDES + (
    "Rate how far this post corrects an EXAGGERATED belief a left reader may hold about "
    "what a group thinks, feels or wants - how extreme, hostile, unanimous or dangerous "
    "the right's voters are, or how united or hostile the left itself is - with "
    "evidence, experience or argument. Posts that spread such beliefs rate low." + base.LANG
)

VIOLENCE_INSTRUCTIONS = SIDES + (
    "Rate the norm toward political violence, threats, harassment or intimidation that "
    "this post models for a left reader. Low endorses or excuses it; the middle has "
    "nothing to say about it; high rejects it - and highest when it rejects violence, "
    "threats or harassment by the LEFT or aimed at the right." + base.LANG
)

SPEAKER_INSTRUCTIONS = (
    "The writer of this post speaks as part of the political left - progressive, "
    "anti-Trump or anti-AfD - addressing or speaking for that side."
)

CRITIC_CRITERIA = {
    "none": "The post criticises no camp's hostility, contempt, exaggeration or attacks.",
    "left": "The post criticises hostility, contempt, generalisation, exaggeration or "
            "attacks coming from the LEFT - progressives, Democrats, Greens, SPD, Linke, "
            "activists, or left Bluesky users.",
    "right": "The post criticises hostility, contempt, lies or attacks coming from the "
             "RIGHT - Trump, MAGA, Republicans, AfD, conservatives, right-wing media.",
    "general": "The post criticises hostility or violence in general, from whichever side.",
    "other": "The post criticises hostility from some other camp - a government abroad, a "
             "religious or national group, a fandom - not the left-right divide.",
}


def questions():
    return {
        "ig_norm": {"type": "score", "instructions": NORM_INSTRUCTIONS,
                    "criteria": base.NORM_LEVELS},
        "ig_correction": {"type": "score", "instructions": CORRECTION_INSTRUCTIONS,
                          "criteria": base.CORRECTION_LEVELS},
        "og_humanize": {"type": "score", "instructions": HUMANIZE_INSTRUCTIONS,
                        "criteria": base.HUMANIZE_LEVELS},
        "norm_correct": {"type": "score", "instructions": NORM_CORRECT_INSTRUCTIONS,
                         "criteria": base.NORM_CORRECT_LEVELS},
        "violence_norm": {"type": "score", "instructions": VIOLENCE_INSTRUCTIONS,
                          "criteria": base.VIOLENCE_LEVELS},
        "ig_speaker": {"type": "noul", "instructions": SPEAKER_INSTRUCTIONS},
        "critic_target": {
            "type": "choice",
            "instructions": SIDES + "Whose hostility, contempt, exaggeration or attacks "
                                    "does this post criticise?",
            "criteria": CRITIC_CRITERIA,
        },
    }


SCORES = ("ig_norm", "ig_correction", "og_humanize", "norm_correct", "violence_norm")


def fields(a):
    """`divide` and `outgroup` are side-independent, so they are not re-asked; the
    v2 answers in the same ledger row serve."""
    out = {VERSION_FIELD: VERSION}
    for k in SCORES:
        out[PREFIX + k] = a[k]["score"]
        out[PREFIX + k + "_probs"] = a[k]["probabilities"]
        out[PREFIX + k + "_conf"] = a[k]["confidence"]
    out[PREFIX + "ig_speaker"] = a["ig_speaker"]["noul"]
    out[PREFIX + "critic_target"] = a["critic_target"]["choice"]
    out[PREFIX + "critic_target_probs"] = a["critic_target"]["probabilities"]
    return out
