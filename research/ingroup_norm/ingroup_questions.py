"""In-group norm signalling toward the out-group, as TypeSafe System One questions.

You & Lee (2024, PNAS Nexus, "Explanations of and interventions against affective
polarization cannot afford to ignore the power of ingroup norm perception"):
partisans' perception of their OWN group's norm of negativity toward the
out-group is exaggerated, and it predicts their affective polarization 8.4x
better than outgroup meta-perception does; for support of partisan violence it
explains 52% of the variance against 0%. Correcting the ingroup norm perception
reduced polarization in their pilot. Blair et al. (2025) find the same from the
other side: a workshop teaching people to challenge in-group members who use
polarizing language lowered animosity, cross-partisan contact did not.

So the unit that matters in a post is not hostility as such but the NORM it
signals to the writer's own side: what 'people like us' feel and do about
'them'. A feed could surface the rare posts in which an in-group voice models
restraint, corrects the in-group, or humanises the other side.

On a candidate pool these are rare and most intergroup posts point the other way,
so every scale is built to resolve the faint end: a concession, a 'to be fair',
a 'not all of them', a named individual on the other side treated decently,
before anything resembling an explicit call to the in-group.

Every Score has ten levels, the API maximum; each level is judged alone, without
its neighbours or its number (jev docs), so each is a self-contained situation.
The answer is a probability-weighted mean over levels 0-9; add 1 for a 1-10 scale.

  ig_norm        bipolar composite: the norm toward the out-group that the post
                 models for the writer's own side, from rallying to violence (1)
                 through no intergroup content (5) to leading the in-group away
                 from hostility (10).
  ig_correction  how strongly the writer pushes back on hostility, generalising
                 or polarising language FROM THEIR OWN SIDE (Blair et al.).
  og_humanize    how far the post presents out-group members as reasonable,
                 good-faith, individual people with shared concerns.
  norm_correct   how far the post corrects an exaggerated belief about what a
                 group thinks -- the You & Lee intervention itself.
  violence_norm  bipolar: the norm toward political / intergroup violence the
                 post models, their second outcome.
  ig_speaker     Noul: the writer speaks as a member of one side of the divide
                 at stake -- the precondition for any of this to be an IN-group
                 signal.
  divide         Choice: which intergroup divide the post is about.
  critic_target  Choice: whose hostility the post criticises -- own side, other
                 side, both/general, none. Separates in-group correction (the
                 signal) from condemning the opponent (an in-group rally).
  outgroup       the 0-8 hostility ladder of time_sample/fusion_questions.py,
                 reused verbatim so the two studies are comparable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "time_sample"))

import fusion_questions  # noqa: E402

# v2: v1 scored condemning the OTHER side's hostility as an in-group correction
# (7 of the top 16 ig_norm posts); that is an in-group rally, the reverse of the
# signal. Both rubrics now exclude it explicitly, and `critic_target` asks the
# question directly so it can be filtered on.
VERSION = "ingroup-v2"
VERSION_FIELD = "ingroup_version"

LANG = (" Posts may be in any language, most often German or English; judge them on "
        "the same scale without translating.")

NORM_INSTRUCTIONS = (
    "Rate the norm toward an opposing group that this social media post models for "
    "the writer's OWN side: what it tells people like the writer they should feel or "
    "do about 'them' - a rival political camp or party and its supporters, another "
    "nation, ethnicity, religion, generation, class or any 'them'. Low means the post "
    "teaches its own side hostility; high means it teaches its own side restraint, "
    "fairness or goodwill toward the other side. Criticising a policy, an "
    "institution or one politician's act is not hostility toward a group of "
    "people. Condemning the OTHER side's hatred, attacks or lack of empathy, or "
    "showing solidarity with one's own side's victims, rallies one's own side "
    "against them and is not restraint." + LANG
)

NORM_LEVELS = [
    "The writer urges or celebrates violence, punishment or the removal of rights "
    "against the other group, rallying their own side to act against 'them'.",

    "The writer rallies their own side against the other group with dehumanising "
    "or contemptuous language - vermin, scum, idiots, traitors - or paints them as "
    "an enemy to be defeated.",

    "The writer mocks or sneers at the other group for an audience of their own "
    "side, or blames 'them' as a whole for problems - 'typical X', 'they only "
    "want to destroy'.",

    "The writer frames a debate as us against them and sides firmly with their "
    "own group, criticising the other group's members civilly but with no "
    "concession to them.",

    "The post has no intergroup content, or mentions groups only neutrally: "
    "personal life, information, news, jokes, or criticism of a policy or one "
    "person's act without turning it against a group of people.",

    "Within a partisan or critical post, the writer makes a faint concession to the "
    "other side - 'to be fair', 'they have a point there', 'I get why people "
    "worry about this' - or criticises leaders while sparing their ordinary "
    "supporters.",

    "The writer treats a member of the other group decently or with respect - "
    "thanks, praises or agrees with someone from the other side, or names common "
    "ground or a value both sides share.",

    "The writer gently questions a generalisation their own side makes - 'not all "
    "of them', 'most of them are just worried', 'we shouldn't stoop to that' - "
    "or asks their side to listen to the other.",

    "The writer tells their own side that hostility toward the other group is not "
    "who 'we' are, or corrects an exaggerated belief about how much their side, or "
    "the other side, hates or threatens the other.",

    "The writer explicitly calls on their own side to reject contempt or violence "
    "against the other group, condemns an attack or slur committed by their own "
    "side, or argues for reconciliation and working together.",
]

CORRECTION_INSTRUCTIONS = (
    "Rate how strongly the WRITER of this social media post pushes back on hostility, "
    "contempt, generalisation or polarising language coming from THEIR OWN side - "
    "their own political camp, party, nation, community, fandom or other group - "
    "toward another group. Disagreeing with an opponent is not this, and neither is "
    "condemning the hostility, lies or attacks of the OTHER side, however "
    "strongly; only correcting one's own allies, fellow members or own camp "
    "counts." + LANG
)

CORRECTION_LEVELS = [
    "The post does not concern the writer's own side's attitude toward another "
    "group at all.",

    "The writer joins in or amplifies their own side's hostility, mockery or "
    "generalisation about another group.",

    "The writer is on their own side's line about another group without adding or "
    "questioning anything - reports, cheers or shares it.",

    "The writer expresses slight discomfort with how their own side talks about "
    "another group - a sigh, an 'eh', a 'this isn't helpful' - without elaborating.",

    "The writer adds a qualifier to their own side's claim about another group - "
    "'well, not all of them', 'it's more complicated', 'that's a bit unfair'.",

    "The writer reminds their own side of fairness or accuracy toward another group "
    "in general terms - 'let's stick to the facts', 'we should argue against "
    "positions, not people'.",

    "The writer disagrees openly with a specific statement, slur or joke from "
    "someone on their own side aimed at another group.",

    "The writer explains to their own side why their hostility toward another "
    "group is wrong or counter-productive, giving reasons.",

    "The writer calls out their own side's leaders, media or prominent voices for "
    "stoking hostility against another group.",

    "The writer condemns violence, threats or dehumanisation committed by their "
    "own side against another group, or asks their own side to change course.",
]

HUMANIZE_INSTRUCTIONS = (
    "Rate how far this social media post presents members of a group the writer is "
    "NOT part of - especially an opposing political camp, party, nation, religion or "
    "other rival group - as reasonable, good-faith, individual people. Neutral news "
    "about a group is not this; nor is praise of a group the writer belongs to." + LANG
)

HUMANIZE_LEVELS = [
    "The post presents another group's members as less than human, evil or "
    "inherently dangerous.",

    "The post presents another group's members as stupid, malicious, or all the "
    "same - a faceless bloc.",

    "The post mentions another group critically or as a bloc, without "
    "contempt but without any individual face or good motive.",

    "The post does not portray any other group's members, or mentions one only "
    "neutrally.",

    "The post concedes that some members of the other group mean well, or that "
    "their concerns are understandable, even while disagreeing.",

    "The post shows an individual from the other group as an ordinary person - a "
    "name, a story, a daily life, a feeling - rather than as a representative.",

    "The post points to something both groups value, fear or want, or to a "
    "situation they share.",

    "The post describes a positive personal encounter, friendship, family tie or "
    "cooperation with someone from the other group.",

    "The post explains the other group's view from the inside, in terms its "
    "members would recognise, so that it makes sense to the writer's side.",

    "The post celebrates people from opposing groups working, grieving or "
    "succeeding together, presenting the divide as smaller than it looks.",
]

NORM_CORRECT_INSTRUCTIONS = (
    "Rate how far this social media post corrects an EXAGGERATED belief about what a "
    "group of people thinks, feels or wants - how extreme, hostile, unanimous or "
    "dangerous the writer's own side or another side is - with evidence, experience "
    "or argument. Posts that spread such beliefs rate low." + LANG
)

NORM_CORRECT_LEVELS = [
    "The post spreads an exaggerated picture of a group - 'all of them want X', "
    "'they hate us', 'everyone on our side agrees we must fight them'.",

    "The post implies, without stating it, that a whole group is extreme, hostile "
    "or unanimous.",

    "The post says nothing about what a group of people thinks, feels or wants.",

    "The post reports a group's opinions neutrally - a poll result, a statement - "
    "with no claim that it is more or less extreme than people believe.",

    "The post hints that a group is more varied than people assume - 'many of them "
    "don't actually think that'.",

    "The post shares a personal experience that contradicts a stereotype of what a "
    "group thinks or wants.",

    "The post says that most people on the writer's own side or the other side are "
    "more moderate, less hostile or more divided than the loud voices suggest.",

    "The post cites a survey, study or data showing that a group's views are less "
    "extreme or hostile than commonly believed.",

    "The post explicitly tells readers that they overestimate how much the other "
    "side, or their own side, hates or opposes the other.",

    "The post makes correcting such misperceptions its main point, with evidence, "
    "and draws the conclusion that the sides are less divided than they think.",
]

VIOLENCE_INSTRUCTIONS = (
    "Rate the norm toward violence, threats, harassment or intimidation between "
    "groups - political, ethnic, national, religious or other - that this social "
    "media post models for its readers. Low endorses such violence; the middle has "
    "nothing to say about it; high rejects it, especially when committed by the "
    "writer's own side." + LANG
)

VIOLENCE_LEVELS = [
    "The post calls for, threatens, or celebrates violence against members of a "
    "group.",

    "The post excuses or justifies violence, threats or harassment against a group "
    "- 'they had it coming', 'what did they expect'.",

    "The post uses violent or martial imagery about a group in a way that "
    "normalises it - 'fight', 'war', 'crush them' - without literally calling for "
    "violence.",

    "The post reports violence or threats between groups neutrally, without "
    "taking a stance on them.",

    "The post has nothing to do with violence, threats or intimidation between "
    "groups.",

    "The post expresses sadness, shock or concern about violence or threats between "
    "groups.",

    "The post condemns violence, threats or harassment committed by the other side "
    "against the writer's side.",

    "The post condemns intergroup violence in general terms, whoever commits it.",

    "The post condemns violence, threats or harassment committed by the writer's "
    "own side, or by people the writer might be expected to support.",

    "The post actively asks the writer's own side to stay peaceful, de-escalate, "
    "or stand up for the safety of their opponents.",
]

SPEAKER_INSTRUCTIONS = (
    "The writer speaks as a member of one side of the group divide the post is about "
    "- a political camp, party, nation, religion or other group - addressing or "
    "speaking for that side, rather than as an outside observer or neutral reporter."
)

DIVIDE_CRITERIA = {
    "none": "The post is not about any divide between groups of people.",
    "left_right": "Political camps, parties or ideologies - left vs right, Democrats vs "
                  "Republicans, Greens vs AfD, progressives vs conservatives, MAGA.",
    "migration_ethnic": "Migrants, refugees, ethnic or racial groups, natives vs newcomers.",
    "national": "Nations, states or peoples - Israel/Palestine, Russia/Ukraine, East/West.",
    "religion": "Religious groups - Muslims, Christians, Jews, atheists.",
    "gender_sexuality": "Gender or sexuality - men vs women, trans people, queer people.",
    "generation_class": "Generations, classes, urban vs rural, rich vs poor, elites vs people.",
    "other": "Any other group divide - fandoms, professions, platforms, online communities.",
}


CRITIC_CRITERIA = {
    "none": "The post criticises no group's or camp's hostility, contempt or attacks.",
    "own_side": "The writer criticises hostility, contempt, generalisation or attacks "
                "coming from their OWN side, camp, community or fellow members.",
    "other_side": "The writer criticises hostility, contempt, lies or attacks coming from "
                  "an opposing side, camp, party, government or group.",
    "general": "The writer criticises hostility or violence in general, from whichever side, "
               "without pointing at one camp.",
}


def questions():
    return {
        "ig_norm": {"type": "score", "instructions": NORM_INSTRUCTIONS, "criteria": NORM_LEVELS},
        "ig_correction": {"type": "score", "instructions": CORRECTION_INSTRUCTIONS,
                          "criteria": CORRECTION_LEVELS},
        "og_humanize": {"type": "score", "instructions": HUMANIZE_INSTRUCTIONS,
                        "criteria": HUMANIZE_LEVELS},
        "norm_correct": {"type": "score", "instructions": NORM_CORRECT_INSTRUCTIONS,
                         "criteria": NORM_CORRECT_LEVELS},
        "violence_norm": {"type": "score", "instructions": VIOLENCE_INSTRUCTIONS,
                          "criteria": VIOLENCE_LEVELS},
        "ig_speaker": {"type": "noul", "instructions": SPEAKER_INSTRUCTIONS},
        "divide": {
            "type": "choice",
            "instructions": "Which divide between groups of people is this post mainly about?",
            "criteria": DIVIDE_CRITERIA,
        },
        "critic_target": {
            "type": "choice",
            "instructions": ("Whose hostility, contempt, polarising language or attacks "
                             "toward another group does the writer of this post criticise?"),
            "criteria": CRITIC_CRITERIA,
        },
        "outgroup": {
            "type": "score",
            "instructions": fusion_questions.OUTGROUP_INSTRUCTIONS,
            "criteria": fusion_questions.OUTGROUP_LEVELS,
        },
    }


SCORES = ("ig_norm", "ig_correction", "og_humanize", "norm_correct", "violence_norm", "outgroup")


def fields(a):
    out = {VERSION_FIELD: VERSION}
    for k in SCORES:
        out[k] = a[k]["score"]
        out[k + "_probs"] = a[k]["probabilities"]
        out[k + "_conf"] = a[k]["confidence"]
    out["ig_speaker"] = a["ig_speaker"]["noul"]
    out["divide"] = a["divide"]["choice"]
    out["divide_probs"] = a["divide"]["probabilities"]
    out["critic_target"] = a["critic_target"]["choice"]
    out["critic_target_probs"] = a["critic_target"]["probabilities"]
    return out
