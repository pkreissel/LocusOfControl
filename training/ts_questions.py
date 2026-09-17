"""The LoC rubric of label_prompts.py, expressed as TypeSafe System One questions.

Three scorers share one request. TypeSafe bills the state once per call and
evaluates each question in isolation against it, so asking all three costs
barely more than asking the best one -- and which one is best is the thing
under test.

  loc_score   the v4 0-9 graded axis, folded onto 5 ordered levels (the Score
              primitive caps at 10 and the v4 bands are already 5 wide).
              Answer is a probability-weighted mean in [0,4] -> a ranker.
  loc_class   the v3 three-way Label as a Choice. P(internal) is the direct
              analogue of the deployed vector's sigmoid output.
  agency      a single Noul, the cheapest possible probe, as a floor.
"""

from label_prompts import DEFINITIONS_V3

# v4's 0-9 bands, one level per band pair. Descriptions are situations, not
# degrees, per the Score docs -- the abstract band names ("mild external") are
# kept only as a lead-in to the concrete cues the teacher prompt spells out.
LOC_LEVELS = [
    "Strong external: pronounced helplessness, fatalism, victimhood, or anxiety "
    "about uncontrollable events, with no path to act. The writer or reader is a "
    "passive victim of corporations, governments, or macro-events; outcomes are "
    "credited to fixed causes - innate talent, luck - or to fatalistic "
    "self-condemnation.",

    "Mild external: leans disempowering. Demands that politicians, institutions, "
    "or other remote powers finally act; worry about events the reader cannot "
    "touch; reports of threatening, dangerous, or tragic events with no solution "
    "offered; sarcasm that mocks individual action as futile against systemic "
    "forces.",

    "Neutral: conversational posts, jokes, greetings, everyday life (food, sports "
    "commentary, small talk), bare links, routine or administrative news that "
    "induces no anxiety, or a mere report that some third party did something. No "
    "psychological framing of control or agency either way.",

    "Internal but abstract: fosters agency while staying general - a vague "
    "intention, a philosophy, or a realization without a concrete situated "
    "action ('I need to do better', 'you make your own luck'). Also: owning a "
    "failure by naming a controllable cause, or crediting a success to effort, "
    "practice, or a chosen strategy.",

    "Internal and concrete: fosters agency AND names a specific tangible "
    "proactive step, a completed action, or an achieved result - including "
    "civic engagement and local collective action where the participants have "
    "direct agency ('we must invest in our neighbourhood garden').",
]

LOC_INSTRUCTIONS = (
    "Rate how this social media post affects a reader's Locus of Control: lower "
    "means a stronger EXTERNAL locus (helplessness), higher a stronger INTERNAL "
    "locus (agency). Judge the EMOTIONAL INTENT, SUBTEXT, and overall EMOTIONAL "
    "ARC, not just the literal actors. If a post opens in anxiety but resolves "
    "into a personal action plan, weigh the resolution. Posts may be in any "
    "language, most often German or English; judge the German ones on the same "
    "scale without translating."
)

CLASS_CRITERIA = {
    "internal": (
        "Fosters the reader's sense of agency: constructive solutions, proactive "
        "steps the reader, author, or ordinary people can take, personal ownership "
        "of emotions, actions, or outcomes, civic engagement, or united community "
        "action overcoming obstacles. Success credited to controllable factors - "
        "effort, practice, learnable skill, chosen strategy. Owning a failure by "
        "naming its controllable causes or intending to correct course."
    ),
    "external": (
        "Fosters helplessness, fatalism, or anxiety: the writer or reader as passive "
        "victim of large corporations, governments, or macro-events; waiting on "
        "top-down authorities to act; demanding accountability from remote powers, "
        "even when the vocabulary sounds active; deep anxiety about uncontrollable "
        "events; reporting negative or worrying events without any solutions, a "
        "factual tone notwithstanding; or attributing outcomes to fixed causes - "
        "innate talent, luck, fatalistic self-condemnation."
    ),
    "neutral": (
        "Purely conversational posts, jokes, greetings, everyday life (food, sports "
        "commentary, mundane event reports, small talk), bare links, or anything "
        "without psychological framing of control or agency. A mere report that a "
        "third party did something is neutral unless it credits controllable "
        "factors or reports a threatening event with no solution."
    ),
}

AGENCY_INSTRUCTIONS = (
    "The post leaves a reader feeling more able to act on their own life - it "
    "offers a proactive step, owns an outcome, or shows effort paying off - "
    "rather than more helpless in the face of forces beyond their control."
)


def questions():
    return {
        "loc_score": {
            "type": "score",
            "instructions": LOC_INSTRUCTIONS,
            "criteria": LOC_LEVELS,
        },
        "loc_class": {
            "type": "choice",
            "instructions": (
                "Which Locus-of-Control class does this post belong to? "
                + DEFINITIONS_V3.split("Key Guidelines")[0].strip()
            ),
            "criteria": CLASS_CRITERIA,
        },
        "agency": {
            "type": "noul",
            "instructions": AGENCY_INSTRUCTIONS,
        },
    }
