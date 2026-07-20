"""Single source of truth for all Locus-of-Control labeling prompts.

Versions:
  v1 - the original prompts. NOTE: teacher (LM Studio) and validator (Gemini/
       OpenRouter) historically used DIFFERENT definitions; both are preserved
       here byte-for-byte so existing datasets stay reproducible.
  v2 - unified definitions for teacher and validator, extended with decision
       rules for the failure modes found in vector_errors.json: agentic
       vocabulary with external reliance, third-party effort attributions,
       and event reports / small talk as neutral. The v2 teacher answers with
       a single letter ("Label: I/E/N") so all three classes appear in
       top_logprobs, giving real soft labels instead of the -20 clamp.

Select at runtime with LOC_PROMPT_VERSION=v1|v2 (default v1).
"""

import os

LABELS = ["internal", "external", "neutral"]
LETTER_TO_LABEL = {"i": "internal", "e": "external", "n": "neutral"}

# ---------------------------------------------------------------------------
# v1 (original, unchanged)
# ---------------------------------------------------------------------------

# Teacher prompt of dataset_annotated_lmstudio_soft.jsonl (short definitions,
# <think> reasoning, one-word answer).
TEACHER_PROMPT_V1 = (
    "You are an expert psycholinguist analyzing how news and social media text implicitly affects a reader's Locus of Control. Focus on the EMOTIONAL INTENT and SUBTEXT, not just the literal actors in the sentence.\n"
    "- 'internal': The text fosters agency, offers constructive solutions, or highlights proactive steps. (Empowering, proactive, solution-oriented).\n"
    "- 'external': The text fosters helplessness, fatalism, or anxiety. It positions the writer/reader as a passive victim of large corporations or macro-events. (Disempowering, fatalistic, anxious).\n"
    "- 'neutral': The text is purely conversational, a joke, irrelevant, or lacks psychological framing regarding control or agency.\n\n"
    "First, analyze the text step-by-step. Write your reasoning inside <think>...</think> tags.\n"
    "After closing the think tag, write EXACTLY ONE WORD as your final decision: 'internal', 'external', or 'neutral'.\n"
    "Example format:\n"
    "<think>The text shows... Therefore, it fosters agency.</think> internal"
)

# Validator prompt of dataset_validation_gemini.jsonl, dataset_annotated.jsonl
# and dataset_annotated_openrouter.jsonl (extended definitions, one word,
# no reasoning). Note the extra clauses the v1 teacher never saw.
VALIDATION_PROMPT_V1 = (
    "You are an expert psycholinguist analyzing how news and social media text implicitly affects a reader's Locus of Control. Focus on the EMOTIONAL INTENT and SUBTEXT, not just the literal actors in the sentence.\n"
    "- 'internal': The text fosters agency, offers constructive solutions, or highlights proactive steps. It includes advocating for civic engagement, taking personal ownership of emotions/actions, or showing how united community action overcomes obstacles. (Empowering, proactive, solution-oriented).\n"
    "- 'external': The text fosters helplessness, fatalism, or anxiety. It positions the writer/reader as a passive victim of large corporations, governments, or macro-events. It includes waiting on top-down authorities to act, demanding accountability from others (which implies reliance on external powers), expressing deep anxiety about uncontrollable events, or reporting negative events without any solutions. (Disempowering, fatalistic, anxious).\n"
    "- 'neutral': The text is purely conversational, a joke, irrelevant, or entirely lacks any psychological framing regarding control or agency.\n"
    "You must respond with EXACTLY ONE WORD: 'internal', 'external', or 'neutral'. Do not include any explanations or punctuation."
)

# Historical experiment kept for reference (used only by test_annotate.py):
# counts "demanding accountability" as internal instead of external.
VALIDATION_PROMPT_V1_ACCOUNTABILITY_AS_INTERNAL = (
    "You are an expert psycholinguist analyzing how news and social media text implicitly affects a reader's Locus of Control. Focus on the EMOTIONAL INTENT and SUBTEXT, not just the literal actors in the sentence.\n"
    "- 'internal': The text fosters agency, offers constructive solutions, or highlights proactive steps. It includes advocating for civic engagement, demanding accountability, taking personal ownership of emotions/actions, or showing how united community action overcomes obstacles. (Empowering, proactive, solution-oriented).\n"
    "- 'external': The text fosters helplessness, fatalism, or anxiety. It positions the writer/reader as a passive victim of large corporations, governments, or macro-events. It includes waiting on top-down authorities to act, expressing deep anxiety about uncontrollable events, or reporting negative events without any solutions. (Disempowering, fatalistic, anxious).\n"
    "- 'neutral': The text is purely conversational, a joke, irrelevant, or entirely lacks any psychological framing regarding control or agency.\n"
    "You must respond with EXACTLY ONE WORD: 'internal', 'external', or 'neutral'. Do not include any explanations or punctuation."
)

# Rewrite prompt shared by create_counterexamples.py and
# create_hard_counterexamples.py.
COUNTEREXAMPLE_PROMPT = (
    "You are an expert psycholinguist and writer. Your task is to rewrite the provided text to have the exact OPPOSITE locus of control.\n"
    "- If the original text has an 'internal' locus of control (focusing on personal agency and proactive steps), rewrite it to have an 'external' locus of control (focusing on external circumstances, luck, or structural factors outside one's control).\n"
    "- If the original text has an 'external' locus of control, rewrite it to have an 'internal' locus of control.\n"
    "Maintain the exact same core topic, language, and approximate length. Change ONLY the psychological framing.\n"
    "Return ONLY the rewritten text. Do not include any explanations, preambles, or quotation marks."
)

# ---------------------------------------------------------------------------
# v2 (unified teacher/validator definitions + failure-mode decision rules)
# ---------------------------------------------------------------------------

DEFINITIONS_V2 = (
    "You are an expert psycholinguist analyzing how news and social media text implicitly "
    "affects a reader's Locus of Control. Posts may be in any language (often German or "
    "English). Focus on the EMOTIONAL INTENT and SUBTEXT, not just the literal actors in "
    "the sentence.\n"
    "- 'internal': The text fosters the reader's sense of agency: constructive solutions, "
    "proactive steps the reader/author/ordinary people can take, personal ownership of "
    "emotions/actions/outcomes, civic engagement, or united community action overcoming "
    "obstacles. Success (one's own or someone else's) credited to controllable factors - "
    "effort, practice, learnable skill, or chosen strategy - is internal. So is owning a "
    "failure by naming its controllable causes (too little preparation, wrong approach), "
    "especially with intent to correct course. (Empowering, proactive, solution-oriented).\n"
    "- 'external': The text fosters helplessness, fatalism, or anxiety: the writer/reader "
    "as passive victim of large corporations, governments, or macro-events; waiting on "
    "top-down authorities to act; demanding that politicians/institutions/others fix "
    "things, which implies reliance on external powers - even when the vocabulary sounds "
    "active (e.g. 'we must invest', 'X needs to finally act'); deep anxiety about "
    "uncontrollable events; reporting negative or worrying events without any solutions "
    "(this includes plain factual news reports - a factual tone does not make them "
    "neutral); or attributing outcomes to fixed, uncontrollable factors - innate talent, "
    "luck, or global self-condemnation ('I am just a failure') with no path to improve. "
    "(Disempowering, fatalistic, anxious).\n"
    "- 'neutral': Purely conversational posts, jokes, greetings, everyday life (food, "
    "sports commentary, mundane event reports, small talk), bare links, or anything "
    "without psychological framing of control or agency. Mere reports that some third "
    "party did something are neutral, not internal, unless the text credits controllable "
    "factors (effort, practice, strategy) or offers the reader a path to act. A report "
    "of a negative or worrying event with no solution is external, not neutral.\n"
)

TEACHER_PROMPT_V2 = (
    DEFINITIONS_V2
    + "\nThink briefly inside <think>...</think>: at most two short sentences naming the "
    "dominant cue and your choice. Do not translate or restate the text.\n"
    "After closing the think tag, output ONLY this line and nothing else:\n"
    "Label: X\n"
    "where X is a single letter: I (internal), E (external), or N (neutral)."
)

VALIDATION_PROMPT_V2 = (
    DEFINITIONS_V2
    + "You must respond with EXACTLY ONE WORD: 'internal', 'external', or 'neutral'. "
    "Do not include any explanations or punctuation."
)

# ---------------------------------------------------------------------------


def prompt_version():
    return os.environ.get("LOC_PROMPT_VERSION", "v1")


def get_teacher_prompt():
    return TEACHER_PROMPT_V2 if prompt_version() == "v2" else TEACHER_PROMPT_V1


def get_validation_prompt():
    return VALIDATION_PROMPT_V2 if prompt_version() == "v2" else VALIDATION_PROMPT_V1
