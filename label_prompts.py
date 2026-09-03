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
  v3 - refined edge cases around "we must invest" (local vs systemic), mixed
       signals (emotional arc / resolution), neutral boundaries for negative
       event reports, and temporal dynamics in self-condemnation.
  v4 - v3 definitions plus a graded 0-9 locus-of-control Score (read from digit
       logprobs) for learning-to-rank. Keeps the v3 Label/Level lines so all
       existing consumers are unaffected; adds loc_score / loc_score_logprobs.

Select at runtime with LOC_PROMPT_VERSION=v1|v2|v3|v4 (default v1).
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
# v3 (refined edge cases, mixed signals, and tone calibration)
# ---------------------------------------------------------------------------

DEFINITIONS_V3 = (
    "You are an expert psycholinguist analyzing how news and social media text implicitly "
    "affects a reader's Locus of Control. Posts may be in any language (often German or "
    "English). Focus on the EMOTIONAL INTENT, SUBTEXT, and the overall EMOTIONAL ARC, "
    "not just the literal actors.\n\n"
    "Key Guidelines for Ambiguities:\n"
    "1. Mixed-Signal Posts & Tone Calibration: If a post begins with anxiety but resolves "
    "into a personal action plan or constructive takeaway (e.g., 'I'm worried about climate "
    "change, so I'll recycle more'), weigh the resolution heavily; this is 'internal'. If it "
    "stays mired in worry without resolution, it is 'external'.\n"
    "2. Demands vs. Collective Action: Demanding that remote institutions or politicians fix "
    "things is 'external' (reliance on top-down authority). A call to collective, local "
    "action where participants have direct agency (e.g., 'we must invest in our neighborhood "
    "garden') is 'internal'.\n"
    "3. Event Reports: Routine or administrative news (e.g., 'The mayor announced a new budget') "
    "that does not inherently induce anxiety is 'neutral'. Reports of threatening, dangerous, "
    "or tragic events without solutions foster anxiety and are 'external'.\n"
    "4. Temporal Dynamics of Failure: Global self-condemnation with no path forward ('I am "
    "just a failure') is 'external'. Acknowledging a failure with an intent to understand or "
    "correct it ('I'm failing and need to figure out why') is 'internal'.\n"
    "5. Cynicism & Sarcasm toward Agency: Posts that sarcastically suggest personal action "
    "or mock the idea of individual responsibility (e.g., 'Oh sure, my recycling will stop "
    "global warming' or 'Just pull yourself up by your bootstraps!') reject the efficacy of "
    "personal agency. Because the subtext emphasizes that individual effort is futile against "
    "overwhelming systemic forces, these are 'external'.\n\n"
    "- 'internal': The text fosters the reader's sense of agency: constructive solutions, "
    "proactive steps the reader/author/ordinary people can take, personal ownership of "
    "emotions/actions/outcomes, civic engagement, or united community action overcoming "
    "obstacles. Success (one's own or someone else's) credited to controllable factors - "
    "effort, practice, learnable skill, or chosen strategy - is internal. So is owning a "
    "failure by naming its controllable causes or intending to correct course. (Empowering, "
    "proactive, solution-oriented).\n"
    "- 'external': The text fosters helplessness, fatalism, or anxiety: the writer/reader "
    "as passive victim of large corporations, governments, or macro-events; waiting on "
    "top-down authorities to act; demanding accountability from remote powers; deep anxiety "
    "about uncontrollable events; reporting negative or worrying events without any solutions; "
    "or attributing outcomes to fixed, uncontrollable factors - innate talent, luck, or "
    "fatalistic self-condemnation. (Disempowering, fatalistic, anxious).\n"
    "- 'neutral': Purely conversational posts, jokes, greetings, everyday life (food, "
    "sports commentary, mundane event reports, small talk), bare links, or anything "
    "without psychological framing of control or agency. Mere reports of third-party actions "
    "are neutral unless they credit controllable factors (internal) or report threatening/anxiety-inducing "
    "events without solutions (external).\n"
)

TEACHER_PROMPT_V3 = (
    DEFINITIONS_V3
    + "\nAdditionally, if the label is 'internal', you must assess its Concreteness on a 3-level scale:\n"
    "- Level 1 (Vague / General): Abstract thoughts, general philosophies, or vague statements without specific context. (e.g., 'I need to do better in life', 'You make your own luck').\n"
    "- Level 2 (Specific Context / Intention): Relates to a specific, tangible situation, but describes an intention, realization, or minor choice rather than a completed concrete action. (e.g., 'I really need to study for my math test tonight instead of playing games').\n"
    "- Level 3 (Concrete Action / Result): Describes a specific, tangible situation AND a concrete action taken or specific result achieved. (e.g., 'I turned off my phone for two hours and finally finished my entire math assignment').\n"
    "If the label is 'external' or 'neutral', always output 'Level: 1'.\n\n"
    "Think briefly inside <think>...</think>: at most two short sentences naming the dominant cue and your choice. Do not translate or restate the text.\n"
    "After closing the think tag, output EXACTLY these two lines and nothing else:\n"
    "Label: X\n"
    "Level: Y\n"
    "where X is a single letter: I (internal), E (external), or N (neutral), and Y is a single digit: 1, 2, or 3."
)

VALIDATION_PROMPT_V3 = (
    DEFINITIONS_V3
    + "You must respond with EXACTLY ONE WORD: 'internal', 'external', or 'neutral'. "
    "Do not include any explanations or punctuation."
)

# ---------------------------------------------------------------------------
# v4 (graded 0-9 locus-of-control Score for learning-to-rank)
# ---------------------------------------------------------------------------
#
# v4 keeps the v3 Label/Level lines (so raw_logprobs / level_logprobs stay
# populated and every existing consumer is unaffected) and ADDS a single-digit
# `Score:` line. The score is a genuinely graded relevance read out from the
# digit-token logprobs (see annotate_lmstudio_soft_labels.py): unlike the
# near-one-hot Label distribution, adjacent digits carry real probability mass,
# so the expected score is a smooth 0-9 target. The single axis folds direction
# (external<->internal) and intensity (concreteness) together and supersedes the
# P(internal)*E[level] product for ranking.

TEACHER_PROMPT_V4 = (
    DEFINITIONS_V3
    + "\nAdditionally, you must rate the text's Locus of Control on a single 0-9 scale, where "
    "higher means a stronger INTERNAL locus (more agency) and lower means a stronger EXTERNAL "
    "locus (more helplessness). Use the full range:\n"
    "- 0-1 (Strong external): pronounced helplessness, fatalism, victimhood, or anxiety about "
    "uncontrollable events, with no path to act.\n"
    "- 2-3 (Mild external): leans disempowering - demands on remote authorities, worry, or "
    "negative event reports without solutions.\n"
    "- 4-5 (Neutral): conversational, mundane, or purely factual with no framing of control or "
    "agency (4 if it leans slightly external, 5 if perfectly neutral).\n"
    "- 6-7 (Internal, abstract): fosters agency but stays general - a vague intention, "
    "philosophy, or realization without a concrete, situated action.\n"
    "- 8-9 (Internal, concrete): fosters agency AND names a specific, tangible proactive step, "
    "completed action, or achieved result (9 for the most concrete and empowering).\n\n"
    "Also emit the v3 Label and Level for compatibility. The Score must be consistent with the "
    "Label (external/neutral score <=5, internal score >=6) and with the Level for internal "
    "posts (Level 1 -> 6, Level 2 -> 7, Level 3 -> 8-9).\n\n"
    "Think briefly inside <think>...</think>: at most two short sentences naming the dominant cue "
    "and your choice. Do not translate or restate the text.\n"
    "After closing the think tag, output EXACTLY these three lines and nothing else:\n"
    "Label: X\n"
    "Level: Y\n"
    "Score: Z\n"
    "where X is a single letter I (internal), E (external), or N (neutral); Y is a single digit "
    "1, 2, or 3; and Z is a single digit 0-9."
)

VALIDATION_PROMPT_V4 = VALIDATION_PROMPT_V3

# ---------------------------------------------------------------------------
# assist (helpful-request vs inert-question ranking axis)
# ---------------------------------------------------------------------------
#
# A separate ranking axis from the Locus-of-Control scheme. The polarity is the
# ACTIONABILITY OF A REQUEST: does the author ask for help or advice about a
# concrete, real problem a respondent could act on (helpful), or is the
# question rhetorical / engagement-bait / circular with no practical
# resolution (inert)? Offers, answers, tutorials, and how-tos are out of scope
# (neutral) - only REQUESTS are classified on the axis. The graded 0-9 Score
# folds direction (inert<->helpful) and specificity together for learning-to-rank,
# exactly like v4's loc_score. Output format is identical to v4 (Label/Level/
# Score) so annotate_lmstudio_soft_labels.py parses it unchanged once the
# letter map is switched.

LETTER_TO_LABEL_ASSIST = {"h": "helpful", "r": "inert", "n": "neutral"}
LABELS_ASSIST = ["helpful", "inert", "neutral"]

DEFINITIONS_ASSIST = (
    "You are an expert content analyst classifying social media posts by whether they contain a "
    "REQUEST for help or advice that leads somewhere actionable versus a question that leads "
    "nowhere. The axis is the ACTIONABILITY OF THE REQUEST: does the author ask for help or advice "
    "about a concrete, real problem that a respondent could act on - or is the question "
    "rhetorical, circular, or engagement-seeking, with no practical resolution? Posts may be in "
    "any language (often German or English). Focus on the REQUEST'S INTENT AND SPECIFICITY, not "
    "just whether it contains a question mark.\n\n"
    "Scope: only REQUESTS for help or advice are classified as 'helpful' or 'inert'. Offers of "
    "help, answers, solutions, tutorials, and how-tos are NOT requests - they are 'neutral' "
    "regardless of how specific or practical they are.\n\n"
    "Key Guidelines for Ambiguities:\n"
    "1. Situated vs Rhetorical Requests: A request tied to a real problem the author faces, with "
    "context ('How do I fix this TypeError in my pytest fixture?') is 'helpful'. A question asked "
    "to spark debate or perform expertise with no real problem behind it ('Can AI really replace "
    "humans?', 'Why is modern software so bloated?') is 'inert'.\n"
    "2. Genuine Help-Seeking vs Engagement-Bait: Genuine requests include context, error "
    "messages, or specifics that narrow the problem. Engagement-bait ('Did you know that...?', "
    "'Who else thinks...?', 'Agree?', 'What's your hot take?') invites endless replies but "
    "resolves nothing - it is 'inert'.\n"
    "3. Concrete vs Generic Advice-Seeking: A request for advice on a specific, situated decision "
    "or problem ('Should I use drizzle or prisma for a Turso-backed app with 50k rows?') is "
    "'helpful'. A request for generic life advice with no situation ('How do I become more "
    "confident?') is 'inert'.\n"
    "4. Circular / Philosophical: Questions that are interesting to discuss but have no practical "
    "resolution ('What is the meaning of success?', 'Is free will real?') are 'inert'. A question "
    "can be sincere and still lead nowhere if no answer would change what the asker does.\n"
    "5. Requests vs Offers: A post asking for help is in scope. A post offering help ('DM me, I "
    "just did this migration') is an offer, not a request - it is 'neutral'.\n"
    "6. Requests vs Answers: A post providing a solution or answer to someone else's question is "
    "'neutral', not 'helpful', even if it is specific and practical. Only the request itself is "
    "classified.\n"
    "7. Requests vs Tutorials: A how-to or tutorial ('Here's how I configured nginx for websocket "
    "proxying') is 'neutral'. If the author is asking how to do it, that is a request and may be "
    "'helpful' or 'inert' depending on specificity.\n"
    "8. Opinions / Statements: Pure opinions, hot takes, or statements with no request are "
    "'neutral'. 'inert' is specifically a question that poses as a request but leads nowhere - not "
    "mere opinion-sharing.\n"
    "9. Help-Seeking vs Promotional Asks: A request for the reader to buy, subscribe, donate, "
    "vote, or sign up is a promotional ask, not a request for help or advice - it is 'neutral' "
    "(or 'inert' if phrased as engagement-bait). Only requests where the author seeks help, "
    "advice, or knowledge for themselves are in scope.\n\n"
    "- 'helpful': The post is a genuine, situated REQUEST for help or advice: a concrete question "
    "about a real problem with enough context that a respondent could act on it. The request "
    "leads somewhere. (Actionable, situated, specific).\n"
    "- 'inert': The post is a question that leads nowhere: rhetorical or engagement-bait "
    "questions, circular debate prompts, abstract philosophical questions with no practical "
    "resolution, generic advice-solicitation with no situation, or opinion-solicitation that "
    "won't produce action. It poses as a request but nothing resolves. (Rhetorical, circular, "
    "unactionable).\n"
    "- 'neutral': Everything else: statements, opinions, news, jokes, chat, AND offers of help, "
    "answers, solutions, tutorials, or how-tos (these are not requests).\n"
)

TEACHER_PROMPT_ASSIST = (
    DEFINITIONS_ASSIST
    + "\nAdditionally, if the label is 'helpful', you must assess the request's Specificity on a "
    "3-level scale:\n"
    "- Level 1 (Vague): A general request with no specifics ('Can someone help me with Python?', "
    "'Any advice on being more productive?').\n"
    "- Level 2 (Situated): A specific problem or context, but missing some detail a respondent "
    "would need ('I keep getting a TypeError when I run pytest, anyone seen this?').\n"
    "- Level 3 (Concrete): A specific problem AND concrete detail - error messages, part "
    "numbers, code snippets, version info, what was already tried ('Getting fixture-not-found in "
    "pytest async with pytest-asyncio 0.23 on Python 3.12, tried @pytest.fixture already, didn't "
    "work - what am I missing?').\n"
    "If the label is 'inert' or 'neutral', always output 'Level: 1'.\n\n"
    "Additionally, you must rate the post's Actionability on a single 0-9 scale, where higher "
    "means the request is more concrete and actionable and lower means the question is more "
    "rhetorical and leading-nowhere. Use the full range:\n"
    "- 0-1 (Strong inert): blatant engagement-bait, clickbait questions, circular debate with no "
    "possible resolution, abstract philosophy, trivia-as-performance.\n"
    "- 2-3 (Mild inert): leans unactionable - somewhat broad advice-solicitation, opinion polls, "
    "questions that won't produce action.\n"
    "- 4-5 (Neutral): no request present, or perfectly balanced between rhetorical and "
    "actionable.\n"
    "- 6-7 (Helpful, general): a genuine request but somewhat general - real problem without full "
    "specifics.\n"
    "- 8-9 (Helpful, concrete): a genuine request AND highly specific - error messages, part "
    "numbers, code, version info, what was tried (9 for the most concrete and actionable "
    "request).\n\n"
    "The Score must be consistent with the Label (inert/neutral score <=5, helpful score >=6) "
    "and with the Level for helpful posts (Level 1 -> 6, Level 2 -> 7, Level 3 -> 8-9).\n\n"
    "Think briefly inside " + "<think>" + "..." + "</think>" + ": at most two short sentences "
    "naming the dominant cue and your choice. Do not translate or restate the text.\n"
    "After closing the think tag, output EXACTLY these three lines and nothing else:\n"
    "Label: X\n"
    "Level: Y\n"
    "Score: Z\n"
    "where X is a single letter H (helpful), R (inert), or N (neutral); Y is a single digit "
    "1, 2, or 3; and Z is a single digit 0-9."
)


def prompt_version():
    return os.environ.get("LOC_PROMPT_VERSION", "v1")


def get_teacher_prompt():
    v = prompt_version()
    if v == "assist":
        return TEACHER_PROMPT_ASSIST
    if v == "v4":
        return TEACHER_PROMPT_V4
    if v == "v3":
        return TEACHER_PROMPT_V3
    return TEACHER_PROMPT_V2 if v == "v2" else TEACHER_PROMPT_V1


def get_validation_prompt():
    v = prompt_version()
    if v in ("v3", "v4"):
        return VALIDATION_PROMPT_V3
    return VALIDATION_PROMPT_V2 if v == "v2" else VALIDATION_PROMPT_V1
