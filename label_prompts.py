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

Separate ranking axes (not LoC versions), selected the same way:
  assist - actionability of a request for help (helpful/inert/neutral).
  virtue - intellectual virtue vs dogmatism in how a contestable claim is
           held (virtuous/dogmatic/neutral).

Select at runtime with LOC_PROMPT_VERSION=v1|v2|v3|v4|assist|virtue (default v1).
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
    "advice, or knowledge for themselves are in scope.\n"
    "10. Reading Audience vs Specific Third Party: a request is only 'helpful' if a READER of the "
    "post could plausibly answer it. A request aimed at one named operator, service desk, "
    "company, or individual who is not the audience - a support ticket, a booking or account "
    "query, an @-mention asking one specific person to do something - is 'neutral', however "
    "concrete and urgent it is. The same holds for posts from automated relay or log accounts "
    "that reproduce somebody else's operational exchange (radio traffic, dispatch messages, "
    "ticket feeds): the request was never addressed to the reader, so it is 'neutral'.\n\n"
    "- 'helpful': The post is a genuine, situated REQUEST for help or advice, addressed to whoever "
    "reads it: a concrete question about a real problem with enough context that a reader could "
    "act on it and give the author a useful answer. The request leads somewhere. (Actionable, "
    "situated, specific, answerable by the audience).\n"
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
    "Additionally, you must rate the post's Answerability on a single 0-9 scale: how well could a "
    "reader who happens to know the subject actually help this author? Higher means a reader "
    "could give a genuinely useful answer; lower means no answer would change anything. Use the "
    "full range:\n"
    "- 0-1 (Strong inert): blatant engagement-bait, clickbait questions, circular debate with no "
    "possible resolution, abstract philosophy, trivia-as-performance.\n"
    "- 2-3 (Mild inert): leans unactionable - somewhat broad advice-solicitation, opinion polls, "
    "questions that won't produce action.\n"
    "- 4-5 (Neutral): no request present, or perfectly balanced between rhetorical and "
    "actionable.\n"
    "- 6 (Helpful, but a reader would have to ask a clarifying question before being able to "
    "answer at all).\n"
    "- 7 (Helpful and answerable, but broad - many equally valid answers, little to narrow them "
    "down).\n"
    "- 8 (Helpful and well-specified - a knowledgeable reader could give a directly useful "
    "answer straight away).\n"
    "- 9 (Helpful and fully specified - error text, model or part number, version, what was "
    "already tried; one reply could resolve it).\n\n"
    "The Score band is a HARD constraint, checked before you answer: if the Label is H the Score "
    "MUST be 6, 7, 8 or 9 - never lower, however unanswerable you judge the request to be. If a "
    "request feels too unanswerable to deserve a 6, that is a signal the Label itself should be "
    "R, not a reason to write H with a low Score. If the Label is R or N the Score MUST be 0-5. "
    "Within its band the Score is an independent judgement from the Level: Level "
    "measures how much detail the author supplied, Score measures how usefully a reader could "
    "answer. Two posts at the same Level must get different Scores when one is easier to answer "
    "usefully than the other, and a short question can score high if it is precise. Spread your "
    "scores across the band - do not default to the same digit for every helpful post.\n\n"
    "Think briefly inside " + "<think>" + "..." + "</think>" + ": at most two short sentences "
    "naming the dominant cue and your choice. Do not translate or restate the text.\n"
    "After closing the think tag, output EXACTLY these three lines and nothing else:\n"
    "Label: X\n"
    "Level: Y\n"
    "Score: Z\n"
    "where X is a single letter H (helpful), R (inert), or N (neutral); Y is a single digit "
    "1, 2, or 3; and Z is a single digit 0-9."
)

VALIDATION_PROMPT_ASSIST = (
    DEFINITIONS_ASSIST
    + "You must respond with EXACTLY ONE WORD: 'helpful', 'inert', or 'neutral'. "
    "Do not include any explanations or punctuation."
)


# ---------------------------------------------------------------------------
# virtue (intellectual-virtue vs dogmatism ranking axis)
# ---------------------------------------------------------------------------
#
# A third ranking axis, independent of Locus-of-Control and of assist. The
# polarity is EPISTEMIC CONDUCT: how does the author hold and handle a
# contestable claim? Virtuous conduct = humility about one's own limits,
# proportioned confidence, weighing competing considerations or values,
# scoping claims instead of generalizing, engaging the strongest opposing
# reading, naming what would change one's mind, publicly updating. Dogmatic
# conduct = blanket statements about whole groups, certainty out of proportion
# to the evidence, single-cause explanations, unfalsifiable framings,
# strawmanning and motive attribution, contempt for anyone who disagrees.
#
# Deliberately NOT part of the axis: whether the claim is true, which side of
# a dispute the author is on, and how politely it is phrased. A correct claim
# asserted with contempt is dogmatic; a mistaken claim held tentatively with
# reasons is virtuous.
#
# Known confound to watch in training: epistemic virtue correlates with
# register and length (long, essayistic, academic posts). The synthetic
# counterexample step (create_synthetic_counterexamples.py) matters more here
# than on the other axes - it produces same-length, same-voice rewrites that
# flip only the epistemic framing, which is what breaks the "nuance = long
# words" shortcut the embedding would otherwise learn.
#
# Output format is identical to v4/assist (Label/Level/Score) so
# annotate_lmstudio_soft_labels.py parses it unchanged.

LETTER_TO_LABEL_VIRTUE = {"v": "virtuous", "d": "dogmatic", "n": "neutral"}
LABELS_VIRTUE = ["virtuous", "dogmatic", "neutral"]

DEFINITIONS_VIRTUE = (
    "You are an expert analyst of reasoning quality, classifying social media posts by the "
    "INTELLECTUAL VIRTUE OF THE AUTHOR'S CONDUCT: how the author holds and handles a "
    "contestable claim. Does the post show humility, proportioned confidence, and the weighing "
    "of competing considerations - or does it show dogmatism: blanket statements, unearned "
    "certainty, and closure to any counter-argument? Posts may be in any language (often German "
    "or English).\n\n"
    "Judge the CONDUCT, never the position. You are not assessing whether the claim is true, "
    "which side of a dispute the author is on, or whether you agree. A claim you consider "
    "correct, asserted with contempt and without scope, is 'dogmatic'. A claim you consider "
    "wrong, held tentatively and supported with reasons, is 'virtuous'. Political, religious, "
    "and cultural positions of every kind are treated identically.\n\n"
    "Scope: the axis only applies to posts that make or discuss a CONTESTABLE claim - something "
    "a reasonable person could dispute. Chat, jokes, logistics, personal narrative, links, "
    "announcements, plain factual reports, and questions carry no epistemic conduct and are "
    "'neutral'.\n\n"
    "Key Guidelines for Ambiguities:\n"
    "1. Hedging Theater: a hedge counts only if it actually governs the claim. 'I might be "
    "wrong, but they are all corrupt', 'just my opinion, but anyone who thinks X is an idiot', "
    "'not all X, but X people always Y' are 'dogmatic' - the disclaimer is decoration on an "
    "unscoped blanket claim. Genuine humility limits what is being asserted.\n"
    "2. Nuance-Signalling vs Real Weighing: 'it's complicated', 'there are many sides to this', "
    "'reality is messy' with nothing named is NOT virtuous - it is 'neutral' at best. Real "
    "weighing names at least one specific consideration that cuts against the author's own "
    "position, or two specific values in tension, and says how they trade off.\n"
    "3. Register Is Not Virtue: academic vocabulary, citations, thread length, and calm tone do "
    "not make a post virtuous; a long, well-written essay can be thoroughly 'dogmatic'. A short, "
    "casual post can be 'virtuous' ('huh, i assumed the opposite - what am i missing?'). Judge "
    "the epistemic move, not the prose.\n"
    "4. Confidence Is Allowed: stating a firm conclusion is not dogmatic when the confidence is "
    "backed by reasons and the claim is scoped to what those reasons support. Dogmatism is "
    "UNEARNED absoluteness: no reasons, no scope, and no conceivable counter-evidence.\n"
    "5. Disagreement Is Allowed: arguing forcefully against a named argument, with reasons, is "
    "'virtuous' or 'neutral'. What makes disagreement 'dogmatic' is attacking the person or "
    "imputing motives ('they only say this because they're paid to') instead of the argument, "
    "strawmanning the opposing view, or treating disagreement itself as proof of bad faith.\n"
    "6. Groups vs Actors: unscoped claims about a broad group ('X people are all Y', 'everyone "
    "who votes Z is W') are the paradigm case of 'dogmatic'. A scoped claim about specific "
    "actors and their documented actions is not dogmatic, however harsh.\n"
    "7. Anecdote Scope: telling one's own experience as an experience is 'neutral'. Generalizing "
    "one experience into a universal law ('this happened to me, which proves X always...') is "
    "'dogmatic'. Noting the limits of one's own sample ('n=1, but...') is 'virtuous'.\n"
    "8. Manufactured Doubt: uncertainty must be proportionate. Treating a well-established "
    "matter as wide open, or demanding impossible proof in order to dismiss evidence, is not "
    "humility - it is 'dogmatic' when it serves to shut an argument down, 'neutral' otherwise.\n"
    "9. Facts and Values Both Count: calibration about evidence ('the data is thin, this could "
    "go either way') and weighing values ('I want both the openness and the safety here and I "
    "can't have all of both') are equally 'virtuous'.\n"
    "10. Reporting Someone Else's Dogmatism: when a post quotes, screenshots, or replies to a "
    "sweeping claim, classify the AUTHOR'S OWN conduct. A quote-post that only dunks is "
    "'dogmatic'; one that engages what the quoted claim actually argues is 'virtuous'; one that "
    "merely relays it without comment is 'neutral'.\n"
    "11. Jokes and Hyperbole: comic exaggeration, satire, and obvious bits are 'neutral', not "
    "dogmatic - unless the joke's payload is a serious unscoped claim about a group.\n"
    "12. Questions: a question is not virtuous merely for being a question - whether a request "
    "is answerable is a different axis entirely. A question is 'virtuous' only when it shows the "
    "author reconsidering their own view ('what's the best argument against my position here?'); "
    "rhetorical questions that assert a blanket claim ('how can anyone still believe X?') are "
    "'dogmatic'; ordinary information-seeking questions are 'neutral'.\n"
    "13. Updating: explicitly revising a stated prior position ('I argued the opposite last "
    "year; this changed my mind because...') is the strongest form of the virtue. Vague "
    "'I've grown a lot' with no content is 'neutral'.\n"
    "14. Default to Neutral - this is the most important rule and the easiest to get wrong. "
    "Before labelling a post 'dogmatic' you must be able to name the specific move that makes it "
    "so: the unscoped claim about a whole group, the certainty asserted with no reasons, the "
    "imputed motive, the contempt standing in for an argument, the framing no evidence could "
    "touch. If you cannot name one, the label is 'neutral'. The following are NOT dogmatism, "
    "however strongly worded: criticism of named actors and their specific actions; anger, "
    "complaint, and venting; sarcasm or mockery aimed at a particular act rather than at a class "
    "of people; taste claims and enthusiasm ('still the best series ever made'); hyperbole no "
    "reader would take literally; hedged relays of rumour or reporting ('apparently they...'); "
    "and plain factual reports of what someone did, even when the facts are selected to make a "
    "point. Having a side is not a vice. Most posts on a political, moral, or cultural topic are "
    "'neutral' on this axis, and an axis that fires on every heated political post is broken.\n\n"
    "- 'virtuous': The post handles a contestable claim well: it acknowledges the limits of the "
    "author's own knowledge or sample, proportions confidence to the evidence, scopes claims to "
    "the cases they cover, weighs competing considerations or values against each other, engages "
    "the strongest version of an opposing view, names what would change the author's mind, or "
    "publicly updates a prior position. (Humble, calibrated, weighing, scoped, revisable).\n"
    "- 'dogmatic': The post handles a contestable claim badly: blanket generalizations about "
    "whole groups, certainty out of proportion to the evidence given, single-cause explanations "
    "of complicated things, framings no evidence could ever contradict, strawmanning or motive "
    "attribution, contempt or ridicule in place of argument, appeals to obviousness ('anyone can "
    "see', 'the only explanation is'), or treating any disagreement as bad faith. One of these "
    "moves must be identifiably present - strength of feeling alone is not enough. (Absolute, "
    "unscoped, closed, contemptuous).\n"
    "- 'neutral': Everything else, and this is the largest class by far: chat, jokes, logistics, "
    "announcements, links, personal narrative, plain factual reports, ordinary questions, and any "
    "post that makes no contestable claim - as well as posts that gesture at nuance without "
    "naming any actual consideration, and opinionated posts that take a side, complain, or "
    "criticise without committing any of the dogmatic moves listed above.\n"
)

TEACHER_PROMPT_VIRTUE = (
    DEFINITIONS_VIRTUE
    + "\nAdditionally, if the label is 'virtuous', you must assess the Depth of the epistemic "
    "work on a 3-level scale:\n"
    "- Level 1 (Stated): the virtue is asserted but not demonstrated - a tentative tone, 'I "
    "could be wrong', 'I don't know enough about this', with no reasons or considerations "
    "given.\n"
    "- Level 2 (Demonstrated): at least one SPECIFIC counter-consideration, limitation, or "
    "competing value is named, and it actually bears on the claim ('the study was self-reported, "
    "so I only half trust it').\n"
    "- Level 3 (Worked through): named considerations are weighed against each other and a "
    "scoped conclusion is reached or explicitly declined - the strongest opposing case is "
    "engaged, the evidence that would settle it is named, or a stated prior position is "
    "revised.\n"
    "If the label is 'dogmatic' or 'neutral', always output 'Level: 1'.\n\n"
    "Additionally, you must rate the post's EPISTEMIC CARE on a single 0-9 Score. The Label and "
    "the Score answer different questions. The Label asks a threshold question: does this post "
    "cross into virtue or into dogmatism? The Score asks a continuous one: how carefully is the "
    "claim held, compared to every other post you see? Most posts do not cross either threshold, "
    "and they still differ enormously in care - that difference is exactly what the Score is "
    "for.\n"
    "- 0-1: unscoped blanket claims about groups, contempt or ridicule instead of argument, "
    "unfalsifiable framings, disagreement treated as bad faith.\n"
    "- 2-3: overgeneralization, certainty beyond the reasons given, single-cause explanation, "
    "motive attribution, dunking, or hedging theater.\n"
    "- 4: a claim or opinion asserted flatly, with no care shown and no dogmatic move committed "
    "either - or a post with nothing to be careful about at all (chat, logistics, a bare link, an "
    "announcement, a joke).\n"
    "- 5: a claim held with visible ordinary caution - a reason given for it, a factual report "
    "kept separate from the author's reading of it, a criticism scoped to a named actor and their "
    "specific actions.\n"
    "- 6: a TRACE of the virtue that falls short of the V threshold - a passing hedge that really "
    "does limit the claim, a small concession to the other side, an acknowledged limit ('for me "
    "anyway', 'at least in my corner of this'). Not virtuous enough to label V, but plainly more "
    "careful than a flat assertion.\n"
    "- 7: one specific limitation or counter-consideration is named, or the claim is explicitly "
    "scoped to the cases it covers.\n"
    "- 8: genuine weighing - two or more specific considerations traded off against each other, "
    "or the strongest opposing case engaged on its merits.\n"
    "- 9: self-implicating - a stated prior is revised, the evidence that would change the "
    "author's mind is named, or a scoped conclusion is reached after weighing considerations that "
    "cut against them.\n\n"
    "The Score is NOT a function of the Label and must never be clipped to fit it. A post you "
    "label N may score 6 or 7 when it shows real care that still falls short of the threshold - "
    "do NOT push it down to 4 or 5 just because the Label came out N. Equally, do not inflate a "
    "flat assertion to 6 because the topic is serious. Scores 8 and 9 describe posts that "
    "genuinely weigh or revise, and those will normally also be labelled V; 0 and 1 are reserved "
    "for the harshest dogmatism.\n"
    "Discriminate carefully in the 4-to-7 range, which is where most posts live: distinguishing a "
    "flat assertion (4) from a reasoned one (5) from one that concedes something (6) is the most "
    "useful judgement you make. Never default to the same digit for every post that is not "
    "obviously virtuous or dogmatic - a run of identical Scores means you have stopped "
    "discriminating.\n\n"
    "Think briefly inside " + "<think>" + "..." + "</think>" + ": at most two short sentences "
    "naming the dominant cue and your choice. Do not translate or restate the text. Do not state "
    "whether you agree with the author.\n"
    "After closing the think tag, output EXACTLY these three lines and nothing else:\n"
    "Label: X\n"
    "Level: Y\n"
    "Score: Z\n"
    "where X is a single letter V (virtuous), D (dogmatic), or N (neutral); Y is a single digit "
    "1, 2, or 3; and Z is a single digit 0-9."
)

VALIDATION_PROMPT_VIRTUE = (
    DEFINITIONS_VIRTUE
    + "You must respond with EXACTLY ONE WORD: 'virtuous', 'dogmatic', or 'neutral'. "
    "Do not include any explanations or punctuation."
)


def axis_labels():
    """Label set of the active axis: [positive, negative, neutral]."""
    v = prompt_version()
    if v == "assist":
        return LABELS_ASSIST
    if v == "virtue":
        return LABELS_VIRTUE
    return LABELS


def prompt_version():
    return os.environ.get("LOC_PROMPT_VERSION", "v1")


def get_teacher_prompt():
    v = prompt_version()
    if v == "virtue":
        return TEACHER_PROMPT_VIRTUE
    if v == "assist":
        return TEACHER_PROMPT_ASSIST
    if v == "v4":
        return TEACHER_PROMPT_V4
    if v == "v3":
        return TEACHER_PROMPT_V3
    return TEACHER_PROMPT_V2 if v == "v2" else TEACHER_PROMPT_V1


def get_validation_prompt():
    v = prompt_version()
    if v == "virtue":
        return VALIDATION_PROMPT_VIRTUE
    if v == "assist":
        return VALIDATION_PROMPT_ASSIST
    if v in ("v3", "v4"):
        return VALIDATION_PROMPT_V3
    return VALIDATION_PROMPT_V2 if v == "v2" else VALIDATION_PROMPT_V1
