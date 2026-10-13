"""The LoC rubric of ts_questions.py, rewritten to fit Laya's head budget.

Laya packs the instructions and every option into `head_max_len` tokens (256 on
laya-multilingual, shared by all options), and hard-truncates each option to 48
tokens. ts_questions.py was written for an API with a 1024-token state and no
such cap, so its options run 49-93 tokens and its score question leaves an
option budget of 256 - 5*48 = 16 tokens for the instructions. Fed verbatim,
Laya sees five levels with their tails cut off and a one-line instruction.

That is a property of the harness, not of the rubric, so it cannot be what the
model is judged on. Same axis, same five bands, same three classes, same noul,
compressed until nothing is clipped -- `check()` asserts it. The numbers from
this file are Laya's best case; the numbers from ts_questions.py are what a
drop-in replacement of the TypeSafe call would actually produce, and
eval_laya_vs_jev.py reports both.
"""

LEVELS = [
    "Strong external: helplessness, fatalism, victimhood, or dread of what "
    "nobody can control.",

    "Mild external: worry about distant events, demands that institutions act, "
    "or bad news with no way to act on it.",

    "Neutral: small talk, jokes, everyday life, bare links, routine news. No "
    "framing of control either way.",

    "Internal but abstract: general agency - an intention, a philosophy, or "
    "credit given to effort or strategy.",

    "Internal and concrete: agency plus a specific step taken, planned, or "
    "achieved.",
]

INSTRUCTIONS = (
    "Rate how this post affects a reader's Locus of Control: low means EXTERNAL "
    "(helplessness), high means INTERNAL (agency). Judge emotional intent, in "
    "any language."
)

CLASS_CRITERIA = {
    "internal": "Fosters agency: solutions, proactive steps the reader or "
                "ordinary people can take, ownership of one's own actions and "
                "outcomes, community action, success credited to effort or skill.",
    "external": "Fosters helplessness: passive victim of corporations, "
                "governments, or macro-events; waiting on remote authorities; "
                "anxiety or bad news with no way to act; outcomes credited to "
                "luck or fate.",
    "neutral":  "Small talk, jokes, greetings, everyday life, sport, food, bare "
                "links, routine news, or a plain report that a third party did "
                "something. No framing of control or agency.",
}

AGENCY_INSTRUCTIONS = (
    "The post leaves a reader feeling more able to act on their own life - it "
    "offers a proactive step, owns an outcome, or shows effort paying off - "
    "rather than more helpless in the face of forces beyond their control."
)


def questions():
    return {
        "loc_score": {"type": "score", "instructions": INSTRUCTIONS, "criteria": LEVELS},
        "loc_class": {"type": "choice",
                      "instructions": "Which Locus-of-Control class does this social media post "
                                      "put its reader in? Judge the emotional intent and subtext, "
                                      "in whatever language the post is written.",
                      "criteria": CLASS_CRITERIA},
        "agency": {"type": "noul", "instructions": AGENCY_INSTRUCTIONS},
    }


def check(agent, qs=None, verbose=True):
    """Assert nothing is clipped: every option under 48 tokens, and the
    instructions fitting the budget the options leave behind."""
    from laya.common import render_options
    head_max = agent.cfg.get("head_max_len", 192)
    ok = True
    for qid, q in (qs or questions()).items():
        qq = agent._to_internal(q)
        opts = [len(agent.tok(" " + o, add_special_tokens=False)["input_ids"])
                for o in render_options(qq)]
        ins = len(agent.tok("%s question: %s" % (qq["t"], qq["ins"]), add_special_tokens=False)["input_ids"])
        budget = head_max - sum(min(o, 48) for o in opts)
        fits = max(opts) <= 48 and ins <= budget
        ok &= fits
        if verbose:
            print(f"  {qid:10} options {opts} (cap 48)  instructions {ins} (budget {budget})  "
                  f"{'ok' if fits else 'CLIPPED'}")
    return ok
