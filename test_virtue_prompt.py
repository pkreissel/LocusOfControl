"""Run the virtue-axis edge cases through the llama.cpp server.

Each case targets one failure mode the definitions are written to prevent:
hedging theater, contentless nuance-signalling, the register trap (an
essayistic but closed post), scoped harshness that is NOT dogmatism, and
position-independence (the same epistemic conduct must score the same
whichever side it argues for).
"""
import asyncio
import os
import re

from openai import AsyncOpenAI
from label_prompts import get_teacher_prompt

API_BASE = os.environ.get("LOC_API_BASE", "http://localhost:8080/v1")
MODEL_NAME = os.environ.get("LOC_TEACHER_MODEL", "muse-glimmer-30B-kquant-17gb.gguf")

EDGE_CASES = [
    {
        # Real weighing: two named values in tension, traded off explicitly.
        "expected": "virtuous, L3, S8",
        "text": "I want the bike lane and I want my mum to be able to park near the surgery, and on Kirchstrasse you genuinely cannot have both. I lean lane because the crash numbers there are bad, but the people arguing for the parking aren't being NIMBYs, they just weigh the same street differently than I do.",
    },
    {
        # Strongest form: a stated prior is revised, with the reason for the update.
        "expected": "virtuous, L3, S9",
        "text": "I spent two years telling people remote work was strictly better and I was wrong about juniors. Watched three of them stall out with no one to overhear. Still think it's better for seniors, but I was generalising from my own situation and I should have noticed that.",
    },
    {
        # Virtue asserted only - tentative, no reasons. Should sit at the band floor.
        "expected": "virtuous, L1, S6",
        "text": "honestly i don't know enough about grid storage to have an opinion here, everything i've read contradicts everything else i've read",
    },
    {
        # Scoped self-knowledge: names the limit of the sample (n=1) instead of hiding it.
        "expected": "virtuous, L2, S7",
        "text": "my experience with the clinic was awful but that's one appointment with one doctor, i have no idea if it's the place or just a bad day. anyone else been?",
    },
    {
        # Genuine self-questioning question - virtuous because the author's own view is at stake.
        "expected": "virtuous, L2, S7",
        "text": "what's the strongest argument AGAINST rent caps? i keep only reading people who agree with me and it's making me stupid",
    },
    {
        # German, scoped and calibrated.
        "expected": "virtuous, L2, S7",
        "text": "Die Studie ist interessant, aber es sind 340 Leute aus einer einzigen Uni und alles selbst berichtet. Ich glaube den Trend, die Effektgroesse eher nicht.",
    },
    {
        # Hedging theater: the disclaimer decorates an unscoped blanket claim.
        "expected": "dogmatic, S2",
        "text": "I could be wrong but every single person who works in consulting has never produced one useful thing in their entire life. just my two cents obviously",
    },
    {
        # Blanket group claim + contempt in place of argument.
        "expected": "dogmatic, S0-1",
        "text": "there is no such thing as a good landlord. none. if you disagree you are a landlord or you want to be one, those are the only two options",
    },
    {
        # Register trap: long, calm, essayistic, cites a source - and completely closed.
        "expected": "dogmatic, S2",
        "text": "It is worth being precise about the mechanism here. As the 2019 literature establishes beyond serious dispute, the causal arrow runs in exactly one direction, and the persistence of the contrary view in public discourse can be explained entirely by the funding incentives of the institutions that promote it. There is simply no version of this where the sceptics are arguing in good faith.",
    },
    {
        # Motive attribution instead of engaging the argument.
        "expected": "dogmatic, S1",
        "text": "she's not making an argument, she's auditioning for a think tank job. that's the whole post. that's all it ever is with these people",
    },
    {
        # Anecdote generalised into a universal law.
        "expected": "dogmatic, S3",
        "text": "took the train once, it was 40 minutes late. this country cannot do infrastructure and never will be able to, it's just not in us",
    },
    {
        # Rhetorical question that asserts a blanket claim.
        "expected": "dogmatic, S2",
        "text": "how can anyone with a functioning brain still believe this in 2026?? genuinely asking",
    },
    {
        # Harsh but SCOPED to specific actors and documented actions - not dogmatism.
        "expected": "neutral, S5 (not dogmatic)",
        "text": "The filing says the three named directors signed off on the payment on 4 March and the minutes back it up. They should not be running a listed company.",
    },
    {
        # Nuance-signalling with nothing named - the classic false positive.
        "expected": "neutral, S5",
        "text": "as always the truth is somewhere in the middle here. it's a complex issue with many valid perspectives and i think we'd all do well to remember that",
    },
    {
        # Hyperbole as a bit - not a serious claim about a group.
        "expected": "neutral, S5",
        "text": "everyone who puts pineapple on pizza should be tried at the Hague. i don't make the rules",
    },
    {
        # Plain factual report, no contestable claim.
        "expected": "neutral, S5",
        "text": "Bahnstrecke zwischen Koeln und Duesseldorf ab Montag wegen Bauarbeiten gesperrt, Ersatzverkehr mit Bussen.",
    },
    # --- Regression cases from the 62-post base-rate probe (2026-09-07). All six
    # were labelled 'dogmatic' by the pre-guideline-14 prompt; only the last one
    # should be. They guard the dogmatic/neutral boundary against the axis
    # degenerating into a heated-politics detector.
    {
        # hedged relay of a rumour, not the author's own claim (was D/S2)
        "expected": "neutral",
        "text": "Apparently they snuck POTUS off the Qatari plane in the food truck...\nthen after the decoy plane landed they snuck him back on there as if nothing happened, so he could be seen deplaning.",
    },
    {
        # comic hyperbole (rule 11), not a serious group claim (was D/S2)
        "expected": "neutral",
        "text": "I would seriously trust my 17 yr old to better run this country, and I’m not even sure he can name all 3 branches of government. 🤣🤣\n\nBut he’s not delusional, doesn’t cheat, and would probably seek advice from advisors, so there’s that.\n\nMeanwhile, I guess we’re safe from the lady who laughs funny. 🤦🏻‍♂️",
    },
    {
        # taste claim / enthusiasm, not an epistemic failure (was D/S2)
        "expected": "neutral",
        "text": "Ich binge immer noch ER. Und das ist immer noch eine der besten TV-Serien, die je produziert wurden. Nichts daran fühlt sich so alt an wie es ist, im Gegensatz zu vielen anderen Serien, die in den 90ern gestartet sind. \nTrotz der fehlenden Smartphones und der prähistorischen Computer...",
    },
    {
        # plain factual report, however damning the selection (was D/S2)
        "expected": "neutral",
        "text": "In 2025, the CDC removed cyclospora from a program that actively tracks food-borne pathogens.\n\nUnder Trump, the CDC has also failed to implement a food traceability rule that requires barcodes that allow public health officials to track it to its source within 24 hours. \n\nA truly shit regime.",
    },
    {
        # complaint, no dogmatic move present (was D/S2)
        "expected": "neutral",
        "text": "Shit’s too expensive, and America’s CEOs just gave themselves a huge raise.",
    },
    {
        # unscoped moral absolute - must STAY dogmatic after the fix (was D/S2)
        "expected": "dogmatic",
        "text": "I see Luigi Mangione as a hero!   The wealth of insurance companies is appalling.\n\nThe system needs change & being nice isn’t working.\n\nFree article, no paywall.\n\nwww.nytimes.com/live/2026/08...",
    },
]

# Think-tag tokens used by the prompt, built from char codes to avoid
# the tooling interpreting them as markup delimiters.
OPEN_TAG = chr(60) + "think" + chr(62)      # <think>
CLOSE_TAG = chr(60) + "/think" + chr(62)    # </think>


def parse_response(content):
    label = level = score = None
    after = content.rsplit(CLOSE_TAG, 1)[-1]
    m = re.search(r"Label:\s*([A-Za-z])", after)
    if m:
        label = m.group(1).upper()
    m = re.search(r"Level:\s*([1-3])", after)
    if m:
        level = m.group(1)
    m = re.search(r"Score:\s*([0-9])", after)
    if m:
        score = m.group(1)
    return label, level, score


async def classify(client, post, system_prompt, sem):
    async with sem:
        messages = [
            {"role": "system", "content": system_prompt + "\nReasoning Strength: high"},
            {"role": "user", "content": f"Text: {post['text']}"},
        ]
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                top_p=0.95,
                max_tokens=4096,
                logprobs=True,
                top_logprobs=10,
            )
            content = completion.choices[0].message.content or ""
            label, level, score = parse_response(content)
            return content, label, level, score
        except Exception as e:
            return f"ERROR: {e}", None, None, None


async def main():
    os.environ["LOC_PROMPT_VERSION"] = "virtue"
    system_prompt = get_teacher_prompt()
    client = AsyncOpenAI(base_url=API_BASE, api_key="test-key")
    sem = asyncio.Semaphore(2)

    tasks = [asyncio.create_task(classify(client, p, system_prompt, sem)) for p in EDGE_CASES]

    for i, (p, fut) in enumerate(zip(EDGE_CASES, tasks)):
        content, label, level, score = await fut
        print(f"=== POST {i} === expected: {p['expected']}", flush=True)
        print(f"  got: Label={label} Level={level} Score={score}", flush=True)
        o = content.find(OPEN_TAG)
        c = content.find(CLOSE_TAG)
        if 0 <= o < c:
            print(f"  reasoning: {content[o+len(OPEN_TAG):c].strip()[:200]}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())
