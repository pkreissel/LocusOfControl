"""Run 10 edge-case posts through the llama.cpp server with the assist prompt."""
import asyncio
import os
import re

from openai import AsyncOpenAI
from label_prompts import get_teacher_prompt

API_BASE = os.environ.get("LOC_API_BASE", "http://localhost:8080/v1")
MODEL_NAME = os.environ.get("LOC_TEACHER_MODEL", "muse-glimmer-30B-kquant-17gb.gguf")

EDGE_CASES = [
    {
        "expected": "helpful, L3, S8",
        "text": "help me figure out what's going on in the photos from a house listing. these are identical shots, staged vs. non-staged. the camera doesn't move so i don't think a person took both photos. i'd guess AI but does AI know how to put a kurt vile record on the wall, or a neil young book on the shelf?",
    },
    {
        "expected": "helpful, L3, S8",
        "text": "Does anyone have a good, accessible, well-researched article documenting how the NYTimes contributed to anti-trans backlash? I'm teaching a credit-bearing info lit class this fall, and since the standard syllabus includes reading the Times, I'd like something that pushes back.",
    },
    {
        "expected": "helpful, L2, S7",
        "text": "Weird question: anyone know any artist studios doing teardown art of laptops?\n\nI'm thinking of something like GridStudio or Xreart, but bigger than phones / game controllers",
    },
    {
        "expected": "helpful, L2, S7",
        "text": "two great job offers dunno which to take, how do i decide? No pros and cons spreadsheet my mother has suggested that for the last 20 yrs and i'm doing great refusing her good idea",
    },
    {
        "expected": "inert, S2",
        "text": "how to get ai to ballmer peak\ncan claude drink\nhow to put alcohol in computer which port",
    },
    {
        "expected": "inert, S1",
        "text": "Why am I getting ANOTHER emergency alert at 10.45 at night????",
    },
    {
        "expected": "inert, S2",
        "text": "Does anyone in this admin play cards? Or are they just too scared to point out this obvious fact so they just let the president keep owning himself?",
    },
    {
        "expected": "neutral, S4",
        "text": 'People always ask "how do I buy your in a way that gets the most money in your pocket?" I get paid a flat rate. Buy it wherever.',
    },
    {
        "expected": "neutral, S4",
        "text": "Anyway, now that I've calmed down,  does anyone fancy coming over for my bbq tomorrow? We're all setting off lanterns and then there'll be a fireworks display.",
    },
    {
        "expected": "neutral/inert, S4",
        "text": "My book is now at #22! Can you help me break into the top 20 by grabbing a copy? If you've enjoyed my content over the last decade (which never has and never will be paywalled), please support my work here: www.harpercollins.com/pages/theday...",
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
    os.environ["LOC_PROMPT_VERSION"] = "assist"
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
