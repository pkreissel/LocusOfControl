"""Generate synthetic counterexamples around the vector's decision boundary.

Hard posts = training posts the new vector misranks against the teacher:
  lures  - teacher says non-internal, vector scores in the top percentiles
  misses - teacher says internal, vector scores in the bottom percentiles
Each hard post is rewritten by Gemini to the OPPOSITE side of the boundary
(same topic/language/length, flipped psychological framing), then the rewrite
is teacher-annotated locally (LM Studio, v2 prompt) for real soft labels.
Gemini handles the constrained creative rewrite (local reasoning models
overthink it into token-limit failures); the local teacher does the labeling.

Output: dataset_synthetic_counterexamples.jsonl (same schema as teacher files,
uri suffixed with #synth). Resumable. Needs GAI_API in .env.

Run: uv run --with numpy --with openai --with google-genai --with python-dotenv \
     --with tqdm python create_synthetic_counterexamples.py
"""

import json
import os
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from tqdm import tqdm

from label_prompts import DEFINITIONS_V2, LABELS, LETTER_TO_LABEL, TEACHER_PROMPT_V2

TEACHER_FILE = os.environ.get("LOC_TEACHER_FILE", "dataset_annotated_bonsai_v2.jsonl")
EMB_CACHE = os.environ.get("LOC_EMB_CACHE", "embeddings_bonsai_jina.npz")
VECTOR = os.environ.get("LOC_VECTOR", "internal_vector_bonsai.npy")
OUT = os.environ.get("LOC_SYNTH_OUT", "dataset_synthetic_counterexamples.jsonl")
MODEL = os.environ.get("LOC_TEACHER_MODEL", "ternary-bonsai-27b-mlx")
N_PER_SIDE = int(os.environ.get("LOC_SYNTH_PER_SIDE", "100"))

REWRITE_PROMPT = (
    "You are an expert psycholinguist and writer. Rewrite the given social media post so "
    "that its Locus-of-Control framing becomes '{target}' under these definitions:\n\n"
    + DEFINITIONS_V2 +
    "\nHard constraints:\n"
    "- Same core topic and same language (German stays German, English stays English).\n"
    "- Same length as the original, within about 20 percent. Do not pad.\n"
    "- Preserve the author's casual social-media voice exactly: keep hashtags, emoji, "
    "slang, lowercase quirks, imperfect punctuation. The rewrite must read like a real "
    "post by the same person, NOT like polished prose.\n"
    "- Change as little as possible - flip only the framing of control and agency. "
    "Avoid therapy-speak and stock phrases like 'conscious choice', 'deliberate', "
    "'I chose to', 'embraced'. Subtle wording shifts beat added sentences.\n"
    "Reply with ONLY the rewritten post text - no explanations, no quotation marks."
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def p_internal(raw_logprobs):
    logits = np.array([raw_logprobs[l] for l in LABELS])
    e = np.exp(logits - logits.max())
    return e[0] / e.sum()


def call(client, system, user, max_tokens=2000):
    c = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        logprobs=True, top_logprobs=10, temperature=0.4, max_tokens=max_tokens)
    return c


def teacher_annotate(client, text):
    for attempt in range(4):
        c = call(client, TEACHER_PROMPT_V2, f"Text: {text}")
        content = c.choices[0].logprobs.content if c.choices[0].logprobs else None
        if not content:
            continue
        idx = next((i for i in range(len(content) - 1, -1, -1)
                    if LETTER_TO_LABEL.get(content[i].token.strip().lower().rstrip("."))), None)
        if idx is None:
            continue
        lp = {l: -20.0 for l in LABELS}
        for item in content[idx].top_logprobs:
            label = LETTER_TO_LABEL.get(item.token.strip().lower().rstrip("."))
            if label:
                lp[label] = max(lp[label], item.logprob)
        probs = {k: np.exp(v) for k, v in lp.items()}
        total = sum(probs.values())
        return lp, [probs[l] / total for l in LABELS], max(lp, key=lp.get)
    raise ValueError("annotation failed after retries")


def main():
    val_uris = {r["uri"] for r in load_jsonl("dataset_validation_gemini.jsonl")}
    rows = [r for r in load_jsonl(TEACHER_FILE)
            if r.get("text") and r.get("raw_logprobs") and r["uri"] not in val_uris]
    X = np.nan_to_num(np.load(EMB_CACHE, allow_pickle=True)["X"])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert len(rows) == len(X)
    v = np.load(VECTOR)
    cos = X @ v
    pct = np.argsort(np.argsort(cos)) / (len(cos) - 1)
    p = np.array([p_internal(r["raw_logprobs"]) for r in rows])

    lure_idx = [i for i in np.argsort(-(pct - p)) if p[i] < 0.3 and pct[i] > 0.7][:N_PER_SIDE]
    miss_idx = [i for i in np.argsort(pct - p) if p[i] > 0.7 and pct[i] < 0.3][:N_PER_SIDE]
    hard = ([(i, "internal") for i in lure_idx]      # lures -> rewrite TO internal
            + [(i, "external") for i in miss_idx])   # missed internals -> rewrite TO external
    print(f"hard posts: {len(lure_idx)} lures, {len(miss_idx)} misses")

    done = {r["uri"] for r in load_jsonl(OUT)} if os.path.exists(OUT) else set()
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    load_dotenv()
    gclient = genai.Client(api_key=os.environ["GAI_API"])
    with open(OUT, "a", encoding="utf-8") as f:
        for i, target in tqdm(hard, desc="Generating counterexamples"):
            uri = rows[i]["uri"] + f"#synth-{target}"
            if uri in done:
                continue
            try:
                time.sleep(4.1)  # Gemini free tier: 15 RPM
                r = gclient.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=f"Post:\n{rows[i]['text'][:2000]}",
                    config=types.GenerateContentConfig(
                        system_instruction=REWRITE_PROMPT.format(target=target),
                        temperature=0.7),
                )
                new_text = (r.text or "").strip()
                if not new_text or len(new_text) < 10:
                    raise ValueError("empty rewrite")
                lp, soft, label = teacher_annotate(client, new_text)
                f.write(json.dumps({
                    "uri": uri, "text": new_text, "source_uri": rows[i]["uri"],
                    "source_text": rows[i]["text"], "rewrite_target": target,
                    "raw_logprobs": lp, "soft_labels": soft, "loc_label": label,
                }) + "\n")
                f.flush()
            except Exception as e:
                tqdm.write(f"error on {uri}: {e}")

    recs = load_jsonl(OUT)
    from collections import Counter
    hit = sum(1 for r in recs if r["loc_label"] == r["rewrite_target"])
    print(f"\n{len(recs)} counterexamples; teacher label matches rewrite target: "
          f"{hit}/{len(recs)}")
    print("teacher labels:", dict(Counter(r["loc_label"] for r in recs)))


if __name__ == "__main__":
    main()
