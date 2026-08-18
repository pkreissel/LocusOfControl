"""Generate synthetic counterexamples around the V3 vector's decision boundary.

Hard posts = training posts the new vector misranks against the teacher:
  lures  - teacher says non-internal, vector scores in the top percentiles
  misses - teacher says internal, vector scores in the bottom percentiles
Each hard post is rewritten by Gemini to the OPPOSITE side of the boundary
(same topic/language/length, flipped psychological framing), then the rewrite
is teacher-annotated locally (llama.cpp, v3 prompt) for real soft labels and levels.

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

from label_prompts import DEFINITIONS_V3, LABELS, LETTER_TO_LABEL, TEACHER_PROMPT_V3, prompt_version

TEACHER_FILE = os.environ.get("LOC_TEACHER_FILE", "dataset_annotated_lmstudio_soft.jsonl")
EMB_CACHE = os.environ.get("LOC_EMB_CACHE", "embeddings_lmstudio_soft_v3_jina.npz")
VECTOR = os.environ.get("LOC_VECTOR", "internal_vector.npy")
OUT = os.environ.get("LOC_SYNTH_OUT", "dataset_synthetic_counterexamples.jsonl")
MODEL = os.environ.get("LOC_TEACHER_MODEL", "meta/muse-glimmer")
API_BASE = os.environ.get("LOC_API_BASE", "http://127.0.0.1:8080/v1")
N_PER_SIDE = int(os.environ.get("LOC_SYNTH_PER_SIDE", "100"))

REWRITE_PROMPT = (
    "You are an expert psycholinguist and writer. Rewrite the given social media post so "
    "that its Locus-of-Control framing becomes '{target}' under these definitions:\n\n"
    + DEFINITIONS_V3 +
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


def get_target_score(raw_logprobs, level_logprobs):
    logits = np.array([raw_logprobs.get(l, -20.0) for l in LABELS])
    e = np.exp(logits - logits.max())
    p_int = float(e[0] / e.sum())
    
    if not level_logprobs:
        exp_lvl = 1.0
    else:
        logits = np.array([level_logprobs.get(str(i), -20.0) for i in (1, 2, 3)])
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        exp_lvl = float(probs[0]*1 + probs[1]*2 + probs[2]*3)
        
    return p_int * exp_lvl


def call(client, system, user, max_tokens=2000):
    c = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system + "\nReasoning Strength: high"},
                  {"role": "user", "content": user}],
        logprobs=True, top_logprobs=10, temperature=1.0, top_p=0.95, max_tokens=max_tokens)
    return c


def token_to_label(token):
    t = token.strip().lower().replace(".", "").replace("\n", "")
    return LETTER_TO_LABEL.get(t)

def token_to_level(token):
    t = token.strip().replace(".", "").replace("\n", "")
    if t in ["1", "2", "3"]:
        return int(t)
    return None


def teacher_annotate(client, text):
    for attempt in range(4):
        c = call(client, TEACHER_PROMPT_V3, f"Text: {text}")
        content = c.choices[0].logprobs.content if c.choices[0].logprobs else None
        if not content:
            continue
            
        label_idx = None
        level_idx = None

        for idx in range(len(content) - 1, -1, -1):
            if level_idx is None and token_to_level(content[idx].token) is not None:
                level_idx = idx
            elif label_idx is None and token_to_label(content[idx].token) is not None:
                label_idx = idx
            if label_idx is not None and level_idx is not None:
                break
                
        if label_idx is None:
            continue
            
        class_logprobs = {"internal": -20.0, "external": -20.0, "neutral": -20.0}
        level_logprobs = {"1": -20.0, "2": -20.0, "3": -20.0}

        decision_token_obj = content[label_idx]
        for item in decision_token_obj.top_logprobs:
            label = token_to_label(item.token)
            if label is not None:
                class_logprobs[label] = max(class_logprobs[label], item.logprob)
                
        if level_idx is not None:
            level_token_obj = content[level_idx]
            for item in level_token_obj.top_logprobs:
                level = token_to_level(item.token)
                if level is not None:
                    level_logprobs[str(level)] = max(level_logprobs[str(level)], item.logprob)
                    
        probs = {k: np.exp(v) for k, v in class_logprobs.items()}
        total_prob = sum(probs.values())
        if total_prob > 0:
            normalized_probs = {k: (v / total_prob) for k, v in probs.items()}
        else:
            normalized_probs = {"internal": 0.33, "external": 0.33, "neutral": 0.34}
            
        level_probs = {k: np.exp(v) for k, v in level_logprobs.items()}
        total_level_prob = sum(level_probs.values())
        if total_level_prob > 0:
            normalized_level_probs = {k: (v / total_level_prob) for k, v in level_probs.items()}
        else:
            normalized_level_probs = {"1": 1.0, "2": 0.0, "3": 0.0}
            
        soft_labels = [normalized_probs["internal"], normalized_probs["external"], normalized_probs["neutral"]]
        loc_label = max(normalized_probs, key=normalized_probs.get)
        soft_levels = [normalized_level_probs["1"], normalized_level_probs["2"], normalized_level_probs["3"]]
        loc_level = int(max(normalized_level_probs, key=normalized_level_probs.get))
        
        return class_logprobs, soft_labels, loc_label, level_logprobs, soft_levels, loc_level

    raise ValueError("annotation failed after retries")


def main():
    val_uris = {r["uri"] for r in load_jsonl("dataset_validation_gemini.jsonl")}
    rows = [r for r in load_jsonl(TEACHER_FILE)
            if r.get("text") and r.get("raw_logprobs") and r["uri"] not in val_uris]
            
    # Load V3 Jina Embeddings
    X = np.nan_to_num(np.load(EMB_CACHE, allow_pickle=True)["X"])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert len(rows) == len(X), "Mismatch between rows and embeddings"
    
    # Load V3 Vector
    v = np.load(VECTOR)
    cos = X @ v
    
    # Calculate Teacher Target Score
    p = np.array([get_target_score(r["raw_logprobs"], r.get("level_logprobs")) for r in rows])
    
    # Convert cos and p to percentiles for scale-invariant comparison
    cos_pct = np.argsort(np.argsort(cos)) / (len(cos) - 1)
    p_pct = np.argsort(np.argsort(p)) / (len(p) - 1)

    # Lures: Teacher says low (bottom 30%), Vector says high (top 30%)
    lure_idx = [i for i in np.argsort(-(cos_pct - p_pct)) if p_pct[i] < 0.3 and cos_pct[i] > 0.7][:N_PER_SIDE]
    
    # Misses: Teacher says high (top 30%), Vector says low (bottom 30%)
    miss_idx = [i for i in np.argsort(cos_pct - p_pct) if p_pct[i] > 0.7 and cos_pct[i] < 0.3][:N_PER_SIDE]
    
    hard = ([(i, "internal") for i in lure_idx]      # lures -> rewrite TO internal
            + [(i, "external") for i in miss_idx])   # missed internals -> rewrite TO external
    print(f"hard posts: {len(lure_idx)} lures, {len(miss_idx)} misses")

    done = {r["uri"] for r in load_jsonl(OUT)} if os.path.exists(OUT) else set()
    client = OpenAI(base_url=API_BASE, api_key="test-key")
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
                    
                # Score the new text using local llama.cpp V3 model
                lp, soft, label, lvl_lp, soft_lvl, loc_lvl = teacher_annotate(client, new_text)
                
                f.write(json.dumps({
                    "uri": uri, "text": new_text, "source_uri": rows[i]["uri"],
                    "source_text": rows[i]["text"], "rewrite_target": target,
                    "raw_logprobs": lp, "soft_labels": soft, "loc_label": label,
                    "level_logprobs": lvl_lp, "soft_levels": soft_lvl, "loc_level": loc_lvl
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
