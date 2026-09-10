"""Generate synthetic counterexamples around the V3 vector's decision boundary.

Hard posts = training posts the new vector misranks against the teacher:
  lures  - teacher says non-internal, vector scores in the top percentiles
  misses - teacher says internal, vector scores in the bottom percentiles
Each hard post is rewritten by Gemini to the OPPOSITE side of the boundary
(same topic/language/length, flipped psychological framing), then the rewrite
is teacher-annotated locally (llama.cpp, v3 prompt) for real soft labels and levels.

Output: dataset_synthetic_counterexamples.jsonl (same schema as teacher files,
uri suffixed with #synth). Resumable. Needs GAI_API in .env.

Before the first Gemini call the script prints a pre-flight report on the mined
hard set (composition, expected land rate, expected usable rows, a sample to
eyeball). It costs no API calls. LOC_SYNTH_PREFLIGHT controls it:
  report  (default) print the diagnosis and generate anyway
  gate    abort with exit code 2 if the lure side looks unrewritable
  only    print the diagnosis and exit 0 without generating
  off     skip it

Run: uv run --with numpy --with openai --with google-genai --with python-dotenv \
     --with tqdm python create_synthetic_counterexamples.py
"""

import json
import os
import re
import sys
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from tqdm import tqdm

from label_prompts import (DEFINITIONS_V3, DEFINITIONS_ASSIST, DEFINITIONS_VIRTUE, LABELS,
                           LETTER_TO_LABEL, LETTER_TO_LABEL_ASSIST, LETTER_TO_LABEL_VIRTUE,
                           TEACHER_PROMPT_V3, TEACHER_PROMPT_ASSIST, TEACHER_PROMPT_VIRTUE,
                           prompt_version)

TEACHER_FILE = os.environ.get("LOC_TEACHER_FILE", "dataset_annotated_lmstudio_soft.jsonl")
EMB_CACHE = os.environ.get("LOC_EMB_CACHE", "embeddings_lmstudio_soft_v3_jina.npz")
VECTOR = os.environ.get("LOC_VECTOR", "internal_vector.npy")
OUT = os.environ.get("LOC_SYNTH_OUT", "dataset_synthetic_counterexamples.jsonl")
MODEL = os.environ.get("LOC_TEACHER_MODEL", "meta/muse-glimmer")
API_BASE = os.environ.get("LOC_API_BASE", "http://127.0.0.1:8080/v1")
N_PER_SIDE = int(os.environ.get("LOC_SYNTH_PER_SIDE", "100"))
# Minimum spacing between Gemini calls (free tier: 15 RPM).
GEMINI_MIN_INTERVAL = 4.1
# report (default) = print the pre-flight and generate anyway; gate = abort if the
# lure side looks unrewritable; only = print and exit without generating; off = skip.
PREFLIGHT_MODE = os.environ.get("LOC_SYNTH_PREFLIGHT", "report").strip().lower()
PREFLIGHT_SAMPLE = int(os.environ.get("LOC_SYNTH_PREFLIGHT_SAMPLE", "8"))
# Judgment call, NOT a validated cutoff. Report section 8 lists lure-side agent
# pronoun rate among the three signals that FAILED to discriminate: 54% on the Jina
# run that gained +0.020 AUC, 47% on the MiniLM run that gained +0.001. No threshold
# can separate those. Two further reasons not to read this as a decision rule: the
# narrow pronoun list below scores that same MiniLM lure set at 34%, not 47%, so the
# absolute number moves with the inventory used; and the corpus base rate it should
# be read against moves with the corpus (29% here, 31% in the report).
# 0.25 is therefore set *below the corpus base rate*, so `gate` only fires when the
# mined lures are actively depleted of first person relative to the corpus at large -
# the signature of the category error in section 7.1 (agentic vocabulary in agent-free
# text), where rewriting has nothing to work with. It is a tripwire, not a decision.
PREFLIGHT_MIN_PRONOUN = float(os.environ.get("LOC_SYNTH_PREFLIGHT_MIN_PRONOUN", "0.25"))

# Measured land rates (teacher label == rewrite target) from the 199-row MiniLM run
# and the Jina run before it: lures carrying a first-person agent 85%, lures without
# one 68%, misses 95% either way. The 68% failures come back labelled `neutral` -
# near-duplicates of their source that dilute the training set rather than teach it.
LAND_LURE_AGENT = 0.85
LAND_LURE_NO_AGENT = 0.68
LAND_MISS = 0.95

# First-person agent pronouns, English and German. Deliberately narrow: this is the
# "is there anybody in this post to give agency to" probe from report section 7.1,
# not a full pronoun inventory.
AGENT_PRONOUNS = re.compile(
    r"\b(i|my|me|we|our|us|ich|mir|mein|meine|wir|uns|unser)\b", re.IGNORECASE)


# Detect if we are in assist mode
if prompt_version() == "assist":
    USE_ASSIST = True
    _LABELS = ["helpful", "inert", "neutral"]
    _POSITIVE = "helpful"
    _NEGATIVE = "inert"
    _DEFS = DEFINITIONS_ASSIST
    _T_PROMPT = TEACHER_PROMPT_ASSIST
    _L2L = LETTER_TO_LABEL_ASSIST
elif prompt_version() == "virtue":
    USE_ASSIST = True
    _LABELS = ["virtuous", "dogmatic", "neutral"]
    _POSITIVE = "virtuous"
    _NEGATIVE = "dogmatic"
    _DEFS = DEFINITIONS_VIRTUE
    _T_PROMPT = TEACHER_PROMPT_VIRTUE
    _L2L = LETTER_TO_LABEL_VIRTUE
else:
    USE_ASSIST = False
    _LABELS = LABELS
    _POSITIVE = "internal"
    _NEGATIVE = "external"
    _DEFS = DEFINITIONS_V3
    _T_PROMPT = TEACHER_PROMPT_V3
    _L2L = LETTER_TO_LABEL


REWRITE_PROMPT = (
    "You are an expert psycholinguist and writer. Rewrite the given social media post so "
    "that its framing on the labelling axis below becomes '{target}' under these definitions:"
    "\n\n"
    + _DEFS +
    "\nHard constraints:\n"
    "- Same core topic and same language (German stays German, English stays English).\n"
    "- Same length as the original, within about 20 percent. Do not pad.\n"
    "- Preserve the author's casual social-media voice exactly: keep hashtags, emoji, "
    "slang, lowercase quirks, imperfect punctuation. The rewrite must read like a real "
    "post by the same person, NOT like polished prose.\n"
    "- Change as little as possible - flip only the framing the axis measures (for the "
    "virtue axis: how the claim is held - scope, hedging, weighing - NOT the position taken, "
    "the topic, or the vocabulary level). "
    "Avoid therapy-speak and stock phrases like 'conscious choice', 'deliberate', "
    "'I chose to', 'embraced'. Subtle wording shifts beat added sentences.\n"
    "Reply with ONLY the rewritten post text - no explanations, no quotation marks."
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_target_score(raw_logprobs, level_logprobs):
    logits = np.array([raw_logprobs.get(l, -20.0) for l in _LABELS])
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
    return _L2L.get(t) if USE_ASSIST else LETTER_TO_LABEL.get(t)

def token_to_level(token):
    t = token.strip().replace(".", "").replace("\n", "")
    if t in ["1", "2", "3"]:
        return int(t)
    return None


def teacher_annotate(client, text):
    for attempt in range(4):
        c = call(client, _T_PROMPT, f"Text: {text}")
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
            
        class_logprobs = {lbl: -20.0 for lbl in _LABELS}
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
            normalized_probs = {_POSITIVE: 0.33, _NEGATIVE: 0.33, "neutral": 0.34}
            
        level_probs = {k: np.exp(v) for k, v in level_logprobs.items()}
        total_level_prob = sum(level_probs.values())
        if total_level_prob > 0:
            normalized_level_probs = {k: (v / total_level_prob) for k, v in level_probs.items()}
        else:
            normalized_level_probs = {"1": 1.0, "2": 0.0, "3": 0.0}
            
        soft_labels = [normalized_probs.get(_POSITIVE, 0), normalized_probs.get(_NEGATIVE, 0),
                       normalized_probs.get("neutral", 0)]
        loc_label = max(normalized_probs, key=normalized_probs.get)
        soft_levels = [normalized_level_probs["1"], normalized_level_probs["2"], normalized_level_probs["3"]]
        loc_level = int(max(normalized_level_probs, key=normalized_level_probs.get))
        
        return class_logprobs, soft_labels, loc_label, level_logprobs, soft_levels, loc_level

    raise ValueError("annotation failed after retries")


def has_agent(text):
    return bool(AGENT_PRONOUNS.search(text or ""))


def preflight(rows, lure_idx, miss_idx, budget):
    """Print what the mined hard set looks like before any Gemini call is made.

    Everything here is computed from data already in memory, so it costs nothing.
    It answers the one question report section 8 says to ask before spending quota:
    could these posts plausibly have been written the other way? There is no
    validated cheap predictor of payoff - the three that were tested (upweighting the
    mined hard posts, corpus-level agent-pronoun rate, cross-space error overlap) all
    failed to separate the Jina run that gained +0.020 AUC from the MiniLM run that
    gained +0.001. What is reported below is composition and expected yield, which
    bound how many rows are usable; they do NOT predict whether the rows help.

    Returns the lure-side agent-pronoun fraction so the caller can gate on it.
    """
    base_rate = sum(has_agent(r["text"]) for r in rows) / max(len(rows), 1)
    lure_agent = [i for i in lure_idx if has_agent(rows[i]["text"])]
    lure_frac = len(lure_agent) / len(lure_idx) if lure_idx else 0.0

    print("\npre-flight (no API calls; report section 8)")
    print(f"  corpus base rate, first-person agent pronoun: {base_rate:.0%} "
          f"of {len(rows)} posts")
    print(f"  lures  with agent pronoun: {len(lure_agent)}/{len(lure_idx)} "
          f"({lure_frac:.0%}, {100 * (lure_frac - base_rate):+.0f}pt vs corpus)")
    print(f"  misses with agent pronoun: "
          f"{sum(1 for i in miss_idx if has_agent(rows[i]['text']))}/{len(miss_idx)} "
          f"(not predictive: the miss side lands {LAND_MISS:.0%} either way)")

    # Expected usable rows: the lure side splits by agent pronoun, the miss side does
    # not. Scale to the remaining budget, since the run is resumable.
    n_hard = len(lure_idx) + len(miss_idx)
    exp_land = (len(lure_agent) * LAND_LURE_AGENT
                + (len(lure_idx) - len(lure_agent)) * LAND_LURE_NO_AGENT
                + len(miss_idx) * LAND_MISS)
    exp_rate = exp_land / n_hard if n_hard else 0.0
    print(f"  expected land rate {exp_rate:.0%} "
          f"(lure {LAND_LURE_AGENT:.0%} with agent / {LAND_LURE_NO_AGENT:.0%} without, "
          f"miss {LAND_MISS:.0%})")
    print(f"  expected usable rows: ~{round(exp_rate * budget)} of {budget} to generate "
          f"(~{budget * 28 / 60:.0f} min, {budget} Gemini calls; free tier is 500/day)")
    if USE_ASSIST:
        print("  note: those land rates were measured on the LoC axis; on this axis "
              "they are an extrapolation")

    print("\n  sample of the mined hard set - ask of each: could this post plausibly "
          "have been written the other way?")
    # Interleave the two sides and stride through each so the sample is not just the
    # most extreme rows; section 8 says to eyeball ~20, this prints a starting handful.
    sample = []
    for side, idx in (("lure", lure_idx), ("miss", miss_idx)):
        take = max(PREFLIGHT_SAMPLE // 2, 1)
        step = max(len(idx) // take, 1) if idx else 1
        sample += [(side, i) for i in idx[::step][:take]]
    for side, i in sample:
        text = " ".join((rows[i]["text"] or "").split())
        agent = "agent" if has_agent(text) else "no-agent"
        print(f"    [{side}/{agent:8s}] {text[:150]}")

    print("\n  caveats that no pre-flight number can cover:")
    print("    - payoff tracked baseline headroom, not composition: Jina at val AUC "
          "0.772 gained +0.020, MiniLM at 0.795 gained +0.001. Ask first whether this "
          "probe is visibly worse than a sibling probe on the same data.")
    print("    - rewrite yield does not predict payoff: the MiniLM batch landed 84% "
          "and still moved AUC by +0.001.")
    print("    - after retraining, check the external-leak column, not just AUC. Every "
          "counterexample run moved external suppression, twice for the worse "
          "(+0.036, +0.061, P(worse) = 1.000).")
    print("    - counterexamples never transfer between embedding spaces or axes; "
          "these rows are only valid for " + os.path.basename(VECTOR) + ".")

    if lure_frac < PREFLIGHT_MIN_PRONOUN:
        verdict = (f"VERDICT: lure side looks unrewritable - only {lure_frac:.0%} of the "
                   f"lures have a first-person agent to make agentic. A lure set of "
                   f"headlines and product announcements means a category error "
                   f"(agentic vocabulary in agent-free text); the fix is more real "
                   f"negatives of that category, not rewrites of them.")
    else:
        verdict = (f"VERDICT: lure side is rewritable in principle ({lure_frac:.0%} carry "
                   f"an agent pronoun) - expect ~{round(exp_rate * budget)} usable rows. "
                   f"This bounds yield, not payoff; headroom decides payoff.")
        if lure_frac <= base_rate:
            verdict += (" Note the lures are no more first-person than the corpus, so "
                        "the sample above is doing the real work here.")
    print("\n  " + verdict)
    return lure_frac


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
    
    hard = ([(i, _POSITIVE) for i in lure_idx]      # lures -> rewrite TO positive
            + [(i, _NEGATIVE) for i in miss_idx])   # missed positives -> rewrite TO negative
    print(f"hard posts: {len(lure_idx)} lures, {len(miss_idx)} misses")

    done = {r["uri"] for r in load_jsonl(OUT)} if os.path.exists(OUT) else set()

    if PREFLIGHT_MODE != "off":
        remaining = sum(1 for i, t in hard if rows[i]["uri"] + f"#synth-{t}" not in done)
        lure_frac = preflight(rows, lure_idx, miss_idx, remaining)
        if PREFLIGHT_MODE == "only":
            return
        if PREFLIGHT_MODE == "gate" and lure_frac < PREFLIGHT_MIN_PRONOUN:
            print("\naborting: LOC_SYNTH_PREFLIGHT=gate and the lure side is below the "
                  f"{PREFLIGHT_MIN_PRONOUN:.0%} agent-pronoun tripwire. Set "
                  "LOC_SYNTH_PREFLIGHT=report to generate anyway.")
            sys.exit(2)

    last_gemini = 0.0  # monotonic timestamp of the last Gemini call; see the pacing below
    client = OpenAI(base_url=API_BASE, api_key="test-key")
    load_dotenv()
    gclient = genai.Client(api_key=os.environ["GAI_API"])
    with open(OUT, "a", encoding="utf-8") as f:
        for i, target in tqdm(hard, desc="Generating counterexamples"):
            uri = rows[i]["uri"] + f"#synth-{target}"
            if uri in done:
                continue
            try:
                # Gemini's free tier allows 15 RPM. Pace against the *previous* call
                # rather than sleeping unconditionally: teacher_annotate() below runs
                # a 30B model locally and already holds this loop near 2 RPM, so a
                # fixed 4.1s sleep was pure dead time (~14% of the run). It still
                # blocks properly if the teacher ever gets fast enough to matter.
                wait = GEMINI_MIN_INTERVAL - (time.monotonic() - last_gemini)
                if wait > 0:
                    time.sleep(wait)
                last_gemini = time.monotonic()
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
