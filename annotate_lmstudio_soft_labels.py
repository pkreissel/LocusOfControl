import asyncio
import json
import os
import time
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from label_prompts import (LETTER_TO_LABEL, LETTER_TO_LABEL_ASSIST, LETTER_TO_LABEL_VIRTUE,
                           axis_labels, get_teacher_prompt, prompt_version)

INPUT_FILE = os.environ.get("LOC_INPUT_FILE", "dataset_raw.jsonl")
OUTPUT_FILE = os.environ.get("LOC_OUTPUT_FILE", "dataset_annotated_lmstudio_soft.jsonl")
API_BASE = os.environ.get("LOC_API_BASE", "http://localhost:8080/v1")
MAX_CONCURRENT = int(os.environ.get("LOC_MAX_CONCURRENT", "4"))
MODEL_NAME = os.environ.get("LOC_TEACHER_MODEL", "meta/muse-glimmer")
TOP_LOGPROBS = int(os.environ.get("LOC_TOP_LOGPROBS", "10"))
TEMPERATURE = float(os.environ.get("LOC_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("LOC_TOP_P", "0.95"))
# Cap generation. Without this a post that sends the teacher into a long <think>
# block runs to the context limit and stalls its worker; run_all*.sh already
# exported LOC_MAX_TOKENS, it was just never read here.
MAX_TOKENS = int(os.environ.get("LOC_MAX_TOKENS", "1500"))
REQUEST_TIMEOUT = float(os.environ.get("LOC_REQUEST_TIMEOUT", "180"))
# A failed post used to be dropped silently, which quietly shrinks a long run.
# llama.cpp splits its context across parallel slots, so under contention a
# request can come back "Context size has been exceeded" purely transiently.
MAX_RETRIES = int(os.environ.get("LOC_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.environ.get("LOC_RETRY_BACKOFF", "5"))


async def annotate_post(client, post, sem, system_prompt):
    for attempt in range(MAX_RETRIES):
        result = await _annotate_once(client, post, sem, system_prompt, attempt)
        if result is not None:
            return result
        if attempt + 1 < MAX_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF * (2 ** attempt))
    tqdm.write(f"Giving up on {post.get('uri')} after {MAX_RETRIES} attempts.")
    return None


async def _annotate_once(client, post, sem, system_prompt, attempt):
    async with sem:
        try:
            # We add a hint for controllable reasoning strength if supported by glimmer
            messages = [
                {"role": "system", "content": system_prompt + "\nReasoning Strength: high"},
                {"role": "user", "content": f"Text: {post['text']}"}
            ]
            
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
            )
            
            logprobs = completion.choices[0].logprobs
            if logprobs is None or logprobs.content is None:
                raise ValueError("Server returned empty logprobs.")
                
            logprobs_content = logprobs.content
            
            is_v4 = prompt_version() == "v4"
            is_assist = prompt_version() == "assist"
            is_virtue = prompt_version() == "virtue"
            # Axes that emit the Label/Level/Score triple (v4 included).
            is_graded = is_v4 or is_assist or is_virtue

            def token_to_label(token):
                t = token.strip().lower().replace(".", "").replace("\n", "")
                if is_assist:
                    return LETTER_TO_LABEL_ASSIST.get(t)
                if is_virtue:
                    return LETTER_TO_LABEL_VIRTUE.get(t)
                if is_v4 or prompt_version() in ["v2", "v3"]:
                    return LETTER_TO_LABEL.get(t)
                return t if t in ["internal", "external", "neutral"] else None

            def token_to_level(token):
                t = token.strip().replace(".", "").replace("\n", "")
                if t in ["1", "2", "3"]:
                    return int(t)
                return None

            def token_to_score(token):
                t = token.strip().replace(".", "").replace("\n", "")
                if t in [str(d) for d in range(10)]:
                    return int(t)
                return None

            label_idx = None
            level_idx = None
            score_idx = None

            # Output ends with (v3/v4/assist) Label/Level/Score. Scanning
            # backward, v4's and assist's last digit is the Score, the next is the Level; v3 has
            # only the Level digit.
            for idx in range(len(logprobs_content) - 1, -1, -1):
                tok = logprobs_content[idx].token
                if is_graded and score_idx is None and token_to_score(tok) is not None:
                    score_idx = idx
                elif level_idx is None and token_to_level(tok) is not None:
                    level_idx = idx
                elif label_idx is None and token_to_label(tok) is not None:
                    label_idx = idx
                if label_idx is not None and level_idx is not None and (
                    not is_graded or score_idx is not None
                ):
                    break

            labels = axis_labels()
            class_logprobs = {lbl: -20.0 for lbl in labels}
            level_logprobs = {"1": -20.0, "2": -20.0, "3": -20.0}
            score_logprobs = {str(d): -20.0 for d in range(10)}

            if label_idx is not None:
                decision_token_obj = logprobs_content[label_idx]
                for item in decision_token_obj.top_logprobs:
                    label = token_to_label(item.token)
                    if label is not None:
                        class_logprobs[label] = max(class_logprobs[label], item.logprob)
                        
            if level_idx is not None:
                level_token_obj = logprobs_content[level_idx]
                for item in level_token_obj.top_logprobs:
                    level = token_to_level(item.token)
                    if level is not None:
                        level_logprobs[str(level)] = max(level_logprobs[str(level)], item.logprob)

            if score_idx is not None:
                score_token_obj = logprobs_content[score_idx]
                for item in score_token_obj.top_logprobs:
                    score = token_to_score(item.token)
                    if score is not None:
                        score_logprobs[str(score)] = max(score_logprobs[str(score)], item.logprob)

            probs = {k: np.exp(v) for k, v in class_logprobs.items()}
            total_prob = sum(probs.values())
            if total_prob > 0:
                normalized_probs = {k: (v / total_prob) for k, v in probs.items()}
            else:
                pos, neg, neu = axis_labels()
                normalized_probs = {pos: 0.33, neg: 0.33, neu: 0.34}
                
            level_probs = {k: np.exp(v) for k, v in level_logprobs.items()}
            total_level_prob = sum(level_probs.values())
            if total_level_prob > 0:
                normalized_level_probs = {k: (v / total_level_prob) for k, v in level_probs.items()}
            else:
                normalized_level_probs = {"1": 1.0, "2": 0.0, "3": 0.0} # fallback
                
            post["soft_labels"] = [normalized_probs.get("internal",0), normalized_probs.get("external",0), normalized_probs.get("neutral",0)]
            post['raw_logprobs'] = class_logprobs
            post['loc_label'] = max(normalized_probs, key=normalized_probs.get)
            
            post['soft_levels'] = [normalized_level_probs["1"], normalized_level_probs["2"], normalized_level_probs["3"]]
            post['level_logprobs'] = level_logprobs
            post['loc_level'] = int(max(normalized_level_probs, key=normalized_level_probs.get))

            # v4: graded 0-9 locus-of-control score. Expected value over the digit
            # distribution (softmax of the raw digit logprobs, T=1) gives a smooth,
            # high-resolution ranking target. loc_score_logprobs is stored raw so
            # downstream can re-apply its own temperature (like raw_logprobs).
            if score_idx is not None:
                digits = np.array([score_logprobs[str(d)] for d in range(10)])
                e = np.exp(digits - digits.max())
                score_probs = e / e.sum()
                post['loc_score_logprobs'] = score_logprobs
                post['loc_score'] = float((score_probs * np.arange(10)).sum())
            else:
                post['loc_score_logprobs'] = None
                post['loc_score'] = None

            post['raw_response'] = completion.choices[0].message.content
            
            return post
        except Exception as e:
            tqdm.write(f"  attempt {attempt + 1}/{MAX_RETRIES} failed: {type(e).__name__}")
            tqdm.write(f"Error on post {post.get('uri')}: {e}")
            return None

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"{INPUT_FILE} not found. Please run the stream script first.")
        return

    client = AsyncOpenAI(base_url=API_BASE, api_key="test-key")

    annotated_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)
                annotated_ids.add(data['uri'])
        print(f"Found {len(annotated_ids)} existing annotations in {OUTPUT_FILE}. Resuming...")

    with open(INPUT_FILE, 'r') as f:
        posts = [json.loads(line) for line in f]

    posts_to_annotate = [p for p in posts if p['uri'] not in annotated_ids]
    print(f"{len(posts_to_annotate)} posts left to annotate using {API_BASE} with max concurrency {MAX_CONCURRENT}...")

    if not posts_to_annotate:
        return

    system_prompt = get_teacher_prompt()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    
    start_time = time.time()
    
    tasks = [annotate_post(client, p, sem, system_prompt) for p in posts_to_annotate]
    
    success_count = 0
    with open(OUTPUT_FILE, 'a') as f:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Annotating via {MODEL_NAME}"):
            result = await fut
            if result:
                f.write(json.dumps(result) + '\n')
                f.flush()
                success_count += 1
            
    elapsed = time.time() - start_time
    print(f"\nAnnotation complete. Successfully saved {success_count} posts to {OUTPUT_FILE}.")
    if success_count > 0:
        print(f"Total time for this batch: {elapsed:.2f} seconds ({success_count/elapsed:.2f} posts/sec)")

if __name__ == '__main__':
    asyncio.run(main())
