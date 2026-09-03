import asyncio
import json
import os
import time
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from label_prompts import LETTER_TO_LABEL, get_teacher_prompt, prompt_version

INPUT_FILE = "dataset_raw.jsonl"
OUTPUT_FILE = "dataset_annotated_llama_cpp_test.jsonl"
BATCH_SIZE = 20 # Small test batch
MAX_CONCURRENT = 4

# Optimal settings researched for meta/muse-glimmer:
MODEL_NAME = "meta/muse-glimmer"
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_LOGPROBS = 10  # Reduced to 10 as many servers cap here, but can increase if supported

async def annotate_post(client, post, sem, system_prompt):
    async with sem:
        try:
            # We add a hint for controllable reasoning strength if supported by glimmer
            # (though the prompt already says <think>...)
            messages = [
                {"role": "system", "content": system_prompt + "\nReasoning Strength: high"},
                {"role": "user", "content": f"Text: {post['text']}"}
            ]
            
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
            )
            
            logprobs = completion.choices[0].logprobs
            if logprobs is None or logprobs.content is None:
                raise ValueError("Server returned empty logprobs.")
                
            logprobs_content = logprobs.content
            
            def token_to_label(token):
                t = token.strip().lower().replace(".", "").replace("\n", "")
                if prompt_version() in ["v2", "v3"]:
                    return LETTER_TO_LABEL.get(t)
                return t if t in ["internal", "external", "neutral"] else None

            target_idx = None
            for idx in range(len(logprobs_content) - 1, -1, -1):
                if token_to_label(logprobs_content[idx].token) is not None:
                    target_idx = idx
                    break
                    
            class_logprobs = {"internal": -20.0, "external": -20.0, "neutral": -20.0}

            if target_idx is not None:
                decision_token_obj = logprobs_content[target_idx]
                for item in decision_token_obj.top_logprobs:
                    label = token_to_label(item.token)
                    if label is not None:
                        class_logprobs[label] = max(class_logprobs[label], item.logprob)
            else:
                # Fallback if it didn't strictly follow the grammar
                pass
                
            probs = {k: np.exp(v) for k, v in class_logprobs.items()}
            total_prob = sum(probs.values())
            
            if total_prob > 0:
                normalized_probs = {k: (v / total_prob) for k, v in probs.items()}
            else:
                normalized_probs = {"internal": 0.33, "external": 0.33, "neutral": 0.34} # fallback
                
            post['soft_labels'] = [normalized_probs["internal"], normalized_probs["external"], normalized_probs["neutral"]]
            post['raw_logprobs'] = class_logprobs
            post['loc_label'] = max(normalized_probs, key=normalized_probs.get)
            post['raw_response'] = completion.choices[0].message.content
            
            return post
        except Exception as e:
            print(f"Error on post {post.get('uri')}: {e}")
            return None

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"{INPUT_FILE} not found.")
        return

    # Using API base from environment or default to llama.cpp port
    api_base = os.environ.get("LOC_API_BASE", "http://localhost:8080/v1")
    client = AsyncOpenAI(base_url=api_base, api_key="test-key")
    
    with open(INPUT_FILE, 'r') as f:
        posts = [json.loads(line) for line in f]
        
    # Take a small test batch
    posts_to_annotate = posts[:BATCH_SIZE]
    print(f"Annotating {len(posts_to_annotate)} posts using {api_base} with max concurrency {MAX_CONCURRENT}...")
    
    system_prompt = get_teacher_prompt()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    
    start_time = time.time()
    
    tasks = [annotate_post(client, p, sem, system_prompt) for p in posts_to_annotate]
    
    annotated_posts = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        if result:
            annotated_posts.append(result)
            
    elapsed = time.time() - start_time
            
    with open(OUTPUT_FILE, 'w') as f:
        for p in annotated_posts:
            f.write(json.dumps(p) + '\n')
            
    print(f"Saved {len(annotated_posts)} annotated posts to {OUTPUT_FILE}")
    print(f"Total time for {len(annotated_posts)} posts: {elapsed:.2f} seconds ({len(annotated_posts)/elapsed:.2f} posts/sec)")

if __name__ == '__main__':
    asyncio.run(main())
