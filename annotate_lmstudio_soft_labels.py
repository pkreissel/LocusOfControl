import json
import os
import time
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from label_prompts import LETTER_TO_LABEL, get_teacher_prompt, prompt_version

INPUT_FILE = os.environ.get("LOC_INPUT_FILE", "dataset_raw.jsonl")
OUTPUT_FILE = os.environ.get("LOC_OUTPUT_FILE", "dataset_annotated_lmstudio_soft.jsonl")
BATCH_SAVE = 10
# Chosen model for psycholinguistic reasoning:
MODEL_NAME = os.environ.get("LOC_TEACHER_MODEL", "tess-4-9b")
TOP_LOGPROBS = int(os.environ.get("LOC_TOP_LOGPROBS", "15"))  # MLX models cap at 10
TEMPERATURE = os.environ.get("LOC_TEMPERATURE")  # unset -> LM Studio default
MAX_TOKENS = os.environ.get("LOC_MAX_TOKENS")  # unset -> no cap; guard against runaway reasoning

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"{INPUT_FILE} not found. Please run the stream script first.")
        return

    # Connect to local LM Studio instance
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

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
    print(f"{len(posts_to_annotate)} posts left to annotate.")

    if not posts_to_annotate:
        return

    system_prompt = get_teacher_prompt()
    print(f"Using teacher prompt version: {prompt_version()}")

    # v1 answers with a full word; v2 answers "Label: X" with a single letter so
    # that all three classes reliably appear in top_logprobs (real soft labels).
    def token_to_label(token):
        t = token.strip().lower().replace(".", "").replace("\n", "")
        if prompt_version() == "v2":
            return LETTER_TO_LABEL.get(t)
        return t if t in ["internal", "external", "neutral"] else None

    with open(OUTPUT_FILE, 'a') as f:
        for i, post in enumerate(tqdm(posts_to_annotate, desc=f"Annotating via LM Studio ({MODEL_NAME})")):
            try:
                kwargs = {}
                if TEMPERATURE is not None:
                    kwargs["temperature"] = float(TEMPERATURE)
                if MAX_TOKENS is not None:
                    kwargs["max_tokens"] = int(MAX_TOKENS)
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Text: {post['text']}"}
                    ],
                    logprobs=True,
                    top_logprobs=TOP_LOGPROBS,
                    **kwargs
                )
                
                logprobs = completion.choices[0].logprobs
                if logprobs is None or logprobs.content is None:
                    raise ValueError("LM Studio returned empty logprobs.")
                logprobs_content = logprobs.content
                
                # Scan backwards to find the final classification token
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
                    raise ValueError("Could not find strict label at end of generation.")
                    
                # Convert to normalized probabilities
                probs = {k: np.exp(v) for k, v in class_logprobs.items()}
                total_prob = sum(probs.values())
                
                if total_prob > 0:
                    normalized_probs = {k: (v / total_prob) for k, v in probs.items()}
                else:
                    raise ValueError("Invalid logprobs (all zeros or extremely low).")
                
                probs_list = [
                    normalized_probs["internal"], 
                    normalized_probs["external"], 
                    normalized_probs["neutral"]
                ]
                
                post['soft_labels'] = probs_list
                post['raw_logprobs'] = class_logprobs # Save raw logprobs so temperature scaling can be applied later
                post['loc_label'] = max(normalized_probs, key=normalized_probs.get) # Best guess hard label for fallback
                post['raw_response'] = completion.choices[0].message.content # Store reasoning for debugging
                
                f.write(json.dumps(post) + '\n')
                
                if i > 0 and i % BATCH_SAVE == 0:
                    f.flush()
                    
            except Exception as e:
                tqdm.write(f"\nAPI Error: {e}")
                time.sleep(2)
                
    print("\nAnnotation complete.")

if __name__ == '__main__':
    main()
