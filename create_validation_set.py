import json
import os
import time
import random
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

from label_prompts import get_validation_prompt, prompt_version

INPUT_FILE = "dataset_raw.jsonl"
OUTPUT_FILE = "dataset_validation_gemini.jsonl"
NUM_VALIDATION = 500

def main():
    load_dotenv()
    
    if not os.path.exists(INPUT_FILE):
        print(f"{INPUT_FILE} not found.")
        return

    api_key = os.environ.get('GAI_API')
    if not api_key:
        print("Please set the GAI_API environment variable.")
        return

    client = genai.Client(api_key=api_key)

    annotated_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)
                annotated_ids.add(data['uri'])
        print(f"Found {len(annotated_ids)} existing Gemini annotations. Resuming...")

    # Load all posts
    with open(INPUT_FILE, 'r') as f:
        all_posts = [json.loads(line) for line in f]
        
    # Sample 500 deterministically
    random.seed(42)
    val_candidates = random.sample(all_posts, NUM_VALIDATION)

    posts_to_annotate = [p for p in val_candidates if p['uri'] not in annotated_ids]
    print(f"{len(posts_to_annotate)} validation posts left to annotate.")

    if not posts_to_annotate:
        return

    system_prompt = get_validation_prompt()
    print(f"Using validation prompt version: {prompt_version()}")

    with open(OUTPUT_FILE, 'a') as f:
        for i, post in enumerate(tqdm(posts_to_annotate, desc="Annotating Validation Set")):
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    time.sleep(4.1) # strictly respect 15 RPM free tier limits
                    
                    response = client.models.generate_content(
                        model='gemini-3.1-flash-lite',
                        contents=f"Text: {post['text']}",
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.0
                        )
                    )
                    
                    raw_response = response.text.strip().lower()
                    
                    if "internal" in raw_response:
                        final_label = "internal"
                    elif "external" in raw_response:
                        final_label = "external"
                    else:
                        final_label = "neutral"
                        
                    post['loc_label'] = final_label
                    f.write(json.dumps(post) + '\n')
                    f.flush()
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "503" in error_msg:
                        tqdm.write(f"Rate limited/Busy. Retrying in 10s... (Attempt {attempt+1})")
                        time.sleep(10)
                    else:
                        tqdm.write(f"API Error: {e}")
                        time.sleep(5)
                
    print("\nValidation annotation complete.")

if __name__ == '__main__':
    main()
