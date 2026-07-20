"""Re-annotate the 500 validation posts with the current Gemini v2 prompt.

Reads dataset_validation_gemini.jsonl (posts + v1 labels), writes
dataset_validation_gemini_v2.jsonl with loc_label (v2) and loc_label_v1.
Resumable; run again to fill any failures.

Run: uv run --with google-genai --with python-dotenv --with tqdm python annotate_validation_v2.py
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from label_prompts import get_validation_prompt, prompt_version

OUTPUT_FILE = os.environ.get("LOC_VAL_OUTPUT", "dataset_validation_gemini_v2.jsonl")
GEMINI_MODEL = "gemini-3.1-flash-lite"


def main():
    load_dotenv()
    client = genai.Client(api_key=os.environ["GAI_API"])
    print(f"Using validation prompt version: {prompt_version()}")

    with open("dataset_validation_gemini.jsonl", encoding="utf-8") as f:
        posts = [json.loads(l) for l in f if l.strip()]

    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            done = {json.loads(l)["uri"] for l in f if l.strip()}
        print(f"Resuming: {len(done)} already annotated.")

    todo = [p for p in posts if p["uri"] not in done]
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for post in tqdm(todo, desc="Gemini v2 validation"):
            for attempt in range(8):
                try:
                    time.sleep(4.1)  # 15 RPM free tier
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=f"Text: {post['text']}",
                        config=types.GenerateContentConfig(
                            system_instruction=get_validation_prompt(), temperature=0.0),
                    )
                    raw = response.text.strip().lower()
                    label = next((l for l in ["internal", "external"] if l in raw), "neutral")
                    record = {"uri": post["uri"], "text": post["text"],
                              "loc_label": label, "loc_label_v1": post.get("loc_label")}
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    break
                except Exception as e:
                    msg = str(e).lower()
                    wait = 20 if ("429" in msg or "503" in msg) else 5
                    tqdm.write(f"Error ({post['uri']}): {e}; retry in {wait}s")
                    time.sleep(wait)

    with open(OUTPUT_FILE, encoding="utf-8") as f:
        n = sum(1 for l in f if l.strip())
    print(f"Done: {n}/{len(posts)} annotated.")


if __name__ == "__main__":
    main()
