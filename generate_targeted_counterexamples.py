import os
import json
import time
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from create_synthetic_counterexamples import teacher_annotate

OUT = os.environ.get("LOC_SYNTH_OUT", "dataset_synthetic_counterexamples.jsonl")
API_BASE = os.environ.get("LOC_API_BASE", "http://127.0.0.1:8080/v1")

def main():
    load_dotenv()
    gclient = genai.Client(api_key=os.environ["GAI_API"])
    client = OpenAI(base_url=API_BASE, api_key="test-key")
    
    PROMPTS = [
        ("external", "Generate 10 highly realistic German social media posts discussing climate change or politics. They MUST be written with an extreme doomer, helpless, cynical, or heavily sarcastic attitude. They must complain about systemic issues and express that individuals have no power (External Locus of Control). Use casual language, maybe some emojis, and keep them under 280 characters each. Separate each post with a line containing '---'."),
        ("internal", "Generate 10 highly realistic German social media posts discussing climate change or politics. They MUST be written with a highly agentic, solution-oriented, self-efficacious attitude. They must focus on what the author is personally doing or how individuals can make a concrete difference, no matter how small (Internal Locus of Control). Use casual language, maybe some emojis, and keep them under 280 characters each. Separate each post with a line containing '---'."),
    ]
    
    total_generated = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for target, prompt in PROMPTS:
            print(f"Generating {target} examples...")
            for batch in tqdm(range(5)): # 5 batches of 10 = 50 each
                try:
                    time.sleep(4.1) # rate limit
                    r = gclient.models.generate_content(
                        model="gemini-3.1-flash-lite",
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.8),
                    )
                    posts = [p.strip() for p in r.text.split("---") if len(p.strip()) > 10]
                    for p_text in posts:
                        try:
                            lp, soft, label, lvl_lp, soft_lvl, loc_lvl = teacher_annotate(client, p_text)
                            uri = f"at://did:plc:synth/app.bsky.feed.post/synth-targeted-{target}-{np.random.randint(1000000)}"
                            f.write(json.dumps({
                                "uri": uri, "text": p_text, "source_uri": "targeted_generation",
                                "source_text": "", "rewrite_target": target,
                                "raw_logprobs": lp, "soft_labels": soft, "loc_label": label,
                                "level_logprobs": lvl_lp, "soft_levels": soft_lvl, "loc_level": loc_lvl
                            }) + "\n")
                            f.flush()
                            total_generated += 1
                        except Exception as e:
                            pass # skip annotation failures
                except Exception as e:
                    print(f"Generation error: {e}")
                    
    print(f"Successfully generated and annotated {total_generated} targeted counterexamples to {OUT}.")

if __name__ == '__main__':
    main()
