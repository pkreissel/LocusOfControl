"""Resolve the authors of a labelled corpus to their public Bluesky profiles.

Keeps handle, display name and bio -- the only fields infer_gender.py reads.
Appends as it goes and skips DIDs already fetched, so a killed run resumes.

Env:
  GB_CORPUS    labelled corpus jsonl  (default training/data/dataset_annotated_decider_net20k.jsonl)
  GB_PROFILES  profiles jsonl out     (default research/data/gender_bias/profiles.jsonl)
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "de_study"))
from hydrate_engagement import Api, chunks, CHUNK  # noqa: E402

CORPUS = Path(os.environ.get("GB_CORPUS", ROOT / "training/data/dataset_annotated_decider_net20k.jsonl"))
OUT = Path(os.environ.get("GB_PROFILES", ROOT / "research/data/gender_bias/profiles.jsonl"))


def main():
    dids = list(dict.fromkeys(json.loads(l)["did"] for l in open(CORPUS, encoding="utf-8")))
    done = set()
    if OUT.exists():
        done = {json.loads(l)["did"] for l in open(OUT, encoding="utf-8")}
    todo = [d for d in dids if d not in done]
    print(f"{len(dids)} authors, {len(todo)} to fetch", flush=True)
    api = Api(qps=8.0)
    with open(OUT, "a", encoding="utf-8") as out:
        for i, batch in enumerate(chunks(todo, CHUNK)):
            data = api.get("app.bsky.actor.getProfiles", {"actors": batch}) or {}
            got = {p["did"]: p for p in data.get("profiles") or []}
            for did in batch:
                p = got.get(did)
                out.write(json.dumps({
                    "did": did, "available": p is not None,
                    "handle": p and p.get("handle"),
                    "display_name": p and p.get("displayName"),
                    "description": p and p.get("description"),
                    "followers_count": p and p.get("followersCount"),
                }, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                out.flush()
                print(f"  {min((i + 1) * CHUNK, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
