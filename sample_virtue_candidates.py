"""Enrichment sampler for the virtue axis.

Random social-media sampling yields almost no 'virtuous' posts - a 62-post
probe of dataset_raw.jsonl on 2026-09-07 returned 0. That is the same class
starvation the assist axis hit, and the same remedy applies: bias the pool
towards posts that could plausibly carry the positive class, then let the
teacher decide. The markers below are surface cues for epistemic conduct
(hedging, revision, weighing, scoping) in English and German; they are
deliberately loose, because their job is to raise the base rate, not to label
anything. Precision is the teacher's job.

Emits a mixed file so the two pools stay comparable in one annotation run:
each row carries pool="enriched" or pool="random".

  LOC_SAMPLE_POOLS   comma-separated input files (default dataset_raw.jsonl)
  LOC_SAMPLE_OUT     output path
  LOC_SAMPLE_N       enriched posts to emit (default 150)
  LOC_SAMPLE_N_RAND  random control posts to emit (default 150)
  LOC_SAMPLE_MIN_LEN minimum text length (default 80)
"""
import json
import os
import random
import re

POOLS = os.environ.get("LOC_SAMPLE_POOLS", "dataset_raw.jsonl").split(",")
OUT = os.environ.get("LOC_SAMPLE_OUT", "dataset_raw_virtue_selection.jsonl")
N_ENRICHED = int(os.environ.get("LOC_SAMPLE_N", "150"))
N_RANDOM = int(os.environ.get("LOC_SAMPLE_N_RAND", "150"))
MIN_LEN = int(os.environ.get("LOC_SAMPLE_MIN_LEN", "80"))
SEED = int(os.environ.get("LOC_SAMPLE_SEED", "11"))

# Weighted surface markers. Weights are a crude prior on how often the marker
# co-occurs with real epistemic work, not a score to be trusted downstream.
MARKERS = [
    # revision / updating - the strongest signal
    (3.0, r"\b(i (was|used to be) wrong|changed my mind|i stand corrected|i've updated|"
          r"i was too (harsh|quick)|thanks for the correction|geirrt|umgestimmt|"
          r"ich lag falsch|revidiere)\b"),
    # explicit uncertainty / humility
    (2.0, r"\b(i could be wrong|i might be wrong|i'm not sure|i don't know enough|"
          r"correct me if|happy to be corrected|genuinely unsure|no idea if|"
          r"ich k(o|ö)nnte mich irren|bin mir nicht sicher|keine ahnung ob|"
          r"lasse mich gern korrigieren)\b"),
    # weighing / concession
    (2.0, r"\b(on the other hand|to be fair|that said|the strongest (case|argument)|"
          r"steelman|both things can be true|i want both|trade-?off|"
          r"andererseits|zugegeben|fairerweise|beides stimmt)\b"),
    # calibration / scoping the claim
    (1.5, r"\b(n=1|small sample|self-?reported|the (study|data) (is|are) thin|"
          r"anecdotal|only one (data ?point|study)|effect size|confidence interval|"
          r"stichprobe|anekdotisch|selbst berichtet)\b"),
    # falsifiability
    (2.0, r"\b(what would change my mind|would convince me|if it turned out|"
          r"what am i missing|was (uebersehe|übersehe) ich|was spricht dagegen)\b"),
    # weaker hedges - common, so low weight
    (0.7, r"\b(seems to me|i think|arguably|probably|my guess is|as far as i can tell|"
          r"soweit ich (weiss|weiß)|vermutlich|mein eindruck)\b"),
]
COMPILED = [(w, re.compile(p, re.I)) for w, p in MARKERS]


def score(text):
    s = sum(w for w, rx in COMPILED if rx.search(text))
    # Epistemic work needs room; give longer posts a mild edge without letting
    # length dominate (the register confound this axis has to avoid).
    return s + min(len(text), 600) / 600.0 if s else 0.0


def load(paths):
    seen, rows = set(), []
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skipping missing pool {p})")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                t = (r.get("text") or "").strip()
                if len(t) < MIN_LEN or r.get("uri") in seen:
                    continue
                seen.add(r.get("uri"))
                rows.append(r)
    return rows


def main():
    random.seed(SEED)
    rows = load(POOLS)
    print(f"{len(rows)} eligible posts from {len(POOLS)} pool(s)")

    scored = [(score(r["text"]), r) for r in rows]
    hits = [(s, r) for s, r in scored if s > 0]
    hits.sort(key=lambda x: -x[0])
    print(f"{len(hits)} carry at least one marker ({len(hits)/max(len(rows),1)*100:.1f}%)")

    # Take the top band, but sample within it so the file is not just the
    # longest posts in the corpus.
    band = hits[: N_ENRICHED * 3]
    enriched = [r for _, r in random.sample(band, min(N_ENRICHED, len(band)))]
    chosen = {r.get("uri") for r in enriched}
    control = random.sample([r for r in rows if r.get("uri") not in chosen],
                            min(N_RANDOM, max(len(rows) - len(chosen), 0)))

    with open(OUT, "w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps({**r, "pool": "enriched"}, ensure_ascii=False) + "\n")
        for r in control:
            f.write(json.dumps({**r, "pool": "random"}, ensure_ascii=False) + "\n")
    print(f"wrote {len(enriched)} enriched + {len(control)} random -> {OUT}")


if __name__ == "__main__":
    main()
