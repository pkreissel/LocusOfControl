"""Infer an author's gender presentation from their public profile.

Two signals, in order of precision:
  1. Pronouns stated in the bio or display name (she/her, sie/ihr, elle, he/him,
     er/ihm, il/lui; they/them and neopronouns -> "nonbinary").
  2. The first given name in the display name, looked up in gender_guesser's
     name dictionary (covers German and most European names). Titles like
     "Dr" are skipped. The handle is NOT used: it is too often a nickname.

Anything else is "unknown" -- organisations, nicknames, empty names. A name is
a proxy for how an author presents, not their gender; report it as that.

Env:
  GB_PROFILES  profiles jsonl in  (default research/data/gender_bias/profiles.jsonl)
  GB_GENDER    gender jsonl out   (default research/data/gender_bias/author_gender.jsonl)
"""
import collections
import json
import os
import re
from pathlib import Path

import gender_guesser.detector as gd

ROOT = Path(__file__).resolve().parents[2]
PROFILES = Path(os.environ.get("GB_PROFILES", ROOT / "research/data/gender_bias/profiles.jsonl"))
OUT = Path(os.environ.get("GB_GENDER", ROOT / "research/data/gender_bias/author_gender.jsonl"))

PRONOUNS = [
    ("nonbinary", r"\b(they\s*/\s*them|they\s*/\s*she|they\s*/\s*he|she\s*/\s*they|he\s*/\s*they|xe\s*/\s*xem|it\s*/\s*its|any\s*/\s*all|any pronouns|nonbinary|non-binary|enby)\b"),
    ("female", r"\b(she\s*/\s*her|sie\s*/\s*ihr|elle\s*/\s*la|elle\s*/\s*elle)\b"),
    ("male", r"\b(he\s*/\s*him|er\s*/\s*ihm|er\s*/\s*ihn|il\s*/\s*lui|il\s*/\s*il)\b"),
]
TITLES = {"dr", "prof", "mr", "mrs", "ms", "miss", "sir", "dame", "herr", "frau",
          "the", "doc", "rev", "dipl", "ing", "mag"}
GUESSER = {"female": "female", "mostly_female": "female",
           "male": "male", "mostly_male": "male"}


def from_pronouns(text):
    t = (text or "").lower()
    for label, pat in PRONOUNS:
        if re.search(pat, t):
            return label
    return None


def first_name(display_name):
    for tok in re.findall(r"[^\W\d_]+", display_name or ""):
        if tok.lower().rstrip(".") in TITLES or len(tok) < 2:
            continue
        return tok
    return None


def main():
    det = gd.Detector(case_sensitive=False)
    counts = collections.Counter()
    with open(OUT, "w", encoding="utf-8") as out:
        for line in open(PROFILES, encoding="utf-8"):
            p = json.loads(line)
            gender, source, name = "unknown", None, None
            g = from_pronouns(f"{p.get('display_name') or ''} {p.get('description') or ''}")
            if g:
                gender, source = g, "pronouns"
            else:
                name = first_name(p.get("display_name"))
                if name:
                    g = GUESSER.get(det.get_gender(name))
                    if g:
                        gender, source = g, "name"
            counts[(gender, source)] += 1
            out.write(json.dumps({"did": p["did"], "gender": gender, "source": source,
                                  "name": name}, ensure_ascii=False) + "\n")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k[0]:9s} {str(k[1]):9s} {v}")


if __name__ == "__main__":
    main()
