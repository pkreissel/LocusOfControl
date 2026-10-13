"""Is Laya failing on the LoC axis, or failing in general?

The LoC rubric is subjective, and a model can miss it for uninteresting reasons:
the axis is hard, the rubric is long, the labels are Gemini's. So before reading
anything into eval_laya_vs_jev.py's numbers, ask the checkpoint three questions
whose answers are in the post itself and need no rubric at all -- the declared
language tag, whether the text contains a question mark, whether it contains a
link. A System One model that cannot do these cannot be said to have been tested
on locus of control.

Ground truth comes from the corpus, not from a labeller: `langs[0]`, `'?' in
text`, and a URL match. The majority-class rate is printed beside each accuracy
because two of the three are unbalanced.

Run: HF_HOME=/Volumes/Daten/huggingface ../../tmp_venv/bin/python eval_laya_floor.py
"""

import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VAL_FILE = os.path.join(ROOT, "training", "data", "dataset_validation_gemini.jsonl")
OUT_JSON = os.path.join(ROOT, "research", "data", "evals", "REPORT_laya_floor.json")
MODELS = ["convaiinnovations/laya-multilingual", "convaiinnovations/laya"]

QUESTIONS = {
    "lang": {"type": "choice",
             "instructions": "What language is this post written in?",
             "criteria": {"german": "the post is written in German",
                          "english": "the post is written in English",
                          "other": "the post is written in some other language"}},
    "has_question": {"type": "noul",
                     "instructions": "The post contains a question mark."},
    "has_link": {"type": "noul",
                 "instructions": "The post contains a web link or URL."},
}
URL = re.compile(r"https?://|www\.|\w+\.(com|org|net|de|io|co|uk)\b")


def main():
    import laya
    rows = [r for r in (json.loads(l) for l in open(VAL_FILE, encoding="utf-8") if l.strip())
            if r.get("text")]
    lang = np.array([(r.get("langs") or ["?"])[0].split("-")[0] for r in rows])
    de_en = np.isin(lang, ("de", "en"))
    gold_lang = np.where(lang == "de", "german", "english")
    q_mark = np.array(["?" in r["text"] for r in rows])
    has_url = np.array([bool(URL.search(r["text"])) for r in rows])

    report = {"n": len(rows), "n_de_en": int(de_en.sum()), "models": {}}
    print(f"{len(rows)} posts; {de_en.sum()} tagged de or en "
          f"({(lang[de_en] == 'en').mean():.0%} of those English)")
    print(f"question mark in {q_mark.mean():.1%}, link in {has_url.mean():.1%}\n")

    for model_id in MODELS:
        agent = laya.load(model_id)
        A = [agent.predict(r["text"], QUESTIONS)["answers"] for r in rows]
        pred = np.array([a["lang"]["choice"] for a in A])
        acc = float((pred[de_en] == gold_lang[de_en]).mean())
        base = float(max((gold_lang[de_en] == c).mean() for c in ("german", "english")))
        qa = float(roc_auc_score(q_mark, [a["has_question"]["noul"] for a in A]))
        la = float(roc_auc_score(has_url, [a["has_link"]["noul"] for a in A]))
        print(f"{model_id}")
        print(f"  language de/en   acc {acc:.3f}   (majority class {base:.3f})")
        print(f"  '?' present      AUC {qa:.3f}")
        print(f"  link present     AUC {la:.3f}")
        report["models"][model_id] = {"lang_acc": acc, "lang_majority": base,
                                      "question_mark_auc": qa, "link_auc": la}
        del agent

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
