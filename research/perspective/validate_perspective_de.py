"""Does each Perspective attribute actually mean anything on German text?

The API will *return* all 23 attributes for German text as long as it is submitted
as `languages=["en"]` -- but returning a number is not the same as measuring the
construct. Only the 6 core attributes have German validation behind them. This
script separates the two questions:

  1. Language-parameter invariance (core six only, the ones `de` accepts):
     score the same German texts under `languages=["de"]` and `["en"]` and
     correlate. Near-1.0 confirms the parameter is a hint, not a model selector,
     so `en` submission does not distort the core scores.

  2. Degeneracy screen (all 23): an attribute whose German distribution is
     saturated (nearly all mass above .9 or below .1) or whose German-vs-English
     distributions barely overlap is keying on "this text is not English" rather
     than on the construct. Such attributes carry no usable within-German variance
     and must not enter the engagement regression.

Outputs a JSON report plus a keep/drop verdict per attribute.
"""
import argparse
import json
import os
import random
import statistics
import sys
import threading
import queue

import requests
from dotenv import load_dotenv

from score_perspective_de import ATTRS, CORE, URL, score_one

# Judgement thresholds for the degeneracy screen.
#
# Two failure modes must not be conflated:
#
#   language artifact -- the attribute saturates HIGH on German while sitting low
#     on comparable English text. It is scoring "this is not English", not the
#     construct. Invalid; must not enter the regression.
#
#   rare construct -- the attribute sits near zero for almost every post. That is
#     what a *correct* THREAT detector does on a random sample of German posts.
#     Valid, but with few positives it has little power, so it is flagged rather
#     than dropped: silently discarding it would throw away a real finding.
SATURATION = 0.90     # >=90% of mass in the top or bottom decile
MIN_SD = 0.05         # near-constant
OVERLAP_MIN = 0.35    # German/English overlap below this, with high saturation
POSITIVE_AT = 0.50    # score above this counts as a "positive" for power accounting
MIN_POSITIVES = 100   # expected positives AT THE STUDY N below this -> underpowered.
                      # Judged against the study sample, not the audit sample: an
                      # attribute with 16 positives in 600 audit posts has ~270 at
                      # n=10,000, which is plenty.


def score_many(texts, key, languages, concurrency=10):
    """Score a list of texts, preserving order. None where scoring failed."""
    results = [None] * len(texts)
    q = queue.Queue()
    for i, t in enumerate(texts):
        q.put((i, t))
    session = requests.Session()
    lock = threading.Lock()

    def work():
        while True:
            try:
                i, t = q.get_nowait()
            except queue.Empty:
                return
            body_scores, _err = score_one_lang(session, key, t, languages)
            with lock:
                results[i] = body_scores
            q.task_done()

    threads = [threading.Thread(target=work, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def score_one_lang(session, key, text, languages):
    """score_one() pins languages=['en']; the invariance test needs ['de'] too."""
    import time as _t
    attrs = ATTRS if languages == ["en"] else CORE
    body = {"comment": {"text": text[:20000]}, "languages": languages,
            "requestedAttributes": {a: {} for a in attrs}, "doNotStore": True}
    delay = 1.0
    for attempt in range(5):
        try:
            r = session.post(URL, params={"key": key}, json=body, timeout=45)
        except requests.RequestException:
            _t.sleep(delay); delay *= 2
            continue
        if r.status_code == 200:
            s = r.json().get("attributeScores", {})
            return {a: s.get(a, {}).get("summaryScore", {}).get("value") for a in attrs}, None
        if r.status_code in (429, 500, 503):
            _t.sleep(delay); delay *= 2
            continue
        return None, f"http:{r.status_code}"
    return None, "exhausted"


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 10:
        return None
    xs2 = [p[0] for p in pairs]; ys2 = [p[1] for p in pairs]
    mx, my = statistics.fmean(xs2), statistics.fmean(ys2)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = sum((x - mx) ** 2 for x in xs2) ** 0.5
    dy = sum((y - my) ** 2 for y in ys2) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def overlap(a, b, bins=20):
    """Histogram intersection of two score distributions on [0,1]."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        return None
    ha = [0] * bins; hb = [0] * bins
    for v in a:
        ha[min(int(v * bins), bins - 1)] += 1
    for v in b:
        hb[min(int(v * bins), bins - 1)] += 1
    return sum(min(ha[i] / len(a), hb[i] / len(b)) for i in range(bins))


def describe(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    q = lambda p: v[min(int(p * n), n - 1)]  # noqa: E731
    return {
        "n": n,
        "mean": round(statistics.fmean(v), 4),
        "sd": round(statistics.pstdev(v), 4) if n > 1 else 0.0,
        "p01": round(q(0.01), 4), "p25": round(q(0.25), 4), "p50": round(q(0.50), 4),
        "p75": round(q(0.75), 4), "p99": round(q(0.99), 4),
        "frac_gt_0.9": round(sum(1 for x in v if x > 0.9) / n, 4),
        "frac_lt_0.1": round(sum(1 for x in v if x < 0.1) / n, 4),
        "n_positive": sum(1 for x in v if x >= POSITIVE_AT),
        "frac_positive": round(sum(1 for x in v if x >= POSITIVE_AT) / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="de_pool_raw.jsonl")
    ap.add_argument("--en-sample", required=True, help="jsonl of English posts for contrast")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--study-n", type=int, default=10000,
                    help="size of the study sample the power screen is judged against")
    ap.add_argument("--raw-out", default="de_validity_raw_scores.json",
                    help="cache the audit scores so verdicts can be re-derived for free")
    ap.add_argument("--out", default="REPORT_perspective_de_validity.json")
    args = ap.parse_args()

    load_dotenv()
    key = os.environ["PERSPECTIVE_API_KEY"]
    rng = random.Random(args.seed)

    # German texts: confidently German by detector, long enough to carry signal.
    de = []
    with open(args.pool, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("det_lang") == "de" and (r.get("det_conf") or 0) >= 0.90 \
                    and r.get("text_len", 0) >= 40:
                de.append(r["text"])
    rng.shuffle(de)
    de = de[:args.n]

    en = []
    with open(args.en_sample, "r", encoding="utf-8") as f:
        for line in f:
            try:
                en.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    rng.shuffle(en)
    en = en[:args.n]

    print(f"German texts: {len(de)}, English contrast: {len(en)}", flush=True)

    print("scoring German as en (23 attrs)...", flush=True)
    de_en = score_many(de, key, ["en"])
    print("scoring German as de (6 core attrs)...", flush=True)
    de_de = score_many(de, key, ["de"])
    print("scoring English as en (23 attrs)...", flush=True)
    en_en = score_many(en, key, ["en"])

    with open(args.raw_out, "w", encoding="utf-8") as f:
        json.dump({"de_as_en": de_en, "de_as_de": de_de, "en_as_en": en_en}, f)

    col = lambda rows, a: [(r or {}).get(a) for r in rows]  # noqa: E731

    report = {"n_de": len(de), "n_en": len(en), "seed": args.seed, "attributes": {}}

    for a in ATTRS:
        de_vals = col(de_en, a)
        en_vals = col(en_en, a)
        entry = {
            "german_dist": describe(de_vals),
            "english_dist": describe(en_vals),
            "de_en_overlap": round(overlap(de_vals, en_vals), 4) if overlap(de_vals, en_vals) is not None else None,
        }
        if a in CORE:
            entry["lang_param_r"] = round(pearson(de_vals, col(de_de, a)), 5) \
                if pearson(de_vals, col(de_de, a)) is not None else None
            entry["accepts_de_param"] = True
        else:
            entry["accepts_de_param"] = False

        d = entry["german_dist"] or {}
        ovl = entry["de_en_overlap"]
        artifact, power = [], []

        # Invalidity: saturating high on German but not on English.
        if d.get("frac_gt_0.9", 0) >= SATURATION and ovl is not None and ovl < OVERLAP_MIN:
            artifact.append(f"saturates high on German ({d['frac_gt_0.9']:.0%} above .9) "
                            f"while English overlap is only {ovl:.2f} -- scoring the "
                            f"language, not the construct")
        if a in CORE and entry.get("lang_param_r") is not None and entry["lang_param_r"] < 0.95:
            artifact.append(f"language parameter changes the score (r={entry['lang_param_r']})")

        # Low power: a rare construct is not a broken one. Extrapolate the audit
        # positive rate to the study sample before judging.
        exp_pos = (d.get("frac_positive") or 0) * args.study_n
        entry["expected_positives_at_study_n"] = int(round(exp_pos))
        if exp_pos < MIN_POSITIVES:
            power.append(f"positive rate {d.get('frac_positive'):.4f} implies only "
                         f"~{int(round(exp_pos))} posts above {POSITIVE_AT} at "
                         f"n={args.study_n} -- too few to detect an effect")
        elif d.get("sd", 0) < MIN_SD:
            power.append(f"near-constant on German (sd={d.get('sd')})")

        if artifact:
            entry["verdict"] = "drop"
        elif power:
            entry["verdict"] = "low_power"
        else:
            entry["verdict"] = "keep"
        entry["reasons"] = artifact + power
        report["attributes"][a] = entry
        print(f"{a:32s} {entry['verdict']:10s} "
              f"sd={d.get('sd')} pos={d.get('n_positive')}/{d.get('n')} "
              f"~{entry['expected_positives_at_study_n']}@n={args.study_n} "
              f"ovl={ovl} "
              + ("| " + "; ".join(entry["reasons"]) if entry["reasons"] else ""), flush=True)

    report["thresholds"] = {"saturation": SATURATION, "min_sd": MIN_SD,
                            "overlap_min": OVERLAP_MIN, "positive_at": POSITIVE_AT,
                            "min_expected_positives": MIN_POSITIVES,
                            "study_n": args.study_n}
    # `keep` drives the analysis table: valid attributes, including the underpowered
    # ones, which are carried with a flag so a null result can be read as "no power"
    # rather than "no effect".
    report["keep"] = [a for a in ATTRS if report["attributes"][a]["verdict"] in ("keep", "low_power")]
    report["well_powered"] = [a for a in ATTRS if report["attributes"][a]["verdict"] == "keep"]
    report["low_power"] = [a for a in ATTRS if report["attributes"][a]["verdict"] == "low_power"]
    report["drop"] = [a for a in ATTRS if report["attributes"][a]["verdict"] == "drop"]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwell-powered {len(report['well_powered'])}: {', '.join(report['well_powered'])}")
    print(f"low-power (kept, flagged) {len(report['low_power'])}: {', '.join(report['low_power'])}")
    print(f"dropped as language artifacts {len(report['drop'])}: {', '.join(report['drop'])}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
