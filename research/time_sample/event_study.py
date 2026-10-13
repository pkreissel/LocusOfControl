"""Before/after the US presidential and German federal elections, German sample.

Events (UTC):
  us_election   2024-11-06T00:00  US polls closing, results through the night.
                Confounded: the German coalition collapsed the evening of 6 Nov.
  de_election   2025-02-23T17:00  exit polls at 18:00 CET.
                The 28-day pre-window contains the 29 Jan Merz/AfD Bundestag vote
                and the protests after it.

For each event and window w in (7, 14, 28) days: weighted mean of each measure
in [E-w, E) vs [E, E+w), difference with a bootstrap CI. Three versions:

  all        every first-language-German post in the windows
  existing   only accounts created before E-w (PLC directory creation date), so
             the X-exodus newcomers that arrive around both events cannot carry
             the difference
  DiD        the 'existing' difference minus the same difference one year earlier
             on the same calendar dates (placebo), removing seasonal drift.
             The placebo also uses only accounts existing at its own E-w.

Account creation dates are cached in research/data/time_sample/did_created.json
(fetched from plc.directory's audit log: the first operation's createdAt).

Run: /usr/bin/python3 research/time_sample/event_study.py
"""

import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data" / "time_sample"
CACHE = DATA / "did_created.json"
REPORT_MD = DATA / "event_study.md"

EVENTS = {
    "us_election": datetime(2024, 11, 6, 0, 0, tzinfo=timezone.utc),
    "de_election": datetime(2025, 2, 23, 17, 0, tzinfo=timezone.utc),
}
WINDOWS = (7, 14, 28)
B = 1000
rng = np.random.default_rng(5)

MEASURES = {
    "fusion": lambda r: r["fusion_score"],
    "fusion>=2.5": lambda r: float(r["fusion_score"] >= 2.5),
    "fusion_political": lambda r: float(r["fusion_score"] >= 1.5
                                        and r["fusion_target"] in ("political", "leader")),
    "fusion_nation": lambda r: float(r["fusion_score"] >= 1.5
                                     and r["fusion_target"] == "nation_ethnic"),
    "we": lambda r: r["fusion_we"],
    "outgroup": lambda r: r["outgroup"],
    "outgroup>=3.5": lambda r: float(r["outgroup"] >= 3.5),
    "loc_ext": lambda r: r["ts_class_probs"]["external"],
    "loc_int": lambda r: r["ts_class_probs"]["internal"],
    "p_toxicity": lambda r: r["persp"]["TOXICITY"],
    "p_insult": lambda r: r["persp"]["INSULT"],
    "p_identity_attack": lambda r: r["persp"]["IDENTITY_ATTACK"],
}


def ts(s):
    return datetime.fromisoformat(s[:19]).replace(tzinfo=timezone.utc)


def load():
    sc = {}
    for line in open(DATA / "scores_typesafe.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "outgroup" in r:
            sc[r["uri"]] = r
    pp = {}
    for line in open(DATA / "perspective_de.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("scores"):
            pp[r["uri"]] = r["scores"]
    rows = []
    for line in open(DATA / "time_sample_de.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if p["uri"] not in sc or p["uri"] not in pp or not p.get("weight"):
            continue
        if not (p["langs"] or [""])[0].lower().startswith("de"):
            continue
        rows.append({**p, **sc[p["uri"]], "persp": pp[p["uri"]], "ts": ts(p["created_at"])})
    return rows


def fetch_created(dids):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [d for d in dids if d not in cache and d.startswith("did:plc:")]

    def one(did):
        for i in range(4):
            try:
                with urllib.request.urlopen(f"https://plc.directory/{did}/log/audit",
                                            timeout=30) as resp:
                    log = json.load(resp)
                return did, log[0]["createdAt"] if log else None
            except Exception:
                time.sleep(2 ** i)
        return did, None

    if todo:
        print(f"fetching creation dates for {len(todo)} accounts from plc.directory", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, (did, created) in enumerate(pool.map(one, todo), 1):
                cache[did] = created
                if i % 500 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
                    CACHE.write_text(json.dumps(cache))
        CACHE.write_text(json.dumps(cache))
    return cache


def window(rows, e, w, created=None):
    lo, hi = e - timedelta(days=w), e + timedelta(days=w)
    sel = [r for r in rows if lo <= r["ts"] < hi]
    if created is not None:
        sel = [r for r in sel if created.get(r["did"]) and ts(created[r["did"]]) < lo]
    pre = [r for r in sel if r["ts"] < e]
    post = [r for r in sel if r["ts"] >= e]
    return pre, post


def arrays(rs, f):
    return (np.array([r["weight"] for r in rs]), np.array([f(r) for r in rs], dtype=float))


def wmean(w, y):
    return float((w * y).sum() / w.sum())


def boot_diff(pre, post, f, placebo=None):
    """post - pre (minus the placebo's post - pre), with a bootstrap 95% CI that
    resamples each of the (two or four) groups independently."""
    groups = [arrays(pre, f), arrays(post, f)]
    if placebo:
        groups += [arrays(placebo[0], f), arrays(placebo[1], f)]
    sign = [-1, 1, 1, -1][: len(groups)]
    est = sum(s * wmean(w, y) for s, (w, y) in zip(sign, groups))
    bs = np.zeros(B)
    for (w, y), s in zip(groups, sign):
        idx = rng.integers(0, len(y), size=(B, len(y)))
        ww, yy = w[idx], y[idx]
        bs += s * (ww * yy).sum(1) / ww.sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def main():
    rows = load()
    span = []
    for e in EVENTS.values():
        for shift in (0, 365):
            c = e - timedelta(days=shift)
            span.append((c - timedelta(days=max(WINDOWS)), c + timedelta(days=max(WINDOWS))))
    dids = {r["did"] for r in rows if any(a <= r["ts"] < b for a, b in span)}
    created = fetch_created(sorted(dids))
    missing = sum(1 for d in dids if not created.get(d))
    print(f"{len(dids)} accounts in the event windows, {missing} without a creation date")

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    for name, e in EVENTS.items():
        pe = e - timedelta(days=365)
        say(f"# {name}  E = {e:%Y-%m-%d %H:%M} UTC  (placebo {pe:%Y-%m-%d})\n")
        for w in WINDOWS:
            pre_a, post_a = window(rows, e, w)
            pre_x, post_x = window(rows, e, w, created)
            ppre, ppost = window(rows, pe, w, created)
            new_pre = 1 - len(pre_x) / len(pre_a)
            new_post = 1 - len(post_x) / len(post_a)
            say(f"## ±{w} days: n pre/post {len(pre_a)}/{len(post_a)}; existing-account "
                f"{len(pre_x)}/{len(post_x)} (posts from accounts newer than E-{w}d: "
                f"{100*new_pre:.0f}% pre, {100*new_post:.0f}% post); placebo {len(ppre)}/{len(ppost)}\n")
            say("| measure | pre | post | Δ all [95% CI] | Δ existing [95% CI] | DiD vs year before [95% CI] |")
            say("|---|---|---|---|---|---|")
            for m, f in MEASURES.items():
                d_all = boot_diff(pre_a, post_a, f)
                d_ex = boot_diff(pre_x, post_x, f)
                d_did = boot_diff(pre_x, post_x, f, placebo=(ppre, ppost))

                def cell(d):
                    star = " *" if d[1] > 0 or d[2] < 0 else ""
                    return f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]{star}"
                say(f"| {m} | {wmean(*arrays(pre_a, f)):.4f} | {wmean(*arrays(post_a, f)):.4f} | "
                    f"{cell(d_all)} | {cell(d_ex)} | {cell(d_did)} |")
            say()
    say("`*` = 95% CI excludes 0. 12 measures x 3 windows x 3 contrasts per event: "
        "expect ~5 starred cells per event by chance alone.")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {REPORT_MD}")


if __name__ == "__main__":
    main()
