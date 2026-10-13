"""Audit how much locus-of-control content different Bluesky feeds serve.

For each feed: read what it serves from the public AppView (collect_feed_posts.py,
no auth), score the posts the deployment would embed (>= 6 words) with the
fine-tuned decider-2b (research/evals/score_loc_decider.py, local weights, no
spend), and compare feeds on the share of internal / external / neutral posts.

Steps are resumable and independent:

  collect   write research/data/feed/audit/posts_<slug>.jsonl per feed
  score     one decider run over the de-duplicated union (the model loads once)
  report    join, print, and write docs/reports/REPORT_feed_audit.md

Run (from anywhere):
  python research/feed/audit_feeds.py collect --popular 20 --max-posts 200
  HF_HOME=/Volumes/Daten/huggingface tmp_venv/bin/python research/feed/audit_feeds.py score
  python research/feed/audit_feeds.py report

Caveats that belong next to any number from this:
  * getFeed is called anonymously, so personalised feeds (For You, Mutuals,
    Popular With Friends) show what a viewer with no graph gets, not what a
    member sees.
  * The judge is a distilled 2B model (gold AP 0.769), not ground truth, and its
    class labels are argmax of a three-way distribution. Both the argmax share and
    the mean probability are reported; read differences between feeds, not the
    absolute level.
  * A feed's cursor walks backwards in time, so the sample is the newest N posts
    the feed returns, at collection time.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "research", "data", "feed", "audit")
REPORT = os.path.join(ROOT, "docs", "reports", "REPORT_feed_audit.md")
APPVIEW = os.environ.get("LOC_APPVIEW", "https://public.api.bsky.app")
ADAPTER = os.path.join(ROOT, "research", "data", "laya_loc", "decider_short")
SCORES = os.path.join(DATA, "scores_decider_ft.jsonl")
UNION = os.path.join(DATA, "union_scorable.jsonl")

# Named by the user, plus the repo's own feed. feed-forward is the deployed LoC feed.
FIXED = {
    "feed-forward": "at://did:plc:xo2uz55ugf2yguhjryahasuf/app.bsky.feed.generator/feed-forward",
    "mysky-your-feed": "at://did:plc:wrmpulygwvuhjn2c3jbalgqj/app.bsky.feed.generator/your-feed",
    "eurosky-fu": "at://did:plc:ooensn4mr5mhznzypvxelfa3/app.bsky.feed.generator/fu",
}


def api(path, params):
    url = f"{APPVIEW}/xrpc/{path}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def feed_list(n_popular):
    feeds = dict(FIXED)
    if n_popular:
        for f in api("app.bsky.unspecced.getPopularFeedGenerators", {"limit": n_popular})["feeds"]:
            if f["uri"] not in feeds.values():
                feeds[slug(f["displayName"]) or f["uri"].rsplit("/", 1)[-1]] = f["uri"]
    return feeds


def path_posts(name):
    return os.path.join(DATA, f"posts_{name}.jsonl")


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_collect(a):
    os.makedirs(DATA, exist_ok=True)
    feeds = feed_list(a.popular)
    with open(os.path.join(DATA, "feeds.json"), "w") as f:
        json.dump(feeds, f, indent=1)
    if os.environ.get("LOC_AUTH") == "1":
        sys.path.insert(0, HERE)
        import collect_feed_posts as C
        C.login()  # one session for every feed: createSession is rate-limited
        os.environ["LOC_AUTH_JWT"], os.environ["LOC_AUTH_PDS"] = C.AUTH.split()[1], C.BASE
    for name, uri in feeds.items():
        if len(load_jsonl(path_posts(name))) >= a.max_posts:
            continue
        print(f"=== {name}  {uri}")
        subprocess.run([sys.executable, os.path.join(HERE, "collect_feed_posts.py"),
                        "--feed", uri, "--out", path_posts(name),
                        "--max-posts", str(a.max_posts)], check=False)


def cmd_score(a):
    feeds = json.load(open(os.path.join(DATA, "feeds.json")))
    union, seen = [], set()
    for name in feeds:
        for r in load_jsonl(path_posts(name)):
            if r["enough_text"] and r["text"].strip() and r["uri"] not in seen:
                seen.add(r["uri"])
                union.append(r)
    with open(UNION, "w", encoding="utf-8") as f:
        for r in union:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(union)} unique scorable posts across {len(feeds)} feeds")
    env = dict(os.environ, LOC_DECIDER_RUBRIC="short", LOC_DECIDER_ADAPTER=ADAPTER,
               PYTHONPATH=os.path.join(ROOT, "training"))
    env.setdefault("HF_HOME", "/Volumes/Daten/huggingface")
    subprocess.run([os.path.join(ROOT, "tmp_venv", "bin", "python"),
                    os.path.join(ROOT, "research", "evals", "score_loc_decider.py"),
                    UNION, SCORES], env=env, check=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmd_report(a):
    feeds = json.load(open(os.path.join(DATA, "feeds.json")))
    sc = {r["uri"]: r for r in load_jsonl(SCORES) if "error" not in r}
    rows = []
    for name, uri in feeds.items():
        posts = load_jsonl(path_posts(name))
        got = [sc[p["uri"]] for p in posts if p["uri"] in sc]
        langs = defaultdict(int)
        for p in posts:
            langs[(p["langs"] or ["?"])[0]] += 1
        top_lang = max(langs, key=langs.get) if langs else "-"
        n = len(got)
        if n == 0:
            rows.append(dict(name=name, served=len(posts), n=0, top_lang=top_lang))
            continue
        cls = [g["ts_class"] for g in got]
        pi = np.array([g["ts_class_probs"]["internal"] for g in got])
        pe = np.array([g["ts_class_probs"]["external"] for g in got])
        ki, ke = cls.count("internal"), cls.count("external")
        # The feed consumes the top of a ranking, so the feed's own first 25 matter
        # as much as its average: draw_rank is the order the feed served them in.
        first = sorted((p for p in posts if p["uri"] in sc), key=lambda p: p["draw_rank"])[:25]
        f_ext = np.mean([sc[p["uri"]]["ts_class"] == "external" for p in first])
        f_int = np.mean([sc[p["uri"]]["ts_class"] == "internal" for p in first])
        rows.append(dict(
            name=name, served=len(posts), n=n, top_lang=top_lang,
            internal=ki / n, int_ci=wilson(ki, n), external=ke / n, ext_ci=wilson(ke, n),
            neutral=cls.count("neutral") / n, p_int=pi.mean(), p_ext=pe.mean(),
            net=pi.mean() - pe.mean(), score=np.mean([g["ts_score"] for g in got]),
            first25_int=f_int, first25_ext=f_ext))

    scored = sorted((r for r in rows if r["n"]), key=lambda r: -r["net"])
    empty = [r for r in rows if not r["n"]]
    lines = ["| feed | lang | n | internal | external | neutral | mean P(int)−P(ext) | first-25 int/ext |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in scored:
        lines.append(f"| {r['name']} | {r['top_lang']} | {r['n']} "
                     f"| {r['internal']:.0%} [{r['int_ci'][0]:.0%}–{r['int_ci'][1]:.0%}] "
                     f"| {r['external']:.0%} [{r['ext_ci'][0]:.0%}–{r['ext_ci'][1]:.0%}] "
                     f"| {r['neutral']:.0%} | {r['net']:+.3f} "
                     f"| {r['first25_int']:.0%} / {r['first25_ext']:.0%} |")
    for r in empty:
        lines.append(f"| {r['name']} | {r['top_lang']} | 0 | – | – | – | – | served {r['served']} posts |")
    table = "\n".join(lines)
    print(table)

    allp = np.concatenate([[g["ts_class"] for g in sc.values()]])
    md = f"""# Feed audit: internal vs. external locus of control

Judge: fine-tuned decider-2b (`decider_ft`, short rubric), local weights. Posts are
what each feed returned to an **anonymous** `getFeed` call, restricted to posts the
deployment would embed (≥ 6 words). Classes are the argmax of the decider's
three-way distribution; brackets are 95% Wilson intervals. Sorted by mean
P(internal) − P(external).

{table}

Corpus-wide across {len(sc)} unique scored posts: {np.mean(allp == 'internal'):.0%} internal,
{np.mean(allp == 'external'):.0%} external, {np.mean(allp == 'neutral'):.0%} neutral.

## Read this before quoting a number

- The judge is a distilled model (gold AP 0.769 vs. the paid teacher's 0.788), not
  ground truth. Compare feeds with each other; do not read the absolute shares as
  the prevalence of anything.
- Anonymous requests: personalised feeds show what a viewer with no graph gets.
- Samples are the newest N posts each feed returned at collection time and differ in
  language mix (column `lang` is the most common first-listed language), and the
  decider was validated on a mixed EN/DE gold set only.
- "first-25" is the first 25 posts in the order the feed served them, n = 25, so each
  step is 4 points: treat it as indicative.
"""
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--popular", type=int, default=20)
    c.add_argument("--max-posts", type=int, default=200)
    sub.add_parser("score")
    sub.add_parser("report")
    a = ap.parse_args()
    {"collect": cmd_collect, "score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
