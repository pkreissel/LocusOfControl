"""Summarise the feed_logs table: latency, failures, and the gaps between them.

One row per getFeedSkeleton request, written by the edge script (see
bunny_feed/src/schema.ts). This reads the live Turso DB over its HTTP API --
read-only, no libsql client needed -- and answers the questions the [feed]
console line could not: how slow the tail is, which stage owns it, how often a
request failed and why.

    python feed_log_stats.py [--hours 24] [--errors-only]

Note what this cannot show. A row is written when a request *finishes*, so a
request the platform killed mid-flight leaves no row at all. That is what the
GAPS section is for: an unusually long stretch with no rows at all, next to
neighbouring traffic, is the shape a hard stall leaves behind.
"""
import argparse, json, os, sys, urllib.request
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv('bunny_feed/.env'); load_dotenv('.env')


def query(sql, args=()):
    url = os.environ.get('PROD_DATABASE_URL') or os.environ.get('BUNNY_DATABASE_URL')
    token = os.environ.get('PROD_DATABASE_AUTH_TOKEN') or os.environ.get('BUNNY_DATABASE_AUTH_TOKEN')
    if not url:
        sys.exit("no PROD_DATABASE_URL in bunny_feed/.env")
    http = url.replace('libsql://', 'https://').rstrip('/')
    body = {"requests": [
        {"type": "execute", "stmt": {"sql": sql,
         "args": [{"type": "integer", "value": str(a)} if isinstance(a, int)
                  else {"type": "text", "value": str(a)} for a in args]}},
        {"type": "close"}]}
    req = urllib.request.Request(http + '/v2/pipeline', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read())
    first = res['results'][0]
    if first.get('type') == 'error':
        sys.exit(f"query failed: {first['error']['message']}")
    rows = first['response']['result']['rows']
    cols = [c['name'] for c in first['response']['result']['cols']]
    out = []
    for row in rows:
        rec = {}
        for c, cell in zip(cols, row):
            v = cell.get('value')
            if cell.get('type') == 'integer' and v is not None: v = int(v)
            elif cell.get('type') == 'float' and v is not None: v = float(v)
            elif cell.get('type') == 'null': v = None
            rec[c] = v
        out.append(rec)
    return out


def pct(vals, p):
    if not vals: return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))]


def ts(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=int, default=24)
    ap.add_argument('--errors-only', action='store_true')
    a = ap.parse_args()
    since = int(__import__('time').time() * 1000) - a.hours * 3600_000

    rows = query("SELECT * FROM feed_logs WHERE ts >= ? ORDER BY ts", (since,))
    if not rows:
        print(f"no rows in the last {a.hours}h "
              "(nothing served, or the deploy that writes them is not live yet)")
        return

    print(f"=== feed_logs, last {a.hours}h: {len(rows)} requests "
          f"({ts(rows[0]['ts'])} .. {ts(rows[-1]['ts'])}) ===\n")

    bad = [r for r in rows if r.get('errors')]
    print(f"failures: {len(bad)}/{len(rows)} ({100*len(bad)/len(rows):.1f}%)")
    by_err = {}
    for r in bad:
        # One request can carry several stage failures.
        for e in str(r['errors']).split(','):
            key = e.strip()
            by_err.setdefault(key, []).append(r)
    for key, rs in sorted(by_err.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(rs):5d}  {key}")
        for r in rs[-3:]:
            print(f"           {ts(r['ts'])}  ms={r['ms']} cand={r['cand']} pass={r['pass']}")
    empty = [r for r in rows if r.get('page') == 'fresh' and (r.get('pass') or 0) == 0]
    print(f"  {len(empty):5d}  empty feeds served (fresh page, 0 posts ranked)")
    if a.errors_only:
        return

    print("\nlatency by path (ms):")
    print(f"  {'path':>14} {'n':>6} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}")
    groups = {}
    for r in rows:
        groups.setdefault(f"{r.get('cache')}/{r.get('page') or '-'}", []).append(r)
    for name, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        v = [r['ms'] for r in rs if r.get('ms') is not None]
        if not v: continue
        print(f"  {name:>14} {len(v):6d} {pct(v,50):7d} {pct(v,95):7d} {pct(v,99):7d} {max(v):7d}")

    fresh = [r for r in rows if r.get('page') == 'fresh']
    if fresh:
        print(f"\nwhere the time goes on a fresh page (n={len(fresh)}, ms):")
        print(f"  {'stage':>10} {'p50':>7} {'p95':>7} {'max':>7}")
        for col, label in [('t_seen', 'seen'), ('t_cand', 'candidates'), ('t_embed', 'embed'),
                           ('t_rank', 'rank'), ('t_store', 'store'), ('ms', 'TOTAL')]:
            v = [r[col] for r in fresh if r.get(col) is not None]
            if v:
                print(f"  {label:>10} {pct(v,50):7d} {pct(v,95):7d} {max(v):7d}")
        pool = [r for r in fresh if r.get('cand_raw') is not None]
        if pool:
            print(f"\npipeline funnel (median of {len(pool)} fresh pages):")
            for col, label in [('cand_raw', 'GE returned'), ('cand', 'past text gates'),
                               ('pre_hydrated', 'hydrated'), ('pre_kept', 'to Jina'),
                               ('pass', 'ranked')]:
                v = [r[col] for r in pool if r.get(col) is not None]
                if v: print(f"  {label:>16}: {pct(v,50)}")
            fl = [r['pre_floor'] for r in pool if r.get('pre_floor')]
            if fl: print(f"  {'shortlist floor':>16}: p={pct(fl,50):.4f}")

    # Absences. A request that never finished writes no row, so a hard stall is a
    # hole in `ts`, not a slow row -- worth seeing next to the traffic around it.
    gaps = []
    for prev, cur in zip(rows, rows[1:]):
        gaps.append((cur['ts'] - prev['ts'], prev['ts'], cur['ts']))
    gaps.sort(reverse=True)
    print("\nlargest gaps with no rows at all (a stall leaves a hole, not a slow row):")
    for g, t0, t1 in gaps[:5]:
        print(f"  {g/1000:8.1f}s  {ts(t0)} -> {ts(t1)}")


if __name__ == '__main__':
    main()
