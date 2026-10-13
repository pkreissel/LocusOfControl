// Final feed order of each dumped pool, at a sweep of locWeight values.
//
// Continues index.ts from the paid scores on: the Jina gate for bypassed posts,
// fuseByRank, the sort, the MAX_FEED_LENGTH cut, then the final order through the
// production blendLocAndEngagement. Selection is identical across weights, so each
// pool yields one list and several orderings of it. Not reproduced: germanFirst
// (reader language), since these readers have no Accept-Language to go on.
//
// Run from bunny_feed/:
//   npx tsx ../research/feed_order/order_eval.ts pools_<date>
// Reads pools_<date>.jsonl + pools_<date>_jina.jsonl, writes pools_<date>_orders.jsonl.

import * as fs from 'fs';
import * as path from 'path';
import { FEEDS } from '../../bunny_feed/src/feeds';
import { fuseByRank, blendLocAndEngagement } from '../../bunny_feed/src/fuse';

const DATA = path.join(__dirname, '..', 'data', 'feed_order');
const MAX_FEED_LENGTH = 150;   // index.ts
const WEIGHTS = [0, 0.25, 0.5, 0.65, 0.8, 1];
const feed = FEEDS['feed-forward'];

// Copy of index.ts generatorRank (index.ts cannot be imported: it serves at load).
function generatorRank(cands: any[]): number[] {
  const groups = new Map<string, number[]>();
  cands.forEach((c, i) => {
    const key = typeof c.generator_name === 'string' ? c.generator_name : '?';
    const g = groups.get(key);
    if (g) g.push(i); else groups.set(key, [i]);
  });
  const pct = new Array<number>(cands.length).fill(0.5);
  for (const idxs of groups.values()) {
    if (idxs.length === 1) continue;
    idxs.sort((a, b) => (cands[b].geScore ?? -Infinity) - (cands[a].geScore ?? -Infinity));
    const last = idxs.length - 1;
    idxs.forEach((ci, pos) => { pct[ci] = (last - pos) / last; });
  }
  return pct;
}

const readJsonl = (f: string) => fs.readFileSync(f, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));

const stem = process.argv[2];
const rows = readJsonl(path.join(DATA, `${stem}.jsonl`));
const jina = new Map<string, number>(readJsonl(path.join(DATA, `${stem}_jina.jsonl`)).map(r => [r.uri, r.jinaScore]));

const pools = new Map<string, any[]>();
for (const r of rows) (pools.get(r.pool) ?? pools.set(r.pool, []).get(r.pool)!).push(r);

const out: string[] = [];
for (const [pool, cands] of pools) {
  const sel: any[] = [];
  for (const c of cands) {
    const p = jina.get(c.uri) ?? null;
    if (c.bypassed && (p === null || p < feed.threshold)) continue;
    sel.push({ ...c, jinaScore: p, score: p ?? c.minilmScore ?? 0 });
  }
  fuseByRank(sel);
  sel.sort((a, b) => (b.score || 0) - (a.score || 0));
  if (sel.length > MAX_FEED_LENGTH) sel.length = MAX_FEED_LENGTH;
  const selRank = generatorRank(sel);
  const geRankOf = new Map(sel.map((c, i) => [c.uri, selRank[i]]));

  for (const w of WEIGHTS) {
    const list = [...sel];
    blendLocAndEngagement(list, generatorRank(list), w);
    out.push(JSON.stringify({
      pool, weight: w,
      uris: list.map(c => c.uri),
      geRank: list.map(c => geRankOf.get(c.uri)),
      fused: list.map(c => c.score),
      jina: list.map(c => c.jinaScore),
      generator: list.map(c => c.generator_name),
    }));
  }
  console.log(`${pool}: ${cands.length} paid -> ${sel.length} in feed`);
}
fs.writeFileSync(path.join(DATA, `${stem}_orders.jsonl`), out.join('\n') + '\n');
console.log(`wrote ${stem}_orders.jsonl`);
