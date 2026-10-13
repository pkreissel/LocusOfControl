// Rebuild signed-in feed-forward candidate pools offline, up to the paid stage.
//
// Reproduces index.ts steps (2)-(3a) with the feed's own modules: the GE call with
// the same body, dedup, the minWords gate, MiniLM scoring, the German bypass and
// the 150 shortlist. What it writes is the paid pool -- the posts the feed would
// send to Jina -- with everything the final order needs later (generator, GE
// score, MiniLM probability, bypass flag). Jina scoring happens locally in
// score_pools.py; order_eval.ts does the gate, fuse, cut and final ordering.
//
// Differences from a live request, all deliberate: no seen-post exclusion (no
// prod DB read), no score cache, no deadline.
//
// Run from bunny_feed/ (for its node_modules):
//   npx tsx ../research/feed_order/dump_pools.ts <handle-or-did> ...

import * as fs from 'fs';
import * as path from 'path';
import { config } from 'dotenv';
import { FEEDS, bypassBudget } from '../../bunny_feed/src/feeds';
import { decodeEmbedding } from '../../bunny_feed/src/minilm';
import { hasEnoughText } from '../../bunny_feed/src/text';

const HERE = __dirname;
const OUT_DIR = path.join(HERE, '..', 'data', 'feed_order');
config({ path: path.join(HERE, '..', '..', 'bunny_feed', '.env') });

const feed = FEEDS['feed-forward'];
const pf = feed.prefilter!;

async function resolveDid(who: string): Promise<string> {
  if (who.startsWith('did:')) return who;
  const r = await fetch(`https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?handle=${encodeURIComponent(who)}`);
  if (!r.ok) throw new Error(`resolveHandle ${who}: ${r.status}`);
  return (await r.json()).did;
}

async function pool(did: string) {
  const r = await fetch('https://api.greenearth.social/candidates/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': process.env.GREEN_EARTH_API_KEY! },
    body: JSON.stringify({
      generators: [
        { name: 'followed_users', weight: 1.0 },
        { name: 'network_likes', weight: 1.0 },
        { name: 'two_tower', weight: 1.0 },
      ],
      infill: 'popularity', user_did: did,
      num_candidates: feed.numCandidates ?? 500, max_age_hours: 72, hydrate_embeddings: true,
    }),
  });
  if (!r.ok) throw new Error(`GE generate ${did}: ${r.status}`);
  const raw: any[] = (await r.json()).candidates || [];

  const seen = new Set<string>();
  const cands = raw.filter(c => c?.at_uri && !seen.has(c.at_uri) && (seen.add(c.at_uri), true))
    .map(c => ({
      uri: c.at_uri as string, text: (c.content || '') as string,
      generator_name: c.generator_name, geScore: typeof c.score === 'number' ? c.score : null,
      emb: c.minilm_l12_embedding as string | undefined,
    }))
    .filter(c => hasEnoughText(c.text, feed.minWords ?? 0));

  const routed = cands.map(c => {
    const e = decodeEmbedding(c.emb);
    return { c, p: e ? pf.scorer.score(e).probability : null, bypass: pf.bypass!(c.text) };
  });
  const detour = routed.filter(x => x.bypass).sort((a, b) => (b.p ?? -Infinity) - (a.p ?? -Infinity));
  const taken = detour.slice(0, bypassBudget(
    pf.shortlist, detour.length, routed.length - detour.length, pf.bypassFloor ?? pf.bypassLimit!, pf.bypassLimit!));
  const kept = routed.filter(x => !x.bypass && x.p !== null)
    .sort((a, b) => (b.p as number) - (a.p as number)).slice(0, pf.shortlist);

  const rows = [
    ...taken.map(x => ({ ...x, bypassed: true })),
    ...kept.map(x => ({ ...x, bypassed: false })),
  ].map(x => ({
    uri: x.c.uri, text: x.c.text, generator_name: x.c.generator_name, geScore: x.c.geScore,
    minilmScore: x.bypassed ? null : x.p, bypassed: x.bypassed,
  }));
  return { raw: raw.length, gated: cands.length, rows };
}

async function main() {
  const who = process.argv.slice(2);
  if (!who.length) throw new Error('usage: dump_pools.ts <handle-or-did> ...');
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 10);
  const out = path.join(OUT_DIR, `pools_${stamp}.jsonl`);
  const f = fs.openSync(out, 'a');
  for (const w of who) {
    try {
      const did = await resolveDid(w);
      const p = await pool(did);
      for (const r of p.rows) fs.writeSync(f, JSON.stringify({ pool: w, did, ...r }) + '\n');
      console.log(`${w}: ${p.raw} raw -> ${p.gated} past minWords -> ${p.rows.length} paid (${p.rows.filter(r => r.bypassed).length} bypassed)`);
    } catch (e: any) {
      console.error(`${w}: ${e.message}`);
    }
  }
  fs.closeSync(f);
  console.log(`wrote ${out}`);
}

main();
