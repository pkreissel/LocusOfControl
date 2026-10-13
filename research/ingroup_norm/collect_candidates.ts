// Pull raw feed-forward candidates from GreenEarth for the in-group norm study.
//
// Same GE request as the feed (and research/feed_order/dump_pools.ts), but the
// pool is cut at the minWords gate only: no MiniLM shortlist and no German
// bypass, because those select on LoC and would hand the study a pool already
// tilted toward agency posts. Dedup across pools by URI.
//
// Run from bunny_feed/ (for its node_modules):
//   npx tsx ../research/ingroup_norm/collect_candidates.ts <per-pool> <handle-or-did> ...

import * as fs from 'fs';
import * as path from 'path';
import { config } from 'dotenv';
import { FEEDS } from '../../bunny_feed/src/feeds';
import { hasEnoughText } from '../../bunny_feed/src/text';

const HERE = __dirname;
const OUT_DIR = path.join(HERE, '..', 'data', 'ingroup_norm');
config({ path: path.join(HERE, '..', '..', 'bunny_feed', '.env') });

const feed = FEEDS['feed-forward'];

async function resolveDid(who: string): Promise<string> {
  if (who.startsWith('did:')) return who;
  const r = await fetch(`https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?handle=${encodeURIComponent(who)}`);
  if (!r.ok) throw new Error(`resolveHandle ${who}: ${r.status}`);
  return (await r.json()).did;
}

async function candidates(did: string, n: number) {
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
      num_candidates: n, max_age_hours: 72,
    }),
  });
  if (!r.ok) throw new Error(`GE generate ${did}: ${r.status}`);
  return ((await r.json()).candidates || []) as any[];
}

async function main() {
  const [per, ...who] = process.argv.slice(2);
  if (!who.length) throw new Error('usage: collect_candidates.ts <per-pool> <handle-or-did> ...');
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const out = path.join(OUT_DIR, `candidates_${new Date().toISOString().slice(0, 10)}.jsonl`);
  const seen = new Set<string>();
  const rows: any[] = [];
  for (const w of who) {
    const did = await resolveDid(w);
    const raw = await candidates(did, Number(per));
    let kept = 0;
    for (const c of raw) {
      const text = (c?.content || '') as string;
      if (!c?.at_uri || seen.has(c.at_uri) || !hasEnoughText(text, feed.minWords ?? 0)) continue;
      seen.add(c.at_uri);
      rows.push({ pool: w, uri: c.at_uri, text, generator_name: c.generator_name,
                  geScore: typeof c.score === 'number' ? c.score : null });
      kept++;
    }
    console.log(`${w}: ${raw.length} raw -> ${kept} kept`);
  }
  fs.writeFileSync(out, rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  console.log(`wrote ${rows.length} -> ${out}`);
}

main();
