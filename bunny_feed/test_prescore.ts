// Checks that the edge prescore job calibrates TypeSafe answers the way the
// offline pipeline does.
//
// calibrate_typesafe_to_vector.py fitted a quantile map on the silberskeets pool
// and wrote scores_push_silberskeets.jsonl; export_prescore_assets.py froze the
// same pool into prescore_calibration.json. Feeding that pool's own TypeSafe
// answers back through prescore.ts' bandKey + calibrate must land on the offline
// scores, up to how ties are broken (the offline sort broke them arbitrarily,
// calibrate() takes the middle of a tied block), and must flip no gate verdict
// that a tie cannot explain.
//
// Run from bunny_feed/:  npx tsx test_prescore.ts
import * as fs from 'fs';
import * as path from 'path';
import { bandKey, calibrate, textKey } from './src/prescore';
import { resolveFeed } from './src/feeds';
import calibration from './src/prescore_calibration.json';

const DATA = path.resolve(__dirname, '../research/data/feed');
const load = (f: string) => fs.readFileSync(path.join(DATA, f), 'utf-8')
  .split('\n').filter(l => l.trim()).map(l => JSON.parse(l));

let failures = 0;
const check = (ok: boolean, msg: string) => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failures++;
};

const feed = resolveFeed('feed-forward');
check(calibration.vector_version === feed.scorer.version,
  `calibration fitted for the deployed vector (${calibration.vector_version} vs ${feed.scorer.version})`);

// Synthetic boundaries of the ladder.
const n = calibration.ladder.length;
check(calibrate([-1, 0, 0]) === calibration.ladder[0], 'a key below every reference key gets the lowest probability');
check(calibrate([2, 1, 9]) === calibration.ladder[n - 1], 'a key above every reference key gets the highest probability');
check(calibrate([0, 0.1, 1]) < calibrate([1, 0.1, 1]), 'external is floored below any non-external post');

// The band floor itself, against calibrate_typesafe_to_vector.KEYS['band'].
check(bandKey({ internal: 0.3, external: 0.55, neutral: 0.15 }, 2)[0] === 0, 'external majority floors');
check(bandKey({ internal: 0.3, external: 0.35, neutral: 0.35 }, 2)[0] === 1, 'external tied with neutral does not floor');
check(bandKey({ internal: 0.9, external: 0.05 }, 3)[0] === 1, 'missing neutral is read as the remainder');

// Against the offline pipeline's output.
const ts = new Map(load('scores_loc_typesafe_silberskeets.jsonl')
  .filter((r: any) => !r.error).map((r: any) => [r.uri, r]));
const push = load('scores_push_silberskeets.jsonl');
const gate = feed.threshold;
let diffSum = 0, maxDiff = 0, flips = 0, flipsOutsideTies = 0, nCmp = 0;
for (const p of push) {
  const t: any = ts.get(p.uri);
  if (!t) continue;
  const key = bandKey(t.ts_class_probs, t.ts_score);
  const edge = calibrate(key);
  const d = Math.abs(edge - p.vector_prob);
  diffSum += d; maxDiff = Math.max(maxDiff, d); nCmp++;
  if ((edge >= gate) !== (p.vector_prob >= gate)) {
    flips++;
    // A flip is only explained by a tie if the offline score sits inside the
    // range the tied block spans on the ladder.
    const k4 = [key[0], Math.round(key[1] * 1e4) / 1e4, Math.round(key[2] * 1e4) / 1e4];
    const tied = calibration.keys.map((k, i) => [k, i] as const)
      .filter(([k]) => k[0] === k4[0] && k[1] === k4[1] && k[2] === k4[2]).map(([, i]) => calibration.ladder[i]);
    // 1e-6: the exported ladder is rounded to six decimals.
    const inBlock = tied.length > 1
      && p.vector_prob >= Math.min(...tied) - 1e-6 && p.vector_prob <= Math.max(...tied) + 1e-6;
    if (!inBlock) flipsOutsideTies++;
  }
}
console.log(`\n  ${nCmp} posts: mean |edge - offline| ${(diffSum / nCmp).toFixed(5)}, max ${maxDiff.toFixed(5)}`);
console.log(`  gate ${gate}: ${flips} verdicts differ, ${flipsOutsideTies} of them outside a tied block`);
check(nCmp === push.length, 'every offline push row has a TypeSafe answer');
check(diffSum / nCmp < 0.005, 'mean difference under 0.005');
check(flipsOutsideTies === 0, 'no gate flip that a tie does not explain');

// text_key parity with score_loc_typesafe.text_key(), which the local ledger
// dedupes on. Expected values computed with the Python function.
(async () => {
  const fixtures: [string, string][] = [
    ['Hallo  Welt\n foo', '067148e2fc2979af'],
    ['  a\tb ', 'c8687a08aa5d6ed2'],
    ['Ünïcode\u00a0space x', 'f733f3eca247c87f'],
  ];
  for (const [text, want] of fixtures) {
    check(await textKey(text) === want, `text_key of ${JSON.stringify(text)} matches Python`);
  }
  console.log(failures ? `\n${failures} failed` : '\nall passed');
  process.exit(failures ? 1 : 0);
})();
