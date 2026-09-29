// fuseByRank: the ordering rule the feed serves.
//
// Run: npx tsx test_fuse.ts
//
// The case this file exists for is the third one. A German candidate reaches the
// paid stage with minilmScore null BY CONSTRUCTION (prefilter.bypass routed it
// around a scorer that cannot read it), not by failure — and before the fusion
// rule handled a missing free score, that null sorted to the bottom of the MiniLM
// ranking and dragged a post Jina had ranked first into the middle of the feed.
// Routing German around the free stage and then burying it on the free stage's
// ranking would have cancelled the whole change out.

import { fuseByRank, blendLocAndEngagement, type Fusable } from './src/fuse';

let failures = 0;
function check(name: string, ok: boolean, detail = '') {
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
}
/** Candidate ids, best first. */
function order(cands: (Fusable & { id: string })[]): string[] {
  fuseByRank(cands);
  return [...cands].sort((a, b) => (b.score || 0) - (a.score || 0)).map(c => c.id);
}

console.log('both scores present');
{
  const o = order([
    { id: 'agree-top', minilmScore: 0.99, jinaScore: 0.99 },
    { id: 'middling', minilmScore: 0.50, jinaScore: 0.50 },
    { id: 'agree-bottom', minilmScore: 0.10, jinaScore: 0.10 },
  ]);
  check('agreement orders as both spaces agree', o[0] === 'agree-top' && o[2] === 'agree-bottom', o.join(' > '));
}
{
  const cands = [
    { id: 'jina-only-likes', minilmScore: 0.10, jinaScore: 0.99 },
    { id: 'both-middling', minilmScore: 0.50, jinaScore: 0.50 },
    { id: 'mini-only-likes', minilmScore: 0.99, jinaScore: 0.10 },
  ];
  fuseByRank(cands);
  // Documented property of the plain mean of ranks: 1+1 ties with 0+2. Asserted
  // so that swapping in a consistency-rewarding rule (RRF) has to come here and
  // re-read the note in fuse.ts about what that invalidates.
  const [a, b, c] = cands.map(x => x.score);
  check('a split verdict ties with consistent middling', a === b && b === c,
        cands.map(x => `${x.id}=${x.score}`).join(' '));
}

console.log('\none score missing');
{
  const o = order([
    { id: 'de-top-of-jina', minilmScore: null, jinaScore: 0.99 },
    { id: 'en-mid', minilmScore: 0.50, jinaScore: 0.50 },
    { id: 'en-low', minilmScore: 0.40, jinaScore: 0.10 },
  ]);
  check('a bypassed candidate is ranked on the score it has', o[0] === 'de-top-of-jina', o.join(' > '));
}
{
  const o = order([
    { id: 'de-bottom-of-jina', minilmScore: null, jinaScore: 0.05 },
    { id: 'en-mid', minilmScore: 0.50, jinaScore: 0.50 },
    { id: 'en-top', minilmScore: 0.90, jinaScore: 0.90 },
  ]);
  check('...and not promoted by it either', o[2] === 'de-bottom-of-jina', o.join(' > '));
}
{
  const o = order([
    { id: 'jina-failed', minilmScore: 0.99, jinaScore: null },
    { id: 'en-mid', minilmScore: 0.50, jinaScore: 0.50 },
    { id: 'en-low', minilmScore: 0.10, jinaScore: 0.10 },
  ]);
  check('a failed Jina chunk still falls back to the free rank', o[0] === 'jina-failed', o.join(' > '));
}
{
  const cands = [
    { id: 'neither', minilmScore: null, jinaScore: null },
    { id: 'both', minilmScore: 0.50, jinaScore: 0.50 },
  ];
  const o = order(cands);
  check('a candidate with no score at all sorts last, not NaN',
        o[1] === 'neither' && Number.isFinite(cands[0].score), String(cands[0].score));
}

console.log('\nedge cases');
{
  const empty: Fusable[] = [];
  fuseByRank(empty);
  check('empty list does not throw', true);
  const one = [{ id: 'solo', minilmScore: 0.5, jinaScore: 0.5, score: 0 }];
  fuseByRank(one);
  check('single candidate gets a finite score', Number.isFinite(one[0].score), String(one[0].score));
}

console.log('\nfinal order: LoC blended with engagement');
{
  // LoC order a > b > c; engagement order c > b > a.
  const mk = () => [
    { id: 'a', score: 0.99 }, { id: 'b', score: 0.90 }, { id: 'c', score: 0.87 },
  ];
  const ge = [0, 0.5, 1];
  const at = (w: number) => { const c = mk(); blendLocAndEngagement(c, ge, w); return c.map(x => x.id).join(' > '); };
  check('weight 0 is pure engagement', at(0) === 'c > b > a', at(0));
  check('weight 1 is pure LoC', at(1) === 'a > b > c', at(1));
  check('weight 0.65 lets LoC lead', at(0.65) === 'a > b > c', at(0.65));
  check('weight 0.5 ties break on LoC score', at(0.5) === 'a > b > c', at(0.5));
  const one = [{ id: 'solo', score: 0.9 }];
  blendLocAndEngagement(one, [0.5], 0.65);
  check('single candidate survives', one[0].id === 'solo');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
