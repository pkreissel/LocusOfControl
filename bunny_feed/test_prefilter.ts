// bypassBudget(): how much of the paid stage the language detour may spend.
//
// Run: npx tsx test_prefilter.ts
//
// The numbers here are the two ends of the range this deployment actually draws.
// A live pool is ~1.4% German (10 of 708 measured), the labelled corpus is 22.6%,
// and one German-reading user's own liked posts are 71% — and a single fixed cap
// is right at none of them at once. That is what the share arithmetic is for, and
// what these cases pin.

import { bypassBudget } from './src/feeds';

const SHORTLIST = 150, FLOOR = 50, MAX = 150;
let failures = 0;
function eq(name: string, got: number, want: number) {
  const ok = got === want;
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${name} — got ${got}, want ${want}`);
  if (!ok) failures++;
}
const budget = (n: number, rest: number) => bypassBudget(SHORTLIST, n, rest, FLOOR, MAX);

console.log('the pools this actually sees');
// 1.4% German: proportional is 2, the floor buys all ten for ~7% more Jina.
eq('live pool, 10 of 708', budget(10, 698), 10);
// 22.6%: proportional (44) is under the floor, so the floor still decides.
eq('corpus mix, 160 of 706', budget(160, 546), 50);
// 50%: proportional is the full shortlist — parity, and the ceiling exactly binds.
eq('half German, 353 of 706', budget(353, 353), 150);
// 71%: proportional would be 367; the ceiling holds the paid stage to 2x.
eq('one reader\'s likes, 501 of 706', budget(501, 205), 150);

console.log('\nboundaries');
eq('nothing flagged spends nothing', budget(0, 706), 0);
eq('never more than there are to take', budget(3, 700), 3);
eq('an all-German pool falls back to the ceiling', budget(400, 0), 150);
eq('a disabled detour (max 0) spends nothing', bypassBudget(SHORTLIST, 100, 600, 50, 0), 0);
eq('floor above ceiling, plenty to take: ceiling wins',
   bypassBudget(SHORTLIST, 400, 600, 200, 150), 150);
eq('floor above both ceiling and supply: supply wins',
   bypassBudget(SHORTLIST, 100, 600, 200, 150), 100);
eq('empty pool', budget(0, 0), 0);

console.log('\nthe property the arithmetic exists for');
// Being on the German side of the split must not itself cost you paid slots.
for (const [n, rest] of [[160, 546], [353, 353], [80, 626]] as const) {
  const got = budget(n, rest) / n;
  const want = Math.min(1, SHORTLIST / rest);
  const ok = got >= Math.min(want, 1) - 1e-9 || budget(n, rest) === MAX;
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${n}/${rest}: German ${(got * 100).toFixed(1)}% vs rest ${(want * 100).toFixed(1)}%`);
  if (!ok) failures++;
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
