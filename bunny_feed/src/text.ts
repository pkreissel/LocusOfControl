// Text gates shared by the feed request (index.ts) and the prescore job
// (prescore.ts). Both must agree on what is worth scoring, or the job would pay
// for posts the feed then drops before looking up their score.

// Length gate behind FeedConfig.minWords: is there enough text here to be worth
// embedding? URLs are stripped first (a bare link says nothing the vector can
// read), and a whitespace token only counts as a word if it holds a letter or
// digit, so a row of emoji or ellipses counts as zero rather than several.
//
// The character escape hatch exists because whitespace tokenizing badly
// undercounts scripts that don't delimit words with spaces: measured on a live
// 500-candidate pool it rescued 10 posts the word count alone would have cut,
// 7 of them substantial Japanese or Thai posts. Counting only letters and
// digits keeps the escape from being opened by a wall of emoji.
const URL_RE = /(?:https?:\/\/|www\.)\S+/g;
const WORDY_RE = /[\p{L}\p{N}]/u;
export function hasEnoughText(text: string, minWords: number): boolean {
  const stripped = (text || '').replace(URL_RE, ' ');
  let words = 0, chars = 0;
  for (const tok of stripped.split(/\s+/)) {
    if (!WORDY_RE.test(tok)) continue;
    words++;
    for (const ch of tok) if (WORDY_RE.test(ch)) chars++;
  }
  return words >= minWords || chars >= minWords * 5;
}
