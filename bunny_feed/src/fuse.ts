// Combining two vectors' verdicts on the same post.
//
// Lives apart from index.ts so it can be exercised directly by a test; the edge
// entrypoint calls serve() at module load and cannot be imported.

/**
 * The scoring fields fuseByRank reads and writes. Structural, so index.ts's
 * Candidate satisfies it without this module knowing anything about candidates.
 */
export type Fusable = {
  minilmScore?: number | null;
  jinaScore?: number | null;
  score?: number;
  [key: string]: any;
};

/**
 * Reorder a shortlist by the mean of its two normalised ranks, writing the result
 * back into each candidate's `score`.
 *
 * Ranks rather than raw scores because the two probabilities come from different
 * vectors with different calibrations — averaging them directly would just let
 * whichever is more spread out win. Fusing is worth a call because the two spaces
 * disagree: Spearman 0.46 on live candidates, sharing only 9 of their top 50. On
 * the validation set the fused order beat both (AUC 0.826 against 0.795 and 0.794,
 * external leak 0.245 against 0.264 and 0.286), and on live posts it held MiniLM's
 * 62% internal while cutting external leak from 12% to 6%.
 *
 * A candidate with only one usable score keeps that rank on both sides, so it is
 * ranked on the signal it has rather than buried by the one it lacks. That covers
 * both directions: a failed Jina chunk (a Jina outage degrades the order to the
 * MiniLM one rather than sinking every affected post) and a candidate that skipped
 * the free stage because the free scorer could not read its language (see
 * prefilter.bypass in feeds.ts — those arrive with minilmScore null by
 * construction, not by failure, and are already past the paid gate).
 *
 * Known property, not a bug: the mean of ranks is linear, so "middling in both"
 * ties exactly with "top of one, bottom of the other" (ranks 1+1 == 0+2). A
 * consistency-rewarding rule such as reciprocal rank fusion would break that tie
 * in favour of agreement, and may well be better — but every number quoted above
 * was measured with the plain mean, so switching the rule would mean re-running
 * that evaluation rather than inheriting its results. See REPORT_ge_embeddings.md.
 */
export function fuseByRank(cands: Fusable[]): void {
  const n = cands.length;
  if (n === 0) return;
  const ranksBy = (pick: (c: Fusable) => number | null | undefined): number[] => {
    const order = cands.map((_, i) => i).sort((a, b) => {
      const va = pick(cands[a]), vb = pick(cands[b]);
      return (vb ?? -Infinity) - (va ?? -Infinity);
    });
    const r = new Array<number>(n);
    order.forEach((ci, pos) => { r[ci] = pos; });
    return r;
  };
  const rMini = ranksBy(c => c.minilmScore);
  const rJina = ranksBy(c => c.jinaScore);
  for (let i = 0; i < n; i++) {
    const m = cands[i].minilmScore == null ? rJina[i] : rMini[i];
    const j = cands[i].jinaScore == null ? rMini[i] : rJina[i];
    // Best rank 0 -> score 1, worst -> ~0, so `score` stays "higher is better" for
    // the sort below and for GE's candidate_score model.
    cands[i].score = 1 - ((m + j) / 2) / n;
  }
}
