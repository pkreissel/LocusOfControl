import { createScorer, type Scorer, type VectorData } from './vector';
import locVectorData from './vector.json';
import helpfulVectorData from './vector_helpful.json';

// --- Feed registry ---------------------------------------------------------
//
// One Bunny deployment serves every feed published under this did:web. Bluesky
// tells them apart by the `feed` query parameter on getFeedSkeleton, whose rkey
// is the record key used when publishing (publish_feed.py). Each entry pairs
// that rkey with the trained vector that ranks it and the probability threshold
// calibrated for that vector.
//
// A threshold is NOT portable between vectors. It is a sigmoid probability, and
// each retrain moves the score distribution underneath it — that is how the LoC
// gate once sat at 0.55 and rejected literally nothing. Every number below was
// measured against its own vector's distribution; recalibrate on every retrain.

export type FeedConfig = {
  /** Record key of the app.bsky.feed.generator record. */
  rkey: string;
  /** Human label, used only in logs and diagnostics headers. */
  name: string;
  /** Scorer built from this feed's trained vector. */
  scorer: Scorer;
  /**
   * Minimum calibrated probability a candidate must reach to survive into the
   * reranker. Tuned as a keep-rate against 500 GE candidates, not as a
   * precision target: the vector is a pre-filter and GE's ranker (blending the
   * score via RANK_SIM_WEIGHT) decides the final order.
   */
  threshold: number;
  /**
   * Force the unauthenticated path, ignoring the caller's DID. Used by feeds
   * that are meant to be identical for everyone, so one cached copy is correct
   * to serve to all readers.
   */
  forcePublic?: boolean;
  /**
   * Number of parallel GE candidate calls. GE caps num_candidates at 1000 per
   * request, and the popularity/network_likes/two_tower mix is effectively
   * deterministic (repeat calls overlap 99%), so a bigger pool has to come from
   * `random_posts`, whose calls overlap 0%. Call 1 is the normal mix; the rest
   * are random_posts. Measured: fan 4 -> ~4000 unique in ~2.2s, fan 8 -> ~8000
   * but ~13s, which is too slow for a feed request. Default 1 = today's
   * behavior, one call, no random_posts.
   */
  fanout?: number;
  /**
   * Candidates requested per GE call. GE's ceiling is 1000; the default stays
   * at the original 500 so feeds that did not ask for a bigger pool keep their
   * existing cost and behavior.
   */
  numCandidates?: number;
  /**
   * Cheap text gate applied to candidates BEFORE embedding. Free (no API call)
   * and the only thing that makes a large fanned-out pool affordable: it cuts
   * ~4000 candidates to the few hundred worth paying Jina for.
   */
  candidateFilter?: RegExp;
  /**
   * What decides the final order. 'ge' hands survivors to GE's reranker (the
   * original behavior). 'score' orders by this feed's own vector score and
   * skips the rerank call — correct when the vector *is* the editorial
   * criterion, since GE's ranker optimizes engagement and reorders almost
   * independently of it (measured Spearman 0.277 against the assist score).
   */
  rankBy?: 'ge' | 'score';
};

// Locus-of-control axis: posts showing agency over posts showing helplessness.
//
// 0.88 was recalibrated for the vector retrained in 4a9286e, which shifted the
// score distribution up: its cosines span roughly -0.22..+0.25, so with scale
// 6.542 / bias 1.807 nothing scores below ~0.59 and the previous 0.55 rejected
// *nothing* — measured over 17.5k dataset texts and 993 live two-tower
// candidates alike. 0.88 restores the "passes ~50%" intent: it keeps 47% of
// corpus texts and 34% of live candidates (~172 of 500, still clear of
// MAX_RERANK_CANDIDATES), and on the held-out validation set keeps 76% of
// internal posts against 26% of external ones. Going to 0.90 sharpens that to
// 54%/12% but leaves only ~104 live candidates, which would starve the
// reranker — raise num_candidates first if you want that operating point.
const LOC: Omit<FeedConfig, 'rkey' | 'name'> = {
  scorer: createScorer(locVectorData as VectorData),
  threshold: 0.88,
};

export const FEEDS: Record<string, FeedConfig> = {
  'feed-forward': { rkey: 'feed-forward', name: 'Feed Forward', ...LOC },
  // Same vector, but served identically to everyone (no personalization).
  'selbstwirksamkeit': {
    rkey: 'selbstwirksamkeit',
    name: 'Selbstwirksamkeit',
    ...LOC,
    forcePublic: true,
  },
  // Assist axis: genuine, situated requests for help or advice that a reader
  // could actually answer, over questions that lead nowhere.
  //
  // 0.90 keeps ~24% of a firehose-random sample — about 120 of 500 GE
  // candidates, which is where MAX_RERANK_CANDIDATES sits — while retaining
  // 94.9% of the posts an independent Gemini-labelled set called helpful.
  // 0.92 would keep only ~6% (~31 candidates) and starve the reranker; raise
  // num_candidates first if you want that sharper operating point.
  //
  // The candidate pipeline matters as much as the vector here. Served straight
  // off GE's default pool this feed returned 0 genuine requests in 30 posts:
  // only 20% of what GE surfaces even contains a question mark, because its
  // generators optimize for engagement and a real help request usually sits
  // unanswered. The vector is fine — it scores AUC 0.867 on a question-filtered
  // pool — it was just being asked to find ~7 needles in 500 general posts.
  //
  // So: fan out to ~4000 candidates, keep only the question-bearing ones (~304,
  // versus 78 from a single call), and order by the assist score rather than by
  // GE's engagement ranker.
  'helpful-feed': {
    rkey: 'helpful-feed',
    name: 'Helpful Feed',
    scorer: createScorer(helpfulVectorData as VectorData),
    threshold: 0.90,
    fanout: 4,
    numCandidates: 1000,
    candidateFilter: /\?/,
    rankBy: 'score',
  },
};

// Requests for an unknown rkey still have to be answered with something, and the
// locus-of-control feed is the original one this deployment served.
export const DEFAULT_FEED = FEEDS['feed-forward'];

export function resolveFeed(rkey: string): FeedConfig {
  return FEEDS[rkey] ?? DEFAULT_FEED;
}

// Every distinct scoring vector in the registry, de-duplicated by fingerprint.
//
// Embedding a post is the expensive step (a billed Jina call); turning that
// embedding into one feed's probability is a dot product. So when a candidate
// misses the cache we score it against ALL of these at once and store a row per
// version, and a later request for a different feed finds it already cached
// instead of paying Jina for the same text twice. Feeds sharing a vector
// (feed-forward and selbstwirksamkeit) collapse to one entry here — writing the
// same (uri, version) twice in one statement is an error, not a no-op.
export const ALL_SCORERS: Scorer[] = (() => {
  const byVersion = new Map<string, Scorer>();
  for (const f of Object.values(FEEDS)) {
    if (!byVersion.has(f.scorer.version)) byVersion.set(f.scorer.version, f.scorer);
  }
  return [...byVersion.values()];
})();
