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
  'helpful-feed': {
    rkey: 'helpful-feed',
    name: 'Helpful Feed',
    scorer: createScorer(helpfulVectorData as VectorData),
    threshold: 0.90,
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
