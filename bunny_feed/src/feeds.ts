import { createScorer, type Scorer, type VectorData } from './vector';
import locVectorData from './vector.json';
import locMinilmVectorData from './vector_minilm.json';

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
//
// Nor is one portable between candidate *pools*. A threshold set as a percentile
// describes the pool it was measured on, and this deployment draws two: GE's
// `popularity` mix for anonymous requests and `followed_users` for signed-in ones.
// The same 0.9356 was the 80th percentile of the first and the 87th of the second,
// which quietly cut the signed-in feed to a third of its length (see the note on
// LOC.prefilter). Prefer ranking to an absolute cut wherever a stage only has to
// choose "the best N of whatever arrived".

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
   * Minimum length a candidate must reach to be embedded at all, in words. Free
   * like `candidateFilter`, and applied before it. A handful of words carries no
   * readable position on any of these axes, so the score such a post gets is
   * noise that the threshold then promotes or drops at random -- measured on a
   * live pool, "overwhelmed" scored 0.901 and a row of four 👇 emoji scored
   * 0.884, both comfortably over the 0.88 LoC gate. Dropping them early also
   * saves their share of the Jina call. See hasEnoughText() in index.ts for how
   * the count treats URLs, emoji and space-less scripts. Unset = no gate.
   */
  minWords?: number;
  /**
   * What decides the final order. 'ge' hands survivors to GE's reranker (the
   * original behavior). 'score' orders by this feed's own vector score and
   * skips the rerank call — correct when the vector *is* the editorial
   * criterion, since GE's ranker optimizes engagement and reorders almost
   * independently of it (measured Spearman 0.277 against the assist score).
   */
  rankBy?: 'ge' | 'score';
  /**
   * Free first-pass ranking over Green Earth's hydrated MiniLM embeddings, so the
   * paid Jina stage only ever sees a shortlist.
   *
   * This is not a cheap approximation of the Jina vector. Trained on the same rows
   * through the same harness, the MiniLM probe matched it on held-out validation
   * (AUC 0.795 vs 0.794) and beat it on live posts judged blind by the validation
   * prompt: 62% internal in its top 50 against Jina's 42%. What it is worse at is
   * suppressing the negative pole (12% external vs 2%), and the two agree only at
   * Spearman 0.46 — which is exactly why `fuse` below is worth the Jina call.
   * Full evidence: REPORT_ge_embeddings.md.
   *
   * Leave undefined to keep the original single-stage behaviour, and note the
   * request falls back to it automatically if hydration stops arriving.
   */
  prefilter?: {
    /** Scorer built from a vector trained in the MiniLM space (see embed_minilm.py). */
    scorer: Scorer;
    /**
     * How many of the top-scoring candidates reach the paid stage.
     *
     * This is the entire selection rule — the pool is ranked by the prefilter
     * scorer and cut here, with no absolute floor. It is therefore also the whole
     * cost control: the paid stage never sees more than this many posts, however
     * big the pool grows. The quality floor is `threshold` above, applied to the
     * Jina score after this cut. See the note on LOC.prefilter for why an absolute
     * floor could not do this job.
     */
    shortlist: number;
  };
  /**
   * Order the shortlist by the mean of its two normalised ranks (MiniLM and Jina)
   * rather than by either score alone. Measured on the validation set, fusion beat
   * both single spaces (AUC 0.826 vs 0.795/0.794, external leak 0.245 vs
   * 0.264/0.286) and on live posts it held MiniLM's 62% internal while cutting the
   * external leak from 12% to 6%. Ignored unless `prefilter` is set.
   */
  fuse?: boolean;
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
  // Locus of control is a claim about agency, and a post has to say something
  // for that claim to be readable at all. Measured against a live 500-candidate
  // pool: 6 words cuts the pool to 396 (23 of the drops had *empty* text, which
  // was being embedded and scored) and the survivors still clear the 0.88 gate
  // 265 deep -- far above MAX_RERANK_CANDIDATES, so the reranker is not starved.
  // Going to 8 buys little (245 through) and starts cutting real one-liners.
  minWords: 6,
  // Ranks every candidate for free over GE's hydrated embeddings and hands the top
  // `shortlist` to Jina. Rank only: there is deliberately no probability floor on
  // this stage.
  //
  // There was one (0.9356, the 80th percentile of this vector's probability over
  // 2143 live candidates), and it silently became the selector instead of the
  // safety net it was documented as. A calibrated percentile describes the *pool*
  // it was measured on, and this deployment draws two different pools: anonymous
  // requests get GE's `popularity` mix under a fallback DID, signed-in requests get
  // `followed_users`. Measured 2026-09-11, the anon pool sat at p80 = 0.9457, so
  // 0.9356 kept 30% (113 of 375) and the feed filled its 100-post cap -- while the
  // signed-in pool sat at p80 = 0.9223, which made the same number the ~87th
  // percentile and kept 13%:
  //
  //   signed-in, 500 candidates -> 470 past minWords -> 59 past the floor
  //     -> 41 past the Jina gate. With 1000 seen URIs excluded: 39 -> 29.
  //
  // So a regular reader's feed ended after ~29 posts against a cap of 100, and
  // `shortlist` (150) never bound at all. Ranking instead: the same two pools give
  // 100 and 96 past the Jina gate, i.e. the cap, with the shortlist's own floor
  // landing around p = 0.90 on the MiniLM distribution.
  //
  // Nothing about that weakens the axis, because the floor was never what enforced
  // it: the paid Jina gate (`threshold` above, 0.88) is, and it still rejects a
  // third of the shortlist. Survivors' Jina probability was unchanged at p50 =
  // 0.911 -- ranking widened what the gate got to choose from, it did not lower the
  // bar. If a pool is ever thin enough that the top 150 are genuinely all rejects,
  // that gate is what says so; X-Prefilter reports the shortlist's floor
  // probability (p=) on every response so the operating point stays observable.
  //
  // 150 was chosen against MAX_RERANK_CANDIDATES (100) with room for Jina chunk
  // failures. On the labelled corpus a 20% prefilter preserved 90% of the top 30
  // that full fusion over everything would have chosen, and precision@30 was flat
  // from 15% upward -- the shortlist is not where quality is lost. It cuts the
  // paid stage from ~364 embeddings per uncached request to <=150.
  prefilter: {
    scorer: createScorer(locMinilmVectorData as VectorData),
    shortlist: 150,
  },
  fuse: true,
};

// Note: only feeds whose record points at did:web:feedforward.b-cdn.net reach
// this worker at all. The account also publishes climate-optim, but that record
// names did:web:api.graze.social and is served entirely by Graze, so nothing
// here applies to it. (fem-feed pointed at did:web:femfeed.b-cdn.net, whose
// service was returning 502; the record was deleted 2026-09-07.) DEFAULT_FEED
// therefore only ever catches a typo or an unpublished rkey.
//
// The assist axis ('helpful-feed') was tried and scrapped: GE's candidate pool
// cannot supply enough genuine help requests to build on. It caps at 1000
// candidates per request, only ~7 per 1000 match a help cue, the
// popularity/network_likes/two_tower mix repeats itself 99% across calls, and
// fanning out random_posts to ~8000 takes ~13s. src/vector_helpful.json is kept
// (unreferenced) because *.npy is gitignored, so it is the only committed copy
// of that trained vector.
export const FEEDS: Record<string, FeedConfig> = {
  'feed-forward': { rkey: 'feed-forward', name: 'Feed Forward', ...LOC },
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
// collapse to one entry here — writing the same (uri, version) twice in one
// statement is an error, not a no-op. With a single feed registered this is a
// one-element list and the pass is a no-op; it earns its keep the moment a
// second vector is added.
export const ALL_SCORERS: Scorer[] = (() => {
  const byVersion = new Map<string, Scorer>();
  for (const f of Object.values(FEEDS)) {
    if (!byVersion.has(f.scorer.version)) byVersion.set(f.scorer.version, f.scorer);
  }
  return [...byVersion.values()];
})();
