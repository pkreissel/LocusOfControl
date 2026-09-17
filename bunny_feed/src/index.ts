import { jwtDecode } from 'jwt-decode';
import { getEmbeddingsBatched, EMBEDDING_MODEL, JINA_TIMEOUT_MS } from './jina';
import { decodeEmbedding } from './minilm';
import { fuseByRank } from './fuse';
import { resolveFeed, ALL_SCORERS, bypassBudget, type FeedConfig } from './feeds';
import { recordEmbeddingUsage, recordInteractions, getSeenUris, pruneOldSeenInteractions, getCachedScores, putScores, pruneOldScores, putFeedSession, getFeedSession, getLatestFeedSession, pruneOldFeedSessions, recordFeedLog, pruneOldFeedLogs, type FeedLogRow } from './db';
import * as BunnySDK from "@bunny.net/edgescript-sdk";
import { siteResponse } from './site';

console.log("Bunny Edge Script module initialized");

// Drizzle wraps every driver failure in a `DrizzleQueryError` whose message is
// just "Failed query: <sql>" — the real cause (Turso HTTP status/body, or a
// runtime `fetch` error) is nested in `.cause`. Walk the chain so logs show why
// a query actually failed instead of the useless wrapper text.
function errChain(e: any): string {
  const parts: string[] = [];
  for (let cur = e, depth = 0; cur && depth < 6; cur = cur.cause, depth++) {
    parts.push(cur?.message ?? String(cur));
  }
  return parts.join(' <- ');
}

// Each feed's vector-probability gate now lives with its vector in feeds.ts,
// because the number is only meaningful relative to one vector's score
// distribution: the LoC gate sat at 0.55 for a while and rejected literally
// nothing after a retrain moved the distribution out from under it. See the
// calibration notes on each FeedConfig.

// Maximum number of candidates to send to the final GE reranker (rank/predict)
const MAX_RERANK_CANDIDATES = 100;

// Minimum share of candidates that must carry a hydrated embedding before the free
// prefilter is trusted with the pool. Measured coverage after the text gates was
// 100% (every un-hydrated candidate had empty text, which minWords already drops),
// so this only ever fires if GE changes what it returns — in which case the request
// pays for the old single-stage path instead of serving a feed ranked on a fraction
// of the pool.
const MIN_HYDRATION_COVERAGE = 0.8;

// Weight of the Jina vector-similarity signal in GE's reranker, relative to the
// two_tower model (fixed at 1.0). We stash each surviving candidate's similarity
// probability in its `score` field and enable GE's `candidate_score` model, so
// the ranker itself blends "learned relevance" with "semantic similarity" (and
// handles the scale normalization) instead of us fusing scores at the edge.
const RANK_SIM_WEIGHT = 2.5;
declare const process: any;
declare const Bunny: any;

// Run best-effort work *after* the response is sent. Bunny's edge runtime keeps
// the isolate alive until the promise settles (`Bunny.v1.waitUntil`), which is
// the sanctioned way to do post-response IO (cache refresh, telemetry, our table
// prunes) without adding latency to the request — unlike an un-registered
// dangling promise, whose deferred `fetch` escapes the request's async context
// and throws (see the transport note in db.ts). Callers MUST attach their own
// `.catch`: a rejected waitUntil promise still settles, but we never want it to
// surface as an unhandled rejection or break the response. Off the edge
// (local/test) there is no isolate to keep alive, so the already-running promise
// is simply left to complete unawaited.
function runAfterResponse(promise: Promise<unknown>): void {
  if (hasWaitUntil()) {
    Bunny.v1.waitUntil(promise);
  }
}

// Whether this runtime actually provides waitUntil. The SDK ships no typings for
// it (we reach for the `Bunny` global directly), so "is it there?" is only
// answerable at runtime — and every deferred write silently becomes a no-op if it
// isn't. Surfaced on the response as X-Wait-Until so a single curl confirms it on
// the live deployment, rather than inferring it from missing rows days later.
function hasWaitUntil(): boolean {
  return typeof Bunny !== 'undefined' && typeof Bunny.v1?.waitUntil === 'function';
}

function getEnv(key: string): string | undefined {
  if (typeof Bunny !== 'undefined' && Bunny.env) return Bunny.env[key];
  if (typeof process !== 'undefined' && process.env) return process.env[key];
  return undefined;
}

// The canonical public base URL the feed is published under (see
// publish_feed.py). Used to serve the did:web document's serviceEndpoint, which
// must match the origin Bluesky reaches the feed generator at. Override via
// PUBLIC_BASE_URL if the deployment ever moves; the request origin is not used
// as a fallback since it varies by isolate warm-up.
const DEFAULT_PUBLIC_BASE_URL = 'https://feedforward.b-cdn.net';
function getPublicBaseUrl(_requestUrl: URL): string {
  const configured = getEnv('PUBLIC_BASE_URL');
  if (configured) return configured.replace(/\/+$/, '');
  return DEFAULT_PUBLIC_BASE_URL;
}

type Candidate = {
  uri: string; cid?: string; text: string;
  /** Base64 embedding from GE's hydrate_embeddings; see minilm.ts. */
  minilm_l12_embedding?: string;
  /** Free prefilter score, kept so fuseByRank can read it back. */
  minilmScore?: number | null;
  /** Paid score for the same post. Null means its Jina chunk failed. */
  jinaScore?: number | null;
  [key: string]: any;
};

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
function hasEnoughText(text: string, minWords: number): boolean {
  const stripped = (text || '').replace(URL_RE, ' ');
  let words = 0, chars = 0;
  for (const tok of stripped.split(/\s+/)) {
    if (!WORDY_RE.test(tok)) continue;
    words++;
    for (const ch of tok) if (WORDY_RE.test(ch)) chars++;
  }
  return words >= minWords || chars >= minWords * 5;
}
type SkeletonBody = { feed: { post: string }[]; cursor?: string };

const SKELETON_TTL_SEC = 60;

// --- Feed request diagnostics ---------------------------------------------
// One structured line per getFeedSkeleton request (hit AND miss) so we can see
// whether each page was served personalized or fell back to the public
// (English) AppView — and, on cache hits, whether a personalized key is
// re-serving a body that was produced by an earlier public fallback.
function logFeed(fields: Record<string, unknown>) {
  try { console.log('[feed] ' + JSON.stringify(fields)); } catch { /* never break the response */ }
  // The same line, kept where it can be queried. console output is not
  // aggregatable and does not survive, and every question worth asking of these
  // numbers is an aggregate one over time ("what does the 99th percentile look
  // like", "how often does GE come back empty", "was that bad minute Jina or the
  // ranker"). Deferred past the response, so the reader pays nothing for it, and
  // best-effort inside recordFeedLog, so diagnostics can never fail a feed.
  try {
    runAfterResponse(recordFeedLog(toFeedLogRow(fields)));
    if (Math.random() < FEED_LOG_PRUNE_PROBABILITY) {
      runAfterResponse(
        pruneOldFeedLogs(Date.now() - FEED_LOG_RETENTION_MS)
          .catch((e: any) => console.error("[feed log] prune failed:", errChain(e)))
      );
    }
  } catch (e: any) {
    console.error("[feed log] dispatch failed:", errChain(e));
  }
}

// How long request diagnostics are kept, and how often a request opportunistically
// prunes past it — same pattern as the other tables, no cron. 30 days is chosen to
// be long enough to compare a bad day against the week before it.
const FEED_LOG_RETENTION_MS = 30 * 24 * 60 * 60 * 1000;
const FEED_LOG_PRUNE_PROBABILITY = 0.02;

// Map one `[feed]` line onto a feed_logs row.
//
// The paths log different shapes (a cache hit knows nothing about candidates; a
// continuation page knows nothing about embedding), so every column but ts/feed is
// nullable and missing keys stay null rather than defaulting to 0 — "this path had
// no embed stage" and "the embed stage took 0ms" must not look alike in a mean.
// Hit-path values come back off cached HTTP headers as strings, hence the coercion.
function toFeedLogRow(f: Record<string, unknown>): FeedLogRow {
  const num = (k: string): number | null => {
    const v = f[k];
    if (v === undefined || v === null || v === '') return null;
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const str = (k: string): string | null => {
    const v = f[k];
    return v === undefined || v === null ? null : String(v);
  };
  return {
    id: crypto.randomUUID(),
    ts: Date.now(),
    feed: str('feed') ?? 'unknown',
    did: str('did'),
    cache: str('cache'),
    page: str('page'),
    personalized: str('personalized'),
    path: str('path'),
    ms: num('ms'),
    gatherMs: num('gatherMs'),
    tSeen: num('tSeen'),
    tPrune: num('tPrune'),
    tCand: num('tCand'),
    tEmbed: num('tEmbed'),
    tRank: num('tRank'),
    tStore: num('tStore'),
    candRaw: num('candRaw'),
    candShort: num('candShort'),
    cand: num('cand'),
    pass: num('pass'),
    off: num('off'),
    pageLimit: num('limit'),
    feedLen: num('feedLen'),
    nextCursor: str('nextCursor'),
    cacheHit: num('cacheHit'),
    cacheMiss: num('cacheMiss'),
    jinaChunks: num('jinaChunks'),
    jinaFailed: num('jinaFailed'),
    preHydrated: num('preHydrated'),
    preKept: num('preKept'),
    preFloor: num('preFloor'),
    preFellBack: str('preFellBack'),
    preBypass: num('preBypass'),
    preBypassDropped: num('preBypassDropped'),
    waitUntil: str('waitUntil'),
    errors: str('errors'),
  };
}

// Deadlines on the two upstream calls a feed request cannot proceed without.
//
// Without one, a hung dependency does not fail — it just stops, and the request
// stalls until the platform kills the isolate. Nothing is written in that case:
// not the [feed] line, not the feed_logs row, because both are written at the end
// of a request that never reaches its end. So the failure this table most needs to
// count would be precisely the one it cannot see.
//
// A deadline converts that silence into a path the code already handles: a
// generate that gives up leaves an empty candidate pool (empty feed, already the
// behaviour when GE errors), and a rank that gives up leaves the vector's own
// order (already the behaviour when the rerank fails). Both finish the request,
// and therefore both land in feed_logs with `errors` set.
//
// Sized against the request's own budget, not each call in isolation. The whole
// pipeline has 10s; generate, embed and rank run in series inside it, so a
// per-call deadline is only honest if the deadlines can all be paid at once. 8s
// for generate could not be: it left 2s for everything downstream, which means
// the deadline would never have fired before the isolate was killed -- exactly
// the silent stall it exists to prevent.
//
// Against measured latency, generate runs 1.7-3.0s at num_candidates 1000 and
// rank ~0.5s. 4s covers the top of the generate range with margin and still
// leaves ~6s for embed, rank and the write. This is the point past which waiting
// longer serves the reader worse than serving what we already have.
const GE_GENERATE_TIMEOUT_MS = 4000;
const GE_RANK_TIMEOUT_MS = 6000;

// A generate that fails *early* leaves budget on the table, and the failures that
// actually show up here -- a dropped connection, a 502 from a rolling deploy, a
// 503 -- are the kind a second call answers. So retry immediately, with every
// attempt sharing the deadline above rather than each getting its own: the stage
// can never run longer than the request is sized against, and the only thing a
// retry spends is time the first failure did not use.
//
// Two conditions gate it, and both are about "early". The failure must be one
// that can differ next time (a 4xx is the same request being wrong, and repeating
// it just burns latency), and at least GE_GENERATE_RETRY_MIN_MS of the budget must
// remain. That floor is 3.0s, the top of the measured generate range: a retry
// either gets a window a generate has actually been observed to finish in, or it
// does not run at all -- a retry that is going to be aborted at the deadline only
// delays the empty pool the reader is already getting. Inside a 4s budget that
// makes "early" mean "failed within a second", which is what the failures worth
// retrying look like: a refused connection or an instant 502 lands in tens of
// milliseconds. A timeout is by construction not early -- it consumed the whole
// budget -- so it is never retried.
const GE_GENERATE_MAX_ATTEMPTS = 2;
const GE_GENERATE_RETRY_MIN_MS = 3000;
const GE_GENERATE_RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

// --- One budget for the whole request --------------------------------------
//
// The caps above, and JINA_TIMEOUT_MS, are per-call stall guards, each sized off
// its own call's measured latency. Summed, they bound nothing: the stages run in
// series inside a request that has 10s, so deadlines that are each individually
// payable still describe a request the platform kills before any of them fires --
// the silent stall they exist to prevent, reintroduced by the deadlines
// themselves.
//
// So the caps stay what they are ("longer than this and the call is hung") and
// the ceiling is enforced once, here: every stage takes min(its cap, what is left
// of the request). A stage that cannot get enough to finish does not start. It
// takes the failure path it already has -- empty pool, cached scores only, the
// vector's own order, no cursor -- and names itself in `errors`, which is the same
// outcome as aborting at the deadline minus the time spent waiting for it.
const REQUEST_BUDGET_MS = 10000;
// Held out of the budget for the work after the last stage: serializing the body,
// the public skeleton cache put, the response. None of that is deadlined, so it
// has to come out of the budget rather than off the end of it.
const RESPONSE_RESERVE_MS = 500;

// Floors. Below these a stage cannot finish even at the fastest it has been
// observed to run, so starting it would spend the reader's remaining time to
// reach the same failure path, later. Measured: generate 1.7-3.0s, one Jina
// round 0.7-3.9s, rank ~0.5s.
const GE_GENERATE_MIN_MS = 1700;
const JINA_MIN_MS = 700;
const GE_RANK_MIN_MS = 500;

// Cap on any single blocking DB call. The driver bounds one HTTP attempt
// (DB_REQUEST_TIMEOUT_MS) but not the retry sequence around it, and these calls
// take no AbortSignal, so they are bounded at the call site instead. An indexed
// round-trip measures ~30ms, so 1.5s only fires when the DB is not answering --
// including the cold-replica attach that has been observed to take tens of
// seconds. Losing that race is the intended outcome under a 10s budget: a cold
// attach cannot be waited out and still leave time for a feed, so the request
// continues without exclusion, or without a cursor, instead of being killed
// mid-wait having logged nothing.
const DB_STAGE_TIMEOUT_MS = 1500;

type Budget = { remaining: () => number; slice: (cap: number) => number };

function requestBudget(t0: number): Budget {
  const deadline = t0 + REQUEST_BUDGET_MS - RESPONSE_RESERVE_MS;
  const remaining = () => Math.max(0, deadline - Date.now());
  return { remaining, slice: (cap: number) => Math.min(cap, remaining()) };
}

// Bound a call that has no AbortSignal of its own -- the DB helpers -- against the
// budget. The work is not cancelled, only stopped being waited on: the timer
// callback rejects this wrapper and issues no fetch, so nothing escapes the
// request's async context (which is what breaks `fetch` on this runtime -- see the
// driver note in db.ts). Safe only because every call wrapped below already has a
// defined outcome for "did not answer in time".
function withDeadline<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  // The call is already in flight by the time we get here (it is an argument), so
  // something has to stay subscribed to it whatever happens next. Once the timer
  // wins the race nobody is, and a rejection arriving after that is an unhandled
  // one -- an isolate-level event raised by a call we had already given up on.
  // This second subscription absorbs exactly that; it is a separate branch and
  // does not take the rejection away from the caller below.
  // Nothing to log here: if the caller was still waiting it logs the rejection
  // itself, and if the timer won, the missed deadline was already logged there.
  p.catch(() => {});
  if (ms <= 0) return Promise.reject(new Error(`${label}: no budget left`));
  let timer: any;
  const guard = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label}: deadline after ${ms}ms`)), ms);
  });
  return Promise.race([p, guard]).finally(() => clearTimeout(timer));
}

// Render one upstream failure for the `errors` column. AbortSignal.timeout rejects
// with a TimeoutError, which is the whole point of the column, so name it as such
// rather than letting it read as a generic fetch failure.
function stageErr(stage: string, e: any): string {
  const name = e?.name;
  const reason = (name === 'TimeoutError' || name === 'AbortError')
    ? 'timeout'
    : (e?.message || String(e));
  return `${stage}:${String(reason).replace(/[,\s]+/g, ' ').slice(0, 120)}`;
}
function shortDid(did: string | null): string {
  return !did ? 'anon' : (did.length > 20 ? did.slice(0, 20) + '…' : did);
}
function cursorTag(c?: string): string {
  return !c ? 'START' : (c.length > 12 ? c.slice(0, 12) + '…' : c);
}

// Prune score-cache rows older than this. Candidate generation asks GE for
// max_age_hours: 72, so a post older than 3 days can never be a candidate again
// and its cached score can never be hit — retention only has to cover that
// window plus a margin for clock skew and for the window being widened. The old
// 30-day retention was ~10x wider than reachable, so ~90% of the table was rows
// that could only ever be scanned past, never read.
const SCORE_CACHE_MAX_AGE_MS = 4 * 24 * 60 * 60 * 1000; // 4 days
// Fraction of feed requests that opportunistically prune stale cache rows,
// keeping the table bounded without a separate cron.
const SCORE_PRUNE_PROBABILITY = 0.02;

// Seen-post exclusion. On each personalized cache miss we pull the user's
// most-recently-shown posts and hand them to candidate generation as
// `exclude_uris`, so paging/refreshing doesn't repeat posts they've already
// viewed. Bounded two ways: at most SEEN_EXCLUDE_LIMIT posts, and never older
// than SEEN_RETENTION_DAYS — we never go back further than the retention window.
// That same window governs the opportunistic prune of "shown" rows (engaged
// rows are kept for training), so read scope and stored data stay in lockstep.
const SEEN_EXCLUDE_LIMIT = 1000;
const SEEN_RETENTION_DAYS = 7;
const SEEN_PRUNE_PROBABILITY = 0.02;

// Cursor pagination. Page 1 runs the full generate→embed→rank pipeline (exactly
// as before) and freezes the ranked URI list in a feed_sessions row; later pages
// slice a window out of that frozen list instead of regenerating. This makes
// intra-session paging cheap and repeat-free regardless of interactionSeen
// timing, while seen-exclusion still runs at page 1 for cross-session freshness.
//
// Cursor wire format: `${sessionId}.${offset}`. sessionId is a UUID (no dots),
// so we split on the last dot. SESSION_TTL_MS must exceed the public skeleton
// cache TTL so a cached public START body keeps pointing at a still-live session.
const SESSION_TTL_MS = 15 * 60 * 1000;
const DEFAULT_PAGE_SIZE = 30;
const MAX_PAGE_SIZE = 100;
const SESSION_PRUNE_PROBABILITY = 0.02;
// Page-1 reuse window. A logged-in START request re-runs the whole pipeline
// today; within this window we instead re-serve the user's most recent frozen
// ranked list, so a reload / crash-retry / quick re-open costs one indexed DB
// read instead of generate→embed→rank. Kept short so seen-post exclusion (baked
// into the list at generation) refreshes soon; within it, re-showing the same
// top-of-feed is the expected reload behavior, not staleness.
const START_REUSE_TTL_MS = 90 * 1000;

function parseLimit(raw: string | null): number {
  const n = raw ? parseInt(raw, 10) : NaN;
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_PAGE_SIZE;
  return Math.min(n, MAX_PAGE_SIZE);
}

function encodeCursor(sessionId: string, offset: number): string {
  return `${sessionId}.${offset}`;
}

// Decode `${sessionId}.${offset}`. Returns null for START (no cursor) or any
// malformed value, so the caller treats it as a fresh page-1 request.
function decodeCursor(raw: string | undefined): { sessionId: string; offset: number } | null {
  if (!raw) return null;
  const dot = raw.lastIndexOf('.');
  if (dot <= 0 || dot === raw.length - 1) return null;
  const sessionId = raw.slice(0, dot);
  const offset = parseInt(raw.slice(dot + 1), 10);
  if (!Number.isFinite(offset) || offset < 0) return null;
  return { sessionId, offset };
}

// The skeleton cache uses the Bunny Edge Cache API. Caching is only an
// optimization, so any cache failure degrades to a miss / no-op and never
// breaks the feed response.
async function safeCacheMatch(cache: Cache, key: string): Promise<Response | undefined> {
  try {
    return await cache.match(key);
  } catch (e) {
    console.error("[cache] match failed (treating as miss):", (e as any)?.message);
    return undefined;
  }
}

async function safeCachePut(cache: Cache, key: string, res: Response): Promise<void> {
  try {
    await cache.put(key, res);
  } catch (e) {
    console.error("[cache] put failed (skipping):", (e as any)?.message);
  }
}

// Score candidates, reusing scores cached in the DB and only sending cache
// misses to Jina. Returns scores aligned to `cands` (null where scoring failed)
// plus the tokens actually billed (misses only) and cache/fan-out counters for
// diagnostics.
//
// The cache stores the *score*, not the embedding: the 1024-dim vector is used
// once, to compute this scalar, and never again. Caching the vector meant moving
// ~11.6 KB per post, so a full 500-candidate cache hit read ~5.9 MB and took ~9s
// — slower than re-embedding all 500 from scratch (~6.5s), which made the cache
// a cost optimization that actively hurt latency. Caching the scalar cuts the
// same read to ~15 KB.
//
// SCORE_CACHE_VERSION scopes entries to the embedding model AND the scoring
// vector, so retraining either invalidates old rows by simple cache miss.
// Also what keeps two feeds' scores for the same post apart: each vector has its
// own fingerprint, and (uri, version) is the score cache's key.
function scoreCacheVersion(scorer: { version: string }): string {
  return `${EMBEDDING_MODEL}:${scorer.version}`;
}

type ScoreResult = {
  scores: (number | null)[];
  tokens: number;
  hits: number;
  misses: number;
  chunks: number;
  failedChunks: number;
};

async function scoreCandidates(
  cands: Candidate[],
  apiKey: string,
  feed: FeedConfig,
  budget: Budget,
  errs: string[],
): Promise<ScoreResult> {
  const scores: (number | null)[] = new Array(cands.length).fill(null);

  // Batch-load cached scores. Best-effort: a DB error just means "all miss".
  // Bounded like every other DB call even though missing the cache makes the Jina
  // stage bigger, not smaller -- a read that has not answered in DB_STAGE_TIMEOUT_MS
  // was never going to return hits in time to save anything.
  const cached = await withDeadline(
    getCachedScores(cands.map(c => c.uri), scoreCacheVersion(feed.scorer)),
    budget.slice(DB_STAGE_TIMEOUT_MS), 'score-cache-read')
    .catch((e: any) => { console.error("[score cache] read failed:", errChain(e)); return new Map<string, number>(); });

  // De-duplicate misses by uri: Jina is billed per unique text, and SQLite
  // rejects upserting the same row twice in one statement.
  const missSlot = new Map<string, number>(); // uri -> index into miss arrays
  const missUris: string[] = [];
  const missTexts: string[] = [];
  let hits = 0;
  for (let i = 0; i < cands.length; i++) {
    const hit = cached.get(cands[i].uri);
    if (hit !== undefined) {
      scores[i] = hit;
      hits++;
    } else if (!missSlot.has(cands[i].uri)) {
      missSlot.set(cands[i].uri, missUris.length);
      missUris.push(cands[i].uri);
      missTexts.push(cands[i].text);
    }
  }

  let tokens = 0, chunks = 0, failedChunks = 0;
  if (missUris.length > 0) {
    // Out of budget before the billed stage: serve whatever the cache gave us
    // (possibly nothing, which is an empty feed) and name it, rather than paying
    // Jina for embeddings that arrive after the request is gone. Deliberately not
    // the rethrow below, even with no hits: nothing is wrong with Jina here.
    const jinaMs = budget.slice(JINA_TIMEOUT_MS);
    if (jinaMs < JINA_MIN_MS) {
      console.error(`[score cache] skipping Jina: ${jinaMs}ms of budget left, serving ${hits} cached score(s)`);
      errs.push('jina:no-budget');
      return { scores, tokens: 0, hits, misses: missUris.length, chunks: 0, failedChunks: 0 };
    }

    // getEmbeddingsBatched throws only when EVERY chunk failed (a total Jina
    // outage). Now that the cache is load-bearing we can ride that out: if some
    // candidates were served from cache, rank those instead of failing the whole
    // request. With no hits either there is nothing to serve, so rethrow and let
    // the handler return its 500 as before.
    let res: Awaited<ReturnType<typeof getEmbeddingsBatched>>;
    try {
      res = await getEmbeddingsBatched(missTexts, apiKey, jinaMs);
    } catch (e: any) {
      if (hits === 0) throw e;
      console.error(`[score cache] all Jina chunks failed; serving ${hits} cached score(s) only:`, e?.message ?? String(e));
      return { scores, tokens: 0, hits, misses: missUris.length, chunks: 0, failedChunks: 0 };
    }
    tokens = res.tokens;
    chunks = res.chunks;
    failedChunks = res.failedChunks;

    // Fold each fresh embedding down to its score. A null embedding means that
    // Jina chunk failed; the candidate stays unscored and drops out of the feed.
    const missScores: (number | null)[] = res.embeddings.map(e => (e ? feed.scorer.score(e).probability : null));

    // Fill every still-empty slot (including duplicate uris) from the results.
    for (let i = 0; i < cands.length; i++) {
      if (scores[i] === null) {
        const slot = missSlot.get(cands[i].uri);
        if (slot !== undefined) scores[i] = missScores[slot];
      }
    }

    // Persist the freshly-computed scores. Awaited rather than deferred past the
    // response: the vector version of this write was 5 sequential ~1.2 MB upserts,
    // which is why it was moved behind waitUntil — but the score payload is ~1500x
    // smaller (one ~50 KB statement), so a single inline round-trip costs ~30ms on
    // a multi-second request. That buys certainty: the cache warming is the entire
    // point of this path, and a deferred write that silently fails would leave the
    // cache permanently cold with no signal. Prunes and telemetry, which nothing
    // depends on, still go through runAfterResponse.
    //
    // Scored against EVERY registered vector, not just the one this request
    // needs, and stored as a row per version. The Jina call is what costs money;
    // each extra scorer is a dot product over an embedding we already hold in
    // memory. Without this, a post embedded for one feed is a version miss for
    // the next and would be embedded — and billed — a second time for the very
    // same text.
    const toStore: { uri: string; score: number; version: string }[] = [];
    for (const scorer of ALL_SCORERS) {
      const version = scoreCacheVersion(scorer);
      for (let k = 0; k < missUris.length; k++) {
        const emb = res.embeddings[k];
        if (!emb) continue;   // failed Jina chunk: nothing to score, for any feed
        toStore.push({ uri: missUris[k], score: scorer.score(emb).probability, version });
      }
    }
    await withDeadline(putScores(toStore), budget.slice(DB_STAGE_TIMEOUT_MS), 'score-cache-write')
      .catch((e: any) => console.error("[score cache] write failed:", errChain(e)));
  }

  // Opportunistically keep the table bounded. Deferred past the response via
  // waitUntil so the prune never adds latency to the request.
  if (Math.random() < SCORE_PRUNE_PROBABILITY) {
    runAfterResponse(
      pruneOldScores(Date.now() - SCORE_CACHE_MAX_AGE_MS)
        .catch((e: any) => console.error("[score cache] prune failed:", errChain(e)))
    );
  }

  return { scores, tokens, hits, misses: missUris.length, chunks, failedChunks };
}

BunnySDK.net.http.serve(async (request: Request) => {
  try {
    return await handleRequest(request);
  } catch (err: any) {
    console.error("[Fetch Event] Unhandled error in handleRequest:", err);
    return new Response(JSON.stringify({ error: "Internal Server Error", details: err.message }), { status: 500 });
  }
});

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);

  // --- FEED GENERATOR SKELETON ENDPOINT ---
  if (url.pathname === '/xrpc/app.bsky.feed.getFeedSkeleton') {
    const t0 = Date.now();
    // Everything below slices its deadline out of this, so the stages can only
    // ever add up to a request the platform will let finish.
    const budget = requestBudget(t0);
    try {
      const apiKey = getEnv('JINA_API_KEY') || 'MISSING_API_KEY';
      if (!apiKey || apiKey === 'MISSING_API_KEY') {
        console.error("[getFeedSkeleton Error]: Missing Jina API Key in environment");
        return new Response(JSON.stringify({ error: "Missing Jina API Key" }), { status: 500 });
      }

      const geApiKey = getEnv('GREEN_EARTH_API_KEY');
      if (!geApiKey) {
        console.error("[getFeedSkeleton Error]: Missing GE API Key in environment");
        return new Response(JSON.stringify({ error: "Missing GE API Key" }), { status: 500 });
      }

      // Extract User DID from ATProto Service JWT
      const authHeader = request.headers.get('Authorization');
      let userDid = null;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        try {
          const decoded: any = jwtDecode(authHeader.substring(7));
          userDid = decoded.iss; // Service JWT issuer is the user's DID
        } catch (e) { }
      }
      const feedUriParam = url.searchParams.get('feed') || '';
      const rkey = feedUriParam.split('/').pop() || 'unknown';
      // Which vector ranks this request, and at what gate. An unknown rkey falls
      // back to the locus-of-control feed rather than erroring.
      const feedConfig = resolveFeed(rkey);

      if (feedConfig.forcePublic) {
        userDid = null; // force unauthenticated path
      }

      const didTag = shortDid(userDid);

      // (1) Serve from the scored-skeleton cache when fresh. Keyed by feed identity +
      // user DID (or "public") + pagination cursor. A hit returns instantly and
      // skips the upstream fetches, embeddings, and the OAuth restore/lock below.
      const startCursor: string | undefined = url.searchParams.get('cursor') || undefined;
      const limit = parseLimit(url.searchParams.get('limit'));
      // `limit` is part of the key: the cached (public) body is already sliced to
      // a page size, so a client asking for a different limit must not be served
      // another's slice.
      const cacheKey = `${rkey}:${userDid || 'public'}:${startCursor || 'START'}:l${limit}`;

      const skeletonCacheUrl = new URL(request.url);
      skeletonCacheUrl.pathname = '/_cache/skeleton';
      skeletonCacheUrl.search = '';
      skeletonCacheUrl.searchParams.set('key', cacheKey);
      const skeletonCacheKeyStr = skeletonCacheUrl.toString();

      const cache = caches.default;
      // Personalized feeds are never cached (see the write path below): every
      // logged-in request recomputes so seen-post exclusion (interactionSeen)
      // takes effect immediately instead of lagging behind the cache TTL. Only
      // the public/unauthenticated feed is served from — and written to — cache.
      const cachedSkeleton = userDid ? undefined : await safeCacheMatch(cache, skeletonCacheKeyStr);

      if (cachedSkeleton && cachedSkeleton.status === 200) {
        const body = await cachedSkeleton.json();
        // Cache hits were previously silent. A public-fallback response cached
        // under a personalized key would be re-served as "English" with no
        // trace — surface the markers we stored on write so it shows in logs.
        logFeed({
          feed: rkey, did: didTag, cursor: cursorTag(startCursor), cache: 'hit',
          personalized: cachedSkeleton.headers.get('X-Personalized'),
          path: cachedSkeleton.headers.get('X-Feed-Path'),
          cand: cachedSkeleton.headers.get('X-Cand-Count'),
          pass: cachedSkeleton.headers.get('X-Pass-Count'),
          feedLen: Array.isArray((body as any)?.feed) ? (body as any).feed.length : null,
          nextCursor: (body as any)?.cursor ? 'y' : 'n',
          ms: Date.now() - t0,
        });
        return new Response(JSON.stringify(body), {
          headers: {
            'Content-Type': 'application/json',
            'X-Cache': 'hit',
            // Never let the Bunny PullZone CDN cache the personalized skeleton:
            // it keys on neither the cursor nor the per-user Authorization/DID, so
            // one cached copy freezes the feed for everyone. This response is
            // already served from our own per-(user,cursor) caches.default; the
            // CDN layer on top must pass through.
            'Cache-Control': 'no-store',
          },
        });
      }

      // (2) Continuation page: the cursor points at a frozen ranked list from an
      // earlier page-1 request. Read it back and slice the next window — no GE,
      // embedding, or rank work. On any miss (expired/pruned session, wrong
      // owner, or a malformed/garbage cursor) we fall through to a fresh
      // generation below, which re-runs seen-exclusion; by the time a session has
      // aged out, interactionSeen for its posts has long since landed, so the
      // regeneration doesn't repeat them.
      const decoded = decodeCursor(startCursor);
      if (decoded) {
        const session = await withDeadline(
          getFeedSession(decoded.sessionId), budget.slice(DB_STAGE_TIMEOUT_MS), 'session-read')
          .catch((e: any) => { console.error("[getFeedSkeleton] session read failed:", errChain(e)); return null; });
        const fresh = !!session && (Date.now() - session.createdAt) < SESSION_TTL_MS;
        const ownerOk = !!session && (session.did || null) === (userDid || null);
        if (session && fresh && ownerOk) {
          const page = session.uris.slice(decoded.offset, decoded.offset + limit);
          const nextOffset = decoded.offset + page.length;
          const body: SkeletonBody = { feed: page.map(uri => ({ post: uri })) };
          if (nextOffset < session.uris.length) body.cursor = encodeCursor(decoded.sessionId, nextOffset);
          logFeed({
            feed: rkey, did: didTag, cursor: cursorTag(startCursor), cache: 'miss', page: 'cont',
            off: decoded.offset, limit, feedLen: page.length,
            nextCursor: body.cursor ? 'y' : 'n', ms: Date.now() - t0,
          });
          return new Response(JSON.stringify(body), {
            headers: {
              'Content-Type': 'application/json',
              'X-Cache': 'miss',
              'X-Page-Path': 'cont',
              // Never let the PullZone CDN cache the personalized skeleton (keys on
              // neither cursor nor DID). Our own per-key stores handle caching.
              'Cache-Control': 'no-store',
              'X-Feed-Version': 'minilm-prefilter-1',
            },
          });
        }
        // Session gone/stale/foreign → fall through to a fresh page 1.
      }

      // (2b) Fresh page-1 reuse. A logged-in START request (no cursor) otherwise
      // re-runs the entire generate→embed→rank pipeline on every reload, crash
      // retry, or quick re-open. If this user produced a ranked list within
      // START_REUSE_TTL_MS, re-serve its first window (and a cursor into it)
      // instead. The frozen list already had interactionSeen exclusion applied
      // when generated; the reuse window is short enough that the only
      // not-yet-excluded posts are ones shown seconds ago — i.e. what's on screen,
      // which a reload should keep showing anyway. Best-effort: any read failure
      // falls through to a fresh generation.
      if (userDid && !startCursor) {
        const latest = await withDeadline(
          getLatestFeedSession(userDid), budget.slice(DB_STAGE_TIMEOUT_MS), 'latest-session-read')
          .catch((e: any) => { console.error("[getFeedSkeleton] latest-session read failed:", errChain(e)); return null; });
        if (latest && latest.uris.length > 0 && (Date.now() - latest.createdAt) < START_REUSE_TTL_MS) {
          const page = latest.uris.slice(0, limit);
          const body: SkeletonBody = { feed: page.map(uri => ({ post: uri })) };
          if (limit < latest.uris.length) body.cursor = encodeCursor(latest.sessionId, limit);
          logFeed({
            feed: rkey, did: didTag, cursor: cursorTag(startCursor), cache: 'miss', page: 'reuse',
            off: 0, limit, feedLen: page.length,
            nextCursor: body.cursor ? 'y' : 'n', ms: Date.now() - t0,
          });
          return new Response(JSON.stringify(body), {
            headers: {
              'Content-Type': 'application/json',
              'X-Cache': 'miss',
              'X-Page-Path': 'reuse',
              // Same reasoning as the other personalized paths: keep the PullZone
              // CDN out of it; our per-user session store already caches.
              'Cache-Control': 'no-store',
              'X-Feed-Version': 'minilm-prefilter-1',
            },
          });
        }
      }

      let candidates: Candidate[] = [];
      const personalized = !!userDid;
      const feedPath = personalized ? 'personalized' : 'public-no-did';
      
      const tG = Date.now();
      const geHeaders = {
        'Content-Type': 'application/json',
        'X-API-Key': geApiKey
      };
      
      const primaryGenerator = userDid ? 'followed_users' : 'popularity';

      // Exclude posts this user has already been shown so paging/refreshing
      // doesn't repeat them. Best-effort: a DB failure just means no exclusion
      // this request. Only meaningful when we know who the user is.
      let tSeen = 0, tPrune = 0, tCand = 0, tEmbed = 0, tRank = 0, tStore = 0;
      // Score-cache diagnostics, surfaced in the [feed] log line and on the
      // response headers so cache warming can be verified without DB access.
      let embedHits = 0, embedMisses = 0, embedChunks = 0, embedFailedChunks = 0;
      // Stage failures seen on this request, joined into the feed_logs `errors`
      // column. Empty is the normal case and stays null in the row. Declared here
      // rather than at the gather stage because the seen read below is the first
      // thing that can fail, and a request that silently stopped excluding posts
      // is exactly the kind of degradation this column exists to show.
      const errs: string[] = [];
      let tS_start = Date.now();
      let excludeUris: string[] = [];
      if (userDid) {
        excludeUris = await withDeadline(
          getSeenUris(userDid, SEEN_EXCLUDE_LIMIT, SEEN_RETENTION_DAYS),
          budget.slice(DB_STAGE_TIMEOUT_MS), 'seen-uris')
          .catch((e: any) => {
            console.error("[getFeedSkeleton] seen-uris read failed:", errChain(e));
            errs.push(stageErr('seen-uris', e));
            return [];
          });
      }
      tSeen = Date.now() - tS_start;

      // Opportunistically age out old "shown" rows so the table stays bounded
      // without a separate cron (engaged rows are kept for training). Deferred
      // past the response via waitUntil, so tPrune now only bounds the dispatch.
      tS_start = Date.now();
      if (Math.random() < SEEN_PRUNE_PROBABILITY) {
        runAfterResponse(
          pruneOldSeenInteractions(SEEN_RETENTION_DAYS)
            .catch((e: any) => console.error("[getFeedSkeleton] seen prune failed:", errChain(e)))
        );
      }
      tPrune = Date.now() - tS_start;

      const tC_start = Date.now();
      let candRaw = 0, candShort = 0;
      let preHydrated = 0, preKept = 0, preFloor = 0, preFellBack = false;
      let preBypass = 0, preBypassDropped = 0;
      try {
        const buildBody = (gens: { name: string; weight: number }[]) => {
          const b: any = {
            generators: gens,
            infill: 'popularity',
            user_did: userDid || 'did:plc:z72i7hdynmk6r22z27h6tvur', // fallback DID if none
            num_candidates: feedConfig.numCandidates ?? 500,
            // Restrict candidates to the last 3 days. GE's allowed values are
            // 6/12/24/48/72/168; the infill generator inherits this window too,
            // so it can't widen eligibility.
            max_age_hours: 72,
            // Return each candidate's MiniLM embedding so the prefilter can rank
            // the whole pool without paying Jina (see minilm.ts). Measured free:
            // 500 candidates came back in 1.4-2.2s hydrated against 3.0-3.3s not,
            // at the cost of ~2KB of base64 each. Always requested, even for feeds
            // with no prefilter configured, so that a feed can be switched over in
            // feeds.ts alone -- and so this stays one code path to reason about.
            hydrate_embeddings: true
          };
          // Only send the key when non-empty so the unauthenticated path stays
          // byte-identical to before and we never send an empty array.
          if (excludeUris.length > 0) b.exclude_uris = excludeUris;
          return b;
        };

        // Call 1 is the normal generator mix. Any further calls are
        // `random_posts`, which is the only generator that returns a different
        // sample each time (the mix repeats itself 99%), so it is what actually
        // grows the pool past GE's 1000-per-request cap.
        const bodies = [buildBody([
          { name: primaryGenerator, weight: 1.0 },
          { name: 'network_likes', weight: 1.0 },
          { name: 'two_tower', weight: 1.0 },
        ])];
        for (let i = 1; i < (feedConfig.fanout ?? 1); i++) {
          bodies.push(buildBody([{ name: 'random_posts', weight: 1.0 }]));
        }

        // What this stage actually gets: its own cap, or what the request has
        // left, whichever is smaller. Computed once and shared by every batch and
        // every retry inside them, so the fan-out cannot multiply the deadline.
        const genBudgetMs = budget.slice(GE_GENERATE_TIMEOUT_MS);
        const skipGenerate = genBudgetMs < GE_GENERATE_MIN_MS;
        if (skipGenerate) {
          console.error(`[getFeedSkeleton] skipping GE generate: ${genBudgetMs}ms of budget left`);
          errs.push('ge-generate:no-budget');
        }

        // One slow or failing call must not lose the others' candidates.
        const batches = skipGenerate ? [] : await Promise.all(bodies.map(async (body) => {
          const payload = JSON.stringify(body);
          const tGen = Date.now();
          let lastErr: any;
          for (let attempt = 1; attempt <= GE_GENERATE_MAX_ATTEMPTS; attempt++) {
            const remaining = genBudgetMs - (Date.now() - tGen);
            if (remaining <= 0) break;
            try {
              const r = await fetch('https://api.greenearth.social/candidates/generate', {
                method: 'POST', headers: geHeaders, body: payload,
                // Whatever is left of the shared budget, so attempt 2 inherits
                // the deadline rather than restarting it.
                signal: AbortSignal.timeout(remaining),
              });
              if (!r.ok) {
                const httpErr: any = new Error(`GE Generate failed: ${r.status}`);
                httpErr.retryable = GE_GENERATE_RETRYABLE_STATUS.has(r.status);
                throw httpErr;
              }
              return (await r.json()).candidates || [];
            } catch (e: any) {
              lastErr = e;
              const elapsed = Date.now() - tGen;
              // A thrown fetch/network error carries no flag and is retryable; a
              // status we classified carries its verdict; a timeout is not early.
              const retryable = e?.retryable !== false
                && e?.name !== 'TimeoutError' && e?.name !== 'AbortError';
              if (!retryable || attempt === GE_GENERATE_MAX_ATTEMPTS
                  || genBudgetMs - elapsed < GE_GENERATE_RETRY_MIN_MS) break;
              console.warn(`[getFeedSkeleton] GE generate failed after ${elapsed}ms (attempt ${attempt}/${GE_GENERATE_MAX_ATTEMPTS}), retrying:`, e?.message || String(e));
            }
          }
          // Only a batch that failed every attempt it was allowed lands in `errors`.
          // A retry that succeeded produced candidates, so counting it here would
          // both overstate the failure rate and, because an empty pool is only named
          // `ge-generate:empty` when nothing else was recorded, mask a genuinely
          // empty one. Retries are visible in the console.warn above instead.
          console.error("[getFeedSkeleton] GE generate call failed:", lastErr?.message || String(lastErr));
          errs.push(stageErr('ge-generate', lastErr));
          return [];
        }));

        const seenUri = new Set<string>();
        const merged: Candidate[] = [];
        for (const batch of batches) {
          for (const c of batch as any[]) {
            if (!c?.at_uri || seenUri.has(c.at_uri)) continue;
            seenUri.add(c.at_uri);
            merged.push({ ...c, uri: c.at_uri, text: c.content || '' });
          }
        }
        candRaw = merged.length;

        // Cheap text gates before any embedding. This is what makes a fanned-out
        // pool affordable: a regex and a word count are free, Jina is not.
        // Feeds that configure neither keep the whole pool, exactly as before.
        const minWords = feedConfig.minWords ?? 0;
        candidates = merged.filter(c => {
          // Length first, and counted separately, so the [feed] line reports how
          // much of the pool is one-liners independently of the regex gate.
          if (minWords > 0 && !hasEnoughText(c.text, minWords)) { candShort++; return false; }
          return !feedConfig.candidateFilter || feedConfig.candidateFilter.test(c.text);
        });
      } catch (err: any) {
        console.error("[getFeedSkeleton] GE generate fetch failed:", err?.message || String(err));
        errs.push(stageErr('ge-gather', err));
      }
      // An empty pool is not an error any call reported -- every batch can return
      // 200 with nothing in it -- but it is the one outcome that guarantees an
      // empty feed, so name it rather than leaving it to be inferred from cand=0.
      if (candRaw === 0 && !errs.length) errs.push('ge-generate:empty');
      tCand = Date.now() - tC_start;
      
      const gatherMs = Date.now() - tG;
      const filteredCandidates: Candidate[] = [];

      if (candidates.length > 0) {
        const tE_start = Date.now();

        // (3a) Free first pass. Rank the WHOLE pool on GE's hydrated embeddings and
        // keep only the top `shortlist`, so the billed stage below sees ~150
        // candidates instead of every survivor of the text gates. Costs one dot
        // product each (~5us; 4000 candidates measured at ~20ms total), no API
        // call, no cache.
        //
        // Rank, not threshold. This stage's only job is to pick the best N of
        // whatever the pool happens to contain; an absolute floor here made the
        // feed's length a property of how the pool's scores happened to be
        // distributed that day, which is how signed-in readers ended up with ~29
        // posts against a 100-post cap. The quality decision belongs to the paid
        // gate below, whose threshold is applied per-feed to the Jina score. See
        // LOC.prefilter in feeds.ts for the measurements.
        let paidPool = candidates;
        let miniScores: (number | null)[] | null = null;
        // Per paid-pool candidate: did it get here by bypassing the free stage?
        // Those are the ones `threshold` still applies to below.
        let bypassed: boolean[] | null = null;
        const pf = feedConfig.prefilter;
        if (pf) {
          const mini = candidates.map(c => {
            const e = decodeEmbedding(c.minilm_l12_embedding);
            return e ? pf.scorer.score(e).probability : null;
          });
          preHydrated = mini.reduce((n: number, m) => n + (m === null ? 0 : 1), 0);
          // Coverage check, not an assumption. Every candidate seen without an
          // embedding in testing had *empty* text and was already dropped by
          // minWords, so real coverage after the gates was 100% -- but if GE ever
          // stops hydrating, silently prefiltering on a handful of embeddings would
          // empty the feed. Below the floor we fall back to the original
          // single-stage path: more expensive, and correct.
          if (preHydrated >= candidates.length * MIN_HYDRATION_COVERAGE) {
            // Candidates the free scorer cannot read (today: German — see
            // pf.bypass) are pulled out BEFORE the ranking, not ranked and then
            // rescued. Ranking them first would be the bug: their scores are the
            // near-chance numbers this whole detour exists because of, so they
            // would take shortlist places off English posts whose scores mean
            // something, and lose the German posts that scored badly anyway.
            //
            // Both halves of the config or neither: a `bypass` with no budget to
            // spend would route German candidates out of the free stage and then
            // drop every one of them for want of a place, which is worse than the
            // bad ranking this replaces. Missing limit = no detour, today's path.
            const bypassMax = pf.bypass ? (pf.bypassLimit ?? 0) : 0;
            const bypassFn = bypassMax > 0 ? pf.bypass : undefined;
            const routed = candidates.map((c, i) => ({
              c, p: mini[i], bypass: bypassFn ? bypassFn(c.text) : false,
            }));
            const kept = routed
              .filter(x => !x.bypass && x.p !== null)
              .sort((a, b) => (b.p as number) - (a.p as number))
              .slice(0, pf.shortlist);
            // Ordered by the same free score the group was routed around — which
            // is not the contradiction it looks like. What the measurement kills
            // is comparing a German score against an English one: on a shared
            // scale German is depressed, so a global cut keeps 15% of German
            // against 21% of English. WITHIN the group that bias is common to
            // every candidate and cancels, and what is left still ranks above
            // chance (AUC 0.618 on the routed group, against the 0.500 an
            // arbitrary order earns). Measured on the corpus at a binding cap,
            // ordering the detour this way rather than by GE's pool order lifts
            // recall of routed band-positives from 27% to 47% at identical cost.
            // Unhydrated candidates have no score at all and sort last.
            const detour = routed.filter(x => x.bypass)
              .sort((a, b) => (b.p ?? -Infinity) - (a.p ?? -Infinity));
            // Share-matched against the rest of the pool; see bypassBudget().
            const taken = detour.slice(0, bypassBudget(
              pf.shortlist, detour.length, routed.length - detour.length,
              pf.bypassFloor ?? bypassMax, bypassMax));
            preBypass = taken.length;
            preBypassDropped = detour.length - taken.length;
            paidPool = [...kept.map(x => x.c), ...taken.map(x => x.c)];
            miniScores = [...kept.map(x => x.p), ...taken.map(() => null)];
            bypassed = [...kept.map(() => false), ...taken.map(() => true)];
            preKept = paidPool.length;
            // Where the cut actually landed on this pool. With no fixed floor this
            // is the only way to see the stage's operating point, and it is what a
            // future recalibration of the paid gate has to be read against. The
            // bypassed tail has no score of its own and is not part of it.
            preFloor = kept.length ? (kept[kept.length - 1].p as number) : 0;
          } else {
            preFellBack = true;
            console.error(`[getFeedSkeleton] prefilter disabled: only ${preHydrated}/${candidates.length} candidates hydrated`);
          }
        }

        // (3b) Reuse cached scores; only cache-missed posts are sent to Jina.
        const scored = await scoreCandidates(paidPool, apiKey, feedConfig, budget, errs);
        embedHits = scored.hits;
        embedMisses = scored.misses;
        embedChunks = scored.chunks;
        embedFailedChunks = scored.failedChunks;
        // Best-effort cost tracking. Deferred past the response via waitUntil:
        // nothing in the request depends on it, and awaiting it put a full DB
        // round-trip on the hot path purely for daily token accounting.
        runAfterResponse(
          recordEmbeddingUsage(scored.tokens).catch((e: any) =>
            console.error("[getFeedSkeleton] failed to record token usage:", errChain(e)))
        );

        for (let i = 0; i < paidPool.length; i++) {
          const p = scored.scores[i];
          if (miniScores) {
            // Already selected by the free pass, so `threshold` is not applied a
            // second time here: it gates the Jina distribution, and re-gating a
            // shortlist that was chosen on a different vector would just discard
            // the disagreements this feed is paying Jina to hear. A null score is
            // a failed Jina chunk rather than a rejection, so the candidate stays
            // and is ranked on its free score alone.
            //
            // A bypassed candidate is the exception on both counts. Nothing has
            // judged it yet, so the Jina gate is its only one and does apply — and
            // if its Jina chunk failed it has no usable score at all (its free
            // score is the near-chance number it was routed around), so it drops
            // rather than riding into the feed on a meaningless rank.
            if (bypassed?.[i] && (p === null || p < feedConfig.threshold)) continue;
            paidPool[i].minilmScore = miniScores[i];
            paidPool[i].jinaScore = p;
            // Overwrite the generator's popularity `score` (unused after ranking)
            // with the vector-similarity probability so GE's `candidate_score`
            // model can blend it into the final rank order (see RANK_SIM_WEIGHT).
            paidPool[i].score = p ?? miniScores[i] ?? 0;
            filteredCandidates.push(paidPool[i]);
          } else if (p !== null && p >= feedConfig.threshold) {
            paidPool[i].score = p;
            filteredCandidates.push(paidPool[i]);
          }
        }

        // Sort descending by vector probability and enforce maximum limit
        if (miniScores && feedConfig.fuse) {
          fuseByRank(filteredCandidates);
        }
        filteredCandidates.sort((a, b) => (b.score || 0) - (a.score || 0));
        if (filteredCandidates.length > MAX_RERANK_CANDIDATES) {
          filteredCandidates.length = MAX_RERANK_CANDIDATES;
        }

        tEmbed = Date.now() - tE_start;
        
        // Fixed once, so the branch below and the signal it hands the fetch are
        // deciding on the same number.
        const rankMs = budget.slice(GE_RANK_TIMEOUT_MS);
        if (filteredCandidates.length > 0 && feedConfig.rankBy === 'score') {
          // The vector IS the editorial criterion for this feed, and
          // filteredCandidates is already sorted by it. Handing that order to
          // GE's engagement ranker scrambles it — measured Spearman 0.277
          // between served rank and assist score — so skip the rerank call
          // entirely. Also saves a round-trip.
          tRank = 0;
        } else if (filteredCandidates.length > 0 && rankMs < GE_RANK_MIN_MS) {
          // No budget left to rank in. The vector's own order is already the
          // fallback whenever the rerank fails, so take it now instead of paying
          // for a call that would be aborted before it returned.
          console.error(`[getFeedSkeleton] skipping GE rank: ${rankMs}ms of budget left`);
          errs.push('ge-rank:no-budget');
          tRank = 0;
        } else if (filteredCandidates.length > 0) {
          const tR_start = Date.now();
          // (4) Rank via Green Earth
          try {
            const rankRes = await fetch('https://api.greenearth.social/rank/predict', {
              method: 'POST',
              headers: geHeaders,
              body: JSON.stringify({
                candidates: filteredCandidates,
                models: [
                  { name: 'heavy_ranker', weight: 1.0 },
                  // Blend in vector similarity (stashed in each candidate's
                  // `score` above) so more semantically-similar posts rank higher.
                  { name: 'candidate_score', weight: RANK_SIM_WEIGHT },
                ],
                user_did: userDid || 'did:plc:z72i7hdynmk6r22z27h6tvur'
              }),
              signal: AbortSignal.timeout(rankMs),
            });
            if (!rankRes.ok) errs.push(`ge-rank:http-${rankRes.status}`);
            if (rankRes.ok) {
                const rankData = await rankRes.json();
                if (rankData.rankings && rankData.rankings.length > 0) {
                    const rankedUris = rankData.rankings.map((r: any) => r.at_uri);
                    filteredCandidates.sort((a, b) => {
                        const idxA = rankedUris.indexOf(a.uri);
                        const idxB = rankedUris.indexOf(b.uri);
                        return (idxA === -1 ? 9999 : idxA) - (idxB === -1 ? 9999 : idxB);
                    });
                }
            }
          } catch (e: any) {
              console.error("[getFeedSkeleton] GE rank failed:", e?.message || String(e));
              errs.push(stageErr('ge-rank', e));
          }

          // NOTE: the GE /diversify step was removed. It re-sorts candidates by
          // the generator's popularity `score` and ignores the input order, so
          // it silently overwrote the rank order above (the reranker had no
          // visible effect). It also expects minilm embeddings, which this feed
          // doesn't produce (we embed with Jina), so it couldn't diversify
          // anyway. Rank order is now the final feed order.
          tRank = Date.now() - tR_start;
        }
      }

      const tSt_start = Date.now();
      // The full ranked list for this scroll session. Page 1 freezes it in a
      // feed_sessions row and serves only the first `limit` posts; the cursor
      // lets later pages slice further into this exact ordering without
      // regenerating. (Previously the entire list was returned as a single page.)
      const rankedUris = filteredCandidates.map(c => c.uri);
      let sessionStored = false;
      let sessionId = '';
      if (rankedUris.length > 0) {
        sessionId = crypto.randomUUID();
        // A write that misses its slice yields no cursor -- the client gets page 1
        // and a refresh regenerates -- which is already what a failed write does.
        sessionStored = await withDeadline(
          putFeedSession(sessionId, userDid, rankedUris),
          budget.slice(DB_STAGE_TIMEOUT_MS), 'session-write')
          .catch((e: any) => {
            console.error("[getFeedSkeleton] session write failed:", errChain(e));
            errs.push(stageErr('session-write', e));
            return false;
          });
      }

      // Opportunistically prune expired sessions so the table stays bounded.
      // Deferred past the response via waitUntil so it adds no request latency.
      if (Math.random() < SESSION_PRUNE_PROBABILITY) {
        runAfterResponse(
          pruneOldFeedSessions(Date.now() - SESSION_TTL_MS)
            .catch((e: any) => console.error("[getFeedSkeleton] session prune failed:", errChain(e)))
        );
      }
      tStore = Date.now() - tSt_start;

      const page = rankedUris.slice(0, limit);
      const feed: { post: string }[] = page.map(uri => ({ post: uri }));

      // Emit a continuation cursor only when the session was stored AND more
      // posts remain past this page; omitting it signals "end of feed". A failed
      // session write yields no cursor — the client gets page 1 and a refresh
      // regenerates, rather than a cursor that could only miss and regenerate.
      const body: SkeletonBody = { feed };
      if (sessionStored && limit < rankedUris.length) {
        body.cursor = encodeCursor(sessionId, limit);
      }

      // Only cache the public/unauthenticated feed. Personalized (logged-in)
      // feeds are never cached: caching them lagged interactionSeen exclusion by
      // up to SKELETON_TTL_SEC (the cached page kept re-serving already-seen
      // posts). Recomputing every personalized request keeps exclusion immediate;
      // the public feed still caches since it has no per-user state to stale.
      if (!userDid) {
        const resToCache = new Response(JSON.stringify(body), {
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': `s-maxage=${SKELETON_TTL_SEC}`,
            // Markers so a later cache HIT isn't blind: whether this cached body
            // was personalized, and how it was produced.
            'X-Personalized': String(personalized),
            'X-Feed-Path': feedPath,
            'X-Cand-Count': String(candidates.length),
            // Total that passed the vector filter (the full frozen list), not the
            // page length — preserves this marker's meaning now that we page.
            'X-Pass-Count': String(rankedUris.length),
          }
        });
        await safeCachePut(cache, skeletonCacheKeyStr, resToCache);
      }

      logFeed({
        feed: rkey, did: didTag, cursor: cursorTag(startCursor), cache: 'miss', page: 'fresh',
        candRaw, candShort, cand: candidates.length, pass: rankedUris.length,
        off: 0, limit, feedLen: feed.length,
        nextCursor: body.cursor ? 'y' : 'n',
        gatherMs, ms: Date.now() - t0,
        tSeen, tPrune, tCand, tEmbed, tRank, tStore,
        cacheHit: embedHits, cacheMiss: embedMisses,
        jinaChunks: embedChunks, jinaFailed: embedFailedChunks,
        preHydrated, preKept, preFloor: preFloor.toFixed(4), preFellBack: preFellBack ? 'y' : 'n',
        preBypass, preBypassDropped,
        waitUntil: hasWaitUntil() ? 'y' : 'n',
        errors: errs.length ? errs.join(',') : null
      });

      return new Response(JSON.stringify(body), {
        headers: {
          'Content-Type': 'application/json',
          'X-Cache': 'miss',
          'X-Page-Path': 'fresh',
          // Never let the Bunny PullZone CDN cache the personalized skeleton (see
          // the hit path above): its cache key ignores the cursor and the per-user
          // Authorization/DID, so a single cached copy freezes the feed and defeats
          // seen-post exclusion. Our own caches.default already caches per-key.
          'Cache-Control': 'no-store',
          // Diagnostics (readable off the HTTP response, no logs needed):
          // X-Feed-Version confirms which build is actually serving traffic;
          // X-Ranked-Count is the full frozen list length; X-Exclude-Count reports
          // how many seen-post URIs were sent to GE as exclude_uris this request.
          'X-Feed-Version': 'minilm-prefilter-1',
          'X-Ranked-Count': String(rankedUris.length),
          'X-Exclude-Count': String(excludeUris.length),
          // Score-cache diagnostics: hits/misses tell you whether the cache is warming
          // at all, and jina=<chunks> how far the miss batch fanned out. A warm
          // feed shows m=0 (or only genuinely new posts) on a repeat request.
          'X-Score-Cache': `h=${embedHits} m=${embedMisses} jina=${embedChunks}${embedFailedChunks ? ` fail=${embedFailedChunks}` : ''}`,
          // Prefilter diagnostics: how many candidates arrived hydrated, how many
          // survived into the paid stage, the probability the shortlist's last
          // place scored (p=, the cut's real operating point now that the stage
          // ranks rather than thresholds), and whether coverage was too low to
          // prefilter at all (fb=y means this request paid the old full price).
          // de= is the language detour: how many candidates skipped the free stage
          // because it cannot read them, and how many more were over bypassLimit
          // (a nonzero second number means the cap is binding on this pool).
          'X-Prefilter': `hyd=${preHydrated} kept=${preKept} p=${preFloor.toFixed(4)} fb=${preFellBack ? 'y' : 'n'} de=${preBypass}/${preBypass + preBypassDropped}`,
          // Whether deferred (post-response) work is actually supported here; if
          // this reads 'n', prunes and token accounting are silently no-ops.
          'X-Wait-Until': hasWaitUntil() ? 'y' : 'n',
        },
      });
    } catch (e: any) {
      console.error("[getFeedSkeleton] error:", e?.status ?? '', e?.message, e?.stack);
      // A 500 is a feed the reader did not get, which is exactly what this table
      // exists to count -- and the only path that produced no row at all. Logged
      // with what is still in scope out here (the try's locals are not), so a bad
      // minute reads as rows with `errors` set instead of as a hole in the data.
      logFeed({
        feed: (url.searchParams.get('feed') || '').split('/').pop() || 'unknown',
        cache: 'miss', page: 'error', ms: Date.now() - t0,
        waitUntil: hasWaitUntil() ? 'y' : 'n',
        errors: stageErr('request', e),
      });
      return new Response(JSON.stringify({ error: e.message }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (url.pathname === '/xrpc/app.bsky.feed.sendInteractions' && request.method === 'POST') {
    try {
      // Extract User DID from ATProto Service JWT
      const authHeader = request.headers.get('Authorization');
      let userDid = null;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        try {
          const decoded: any = jwtDecode(authHeader.substring(7));
          userDid = decoded.iss; // Service JWT issuer is the user's DID
        } catch (e) { }
      }

      const body = await request.json();
      if (body.interactions && Array.isArray(body.interactions)) {
        // Record every interaction, INCLUDING `interactionSeen`: the "shown"
        // signal is what powers seen-post exclusion during candidate generation
        // (getSeenUris). Batched into one multi-row insert so a scroll burst is a
        // single DB round-trip, not one per post.
        const toRecord = body.interactions
          .filter((i: any) => i && i.item && i.event)
          .map((i: any) => ({ item: i.item, event: i.event, feedContext: i.feedContext }));
        await recordInteractions(userDid, toRecord).catch((e: any) => {
          console.error("[sendInteractions] failed to record interactions:", e?.cause?.message || e?.message);
        });
      }

      return new Response(JSON.stringify({}), { headers: { 'Content-Type': 'application/json' } });
    } catch (e: any) {
      console.error("[sendInteractions] error:", e?.message, e?.stack);
      return new Response(JSON.stringify({ error: e.message }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (url.pathname === '/.well-known/did.json') {
    const publicBaseUrl = getPublicBaseUrl(url);
    const publicHostname = new URL(publicBaseUrl).hostname;
    return new Response(JSON.stringify({
      "@context": ["https://www.w3.org/ns/did/v1"],
      "id": `did:web:${publicHostname}`,
      "service": [
        {
          "id": "#bsky_fg",
          "type": "BskyFeedGenerator",
          "serviceEndpoint": publicBaseUrl
        }
      ]
    }), { headers: { 'Content-Type': 'application/json' } });
  }

  // --- PUBLIC PAGE ---
  // Everything above this point is machine-facing. A human who types the
  // hostname into a browser gets the measurement notes instead: what the vector
  // does to the emotional and civility profile of the posts it admits. Static
  // and self-contained (see site.ts), so it costs no query, no upstream call and
  // no latency the feed path could notice, and the PullZone may cache it hard.
  if (request.method === 'GET' && (url.pathname === '/' || url.pathname === '/findings')) {
    return siteResponse();
  }

  // Unknown paths keep answering 200 with the original plaintext banner rather
  // than 404. describeFeedGenerator is not implemented here and lands in this
  // branch, and some clients read a non-2xx from a feed generator as the service
  // being down — so adding a page at `/` must not change what anything else gets.
  return new Response(`Bunny ATProto Feed Generator - Current time: ${new Date().toISOString()}`, { status: 200 });
}
