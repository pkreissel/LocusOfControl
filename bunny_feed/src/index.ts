import { jwtDecode } from 'jwt-decode';
import { getEmbeddingsBatched } from './jina';
import { hasEnoughText } from './text';
import { decodeEmbedding } from './minilm';
import { fuseByRank, blendLocAndEngagement } from './fuse';
import { resolveFeed, ALL_SCORERS, bypassBudget, scoreCacheVersion, type FeedConfig } from './feeds';
import { readerLangs, prefersGermanOnly, germanFirst } from './lang';
import { recordEmbeddingUsage, recordInteractions, getSeenUris, pruneOldSeenInteractions, getCachedScores, putScores, pruneOldScores, putFeedSession, getFeedSession, getLatestFeedSession, pruneOldFeedSessions, recordFeedLog, pruneOldFeedLogs, type FeedLogRow } from './db';
import * as BunnySDK from "@bunny.net/edgescript-sdk";
import { siteResponse } from './site';
import { runPrescore, PrescoreConfigError } from './prescore';

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

// The entire length of the feed. The list this truncates becomes `rankedUris`,
// which is frozen into the feed_sessions row that every cursor page slices from,
// so a reader scrolls exactly this many posts and then gets no cursor, i.e.
// end-of-feed. (It used to be named for the GE reranker it also capped; that
// call is gone, see (4).)
//
// At 100 that cap was throwing away work already paid for. The rank-based
// prefilter hands the paid stage ~150 shortlisted candidates plus the bypassed
// German ones (measured live: preKept 162 anonymous, 250-300 signed-in), and
// non-bypassed candidates deliberately skip the `threshold` gate, so 200-250
// survive to be sorted here and 100-150 of them were dropped on the floor.
const MAX_FEED_LENGTH = 150;

// Minimum share of candidates that must carry a hydrated embedding before the free
// prefilter is trusted with the pool. Measured coverage after the text gates was
// 100% (every un-hydrated candidate had empty text, which minWords already drops),
// so this only ever fires if GE changes what it returns — in which case the request
// pays for the old single-stage path instead of serving a feed ranked on a fraction
// of the pool.
const MIN_HYDRATION_COVERAGE = 0.8;

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

// The reader's DID, from the ATProto service JWT the appview sends. Only decoded,
// not verified -- the same trust the feed has always extended. Null when the
// request carries no token (a logged-out reader).
function didFromAuth(request: Request): string | null {
  const auth = request.headers.get('Authorization');
  if (!auth || !auth.startsWith('Bearer ')) return null;
  try {
    const decoded: any = jwtDecode(auth.substring(7));
    return typeof decoded?.iss === 'string' ? decoded.iss : null;
  } catch {
    return null;
  }
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
  /**
   * GE's own score for this candidate, as the generator returned it, captured
   * before `score` is overwritten with the vector probability. This is the
   * engagement signal the final order is built from (orderByGeneratorScore). Null
   * when the generator sent no numeric score.
   */
  geScore?: number | null;
  [key: string]: any;
};

// Each candidate's GE score as a rank within its own generator: best 1, worst 0,
// and 0.5 for a generator that contributed a single candidate (neutral, not a
// free top slot). Raw scores cannot be compared across generators -- see the
// ranges measured below -- so this is the only form in which GE's engagement
// signal orders a mixed list.
function generatorRank(cands: Candidate[]): number[] {
  const groups = new Map<string, number[]>();
  cands.forEach((c, i) => {
    const key = typeof c.generator_name === 'string' ? c.generator_name : '?';
    const g = groups.get(key);
    if (g) g.push(i); else groups.set(key, [i]);
  });
  const pct = new Array<number>(cands.length).fill(0.5);
  for (const idxs of groups.values()) {
    if (idxs.length === 1) continue;
    idxs.sort((a, b) => (cands[b].geScore ?? -Infinity) - (cands[a].geScore ?? -Infinity));
    const last = idxs.length - 1;
    idxs.forEach((ci, pos) => { pct[ci] = (last - pos) / last; });
  }
  return pct;
}

// The feed's final order: GE's own candidate score, for posts the vector has
// already selected.
//
// It began as the fallback for when GE's /rank/predict call did not happen, and
// became the only order when that call was removed: it needed the pipeline done
// by ~2.85s to fit the appview's 5s, so this was already what nearly every reader
// got. The alternative is to leave the list in the order the sort above left it,
// descending vector probability. That is the order the shortlist was *selected*
// on, so using it again as the *ordering* means nothing orders it at all.
// Selection and ordering are different jobs.
//
// GE's score is the engagement signal the rerank call used to blend in, it costs
// no round trip, and it arrived with the candidates. Selection is untouched --
// these are still the top MAX_FEED_LENGTH by vector -- so this only reorders posts
// that have all already cleared the axis.
//
// It is NOT sorted on raw `score`, because those are not one scale. Measured over
// a live 500-candidate pool, the three generators in this feed's mix occupy
// disjoint ranges:
//
//   popularity     n=167   10.122 .. 13.271   (median 10.738)
//   network_likes  n=167    1.000 ..  2.000   (median  1.000)
//   two_tower      n=166    0.871 ..  0.909   (median  0.880)
//
// A raw sort therefore puts every popularity candidate above every network_likes
// candidate above every two_tower one, which orders the feed by which generator
// happened to emit a post -- arbitrary, and worse than the vector order it is
// replacing. So each generator's scores are converted to a rank within their own
// group first, the same trick fuseByRank uses to make the MiniLM and Jina
// probabilities comparable, and the groups are then interleaved on that. Best in
// its group scores 1, worst 0, and a generator that contributed a single candidate
// gets 0.5 rather than a free top slot.
//
// Falls back to leaving the order alone when the generator sent no numeric scores
// at all, since sorting every candidate by the same null would shuffle a list that
// at least means something. `errs` records only that fallback; ordering by GE
// score is the normal case and is not an error.
//
// `locWeight` blends the LoC rank back in (see blendLocAndEngagement in fuse.ts):
// at 0 engagement alone orders the list and the vector only breaks ties.
function orderByGeneratorScore(cands: Candidate[], errs: string[], locWeight = 0): void {
  if (!cands.some(c => typeof c.geScore === 'number')) {
    errs.push('order:no-gescore');
    return;
  }
  blendLocAndEngagement(cands, generatorRank(cands), locWeight);
}

type SkeletonBody = { feed: { post: string }[]; cursor?: string };


// --- Feed request diagnostics ---------------------------------------------
// One structured line per getFeedSkeleton request, whichever path served it
// (fresh, cont, reuse, stale, anon, error), so every request is countable.
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

// Deadline on the upstream call a feed request cannot proceed without. (There
// were two; the GE rerank call has been removed -- see (4).)
//
// Without one, a hung dependency does not fail — it just stops, and the request
// stalls until the platform kills the isolate. Nothing is written in that case:
// not the [feed] line, not the feed_logs row, because both are written at the end
// of a request that never reaches its end. So the failure this table most needs to
// count would be precisely the one it cannot see.
//
// A deadline converts that silence into a path the code already handles: a
// generate that gives up leaves an empty candidate pool (empty feed, already the
// behaviour when GE errors). That finishes the request, and therefore lands in
// feed_logs with `errors` set.
//
// Sized against the request's own budget, not each call in isolation. The whole
// pipeline had 10s (now 5s, see below); generate and embed run in series inside it, so a
// per-call deadline is only honest if the deadlines can all be paid at once. 8s
// for generate could not be: it left 2s for everything downstream, which means
// the deadline would never have fired before the isolate was killed -- exactly
// the silent stall it exists to prevent.
//
// Sized against the appview's deadline, which is the only one that decides
// whether a reader sees this feed. See REQUEST_BUDGET_MS: it is 5s, and a
// response that misses it is discarded whole -- the reader gets
// UpstreamFailure/"feed unavailable" no matter how good the feed was.
//
// This cap has been 8s, then 4s, then 6.5s, each sized against generate's own
// measured latency at num_candidates 1000 (p50 3.5s, p90 5.0s, max 6.2s in
// feed_logs). Every one of those numbers is larger than the budget the whole
// request now has, which is the real lesson: at a 1000-candidate pool this stage
// could not be made to fit, so the pool shrank instead (numCandidates 500 in
// feeds.ts). The GE call itself measures 1.2-1.9s there; the rest of t_cand is
// this worker decoding the payload, which halves with it. 2.5s covers the call
// with room for the decode, and the budget takes back anything it cannot pay.
//
// 2.5s was not enough from the edge. Over the week to 2026-09-27, 89 of 318
// fresh requests (28%) hit it and served an empty pool, while the same call made
// from outside Bunny, with a real 1000-URI exclude list, took 1.2-1.5s -- so the
// slow tail is on the edge-to-GE path, not in GE. Raised to 3.2s now that the
// rerank and the session-write reserve no longer need that time: a generate that
// uses all of it leaves ~500ms, below JINA_MIN_MS, so the request skips Jina and
// finishes on cached and MiniLM scores -- a worse-ranked feed instead of none.
const GE_GENERATE_TIMEOUT_MS = 3200;

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
// remain. That floor is the top of the measured generate range at the pool size we
// now ask for (1.4-2.2s hydrated at 500 candidates): a retry either gets a window
// a generate has actually been observed to finish in, or it does not run at all --
// a retry that is going to be aborted at the deadline only delays the empty pool
// the reader is already getting. Inside a 3.2s stage that makes "early" mean
// "failed within ~1s", which covers the failures worth retrying: a
// refused connection or an instant 502 lands in tens of milliseconds. A timeout is
// by construction not early -- it consumed the whole budget -- so it is never
// retried.
const GE_GENERATE_MAX_ATTEMPTS = 2;
const GE_GENERATE_RETRY_MIN_MS = 2200;
const GE_GENERATE_RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

// --- One budget for the whole request --------------------------------------
//
// The caps above, and JINA_TIMEOUT_MS, are per-call stall guards, each sized off
// its own call's measured latency. Summed, they bound nothing: the stages run in
// series inside one request, so deadlines that are each individually payable still
// describe a request that blows its deadline before any of them fires -- the
// silent stall they exist to prevent, reintroduced by the deadlines themselves.
//
// So the caps stay what they are ("longer than this and the call is hung") and
// the ceiling is enforced once, here: every stage takes min(its cap, what is left
// of the request). A stage that cannot get enough to finish does not start. It
// takes the failure path it already has -- empty pool, cached scores only, the
// vector's own order, no cursor -- and names itself in `errors`, which is the same
// outcome as aborting at the deadline minus the time spent waiting for it.
// 10s was right when it was written, and silently stopped being right. The
// appview wrapped its getFeedSkeleton fetch in a hardcoded
// `AbortSignal.timeout(10_000)`, so 10s was exactly the deadline this worker had
// to beat. On 2026-09-01 that became configurable and the default halved:
//
//   -    signal: AbortSignal.any([params.signal, AbortSignal.timeout(10_000)]),
//   +      AbortSignal.timeout(ctx.cfg.feedGenSkeletonTimeout),   // 5 * SECOND
//
// (bluesky-social/atproto cae15d9f, #5473). Nothing here failed, so nothing here
// noticed. A request that misses the appview's deadline is discarded whole: the
// reader gets UpstreamFailure / "feed unavailable" while feed_logs records a
// clean row with a full feed and no errors, because from this side the request
// succeeded. That is the one failure this table cannot see by design, and it was
// the majority case -- 165 of 247 fresh-pipeline requests over the week were past
// 5s, 69% of the signed-in ones, every day between 49% and 88%.
//
// So the budget is the appview's deadline minus the trip it does not cover: DNS,
// TLS and transfer between the appview and this edge, which AbortSignal.timeout
// starts counting before this worker is even entered and none of which is visible
// from in here. 700ms for that, which is deliberately more than a round trip
// should cost: the stages below are deadlined but the work *between* them is not
// -- decoding the GE payload, the dedup merge, the text gates, the prefilter dot
// products, serialising the body -- and that unmeasured remainder was 888ms of the
// 8667ms request that prompted all this. The bail at (3z) is what actually pays
// for it, by reading the clock after the expensive untimed stretch rather than
// assuming it was free.
//
// Prefer being early and slightly stale to being late and discarded: past the
// deadline a perfect feed is worth exactly nothing.
const APPVIEW_SKELETON_TIMEOUT_MS = 5000;
const APPVIEW_NETWORK_MARGIN_MS = 700;
const REQUEST_BUDGET_MS = APPVIEW_SKELETON_TIMEOUT_MS - APPVIEW_NETWORK_MARGIN_MS;
// Held out of the budget for the work after the last stage: serializing the body
// and the response. None of that is deadlined, so it
// has to come out of the budget rather than off the end of it.
const RESPONSE_RESERVE_MS = 500;

// Floors. Below these a stage cannot finish even at the fastest it has been
// observed to run, so starting it would spend the reader's remaining time to
// reach the same failure path, later. Measured at numCandidates 500: generate
// 1.4-2.2s, one Jina round 0.7-3.9s.
const GE_GENERATE_MIN_MS = 1200;
const JINA_MIN_MS = 700;

// Cap on any single blocking DB call. The driver bounds one HTTP attempt
// (DB_REQUEST_TIMEOUT_MS) but not the retry sequence around it, and these calls
// take no AbortSignal, so they are bounded at the call site instead. An indexed
// round-trip measures ~30ms, so 600ms only fires when the DB is not answering --
// including the cold-replica attach that has been observed to take tens of
// seconds. Losing that race is the intended outcome, now more than ever: under a
// 4.5s budget a request makes several of these calls, and 1.5s each was a third of
// the budget apiece for work that normally costs 30ms. A cold attach cannot be
// waited out and still leave time for a feed, so the request continues without
// exclusion, or without a cursor, instead of being killed mid-wait having logged
// nothing.
const DB_STAGE_TIMEOUT_MS = 600;

// Below this much budget left after the candidates are in, no paid stage can run
// (see (3z)), and the request is tagged `budget:gather-overran`. It no longer
// changes what is served -- the stages skip themselves on their own floors and the
// request finishes on cached and free scores -- so this is now only the threshold
// the tag is reported at, kept at its old value so feed_logs stays comparable.
const PIPELINE_FINISH_MIN_MS = 900;

// The stale read is allowed this much even when the budget is gone -- see
// serveStale. It is the last step before an empty screen, and it is one indexed
// round-trip (~30ms) in the normal case.
const STALE_READ_FLOOR_MS = 400;

// The paid embed stage's share of the request. JINA_TIMEOUT_MS (15s) is a
// transport-level stall guard for a call that has no business taking that long,
// and it is far too large to slice a 3.8s request against: as a cap it meant
// "take everything left". Measured, one Jina round is 0.2-2.0s; 1.2s is what this
// request can afford to give it, and the batching already adapts to the deadline
// it is handed.
const JINA_STAGE_TIMEOUT_MS = 1200;

// Nothing is held back for writes any more. The session and score-cache writes
// both run after the response (runAfterResponse), so the only thing left to
// reserve for is the response itself (RESPONSE_RESERVE_MS). The per-stage caps
// still keep one stage from spending everything: the paid stage is the last one
// that waits on the network, and JINA_STAGE_TIMEOUT_MS bounds it.

type Budget = {
  remaining: () => number;
  slice: (cap: number) => number;
};

function requestBudget(t0: number): Budget {
  const deadline = t0 + REQUEST_BUDGET_MS - RESPONSE_RESERVE_MS;
  const remaining = () => Math.max(0, deadline - Date.now());
  const slice = (cap: number) => Math.min(cap, remaining());
  return { remaining, slice };
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
//
// The limit counts distinct posts. It used to count rows, and a post shown twice
// is two rows: 1000 rows was ~830 posts against the ~1200 a daily reader sees in
// a week, so a third of what they had seen could never be excluded.
const SEEN_EXCLUDE_LIMIT = 2000;
// How long GE generate waits for the seen list before starting without it. A
// warm read is ~45ms; a cold one on the edge has been observed past 600ms, and a
// generate that waited for it had to be sent with no exclusion at all -- every
// such list came back 50-75% posts the reader had already seen. Now a slow read
// no longer holds up GE: it finishes alongside the generate (SEEN_READ_TIMEOUT_MS,
// which costs no latency since generate takes longer) and the seen posts are
// dropped from the pool in the worker instead. The only thing lost that way is
// GE backfilling the pool with unseen posts.
const SEEN_WAIT_MS = 300;
const SEEN_READ_TIMEOUT_MS = 2000;
const SEEN_RETENTION_DAYS = 7;
const SEEN_PRUNE_PROBABILITY = 0.02;

// Cursor pagination. Page 1 runs the full generate→embed→rank pipeline (exactly
// as before) and freezes the ranked URI list in a feed_sessions row; later pages
// slice a window out of that frozen list instead of regenerating. This makes
// intra-session paging cheap and repeat-free regardless of interactionSeen
// timing, while seen-exclusion still runs at page 1 for cross-session freshness.
//
// Cursor wire format: `${sessionId}.${offset}`. sessionId is a UUID (no dots),
// so we split on the last dot.
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
// The cache key's version half is scoreCacheVersion() in feeds.ts.

// Batch-load cached scores for `cands`. Best-effort and never rejects: a DB error
// or a missed deadline just means "all miss". Bounded like every other DB call even
// though missing the cache makes the Jina stage bigger, not smaller -- a read that
// has not answered in DB_STAGE_TIMEOUT_MS was never going to return hits in time to
// save anything.
function readScoreCache(cands: Candidate[], feed: FeedConfig, budget: Budget): Promise<Map<string, number>> {
  return withDeadline(
    getCachedScores(cands.map(c => c.uri), scoreCacheVersion(feed.scorer)),
    budget.slice(DB_STAGE_TIMEOUT_MS), 'score-cache-read')
    .catch((e: any) => { console.error("[score cache] read failed:", errChain(e)); return new Map<string, number>(); });
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
  /**
   * The cache read, already in flight. The request starts it as soon as the pool
   * is known (see readScoreCache), so it overlaps the prefilter instead of queueing
   * behind it. May cover more URIs than `cands`; only lookups are done on it.
   */
  cachedP?: Promise<Map<string, number>>,
  /**
   * Whether a candidate without a Jina score still has something to be ranked on.
   * True when the free prefilter scored the pool: a total Jina outage then leaves
   * the shortlist in MiniLM order rather than failing the request, which is what a
   * single failed chunk already does. False on the single-stage path, where Jina
   * is the only score and an outage with no cache hits still has to be an error.
   */
  degradable = false,
): Promise<ScoreResult> {
  const scores: (number | null)[] = new Array(cands.length).fill(null);

  const cached = await (cachedP ?? readScoreCache(cands, feed, budget));

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
    const jinaMs = budget.slice(JINA_STAGE_TIMEOUT_MS);
    if (jinaMs < JINA_MIN_MS) {
      console.error(`[score cache] skipping Jina: ${jinaMs}ms of budget left, serving ${hits} cached score(s)`);
      errs.push('jina:no-budget');
      return { scores, tokens: 0, hits, misses: missUris.length, chunks: 0, failedChunks: 0 };
    }

    // getEmbeddingsBatched throws only when EVERY chunk failed (a total Jina
    // outage). Ride that out whenever there is something left to rank on: cached
    // scores, or the free prefilter's scores for the whole shortlist (`degradable`).
    // Only with neither -- no hits, no prefilter -- is there nothing to serve, and
    // then the handler returns its 500 as before.
    let res: Awaited<ReturnType<typeof getEmbeddingsBatched>>;
    try {
      res = await getEmbeddingsBatched(missTexts, apiKey, jinaMs);
    } catch (e: any) {
      if (hits === 0 && !degradable) throw e;
      console.error(`[score cache] all Jina chunks failed; serving ${hits} cached score(s)${degradable ? ' + prefilter order' : ''}:`, e?.message ?? String(e));
      errs.push(stageErr('jina:all-failed', e));
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

    // Persist the freshly-computed scores, after the response. This was awaited
    // inline for certainty, and that certainty depended on budget held back for
    // the session write behind it. With that write also deferred there is no
    // reserve left to protect it: a Jina call that used its whole slice would
    // leave this write no time, and the cache would silently stop warming.
    // Nothing in this request reads what it writes, and waitUntil is confirmed
    // on this runtime (it carries every feed_logs row). If it ever fails
    // persistently, the signal is cache_hit falling towards 0 in feed_logs.
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
    runAfterResponse(
      putScores(toStore).catch((e: any) => console.error("[score cache] write failed:", errChain(e)))
    );
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

      const feedUriParam = url.searchParams.get('feed') || '';
      const rkey = feedUriParam.split('/').pop() || 'unknown';
      // Which vector ranks this request, and at what gate. An unknown rkey falls
      // back to the locus-of-control feed rather than erroring.
      const feedConfig = resolveFeed(rkey);
      const startCursor: string | undefined = url.searchParams.get('cursor') || undefined;
      const limit = parseLimit(url.searchParams.get('limit'));
      // What this reader's client will show of whatever we return (see
      // readerLangs): a German-only reader gets a German-first list.
      const rLangs = readerLangs(request.headers.get('accept-language'));
      const deReader = prefersGermanOnly(rLangs);
      // Stored with every frozen list, so a list is only ever re-served to a
      // request that would have built it the same way (see feedSessions.langMode).
      const langMode: 'de' | 'any' = deReader ? 'de' : 'any';

      // (1) Signed-in readers only. Everything below is built around a reader:
      // candidates come from who they follow, seen posts are excluded, and the
      // ranked list is frozen under their DID. There used to be a whole second
      // path for logged-out requests (GE's `popularity` mix under a borrowed DID,
      // a shared skeleton cache, a public last-good list), and essentially no
      // traffic ever used it. A logged-out request now gets an empty feed at once,
      // which the app shows as an empty feed rather than an error.
      const userDid = didFromAuth(request);
      if (!userDid) {
        logFeed({ feed: rkey, did: 'anon', cursor: cursorTag(startCursor), cache: 'miss', page: 'anon',
          feedLen: 0, nextCursor: 'n', ms: Date.now() - t0 });
        return new Response(JSON.stringify({ feed: [] } satisfies SkeletonBody), {
          headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Page-Path': 'anon' },
        });
      }
      const didTag = shortDid(userDid);

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
        const ownerOk = !!session && session.did === userDid;
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
              'X-Feed-Version': 'minilm-prefilter-2',
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
      // The seen-post read the fresh pipeline needs below, started now so it runs
      // alongside the (2b) read instead of after it. The two are independent, and
      // in series they each took their own DB_STAGE_TIMEOUT_MS out of the budget:
      // ~30ms apiece normally, but a cold replica attach lands on whichever read
      // comes first, and the second one paid for it again out of the generate
      // stage's time. Only started where it will almost always be used -- a signed-in
      // START request. A cursor request is served by (2) far more often than not, so
      // its seen read is started lazily below, on the rare fall-through.
      const readSeen = (did: string) => withDeadline(
        getSeenUris(did, SEEN_EXCLUDE_LIMIT, SEEN_RETENTION_DAYS),
        budget.slice(SEEN_READ_TIMEOUT_MS), 'seen-uris');
      const seenEarly: Promise<string[]> | null = !startCursor ? readSeen(userDid) : null;
      // Subscribed now, because nothing awaits it until (2b) has finished: a
      // rejection landing in between would otherwise be unhandled at the isolate
      // level. The real handler below still receives the rejection.
      seenEarly?.catch(() => {});

      if (!startCursor) {
        const latest = await withDeadline(
          getLatestFeedSession(userDid, langMode), budget.slice(DB_STAGE_TIMEOUT_MS), 'latest-session-read')
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
          // The early seen read is not needed on this path. Its fetch was issued
          // inside the request, so let it settle rather than abandoning it
          // mid-flight; withDeadline already absorbs its rejection.
          if (seenEarly) runAfterResponse(seenEarly.catch(() => {}));
          return new Response(JSON.stringify(body), {
            headers: {
              'Content-Type': 'application/json',
              'X-Cache': 'miss',
              'X-Page-Path': 'reuse',
              // Keep the PullZone CDN out of it; the session store already caches.
              'Cache-Control': 'no-store',
              'X-Feed-Version': 'minilm-prefilter-2',
            },
          });
        }
      }

      let candidates: Candidate[] = [];
      
      const tG = Date.now();
      const geHeaders = {
        'Content-Type': 'application/json',
        'X-API-Key': geApiKey
      };
      

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
      // tSeen is the time this request still had to wait for the read here, not
      // the read's own duration: a read started early (seenEarly) has usually
      // already finished behind the (2b) read, and then this is ~0.
      let tS_start = Date.now();
      let excludeUris: string[] = [];
      const seenP: Promise<string[] | null> = (seenEarly ?? readSeen(userDid))
        .catch((e: any) => {
          console.error("[getFeedSkeleton] seen-uris read failed:", errChain(e));
          errs.push(stageErr('seen-uris', e));
          return null;
        });
      // Ready within SEEN_WAIT_MS: GE gets it as exclude_uris and backfills the
      // pool with unseen posts. Not ready: GE starts without it, and the read is
      // collected after the gather and applied as a filter (see SEEN_WAIT_MS).
      let seenTimer: any;
      const seenQuick = await Promise.race([
        seenP,
        new Promise<'late'>(resolve => { seenTimer = setTimeout(() => resolve('late'), SEEN_WAIT_MS); }),
      ]);
      clearTimeout(seenTimer);
      const seenLate: Promise<string[] | null> | null = seenQuick === 'late' ? seenP : null;
      if (seenQuick !== 'late') excludeUris = seenQuick ?? [];
      let seenFiltered = 0;
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

      // Serve the most recent good list instead of nothing. Three call sites, one
      // body: the budget bail after the gather stage, the empty-list case at (4b),
      // and any future stage that finds itself with nothing worth sending.
      //
      // Reaches back SESSION_TTL_MS, deliberately wider than the
      // START_REUSE_TTL_MS window at (2b). That one is an optimisation -- it skips
      // work a fresh request would happily redo -- so 90s keeps it honest. This one
      // is the difference between a feed and no feed, so it reaches back the full
      // session lifetime. The cost is seen-exclusion: a list this old can repeat
      // posts already scrolled past, which is strictly better than showing none.
      //
      // The cursor points into the OLD session, which (2) already knows how to
      // read, re-check for ownership and freshness, and page, so nothing new has to
      // be stored for the continuation to work. The public list carries no session
      // of this reader's, so it is served as page 1 without a cursor rather than
      // with one (2) would reject.
      //
      // The read gets STALE_READ_FLOOR_MS even when the budget is spent. That is
      // the one place it is right to spend past the reserve: this is the last
      // thing standing between the reader and an empty screen, and the reserve
      // exists for exactly this sort of tail work. Bounded, so it cannot itself
      // push the response past the appview's deadline.
      const serveStale = async (
        reason: string,
        extra: Record<string, unknown>,
      ): Promise<Response | null> => {
        const stale = await withDeadline(
          getLatestFeedSession(userDid, langMode),
          Math.max(budget.slice(DB_STAGE_TIMEOUT_MS), STALE_READ_FLOOR_MS),
          'stale-fallback-read')
          .catch((e: any) => {
            console.error("[getFeedSkeleton] stale fallback read failed:", errChain(e));
            return null;
          });
        if (!stale || stale.uris.length === 0) return null;
        const ageMs = Date.now() - stale.createdAt;
        if (ageMs >= SESSION_TTL_MS) return null;

        const page = stale.uris.slice(0, limit);
        const body: SkeletonBody = { feed: page.map(uri => ({ post: uri })) };
        if (limit < stale.uris.length) {
          body.cursor = encodeCursor(stale.sessionId, limit);
        }
        // Named in `errors` rather than hidden by the recovery: a request that
        // needed this is still a request whose pipeline did not deliver, and the
        // rate it fires at is the thing worth watching.
        errs.push(reason);
        logFeed({
          feed: rkey, did: didTag, cursor: cursorTag(startCursor), cache: 'miss', page: 'stale',
          off: 0, limit, feedLen: page.length,
          nextCursor: body.cursor ? 'y' : 'n',
          ms: Date.now() - t0,
          waitUntil: hasWaitUntil() ? 'y' : 'n',
          errors: errs.join(','),
          ...extra,
        });
        return new Response(JSON.stringify(body), {
          headers: {
            'Content-Type': 'application/json',
            'X-Cache': 'miss',
            'X-Page-Path': 'stale',
            // Personalized or not, this body is one reader's list and is already
            // served from our own stores; keep the PullZone CDN out of it.
            'Cache-Control': 'no-store',
            'X-Feed-Version': 'minilm-prefilter-2',
            'X-Stale-Age-Ms': String(ageMs),
          },
        });
      };

      const tC_start = Date.now();
      let candRaw = 0, candShort = 0;
      let preHydrated = 0, preKept = 0, preFloor = 0, preFellBack = false;
      let preBypass = 0, preBypassDropped = 0;
      try {
        const buildBody = (gens: { name: string; weight: number }[]) => {
          const b: any = {
            generators: gens,
            infill: 'popularity',
            user_did: userDid,
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
          { name: 'followed_users', weight: 1.0 },
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
            merged.push({
              ...c,
              uri: c.at_uri,
              text: c.content || '',
              // Before anything downstream overwrites `score`. See Candidate.geScore.
              geScore: typeof c.score === 'number' ? c.score : null,
            });
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

      // (3x) The seen list that missed SEEN_WAIT_MS. It has been loading alongside
      // the generate, under its own SEEN_READ_TIMEOUT_MS, so this is normally
      // already settled. Apply it as a filter: the posts GE would have excluded
      // are dropped here instead, and the pool is that much smaller.
      if (seenLate) {
        const late = await seenLate;
        errs.push(late ? 'seen:late' : 'seen:none');
        if (late && late.length) {
          const seen = new Set(late);
          const before = candidates.length;
          candidates = candidates.filter(c => !seen.has(c.uri));
          seenFiltered = before - candidates.length;
          excludeUris = late;
        }
      }

      // (3y) Score-cache read, started the moment the pool is known rather than
      // after the prefilter has picked the paid pool. It covers the whole pool
      // (~474 URIs, three concurrent chunks, the same ~30ms as one), so it runs
      // while the prefilter below does its CPU work, and so the cached Jina scores
      // are already in hand if the budget runs out before anything is paid for.
      // Never rejects; a failure is an empty map, i.e. every candidate a miss.
      const cachedEarly = candidates.length > 0
        ? readScoreCache(candidates, feedConfig, budget)
        : null;

      // (3z) Short on time before the paid stages. This used to stop here and
      // serve the reader's last list, up to SESSION_TTL_MS old: no seen-exclusion
      // since it was built, and nothing at all for a reader with no recent list.
      // With the cache read already in flight the pipeline has a cheaper finish
      // than that. Jina skips itself on a budget this small (JINA_MIN_MS), so the
      // rest of the request is the cache read that is already running, the MiniLM
      // ranking and a local sort; both writes happen after the response. What the
      // reader gets is this request's own pool: cached Jina scores where they
      // exist and the free score for the rest, which is also what `jina:no-budget`
      // has always served. Only if that comes out empty does (4b) fall back to the
      // last list.
      //
      // Tagged under the same name as before so feed_logs stays comparable. The
      // row now reads page=fresh with this in `errors`, and page=stale means (4b)
      // had to step in.
      if (budget.remaining() < PIPELINE_FINISH_MIN_MS) {
        console.error(`[getFeedSkeleton] short of budget after gather: ${budget.remaining()}ms left, finishing on cached + free scores`);
        errs.push('budget:gather-overran');
      }

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
            //
            // Except for a reader whose client hides English (deReader). Sharing
            // the paid budget between the two groups is the right policy when both
            // groups can reach the reader; when only one can, it buys scores for
            // posts that will be dropped on arrival and rations the ones that
            // won't. So that reader's budget goes to the German group first and
            // the English shortlist keeps the remainder — the same number of paid
            // candidates, the same Jina bill, spent where it can be seen.
            const paidBudget = pf.shortlist + bypassMax;
            const taken = detour.slice(0, deReader
              ? Math.min(paidBudget, detour.length)
              : bypassBudget(
                  pf.shortlist, detour.length, routed.length - detour.length,
                  pf.bypassFloor ?? bypassMax, bypassMax));
            const kept = routed
              .filter(x => !x.bypass && x.p !== null)
              .sort((a, b) => (b.p as number) - (a.p as number))
              .slice(0, deReader ? Math.max(0, paidBudget - taken.length) : pf.shortlist);
            preBypass = taken.length;
            preBypassDropped = detour.length - taken.length;
            // Where the cut actually landed on this pool. With no fixed floor this
            // is the only way to see the stage's operating point, and it is what a
            // future recalibration of the paid gate has to be read against. The
            // bypassed tail has no score of its own and is not part of it. Read
            // before the reorder below, while `kept` is still in free-score order.
            preFloor = kept.length ? (kept[kept.length - 1].p as number) : 0;

            // The order the paid pool goes to Jina in. Selection is settled; this
            // only decides which posts land in the early batches, which are the
            // ones that come back when the deadline cuts the stage short (Jina
            // throttles per key, so later requests queue behind earlier ones).
            //
            // Bypassed posts first: without a Jina score they are dropped outright,
            // while an English post without one is still ranked on its free score.
            // Then the English by GE engagement rank, because that is the feed's
            // final order: the posts heading for page 1 get the best chance of a
            // paid score. Nothing else reads this order -- the fuse and the final
            // sort both re-sort.
            const keptRank = generatorRank(kept.map(x => x.c));
            const keptByEngagement = kept
              .map((x, i) => ({ x, r: keptRank[i] }))
              .sort((a, b) => b.r - a.r)
              .map(k => k.x);
            paidPool = [...taken.map(x => x.c), ...keptByEngagement.map(x => x.c)];
            miniScores = [...taken.map(() => null), ...keptByEngagement.map(x => x.p)];
            bypassed = [...taken.map(() => true), ...keptByEngagement.map(() => false)];
            preKept = paidPool.length;
          } else {
            preFellBack = true;
            console.error(`[getFeedSkeleton] prefilter disabled: only ${preHydrated}/${candidates.length} candidates hydrated`);
          }
        }

        // (3b) Reuse cached scores; only cache-missed posts are sent to Jina.
        const scored = await scoreCandidates(
          paidPool, apiKey, feedConfig, budget, errs, cachedEarly ?? undefined, miniScores !== null);
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
            // Overwrite the generator's `score` (kept as geScore) with the vector
            // probability, which fuseByRank and the sort below order on, and which
            // breaks ties in orderByGeneratorScore.
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
        // Before the cut, not after: MAX_FEED_LENGTH is the whole length of
        // the feed, so a German post that falls past it is one this reader can
        // never be shown, however well it scored.
        if (deReader) germanFirst(filteredCandidates);
        if (filteredCandidates.length > MAX_FEED_LENGTH) {
          filteredCandidates.length = MAX_FEED_LENGTH;
        }

        tEmbed = Date.now() - tE_start;

        // (4) Final order. Selection is done -- these are the feed's posts -- and
        // what is left is which of them the reader sees first. There is no remote
        // reranker any more: GE's /rank/predict needed the pipeline finished by
        // ~2.85s to fit inside the appview's 5s, which it almost never was, so the
        // order below was already what nearly every reader got. Doing it here, in
        // the worker, costs no round trip and gives the budget it used to hold
        // for the call back to the paid scoring stage.
        //
        // 'score' keeps the vector's own (fused) order, for a feed whose vector is
        // the whole editorial criterion. The default orders by each generator's
        // own engagement score, which is what the rerank call was there to add.
        const tR_start = Date.now();
        if (filteredCandidates.length > 0 && feedConfig.rankBy !== 'score') {
          orderByGeneratorScore(filteredCandidates, errs, feedConfig.locWeight ?? 0);
        }
        tRank = Date.now() - tR_start;
      }

      // The final order has just sorted on engagement and score, which threads
      // the English back through the list. Re-apply the partition last so the
      // frozen list — and therefore page 1, which is the only page most readers
      // see — leads with what this reader can be shown. Stable, so the order
      // chosen above survives inside each group.
      if (deReader) germanFirst(filteredCandidates);

      const tSt_start = Date.now();

      // The full ranked list for this scroll session. Page 1 freezes it in a
      // feed_sessions row and serves only the first `limit` posts; the cursor
      // lets later pages slice further into this exact ordering without
      // regenerating. (Previously the entire list was returned as a single page.)
      const rankedUris = filteredCandidates.map(c => c.uri);

      // (4b) Nothing to serve. Every stage above has a failure path that ends here
      // with an empty list: GE timed out, GE answered with an empty pool, the text
      // gates dropped everything, Jina scored nothing usable. Each was written as
      // "serve what we have", and what we have at this point is an empty skeleton.
      if (rankedUris.length === 0) {
        const staleRes = await serveStale('fallback:empty', {
          candRaw, candShort, cand: candidates.length, pass: 0,
          gatherMs, tSeen, tPrune, tCand, tEmbed, tRank, tStore,
        });
        if (staleRes) return staleRes;
      }

      // Freeze the list after the response, not before it. The write used to be
      // awaited, and 450ms of every request was held back for it, which is budget
      // the paid stage now gets. The cost is that the cursor goes out before the
      // row exists. That is safe in
      // practice: the next page is only requested once the reader has scrolled
      // through this one, while the write takes ~30ms. If it does fail, the cursor
      // misses at (2) and falls through to a fresh page, the same path an expired
      // session already takes, with seen-exclusion covering what was shown. A
      // failed write is therefore only visible in the console log, not in
      // feed_logs, since that row is written before the write settles.
      let sessionId = '';
      if (rankedUris.length > 0) {
        sessionId = crypto.randomUUID();
        runAfterResponse(
          putFeedSession(sessionId, userDid, rankedUris, langMode)
            .then(ok => { if (!ok) console.error("[getFeedSkeleton] session write reported failure"); })
            .catch((e: any) => console.error("[getFeedSkeleton] session write failed:", errChain(e)))
        );
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

      // Emit a continuation cursor whenever more posts remain past this page;
      // omitting it signals "end of feed". The session behind it is written after
      // the response (see above).
      const body: SkeletonBody = { feed };
      if (sessionId && limit < rankedUris.length) {
        body.cursor = encodeCursor(sessionId, limit);
      }

      logFeed({
        feed: rkey, did: didTag, cursor: cursorTag(startCursor), cache: 'miss', page: 'fresh',
        rlang: rLangs.join(',') || 'none', deFirst: deReader ? 'y' : 'n',
        candRaw, candShort, cand: candidates.length, pass: rankedUris.length,
        off: 0, limit, feedLen: feed.length,
        nextCursor: body.cursor ? 'y' : 'n',
        gatherMs, ms: Date.now() - t0,
        tSeen, tPrune, tCand, tEmbed, tRank, tStore,
        cacheHit: embedHits, cacheMiss: embedMisses,
        jinaChunks: embedChunks, jinaFailed: embedFailedChunks,
        preHydrated, preKept, preFloor: preFloor.toFixed(4), preFellBack: preFellBack ? 'y' : 'n',
        preBypass, preBypassDropped, seenFiltered,
        waitUntil: hasWaitUntil() ? 'y' : 'n',
        errors: errs.length ? errs.join(',') : null
      });

      return new Response(JSON.stringify(body), {
        headers: {
          'Content-Type': 'application/json',
          'X-Cache': 'miss',
          'X-Page-Path': 'fresh',
          // Never let the Bunny PullZone CDN cache the skeleton: its cache key
          // ignores the cursor and the per-user Authorization/DID, so a single
          // cached copy freezes the feed and defeats seen-post exclusion.
          'Cache-Control': 'no-store',
          // Diagnostics (readable off the HTTP response, no logs needed):
          // X-Feed-Version confirms which build is actually serving traffic;
          // X-Ranked-Count is the full frozen list length; X-Exclude-Count reports
          // how many seen-post URIs were sent to GE as exclude_uris this request.
          'X-Feed-Version': 'minilm-prefilter-2',
          // What the reader's client told us it will display, and whether that
          // moved their language to the front of the list.
          'X-Reader-Lang': `${rLangs.join(',') || 'none'}${deReader ? ' de-first' : ''}`,
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
      const userDid = didFromAuth(request);

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

  // --- PRESCORE JOB ---
  // Called by an external cron (.github/workflows/prescore.yml), since Bunny has
  // no scheduler. See prescore.ts. Disabled unless PRESCORE_TOKEN is set, and
  // answers 404 to anything without it, so the route is not advertised.
  if (url.pathname === '/_jobs/prescore' && request.method === 'POST') {
    const token = getEnv('PRESCORE_TOKEN');
    if (!token || request.headers.get('Authorization') !== `Bearer ${token}`) {
      return new Response('Not found', { status: 404 });
    }
    const headers = { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' };
    try {
      const t0 = Date.now();
      const report = await runPrescore();
      console.log('[prescore] ' + JSON.stringify({ ...report, ms: Date.now() - t0 }));
      return new Response(JSON.stringify({ ...report, ms: Date.now() - t0 }), { headers });
    } catch (e: any) {
      console.error('[prescore] failed:', errChain(e));
      const status = e instanceof PrescoreConfigError ? 409 : 500;
      return new Response(JSON.stringify({ error: errChain(e) }), { status, headers });
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
