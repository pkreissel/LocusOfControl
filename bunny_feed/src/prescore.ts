// Prescore job: score another feed's new posts ahead of time, so that if Green
// Earth later offers one of them as a candidate, the feed finds its score in
// post_scores and skips the Jina call inside a request that has 4.3s to finish.
//
// Bunny has no scheduler, so an external cron (.github/workflows/prescore.yml)
// POSTs /_jobs/prescore every 5 minutes. Each call is one Edge Script request, and
// the binding limit on one is 50 subrequests (CPU is milliseconds here: parsing
// a feed page and one comparison per post). So a call does bounded work and
// leaves the rest for the next one:
//
//   1. Read the newest two pages of the source feed from the public AppView.
//      Keep posts the feed itself would score: minWords, under 72h old.
//   2. Look them up in post_scores under the current version: a post with a row
//      is done, whoever wrote it. For the rest, look in typesafe_answers: a post
//      already bought (its post_scores row pruned, lost to a failed write, or
//      filed under an older vector version) is re-mapped from its stored answer,
//      never bought again. A post whose TypeSafe call failed has no row in
//      either and is picked up by the next call.
//   3. Buy what is still missing, newest first, up to what the subrequest budget
//      allows. Record the answers first, with the vector's own score for the same
//      text, then write post_scores.
//
// What gets written depends on PRESCORE_SOURCE:
//
//   typesafe (default)  TypeSafe's judgement, placed on the deployed vector's
//                       probability scale by the fixed quantile map in
//                       prescore_calibration.json. The scale is the vector's, so
//                       the feed's threshold still means what it means; the ORDER
//                       among these posts is the teacher's. Nothing in
//                       post_scores records that a row came from here.
//   vector              the deployed vector's own score via the Jina API, i.e.
//                       exactly what the feed would have computed. Pure latency;
//                       changes nothing about what the feed serves.
//
// Every answer bought is kept in full in typesafe_answers (see schema.ts), which
// is what lets a retrain re-map them (rewarm_cache.ts) and the trainer use them
// (sync_typesafe_ledger.ts).

import { getCachedScores, getTypesafeAnswers, putScores, putTypesafeAnswers, type TypesafeAnswerRow } from './db';
import { resolveFeed, scoreCacheVersion, type FeedConfig } from './feeds';
import { getEmbeddings } from './jina';
import { hasEnoughText } from './text';
import tsQuestions from './ts_questions.json';
import calibration from './prescore_calibration.json';

declare const process: any;
declare const Bunny: any;

function getEnv(key: string): string | undefined {
  if (typeof Bunny !== 'undefined' && Bunny.env) return Bunny.env[key];
  if (typeof process !== 'undefined' && process.env) return process.env[key];
  return undefined;
}

// Silberskeets (sebastian.eurosky.social), the German feed the offline pipeline
// in research/feed/ has been measuring and warming by hand.
const DEFAULT_SOURCE_FEED = 'at://did:plc:ituoear7k6qx3smjfoxhufm4/app.bsky.feed.generator/aaanl422clvee';
const APPVIEW = 'https://public.api.bsky.app';
const TYPESAFE_URL = 'https://api.typesafe.ai/v1/systemone';

// --- Subrequest budget -------------------------------------------------------
//
// Bunny allows 50 per request. Everything the job fetches counts, including the
// DB driver's retries, which happen inside db.ts where this module cannot count
// them. So the budget is spent in advance, from the top:
const SUBREQUEST_LIMIT = 50;
const MAX_PAGES = 2;            // AppView getFeed pages
// post_scores read, typesafe_answers read (both <= 200 uris), answers write,
// post_scores write.
const DB_STATEMENTS = 4;
const DB_RETRY_RESERVE = 4;     // the driver makes up to 3 attempts per statement
const JINA_REQUESTS = 1;        // one request embeds the whole batch
// Whatever is left is the TypeSafe batch, one request per post:
// 50 - 2 - 4 - 4 - 1 = 39, minus one for luck. There are no in-call retries -- a
// failed post has no row and the next call picks it up -- so this number is
// exact, not an estimate.
const TYPESAFE_MAX = SUBREQUEST_LIMIT - MAX_PAGES - DB_STATEMENTS - DB_RETRY_RESERVE - JINA_REQUESTS - 1;
// Bunny's runtime misbehaves above ~8 concurrent operations (see the Jina and
// cache notes elsewhere). 38 posts at 8 wide is ~5 rounds of ~1s.
const TYPESAFE_CONCURRENCY = 8;
const TYPESAFE_TIMEOUT_MS = 20_000;

// A post older than GE's max_age_hours (72, index.ts) can never be a candidate
// again, so scoring it buys nothing.
const MAX_POST_AGE_MS = 72 * 60 * 60 * 1000;

type SourcePost = { uri: string; text: string; createdAt: number };

export type PrescoreReport = {
  tag: string;
  source: 'typesafe' | 'vector';
  pages: number;
  fetched: number;
  eligible: number;
  cached: number;
  missing: number;
  /** Missing from post_scores but already bought: re-mapped, not re-bought. */
  reused: number;
  scored: number;
  failed: number;
  pushed: number;
  /** Still missing after this call; the cron calls again while this is > 0. */
  remaining: number;
  subrequests: number;
  errors: string[];
};

export class PrescoreConfigError extends Error {}

// --- Calibration --------------------------------------------------------------

type BandKey = [number, number, number];

/**
 * calibrate_typesafe_to_vector.py's `band` key: external floored below
 * everything, then P(internal), then the graded Score breaking ties. The Python
 * original and export_prescore_assets.py must agree with this; test_prescore.ts
 * checks it against the offline pipeline's own output.
 */
export function bandKey(classProbs: Record<string, number>, score: number): BandKey {
  const pi = classProbs.internal ?? 0, pe = classProbs.external ?? 0;
  const floor = pe > Math.max(pi, 1 - pi - pe) ? 0 : 1;
  return [floor, pi, score];
}

function cmpKey(a: readonly number[], b: readonly number[]): number {
  return (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]);
}

/**
 * Place `key` among the reference pool's keys and read the vector's probability
 * ladder at that rank. The offline fit gives the i-th smallest key the i-th
 * smallest probability; a new key lands between two reference ranks and is
 * interpolated, and a key tied with a block of reference keys takes the middle
 * of that block (the offline sort broke such ties arbitrarily).
 */
export function calibrate(
  key: BandKey,
  cal: { keys: number[][]; ladder: number[] } = calibration,
): number {
  const { keys, ladder } = cal;
  const n = keys.length;
  let lo = 0, hi = n;
  while (lo < hi) { const m = (lo + hi) >> 1; if (cmpKey(keys[m], key) < 0) lo = m + 1; else hi = m; }
  const first = lo;
  hi = n;
  while (lo < hi) { const m = (lo + hi) >> 1; if (cmpKey(keys[m], key) <= 0) lo = m + 1; else hi = m; }
  const end = lo;
  const t = Math.min(n - 1, Math.max(0, end > first ? (first + end - 1) / 2 : first - 0.5));
  const i = Math.floor(t), f = t - i;
  return i + 1 < n ? ladder[i] * (1 - f) + ladder[i + 1] * f : ladder[i];
}

// --- Upstream calls -------------------------------------------------------------

async function fetchFeedPage(
  feedUri: string, cursor: string | undefined,
): Promise<{ posts: SourcePost[]; cursor?: string }> {
  const params = new URLSearchParams({ feed: feedUri, limit: '100' });
  if (cursor) params.set('cursor', cursor);
  const r = await fetch(`${APPVIEW}/xrpc/app.bsky.feed.getFeed?${params}`, {
    signal: AbortSignal.timeout(10_000),
  });
  if (!r.ok) throw new Error(`appview getFeed ${r.status}`);
  const data: any = await r.json();
  const posts: SourcePost[] = [];
  for (const item of data.feed ?? []) {
    const post = item?.post;
    const createdAt = Date.parse(post?.record?.createdAt ?? '');
    if (!post?.uri || !Number.isFinite(createdAt)) continue;
    posts.push({ uri: post.uri, text: post.record?.text ?? '', createdAt });
  }
  return { posts, cursor: data.cursor };
}

/**
 * The answer fields score_loc_typesafe.py records per post, so a row synced into
 * training/data/typesafe_ledger.jsonl is indistinguishable from one it wrote.
 */
export type TypesafeAnswer = {
  ts_score: number;
  ts_score_probs: Record<string, number>;
  ts_score_conf: number;
  ts_class: string;
  ts_class_probs: Record<string, number>;
  ts_class_conf: number;
  ts_noul: number;
  input_tokens: number;
  model: string;
  text_key: string;
};

/**
 * score_loc_typesafe.text_key(): sha256 of the whitespace-normalised text, first
 * 16 hex. Keys on the billed unit, so the local ledger recognises the same text
 * under another uri. JS \s and Python's str.split() agree on every whitespace
 * character a post plausibly carries (test_prescore.ts checks a few).
 */
export async function textKey(text: string): Promise<string> {
  const norm = text.split(/\s+/).filter(Boolean).join(' ');
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(norm));
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
}

export function answerKey(a: Pick<TypesafeAnswer, 'ts_class_probs' | 'ts_score'>): BandKey {
  return bandKey(a.ts_class_probs, a.ts_score);
}

/** One TypeSafe call. Resolves to the full answer, or throws. */
async function askTypesafe(text: string, apiKey: string, model: string): Promise<TypesafeAnswer> {
  const r = await fetch(TYPESAFE_URL, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ state: text, model, questions: tsQuestions.questions }),
    signal: AbortSignal.timeout(TYPESAFE_TIMEOUT_MS),
  });
  if (!r.ok) throw new Error(`typesafe ${r.status}`);
  const resp: any = await r.json();
  const a = resp?.answers;
  if (!a?.loc_class?.probabilities || typeof a?.loc_score?.score !== 'number' || !a?.agency) {
    throw new Error('typesafe: malformed answer');
  }
  return {
    ts_score: a.loc_score.score,
    ts_score_probs: a.loc_score.probabilities,
    ts_score_conf: a.loc_score.confidence,
    ts_class: a.loc_class.choice,
    ts_class_probs: a.loc_class.probabilities,
    ts_class_conf: a.loc_class.confidence,
    ts_noul: a.agency.noul,
    input_tokens: resp?.usage?.input_tokens ?? 0,
    model,
    text_key: await textKey(text),
  };
}

/** Run `fn` over `items`, at most `width` at a time, never rejecting. */
async function pool<T, R>(items: T[], width: number, fn: (x: T) => Promise<R>): Promise<(R | Error)[]> {
  const out: (R | Error)[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      try { out[i] = await fn(items[i]); } catch (e: any) { out[i] = e instanceof Error ? e : new Error(String(e)); }
    }
  };
  await Promise.all(Array.from({ length: Math.min(width, items.length) }, worker));
  return out;
}

// --- The job ----------------------------------------------------------------------

export async function runPrescore(feed: FeedConfig = resolveFeed('feed-forward')): Promise<PrescoreReport> {
  const source = (getEnv('PRESCORE_SOURCE') || 'typesafe') as 'typesafe' | 'vector';
  if (source !== 'typesafe' && source !== 'vector') {
    throw new PrescoreConfigError(`PRESCORE_SOURCE must be 'typesafe' or 'vector', not '${source}'`);
  }
  const feedUri = getEnv('PRESCORE_FEED') || DEFAULT_SOURCE_FEED;
  const tag = getEnv('PRESCORE_TAG') || 'silberskeets';
  const tsKey = getEnv('TYPESAFE_API_KEY');
  const tsModel = getEnv('TYPESAFE_MODEL') || 'jev-latest';
  const jinaKey = getEnv('JINA_API_KEY');
  if (source === 'typesafe') {
    if (!tsKey) throw new PrescoreConfigError('TYPESAFE_API_KEY is not set');
    // The ladder is one vector's probability distribution. Under any other vector
    // it would put the teacher's order on a scale the feed's threshold was not
    // measured against, so refuse loudly rather than push miscalibrated scores.
    if (calibration.vector_version !== feed.scorer.version) {
      throw new PrescoreConfigError(
        `prescore_calibration.json was fitted for vector ${calibration.vector_version}, ` +
        `the feed runs ${feed.scorer.version}; re-run training/export_prescore_assets.py`);
    }
  }
  // Needed in both modes: in typesafe mode it records the vector's own score next
  // to each answer, which is what makes vector-vs-teacher disagreement queryable.
  if (!jinaKey) throw new PrescoreConfigError('JINA_API_KEY is not set');

  const version = scoreCacheVersion(feed.scorer);
  const minWords = feed.minWords ?? 0;
  const errors: string[] = [];
  let subrequests = 0;
  const batchMax = source === 'typesafe' ? TYPESAFE_MAX : Infinity;

  // (1) Newest posts first, up to MAX_PAGES. In steady state (~20 new posts per 5
  // minutes) page 1 holds everything new; page 2 is what catches a backlog.
  const now = Date.now();
  const byUri = new Map<string, SourcePost>();
  let fetched = 0, pages = 0, cursor: string | undefined;
  while (pages < MAX_PAGES) {
    let page;
    try {
      subrequests++; pages++;
      page = await fetchFeedPage(feedUri, cursor);
    } catch (e: any) {
      errors.push(`appview:${e?.message ?? e}`);
      break;
    }
    fetched += page.posts.length;
    for (const p of page.posts) {
      if (now - p.createdAt > MAX_POST_AGE_MS) continue;
      if (!p.text.trim() || !hasEnoughText(p.text, minWords)) continue;
      byUri.set(p.uri, p);
    }
    cursor = page.cursor;
    // Stop early only when there is no more feed or the page already ran past
    // the age window. Otherwise both budgeted pages are read: which posts are
    // already scored is only known after the cache read, and two pages keep that
    // read to one statement (<= 200 uris).
    const oldest = page.posts.length ? Math.min(...page.posts.map(p => p.createdAt)) : 0;
    if (!cursor || now - oldest > MAX_POST_AGE_MS) break;
  }
  const eligible = [...byUri.values()];

  // (2) What the cache already holds. Throwing here is right: without the read
  // the job cannot tell done from not-done and would buy everything again.
  subrequests++;
  const cached = eligible.length ? await getCachedScores(eligible.map(p => p.uri), version) : new Map();
  const missing = eligible.filter(p => !cached.has(p.uri)).sort((a, b) => b.createdAt - a.createdAt);

  const toStore: { uri: string; score: number; version: string }[] = [];

  // (2b) Already bought, only missing from post_scores: re-map the stored answer.
  // Same reasoning as above for letting a failed read throw -- the alternative is
  // buying again.
  let reused = 0;
  let toBuy = missing;
  if (source === 'typesafe' && missing.length) {
    subrequests++;
    const bought = await getTypesafeAnswers(missing.map(p => p.uri));
    for (const p of missing) {
      const raw = bought.get(p.uri);
      if (!raw) continue;
      toStore.push({ uri: p.uri, score: calibrate(answerKey(JSON.parse(raw))), version });
      reused++;
    }
    toBuy = missing.filter(p => !bought.has(p.uri));
  }
  const batch = toBuy.slice(0, batchMax);

  // (3) Score the batch: the vector's own score for every post (one Jina request),
  // and TypeSafe's answer when that is the source.
  let failed = 0;
  let vectorProbs: (number | null)[] = batch.map(() => null);
  if (batch.length) {
    subrequests += JINA_REQUESTS;
    try {
      const { embeddings } = await getEmbeddings(batch.map(p => p.text), jinaKey!);
      vectorProbs = embeddings.map(e => feed.scorer.score(e).probability);
    } catch (e: any) {
      // In typesafe mode the answers still stand without it; they are recorded
      // with a null vector_prob. In vector mode there is nothing to push.
      errors.push(`jina:${e?.message ?? e}`);
    }
  }

  if (batch.length && source === 'typesafe') {
    subrequests += batch.length;
    const answers = await pool(batch, TYPESAFE_CONCURRENCY, p => askTypesafe(p.text, tsKey!, tsModel));
    const rows: TypesafeAnswerRow[] = [];
    const now2 = Date.now();
    answers.forEach((a, i) => {
      if (a instanceof Error) {
        failed++;
        if (errors.length < 10) errors.push(`typesafe:${a.message}`);
        return;
      }
      const { model, text_key, ...fields } = a;
      rows.push({
        uri: batch[i].uri, text: batch[i].text, textKey: text_key, tag, model,
        answer: JSON.stringify({ ...fields, model, text_key }),
        vectorProb: vectorProbs[i], vectorVersion: feed.scorer.version,
        postCreatedAt: batch[i].createdAt, createdAt: now2,
      });
      toStore.push({ uri: batch[i].uri, score: calibrate(answerKey(a)), version });
    });
    // The record of spend goes first. If it fails, the answers just bought are
    // lost and the next call buys them again -- the one way this job can pay
    // twice -- so it throws and fails the cron run loudly.
    if (rows.length) {
      subrequests++;
      await putTypesafeAnswers(rows);
    }
  } else if (batch.length) {
    batch.forEach((p, i) => {
      const v = vectorProbs[i];
      if (v === null) { failed++; return; }
      toStore.push({ uri: p.uri, score: v, version });
    });
  }

  if (toStore.length) {
    subrequests++;
    await putScores(toStore);
  }

  return {
    tag, source, pages, fetched,
    eligible: eligible.length,
    cached: eligible.length - missing.length,
    missing: missing.length,
    reused,
    scored: batch.length - failed,
    failed,
    pushed: toStore.length,
    remaining: missing.length - toStore.length,
    subrequests,
    errors,
  };
}
