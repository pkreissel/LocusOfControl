import { drizzle } from 'drizzle-orm/sqlite-proxy';
import { embeddingUsage, feedInteractions, feedSessions, postEmbeddings } from './schema';
import { sql, and, eq, lt, gte, inArray, desc, isNotNull } from 'drizzle-orm';

declare const process: any;
declare const Bunny: any;

function getEnv(key: string): string | undefined {
  if (typeof Bunny !== 'undefined' && Bunny.env) return Bunny.env[key];
  if (typeof process !== 'undefined' && process.env) return process.env[key];
  return undefined;
}

// --- Turso HTTP transport -------------------------------------------------
//
// We deliberately do NOT use @libsql/client here. On Bunny's edge runtime the
// libSQL web client pipelines statements through a long-lived `HttpStream` that
// flushes its `fetch` on a *deferred* queue/microtask. That flush escapes the
// active request's async context, and Bunny's `fetch` then throws
// `TypeError: Cannot read properties of undefined (reading 'loop')` because it
// requires a live request/event-loop binding. Every DB call died this way,
// which silently degraded the feed (e.g. seen-post exclusion and embedding
// caching stopped working).
//
// Instead we drive Turso's HTTP pipeline API directly through drizzle's
// sqlite-proxy driver: a single, inline, fully-awaited `fetch` per query — the
// same shape as the Jina/bsky calls that already work in-request.

type TursoValue =
  | { type: 'null' }
  | { type: 'integer'; value: string }
  | { type: 'float'; value: number }
  | { type: 'text'; value: string }
  | { type: 'blob'; base64: string };

function encodeArg(v: any): TursoValue {
  if (v === null || v === undefined) return { type: 'null' };
  switch (typeof v) {
    case 'bigint': return { type: 'integer', value: v.toString() };
    case 'number':
      return Number.isInteger(v)
        ? { type: 'integer', value: String(v) }
        : { type: 'float', value: v };
    case 'boolean': return { type: 'integer', value: v ? '1' : '0' };
    case 'string': return { type: 'text', value: v };
    default: return { type: 'text', value: String(v) };
  }
}

function decodeValue(cell: any): any {
  if (!cell) return null;
  switch (cell.type) {
    case 'null': return null;
    case 'integer': return Number(cell.value);
    case 'float': return cell.value;
    case 'text': return cell.value;
    case 'blob': return cell.base64;
    default: return cell.value ?? null;
  }
}

function toHttpUrl(url: string): string {
  return url
    .replace(/^libsql:\/\//, 'https://')
    .replace(/^wss:\/\//, 'https://')
    .replace(/^ws:\/\//, 'http://');
}

// Turso's edge intermittently returns transient transport failures — most
// commonly HTTP 421 ("It seems we are strugling to contact your database.
// Please retry in a few seconds.") during replica handoff, plus the usual 5xx
// and network drops. These are not query errors; the statement never ran, so a
// retry is safe (SELECTs and our idempotent upserts alike). A genuine SQL error
// ("Turso query failed") is NOT retried — retrying won't fix it.
//
// Retries are immediate (no setTimeout backoff) on purpose: this driver exists
// because work that escapes the active request's async context breaks Bunny's
// `fetch` (see the transport note above), and a timer callback would run outside
// that context. Immediate re-issue keeps every attempt an inline, in-request
// awaited fetch — the shape that actually works here.
const DB_RETRYABLE_STATUS = new Set([421, 429, 500, 502, 503, 504]);
const DB_MAX_ATTEMPTS = 3;

let _drizzleDb: ReturnType<typeof drizzle> | null = null;

export function getDbClient() {
  if (_drizzleDb) return _drizzleDb;
  const rawUrl = getEnv('BUNNY_DATABASE_URL') || getEnv('BUNNY_DB_URL');
  const authToken = getEnv('BUNNY_DATABASE_AUTH_TOKEN') || getEnv('BUNNY_DB_TOKEN');

  if (!rawUrl || !authToken) {
    const msg = "Missing BUNNY_DATABASE_URL or BUNNY_DATABASE_AUTH_TOKEN in environment";
    console.error(`[DB Error]: ${msg}`);
    throw new Error(msg);
  }

  const endpoint = `${toHttpUrl(rawUrl).replace(/\/$/, '')}/v2/pipeline`;

  _drizzleDb = drizzle(async (sqlText, params, method) => {
    const body = JSON.stringify({
      requests: [
        { type: 'execute', stmt: { sql: sqlText, args: params.map(encodeArg) } },
        { type: 'close' },
      ],
    });

    let lastErr: any;
    for (let attempt = 1; attempt <= DB_MAX_ATTEMPTS; attempt++) {
      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
          body,
        });

        if (!res.ok) {
          const errBody = await res.text().catch(() => '');
          const err: any = new Error(`Turso HTTP ${res.status}: ${errBody} (SQL: ${sqlText.slice(0, 100)}...)`);
          err.retryable = DB_RETRYABLE_STATUS.has(res.status);
          throw err;
        }

        const json: any = await res.json();
        const first = json?.results?.[0];
        if (!first || first.type === 'error') {
          // A real SQL error — surface it immediately, never retry.
          throw new Error(`Turso query failed: ${first?.error?.message ?? 'unknown error'} (SQL: ${sqlText.slice(0, 100)}...)`);
        }

        const result = first.response?.result;
        const rows: any[][] = (result?.rows ?? []).map((row: any[]) => row.map(decodeValue));

        // sqlite-proxy expects a single row array for `get`, array-of-rows otherwise.
        return { rows: method === 'get' ? rows[0] : rows };
      } catch (e: any) {
        lastErr = e;
        // Retry transient transport failures: a flagged-retryable HTTP status,
        // or a thrown fetch/network error (which has no `retryable` flag). A
        // "Turso query failed" SQL error is explicitly non-retryable.
        const isQueryError = typeof e?.message === 'string' && e.message.startsWith('Turso query failed');
        const transient = !isQueryError && e?.retryable !== false;
        if (!transient || attempt === DB_MAX_ATTEMPTS) throw e;
        console.warn(`[DB] transient failure (attempt ${attempt}/${DB_MAX_ATTEMPTS}) for SQL '${sqlText.slice(0, 50)}...', retrying:`, e?.message);
      }
    }
    throw lastErr;
  });

  return _drizzleDb;
}

export async function recordEmbeddingUsage(tokens: number) {
  if (!tokens) return;
  const db = getDbClient();
  const day = new Date().toISOString().slice(0, 10); // YYYY-MM-DD (UTC)
  await db.insert(embeddingUsage)
    .values({
      day,
      tokens,
      requests: 1
    })
    .onConflictDoUpdate({
      target: embeddingUsage.day,
      set: {
        tokens: sql`tokens + excluded.tokens`,
        requests: sql`requests + 1`,
        updatedAt: sql`CURRENT_TIMESTAMP`
      }
    });
}

// --- Embedding vector cache ----------------------------------------------
//
// Read/write per-post vectors in single batched round-trips (one SELECT ... IN
// for reads, one multi-row upsert for writes) instead of N per-key edge-cache
// ops. Stored globally in the DB so vectors are shared across all edge PoPs.

// Fetch cached vectors for `uris` produced by `model`. Returns a uri -> vector
// map containing only the hits; misses are simply absent.
export async function getCachedEmbeddings(uris: string[], model: string): Promise<Map<string, number[]>> {
  const out = new Map<string, number[]>();
  if (uris.length === 0) return out;
  const db = getDbClient();
  // Chunk only to stay well under SQLite's bound-parameter limit. Response size
  // is not a concern (Turso happily returns multi-MB result sets), and transient
  // failures are handled by the driver's retry, so keep chunks large to minimise
  // round-trips.
  const CHUNK_SIZE = 100;
  for (let i = 0; i < uris.length; i += CHUNK_SIZE) {
    const chunk = uris.slice(i, i + CHUNK_SIZE);
    const rows = await db
      .select({ uri: postEmbeddings.uri, vec: postEmbeddings.vec })
      .from(postEmbeddings)
      .where(and(inArray(postEmbeddings.uri, chunk), eq(postEmbeddings.model, model)));
    for (const r of rows as { uri: string; vec: string }[]) {
      try { out.set(r.uri, JSON.parse(r.vec)); } catch { /* skip corrupt row */ }
    }
  }
  return out;
}

// Upsert freshly-computed vectors in a single multi-row statement. `entries`
// must be de-duplicated by uri (SQLite rejects updating the same row twice in
// one statement).
export async function putEmbeddings(entries: { uri: string; vec: number[] }[], model: string): Promise<void> {
  if (entries.length === 0) return;
  const db = getDbClient();
  const now = Date.now();
  const CHUNK_SIZE = 100;
  for (let i = 0; i < entries.length; i += CHUNK_SIZE) {
    const chunk = entries.slice(i, i + CHUNK_SIZE);
    await db.insert(postEmbeddings)
      .values(chunk.map(e => ({ uri: e.uri, vec: JSON.stringify(e.vec), model, createdAt: now })))
      .onConflictDoUpdate({
        target: postEmbeddings.uri,
        set: { vec: sql`excluded.vec`, model: sql`excluded.model`, createdAt: sql`excluded.created_at` },
      });
  }
}

// Delete cached vectors older than `cutoffTs` (unix ms). Called opportunistically
// to keep the table bounded.
export async function pruneOldEmbeddings(cutoffTs: number): Promise<void> {
  const db = getDbClient();
  const toDelete = db.select({ uri: postEmbeddings.uri })
    .from(postEmbeddings)
    .where(lt(postEmbeddings.createdAt, cutoffTs))
    .limit(1000);
  await db.delete(postEmbeddings).where(inArray(postEmbeddings.uri, toDelete));
}

// --- Cursor-pagination session store -------------------------------------
//
// A scroll session's frozen ranked URI list. Page 1 stores the full ranked list
// here; later pages read it back and slice a window, so pagination never re-runs
// the generate→embed→rank pipeline and can't repeat posts within the session.

// Store the ranked URI list for a session. Returns true only if the write
// landed, so the caller emits a continuation cursor only when the list is
// actually retrievable. Best-effort: never throws.
export async function putFeedSession(id: string, did: string | null, uris: string[]): Promise<boolean> {
  try {
    const db = getDbClient();
    await db.insert(feedSessions).values({
      id,
      did,
      uris: JSON.stringify(uris),
      createdAt: Date.now(),
    });
    return true;
  } catch (e: any) {
    console.error("[feed session] write failed:", e?.cause?.message || e?.message);
    return false;
  }
}

// Fetch a session's frozen list. Returns null on miss or parse failure so the
// caller falls back to a fresh generation.
export async function getFeedSession(
  id: string,
): Promise<{ uris: string[]; did: string | null; createdAt: number } | null> {
  const db = getDbClient();
  const rows = await db
    .select({ uris: feedSessions.uris, did: feedSessions.did, createdAt: feedSessions.createdAt })
    .from(feedSessions)
    .where(eq(feedSessions.id, id))
    .limit(1);
  const row = (rows as { uris: string; did: string | null; createdAt: number }[])[0];
  if (!row) return null;
  try {
    const uris = JSON.parse(row.uris);
    if (!Array.isArray(uris)) return null;
    return { uris, did: row.did, createdAt: row.createdAt };
  } catch {
    return null;
  }
}

// Delete sessions older than `cutoffTs` (unix ms). Sessions live only for the
// length of a scroll session, so this keeps the table small; called
// opportunistically like the other prunes.
export async function pruneOldFeedSessions(cutoffTs: number): Promise<void> {
  const db = getDbClient();
  const toDelete = db.select({ id: feedSessions.id })
    .from(feedSessions)
    .where(lt(feedSessions.createdAt, cutoffTs))
    .limit(1000);
  await db.delete(feedSessions).where(inArray(feedSessions.id, toDelete));
}

export async function recordInteraction(did: string | null, item: string, event: string, feedContext?: string) {
  const db = getDbClient();
  await db.insert(feedInteractions)
    .values({
      id: crypto.randomUUID(),
      did,
      item,
      event,
      feedContext: feedContext || null
    });
}

// Record a batch of interactions in a single multi-row insert. A scroll burst
// arrives as many `interactionSeen` events at once; inserting them one-await-at-
// a-time would be one Turso round-trip per shown post. Chunked to stay under
// SQLite's bound-parameter limit.
export async function recordInteractions(
  did: string | null,
  interactions: { item: string; event: string; feedContext?: string }[],
): Promise<void> {
  if (interactions.length === 0) return;
  const db = getDbClient();
  const CHUNK_SIZE = 100;
  for (let i = 0; i < interactions.length; i += CHUNK_SIZE) {
    const chunk = interactions.slice(i, i + CHUNK_SIZE);
    await db.insert(feedInteractions).values(
      chunk.map(it => ({
        id: crypto.randomUUID(),
        did,
        item: it.item,
        event: it.event,
        feedContext: it.feedContext || null,
      })),
    );
  }
}

// The interactionSeen event id — the "shown" signal. Both the exclusion read
// below and the seen-row prune key off this single event, so the read scope
// (what we exclude) and the write retention (what we keep) stay in lockstep.
const SEEN_EVENT = 'app.bsky.feed.defs#interactionSeen';

// Fetch the most recent post URIs this user has been *shown*, newest first, so
// candidate generation can exclude them and stop repeating posts across
// pages/sessions. Deliberately scoped to `interactionSeen` only: engagement
// events (likes/reposts/replies/requestMore/Less) must NOT hide a post from the
// feed — a liked or "more like this" post is not a seen post, and excluding on
// them would silently bury content the user engaged with. Bounded two ways: at
// most `limit` rows, and never older than `withinDays` (the same window the
// seen-row prune uses, so we never look back past retention). Backed by the
// (did, created_at) index. created_at is the SQLite CURRENT_TIMESTAMP text
// format, matching datetime('now', ...).
export async function getSeenUris(did: string, limit: number, withinDays: number): Promise<string[]> {
  if (!did) return [];
  const db = getDbClient();
  const rows = await db
    .select({ item: feedInteractions.item })
    .from(feedInteractions)
    .where(and(
      eq(feedInteractions.did, did),
      eq(feedInteractions.event, SEEN_EVENT),
      isNotNull(feedInteractions.item),
      gte(feedInteractions.createdAt, sql`datetime('now', ${`-${withinDays} days`})`),
    ))
    .orderBy(desc(feedInteractions.createdAt))
    .limit(limit);
  const out: string[] = [];
  for (const r of rows as { item: string | null }[]) if (r.item) out.push(r.item);
  return out;
}

// Prune ONLY high-volume `interactionSeen` rows older than the retention window.
// Engaged rows (likes/reposts/replies/quotes/requestLess) are kept indefinitely:
// the offline training pipeline (resolve_interaction_posts.py) reads them, and
// they are rare. Only the "shown" firehose grows unbounded. `created_at` is the
// SQLite CURRENT_TIMESTAMP text format ('YYYY-MM-DD HH:MM:SS'), so we compare
// against datetime('now', ...) which produces the identical format. Keys off the
// shared SEEN_EVENT id (declared above getSeenUris).
export async function pruneOldSeenInteractions(retentionDays: number): Promise<void> {
  const db = getDbClient();
  const cutoff = sql`datetime('now', ${`-${retentionDays} days`})`;

  const toDelete = db.select({ id: feedInteractions.id })
    .from(feedInteractions)
    .where(and(eq(feedInteractions.event, SEEN_EVENT), lt(feedInteractions.createdAt, cutoff)))
    .limit(1000);

  await db.delete(feedInteractions).where(inArray(feedInteractions.id, toDelete));
}
