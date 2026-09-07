import { sqliteTable, text, integer, real, index, primaryKey } from 'drizzle-orm/sqlite-core';
import { sql } from 'drizzle-orm';

export const embeddingUsage = sqliteTable('embedding_usage', {
  day: text('day').primaryKey(),
  tokens: integer('tokens').notNull().default(0),
  requests: integer('requests').notNull().default(0),
  updatedAt: text('updated_at').default(sql`CURRENT_TIMESTAMP`),
});

// DEPRECATED: superseded by `postScores` below and no longer read or written by
// the feed. Kept declared so the one-off migrate_db.ts helper still compiles and
// so the existing table isn't implicitly dropped; safe to remove (along with the
// table) once the live DB has been cleaned up.
//
// Cache of per-post embedding vectors, keyed by AT-URI. Storing the full 1024-dim
// vector as JSON text cost ~11.6 KB per row, so a 500-candidate lookup moved
// ~5.9 MB and took ~9s — slower than simply re-embedding all 500 from scratch
// (~6.5s). The vector was only ever consumed to compute one scalar, so the cache
// could never pay off in latency; `postScores` caches that scalar instead.
export const postEmbeddings = sqliteTable('post_embeddings', {
  uri: text('uri').primaryKey(),
  vec: text('vec').notNull(),
  model: text('model').notNull(),
  createdAt: integer('created_at').notNull(),
}, (t) => [
  // Serves the opportunistic prune (WHERE created_at < cutoff), which otherwise
  // full-scans. created_at is monotonic, so inserts append to the index edge.
  index('post_embeddings_created_at_idx').on(t.createdAt),
]);

// Cache of per-post *scores*, keyed by AT-URI — the feed's embedding cache.
//
// A candidate's embedding is used for exactly one thing: scoreEmbedding() folds
// it against the trained vector into a single probability, which becomes the
// candidate's rank `score`. The vector itself is never used again, so caching it
// (see the deprecated postEmbeddings above) shipped ~11.6 KB per post across the
// wire to recompute one float. This table stores that float: ~1500x less payload,
// which turns a ~9s cached read into ~50ms and makes the cache an actual latency
// win rather than a cost-only one.
//
// `version` scopes entries to BOTH the embedding model and the scoring vector
// (`${EMBEDDING_MODEL}:${scorer.version}`), so changing either one invalidates
// old rows via an ordinary cache miss. Pruned opportunistically by `createdAt`.
//
// The primary key is (uri, version), not uri alone. Once one deployment served
// more than one feed, each with its own vector, a uri-only key meant the two
// feeds fought over a single row: whichever ran last overwrote the other's
// score, and the loser then read its own row back as a version mismatch. Both
// feeds would have missed on essentially every candidate forever, doubling the
// Jina bill while looking like a working cache. The composite key lets one post
// hold one score per scoring model, which is what the version column always
// meant.
export const postScores = sqliteTable('post_scores', {
  uri: text('uri').notNull(),
  score: real('score').notNull(),              // sigmoid probability from the feed's scorer
  version: text('version').notNull(),          // embedding model + scoring-vector fingerprint
  createdAt: integer('created_at').notNull(),  // unix ms, for the prune
}, (t) => [
  primaryKey({ columns: [t.uri, t.version] }),
  // Serves the opportunistic prune (WHERE created_at < cutoff), which otherwise
  // full-scans. created_at is monotonic, so inserts append to the index edge.
  index('post_scores_created_at_idx').on(t.createdAt),
]);

// Frozen ranked list for one scroll session, so cursor pagination pages through
// a stable ordering instead of re-running the expensive generate→embed→rank
// pipeline (and re-hitting the seen-exclusion timing race) on every page. Page 1
// computes the full ranked list, stores it here under an opaque id, and serves a
// window; later pages just slice `uris`. Stored in Turso (not the per-PoP edge
// cache) so a continuation request that lands on a different edge location still
// finds the list — a per-PoP cache would miss and fall back to regeneration,
// which can repeat the current session's not-yet-`interactionSeen` posts. Pruned
// opportunistically by `createdAt` (rows live only for a scroll session).
export const feedSessions = sqliteTable('feed_sessions', {
  id: text('id').primaryKey(),                 // opaque session id (crypto.randomUUID())
  did: text('did'),                            // owner DID (null = public); scopes reads + observability
  uris: text('uris').notNull(),                // JSON array of ranked at-uris (the frozen list)
  createdAt: integer('created_at').notNull(),  // unix ms, for TTL freshness check + prune
}, (t) => [
  // Serves the opportunistic prune (WHERE created_at < cutoff). created_at is
  // monotonic, so inserts append to the index edge.
  index('feed_sessions_created_at_idx').on(t.createdAt),
  // Serves the page-1 reuse read (getLatestFeedSession: WHERE did=? ORDER BY
  // created_at DESC LIMIT 1) so a logged-in START request can reuse the user's
  // most recent frozen ranked list instead of re-running the pipeline.
  index('feed_sessions_did_created_at_idx').on(t.did, t.createdAt),
]);

export const feedInteractions = sqliteTable('feed_interactions', {
  id: text('id').primaryKey(),
  did: text('did'),
  item: text('item'),
  event: text('event'),
  feedContext: text('feed_context'),
  createdAt: text('created_at').default(sql`CURRENT_TIMESTAMP`),
}, (t) => [
  // Serves the seen-post exclusion read (WHERE did=? ORDER BY created_at DESC
  // LIMIT n) as an index-only range scan, and the plain did lookup. Without it
  // that read is a full table scan that degrades as "shown" rows accumulate.
  index('feed_interactions_did_event_created_at_item_idx').on(t.did, t.event, t.createdAt, t.item),
  // Serves the seen-row prune (WHERE event=? AND created_at<?). event is
  // low-cardinality and created_at monotonic, so inserts append to the edge —
  // cheap to maintain even though "shown" rows are now the insert firehose.
  index('feed_interactions_event_created_at_idx').on(t.event, t.createdAt),
]);
