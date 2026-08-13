import { sqliteTable, text, integer, index } from 'drizzle-orm/sqlite-core';
import { sql } from 'drizzle-orm';

export const embeddingUsage = sqliteTable('embedding_usage', {
  day: text('day').primaryKey(),
  tokens: integer('tokens').notNull().default(0),
  requests: integer('requests').notNull().default(0),
  updatedAt: text('updated_at').default(sql`CURRENT_TIMESTAMP`),
});

// Cache of per-post embedding vectors, keyed by AT-URI. Stored in the DB (not
// the per-PoP edge cache) so a vector computed at one edge location is reused
// globally, and so reads/writes batch into a single round-trip. `model` scopes
// entries to the embedding model that produced them (a retrain invalidates old
// rows via a cache miss). `vec` is the float array as JSON text. Pruned
// opportunistically by `createdAt`.
export const postEmbeddings = sqliteTable('post_embeddings', {
  uri: text('uri').primaryKey(),
  vec: text('vec').notNull(),
  model: text('model').notNull(),
  createdAt: integer('created_at').notNull(),
}, (t) => ({
  // Serves the opportunistic prune (WHERE created_at < cutoff), which otherwise
  // full-scans. created_at is monotonic, so inserts append to the index edge.
  createdIdx: index('post_embeddings_created_at_idx').on(t.createdAt),
}));

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
}, (t) => ({
  // Serves the opportunistic prune (WHERE created_at < cutoff). created_at is
  // monotonic, so inserts append to the index edge.
  createdIdx: index('feed_sessions_created_at_idx').on(t.createdAt),
}));

export const feedInteractions = sqliteTable('feed_interactions', {
  id: text('id').primaryKey(),
  did: text('did'),
  item: text('item'),
  event: text('event'),
  feedContext: text('feed_context'),
  createdAt: text('created_at').default(sql`CURRENT_TIMESTAMP`),
}, (t) => ({
  // Serves the seen-post exclusion read (WHERE did=? ORDER BY created_at DESC
  // LIMIT n) as an index-only range scan, and the plain did lookup. Without it
  // that read is a full table scan that degrades as "shown" rows accumulate.
  didCreatedIdx: index('feed_interactions_did_created_at_idx').on(t.did, t.createdAt),
  // Serves the seen-row prune (WHERE event=? AND created_at<?). event is
  // low-cardinality and created_at monotonic, so inserts append to the edge —
  // cheap to maintain even though "shown" rows are now the insert firehose.
  eventCreatedIdx: index('feed_interactions_event_created_at_idx').on(t.event, t.createdAt),
}));
