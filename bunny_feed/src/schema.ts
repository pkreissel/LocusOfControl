import { sqliteTable, text, integer } from 'drizzle-orm/sqlite-core';
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
});

export const feedInteractions = sqliteTable('feed_interactions', {
  id: text('id').primaryKey(),
  did: text('did'),
  item: text('item'),
  event: text('event'),
  feedContext: text('feed_context'),
  createdAt: text('created_at').default(sql`CURRENT_TIMESTAMP`),
});

export const oauthStates = sqliteTable('oauth_states', {
  key: text('key').primaryKey(),
  data: text('data').notNull(),
  updatedAt: text('updated_at').default(sql`CURRENT_TIMESTAMP`),
});

export const oauthSessions = sqliteTable('oauth_sessions', {
  key: text('key').primaryKey(),
  data: text('data').notNull(),
  updatedAt: text('updated_at').default(sql`CURRENT_TIMESTAMP`),
});

export const oauthLocks = sqliteTable('oauth_locks', {
  name: text('name').primaryKey(),
  expiresAt: integer('expires_at').notNull(),
});
