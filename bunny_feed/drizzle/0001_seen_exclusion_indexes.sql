-- Indexes for seen-post exclusion during candidate generation.
--
-- Deployment note: this project syncs schema with `pnpm db:push`
-- (drizzle-kit push, dialect turso), which diffs src/schema.ts against the live
-- DB and creates these indexes directly. This file is an IF-NOT-EXISTS,
-- runnable-by-hand equivalent for applying the same change without a push.

-- Hot read: getSeenUris -> WHERE did=? ORDER BY created_at DESC LIMIT n
CREATE INDEX IF NOT EXISTS `feed_interactions_did_created_at_idx`
  ON `feed_interactions` (`did`, `created_at`);
--> statement-breakpoint

-- Prune: pruneOldSeenInteractions -> WHERE event=? AND created_at<?
CREATE INDEX IF NOT EXISTS `feed_interactions_event_created_at_idx`
  ON `feed_interactions` (`event`, `created_at`);
--> statement-breakpoint

-- Prune: pruneOldEmbeddings -> WHERE created_at<?
CREATE INDEX IF NOT EXISTS `post_embeddings_created_at_idx`
  ON `post_embeddings` (`created_at`);
