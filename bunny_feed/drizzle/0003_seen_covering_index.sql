-- Replace the existing did/createdAt index with a covering index that includes event and item.
-- This makes the getSeenUris hot-read query a true index-only scan.

DROP INDEX IF EXISTS `feed_interactions_did_created_at_idx`;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS `feed_interactions_did_event_created_at_item_idx`
  ON `feed_interactions` (`did`, `event`, `created_at`, `item`);
