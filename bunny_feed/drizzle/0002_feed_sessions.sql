-- Frozen ranked-list store backing cursor pagination (feed_sessions).
--
-- Deployment note: this project syncs schema with `pnpm db:push`
-- (drizzle-kit push, dialect turso), which diffs src/schema.ts against the live
-- DB and creates this table directly. This file is an IF-NOT-EXISTS,
-- runnable-by-hand equivalent for applying the same change without a push.

-- Page 1 freezes its full ranked URI list here under an opaque id; continuation
-- pages read it back and slice a window instead of regenerating.
CREATE TABLE IF NOT EXISTS `feed_sessions` (
  `id` text PRIMARY KEY NOT NULL,
  `did` text,
  `uris` text NOT NULL,
  `created_at` integer NOT NULL
);
--> statement-breakpoint

-- Prune: pruneOldFeedSessions -> WHERE created_at<?
CREATE INDEX IF NOT EXISTS `feed_sessions_created_at_idx`
  ON `feed_sessions` (`created_at`);
