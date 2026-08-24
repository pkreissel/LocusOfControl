CREATE TABLE `embedding_usage` (
	`day` text PRIMARY KEY NOT NULL,
	`tokens` integer DEFAULT 0 NOT NULL,
	`requests` integer DEFAULT 0 NOT NULL,
	`updated_at` text DEFAULT CURRENT_TIMESTAMP
);
--> statement-breakpoint
CREATE TABLE `feed_interactions` (
	`id` text PRIMARY KEY NOT NULL,
	`did` text,
	`item` text,
	`event` text,
	`feed_context` text,
	`created_at` text DEFAULT CURRENT_TIMESTAMP
);
--> statement-breakpoint
CREATE INDEX `feed_interactions_did_event_created_at_item_idx` ON `feed_interactions` (`did`,`event`,`created_at`,`item`);--> statement-breakpoint
CREATE INDEX `feed_interactions_event_created_at_idx` ON `feed_interactions` (`event`,`created_at`);--> statement-breakpoint
CREATE TABLE `feed_sessions` (
	`id` text PRIMARY KEY NOT NULL,
	`did` text,
	`uris` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `feed_sessions_created_at_idx` ON `feed_sessions` (`created_at`);--> statement-breakpoint
CREATE TABLE `post_embeddings` (
	`uri` text PRIMARY KEY NOT NULL,
	`vec` text NOT NULL,
	`model` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `post_embeddings_created_at_idx` ON `post_embeddings` (`created_at`);