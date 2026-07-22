CREATE TABLE `embedding_usage` (
	`day` text PRIMARY KEY NOT NULL,
	`tokens` integer DEFAULT 0 NOT NULL,
	`requests` integer DEFAULT 0 NOT NULL,
	`updated_at` text DEFAULT CURRENT_TIMESTAMP
);
--> statement-breakpoint
CREATE TABLE `feed_interactions` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`did` text,
	`item` text,
	`event` text,
	`feed_context` text,
	`created_at` text DEFAULT CURRENT_TIMESTAMP
);
--> statement-breakpoint
CREATE TABLE `oauth_sessions` (
	`key` text PRIMARY KEY NOT NULL,
	`data` text NOT NULL,
	`updated_at` text DEFAULT CURRENT_TIMESTAMP
);
--> statement-breakpoint
CREATE TABLE `oauth_states` (
	`key` text PRIMARY KEY NOT NULL,
	`data` text NOT NULL,
	`updated_at` text DEFAULT CURRENT_TIMESTAMP
);
--> statement-breakpoint
CREATE TABLE `users` (
	`did` text PRIMARY KEY NOT NULL,
	`handle` text,
	`access_token` text,
	`refresh_token` text,
	`updated_at` text DEFAULT CURRENT_TIMESTAMP
);
