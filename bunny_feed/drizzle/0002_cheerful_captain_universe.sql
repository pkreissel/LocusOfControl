CREATE TABLE `post_scores` (
	`uri` text PRIMARY KEY NOT NULL,
	`score` real NOT NULL,
	`version` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `post_scores_created_at_idx` ON `post_scores` (`created_at`);