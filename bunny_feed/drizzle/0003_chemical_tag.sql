PRAGMA foreign_keys=OFF;--> statement-breakpoint
CREATE TABLE `__new_post_scores` (
	`uri` text NOT NULL,
	`score` real NOT NULL,
	`version` text NOT NULL,
	`created_at` integer NOT NULL,
	PRIMARY KEY(`uri`, `version`)
);
--> statement-breakpoint
INSERT INTO `__new_post_scores`("uri", "score", "version", "created_at") SELECT "uri", "score", "version", "created_at" FROM `post_scores`;--> statement-breakpoint
DROP TABLE `post_scores`;--> statement-breakpoint
ALTER TABLE `__new_post_scores` RENAME TO `post_scores`;--> statement-breakpoint
PRAGMA foreign_keys=ON;--> statement-breakpoint
CREATE INDEX `post_scores_created_at_idx` ON `post_scores` (`created_at`);