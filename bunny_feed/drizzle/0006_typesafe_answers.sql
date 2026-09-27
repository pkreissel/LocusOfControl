CREATE TABLE `typesafe_answers` (
	`uri` text PRIMARY KEY NOT NULL,
	`text` text NOT NULL,
	`text_key` text NOT NULL,
	`tag` text NOT NULL,
	`model` text NOT NULL,
	`answer` text NOT NULL,
	`vector_prob` real,
	`vector_version` text NOT NULL,
	`post_created_at` integer,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `typesafe_answers_created_at_idx` ON `typesafe_answers` (`created_at`);