CREATE TABLE conversation_history (
	id serial PRIMARY KEY NOT NULL,
	conversation_id uuid DEFAULT gen_random_uuid() NOT NULL,
	human_message TEXT NOT NULL,
	ai_message TEXT NOT NULL,
	existing_summary TEXT DEFAULT '',
	created_at timestamp DEFAULT now(),
	created_by varchar(255) DEFAULT NULL,
	updated_at timestamp DEFAULT now(),
	updated_by varchar(255) DEFAULT NULL,
	deleted_at timestamp DEFAULT NULL,
	deleted_by varchar(255) DEFAULT NULL,
	"version" int DEFAULT 1
);