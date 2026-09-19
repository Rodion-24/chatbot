-- 0002: switch the full-text column to the `english` configuration.
--
-- The corpus is the pgvector documentation, which is written in English.
-- The `russian` stemmer in 0001 cannot fold `indexes`/`indexing` onto a
-- common root, which blunts the lexical half of hybrid search.
--
-- `tsv` is a generated column, so it cannot be altered in place: drop and
-- recreate. Postgres recomputes it for every existing row on ADD COLUMN,
-- so no re-ingestion is needed.

DROP INDEX IF EXISTS chunks_tsv_idx;

ALTER TABLE chunks DROP COLUMN tsv;

ALTER TABLE chunks
  ADD COLUMN tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
