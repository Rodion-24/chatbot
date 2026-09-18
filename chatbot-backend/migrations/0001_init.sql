-- 0001_init: base schema for the documentation RAG assistant.
--
-- Note: no HNSW index on chunks.embedding on purpose. We want an exact
-- baseline for recall measurement before an approximate index is introduced.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id         bigserial PRIMARY KEY,
  title      text NOT NULL,
  source_uri text,
  version    text,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE chunks (
  id            bigserial PRIMARY KEY,
  doc_id        bigint REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index   int NOT NULL,
  content       text NOT NULL,
  section_title text,
  meta          jsonb NOT NULL DEFAULT '{}',
  embedding     vector(1536),
  embed_model   text NOT NULL,
  tsv           tsvector GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED
);

CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_doc_id_idx ON chunks (doc_id);
