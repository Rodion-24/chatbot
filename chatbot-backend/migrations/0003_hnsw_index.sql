-- 0003: approximate nearest-neighbour index on chunks.embedding.
--
-- Parameters are the pgvector defaults, confirmed by the sweep in
-- evals/recall_index.py: on this corpus recall@10 is already 100% at
-- ef_search=40 and collapses below ef_search=10, so there is no reason to
-- raise m or ef_construction and pay for a slower build.
--
-- vector_cosine_ops must match the operator the retrievers use (<=>);
-- an index built for a different distance function is simply ignored.
--
-- Note this trades exactness for speed: results may differ from a full
-- scan. Drop this index to restore exact search.

CREATE INDEX chunks_embedding_hnsw_idx ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
