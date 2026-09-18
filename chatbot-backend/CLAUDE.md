# CLAUDE.md

RAG assistant over documentation. FastAPI + Postgres 17/pgvector, fully async, no LangChain.

## Commands

```bash
uv sync                                  # install deps
uv run uvicorn src.api.app:app --reload  # run dev server
uv run python -m src.db.migrate          # apply migrations
uv run ruff check . && uv run ruff format --check .
uv run pytest
docker compose up -d db langfuse         # local infra
```

## Layout

- `src/api/` — FastAPI app, routers, SSE streaming
- `src/providers/` — `LLMProvider` / `EmbeddingProvider` protocols + Anthropic and OpenAI impls, retries, pricing
- `src/retrieval/` — search interface (stub)
- `src/ingest/` — ingestion interface (stub)
- `src/db/` — asyncpg pool, migration runner, repositories
- `src/config.py` — pydantic-settings, env-driven
- `migrations/` — plain SQL, applied in filename order
- `evals/` — golden dataset + runner (stub)

## Rules

- **Everything async.** No blocking calls in handlers — no `requests`, no `time.sleep`, no sync DB drivers. CPU-bound work goes through `asyncio.to_thread`.
- **No `SELECT *`.** List columns explicitly in every query.
- **No LangChain** or similar frameworks. Provider SDKs and httpx directly.
- **Providers go through the protocols** in `src/providers/base.py`. Adding a vendor means a new module plus a branch in `factory.py` — handlers never import a vendor SDK.
- **Every provider call is retried** via `src/providers/retry.py` (exponential backoff, full jitter) and **priced** via `src/providers/pricing.py`.
- **New models need a pricing entry.** `estimate_cost_usd` returns `None` for unknown models so the gap shows up in traces rather than reporting $0.
- **Langfuse decorators** (`@observe` from `src/observability.py`) on anything that calls a model. Tracing degrades to a no-op without keys — never let it break a request.
- **Migrations are forward-only.** Add a new numbered file; don't edit an applied one.

## Deliberately not implemented

Chunking, embeddings generation, retrieval, reranking, evals. The interfaces exist; the bodies raise or return empty.

## Notes

- No HNSW index on `chunks.embedding` — an exact baseline is wanted for recall measurement before adding an approximate index.
- `chunks.tsv` is a generated column using the `russian` text search config; the GIN index is on it.
- `EMBEDDING_DIM` must match `vector(N)` in the migration. Changing it requires a new migration.
- Anthropic has no embeddings endpoint — `EMBEDDING_PROVIDER=anthropic` raises at startup.
