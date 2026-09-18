# docs-rag

RAG assistant over documentation. Skeleton: a working streaming chat endpoint, provider
abstraction, database schema and migrations. Retrieval and ingestion are interfaces only.

## Stack

Python 3.12 · uv · FastAPI (async) · Postgres 17 + pgvector · asyncpg · httpx · pydantic-settings · Langfuse. No LangChain.

## Quick start

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY (or OPENAI_API_KEY)
uv sync
docker compose up -d db langfuse
uv run python -m src.db.migrate
uv run uvicorn src.api.app:app --reload
```

The `db` service publishes on host port **5433** by default (5432 is usually taken by
another local Postgres); override with `DB_HOST_PORT`. Inside compose, services reach it
at `db:5432`.

Or the whole stack in containers:

```bash
docker compose up --build
```

## Endpoints

| Method | Path           | Description                                  |
| ------ | -------------- | -------------------------------------------- |
| `GET`  | `/health`      | Readiness — checks the DB, 503 when down      |
| `GET`  | `/health/live` | Liveness — process only                       |
| `POST` | `/api/chat`    | Question in, answer streamed back over SSE    |

### `POST /api/chat`

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I configure the database?"}'
```

Responds as `text/event-stream` with four event types:

```
event: token
data: {"text": "You "}

event: usage
data: {"model":"claude-opus-5","input_tokens":21,"output_tokens":58,"total_tokens":79,"cost_usd":0.00155,"stop_reason":"end_turn"}

event: done
data: {"ok": true}
```

An `error` event is emitted instead of `done` if the stream fails after headers are sent.

## How it fits together

- **Providers** — `LLMProvider` and `EmbeddingProvider` are `Protocol`s in
  [src/providers/base.py](src/providers/base.py); Anthropic and OpenAI implement them and
  [factory.py](src/providers/factory.py) picks one from `LLM_PROVIDER` / `EMBEDDING_PROVIDER`.
- **Retries** — every provider call goes through
  [with_retry](src/providers/retry.py) (exponential backoff, full jitter) on top of the SDKs'
  own retry handling.
- **Cost** — token usage is read off each response and priced by
  [pricing.py](src/providers/pricing.py); it lands on the SSE `usage` event and the Langfuse span.
- **Tracing** — `@observe` decorators from [src/observability.py](src/observability.py). Without
  Langfuse keys they are no-ops, so the app runs with no observability backend.
- **Database** — asyncpg pool, plain-SQL forward-only migrations
  (`uv run python -m src.db.migrate`), repositories with explicit column lists.

## Schema

`documents` and `chunks` per [migrations/0001_init.sql](migrations/0001_init.sql). `chunks.tsv`
is a generated `tsvector` with a GIN index. There is **no HNSW index** on `chunks.embedding` —
an exact baseline is wanted for recall measurement first.

## Not implemented

Chunking, embedding generation, retrieval, reranking and evals are stubs — see
[src/retrieval/](src/retrieval/), [src/ingest/](src/ingest/) and [evals/](evals/).
