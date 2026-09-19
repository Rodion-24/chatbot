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

## Iterations and metrics

Each row is one change, measured against the 60-question golden set in
`evals/dataset/golden.json`. Run it with `uv run python -m evals.runner`.

| # | Change | Overall | factual | multihop | refusal acc | p50 | $/run |
|---|---|---|---|---|---|---|---|
| 1 | **Baseline** — naive 500/50 chunking, exact vector search, prompt v1 | 88% | 92% | 83% | 78% | 2.28s | $0.24 |
| 2 | **Prompt v2** — answer language and number formatting | 93% | 96% | 83% | **100%** | 1.96s | $0.25 |
| 3 | **Structural chunking** — split on headings, code blocks kept whole, heading path in every chunk | **95%** | 96% | **100%** | 100% | 2.11s | $0.26 |
| 4 | **LLM judge** on `impossible` / `ambiguous` (Haiku 4.5) — grades meaning instead of substrings | 92% | 96% | 100% | 89%* | 2.10s | $0.28 |
| 5 | **`tsv` → `english`** — the stemmer now matches the corpus language | — | — | — | — | — | $0 |
| 6 | **Hybrid search** — vector + full-text fused with RRF, 20 candidates per arm | **97%** | **100%** | 100% | 89%* | 2.07s | $0.33 |

### What each change bought

**Prompt v2.** The baseline answered an English question in German (`#15`):
the rule "answer in the language of the question" was applied to the language
of the retrieved excerpts. Tightening that rule, plus requiring numbers to be
written as the docs write them (`2,000`, not `2.000`), took refusal accuracy
from 78% to 100%.

**Hybrid search.** Dense retrieval alone could not surface literal
identifiers: asked "What should I set `lists` to?", it returned chunks *about*
indexing while the chunk holding `rows / 1000` never made the top 5. Adding a
lexical arm and fusing with RRF fixed that. One detail mattered more than the
fusion itself — `plainto_tsquery` ANDs every term, so a natural-language
question only matches a chunk containing *all* its words; rewriting the query
to OR moved the right chunk from unranked to rank 1. `ambiguous` went 50% →
83%, `factual` back to 100%, and p95 latency dropped from 5.2s to 3.9s.

**Structural chunking.** The naive 500-character window broke **29 of 94**
fenced code blocks and cut lists in half — asked for the three keys to IVFFlat
recall, the bot correctly reported that only the first was visible in its
excerpt. Splitting on markdown headings yields **0 broken code blocks** and
stamps each chunk with its heading path. multihop went from 83% to 100%.

### Retrieval measured on its own

`context_recall` — the share of expected chunks a search actually surfaces —
is scored without calling the model at all:

```bash
uv run python -m evals.runner --mode retrieval
```

| Retriever | context recall | Cost | Wall clock |
|---|---|---|---|
| Vector only | 66% | $0.00 | 8s |
| **Hybrid (RRF)** | **79%** | $0.00 | 8s |

This is the loop used while iterating on chunking and search: a full run costs
$0.33 and four minutes, this costs nothing and finishes before you look away.

### Where it does not work

- **`ambiguous` dropped to 67%.** The metric demands exact substrings for
  questions that have no single right answer — a flaw in the scoring, not
  the assistant.
- **The LLM judge is not reproducible.** Across three runs on identical
  answers, **8 of 15** judged verdicts changed (mostly between `refused` and
  `correct`, which both count as success for `impossible` — but #41 and #57
  swung as far as `incorrect`). The starred 89% above is therefore ±1
  question depending on the run. Substring checks stay the headline metric
  precisely because they are deterministic; the judge is a second opinion on
  the two categories where substrings are structurally unable to decide.
- **`ambiguous` sits at 50%** and is the weakest category. Two of the three
  failures (#26, #59) are genuine retrieval misses — the answer is in the
  README but the search does not surface it.
- **`context_recall` covers 43 of 60 questions.** The 9 `impossible` ones have
  no source by definition; 8 more need their expected chunks picked by hand
  (`uv run python -m evals.find_sources` proposes candidates, it does not
  decide). Chunk ids also shift on every re-ingest, so the field has to be
  regenerated whenever the chunking strategy changes.
- **`#26` still fails.** "Which Postgres versions does pgvector support?" —
  the answer (`Postgres 13+`) sits in an Installation chunk that neither arm
  ranks highly, because the question's wording shares almost no vocabulary
  with it. A reranker over a wider candidate set is the next lever.
- **No HNSW index, deliberately** — exact search is the recall baseline an
  approximate index will be measured against.

## Schema

`documents` and `chunks` per [migrations/0001_init.sql](migrations/0001_init.sql). `chunks.tsv`
is a generated `tsvector` with a GIN index. There is **no HNSW index** on `chunks.embedding` —
an exact baseline is wanted for recall measurement first.

## Not implemented

Chunking, embedding generation, retrieval, reranking and evals are stubs — see
[src/retrieval/](src/retrieval/), [src/ingest/](src/ingest/) and [evals/](evals/).
