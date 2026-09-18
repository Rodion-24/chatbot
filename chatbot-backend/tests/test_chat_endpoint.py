"""End-to-end check of the SSE chat endpoint with a stubbed provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest

from src.api.app import create_app
from src.providers.base import EmbeddingProvider, LLMProvider, Message, StreamResult, Usage
from src.retrieval.base import RetrievedChunk


class StubLLM(LLMProvider):
    name = "stub"

    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas
        self.closed = False

    async def stream_chat(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        result: StreamResult | None = None,
    ) -> AsyncIterator[str]:
        for d in self._deltas:
            yield d
        if result is not None:
            result.model = "claude-opus-5"
            result.stop_reason = "end_turn"
            result.usage = Usage(input_tokens=1000, output_tokens=500)

    async def aclose(self) -> None:
        self.closed = True


class StubEmbedder(EmbeddingProvider):
    """Never called: the retrieval stub below short-circuits before it runs."""

    name = "stub"
    model = "stub-embed"
    dim = 3

    async def embed(self, texts):
        return [[0.0] * self.dim for _ in texts], Usage(input_tokens=1)

    async def aclose(self) -> None:
        pass


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event = None
    for line in body.splitlines():
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:") and event:
            events.append((event, json.loads(line.removeprefix("data:").strip())))
    return events


@pytest.fixture
def stub_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=42, doc_id=3, content="HNSW index docs", score=0.21),
        RetrievedChunk(chunk_id=17, doc_id=3, content="IVFFlat index docs", score=0.34),
    ]


@pytest.fixture
def client(monkeypatch, stub_chunks) -> httpx.AsyncClient:
    """App wired with stub providers, skipping the DB-dependent lifespan."""
    # Retrieval needs a live pool, so replace it wholesale rather than
    # standing up Postgres for an endpoint test.
    async def fake_retrieve(embedder, settings, question):
        return stub_chunks

    monkeypatch.setattr("src.api.routers.chat._retrieve", fake_retrieve)

    app = create_app()
    app.state.llm = StubLLM(["Hello", ", ", "world"])
    app.state.embedder = StubEmbedder()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_liveness(client: httpx.AsyncClient) -> None:
    async with client:
        r = await client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_chat_streams_tokens_usage_and_done(client: httpx.AsyncClient) -> None:
    async with client:
        r = await client.post("/api/chat", json={"question": "hi"})
        body = r.text

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(body)
    kinds = [e for e, _ in events]

    assert kinds[0] == "sources"
    assert kinds.count("token") == 3
    assert "".join(d["text"] for e, d in events if e == "token") == "Hello, world"
    assert kinds[-2:] == ["usage", "done"]

    usage = next(d for e, d in events if e == "usage")
    assert usage["model"] == "claude-opus-5"
    assert usage["input_tokens"] == 1000
    assert usage["output_tokens"] == 500
    assert usage["total_tokens"] == 1500
    # 1000/1e6*$5 + 500/1e6*$25 = 0.005 + 0.0125
    assert usage["cost_usd"] == pytest.approx(0.0175)
    assert usage["stop_reason"] == "end_turn"


async def test_chat_emits_retrieved_sources(
    client: httpx.AsyncClient, stub_chunks: list[RetrievedChunk]
) -> None:
    async with client:
        r = await client.post("/api/chat", json={"question": "hi"})

    sources = next(d for e, d in _parse_sse(r.text) if e == "sources")
    assert [c["chunk_id"] for c in sources["chunks"]] == [42, 17]
    assert sources["chunks"][0]["score"] == pytest.approx(0.21)


async def test_chat_rejects_empty_question(client: httpx.AsyncClient) -> None:
    async with client:
        r = await client.post("/api/chat", json={"question": ""})
    assert r.status_code == 422
