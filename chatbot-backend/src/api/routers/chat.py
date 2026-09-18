"""POST /api/chat — retrieves documentation, then streams an answer over SSE."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import ChatRequest, UsagePayload
from src.config import Settings, get_settings
from src.db import pool as db_pool
from src.generation.context import build_user_message
from src.generation.prompts import SYSTEM_PROMPT, load_prompt
from src.observability import observe, trace_context, update_generation
from src.providers.base import EmbeddingProvider, LLMProvider, Message, StreamResult
from src.providers.pricing import estimate_cost_usd
from src.retrieval.base import RetrievedChunk
from src.retrieval.vector import VectorRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _build_messages(payload: ChatRequest, context: Sequence[RetrievedChunk]) -> list[Message]:
    """Prior turns verbatim, then the question wrapped in its excerpts."""
    messages = [Message(role=m.role, content=m.content) for m in payload.history]
    messages.append(Message(role="user", content=build_user_message(payload.question, context)))
    return messages


@observe(name="retrieve")
async def _retrieve(
    embedder: EmbeddingProvider, settings: Settings, question: str
) -> list[RetrievedChunk]:
    """Search the corpus. A retrieval failure degrades to an unsourced answer
    rather than failing the request — the prompt makes the model say it has
    no documentation to work from."""
    try:
        async with db_pool.acquire() as conn:
            return await VectorRetriever(embedder, conn).search(
                question, top_k=settings.retrieval_top_k
            )
    except Exception:  # noqa: BLE001 - surfaced through an empty context
        logger.exception("retrieval failed; answering without context")
        return []


@observe(name="chat_completion", as_type="generation")
async def _generate(
    llm: LLMProvider,
    settings: Settings,
    payload: ChatRequest,
    context: Sequence[RetrievedChunk],
    result: StreamResult,
) -> AsyncIterator[str]:
    """Yield text deltas and record usage/cost on the Langfuse span."""
    messages = _build_messages(payload, context)

    update_generation(
        input=payload.question,
        metadata={
            "provider": llm.name,
            "history_turns": len(payload.history),
            "context_chunks": len(context),
            "prompt_version": SYSTEM_PROMPT,
        },
    )

    chunks: list[str] = []
    async for delta in llm.stream_chat(
        messages=messages,
        system=load_prompt(),
        max_tokens=settings.max_tokens,
        result=result,
    ):
        chunks.append(delta)
        yield delta

    cost = estimate_cost_usd(result.model, result.usage)
    update_generation(
        output="".join(chunks),
        model=result.model,
        usage_details={
            "input": result.usage.input_tokens,
            "output": result.usage.output_tokens,
            "cache_read_input_tokens": result.usage.cache_read_input_tokens,
            "total": result.usage.total_tokens,
        },
        cost_details=({"total": float(cost)} if cost is not None else None),
        metadata={"stop_reason": result.stop_reason},
    )


def _sse(event: str, data: dict | str) -> dict:
    return {
        "event": event,
        "data": data if isinstance(data, str) else json.dumps(data, ensure_ascii=False),
    }


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> EventSourceResponse:
    settings = get_settings()
    llm: LLMProvider = request.app.state.llm
    embedder: EmbeddingProvider = request.app.state.embedder

    async def event_stream() -> AsyncIterator[dict]:
        result = StreamResult()
        try:
            with trace_context(
                session_id=payload.session_id,
                tags=["chat", llm.name],
                trace_name="chat",
            ):
                context = await _retrieve(embedder, settings, payload.question)
                yield _sse(
                    "sources",
                    {
                        "chunks": [
                            {
                                "chunk_id": c.chunk_id,
                                "doc_id": c.doc_id,
                                "score": round(c.score, 4),
                                "section_title": c.section_title,
                            }
                            for c in context
                        ]
                    },
                )

                async for delta in _generate(llm, settings, payload, context, result):
                    if await request.is_disconnected():
                        logger.info("client disconnected, aborting stream")
                        return
                    yield _sse("token", {"text": delta})

            usage = UsagePayload(
                model=result.model,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cache_read_input_tokens=result.usage.cache_read_input_tokens,
                cache_creation_input_tokens=result.usage.cache_creation_input_tokens,
                total_tokens=result.usage.total_tokens,
                cost_usd=(
                    float(cost)
                    if (cost := estimate_cost_usd(result.model, result.usage)) is not None
                    else None
                ),
                stop_reason=result.stop_reason,
            )
            yield _sse("usage", usage.model_dump())
            yield _sse("done", {"ok": True})
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE event
            # The response has already begun, so an HTTP error code is no
            # longer available — report the failure in-band and stop.
            logger.exception("chat stream failed")
            yield _sse("error", {"message": type(exc).__name__})

    return EventSourceResponse(event_stream())
