"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    #: Prior turns, oldest first. The question is appended to these.
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)
    session_id: str | None = Field(default=None, max_length=128)


class UsagePayload(BaseModel):
    """Terminal SSE `usage` event."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_tokens: int
    cost_usd: float | None = None
    stop_reason: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["up", "down"]
    llm_provider: str
    embedding_provider: str
