"""Provider-agnostic interfaces and value types.

Nothing in here imports a vendor SDK — implementations live in sibling
modules and are wired up in `src/providers/factory.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for a single provider call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


@dataclass(slots=True)
class StreamResult:
    """Filled in by the provider once a stream completes."""

    model: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Streaming chat completion."""

    name: str

    def stream_chat(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        result: StreamResult | None = None,
    ) -> AsyncIterator[str]:
        """Yield response text deltas.

        If `result` is provided the provider must populate it with the final
        model id, usage and stop reason before the iterator is exhausted, so
        the caller can price the request.
        """
        ...

    async def aclose(self) -> None: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Batch text embedding."""

    name: str
    model: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], Usage]:
        """Return one vector per input text, plus token usage."""
        ...

    async def aclose(self) -> None: ...
