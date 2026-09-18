"""OpenAI implementations of LLMProvider and EmbeddingProvider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from openai import AsyncOpenAI

from src.config import Settings
from src.providers.base import EmbeddingProvider, LLMProvider, Message, StreamResult, Usage
from src.providers.retry import with_retry


def _client(settings: Settings) -> AsyncOpenAI:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for the openai provider")
    return AsyncOpenAI(api_key=settings.openai_api_key, max_retries=2)


class OpenAILLMProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.openai_model
        self._client = _client(settings)

    async def stream_chat(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        result: StreamResult | None = None,
    ) -> AsyncIterator[str]:
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)

        async def _open():
            return await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=payload,
                stream=True,
                # Usage is omitted from streamed responses unless requested.
                stream_options={"include_usage": True},
            )

        stream = await with_retry(
            _open, settings=self._settings, op="openai.chat.completions.create"
        )

        model = self._model
        stop_reason: str | None = None
        usage = Usage()

        async for chunk in stream:
            if chunk.model:
                model = chunk.model
            if chunk.usage is not None:
                cached = 0
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                if details is not None:
                    cached = getattr(details, "cached_tokens", 0) or 0
                usage = Usage(
                    input_tokens=(chunk.usage.prompt_tokens or 0) - cached,
                    output_tokens=chunk.usage.completion_tokens or 0,
                    cache_read_input_tokens=cached,
                )
            # The usage-only chunk carries no choices.
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                stop_reason = choice.finish_reason
            if choice.delta and choice.delta.content:
                yield choice.delta.content

        if result is not None:
            result.model = model
            result.stop_reason = stop_reason
            result.usage = usage

    async def aclose(self) -> None:
        await self._client.close()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model = settings.openai_embedding_model
        self.dim = settings.embedding_dim
        self._client = _client(settings)

    async def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], Usage]:
        if not texts:
            return [], Usage()

        async def _call():
            return await self._client.embeddings.create(
                model=self.model,
                input=list(texts),
                dimensions=self.dim,
            )

        response = await with_retry(_call, settings=self._settings, op="openai.embeddings.create")

        # The API may return items out of order; sort by index before use.
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        usage = Usage(input_tokens=response.usage.prompt_tokens or 0)
        return vectors, usage

    async def aclose(self) -> None:
        await self._client.close()
