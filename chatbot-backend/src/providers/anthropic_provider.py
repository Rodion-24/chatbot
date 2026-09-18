"""Anthropic implementation of LLMProvider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from anthropic import AsyncAnthropic

from src.config import Settings
from src.providers.base import LLMProvider, Message, StreamResult, Usage
from src.providers.retry import with_retry


class AnthropicLLMProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        self._settings = settings
        self._model = settings.anthropic_model
        # The SDK retries 429/5xx/connection errors itself; with_retry wraps
        # the call as a whole on top of that.
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            max_retries=2,
        )

    async def stream_chat(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        result: StreamResult | None = None,
    ) -> AsyncIterator[str]:
        payload = [{"role": m.role, "content": m.content} for m in messages]

        async def _open():
            return self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=system or "",
                messages=payload,
            )

        manager = await with_retry(_open, settings=self._settings, op="anthropic.messages.stream")

        async with manager as stream:
            # anthropic>=1.x drops `.text_stream`; iterate events instead.
            async for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield event.delta.text

            final = await stream.get_final_message()

        if result is not None:
            u = final.usage
            result.model = final.model
            result.stop_reason = final.stop_reason
            result.usage = Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            )

    async def aclose(self) -> None:
        await self._client.close()
