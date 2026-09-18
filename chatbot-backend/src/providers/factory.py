"""Provider selection driven by config."""

from __future__ import annotations

from src.config import Settings
from src.providers.base import EmbeddingProvider, LLMProvider


def build_llm_provider(settings: Settings) -> LLMProvider:
    match settings.llm_provider:
        case "anthropic":
            from src.providers.anthropic_provider import AnthropicLLMProvider

            return AnthropicLLMProvider(settings)
        case "openai":
            from src.providers.openai_provider import OpenAILLMProvider

            return OpenAILLMProvider(settings)
        case unknown:  # pragma: no cover - pydantic constrains the literal
            raise ValueError(f"unsupported llm_provider: {unknown}")


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    match settings.embedding_provider:
        case "openai":
            from src.providers.openai_provider import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(settings)
        case "anthropic":
            # Anthropic ships no embeddings endpoint; use OpenAI or Voyage.
            raise ValueError(
                "anthropic does not provide an embeddings API — set EMBEDDING_PROVIDER=openai"
            )
        case unknown:  # pragma: no cover
            raise ValueError(f"unsupported embedding_provider: {unknown}")
