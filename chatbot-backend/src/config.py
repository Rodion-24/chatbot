"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- app ---
    app_name: str = "docs-rag"
    environment: Literal["local", "staging", "prod"] = "local"
    log_level: str = "INFO"
    # Browser origins allowed to call the API. The Vite dev server is 5173;
    # add the deployed frontend origin here before going live.
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- database ---
    database_url: PostgresDsn = Field(
        default="postgresql://postgres:postgres@localhost:5433/docs_rag",
    )
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_command_timeout_s: float = 30.0

    # --- provider selection ---
    llm_provider: ProviderName = "anthropic"
    embedding_provider: ProviderName = "openai"

    # --- anthropic ---
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"

    # --- openai ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- generation ---
    # The system prompt lives in src/generation/prompts/ so it can be
    # versioned and diffed; only its selection belongs in config.
    max_tokens: int = 4096

    # --- retrieval ---
    retrieval_top_k: int = 5
    # "hybrid" fuses vector + full-text with RRF; "vector" is the stage-2
    # dense-only baseline, kept so the two can be compared.
    retriever: Literal["hybrid", "vector"] = "hybrid"

    # --- reranking ---
    # Off by default: it adds a model call to every query, so it has to earn
    # its place on the eval set before being turned on.
    rerank_enabled: bool = False
    # Candidates pulled from the retriever before re-ranking down to top_k.
    rerank_candidates: int = 20
    # A cheap model is enough to order a shortlist, and keeping it distinct
    # from the answering model avoids one model grading its own shortlist.
    rerank_model: str = "claude-haiku-4-5"

    # --- embeddings ---
    # Must match the vector(N) column width in migrations/0001_init.sql.
    embedding_dim: int = 1536

    # --- retries ---
    retry_max_attempts: int = 4
    retry_initial_backoff_s: float = 0.5
    retry_max_backoff_s: float = 8.0

    # --- langfuse ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    @property
    def asyncpg_dsn(self) -> str:
        """asyncpg wants a plain libpq URI, not the SQLAlchemy-style scheme."""
        return str(self.database_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
