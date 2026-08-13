from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False
    )

    env_mode: str = "production"
    app_version: str = "1.0.0"

    # Restricted runtime role — SELECT/INSERT/UPDATE/DELETE only.
    database_url: str
    # Admin/migration role — CREATE/ALTER/DROP. Used only by Alembic.
    database_admin_url: str

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "llm_wiki"

    # LLM adapter (Generator). Provider swap is config-only (D-45, Seam 3).
    llm_provider: str = "claude"  # claude / openai / llama
    llm_model_name: str = "claude-sonnet-4-6"
    # Dormant-provider model names — deliberately separate from llm_model_name so an
    # eventual LLM_PROVIDER switch never sends a Claude-format string to another API.
    llm_model_name_openai: str | None = None
    llm_model_name_llama: str | None = None
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000
    # OpenAI-compatible endpoint override; required for the llama provider.
    llm_base_url: str | None = None

    # Chunker (structure-aware splitting). Token units; wired to RunnableConfig in Step 8.
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embedding (Nomic dense + BM25 sparse, both loaded in-process).
    embedding_model_name: str = "nomic-embed-text-v1.5"
    embedding_batch_size: int = 50
    sparse_model_name: str = "Qdrant/bm25"

    # Retrieval / generation (sealed-envelope values passed via RunnableConfig).
    top_k: int = 5
    quality_threshold: float = 0.65
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    cache_ttl_hours: int = 48

    # Validator (Node 1). Wired to RunnableConfig in Step 8.
    max_file_size_mb: int = 20
    max_query_length: int = 2000

    # FastAPI (Step 12).
    rate_limit_admin: int = 30
    rate_limit_service: int = 10
    rate_limit_employee: int = 10

    # Dead-letter system (Step 14). Host folder surviving container restarts.
    dead_letter_path: str = "/data/dead-letter"

    # MCP server (Step 16). Thin wrapper — talks to FastAPI over HTTP only, never
    # touches Postgres/Qdrant/models directly (Section 4 System Mapping).
    mcp_fastapi_base_url: str = "http://app:8000"
    mcp_rate_limit_per_minute: int = 10
    mcp_query_timeout_seconds: float = 35.0
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8001


@lru_cache
def get_settings() -> Settings:
    return Settings()
