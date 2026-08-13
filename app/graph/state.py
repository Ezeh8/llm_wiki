"""LangGraph state schema (PRD Section 3, 19 fields — D-11, locked) and RunnableConfig
(the sealed envelope — D-12, locked: secrets/config live ONLY in `config["configurable"]`,
never on this TypedDict).

All fields overwrite (LangGraph's default per-key merge for a plain TypedDict — no
reducers/Annotated accumulation, per "All fields overwrite"). None values are routing
signals. `document_file` is a base64 string, cleared by the Chunker node once extraction
is done (#46). Timestamps are ISO-8601 UTC strings. State values are kept as plain
dicts/lists/primitives (not Pydantic instances) so they round-trip cleanly through the
Postgres checkpointer — nodes validate into Pydantic models internally and `.model_dump()`
before returning into state.

Two judgment calls, needed because the schema is locked at exactly 19 fields with no
room to add new ones (D-11):
  - `document_metadata` (typed as a bare `dict`) is used as the general per-operation
    scratch space, not only the ingest example shape {title, source_label,
    changelog_id}. Duplicate Checker stashes `content_hash`/`file_size_bytes` there;
    the Chunker stashes `chunk_count`; a query's optional `filter` (source_label/
    document_id) also lives there as `document_metadata["filter"]`, and FastAPI
    (Step 12) is expected to set `document_metadata["file_type"]` from the uploaded
    filename before invoking the graph, since Validator/Chunker need it and no other
    field carries it.
  - `retrieved_chunks` is typed `list[RetrievedChunk]` in the PRD table, but between
    Retriever and Ranker it actually holds `list[ScoredChunk]`-shaped dicts — carrying
    the separate dense_score/sparse_score the Ranker needs for the 0.7/0.3 weighted sum
    (D-41). The Ranker reduces ScoredChunk -> RetrievedChunk, so `ranked_chunks` is a
    true `list[RetrievedChunk]`-shaped list as documented. See the Step 2 docstring in
    app/adapters/vector_store.py and app/graph/nodes/query_path.py.
"""

from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig as _RunnableConfig

from app.config import Settings, get_settings


class GraphState(TypedDict, total=False):
    operation_type: str  # "query" / "ingest" / "delete" / "update"
    thread_id: str
    session_id: str
    query_text: str
    document_file: str | None
    document_metadata: dict[str, Any]
    document_id: str
    actor: str
    chunks: list[dict]
    embeddings: list[list[float]]
    retrieved_chunks: list[dict]
    ranked_chunks: list[dict]
    answer: str
    citations: list[dict]
    status: str
    error: str | None
    cache_hit: bool
    cache_key: str
    query_embedding: list[float]


STATE_FIELDS: tuple[str, ...] = tuple(GraphState.__annotations__.keys())


class Configurable(TypedDict, total=False):
    """The RunnableConfig `configurable` dict — the sealed envelope (D-12). Built once
    at the FastAPI boundary (Step 12) via `build_run_config`; nodes read config/secrets
    from here exclusively, never from `GraphState`."""

    thread_id: str  # required by the checkpointer to key the run's thread
    llm_api_key: str
    database_url: str
    tenant_id: str | None  # reserved, v2 multi-tenancy (D-29)
    env_mode: str
    llm_provider: str
    llm_model_name: str
    llm_temperature: float
    llm_max_tokens: int
    file_storage_path: str | None
    qdrant_collection: str
    chunk_size: int
    chunk_overlap: int
    embedding_model_name: str
    cache_ttl_hours: int
    hybrid_dense_weight: float
    hybrid_sparse_weight: float
    quality_threshold: float
    top_k: int
    max_file_size_mb: int
    max_query_length: int


def build_run_config(
    *,
    thread_id: str,
    session_id: str,
    operation_type: str,
    actor: str,
    settings: Settings | None = None,
) -> _RunnableConfig:
    """Build the sealed-envelope RunnableConfig once at the FastAPI boundary (Step 12).

    Tags/metadata (thread_id, session_id, operation_type, actor, env_mode, app_version)
    are built here too, per PRD Section 3 "RunnableConfig Tags/Metadata".
    """
    settings = settings or get_settings()
    configurable: Configurable = {
        "thread_id": thread_id,
        "llm_api_key": settings.llm_api_key,
        "database_url": settings.database_url,
        "tenant_id": None,
        "env_mode": settings.env_mode,
        "llm_provider": settings.llm_provider,
        "llm_model_name": settings.llm_model_name,
        "llm_temperature": settings.llm_temperature,
        "llm_max_tokens": settings.llm_max_tokens,
        "file_storage_path": None,
        "qdrant_collection": settings.qdrant_collection,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model_name": settings.embedding_model_name,
        "cache_ttl_hours": settings.cache_ttl_hours,
        "hybrid_dense_weight": settings.hybrid_dense_weight,
        "hybrid_sparse_weight": settings.hybrid_sparse_weight,
        "quality_threshold": settings.quality_threshold,
        "top_k": settings.top_k,
        "max_file_size_mb": settings.max_file_size_mb,
        "max_query_length": settings.max_query_length,
    }
    return _RunnableConfig(
        configurable=configurable,
        tags=[operation_type, settings.env_mode],
        metadata={
            "thread_id": thread_id,
            "session_id": session_id,
            "operation_type": operation_type,
            "actor": actor,
            "env_mode": settings.env_mode,
            "app_version": settings.app_version,
        },
    )
