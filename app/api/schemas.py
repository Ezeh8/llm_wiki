"""Request/response Pydantic models — all shapes locked by PRD Section 5 (API-2).

Reuses app.domain's `Citation`/`RetrievedChunk` directly for the query response's
`citations`/`source_chunks`, since those already match the PRD's payload shapes
exactly (D-11's Pydantic Models section) — no reason to duplicate them here.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain import Citation, RetrievedChunk

# --- query -----------------------------------------------------------------


class QueryFilter(BaseModel):
    source_label: str | None = None
    document_id: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    filter: QueryFilter | None = None
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    source_chunks: list[RetrievedChunk] | None
    cached: bool
    query_id: str
    degraded: bool
    session_id: str


# --- documents ---------------------------------------------------------------


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    source_label: str
    chunk_count: int
    ingested_at: str


class DocumentListItem(BaseModel):
    document_id: str
    title: str
    source_label: str
    ingested_at: str


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]
    next_cursor: str | None


class DocumentDetail(BaseModel):
    document_id: str
    title: str
    source_label: str
    chunk_count: int
    ingested_at: str
    file_type: str
    file_size_bytes: int


class DeleteDocumentResponse(BaseModel):
    deleted: bool = True
    document_id: str


# --- changelog -----------------------------------------------------------------


class ChangelogCreateRequest(BaseModel):
    entry: str = Field(..., min_length=1, max_length=1000)
    document_id: str | None = None


class ChangelogUpdateRequest(BaseModel):
    entry: str | None = Field(None, min_length=1, max_length=1000)
    document_id: str | None = None


class ChangelogResponse(BaseModel):
    changelog_id: str
    entry: str
    document_id: str | None
    actor: str
    created_at: datetime


class ChangelogDetail(ChangelogResponse):
    updated_at: datetime


class ChangelogListResponse(BaseModel):
    items: list[ChangelogResponse]
    next_cursor: str | None


class DeleteChangelogResponse(BaseModel):
    deleted: bool = True
    changelog_id: str


# --- audit -----------------------------------------------------------------


class AuditListItem(BaseModel):
    event_id: str
    event_type: str
    document_id: str | None
    actor: str
    timestamp: datetime
    status: str


class AuditListResponse(BaseModel):
    items: list[AuditListItem]
    next_cursor: str | None


class AuditDetail(BaseModel):
    event_id: str
    event_type: str
    document_id: str | None
    query_text: str | None
    chunks_retrieved: int | None
    answer_text: str | None
    actor: str
    timestamp: datetime
    status: str
    error_detail: str | None


# --- health -----------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded"
    audit_backlog: bool
    audit_poisoned: bool
    cache_stale_risk: bool
    qdrant_connected: bool
    postgres_connected: bool
    embedding_model_available: bool
