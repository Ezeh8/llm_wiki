"""Shared Pydantic domain models.

Includes the two PRD Section 3 state models (RetrievedChunk, Citation) and the
adapter-wall I/O shapes. These are provider-neutral on purpose: callers of the
adapters use these types and never import qdrant/sqlalchemy client types, which is
the whole point of the adapter-wall (D-7).
"""

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    chunk_text: str
    document_id: str
    document_title: str
    score: float
    chunk_index: int


class Citation(BaseModel):
    document_id: str
    document_title: str
    chunk_id: str
    chunk_text: str
    chunk_index: int


class GenerationResult(BaseModel):
    """Standardized LLM adapter output — identical shape across all providers.

    Matches the Generator node's required structured output (PRD Section 3, Node 7):
    {answer, citations:[Citation]}. The Generator node (Step 10) owns prompt wording
    and post-generation grounding verification / degraded fallback; the adapter only
    requests structured output per provider and returns this validated shape.
    """

    answer: str
    citations: list[Citation]


class SparseVector(BaseModel):
    """BM25-style sparse vector, decoupled from qdrant's own model type."""

    indices: list[int]
    values: list[float]


class Chunk(BaseModel):
    """Chunker output (Step 5) — one entry per LangGraph state `chunks: list[dict]` item.

    Carries BOTH text views: `chunk_text` is the original, unenriched text used for
    citations and the Qdrant payload; `embed_text` is the enriched text (document_title +
    section heading prepended) that Step 6 embeds. `chunk_id` = f"{document_id}:{index}"
    is deterministic (Node 12 Storer) and maps to a Qdrant point via Step 2's uuid5.
    """

    chunk_id: str
    chunk_index: int
    document_id: str
    document_title: str
    source_label: str
    section_heading: str | None = None
    chunk_text: str
    embed_text: str


class ChunkWithVectors(BaseModel):
    """One chunk ready to be stored: identity + payload + both vector kinds.

    `ingested_at` (ISO-8601 UTC string, Step 10) travels in the Qdrant payload so the
    Ranker's recency tiebreaker (D-5) can read it straight off a search hit — no
    Postgres round-trip on the hot query path, keeping Ranker "pure logic, no external
    calls" as the PRD specifies (Node 5).
    """

    chunk_id: str
    chunk_index: int
    chunk_text: str
    document_id: str
    document_title: str
    source_label: str
    ingested_at: str
    dense_vector: list[float]
    sparse_vector: SparseVector


class ScoredChunk(BaseModel):
    """A hybrid search hit carrying BOTH component scores.

    The Ranker node (Step 10) combines dense_score and sparse_score via the
    configured weighted sum (0.7/0.3, D-41) and recency tiebreaker, then emits
    RetrievedChunk. Keeping the components separate here is what makes that
    weighted sum possible — a single fused score could not be re-weighted.
    """

    chunk_id: str
    chunk_text: str
    document_id: str
    document_title: str
    chunk_index: int
    ingested_at: str | None = Field(default=None)
    dense_score: float | None = Field(default=None)
    sparse_score: float | None = Field(default=None)
