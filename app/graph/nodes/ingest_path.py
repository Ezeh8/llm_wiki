"""Ingest/update path nodes (PRD Section 3, Nodes 9-12).

All REAL as of Step 9: duplicate_checker, chunker_node, embedding_batcher (Steps
2/4/5/6/7), and storer (Step 9 — delete-existing-first clean slate #21, Postgres
upsert + Qdrant write with rollback #24, 1 retry, cache-flush side effect). Step 11
resolves chunker_node's old/new document_id split for the update path (see its
docstring) — Deleter (app/graph/nodes/delete_path.py) is the other half.
"""

import base64
import hashlib
import uuid
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from app.adapters.postgres import PostgresAdapter
from app.adapters.vector_store import build_vector_store
from app.chunker import chunk_document
from app.db.session import async_session_factory
from app.domain import Chunk, ChunkWithVectors
from app.embedding import EmbeddingError, get_bm25_encoder, get_embedder
from app.graph.nodes._common import call_with_retry, flush_cache_side_effect
from app.graph.state import GraphState
from app.parsers import parse_document
from app.parsers.base import FileParseError


async def duplicate_checker(state: GraphState, config: RunnableConfig) -> dict:
    """Node 9 — content-hash duplicate check. 1 retry on the DB call.

    Computes content_hash/file_size_bytes from document_file here (no other node owns
    this) and stashes them into document_metadata — a bare `dict` in the state schema
    (D-11), so this is additive, not a new top-level field. The uniqueness contract is
    enforced by the `documents.content_hash` unique constraint (Step 1); matching hash
    is treated as duplicate regardless of title.
    """
    raw = base64.b64decode(state["document_file"])
    content_hash = hashlib.sha256(raw).hexdigest()
    metadata = dict(state.get("document_metadata") or {})
    metadata["content_hash"] = content_hash
    metadata["file_size_bytes"] = len(raw)

    async def _lookup():
        async with async_session_factory() as session:
            return await PostgresAdapter(session).get_document_by_content_hash(content_hash)

    existing = await call_with_retry(_lookup, retries=1)
    if existing is not None:
        return {"document_metadata": metadata, "status": "error", "error": "Duplicate document"}
    return {"document_metadata": metadata, "status": "success", "error": None}


async def chunker_node(state: GraphState, config: RunnableConfig) -> dict:
    """Node 10 — parse + structure-aware chunk. Sets document_file=None (state cleanup,
    #46) whether parsing succeeds or fails, so a failed ingest never checkpoints the
    base64 blob further.

    document_id (Step 9 ingest / Step 11 update, resolved here):
    - Plain ingest: no document_id exists yet, so a fresh one is generated (needed
      before chunking, since chunk_id = document_id + chunk_index, Node 12).
    - Update: FastAPI's PUT /documents/{id} (Step 12) seeds `document_id` with the OLD
      document being replaced. The fortified update order (D-2 — "ingest new FIRST,
      verify success, THEN delete old; duplicate temporarily better than a hole")
      requires the new content to land as a genuinely SEPARATE row, not overwrite the
      old one in place — otherwise there's no "old" left to roll back to if the new
      write fails partway. So the OLD id is stashed as
      `document_metadata["previous_document_id"]` (the same "reuse document_metadata
      as scratch space" pattern as content_hash/chunk_count) and a FRESH id is
      generated for the new document. Deleter (Step 11) reads `previous_document_id`
      to know what to clean up after Storer confirms the new write succeeded.

    content_hash/file_size_bytes (Step 11 fix): computed here from `raw`, always —
    NOT read from document_metadata. Originally only Duplicate Checker computed these,
    which works for ingest (Duplicate Checker always runs first) but silently broke
    every update: the update path's routing skips Duplicate Checker entirely (updates
    aren't duplicate-checked — replacing a document on purpose isn't a duplicate), so
    Storer's `metadata["content_hash"]` lookup raised a bare KeyError on every update,
    caught by Storer's own try/except and misreported as a generic store failure. Found
    by actually running the update path end-to-end, not by any unit test. Recomputing
    here removes the implicit "Duplicate Checker must run first" coupling entirely —
    idempotent (same bytes -> same hash) and cheap enough to not bother skipping for
    the ingest path just because Duplicate Checker already computed it once.
    """
    metadata = dict(state.get("document_metadata") or {})
    configurable = config["configurable"]
    incoming_id = state.get("document_id")
    if state.get("operation_type") == "update" and incoming_id:
        metadata["previous_document_id"] = incoming_id
        document_id = str(uuid.uuid4())
    else:
        document_id = incoming_id or str(uuid.uuid4())
    raw = base64.b64decode(state["document_file"])
    metadata["content_hash"] = hashlib.sha256(raw).hexdigest()
    metadata["file_size_bytes"] = len(raw)

    try:
        parsed = parse_document(metadata["file_type"], raw)
    except FileParseError as exc:
        return {"status": "error", "error": str(exc), "document_file": None}

    chunks = chunk_document(
        parsed,
        document_id=document_id,
        document_title=metadata["title"],
        source_label=metadata["source_label"],
        chunk_size=configurable.get("chunk_size", 500),
        chunk_overlap=configurable.get("chunk_overlap", 50),
    )
    metadata["chunk_count"] = len(chunks)

    return {
        "document_id": document_id,
        "document_metadata": metadata,
        "chunks": [c.model_dump() for c in chunks],
        "document_file": None,
        "status": "success",
        "error": None,
    }


async def embedding_batcher(state: GraphState, config: RunnableConfig) -> dict:
    """Node 11 — batches ~50 chunks through Nomic (dense only; PRD Node 11 scope).

    BM25 sparse vectors for storage are computed inside Storer (Step 9), right before
    building ChunkWithVectors — Storer is the only node that needs both vector kinds
    together, so there's no reason to carry sparse vectors through state in between.
    Retry/batching/768-dim validation all live inside Embedder (Step 6).
    """
    chunks = [Chunk.model_validate(c) for c in state.get("chunks", [])]
    try:
        vectors = await get_embedder().embed_chunks(chunks)
    except EmbeddingError as exc:
        # Same Step 20 fix as embedder_query — route_after_embedding_batcher sends this
        # straight to Audit Writer instead of Storer, which unconditionally reads
        # embeddings.
        return {"status": "error", "error": str(exc)}
    return {"embeddings": vectors}


async def storer(state: GraphState, config: RunnableConfig) -> dict:
    """Node 12 — REAL. Order matters (D-21/#24, locked):
      1. delete existing Qdrant chunks for document_id (clean slate — prevents
         duplicate chunks if this node is retried or re-run)
      2. upsert Postgres document metadata
      3. write Qdrant chunks (dense+sparse); if this fails AFTER step 2 succeeded,
         roll back by deleting the Postgres row just written — no ghost documents.
    The whole 3-step sequence gets 1 retry (each step is individually idempotent, so
    retrying from the top is safe). Side effect: flush cache, independent of whether
    the ingest itself succeeds or fails.
    """
    document_id = state["document_id"]
    metadata = state.get("document_metadata") or {}
    chunks = [Chunk.model_validate(c) for c in state.get("chunks", [])]
    dense_vectors = state.get("embeddings") or []
    if len(dense_vectors) != len(chunks):
        return {"status": "error", "error": "embeddings/chunks length mismatch"}

    # Generated once (not per retry attempt) — passed identically to the Postgres row
    # and every chunk's Qdrant payload, so the Ranker's recency read (Step 10) always
    # agrees with documents.ingested_at.
    ingested_at = datetime.now(timezone.utc)
    ingested_at_iso = ingested_at.isoformat()

    sparse_vectors = get_bm25_encoder().encode_chunks(chunks)
    chunks_with_vectors = [
        ChunkWithVectors(
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            chunk_text=chunk.chunk_text,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            source_label=chunk.source_label,
            ingested_at=ingested_at_iso,
            dense_vector=dense,
            sparse_vector=sparse,
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
    ]
    store = build_vector_store()

    async def _write_attempt() -> None:
        await store.delete_by_document_id(document_id)  # step 1: clean slate

        async with async_session_factory() as session:
            adapter = PostgresAdapter(session)
            await adapter.upsert_document(  # step 2: Postgres before Qdrant
                document_id=document_id,
                title=metadata["title"],
                source_label=metadata["source_label"],
                file_type=metadata["file_type"],
                file_size_bytes=metadata["file_size_bytes"],
                content_hash=metadata["content_hash"],
                chunk_count=len(chunks),
                ingested_at=ingested_at,
            )
            try:
                await store.store_chunks(document_id, chunks_with_vectors)  # step 3
            except Exception:
                await adapter.delete_document(document_id)  # rollback: no ghost doc
                await session.commit()
                raise
            await session.commit()

    try:
        await call_with_retry(_write_attempt, retries=1)
        result = {"status": "success", "error": None}
    except Exception as exc:
        result = {"status": "error", "error": f"failed to store document: {exc}"}

    await flush_cache_side_effect()
    return result
