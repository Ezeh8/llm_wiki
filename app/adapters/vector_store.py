"""VectorStore adapter — the only module that imports the Qdrant client.

Committed point/vector schema (Step 7 collection setup MUST match this):
  - Named vectors on every point:
      * dense  "text" — 768 dims (Nomic-embed-text-v1.5), Cosine distance
      * sparse "bm25" — BM25 sparse vector
  - Point id: a deterministic UUIDv5 derived from the human chunk_id
    ("{document_id}:{chunk_index}"). Qdrant point ids must be uint or UUID, so the
    string chunk_id cannot be the id directly; it is kept verbatim in the payload.
  - Payload keys: chunk_id, document_id, document_title, source_label, chunk_index,
    chunk_text, ingested_at (Step 10 — lets the Ranker's recency tiebreaker read a
    document's age straight off a search hit, no Postgres round-trip on the query path).

The adapter never retries (nodes own retries) and validates every output before
returning it (D-7).
"""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from app.adapters.errors import AdapterValidationError
from app.config import get_settings
from app.domain import ChunkWithVectors, ScoredChunk, SparseVector

DENSE_VECTOR_NAME = "text"
SPARSE_VECTOR_NAME = "bm25"
EMBEDDING_DIM = 768

# Stable namespace so a given chunk_id always maps to the same Qdrant point id.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

_REQUIRED_PAYLOAD = ("chunk_id", "document_id", "document_title", "chunk_index", "chunk_text")
# ingested_at is required on write (Step 9's Storer always supplies it) but read
# leniently (older/hand-built points in tests may omit it) — see _to_scored_chunk.


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class VectorStoreAdapter:
    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name

    async def store_chunks(
        self, document_id: str, chunks_with_vectors: list[ChunkWithVectors]
    ) -> int:
        points: list[qm.PointStruct] = []
        for chunk in chunks_with_vectors:
            if chunk.document_id != document_id:
                raise AdapterValidationError(
                    f"chunk {chunk.chunk_id} document_id mismatch for {document_id}"
                )
            if len(chunk.dense_vector) != EMBEDDING_DIM:
                raise AdapterValidationError(
                    f"chunk {chunk.chunk_id} dense vector has {len(chunk.dense_vector)} "
                    f"dims, expected {EMBEDDING_DIM}"
                )
            points.append(
                qm.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR_NAME: chunk.dense_vector,
                        SPARSE_VECTOR_NAME: qm.SparseVector(
                            indices=chunk.sparse_vector.indices,
                            values=chunk.sparse_vector.values,
                        ),
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_title": chunk.document_title,
                        "source_label": chunk.source_label,
                        "chunk_index": chunk.chunk_index,
                        "chunk_text": chunk.chunk_text,
                        "ingested_at": chunk.ingested_at,
                    },
                )
            )

        result = await self._client.upsert(collection_name=self._collection, points=points)
        if result.status != qm.UpdateStatus.COMPLETED:
            raise AdapterValidationError(f"upsert not completed: status={result.status}")
        return len(points)

    async def delete_by_document_id(self, document_id: str) -> None:
        # Naturally idempotent: deleting an absent document_id is a no-op.
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=_document_filter({"document_id": document_id})
            ),
        )

    async def search(
        self,
        query_vector: list[float],
        sparse_vector: SparseVector,
        top_k: int,
        filters: dict | None = None,
    ) -> list[ScoredChunk]:
        if len(query_vector) != EMBEDDING_DIM:
            raise AdapterValidationError(
                f"query vector has {len(query_vector)} dims, expected {EMBEDDING_DIM}"
            )

        query_filter = _document_filter(filters)

        dense_response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        sparse_response = await self._client.query_points(
            collection_name=self._collection,
            query=qm.SparseVector(
                indices=sparse_vector.indices, values=sparse_vector.values
            ),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        merged: dict[str, ScoredChunk] = {}
        for point in dense_response.points:
            chunk = self._to_scored_chunk(point)
            chunk.dense_score = float(point.score)
            merged[chunk.chunk_id] = chunk
        for point in sparse_response.points:
            existing = merged.get(self._payload(point)["chunk_id"])
            if existing is not None:
                existing.sparse_score = float(point.score)
            else:
                chunk = self._to_scored_chunk(point)
                chunk.sparse_score = float(point.score)
                merged[chunk.chunk_id] = chunk

        return list(merged.values())

    @staticmethod
    def _payload(point) -> dict:
        payload = point.payload or {}
        missing = [k for k in _REQUIRED_PAYLOAD if k not in payload]
        if missing:
            raise AdapterValidationError(f"hit {point.id} missing payload fields: {missing}")
        return payload

    def _to_scored_chunk(self, point) -> ScoredChunk:
        payload = self._payload(point)
        return ScoredChunk(
            chunk_id=str(payload["chunk_id"]),
            chunk_text=str(payload["chunk_text"]),
            document_id=str(payload["document_id"]),
            document_title=str(payload["document_title"]),
            chunk_index=int(payload["chunk_index"]),
            ingested_at=payload.get("ingested_at"),
        )


def _document_filter(filters: dict | None) -> qm.Filter | None:
    if not filters:
        return None
    conditions: list[qm.FieldCondition] = []
    for key in ("source_label", "document_id"):
        value = filters.get(key)
        if value is not None:
            conditions.append(
                qm.FieldCondition(key=key, match=qm.MatchValue(value=value))
            )
    return qm.Filter(must=conditions) if conditions else None


def build_vector_store() -> VectorStoreAdapter:
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    return VectorStoreAdapter(client=client, collection_name=settings.qdrant_collection)
