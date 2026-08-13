from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import models as qm

from app.adapters.errors import AdapterValidationError
from app.adapters.vector_store import (
    DENSE_VECTOR_NAME,
    EMBEDDING_DIM,
    SPARSE_VECTOR_NAME,
    VectorStoreAdapter,
    _point_id,
)
from app.domain import ChunkWithVectors, SparseVector


def _chunk(document_id="doc-1", index=0, dims=EMBEDDING_DIM):
    return ChunkWithVectors(
        chunk_id=f"{document_id}:{index}",
        chunk_index=index,
        chunk_text="hello world",
        document_id=document_id,
        document_title="Handbook",
        source_label="hr",
        ingested_at="2026-01-01T00:00:00+00:00",
        dense_vector=[0.1] * dims,
        sparse_vector=SparseVector(indices=[1, 5], values=[0.9, 0.4]),
    )


def _hit(chunk_id, score, index=0):
    return SimpleNamespace(
        id=_point_id(chunk_id),
        score=score,
        payload={
            "chunk_id": chunk_id,
            "document_id": "doc-1",
            "document_title": "Handbook",
            "source_label": "hr",
            "chunk_index": index,
            "chunk_text": "hello world",
        },
    )


def _adapter(client):
    return VectorStoreAdapter(client=client, collection_name="llm_wiki")


async def test_store_chunks_builds_named_and_sparse_vectors():
    client = AsyncMock()
    client.upsert.return_value = SimpleNamespace(status=qm.UpdateStatus.COMPLETED)
    adapter = _adapter(client)

    count = await adapter.store_chunks("doc-1", [_chunk(index=0), _chunk(index=1)])

    assert count == 2
    points = client.upsert.call_args.kwargs["points"]
    assert points[0].id == _point_id("doc-1:0")
    assert len(points[0].vector[DENSE_VECTOR_NAME]) == EMBEDDING_DIM
    assert isinstance(points[0].vector[SPARSE_VECTOR_NAME], qm.SparseVector)
    assert points[0].payload["chunk_id"] == "doc-1:0"


async def test_store_chunks_rejects_wrong_dimension():
    client = AsyncMock()
    adapter = _adapter(client)
    with pytest.raises(AdapterValidationError):
        await adapter.store_chunks("doc-1", [_chunk(dims=10)])
    client.upsert.assert_not_awaited()


async def test_store_chunks_raises_when_not_completed():
    client = AsyncMock()
    client.upsert.return_value = SimpleNamespace(status=qm.UpdateStatus.ACKNOWLEDGED)
    with pytest.raises(AdapterValidationError):
        await _adapter(client).store_chunks("doc-1", [_chunk()])


async def test_search_merges_dense_and_sparse_scores():
    client = AsyncMock()
    client.query_points.side_effect = [
        SimpleNamespace(points=[_hit("doc-1:0", 0.9), _hit("doc-1:1", 0.7, index=1)]),
        SimpleNamespace(points=[_hit("doc-1:0", 0.4)]),
    ]
    results = await _adapter(client).search(
        query_vector=[0.2] * EMBEDDING_DIM,
        sparse_vector=SparseVector(indices=[1], values=[0.5]),
        top_k=5,
    )
    by_id = {c.chunk_id: c for c in results}
    assert by_id["doc-1:0"].dense_score == 0.9
    assert by_id["doc-1:0"].sparse_score == 0.4
    assert by_id["doc-1:1"].dense_score == 0.7
    assert by_id["doc-1:1"].sparse_score is None


async def test_search_rejects_wrong_query_dimension():
    with pytest.raises(AdapterValidationError):
        await _adapter(AsyncMock()).search(
            query_vector=[0.1] * 10,
            sparse_vector=SparseVector(indices=[], values=[]),
            top_k=5,
        )


async def test_search_builds_filter_from_source_label_and_document_id():
    client = AsyncMock()
    client.query_points.side_effect = [
        SimpleNamespace(points=[]),
        SimpleNamespace(points=[]),
    ]
    await _adapter(client).search(
        query_vector=[0.2] * EMBEDDING_DIM,
        sparse_vector=SparseVector(indices=[1], values=[0.5]),
        top_k=5,
        filters={"source_label": "hr", "document_id": "doc-1"},
    )
    query_filter = client.query_points.call_args_list[0].kwargs["query_filter"]
    keys = {c.key for c in query_filter.must}
    assert keys == {"source_label", "document_id"}


async def test_search_raises_on_missing_payload_fields():
    client = AsyncMock()
    bad = SimpleNamespace(id="x", score=0.5, payload={"chunk_id": "doc-1:0"})
    client.query_points.side_effect = [
        SimpleNamespace(points=[bad]),
        SimpleNamespace(points=[]),
    ]
    with pytest.raises(AdapterValidationError):
        await _adapter(client).search(
            query_vector=[0.2] * EMBEDDING_DIM,
            sparse_vector=SparseVector(indices=[1], values=[0.5]),
            top_k=5,
        )


async def test_delete_uses_document_id_filter():
    client = AsyncMock()
    await _adapter(client).delete_by_document_id("doc-1")
    selector = client.delete.call_args.kwargs["points_selector"]
    assert isinstance(selector, qm.FilterSelector)
    assert selector.filter.must[0].key == "document_id"
