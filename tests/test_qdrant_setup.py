from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import models as qm

from app.adapters.vector_store import DENSE_VECTOR_NAME, EMBEDDING_DIM, SPARSE_VECTOR_NAME
from app.domain import Chunk, SparseVector
from app.embedding import BM25Encoder
from app.qdrant_setup import ensure_collection


def _client(exists: bool) -> AsyncMock:
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=exists)
    return client


# --- collection provisioning --------------------------------------------------


async def test_creates_collection_with_correct_dense_and_sparse_config():
    client = _client(exists=False)
    created = await ensure_collection(client)

    assert created is True
    kwargs = client.create_collection.call_args.kwargs
    dense = kwargs["vectors_config"][DENSE_VECTOR_NAME]
    assert dense.size == EMBEDDING_DIM
    assert dense.distance == qm.Distance.COSINE
    sparse = kwargs["sparse_vectors_config"][SPARSE_VECTOR_NAME]
    assert sparse.modifier == qm.Modifier.IDF
    assert set(kwargs["vectors_config"]) == {"text"}
    assert set(kwargs["sparse_vectors_config"]) == {"bm25"}


async def test_idempotent_when_collection_exists():
    client = _client(exists=True)
    created = await ensure_collection(client)
    assert created is False
    client.create_collection.assert_not_called()
    client.delete_collection.assert_not_called()


async def test_recreate_drops_then_creates():
    client = _client(exists=True)
    created = await ensure_collection(client, recreate=True)
    assert created is True
    client.delete_collection.assert_called_once()
    client.create_collection.assert_called_once()


# --- BM25 sparse encoding (wrapper logic, injected model) ---------------------


class FakeSparseModel:
    def __init__(self):
        self.embed_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    def embed(self, texts, **kwargs):
        self.embed_calls.append(list(texts))
        for offset, _ in enumerate(texts):
            yield SimpleNamespace(indices=[10 + offset, 20 + offset], values=[1.5, 0.5])

    def query_embed(self, texts, **kwargs):
        self.query_calls.append(list(texts))
        yield SimpleNamespace(indices=[10], values=[1.0])


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc-1:{i}",
        chunk_index=i,
        document_id="doc-1",
        document_title="Handbook",
        source_label="hr",
        section_heading="Policy",
        chunk_text=f"body {i}",
        embed_text=f"Handbook > Policy\n\nbody {i}",
    )


def test_encode_chunks_returns_aligned_sparse_vectors_over_embed_text():
    model = FakeSparseModel()
    result = BM25Encoder(model).encode_chunks([_chunk(0), _chunk(1)])
    assert all(isinstance(v, SparseVector) for v in result)
    assert result[0].indices == [10, 20] and result[1].indices == [11, 21]
    assert all(isinstance(i, int) for i in result[0].indices)
    assert all(isinstance(v, float) for v in result[0].values)
    # Encodes the enriched embed_text, not chunk_text.
    assert model.embed_calls[0][0].startswith("Handbook > Policy")


def test_encode_query_uses_query_embed_path():
    model = FakeSparseModel()
    vector = BM25Encoder(model).encode_query("reset badge")
    assert isinstance(vector, SparseVector)
    assert vector.indices == [10] and vector.values == [1.0]
    assert model.query_calls == [["reset badge"]]
    assert model.embed_calls == []


def test_encode_documents_empty_returns_empty():
    assert BM25Encoder(FakeSparseModel()).encode_documents([]) == []


# --- real fastembed BM25 (guarded — verifies Engineer Task #2 end-to-end) -----


def test_real_bm25_produces_valid_sparse_vectors():
    try:
        from fastembed import SparseTextEmbedding  # noqa: F401

        encoder = BM25Encoder()
        doc_vectors = encoder.encode_documents(["Reset your badge at the front desk."])
        query_vector = encoder.encode_query("badge reset")
    except Exception as exc:  # noqa: BLE001 — model download/load may be unavailable
        pytest.skip(f"real fastembed BM25 model unavailable: {exc}")

    assert len(doc_vectors) == 1
    doc = doc_vectors[0]
    assert doc.indices and len(doc.indices) == len(doc.values)
    assert all(isinstance(i, int) for i in doc.indices)
    assert all(isinstance(v, float) and v > 0 for v in doc.values)
    assert query_vector.indices and len(query_vector.indices) == len(query_vector.values)
