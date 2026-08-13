"""Nomic embedding integration (PRD Node 3 query embed + Node 11 Embedding Batcher).

Nomic-embed-text-v1.5 is loaded IN-PROCESS via sentence-transformers (Crash Risk #1/#4
describe it as a ~500MB-1GB in-RAM CPU model, not an API), lazily and cached so the weight
load happens once. The model is injectable for testing so the suite never downloads
weights. Nomic requires a task prefix on every input — `search_query:` for queries,
`search_document:` for stored chunks — applied here (separate from the Chunker's
title/heading enrichment already baked into `Chunk.embed_text`).

Retry (1 per PRD) and 768-dim output validation live in this layer; Step 8 wraps these
methods as graph nodes. "Backoff for rate limits" is effectively N/A for local inference
(no remote rate limit) — the retry guards transient CPU/thread hiccups; a configurable
`backoff_seconds` (default 0) is kept so an API-hosted embedding could reuse this path.
"""

import asyncio
import time
from collections import deque
from functools import lru_cache
from typing import Any, Protocol

from app.config import get_settings
from app.domain import Chunk, SparseVector

# Must match the vector store's dense dimension (app/adapters/vector_store.EMBEDDING_DIM).
EMBEDDING_DIM = 768

QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "

# How long a real embedding failure keeps the model reported as unhealthy (Step 20:
# health check must reflect real traffic, not just a synthetic probe). A timestamped
# window, not a single overwriteable flag — same pattern as RateLimitMiddleware's
# sliding window (app/api/middleware.py). A concurrent SUCCESS never clears this early;
# it only ages out by time. That matters under real concurrency: asyncio.to_thread
# offloads only the blocking encode() call, so two concurrent embed calls' completion
# order depends on thread-scheduling timing, not call-start order — a last-write-wins
# boolean could have a failing call's result silently overwritten by a concurrent
# success that happens to finish afterward. A failures-only, age-pruned window can't be
# masked that way.
_FAILURE_WINDOW_SECONDS = 60.0

# CPU-only inference on a ~500MB-1GB transformer doesn't parallelize usefully — extra
# concurrent model.encode() calls just contend for the same cores and each gets slower,
# which is exactly what compounded the Stage 2 health-check pileup incident. Capped low
# (not 1, so one real query/ingest embed and one health probe can still run together;
# not higher, since more only adds contention without adding throughput) so new calls
# queue instead of piling on more concurrently-running threads.
_MAX_CONCURRENT_ENCODES = 2


class EmbeddingError(Exception):
    """Embedding failed after retries."""


class EmbeddingDimensionError(EmbeddingError):
    """A returned vector did not match the expected count/dimension (output validation)."""


class EmbeddingModel(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


def _load_model() -> EmbeddingModel:
    # Imported lazily so importing this module does not pull torch/sentence-transformers.
    from sentence_transformers import SentenceTransformer

    name = get_settings().embedding_model_name
    repo = name if "/" in name else f"nomic-ai/{name}"
    return SentenceTransformer(repo, trust_remote_code=True)


def _as_float_lists(raw: Any) -> list[list[float]]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [[float(value) for value in row] for row in raw]


class Embedder:
    def __init__(
        self,
        model: EmbeddingModel | None = None,
        *,
        batch_size: int | None = None,
        retries: int = 1,
        backoff_seconds: float = 0.0,
        dimension: int = EMBEDDING_DIM,
    ) -> None:
        self._model = model
        self._loaded = model is not None
        self._batch_size = batch_size or get_settings().embedding_batch_size
        self._retries = retries
        self._backoff_seconds = backoff_seconds
        self._dimension = dimension
        self._recent_failures: deque[float] = deque()
        self._encode_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ENCODES)

    def _get_model(self) -> EmbeddingModel:
        if not self._loaded:
            self._model = _load_model()
            self._loaded = True
        return self._model  # type: ignore[return-value]

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into this process yet — used by the
        health check (Step 15) to decide whether a real (fast, warm) check is
        possible without triggering a cold multi-GB load on the health-check path."""
        return self._loaded

    @property
    def recently_failed(self) -> bool:
        """Whether a real embed call (query or chunk, from actual traffic) has failed
        within the last _FAILURE_WINDOW_SECONDS — read by the health check (Step 20) as
        a stronger, unmaskable-by-a-concurrent-success signal than a synthetic probe."""
        now = time.monotonic()
        while self._recent_failures and now - self._recent_failures[0] > _FAILURE_WINDOW_SECONDS:
            self._recent_failures.popleft()
        return len(self._recent_failures) > 0

    async def embed_query(self, query_text: str) -> list[float]:
        vectors = await self._embed([QUERY_PREFIX + query_text])
        return vectors[0]

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Return one 768-dim vector per chunk, aligned to input order (state `embeddings`)."""
        return await self._embed([DOCUMENT_PREFIX + chunk.embed_text for chunk in chunks])

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        results: list[list[float]] = []
        try:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                vectors = await self._encode_with_retry(model, batch)
                self._validate(vectors, len(batch))
                results.extend(vectors)
        except EmbeddingError:
            # Recorded here, not cleared on success elsewhere — see _FAILURE_WINDOW_SECONDS
            # docstring above for why a concurrent success must never erase this.
            self._recent_failures.append(time.monotonic())
            raise
        return results

    async def _encode_with_retry(
        self, model: EmbeddingModel, batch: list[str]
    ) -> list[list[float]]:
        attempts = self._retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self._encode_semaphore:
                    raw = await asyncio.to_thread(
                        model.encode, batch, normalize_embeddings=True
                    )
                return _as_float_lists(raw)
            except asyncio.CancelledError:
                # An outer asyncio.wait_for timing out (health probe's 1.5s budget, or a
                # route's with_timeout) cancels this await — but CancelledError is a
                # BaseException, not Exception, so it silently skips the `except
                # Exception` below AND `_embed`'s `except EmbeddingError`, leaving
                # recently_failed never set. Without recording it here, every later
                # health check keeps re-probing on top of whatever's still running
                # in the background (asyncio.to_thread cannot force-stop an
                # already-started OS thread — that part is unfixable, this only stops
                # NEW pileup from starting). Must re-raise unchanged, never swallowed.
                self._recent_failures.append(time.monotonic())
                raise
            except Exception as exc:  # noqa: BLE001 — normalized into a typed error below
                last_error = exc
                if attempt + 1 < attempts and self._backoff_seconds:
                    await asyncio.sleep(self._backoff_seconds * (attempt + 1))
        raise EmbeddingError(
            f"embedding failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    def _validate(self, vectors: list[list[float]], expected_count: int) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingDimensionError(
                f"expected {expected_count} vectors, got {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingDimensionError(
                    f"expected {self._dimension}-dim vector, got {len(vector)}"
                )


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide singleton; the real Nomic model loads lazily on first embed call."""
    return Embedder()


class SparseModel(Protocol):
    def embed(self, texts: list[str], **kwargs: Any) -> Any: ...
    def query_embed(self, texts: list[str], **kwargs: Any) -> Any: ...


def _load_sparse_model() -> SparseModel:
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(get_settings().sparse_model_name)


def _to_sparse(embedding: Any) -> SparseVector:
    return SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )


class BM25Encoder:
    """BM25 sparse-vector encoder (Qdrant native sparse vectors, D-41).

    fastembed's `Qdrant/bm25` produces term-frequency values for documents and unit
    values for query terms; the IDF component is applied server-side by the collection's
    `Modifier.IDF` (see qdrant_setup). Document and query use DIFFERENT calls — `embed`
    vs `query_embed` — so the query side is weighted correctly.
    """

    def __init__(self, model: SparseModel | None = None) -> None:
        self._model = model
        self._loaded = model is not None

    def _get_model(self) -> SparseModel:
        if not self._loaded:
            self._model = _load_sparse_model()
            self._loaded = True
        return self._model  # type: ignore[return-value]

    def encode_chunks(self, chunks: list[Chunk]) -> list[SparseVector]:
        """One sparse vector per chunk (aligned to input order), over `chunk.embed_text`."""
        return self.encode_documents([chunk.embed_text for chunk in chunks])

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        return [_to_sparse(e) for e in self._get_model().embed(texts)]

    def encode_query(self, query_text: str) -> SparseVector:
        return _to_sparse(next(iter(self._get_model().query_embed([query_text]))))


@lru_cache(maxsize=1)
def get_bm25_encoder() -> BM25Encoder:
    """Process-wide singleton; the fastembed BM25 model loads lazily on first encode."""
    return BM25Encoder()
