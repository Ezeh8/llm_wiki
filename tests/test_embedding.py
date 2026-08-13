import pytest

from app.domain import Chunk
from app.embedding import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIM,
    QUERY_PREFIX,
    Embedder,
    EmbeddingDimensionError,
    EmbeddingError,
    get_embedder,
)


class FakeModel:
    """Injectable stand-in for the sentence-transformers model — no weights downloaded."""

    def __init__(self, *, dim: int = EMBEDDING_DIM, fail_times: int = 0, short: bool = False):
        self.dim = dim
        self.fail_times = fail_times
        self.short = short
        self.calls: list[list[str]] = []
        self.attempts = 0

    def encode(self, texts, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError("transient encode failure")
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            # First component encodes the trailing integer so order is verifiable.
            marker = float(int(text.strip().split()[-1]))
            rows.append([marker] + [0.0] * (self.dim - 1))
        if self.short:
            return rows[:-1]  # deliberately drop one to trigger count validation
        return rows


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc-1:{i}",
        chunk_index=i,
        document_id="doc-1",
        document_title="Handbook",
        source_label="hr",
        section_heading=None,
        chunk_text=f"body {i}",
        embed_text=f"body {i}",
    )


async def test_embed_query_applies_prefix_and_returns_768():
    model = FakeModel()
    vector = await Embedder(model, batch_size=50).embed_query("how to reset badge 7")
    assert len(vector) == EMBEDDING_DIM
    assert model.calls[0][0].startswith(QUERY_PREFIX)


async def test_embed_chunks_prefixes_and_preserves_order_across_batches():
    model = FakeModel()
    chunks = [_chunk(i) for i in range(120)]
    vectors = await Embedder(model, batch_size=50).embed_chunks(chunks)

    assert len(vectors) == 120
    assert all(len(v) == EMBEDDING_DIM for v in vectors)
    # Order preserved: the encoded marker equals the chunk index.
    assert [int(v[0]) for v in vectors] == list(range(120))
    assert model.calls[0][0].startswith(DOCUMENT_PREFIX)


async def test_batching_splits_into_batch_sized_calls():
    model = FakeModel()
    chunks = [_chunk(i) for i in range(120)]
    await Embedder(model, batch_size=50).embed_chunks(chunks)
    assert [len(call) for call in model.calls] == [50, 50, 20]


async def test_empty_input_returns_empty_without_calling_model():
    model = FakeModel()
    assert await Embedder(model).embed_chunks([]) == []
    assert model.calls == []


async def test_wrong_dimension_raises_dimension_error():
    model = FakeModel(dim=100)
    with pytest.raises(EmbeddingDimensionError):
        await Embedder(model).embed_query("bad dim 1")


async def test_count_mismatch_raises_dimension_error():
    model = FakeModel(short=True)
    with pytest.raises(EmbeddingDimensionError):
        await Embedder(model, batch_size=50).embed_chunks([_chunk(i) for i in range(3)])


async def test_retry_recovers_after_one_transient_failure():
    model = FakeModel(fail_times=1)
    vector = await Embedder(model, retries=1).embed_query("recover 5")
    assert len(vector) == EMBEDDING_DIM
    assert model.attempts == 2  # 1 failure + 1 success


async def test_exhausted_retries_raise_embedding_error():
    model = FakeModel(fail_times=2)
    with pytest.raises(EmbeddingError):
        await Embedder(model, retries=1).embed_query("never 9")
    assert model.attempts == 2  # retries=1 -> 2 attempts total


def test_get_embedder_is_cached_singleton():
    assert get_embedder() is get_embedder()


class TestRecentlyFailed:
    """Step 20: the health check reads this instead of relying solely on a synthetic
    probe. Timestamped window, not a single overwriteable flag — see
    _FAILURE_WINDOW_SECONDS docstring in app/embedding.py for why."""

    async def test_false_before_any_call(self):
        embedder = Embedder(FakeModel())
        assert embedder.recently_failed is False

    async def test_true_after_exhausted_retries(self):
        embedder = Embedder(FakeModel(fail_times=2), retries=1)
        with pytest.raises(EmbeddingError):
            await embedder.embed_query("never 9")
        assert embedder.recently_failed is True

    async def test_false_after_a_call_that_succeeds_outright(self):
        embedder = Embedder(FakeModel())
        await embedder.embed_query("fine 1")
        assert embedder.recently_failed is False

    async def test_a_later_success_does_not_clear_a_recent_failure(self):
        """The race this guards against: a failing call and a concurrent/later
        succeeding call must not let the success erase evidence of the failure. Modeled
        here as sequential calls on one Embedder (same underlying state either way) —
        two separate FakeModel instances stand in for what would be two real
        concurrent embed calls, since a single FakeModel's `attempts` counter can't
        represent overlapping in-flight calls."""
        failing = Embedder(FakeModel(fail_times=2), retries=1)
        with pytest.raises(EmbeddingError):
            await failing.embed_query("never 9")
        assert failing.recently_failed is True

        # A concurrent/later call on the SAME embedder instance succeeds...
        succeeding_model = FakeModel()
        failing._model = succeeding_model  # simulates the model recovering mid-process
        await failing.embed_query("fine 1")

        # ...but the earlier failure must still be visible, not silently erased.
        assert failing.recently_failed is True

    def test_ages_out_after_the_window(self):
        import time

        embedder = Embedder(FakeModel())
        # Directly inject a stale failure timestamp rather than sleeping 60s in a test.
        embedder._recent_failures.append(time.monotonic() - 61.0)
        assert embedder.recently_failed is False
