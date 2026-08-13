"""Step 15 — Health endpoint's last remaining flag: embedding_model_available.
The other 6 flags were already real and tested by Steps 9/12/13/14 (postgres/qdrant
connectivity in test_api.py's TestHealth; cache_stale_risk lifecycle in
test_cache_system.py; audit_backlog/poisoned in test_dead_letter.py) — this file adds
only what's new here: the cold/warm/broken/timeout behavior of the embedding check,
including that it never blows the endpoint's 2-second budget (PRD Success Criteria).
"""

import os
import time

import pytest

RUN_INTEGRATION = os.environ.get("RUN_GRAPH_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="RUN_GRAPH_INTEGRATION_TESTS not set to 1"
)


@pytest.fixture(autouse=True)
def _isolated_embedder_singleton():
    """`get_embedder()` is a process-wide `@lru_cache` singleton (Step 6) — these
    tests mutate its `_model`/`_loaded` state directly to simulate cold/warm/broken
    scenarios, which would otherwise leak across tests (including this file's own
    ordering, and any other test that happens to call the real singleton rather than
    monkeypatching the `get_embedder` name in a specific module, as test_api.py/
    test_graph.py/test_cache_system.py all do). Snapshot and restore around each test.
    """
    from app.embedding import get_embedder

    embedder = get_embedder()
    original_model, original_loaded = embedder._model, embedder._loaded
    yield
    embedder._model, embedder._loaded = original_model, original_loaded


@pytest.fixture
def health_client():
    import asyncio

    from fastapi.testclient import TestClient

    from app.api.app import app
    from app.db.session import engine

    asyncio.run(engine.dispose())
    with TestClient(app) as client:
        yield client


def test_cold_unloaded_model_reports_available_without_loading(health_client):
    """Never touched in this process -> optimistic True, no load attempted, fast."""
    from app.embedding import get_embedder

    assert get_embedder().is_loaded is False

    start = time.monotonic()
    r = health_client.get("/health")
    elapsed = time.monotonic() - start

    assert r.status_code == 200
    assert r.json()["embedding_model_available"] is True
    assert elapsed < 2.0
    assert get_embedder().is_loaded is False  # confirms no load was triggered


def test_warm_healthy_model_passes_a_real_check(health_client):
    from app.embedding import get_embedder

    class FakeModel:
        def encode(self, texts, **kwargs):
            return [[0.1] * 768 for _ in texts]

    embedder = get_embedder()
    embedder._model = FakeModel()
    embedder._loaded = True

    r = health_client.get("/health")
    body = r.json()
    assert body["embedding_model_available"] is True
    assert body["status"] == "healthy"


def test_warm_broken_model_fails_and_degrades_overall_status(health_client):
    from app.embedding import get_embedder

    class BrokenModel:
        def encode(self, texts, **kwargs):
            raise RuntimeError("model crashed")

    embedder = get_embedder()
    embedder._model = BrokenModel()
    embedder._loaded = True

    r = health_client.get("/health")
    body = r.json()
    assert body["embedding_model_available"] is False
    assert body["status"] == "degraded"


def test_hung_model_is_cut_off_by_timeout_within_budget(health_client):
    """A model call that never returns must not be allowed to block the health
    endpoint past its 2s budget — proves the asyncio.wait_for timeout actually cuts
    it off rather than just being decorative."""
    from app.embedding import get_embedder

    class HangingModel:
        def encode(self, texts, **kwargs):
            time.sleep(5)
            return [[0.1] * 768 for _ in texts]

    embedder = get_embedder()
    embedder._model = HangingModel()
    embedder._loaded = True

    start = time.monotonic()
    r = health_client.get("/health")
    elapsed = time.monotonic() - start

    assert r.json()["embedding_model_available"] is False
    assert elapsed < 2.0, f"health check must stay under budget, took {elapsed}s"
