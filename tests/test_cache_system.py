"""Step 13 — Cache system verification.

Almost the entire cache system (Postgres table since Step 1, Cache Checker/Writer
since Step 8/10, the flush side effect + cache_stale_risk health flag since Step 9,
TTL self-heal since Step 2) was already built organically while implementing the
query and ingest/delete paths — see BUILD_LOG's Step 13 entry for the full audit. This
file closes the verification gaps that were never explicitly exercised end-to-end:
invalidation actually happening on a real ingest/delete (not just "the flush function
gets called" at the unit level), TTL expiry self-healing through the real Cache
Checker node (not just the adapter directly), and the cache_stale_risk flag's full
set-then-clear lifecycle reflected through the live health endpoint.

Same RUN_GRAPH_INTEGRATION_TESTS gate and mocked-embedder/LLM reasoning as
test_graph.py / test_api.py.
"""

import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import pytest

RUN_INTEGRATION = os.environ.get("RUN_GRAPH_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="RUN_GRAPH_INTEGRATION_TESTS not set to 1"
)


@pytest.fixture
def wired_client(monkeypatch):
    from app.embedding import Embedder

    class FakeModel:
        def encode(self, texts, **kwargs):
            return [[0.1] * 768 for _ in texts]

    fake_embedder = Embedder(model=FakeModel())

    import app.graph.nodes.ingest_path as ingest_mod
    import app.graph.nodes.query_path as query_mod

    monkeypatch.setattr(ingest_mod, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(query_mod, "get_embedder", lambda: fake_embedder)

    from app.domain import Citation, GenerationResult

    class EchoingLLM:
        async def generate(self, *, system, user):
            match = re.search(r"chunk_id=(\S+)", user)
            chunk_id = match.group(1) if match else "unknown"
            return GenerationResult(
                answer="Badges expire after 90 days.",
                citations=[
                    Citation(
                        document_id="d",
                        document_title="Badge Policy",
                        chunk_id=chunk_id,
                        chunk_text="t",
                        chunk_index=0,
                    )
                ],
            )

    monkeypatch.setattr(query_mod, "build_llm_adapter", lambda: EchoingLLM())

    import asyncio

    from fastapi.testclient import TestClient

    from app.api.app import app
    from app.db.session import engine

    asyncio.run(engine.dispose())  # avoid cross-event-loop pooled connections

    with TestClient(app) as client:
        yield client


async def _run_with_fresh_session(fn):
    """See tests/test_api.py's identical helper docstring for why: TestClient runs
    the ASGI app in its own event loop, so DB access from test code that isn't routed
    through the HTTP API needs its own throwaway engine, not the app's shared one."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.config import get_settings

    engine = create_async_engine(get_settings().database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            return await fn(session)
    finally:
        await engine.dispose()


@pytest.fixture
def admin_key(wired_client):
    import asyncio

    from app.adapters.postgres import PostgresAdapter
    from app.api.auth import hash_key

    raw = f"admin-{uuid.uuid4()}"

    async def _seed(session):
        await PostgresAdapter(session).create_api_key(
            key_hash=hash_key(raw), tier="admin", actor_name="Cache Test Admin"
        )
        await session.commit()

    asyncio.run(_run_with_fresh_session(_seed))
    return {"X-API-Key": raw}


def _ingest(client, headers, label: str):
    content = f"# {label}\n\nBody {uuid.uuid4()}.".encode()
    return client.post(
        "/documents",
        headers=headers,
        files={"file": (f"{label}.md", content, "text/markdown")},
        data={"title": label, "source_label": "test"},
    )


class TestCacheInvalidation:
    def test_ingest_flushes_a_previously_cached_answer(self, wired_client, admin_key):
        _ingest(wired_client, admin_key, "First Doc")
        question = f"What does the cache flush test say, {uuid.uuid4()}?"

        first = wired_client.post(
            "/query", headers=admin_key, json={"question": question}
        ).json()
        assert first["cached"] is False

        cached_hit = wired_client.post(
            "/query", headers=admin_key, json={"question": question}
        ).json()
        assert cached_hit["cached"] is True  # confirms it was actually cached

        # Ingesting an unrelated document must flush the ENTIRE cache table
        # (PRD: "Invalidated on any ingest/delete/update" — table-wide, not scoped).
        _ingest(wired_client, admin_key, "Second Doc")

        after_ingest = wired_client.post(
            "/query", headers=admin_key, json={"question": question}
        ).json()
        assert after_ingest["cached"] is False, "stale cache must not survive an ingest"

    def test_delete_flushes_a_previously_cached_answer(self, wired_client, admin_key):
        doc = _ingest(wired_client, admin_key, "To Be Deleted").json()
        question = f"Delete-flush test question {uuid.uuid4()}?"

        wired_client.post("/query", headers=admin_key, json={"question": question})
        assert wired_client.post(
            "/query", headers=admin_key, json={"question": question}
        ).json()["cached"] is True

        r = wired_client.delete(f"/documents/{doc['document_id']}", headers=admin_key)
        assert r.status_code == 204

        after_delete = wired_client.post(
            "/query", headers=admin_key, json={"question": question}
        ).json()
        assert after_delete["cached"] is False


class TestTtlSelfHeal:
    def test_expired_cache_row_is_treated_as_a_miss_by_the_real_cache_checker_node(
        self, wired_client, admin_key
    ):
        """Exercises the actual Cache Checker node (Step 8), not just
        PostgresAdapter.get_cache directly (already unit-tested in
        test_postgres_adapter.py) — proves TTL self-heal holds through the real
        normalize_cache_key -> lookup -> miss-routes-to-Embedder path."""
        from app.adapters.postgres import PostgresAdapter
        from app.graph.nodes.query_path import normalize_cache_key

        question = f"Expired cache question {uuid.uuid4()}?"
        key = normalize_cache_key(question)

        async def _seed_expired(session):
            await PostgresAdapter(session).write_cache(
                cache_key=key,
                question_text=question,
                answer="This stale answer must never be served.",
                citations=[],
                expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            )
            await session.commit()

        import asyncio

        asyncio.run(_run_with_fresh_session(_seed_expired))

        # No document ingested — Retriever will find nothing, so this correctly lands
        # on "insufficient" if it actually re-runs the pipeline instead of serving the
        # expired row. Either way, `cached` must be False.
        result = wired_client.post(
            "/query", headers=admin_key, json={"question": question}
        ).json()
        assert result["cached"] is False
        assert result["answer"] != "This stale answer must never be served."


class TestCacheStaleRiskFlag:
    def test_flush_failure_sets_flag_and_next_success_clears_it(
        self, wired_client, admin_key, monkeypatch
    ):
        # Force the cache-flush's own DB call to fail, simulating a transient outage
        # (D-16/D-24: flush failure must not fail the ingest, only raise the flag).
        import app.graph.nodes._common as common_mod
        from app.adapters.postgres import PostgresAdapter

        original_flush_cache = PostgresAdapter.flush_cache

        async def _broken_flush_cache(self):
            raise RuntimeError("simulated cache flush outage")

        monkeypatch.setattr(PostgresAdapter, "flush_cache", _broken_flush_cache)

        r = _ingest(wired_client, admin_key, "Flush Failure Doc")
        assert r.status_code == 201  # the ingest itself must still succeed

        health = wired_client.get("/health").json()
        assert health["cache_stale_risk"] is True
        assert health["status"] == "degraded"

        # Restore real flush behavior — the next successful flush must clear the flag.
        monkeypatch.setattr(PostgresAdapter, "flush_cache", original_flush_cache)
        r = _ingest(wired_client, admin_key, "Flush Recovery Doc")
        assert r.status_code == 201

        health_after = wired_client.get("/health").json()
        assert health_after["cache_stale_risk"] is False
        assert health_after["status"] == "healthy"
