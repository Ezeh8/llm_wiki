"""Step 14 — Dead-letter system. Real filesystem + real Postgres (gated on
RUN_GRAPH_INTEGRATION_TESTS like the rest of the graph integration suite), since the
whole point of this system is durable file I/O plus DB replay — mocking either side
would leave the actual thing being built unverified.
"""

import os
import uuid

import pytest

RUN_INTEGRATION = os.environ.get("RUN_GRAPH_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="RUN_GRAPH_INTEGRATION_TESTS not set to 1"
)


@pytest.fixture(autouse=True)
async def _fresh_engine_pool():
    """pytest-asyncio gives each test function its own event loop by default, but
    app.db.session.engine's connection pool is a module-level singleton shared across
    the whole pytest process — a pooled connection created in one test's (now-dead)
    loop handed to a later test's different loop is a genuine asyncpg "attached to a
    different loop" failure. Unlike test_api.py/test_cache_system.py (which only need
    this once, right before creating a TestClient), every test in this file touches
    the shared engine directly via async_session_factory (audit_writer, replay,
    list_audit) — including ones that run in isolation and pass fine there, only to
    fail when the combined suite hands them a loop after other files' TestClients
    have already cycled through several. Disposing before every test, not just once,
    is what actually closes the gap.
    """
    from app.db.session import engine

    await engine.dispose()
    yield


@pytest.fixture
def dead_letter_dir(tmp_path, monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("DEAD_LETTER_PATH", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


class TestAuditWriterDeadLettersOnExhaustion:
    async def test_failed_audit_write_does_not_raise_and_is_captured(
        self, dead_letter_dir, monkeypatch
    ):
        from app.adapters.postgres import PostgresAdapter
        from app.dead_letter import has_backlog
        from app.graph.nodes.audit import audit_writer

        async def _broken_write_audit(self, **kwargs):
            raise RuntimeError("simulated postgres outage")

        monkeypatch.setattr(PostgresAdapter, "write_audit", _broken_write_audit)

        thread_id = f"query_{uuid.uuid4()}"
        result = await audit_writer(
            {
                "thread_id": thread_id,
                "operation_type": "query",
                "actor": "tester",
                "status": "success",
                "query_text": "q",
                "answer": "a",
            },
            {"configurable": {}},
        )

        # The whole point: the user's already-completed request must never fail just
        # because its own audit LOG entry couldn't be written.
        assert result == {}
        assert has_backlog() is True


class TestReplay:
    async def test_successful_replay_writes_to_postgres_and_clears_backlog(
        self, dead_letter_dir, monkeypatch
    ):
        from app.adapters.postgres import PostgresAdapter
        from app.dead_letter import has_backlog, replay_dead_letters
        from app.db.session import async_session_factory
        from app.graph.nodes.audit import audit_writer

        async def _broken_write_audit(self, **kwargs):
            raise RuntimeError("simulated postgres outage")

        monkeypatch.setattr(PostgresAdapter, "write_audit", _broken_write_audit)
        thread_id = f"delete_{uuid.uuid4()}"
        await audit_writer(
            {
                "thread_id": thread_id,
                "operation_type": "delete",
                "actor": "tester",
                "status": "success",
                "document_id": "doc-1",
            },
            {"configurable": {}},
        )
        assert has_backlog() is True

        monkeypatch.undo()  # restore the real write_audit
        await replay_dead_letters()

        assert has_backlog() is False
        async with async_session_factory() as session:
            rows = await PostgresAdapter(session).list_audit(limit=50)
        assert any(r.idempotency_key == f"{thread_id}:delete" for r in rows)

    async def test_entry_is_poisoned_after_max_replay_attempts_and_never_retried_again(
        self, dead_letter_dir, monkeypatch
    ):
        from app.adapters.postgres import PostgresAdapter
        from app.dead_letter import (
            MAX_REPLAY_ATTEMPTS,
            _read_entries,
            has_backlog,
            has_poisoned,
            replay_dead_letters,
        )
        from app.graph.nodes.audit import audit_writer

        async def _broken_write_audit(self, **kwargs):
            raise RuntimeError("simulated permanent outage")

        monkeypatch.setattr(PostgresAdapter, "write_audit", _broken_write_audit)
        thread_id = f"ingest_{uuid.uuid4()}"
        await audit_writer(
            {
                "thread_id": thread_id,
                "operation_type": "ingest",
                "actor": "tester",
                "status": "success",
                "document_id": "doc-2",
            },
            {"configurable": {}},
        )

        for _ in range(MAX_REPLAY_ATTEMPTS):
            await replay_dead_letters()

        assert has_backlog() is False
        assert has_poisoned() is True
        from app.dead_letter import _poisoned_path

        poisoned = _read_entries(_poisoned_path())
        assert len(poisoned) == 1
        assert poisoned[0]["retry_count"] == MAX_REPLAY_ATTEMPTS
        assert poisoned[0]["idempotency_key"] == f"{thread_id}:ingest"

        # A further replay must leave the poisoned entry completely untouched.
        await replay_dead_letters()
        assert len(_read_entries(_poisoned_path())) == 1


class TestHealthEndpointReflectsDeadLetterState:
    async def test_audit_backlog_and_poisoned_flags_over_live_http(
        self, dead_letter_dir, monkeypatch
    ):
        from fastapi.testclient import TestClient

        from app.adapters.postgres import PostgresAdapter
        from app.dead_letter import MAX_REPLAY_ATTEMPTS, replay_dead_letters
        from app.db.session import engine
        from app.graph.nodes.audit import audit_writer

        async def _broken_write_audit(self, **kwargs):
            raise RuntimeError("outage")

        monkeypatch.setattr(PostgresAdapter, "write_audit", _broken_write_audit)
        await audit_writer(
            {
                "thread_id": f"query_{uuid.uuid4()}",
                "operation_type": "query",
                "actor": "tester",
                "status": "success",
            },
            {"configurable": {}},
        )
        for _ in range(MAX_REPLAY_ATTEMPTS):
            await replay_dead_letters()

        await engine.dispose()  # already inside pytest-asyncio's loop, no asyncio.run
        from app.api.app import app

        with TestClient(app) as client:
            body = client.get("/health").json()

        assert body["audit_poisoned"] is True
        assert body["status"] == "degraded"
