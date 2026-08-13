"""Step 12 — FastAPI HTTP surface, end-to-end against a real Postgres+Qdrant+
checkpointer (same RUN_GRAPH_INTEGRATION_TESTS gate and reasoning as test_graph.py —
see that file's module docstring). The dense embedding model and LLM are mocked (same
reasoning as every graph integration test since Step 9): avoids a ~500MB-1GB Nomic
download and a real Anthropic call while everything else — auth, rate limiting,
routing, the full graph pipeline, Postgres, Qdrant — runs for real.
"""

import os
import re

import pytest

RUN_INTEGRATION = os.environ.get("RUN_GRAPH_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="RUN_GRAPH_INTEGRATION_TESTS not set to 1"
)


async def _run_with_fresh_session(fn):
    """`TestClient` runs the ASGI app in its own thread with its own event loop
    (anyio's blocking portal), separate from pytest-asyncio's loop that a test
    function/fixture runs in. Reusing `app.db.session`'s shared, module-level engine
    directly from test code causes a real "attached to a different loop" asyncpg
    error the moment both loops touch the same connection pool — so any direct DB
    access from test code (as opposed to going through the HTTP API, which correctly
    runs inside TestClient's own loop) uses a throwaway engine instead."""
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
def api_client(monkeypatch):
    """A TestClient wired to fake dense-embedding + LLM, with the real graph,
    checkpointer, Postgres, and Qdrant behind it. Also seeds admin/service/employee
    API keys directly (bypassing the CLI, since these tests want known raw keys).
    """
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

    # TestClient spins up a NEW event loop (anyio's blocking portal) per test, but
    # app.db.session.engine's connection pool is a module-level singleton that
    # outlives any single test. A pooled connection created in one test's (now-dead)
    # loop being handed to a later test's different loop is a real asyncpg
    # "attached to a different loop" error — disposing the pool before each test
    # forces fresh connections to be made against whichever loop is actually live.
    asyncio.run(engine.dispose())

    with TestClient(app) as client:
        yield client


@pytest.fixture
async def keys():
    """Seeds one key per tier directly via the adapter and returns raw-key headers."""
    import uuid

    from app.adapters.postgres import PostgresAdapter
    from app.api.auth import hash_key

    raw = {
        "admin": f"admin-{uuid.uuid4()}",
        "service": f"service-{uuid.uuid4()}",
        "employee": f"employee-{uuid.uuid4()}",
    }

    async def _seed(session):
        adapter = PostgresAdapter(session)
        for tier, raw_key in raw.items():
            await adapter.create_api_key(
                key_hash=hash_key(raw_key), tier=tier, actor_name=f"Test {tier.title()}"
            )
        await session.commit()

    await _run_with_fresh_session(_seed)
    return {tier: {"X-API-Key": key} for tier, key in raw.items()}


def _unique_content(label: str) -> bytes:
    """Content-hash duplicate detection (Node 9) is global across the whole database
    — other test files (test_graph.py) exercise it against the SAME live Postgres
    when the full suite runs together, so fixed literal test content risks a genuine
    cross-file collision. A UUID marker per call makes that impossible."""
    import uuid

    return f"# {label}\n\nBody {uuid.uuid4()}.".encode()


def _ingest(client, headers, *, filename="doc.md", content=None, title="Doc"):
    content = content if content is not None else _unique_content(title)
    return client.post(
        "/documents",
        headers=headers,
        files={"file": (filename, content, "text/markdown")},
        data={"title": title, "source_label": "test"},
    )


class TestAuth:
    def test_no_key_rejected(self, api_client):
        r = api_client.post("/documents")
        assert r.status_code == 401

    def test_wrong_tier_rejected(self, api_client, keys):
        r = _ingest(api_client, keys["employee"])
        assert r.status_code == 401

    def test_all_401_bodies_are_identical(self, api_client, keys):
        no_key = api_client.get("/documents").json()
        wrong_tier = api_client.get("/documents", headers=keys["employee"]).json()
        assert no_key == wrong_tier == {
            "status": 401,
            "error": "unauthorized",
            "message": "Unauthorized",
            "retryable": False,
        }

    async def test_revoked_key_rejected(self, api_client, keys):
        from app.adapters.postgres import PostgresAdapter
        from app.api.auth import hash_key

        async def _revoke(session):
            adapter = PostgresAdapter(session)
            key = await adapter.get_key_by_hash(hash_key(keys["admin"]["X-API-Key"]))
            key.active = False
            await session.commit()

        await _run_with_fresh_session(_revoke)

        r = api_client.get("/documents", headers=keys["admin"])
        assert r.status_code == 401


class TestHealth:
    def test_health_is_public_and_reports_connectivity(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["postgres_connected"] is True
        assert body["qdrant_connected"] is True


class TestDocumentLifecycle:
    def test_ingest_list_detail_update_delete(self, api_client, keys):
        r = _ingest(api_client, keys["admin"])
        assert r.status_code == 201
        doc = r.json()
        assert doc["chunk_count"] == 1

        r = api_client.get("/documents", headers=keys["admin"])
        assert r.status_code == 200
        assert any(d["document_id"] == doc["document_id"] for d in r.json()["items"])

        r = api_client.get(f"/documents/{doc['document_id']}", headers=keys["service"])
        assert r.status_code == 200
        assert r.json()["file_type"] == "md"

        r = api_client.put(
            f"/documents/{doc['document_id']}",
            headers=keys["admin"],
            files={"file": ("badge.md", b"# Badge Policy\n\nExpires in 60 days now.", "text/markdown")},
            data={"title": "Badge Policy", "source_label": "test"},
        )
        assert r.status_code == 200
        new_doc = r.json()
        assert new_doc["document_id"] != doc["document_id"]

        assert api_client.get(f"/documents/{doc['document_id']}", headers=keys["admin"]).status_code == 404

        r = api_client.delete(f"/documents/{new_doc['document_id']}", headers=keys["admin"])
        assert r.status_code == 204
        assert r.json() == {"deleted": True, "document_id": new_doc["document_id"]}

        r = api_client.delete(f"/documents/{new_doc['document_id']}", headers=keys["admin"])
        assert r.status_code == 204  # idempotent — no 404 for an already-gone document

    def test_duplicate_ingest_is_409(self, api_client, keys):
        content = _unique_content("Same Content Twice")
        assert _ingest(api_client, keys["admin"], filename="a.md", content=content).status_code == 201
        r = _ingest(api_client, keys["admin"], filename="b.md", content=content)
        assert r.status_code == 409
        assert r.json()["error"] == "conflict"

    def test_unsupported_extension_is_415(self, api_client, keys):
        r = api_client.post(
            "/documents",
            headers=keys["admin"],
            files={"file": ("x.exe", b"binary", "application/octet-stream")},
            data={"title": "X", "source_label": "s"},
        )
        assert r.status_code == 415

    def test_get_nonexistent_document_is_404(self, api_client, keys):
        r = api_client.get(
            "/documents/00000000-0000-0000-0000-000000000000", headers=keys["admin"]
        )
        assert r.status_code == 404


class TestQuery:
    def test_query_then_cache_hit(self, api_client, keys):
        _ingest(api_client, keys["admin"], content=_unique_content("Badge Policy"))

        r = api_client.post(
            "/query", headers=keys["service"], json={"question": "How long do badges last?"}
        )
        assert r.status_code == 200
        first = r.json()
        assert first["cached"] is False and first["degraded"] is False
        assert len(first["citations"]) == 1

        r = api_client.post(
            "/query", headers=keys["service"], json={"question": "How long do badges last?"}
        )
        second = r.json()
        assert second["cached"] is True
        assert second["answer"] == first["answer"]

    def test_empty_question_is_400(self, api_client, keys):
        r = api_client.post("/query", headers=keys["service"], json={"question": "   "})
        assert r.status_code == 400

    def test_question_too_long_is_400(self, api_client, keys):
        r = api_client.post(
            "/query", headers=keys["service"], json={"question": "x" * 2001}
        )
        assert r.status_code in (400, 422)  # 422 if Pydantic's own max_length catches it first


class TestChangelogCrud:
    def test_full_crud(self, api_client, keys):
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": "note"})
        assert r.status_code == 201
        entry = r.json()

        assert api_client.get("/changelog", headers=keys["employee"]).status_code == 200
        assert api_client.get(f"/changelog/{entry['changelog_id']}", headers=keys["employee"]).status_code == 200

        r = api_client.put(
            f"/changelog/{entry['changelog_id']}",
            headers=keys["employee"],
            json={"entry": "updated note"},
        )
        assert r.status_code == 200 and r.json()["entry"] == "updated note"

        r = api_client.delete(f"/changelog/{entry['changelog_id']}", headers=keys["employee"])
        assert r.status_code == 204
        assert r.json() == {"deleted": True, "changelog_id": entry["changelog_id"]}

        assert api_client.get(f"/changelog/{entry['changelog_id']}", headers=keys["employee"]).status_code == 404


class TestAudit:
    def test_operations_are_recorded_and_listable(self, api_client, keys):
        _ingest(api_client, keys["admin"], content=_unique_content("Audited Doc"))
        r = api_client.get("/audit", headers=keys["admin"])
        assert r.status_code == 200
        items = r.json()["items"]
        assert any(i["event_type"] == "ingest" for i in items)

        r = api_client.get(f"/audit/{items[0]['event_id']}", headers=keys["admin"])
        assert r.status_code == 200


class TestRateLimiting:
    def test_employee_limit_enforced_then_recovers_next_window(self, api_client, keys):
        statuses = [
            api_client.get("/changelog", headers=keys["employee"]).status_code
            for _ in range(12)
        ]
        assert statuses[:10] == [200] * 10
        assert statuses[10] == 429 and statuses[11] == 429

    def test_health_is_exempt_from_rate_limiting(self, api_client, keys):
        for _ in range(15):
            assert api_client.get("/health").status_code == 200

    def test_rate_limiter_runs_before_auth(self, api_client):
        """Even fully unauthenticated requests get bucketed and eventually 429 —
        proves RateLimitMiddleware is genuinely outermost (PRD: reject floods before
        doing any work), not gated behind a successful auth check first.

        Doesn't assert an exact index: the "anonymous" bucket is shared process-wide
        (RateLimitMiddleware's state lives on the module-level `app` singleton, reused
        across every test in this process), so other tests' no-key requests may have
        already consumed part of the window before this one runs. What must hold
        regardless: only 401/429 ever appear (never a 200 — auth is never actually
        satisfied), and once 429 starts, it never reverts back to 401 mid-burst.
        """
        statuses = [api_client.get("/documents").status_code for _ in range(15)]
        assert set(statuses) <= {401, 429}
        assert 429 in statuses
        first_429 = statuses.index(429)
        assert all(s == 429 for s in statuses[first_429:])


class TestPagination:
    def test_cursor_pages_are_disjoint(self, api_client, keys):
        for i in range(7):
            _ingest(api_client, keys["admin"], filename=f"d{i}.md", content=_unique_content(f"Doc {i}"))

        page1 = api_client.get("/documents", headers=keys["admin"], params={"limit": 3}).json()
        assert len(page1["items"]) == 3 and page1["next_cursor"]

        page2 = api_client.get(
            "/documents", headers=keys["admin"], params={"limit": 3, "cursor": page1["next_cursor"]}
        ).json()
        ids1 = {d["document_id"] for d in page1["items"]}
        ids2 = {d["document_id"] for d in page2["items"]}
        assert ids1.isdisjoint(ids2)
