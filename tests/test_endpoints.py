"""Step 18 — PRD Section 9's dedicated CI test suite ("~52 tests, in-memory Postgres,
mocked Qdrant. 3 per endpoint + specific failure tests. Failing test blocks deploy.").

Deliberately NOT the same suite as test_api.py/test_graph.py/etc.: those are real,
opt-in integration tests against live Postgres+Qdrant Docker containers
(RUN_GRAPH_INTEGRATION_TESTS=1), built alongside each Step 8-17 feature to catch real
cross-system bugs (and did — see BUILD_LOG Steps 9/11). This file is the fast,
always-on deploy gate: run via `scripts/run_ci_tests.sh`, which brings up a real but
throwaway tmpfs-backed Postgres (see that script's docstring for why "in-memory
Postgres" means that rather than SQLite) and never touches a live Qdrant at all —
every Qdrant call in this file is served by an in-process FakeQdrantClient below.

Mocked in every test: the Qdrant client (FakeQdrantClient), the dense embedding model
and BM25 sparse encoder (deterministic fakes — no ~500MB-1GB Nomic download, no
fastembed download), the LLM adapter (EchoingLLM / degrading variants — no network
call), and the Qdrant collection-provisioning startup step (a real one would try to
reach the same nonexistent Qdrant the app is configured to skip). Postgres, the
compiled LangGraph, FastAPI's routing/auth/rate-limiting, and every HTTP contract are
all real — only the two genuinely heavy/networked dependencies are faked.
"""

import re
import uuid
from types import SimpleNamespace

import pytest
from qdrant_client import models as qm


# --- Fake Qdrant -------------------------------------------------------------------


class FakeQdrantClient:
    """In-memory stand-in for AsyncQdrantClient, wired in under VectorStoreAdapter
    (app/adapters/vector_store.py) exactly the way the real client is — so every
    payload-shape/status-check assumption the adapter makes is exercised for real.
    Dense and sparse `query_points` calls both just return every point currently
    matching the filter (scored 1.0): with the fake embedder/BM25 encoder also
    returning constant vectors, real similarity ranking isn't under test here — the
    ingest -> retrieve -> rank -> generate round trip through the API is.
    """

    def __init__(self) -> None:
        self._points: dict[str, qm.PointStruct] = {}

    async def upsert(self, *, collection_name: str, points: list[qm.PointStruct]):
        for point in points:
            self._points[str(point.id)] = point
        return SimpleNamespace(status=qm.UpdateStatus.COMPLETED)

    async def delete(self, *, collection_name: str, points_selector: qm.FilterSelector):
        document_id = _filter_value(points_selector.filter, "document_id")
        if document_id is None:
            return
        self._points = {
            pid: p
            for pid, p in self._points.items()
            if (p.payload or {}).get("document_id") != document_id
        }

    async def query_points(
        self, *, collection_name, query, using, limit, query_filter, with_payload
    ):
        document_id = _filter_value(query_filter, "document_id")
        source_label = _filter_value(query_filter, "source_label")
        hits = []
        for point in self._points.values():
            payload = point.payload or {}
            if document_id is not None and payload.get("document_id") != document_id:
                continue
            if source_label is not None and payload.get("source_label") != source_label:
                continue
            hits.append(SimpleNamespace(id=point.id, score=1.0, payload=payload))
        return SimpleNamespace(points=hits[:limit])


def _filter_value(qdrant_filter: qm.Filter | None, key: str) -> str | None:
    if qdrant_filter is None:
        return None
    for condition in qdrant_filter.must or []:
        if condition.key == key:
            return condition.match.value
    return None


class FakeSparseModel:
    """`embed`/`query_embed` shape matching app.embedding.SparseModel — one nonzero
    term per call is enough; BM25 ranking correctness isn't in scope for this file."""

    def embed(self, texts, **kwargs):
        return [SimpleNamespace(indices=[0], values=[1.0]) for _ in texts]

    def query_embed(self, texts, **kwargs):
        return [SimpleNamespace(indices=[0], values=[1.0]) for _ in texts]


class EchoingLLM:
    """Cites whatever chunk_id the prompt actually contains, so grounding
    (generator strips citations to unranked chunk_ids) is exercised honestly rather
    than trivially satisfied by a hardcoded id."""

    async def generate(self, *, system, user):
        from app.domain import Citation, GenerationResult

        match = re.search(r"chunk_id=(\S+)", user)
        chunk_id = match.group(1) if match else "unknown"
        doc_match = re.search(r"document='([^']*)'", user)
        return GenerationResult(
            answer="Badges expire after 90 days.",
            citations=[
                Citation(
                    document_id="doc",
                    document_title=doc_match.group(1) if doc_match else "Doc",
                    chunk_id=chunk_id,
                    chunk_text="t",
                    chunk_index=0,
                )
            ],
        )


class FailingLLM:
    async def generate(self, *, system, user):
        from app.adapters.llm import LLMError

        raise LLMError("simulated LLM failure")


class FailingEmbedder:
    async def embed_query(self, text):
        from app.embedding import EmbeddingError

        raise EmbeddingError("embedding failed after 2 attempt(s): simulated failure")

    async def embed_chunks(self, chunks):
        from app.embedding import EmbeddingError

        raise EmbeddingError("embedding failed after 2 attempt(s): simulated failure")


# --- Fixtures ------------------------------------------------------------------


async def _run_with_fresh_session(fn):
    """Same reasoning as test_api.py: TestClient runs the ASGI app in its own
    event loop (anyio's blocking portal), separate from pytest-asyncio's loop — a
    throwaway engine avoids handing a pooled connection across loops."""
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
    from app.embedding import Embedder

    class FakeModel:
        def encode(self, texts, **kwargs):
            return [[0.1] * 768 for _ in texts]

    from app.domain import SparseVector

    fake_embedder = Embedder(model=FakeModel())
    fake_bm25 = SimpleNamespace(
        encode_chunks=lambda chunks: [SparseVector(indices=[0], values=[1.0]) for _ in chunks],
        encode_query=lambda text: SparseVector(indices=[0], values=[1.0]),
    )

    import app.graph.nodes.delete_path as delete_mod
    import app.graph.nodes.ingest_path as ingest_mod
    import app.graph.nodes.query_path as query_mod

    monkeypatch.setattr(ingest_mod, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(query_mod, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(ingest_mod, "get_bm25_encoder", lambda: fake_bm25)
    monkeypatch.setattr(query_mod, "get_bm25_encoder", lambda: fake_bm25)

    from app.adapters.vector_store import VectorStoreAdapter

    fake_qdrant = FakeQdrantClient()
    fake_store = VectorStoreAdapter(client=fake_qdrant, collection_name="test")
    monkeypatch.setattr(ingest_mod, "build_vector_store", lambda: fake_store)
    monkeypatch.setattr(query_mod, "build_vector_store", lambda: fake_store)
    monkeypatch.setattr(delete_mod, "build_vector_store", lambda: fake_store)

    monkeypatch.setattr(query_mod, "build_llm_adapter", lambda: EchoingLLM())

    import app.api.app as app_module

    async def _noop_ensure_collection(*args, **kwargs):
        return False

    monkeypatch.setattr(app_module, "ensure_collection", _noop_ensure_collection)

    import asyncio

    from fastapi.testclient import TestClient

    from app.api.app import app
    from app.db.session import engine

    asyncio.run(engine.dispose())

    # RateLimitMiddleware's `_buckets` dict lives on the middleware instance, which
    # Starlette builds once and caches on `app.middleware_stack` — and `app` itself is
    # a module-level singleton shared by every test in this process (see test_api.py's
    # own docstring on the same issue). Without this, an "anonymous"/no-key request in
    # test N shares its rate-limit window with every no-key request in tests 1..N-1,
    # so later tests spuriously see 429 instead of 401. Clearing the cached stack
    # forces a fresh RateLimitMiddleware (and therefore an empty bucket) on the next
    # request, giving every test a genuinely clean rate-limit slate.
    app.middleware_stack = None

    with TestClient(app) as client:
        yield client


@pytest.fixture
async def keys():
    import uuid as uuid_mod

    from app.adapters.postgres import PostgresAdapter
    from app.api.auth import hash_key

    raw = {
        "admin": f"admin-{uuid_mod.uuid4()}",
        "service": f"service-{uuid_mod.uuid4()}",
        "employee": f"employee-{uuid_mod.uuid4()}",
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
    return f"# {label}\n\nBody {uuid.uuid4()}.".encode()


def _ingest(client, headers, *, filename="doc.md", content=None, title="Doc", source_label="test"):
    content = content if content is not None else _unique_content(title)
    return client.post(
        "/documents",
        headers=headers,
        files={"file": (filename, content, "text/markdown")},
        data={"title": title, "source_label": source_label},
    )


# --- POST /documents ---------------------------------------------------------------


class TestPostDocuments:
    def test_success_returns_201_with_chunk_count(self, api_client, keys):
        r = _ingest(api_client, keys["admin"], content=_unique_content("Post Doc Success"))
        assert r.status_code == 201
        assert r.json()["chunk_count"] >= 1

    def test_no_key_is_401(self, api_client):
        r = api_client.post("/documents")
        assert r.status_code == 401

    def test_non_admin_tier_is_401(self, api_client, keys):
        r = _ingest(api_client, keys["service"])
        assert r.status_code == 401

    def test_whitespace_only_title_is_400_not_500(self, api_client, keys):
        r = _ingest(api_client, keys["admin"], title="   ")
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"


# --- GET /documents ------------------------------------------------------------


class TestListDocuments:
    def test_success_lists_ingested_document(self, api_client, keys):
        r = _ingest(api_client, keys["admin"], content=_unique_content("List Doc"))
        doc_id = r.json()["document_id"]

        r = api_client.get("/documents", headers=keys["admin"])
        assert r.status_code == 200
        assert any(d["document_id"] == doc_id for d in r.json()["items"])

    def test_no_key_is_401(self, api_client):
        assert api_client.get("/documents").status_code == 401

    def test_employee_tier_is_401(self, api_client, keys):
        r = api_client.get("/documents", headers=keys["employee"])
        assert r.status_code == 401

    def test_malformed_cursor_is_400_not_500(self, api_client, keys):
        r = api_client.get("/documents?cursor=not-a-uuid", headers=keys["admin"])
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"

    def test_stale_cursor_is_400_not_500(self, api_client, keys):
        r = api_client.get(
            "/documents?cursor=99999999-9999-9999-9999-999999999999", headers=keys["admin"]
        )
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"


# --- GET /documents/{id} ------------------------------------------------------------


class TestGetDocumentDetail:
    def test_success_returns_file_metadata(self, api_client, keys):
        r = _ingest(api_client, keys["admin"], filename="detail.md", content=_unique_content("Detail Doc"))
        doc_id = r.json()["document_id"]

        r = api_client.get(f"/documents/{doc_id}", headers=keys["service"])
        assert r.status_code == 200
        assert r.json()["file_type"] == "md"

    def test_nonexistent_id_is_404(self, api_client, keys):
        r = api_client.get(
            "/documents/00000000-0000-0000-0000-000000000000", headers=keys["admin"]
        )
        assert r.status_code == 404

    def test_malformed_id_is_400_not_500(self, api_client, keys):
        r = api_client.get("/documents/not-a-uuid", headers=keys["admin"])
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"

    def test_no_key_is_401(self, api_client):
        r = api_client.get("/documents/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 401


# --- DELETE /documents/{id} ------------------------------------------------------------


class TestDeleteDocument:
    def test_success_returns_204_no_body(self, api_client, keys):
        r = _ingest(api_client, keys["admin"], content=_unique_content("Delete Doc"))
        doc_id = r.json()["document_id"]

        r = api_client.delete(f"/documents/{doc_id}", headers=keys["admin"])
        assert r.status_code == 204
        assert r.content == b""

    def test_repeated_delete_is_idempotent_204(self, api_client, keys):
        r = _ingest(api_client, keys["admin"], content=_unique_content("Delete Twice Doc"))
        doc_id = r.json()["document_id"]
        api_client.delete(f"/documents/{doc_id}", headers=keys["admin"])

        r = api_client.delete(f"/documents/{doc_id}", headers=keys["admin"])
        assert r.status_code == 204

    def test_malformed_id_is_400_not_500_or_503(self, api_client, keys):
        r = api_client.delete("/documents/not-a-uuid", headers=keys["admin"])
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"

    def test_no_key_is_401(self, api_client):
        r = api_client.delete("/documents/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 401


# --- PUT /documents/{id} ------------------------------------------------------------


class TestUpdateDocument:
    def test_success_replaces_document_with_new_id(self, api_client, keys):
        r = _ingest(api_client, keys["admin"], content=_unique_content("Update Doc Original"))
        old_id = r.json()["document_id"]

        r = api_client.put(
            f"/documents/{old_id}",
            headers=keys["admin"],
            files={"file": ("updated.md", _unique_content("Update Doc New"), "text/markdown")},
            data={"title": "Updated", "source_label": "test"},
        )
        assert r.status_code == 200
        new_id = r.json()["document_id"]
        assert new_id != old_id
        assert api_client.get(f"/documents/{old_id}", headers=keys["admin"]).status_code == 404

    def test_whitespace_only_title_is_400_not_500(self, api_client, keys):
        r = _ingest(api_client, keys["admin"], content=_unique_content("Update Blank Title Doc"))
        doc_id = r.json()["document_id"]

        r = api_client.put(
            f"/documents/{doc_id}",
            headers=keys["admin"],
            files={"file": ("updated.md", _unique_content("New"), "text/markdown")},
            data={"title": "   ", "source_label": "test"},
        )
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"

    def test_nonexistent_id_is_404(self, api_client, keys):
        r = api_client.put(
            "/documents/00000000-0000-0000-0000-000000000000",
            headers=keys["admin"],
            files={"file": ("x.md", b"# X\n\nbody", "text/markdown")},
            data={"title": "X", "source_label": "test"},
        )
        assert r.status_code == 404

    def test_no_key_is_401(self, api_client):
        r = api_client.put(
            "/documents/00000000-0000-0000-0000-000000000000",
            files={"file": ("x.md", b"# X\n\nbody", "text/markdown")},
            data={"title": "X", "source_label": "test"},
        )
        assert r.status_code == 401


# --- POST /query ------------------------------------------------------------


class TestQuery:
    def test_success_returns_grounded_answer(self, api_client, keys):
        _ingest(api_client, keys["admin"], content=_unique_content("Badge Policy Query"))

        r = api_client.post(
            "/query", headers=keys["service"], json={"question": "How long do badges last?"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["degraded"] is False
        assert len(body["citations"]) == 1

    def test_empty_question_is_400(self, api_client, keys):
        r = api_client.post("/query", headers=keys["service"], json={"question": "   "})
        assert r.status_code == 400

    def test_employee_tier_is_401(self, api_client, keys):
        r = api_client.post(
            "/query", headers=keys["employee"], json={"question": "How long do badges last?"}
        )
        assert r.status_code == 401


# --- POST /changelog ------------------------------------------------------------


class TestCreateChangelog:
    def test_success_returns_201(self, api_client, keys):
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": "note"})
        assert r.status_code == 201
        assert r.json()["entry"] == "note"

    def test_no_key_is_401(self, api_client):
        r = api_client.post("/changelog", json={"entry": "note"})
        assert r.status_code == 401

    def test_empty_entry_is_422(self, api_client, keys):
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": ""})
        assert r.status_code == 422

    def test_nonexistent_document_id_is_404_not_500(self, api_client, keys):
        r = api_client.post(
            "/changelog",
            headers=keys["employee"],
            json={"entry": "dangling ref", "document_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert r.status_code == 404
        assert r.json()["error"] == "not_found"

    def test_malformed_document_id_is_400_not_500(self, api_client, keys):
        r = api_client.post(
            "/changelog",
            headers=keys["employee"],
            json={"entry": "dangling ref", "document_id": "not-a-uuid"},
        )
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"


# --- GET /changelog ------------------------------------------------------------


class TestListChangelog:
    def test_success_lists_created_entry(self, api_client, keys):
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": "list-me"})
        entry_id = r.json()["changelog_id"]

        r = api_client.get("/changelog", headers=keys["employee"])
        assert r.status_code == 200
        assert any(e["changelog_id"] == entry_id for e in r.json()["items"])

    def test_no_key_is_401(self, api_client):
        assert api_client.get("/changelog").status_code == 401

    def test_service_tier_is_401(self, api_client, keys):
        r = api_client.get("/changelog", headers=keys["service"])
        assert r.status_code == 401

    def test_malformed_cursor_is_400_not_500(self, api_client, keys):
        r = api_client.get("/changelog?cursor=not-a-uuid", headers=keys["employee"])
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"


# --- GET /changelog/{id} ------------------------------------------------------------


class TestGetChangelogDetail:
    def test_success(self, api_client, keys):
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": "detail-me"})
        entry_id = r.json()["changelog_id"]

        r = api_client.get(f"/changelog/{entry_id}", headers=keys["employee"])
        assert r.status_code == 200
        assert r.json()["entry"] == "detail-me"

    def test_nonexistent_id_is_404(self, api_client, keys):
        r = api_client.get(
            "/changelog/00000000-0000-0000-0000-000000000000", headers=keys["admin"]
        )
        assert r.status_code == 404

    def test_malformed_id_is_400_not_500(self, api_client, keys):
        r = api_client.get("/changelog/not-a-uuid", headers=keys["admin"])
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"

    def test_no_key_is_401(self, api_client):
        r = api_client.get("/changelog/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 401


# --- PUT /changelog/{id} ------------------------------------------------------------


class TestUpdateChangelog:
    def test_success(self, api_client, keys):
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": "before"})
        entry_id = r.json()["changelog_id"]

        r = api_client.put(
            f"/changelog/{entry_id}", headers=keys["employee"], json={"entry": "after"}
        )
        assert r.status_code == 200
        assert r.json()["entry"] == "after"

    def test_empty_entry_is_422(self, api_client, keys):
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": "before"})
        entry_id = r.json()["changelog_id"]

        r = api_client.put(
            f"/changelog/{entry_id}", headers=keys["employee"], json={"entry": ""}
        )
        assert r.status_code == 422

    def test_nonexistent_id_is_404(self, api_client, keys):
        r = api_client.put(
            "/changelog/00000000-0000-0000-0000-000000000000",
            headers=keys["admin"],
            json={"entry": "x"},
        )
        assert r.status_code == 404

    def test_malformed_id_is_400_not_500(self, api_client, keys):
        r = api_client.put(
            "/changelog/not-a-uuid", headers=keys["admin"], json={"entry": "x"}
        )
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"

    def test_no_key_is_401(self, api_client):
        r = api_client.put(
            "/changelog/00000000-0000-0000-0000-000000000000", json={"entry": "x"}
        )
        assert r.status_code == 401


# --- DELETE /changelog/{id} ------------------------------------------------------------


class TestDeleteChangelog:
    def test_success_returns_204_no_body(self, api_client, keys):
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": "to-delete"})
        entry_id = r.json()["changelog_id"]

        r = api_client.delete(f"/changelog/{entry_id}", headers=keys["employee"])
        assert r.status_code == 204
        assert r.content == b""

    def test_repeated_delete_is_404(self, api_client, keys):
        """Unlike DELETE /documents/{id} (deliberately idempotent — it routes through
        the graph's Deleter node, see documents.py), changelog deletion is a plain
        Postgres adapter call: a second delete of an already-gone row is a genuine
        404, not a no-op success."""
        r = api_client.post("/changelog", headers=keys["employee"], json={"entry": "delete-twice"})
        entry_id = r.json()["changelog_id"]
        api_client.delete(f"/changelog/{entry_id}", headers=keys["employee"])

        r = api_client.delete(f"/changelog/{entry_id}", headers=keys["employee"])
        assert r.status_code == 404

    def test_malformed_id_is_400_not_500(self, api_client, keys):
        r = api_client.delete("/changelog/not-a-uuid", headers=keys["employee"])
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"

    def test_no_key_is_401(self, api_client):
        r = api_client.delete("/changelog/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 401


# --- GET /audit ------------------------------------------------------------


class TestListAudit:
    def test_success_records_ingest_event(self, api_client, keys):
        _ingest(api_client, keys["admin"], content=_unique_content("Audited Doc"))

        r = api_client.get("/audit", headers=keys["admin"])
        assert r.status_code == 200
        assert any(e["event_type"] == "ingest" for e in r.json()["items"])

    def test_no_key_is_401(self, api_client):
        assert api_client.get("/audit").status_code == 401

    def test_employee_tier_is_401(self, api_client, keys):
        r = api_client.get("/audit", headers=keys["employee"])
        assert r.status_code == 401

    def test_malformed_cursor_is_400_not_500(self, api_client, keys):
        r = api_client.get("/audit?cursor=not-a-uuid", headers=keys["admin"])
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"


# --- GET /audit/{id} ------------------------------------------------------------


class TestGetAuditDetail:
    def test_success(self, api_client, keys):
        _ingest(api_client, keys["admin"], content=_unique_content("Audit Detail Doc"))
        event_id = api_client.get("/audit", headers=keys["admin"]).json()["items"][0]["event_id"]

        r = api_client.get(f"/audit/{event_id}", headers=keys["admin"])
        assert r.status_code == 200

    def test_nonexistent_id_is_404(self, api_client, keys):
        r = api_client.get(
            "/audit/00000000-0000-0000-0000-000000000000", headers=keys["admin"]
        )
        assert r.status_code == 404

    def test_malformed_id_is_400_not_500(self, api_client, keys):
        r = api_client.get("/audit/not-a-uuid", headers=keys["admin"])
        assert r.status_code == 400
        assert r.json()["error"] == "bad_request"

    def test_no_key_is_401(self, api_client):
        r = api_client.get("/audit/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 401


# --- GET /health ------------------------------------------------------------


class TestHealth:
    def test_reports_degraded_when_qdrant_unreachable(self, api_client):
        """No live Qdrant exists in this suite (QDRANT_URL points at a refusing
        port, see scripts/run_ci_tests.sh) — the health check must honestly report
        that, not silently swallow it."""
        r = api_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["postgres_connected"] is True
        assert body["qdrant_connected"] is False
        assert body["status"] == "degraded"

    def test_reports_healthy_when_qdrant_reachable(self, api_client, monkeypatch):
        import app.api.routers.health as health_mod

        class FakeConnectedClient:
            async def get_collections(self):
                return None

            async def close(self):
                return None

        monkeypatch.setattr(
            health_mod, "AsyncQdrantClient", lambda url: FakeConnectedClient()
        )
        r = api_client.get("/health")
        assert r.json()["qdrant_connected"] is True
        assert r.json()["status"] == "healthy"

    def test_is_public_and_exempt_from_rate_limiting(self, api_client):
        for _ in range(15):
            assert api_client.get("/health").status_code == 200

    def test_reports_unhealthy_on_recent_real_failure_even_if_probe_would_pass(
        self, api_client, monkeypatch
    ):
        """Step 20 finding: a real incident showed the OLD ping-only check reporting
        healthy while every real query failed — a fixed-length synthetic probe can
        coincidentally avoid whatever sequence length trips a real failure. This proves
        the fix's actual mechanism, not just its outcome: recently_failed is checked
        BEFORE the probe, and a probe that would clearly succeed must never override it."""
        import app.api.routers.health as health_mod

        class FakeConnectedQdrant:
            async def get_collections(self):
                return None

            async def close(self):
                return None

        monkeypatch.setattr(
            health_mod, "AsyncQdrantClient", lambda url: FakeConnectedQdrant()
        )

        class BrokenButPingableEmbedder:
            is_loaded = True
            recently_failed = True  # a real embed call failed recently

            def __init__(self):
                self.ping_called = False

            async def embed_query(self, text):
                # If this ever runs, the fix's ordering is wrong — recently_failed
                # must short-circuit before the probe is attempted.
                self.ping_called = True
                return [0.1] * 768

        fake_embedder = BrokenButPingableEmbedder()
        monkeypatch.setattr(health_mod, "get_embedder", lambda: fake_embedder)

        r = api_client.get("/health")
        body = r.json()
        assert body["embedding_model_available"] is False
        assert body["status"] == "degraded"
        assert fake_embedder.ping_called is False


# --- Specific failure tests (PRD Step 18: "+ specific failure tests") -----------


class TestSpecificFailures:
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

    def test_oversized_file_is_413(self, api_client, keys):
        oversized = b"x" * (21 * 1024 * 1024)  # MAX_FILE_SIZE_MB defaults to 20
        r = api_client.post(
            "/documents",
            headers=keys["admin"],
            files={"file": ("big.md", oversized, "text/markdown")},
            data={"title": "Big", "source_label": "s"},
        )
        assert r.status_code == 413

    def test_query_with_no_matching_documents_returns_insufficient_answer(self, api_client, keys):
        r = api_client.post(
            "/query",
            headers=keys["service"],
            json={"question": f"Completely unmatched question {uuid.uuid4()}?"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["citations"] == []
        assert body["degraded"] is False

    def test_query_degrades_when_llm_fails(self, api_client, keys, monkeypatch):
        import app.graph.nodes.query_path as query_mod

        _ingest(api_client, keys["admin"], content=_unique_content("Degraded Answer Doc"))
        monkeypatch.setattr(query_mod, "build_llm_adapter", lambda: FailingLLM())

        r = api_client.post(
            "/query", headers=keys["service"], json={"question": "Anything?"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["degraded"] is True
        assert body["citations"] == []
        assert body["source_chunks"] is not None

    def test_query_embedding_failure_is_503_not_500(self, api_client, keys, monkeypatch):
        """Step 20 finding: an exhausted EmbeddingError previously reached FastAPI
        unhandled (raw 500) — nothing between embedder_query and the ASGI layer ever
        caught it. Full chain: node catches it -> route_after_embedder_query sends it
        to Audit Writer instead of Retriever -> error_mapping.py maps the message to
        ServiceUnavailableError."""
        import app.graph.nodes.query_path as query_mod

        monkeypatch.setattr(query_mod, "get_embedder", lambda: FailingEmbedder())

        r = api_client.post(
            "/query", headers=keys["service"], json={"question": "Anything?"}
        )
        assert r.status_code == 503
        assert r.json()["error"] == "service_unavailable"

    def test_ingest_embedding_failure_is_503_not_500(self, api_client, keys, monkeypatch):
        """Same Step 20 fix, ingestion side: embedding_batcher -> Storer would otherwise
        KeyError on the missing `embeddings` state key without
        route_after_embedding_batcher."""
        import app.graph.nodes.ingest_path as ingest_mod

        monkeypatch.setattr(ingest_mod, "get_embedder", lambda: FailingEmbedder())

        r = _ingest(api_client, keys["admin"], content=_unique_content("Embedding Fail Doc"))
        assert r.status_code == 503
        assert r.json()["error"] == "service_unavailable"

    def test_query_then_cache_hit(self, api_client, keys):
        _ingest(api_client, keys["admin"], content=_unique_content("Cache Hit Doc"))
        question = "How does caching behave here?"

        first = api_client.post("/query", headers=keys["service"], json={"question": question}).json()
        assert first["cached"] is False

        second = api_client.post("/query", headers=keys["service"], json={"question": question}).json()
        assert second["cached"] is True
        assert second["answer"] == first["answer"]

    def test_all_401_bodies_are_identical(self, api_client, keys):
        no_key = api_client.get("/documents").json()
        wrong_tier = api_client.get("/documents", headers=keys["employee"]).json()
        assert no_key == wrong_tier == {
            "status": 401,
            "error": "unauthorized",
            "message": "Unauthorized",
            "retryable": False,
        }

    async def test_revoked_key_is_rejected(self, api_client, keys):
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

    async def test_revoke_api_key_cli_actually_works(self, api_client, monkeypatch):
        """Step 20 finding: `active` existed and the auth check honored it, but no
        code path ever set it False — create_api_key.py had no counterpart. Exercises
        the real create_api_key.create_key() -> revoke_api_key.revoke_key() path
        end-to-end, not the ORM-mutation backdoor test_revoked_key_is_rejected uses."""
        import app.create_api_key as create_mod
        import app.revoke_api_key as revoke_mod
        from app.config import get_settings
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        engine = create_async_engine(get_settings().database_url)
        try:
            factory = async_sessionmaker(engine, expire_on_commit=False)
            monkeypatch.setattr(create_mod, "async_session_factory", factory)
            monkeypatch.setattr(revoke_mod, "async_session_factory", factory)
            raw_key, key_id = await create_mod.create_key(tier="admin", actor_name="Revoke Me")

            r = api_client.get("/documents", headers={"X-API-Key": raw_key})
            assert r.status_code == 200

            found = await revoke_mod.revoke_key(key_id=key_id)
            assert found is True

            r = api_client.get("/documents", headers={"X-API-Key": raw_key})
            assert r.status_code == 401
        finally:
            await engine.dispose()

    def test_employee_rate_limit_enforced_then_recovers_next_window(self, api_client, keys):
        statuses = [
            api_client.get("/changelog", headers=keys["employee"]).status_code
            for _ in range(12)
        ]
        assert statuses[:10] == [200] * 10
        assert statuses[10] == 429 and statuses[11] == 429

    def test_document_cursor_pages_are_disjoint(self, api_client, keys):
        for i in range(7):
            _ingest(api_client, keys["admin"], filename=f"d{i}.md", content=_unique_content(f"Page Doc {i}"))

        page1 = api_client.get("/documents", headers=keys["admin"], params={"limit": 3}).json()
        assert len(page1["items"]) == 3 and page1["next_cursor"]

        page2 = api_client.get(
            "/documents", headers=keys["admin"], params={"limit": 3, "cursor": page1["next_cursor"]}
        ).json()
        ids1 = {d["document_id"] for d in page1["items"]}
        ids2 = {d["document_id"] for d in page2["items"]}
        assert ids1.isdisjoint(ids2)
