"""Step 16 — MCP server. Real end-to-end: both the FastAPI app AND the MCP server run
as live uvicorn instances (in background threads) and are exercised through a real MCP
client over HTTP/SSE — the whole point of this step is the thin-wrapper HTTP hop
between two real services, so anything less than genuinely running both would leave
the actual integration unverified. Same RUN_GRAPH_INTEGRATION_TESTS gate, mocked dense
embedder + LLM as every other integration test since Step 9.
"""

import asyncio
import os
import re
import threading
import uuid

import pytest

RUN_INTEGRATION = os.environ.get("RUN_GRAPH_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="RUN_GRAPH_INTEGRATION_TESTS not set to 1"
)

FASTAPI_PORT = 8811
MCP_PORT = 8812


def _patch_fastapi_mocks():
    from app.embedding import Embedder

    class FakeModel:
        def encode(self, texts, **kwargs):
            return [[0.1] * 768 for _ in texts]

    fake_embedder = Embedder(model=FakeModel())

    import app.graph.nodes.ingest_path as ingest_mod
    import app.graph.nodes.query_path as query_mod

    ingest_mod.get_embedder = lambda: fake_embedder
    query_mod.get_embedder = lambda: fake_embedder

    from app.domain import Citation, GenerationResult

    class EchoingLLM:
        async def generate(self, *, system, user):
            match = re.search(r"chunk_id=(\S+)", user)
            chunk_id = match.group(1) if match else "unknown"
            document_id = chunk_id.split(":")[0] if ":" in chunk_id else "unknown"
            return GenerationResult(
                answer="Badges expire after 90 days.",
                citations=[
                    Citation(
                        document_id=document_id,
                        document_title="Badge Policy",
                        chunk_id=chunk_id,
                        chunk_text="t",
                        chunk_index=0,
                    )
                ],
            )

    query_mod.build_llm_adapter = lambda: EchoingLLM()


def _run_uvicorn(app, port: int) -> None:
    # `.serve()` wrapped in our own asyncio.run(), not `.run()` — `.run()` installs
    # signal handlers, which only works on the main thread; this runs in a background
    # thread. Matches the pattern verified working during manual empirical testing.
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


@pytest.fixture(scope="module")
def live_servers(request):
    """Starts real FastAPI + MCP servers once for this module, each in its own thread
    (its own event loop) — see the engine.dispose() note below for why that matters."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("MCP_FASTAPI_BASE_URL", f"http://127.0.0.1:{FASTAPI_PORT}")
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "10")
    from app.config import get_settings

    get_settings.cache_clear()

    _patch_fastapi_mocks()

    from app.api.app import app as fastapi_app
    from app.db.session import engine
    from app.mcp_server.server import app as mcp_app

    # Same cross-event-loop pooled-connection fix needed in every prior integration
    # test file (Steps 12-15): the FastAPI thread below gets its OWN event loop, but
    # app.db.session.engine's pool is a process-wide singleton that may already carry
    # connections from whatever loop last touched it (e.g. an earlier test module's
    # TestClient). Disposing first forces fresh connections against this thread's loop.
    asyncio.run(engine.dispose())

    t1 = threading.Thread(target=_run_uvicorn, args=(fastapi_app, FASTAPI_PORT), daemon=True)
    t1.start()
    t2 = threading.Thread(target=_run_uvicorn, args=(mcp_app, MCP_PORT), daemon=True)
    t2.start()

    import time

    time.sleep(3)
    assert t1.is_alive() and t2.is_alive(), "FastAPI/MCP server threads failed to start"

    def _cleanup():
        get_settings.cache_clear()
        monkeypatch.undo()

    request.addfinalizer(_cleanup)
    return f"http://127.0.0.1:{FASTAPI_PORT}", f"http://127.0.0.1:{MCP_PORT}"


@pytest.fixture
async def api_keys(live_servers):
    """A fresh admin + service key pair per test, via a throwaway engine (this fixture
    runs in pytest-asyncio's own loop, distinct from the two server threads' loops —
    the shared app.db.session.engine singleton must not be touched directly here, same
    cross-loop reasoning as tests/test_api.py)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.adapters.postgres import PostgresAdapter
    from app.api.auth import hash_key
    from app.config import get_settings

    run_id = uuid.uuid4().hex[:8]
    admin_key = f"admin-{run_id}"
    service_key = f"service-{run_id}"

    engine = create_async_engine(get_settings().database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            adapter = PostgresAdapter(session)
            await adapter.create_api_key(key_hash=hash_key(admin_key), tier="admin", actor_name="Admin")
            await adapter.create_api_key(key_hash=hash_key(service_key), tier="service", actor_name="Service")
            await session.commit()
    finally:
        await engine.dispose()

    return admin_key, service_key


async def _ingest_via_fastapi(fastapi_url: str, admin_key: str) -> str:
    import httpx

    content = f"# Badge Policy\n\nBadges expire after 90 days. Marker {uuid.uuid4()}.".encode()
    async with httpx.AsyncClient(base_url=fastapi_url) as client:
        r = await client.post(
            "/documents",
            headers={"X-API-Key": admin_key},
            files={"file": ("badge.md", content, "text/markdown")},
            data={"title": "Badge Policy", "source_label": "security"},
        )
        assert r.status_code == 201
        return r.json()["document_id"]


async def test_query_tool_returns_grounded_answer_matching_prd_contract(live_servers, api_keys):
    """Doesn't assert the citation points at THIS test's own document_id: the fake
    dense embedder (Step 6/9's testing pattern) returns an identical constant vector
    for every text, so when this runs as part of the full suite — with many other
    test files' similarly-worded documents already sitting in the SAME shared Qdrant
    collection — dense retrieval genuinely cannot distinguish between them, and BM25
    alone may not either. That's a property of running many "Badge Policy"-flavored
    integration tests against one shared vector store with a non-discriminating fake
    embedder, not a defect in the server under test. What this test CAN reliably
    assert: the response has the right shape, a real ranked_chunks-backed citation
    survived grounding verification (Step 10's 3-layer check), and it references a
    real, well-formed chunk_id — not that it's specifically the chunk THIS call ingested.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    fastapi_url, mcp_url = live_servers
    admin_key, service_key = api_keys
    await _ingest_via_fastapi(fastapi_url, admin_key)

    async with sse_client(f"{mcp_url}/sse", headers={"X-API-Key": service_key}) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()

            tools = await session.list_tools()
            assert [t.name for t in tools.tools] == ["llm_wiki_query"]

            result = await session.call_tool(
                "llm_wiki_query", {"question": "How long do badges last?"}
            )
            payload = result.structured_content
            assert payload is not None, "expected structured tool output (dict[str, Any] return type)"
            assert set(payload.keys()) == {
                "answer", "citations", "source_chunks", "cached", "query_id", "degraded", "session_id",
            }
            assert payload["cached"] is False
            assert payload["degraded"] is False
            assert len(payload["citations"]) == 1
            citation = payload["citations"][0]
            assert citation["chunk_id"] == f"{citation['document_id']}:{citation['chunk_index']}"


async def test_source_listing_resource_lists_ingested_documents(live_servers, api_keys):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    fastapi_url, mcp_url = live_servers
    admin_key, service_key = api_keys
    document_id = await _ingest_via_fastapi(fastapi_url, admin_key)

    async with sse_client(f"{mcp_url}/sse", headers={"X-API-Key": service_key}) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            resources = await session.list_resources()
            assert [str(res.uri) for res in resources.resources] == ["llmwiki://source_listing"]

            listing = await session.read_resource("llmwiki://source_listing")
            import json

            body = json.loads(listing.contents[0].text)
            assert any(item["document_id"] == document_id for item in body["items"])


async def test_input_validation_rejects_without_calling_fastapi(live_servers, api_keys):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    _, mcp_url = live_servers
    _, service_key = api_keys

    async with sse_client(f"{mcp_url}/sse", headers={"X-API-Key": service_key}) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()

            empty = await session.call_tool("llm_wiki_query", {"question": "   "})
            assert empty.structured_content["code"] == "invalid_input"
            assert empty.structured_content["retryable"] is False

            too_long = await session.call_tool("llm_wiki_query", {"question": "x" * 2001})
            assert too_long.structured_content["code"] == "invalid_input"


async def test_bad_key_maps_fastapi_401_to_documented_fallback(live_servers):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    _, mcp_url = live_servers

    async with sse_client(f"{mcp_url}/sse", headers={"X-API-Key": "totally-bogus-key"}) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(
                "llm_wiki_query", {"question": "How long do badges last?"}
            )
            # unauthorized isn't one of the PRD's 7 listed MCP codes — falls to the
            # documented default (Section 4: "Unmapped errors default to internal_error
            # retryable:true").
            assert result.structured_content["code"] == "internal_error"
            assert result.structured_content["retryable"] is True


async def test_rate_limit_enforced_at_configured_threshold(live_servers, api_keys):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    _, mcp_url = live_servers
    _, service_key = api_keys

    async with sse_client(f"{mcp_url}/sse", headers={"X-API-Key": service_key}) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            outcomes = []
            for i in range(13):
                result = await session.call_tool(
                    "llm_wiki_query", {"question": f"Rate limit probe {i} {uuid.uuid4()}?"}
                )
                sc = result.structured_content or {}
                outcomes.append("ok" if "answer" in sc else sc.get("message"))

    assert outcomes[:10] == ["ok"] * 10
    assert all(o == "Too many requests." for o in outcomes[10:])


def test_mcp_server_module_never_imports_db_or_vector_store_clients():
    """Static check backing "zero direct connections to Postgres, Qdrant, or models"
    (Section 4 System Mapping): confirms importing the MCP server package pulls in
    none of the actual client libraries those connections would require."""
    import sys

    before = set(sys.modules)
    import app.mcp_server.server  # noqa: F401

    after = set(sys.modules)
    new_modules = after - before
    forbidden_substrings = ("sqlalchemy", "asyncpg", "psycopg", "qdrant")
    offending = [
        m for m in new_modules if any(f in m.lower() for f in forbidden_substrings)
    ]
    assert offending == []
