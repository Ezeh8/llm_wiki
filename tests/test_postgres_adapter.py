"""PostgresAdapter tests.

Requires a real Postgres (UUID/JSONB/ON CONFLICT are PG-specific). Set
TEST_DATABASE_URL to an asyncpg URL for a throwaway DB; the suite is skipped when
it is unset. The full harness (in-memory/containerized PG) lands in Step 18.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.postgres import PostgresAdapter
from app.db.base import Base

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set"
)


def _utcnow():
    return datetime.now(timezone.utc)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess
    await engine.dispose()


async def test_document_crud_and_content_hash_lookup(session):
    adapter = PostgresAdapter(session)
    doc = await adapter.create_document(
        title="Onboarding",
        source_label="hr",
        file_type="pdf",
        file_size_bytes=1234,
        content_hash="hash-1",
        chunk_count=3,
    )
    assert isinstance(doc.document_id, uuid.UUID)

    fetched = await adapter.get_document(doc.document_id)
    assert fetched is not None and fetched.title == "Onboarding"

    by_hash = await adapter.get_document_by_content_hash("hash-1")
    assert by_hash is not None and by_hash.document_id == doc.document_id

    updated = await adapter.update_document(doc.document_id, chunk_count=9)
    assert updated is not None and updated.chunk_count == 9

    assert await adapter.delete_document(doc.document_id) is True
    assert await adapter.get_document(doc.document_id) is None


async def test_changelog_fk_set_null_via_document_delete(session):
    adapter = PostgresAdapter(session)
    doc = await adapter.create_document(
        title="Policy",
        source_label="ops",
        file_type="md",
        file_size_bytes=10,
        content_hash="hash-2",
        chunk_count=1,
    )
    entry = await adapter.create_changelog(
        entry="linked", actor="admin", document_id=doc.document_id
    )
    assert entry.document_id == doc.document_id

    await adapter.delete_document(doc.document_id)
    await session.flush()
    refreshed = await adapter.get_changelog(entry.changelog_id)
    await session.refresh(refreshed)
    assert refreshed is not None and refreshed.document_id is None


async def test_audit_write_is_idempotent_on_key(session):
    adapter = PostgresAdapter(session)
    first = await adapter.write_audit(
        event_type="query",
        actor="svc",
        status="success",
        idempotency_key="query_thread-1_query",
        query_text="q",
    )
    second = await adapter.write_audit(
        event_type="query",
        actor="svc",
        status="success",
        idempotency_key="query_thread-1_query",
        query_text="q",
    )
    assert first.event_id == second.event_id
    rows = await adapter.list_audit(limit=50)
    assert len([r for r in rows if r.idempotency_key == "query_thread-1_query"]) == 1


async def test_cache_write_read_expiry_and_flush(session):
    adapter = PostgresAdapter(session)
    now = _utcnow()
    await adapter.write_cache(
        cache_key="k1",
        question_text="q",
        answer="a",
        citations=[{"chunk_id": "c1"}],
        expires_at=now + timedelta(hours=1),
    )
    live = await adapter.get_cache("k1", now=now)
    assert live is not None and live.answer == "a"

    expired = await adapter.get_cache("k1", now=now + timedelta(hours=2))
    assert expired is None

    flushed = await adapter.flush_cache()
    assert flushed == 1
    assert await adapter.get_cache("k1", now=now) is None


async def test_api_key_lookup_by_hash(session):
    adapter = PostgresAdapter(session)
    created = await adapter.create_api_key(
        key_hash="abc123", tier="admin", actor_name="root"
    )
    found = await adapter.get_key_by_hash("abc123")
    assert found is not None and found.key_id == created.key_id and found.active is True
    assert await adapter.get_key_by_hash("nope") is None


async def test_upsert_document_inserts_then_updates(session):
    adapter = PostgresAdapter(session)
    doc_id = uuid.uuid4()
    first = await adapter.upsert_document(
        document_id=doc_id,
        title="v1",
        source_label="s",
        file_type="txt",
        file_size_bytes=1,
        content_hash="hash-upsert",
        chunk_count=1,
    )
    assert first.title == "v1"

    second = await adapter.upsert_document(
        document_id=doc_id,
        title="v2",
        source_label="s",
        file_type="txt",
        file_size_bytes=2,
        content_hash="hash-upsert",
        chunk_count=2,
    )
    assert second.document_id == doc_id
    assert second.title == "v2" and second.chunk_count == 2

    fetched = await adapter.get_document(doc_id)
    assert fetched is not None and fetched.title == "v2"


async def test_health_flag_set_get_defaults_false(session):
    adapter = PostgresAdapter(session)
    assert await adapter.get_health_flag("cache_stale_risk") is False

    await adapter.set_health_flag("cache_stale_risk", True)
    assert await adapter.get_health_flag("cache_stale_risk") is True

    await adapter.set_health_flag("cache_stale_risk", False)
    assert await adapter.get_health_flag("cache_stale_risk") is False
    assert await adapter.get_health_flags() == {"cache_stale_risk": False}


async def test_documents_keyset_pagination(session):
    adapter = PostgresAdapter(session)
    made = []
    for i in range(5):
        made.append(
            await adapter.create_document(
                title=f"d{i}",
                source_label="s",
                file_type="txt",
                file_size_bytes=1,
                content_hash=f"h{i}",
                chunk_count=1,
            )
        )
    await session.flush()
    page1 = await adapter.list_documents(limit=2)
    assert len(page1) == 2
    page2 = await adapter.list_documents(limit=2, cursor=page1[-1].document_id)
    assert len(page2) == 2
    ids1 = {d.document_id for d in page1}
    ids2 = {d.document_id for d in page2}
    assert ids1.isdisjoint(ids2)
