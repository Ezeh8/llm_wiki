"""Postgres adapter — the only module (besides Alembic) that touches the ORM/session.

Wraps all Postgres operations behind a provider-neutral surface (D-7). Transaction
control (commit/rollback) is owned by the caller's session lifecycle
(session-per-request, Step 12); these methods flush so DB-side effects and constraint
checks happen, but never commit. The adapter never retries — nodes own retries.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.errors import AdapterValidationError
from app.db.models import ApiKey, AuditLog, CacheEntry, Changelog, Document, HealthFlag


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class PostgresAdapter:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- documents -------------------------------------------------------------

    async def create_document(
        self,
        *,
        title: str,
        source_label: str,
        file_type: str,
        file_size_bytes: int,
        content_hash: str,
        chunk_count: int,
        document_id: uuid.UUID | str | None = None,
    ) -> Document:
        document = Document(
            title=title,
            source_label=source_label,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            content_hash=content_hash,
            chunk_count=chunk_count,
        )
        if document_id is not None:
            document.document_id = _as_uuid(document_id)
        self._session.add(document)
        await self._session.flush()
        if document.document_id is None:
            raise AdapterValidationError("create_document produced no document_id")
        return document

    async def upsert_document(
        self,
        *,
        document_id: uuid.UUID | str,
        title: str,
        source_label: str,
        file_type: str,
        file_size_bytes: int,
        content_hash: str,
        chunk_count: int,
        ingested_at: datetime | None = None,
    ) -> Document:
        """Node 12 Storer spec: "write document metadata to Postgres (upsert by
        document_id)" — a genuine upsert (not create_document's plain INSERT), so a
        Storer retry or a same-document_id re-write never raises on conflict.

        `ingested_at`: Storer (Step 9/10) generates this ONCE per write attempt and
        passes it explicitly, on both the insert and the ON CONFLICT update branch —
        so a Storer-internal retry of the SAME attempt keeps a consistent timestamp
        (not a fresh one per retry), and it stays identical to the value stamped into
        each chunk's Qdrant payload for the Ranker's recency tiebreaker (Step 10).
        """
        doc_id = _as_uuid(document_id)
        values = {
            "document_id": doc_id,
            "title": title,
            "source_label": source_label,
            "file_type": file_type,
            "file_size_bytes": file_size_bytes,
            "content_hash": content_hash,
            "chunk_count": chunk_count,
        }
        if ingested_at is not None:
            values["ingested_at"] = ingested_at
        stmt = (
            pg_insert(Document)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["document_id"],
                set_={k: v for k, v in values.items() if k != "document_id"},
            )
            .returning(Document)
        )
        # populate_existing: without it, an ORM-enabled RETURNING for a row already in
        # this session's identity map (e.g. a prior upsert of the same document_id in
        # the same session) returns the stale cached object instead of the freshly
        # updated one — a real bug caught by test_upsert_document_inserts_then_updates.
        result = await self._session.execute(stmt, execution_options={"populate_existing": True})
        document = result.scalar_one_or_none()
        if document is None:
            raise AdapterValidationError(f"upsert_document returned no row for {doc_id}")
        return document

    async def get_document(self, document_id: str | uuid.UUID) -> Document | None:
        return await self._session.get(Document, _as_uuid(document_id))

    async def get_document_by_content_hash(self, content_hash: str) -> Document | None:
        result = await self._session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self, *, limit: int = 20, cursor: str | uuid.UUID | None = None
    ) -> list[Document]:
        stmt = select(Document).order_by(
            Document.ingested_at.desc(), Document.document_id.desc()
        )
        stmt = await self._apply_cursor(
            stmt, Document, (Document.ingested_at, Document.document_id), cursor, "document_id"
        )
        result = await self._session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    async def update_document(
        self, document_id: str | uuid.UUID, **fields
    ) -> Document | None:
        document = await self.get_document(document_id)
        if document is None:
            return None
        allowed = {
            "title",
            "source_label",
            "file_type",
            "file_size_bytes",
            "content_hash",
            "chunk_count",
        }
        for key, value in fields.items():
            if key not in allowed:
                raise AdapterValidationError(f"cannot update unknown document field: {key}")
            setattr(document, key, value)
        await self._session.flush()
        return document

    async def delete_document(self, document_id: str | uuid.UUID) -> bool:
        result = await self._session.execute(
            delete(Document).where(Document.document_id == _as_uuid(document_id))
        )
        return result.rowcount > 0

    # --- changelog -------------------------------------------------------------

    async def create_changelog(
        self,
        *,
        entry: str,
        actor: str,
        document_id: str | uuid.UUID | None = None,
    ) -> Changelog:
        changelog = Changelog(
            entry=entry,
            actor=actor,
            document_id=_as_uuid(document_id) if document_id is not None else None,
        )
        self._session.add(changelog)
        await self._session.flush()
        if changelog.changelog_id is None:
            raise AdapterValidationError("create_changelog produced no changelog_id")
        return changelog

    async def get_changelog(self, changelog_id: str | uuid.UUID) -> Changelog | None:
        return await self._session.get(Changelog, _as_uuid(changelog_id))

    async def list_changelog(
        self, *, limit: int = 20, cursor: str | uuid.UUID | None = None
    ) -> list[Changelog]:
        stmt = select(Changelog).order_by(
            Changelog.created_at.desc(), Changelog.changelog_id.desc()
        )
        stmt = await self._apply_cursor(
            stmt,
            Changelog,
            (Changelog.created_at, Changelog.changelog_id),
            cursor,
            "changelog_id",
        )
        result = await self._session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    async def update_changelog(
        self,
        changelog_id: str | uuid.UUID,
        *,
        entry: str | None = None,
        document_id: str | uuid.UUID | None = None,
        set_document_null: bool = False,
    ) -> Changelog | None:
        changelog = await self.get_changelog(changelog_id)
        if changelog is None:
            return None
        if entry is not None:
            changelog.entry = entry
        if set_document_null:
            changelog.document_id = None
        elif document_id is not None:
            changelog.document_id = _as_uuid(document_id)
        await self._session.flush()
        return changelog

    async def delete_changelog(self, changelog_id: str | uuid.UUID) -> bool:
        result = await self._session.execute(
            delete(Changelog).where(Changelog.changelog_id == _as_uuid(changelog_id))
        )
        return result.rowcount > 0

    # --- audit (append-only) ---------------------------------------------------

    async def write_audit(
        self,
        *,
        event_type: str,
        actor: str,
        status: str,
        idempotency_key: str,
        document_id: str | None = None,
        query_text: str | None = None,
        chunks_retrieved: int | None = None,
        answer_text: str | None = None,
        error_detail: str | None = None,
        timestamp: datetime | None = None,
    ) -> AuditLog:
        # Insert-or-noop on the unique idempotency_key so a replayed audit write for
        # the same (thread_id + event_type) never duplicates and never errors.
        values = {
            "event_id": uuid.uuid4(),
            "event_type": event_type,
            "actor": actor,
            "status": status,
            "idempotency_key": idempotency_key,
            "document_id": document_id,
            "query_text": query_text,
            "chunks_retrieved": chunks_retrieved,
            "answer_text": answer_text,
            "error_detail": error_detail,
            "timestamp": timestamp or _utcnow(),
        }
        stmt = (
            pg_insert(AuditLog)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["idempotency_key"])
            .returning(AuditLog)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            return row
        existing = await self._session.execute(
            select(AuditLog).where(AuditLog.idempotency_key == idempotency_key)
        )
        row = existing.scalar_one_or_none()
        if row is None:
            raise AdapterValidationError(
                f"audit write neither inserted nor found key={idempotency_key}"
            )
        return row

    async def get_audit(self, event_id: str | uuid.UUID) -> AuditLog | None:
        return await self._session.get(AuditLog, _as_uuid(event_id))

    async def list_audit(
        self, *, limit: int = 20, cursor: str | uuid.UUID | None = None
    ) -> list[AuditLog]:
        stmt = select(AuditLog).order_by(
            AuditLog.timestamp.desc(), AuditLog.event_id.desc()
        )
        stmt = await self._apply_cursor(
            stmt, AuditLog, (AuditLog.timestamp, AuditLog.event_id), cursor, "event_id"
        )
        result = await self._session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    # --- cache -----------------------------------------------------------------

    async def get_cache(
        self, cache_key: str, *, now: datetime | None = None
    ) -> CacheEntry | None:
        # Expired rows are not served (stale cache self-heals on TTL expiry).
        entry = await self._session.get(CacheEntry, cache_key)
        if entry is None:
            return None
        if entry.expires_at <= (now or _utcnow()):
            return None
        return entry

    async def write_cache(
        self,
        *,
        cache_key: str,
        question_text: str,
        answer: str,
        citations: list | dict,
        expires_at: datetime,
        created_at: datetime | None = None,
    ) -> CacheEntry:
        created = created_at or _utcnow()
        stmt = (
            pg_insert(CacheEntry)
            .values(
                cache_key=cache_key,
                question_text=question_text,
                answer=answer,
                citations=citations,
                created_at=created,
                expires_at=expires_at,
            )
            .on_conflict_do_update(
                index_elements=["cache_key"],
                set_={
                    "question_text": question_text,
                    "answer": answer,
                    "citations": citations,
                    "created_at": created,
                    "expires_at": expires_at,
                },
            )
            .returning(CacheEntry)
        )
        # populate_existing: same identity-map staleness fix as upsert_document above —
        # re-caching an already-session-tracked cache_key must return the fresh row.
        result = await self._session.execute(stmt, execution_options={"populate_existing": True})
        entry = result.scalar_one_or_none()
        if entry is None:
            raise AdapterValidationError(f"cache write returned no row for {cache_key}")
        return entry

    async def flush_cache(self) -> int:
        result = await self._session.execute(delete(CacheEntry))
        return result.rowcount or 0

    # --- api keys --------------------------------------------------------------

    async def get_key_by_hash(self, key_hash: str) -> ApiKey | None:
        # Returns regardless of the active flag; the auth layer (Step 12) checks
        # active and returns an identical 401 for every failure mode.
        result = await self._session.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash)
        )
        return result.scalar_one_or_none()

    async def create_api_key(
        self, *, key_hash: str, tier: str, actor_name: str, active: bool = True
    ) -> ApiKey:
        api_key = ApiKey(
            key_hash=key_hash, tier=tier, actor_name=actor_name, active=active
        )
        self._session.add(api_key)
        await self._session.flush()
        if api_key.key_id is None:
            raise AdapterValidationError("create_api_key produced no key_id")
        return api_key

    async def set_api_key_active(
        self, key_id: str | uuid.UUID, *, active: bool
    ) -> ApiKey | None:
        api_key = await self._session.get(ApiKey, _as_uuid(key_id))
        if api_key is None:
            return None
        api_key.active = active
        await self._session.flush()
        return api_key

    # --- health flags ------------------------------------------------------------

    async def set_health_flag(self, name: str, value: bool) -> None:
        """Durable operational flags (e.g. cache_stale_risk, D-16/D-24 fortifications).
        Postgres-backed rather than a file, unlike the dead-letter system (Step 14) —
        the PRD only calls out a persistent volume for dead-letter, and everything else
        in this app is Postgres-backed, so a small table is the consistent choice."""
        stmt = (
            pg_insert(HealthFlag)
            .values(name=name, value=value, updated_at=_utcnow())
            .on_conflict_do_update(
                index_elements=["name"], set_={"value": value, "updated_at": _utcnow()}
            )
        )
        await self._session.execute(stmt)

    async def get_health_flag(self, name: str) -> bool:
        flag = await self._session.get(HealthFlag, name)
        return flag.value if flag is not None else False

    async def get_health_flags(self) -> dict[str, bool]:
        result = await self._session.execute(select(HealthFlag))
        return {row.name: row.value for row in result.scalars().all()}

    # --- helpers ---------------------------------------------------------------

    async def _apply_cursor(self, stmt, model, sort_cols, cursor, pk_name):
        if cursor is None:
            return stmt
        pk_col = getattr(model, pk_name)
        anchor = await self._session.execute(
            select(*sort_cols).where(pk_col == _as_uuid(cursor))
        )
        row = anchor.first()
        if row is None:
            raise AdapterValidationError(f"pagination cursor not found: {cursor}")
        return stmt.where(tuple_(*sort_cols) < tuple_(*[row[0], row[1]]))
