import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    source_label: Mapped[str] = mapped_column(String(100), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class Changelog(Base):
    __tablename__ = "changelog"

    changelog_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # entry max-length (1000) is enforced at the Pydantic layer; DB column is plain text.
    entry: Mapped[str] = mapped_column(Text, nullable=False)
    # SET NULL on document delete — changelog entries survive with a blank link (D-4).
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.document_id", ondelete="SET NULL"),
        nullable=True,
    )
    actor: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


class AuditLog(Base):
    __tablename__ = "audit_log"
    # Append-only: the restricted runtime role is granted SELECT + INSERT only
    # (no UPDATE/DELETE) — enforced at the DB privilege level in the migration.
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_audit_log_idempotency_key"),)

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    document_id: Mapped[str | None] = mapped_column(String, nullable=True)
    query_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunks_retrieved: Mapped[int | None] = mapped_column(Integer, nullable=True)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Concatenation of thread_id + event_type; unique to make audit writes idempotent.
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False)


class CacheEntry(Base):
    __tablename__ = "cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list | dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class HealthFlag(Base):
    """Durable operational health flags (e.g. cache_stale_risk — D-16/D-24). Read by
    the health endpoint (Step 15); set by Storer/Deleter's cache-flush side effect
    (Step 9/11). Not part of the PRD's Section 5 table list — added in Step 9 as the
    storage mechanism for a behavior the PRD specifies without naming one; see
    BUILD_LOG."""

    __tablename__ = "health_flags"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    tier: Mapped[str] = mapped_column(String, nullable=False)  # admin / service / employee
    actor_name: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
