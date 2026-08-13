"""initial schema: tables, restricted runtime role, grants

Revision ID: 0001
Revises:
Create Date: 2026-08-06

Creates the five core tables and provisions the least-privilege runtime role.

Role model (two roles, two privilege levels):
  - The admin/migration role is the identity Alembic connects as
    (DATABASE_ADMIN_URL) and owns every object it creates, so it alone can
    CREATE/ALTER/DROP. It is provisioned once outside migrations (see BUILD_LOG).
  - The restricted runtime role is created here. Its name and password are read
    from DATABASE_URL so the whole setup is reproducible from the two env vars.
    It gets SELECT/INSERT/UPDATE/DELETE on the mutable tables and SELECT/INSERT
    ONLY on audit_log (append-only). It is never granted ownership, TRUNCATE, or
    DDL, so it cannot DROP or TRUNCATE anything.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import make_url

from app.config import get_settings

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_RW_TABLES = ("documents", "changelog", "cache", "api_keys")


def _runtime_role() -> tuple[str, str, str]:
    url = make_url(get_settings().database_url)
    if not url.username or url.password is None:
        raise RuntimeError("DATABASE_URL must include the restricted role username and password")
    if '"' in url.username:
        raise RuntimeError("restricted role name must not contain double quotes")
    return url.username, url.password, url.database


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _create_tables() -> None:
    op.create_table(
        "documents",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("source_label", sa.String(length=100), nullable=False),
        sa.Column("file_type", sa.String(length=10), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("document_id", name="pk_documents"),
        sa.UniqueConstraint("content_hash", name="uq_documents_content_hash"),
    )

    op.create_table(
        "changelog",
        sa.Column("changelog_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entry", sa.Text(), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("changelog_id", name="pk_changelog"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            name="fk_changelog_document_id_documents",
            ondelete="SET NULL",
        ),
    )

    op.create_table(
        "audit_log",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("chunks_retrieved", sa.Integer(), nullable=True),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("event_id", name="pk_audit_log"),
        sa.UniqueConstraint("idempotency_key", name="uq_audit_log_idempotency_key"),
    )

    op.create_table(
        "cache",
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("cache_key", name="pk_cache"),
    )

    op.create_table(
        "api_keys",
        sa.Column("key_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("tier", sa.String(), nullable=False),
        sa.Column("actor_name", sa.String(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key_id", name="pk_api_keys"),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )


def _provision_runtime_role() -> None:
    role, password, database = _runtime_role()
    role_ident = _quote_ident(role)
    password_literal = "'" + password.replace("'", "''") + "'"

    op.execute(
        sa.text(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
                    CREATE ROLE {role_ident} LOGIN PASSWORD {password_literal}
                        NOSUPERUSER NOCREATEDB NOCREATEROLE;
                END IF;
            END
            $$
            """
        )
    )

    if database:
        op.execute(f"GRANT CONNECT ON DATABASE {_quote_ident(database)} TO {role_ident}")
    op.execute(f"GRANT USAGE ON SCHEMA public TO {role_ident}")

    rw_list = ", ".join(_RW_TABLES)
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {rw_list} TO {role_ident}")
    # audit_log is append-only — INSERT/SELECT only, deliberately no UPDATE/DELETE.
    op.execute(f"GRANT SELECT, INSERT ON audit_log TO {role_ident}")


def upgrade() -> None:
    _create_tables()
    _provision_runtime_role()


def downgrade() -> None:
    role, _password, database = _runtime_role()
    role_ident = _quote_ident(role)

    op.execute(f"REVOKE ALL ON audit_log FROM {role_ident}")
    rw_list = ", ".join(_RW_TABLES)
    op.execute(f"REVOKE ALL ON {rw_list} FROM {role_ident}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {role_ident}")
    if database:
        op.execute(f"REVOKE CONNECT ON DATABASE {_quote_ident(database)} FROM {role_ident}")

    op.drop_table("api_keys")
    op.drop_table("cache")
    op.drop_table("audit_log")
    op.drop_table("changelog")
    op.drop_table("documents")

    op.execute(f"DROP ROLE IF EXISTS {role_ident}")
