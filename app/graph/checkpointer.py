"""LangGraph checkpointer (D-19, Engineer Verification Task #1 — verified against
langgraph-checkpoint-postgres 3.1.1): the class is
`langgraph.checkpoint.postgres.aio.AsyncPostgresSaver`, an ASYNC CONTEXT MANAGER
(`async with AsyncPostgresSaver.from_conn_string(conn_string) as cp`), not a plain
constructor. It requires the `psycopg` (v3, with the `binary` extra) driver — a
different driver than the app's `asyncpg`/SQLAlchemy stack — so its connection strings
must be the plain `postgresql://` form, not `postgresql+asyncpg://`.

Same Postgres instance as the app, fault-tolerance only (D-19) — not a source of truth.

Bootstrap needs CREATE TABLE (`checkpointer.setup()`), which the restricted runtime
role cannot do (Step 1's least-privilege model: only the admin role has DDL). So
`setup_checkpointer_schema()` runs via DATABASE_ADMIN_URL and additionally grants the
restricted role rights on the four tables AsyncPostgresSaver creates
(`checkpoint_migrations`, `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) —
without that grant the restricted role could not read or write its own checkpoints at
runtime. Idempotent: `setup()` and the grants are both safe to call on every startup,
mirroring `qdrant_setup.ensure_collection`.

Ongoing runtime graph invocations use `build_checkpointer()`, connected as the
restricted role (DATABASE_URL) — consistent with "restricted role: runtime app".
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.engine import make_url

from app.config import get_settings

CHECKPOINT_TABLES: tuple[str, ...] = (
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
)


def to_psycopg_conn_string(sqlalchemy_url: str) -> str:
    """`postgresql+asyncpg://...` (SQLAlchemy/asyncpg) -> `postgresql://...` (psycopg)."""
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def setup_checkpointer_schema() -> None:
    """One-time (idempotent) admin bootstrap. Call at app startup (Step 12)."""
    settings = get_settings()
    admin_conn = to_psycopg_conn_string(settings.database_admin_url)
    runtime_role = make_url(settings.database_url).username
    if not runtime_role:
        raise RuntimeError("DATABASE_URL must include the restricted role username")

    async with AsyncPostgresSaver.from_conn_string(admin_conn) as checkpointer:
        await checkpointer.setup()

    role_ident = _quote_ident(runtime_role)
    table_list = ", ".join(CHECKPOINT_TABLES)
    async with await psycopg.AsyncConnection.connect(admin_conn, autocommit=True) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table_list} TO {role_ident}"
            )


@asynccontextmanager
async def build_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Runtime checkpointer, connected as the restricted role (DATABASE_URL)."""
    conn_string = to_psycopg_conn_string(get_settings().database_url)
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        yield checkpointer
