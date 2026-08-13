"""Shared node-layer helpers.

Adapters never retry (D-7) — retry policy belongs to the node layer, at whatever count
the PRD assigns each node (e.g. Duplicate Checker 1, Retriever 1, Audit Writer 3, Cache
Checker/Writer 0 — best-effort, so those nodes simply don't call this at all).
"""

from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.adapters.postgres import PostgresAdapter
from app.db.session import async_session_factory

T = TypeVar("T")


async def call_with_retry(fn: Callable[[], Awaitable[T]], *, retries: int) -> T:
    attempts = retries + 1
    last_exc: Exception | None = None
    for _attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 — re-raised below once retries are exhausted
            last_exc = exc
    assert last_exc is not None
    raise last_exc


async def flush_cache_side_effect() -> None:
    """Shared by Storer (Step 9) and Deleter (Step 11) — both mutate the knowledge
    base and must invalidate all cached answers afterward (D-16). A side effect, not a
    gate: 1 retry, and a failure never fails the write/delete it's attached to — it
    only sets the `cache_stale_risk` health flag (D-16/D-24). A later successful flush
    clears the flag again (best-effort; the PRD only says "set" it on failure, but
    leaving it permanently true after one transient blip would make the health
    endpoint reflect history instead of current state)."""

    async def _flush():
        async with async_session_factory() as session:
            await PostgresAdapter(session).flush_cache()
            await session.commit()

    async def _set_flag(value: bool) -> None:
        try:
            async with async_session_factory() as session:
                await PostgresAdapter(session).set_health_flag("cache_stale_risk", value)
                await session.commit()
        except Exception:
            pass  # best-effort — a flag-write failure must not fail the caller either

    try:
        await call_with_retry(_flush, retries=1)
        await _set_flag(False)
    except Exception:
        await _set_flag(True)
