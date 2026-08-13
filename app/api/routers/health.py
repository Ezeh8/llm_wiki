"""Health route (PRD Section 5): public, no auth, unlimited rate (middleware.py
exempts it explicitly). All 7 flags real as of Step 15.

**`embedding_model_available` — the one flag that needed real design thought.** The
PRD's own Success Criteria says the health endpoint must respond "under 2 seconds
(including the embedding model check)" — but Nomic is a ~500MB-1GB CPU model
(Crash Risk #1) that can take far longer than 2s to cold-load, and a health check
should not itself trigger a multi-GB download as a side effect anyway. Resolution:
  - If the model has NEVER been loaded in this process yet (`Embedder.is_loaded` is
    False — Step 6's lazy-singleton flag, made public in Step 15), report `True`
    WITHOUT attempting a load. This is "no failure has been observed," not "actively
    confirmed" — a judgment call, but the alternative (attempting a cold load on the
    health path) both blows the 2s budget on a fresh deploy's very first health check
    and makes health-checking itself responsible for triggering the download.
  - If the model IS already loaded (a real query has happened before, or it warmed up
    some other way), check `Embedder.recently_failed` first — has a REAL embed call
    (query or ingest traffic) failed in the last 60 seconds? (Step 20: added after a
    real incident where a synthetic "ping" probe kept passing while every real query
    failed — a fixed-length probe isn't representative of real query lengths, and can
    miss a failure mode tied to input length.) If no recent real failure, fall back to
    one real, cheap `embed_query("ping")` bounded by a `asyncio.wait_for` timeout well
    under the 2s budget — this second check catches a different failure mode
    (stuck/hung model that hasn't been touched by real traffic yet), which
    failure-tracking alone can't see.
"""

import asyncio

from fastapi import APIRouter
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text

from app.adapters.postgres import PostgresAdapter
from app.api.schemas import HealthResponse
from app.config import get_settings
from app.db.session import async_session_factory
from app.dead_letter import has_backlog, has_poisoned
from app.embedding import get_embedder

router = APIRouter(tags=["health"])

_EMBEDDING_CHECK_TIMEOUT_SECONDS = 1.5


async def _postgres_connected() -> bool:
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _qdrant_connected() -> bool:
    client = AsyncQdrantClient(url=get_settings().qdrant_url)
    try:
        await client.get_collections()
        return True
    except Exception:
        return False
    finally:
        await client.close()


async def _cache_stale_risk() -> bool:
    try:
        async with async_session_factory() as session:
            return await PostgresAdapter(session).get_health_flag("cache_stale_risk")
    except Exception:
        # Can't confirm the flag is clear — treat conservatively as at-risk.
        return True


async def _embedding_model_available() -> bool:
    embedder = get_embedder()
    if not embedder.is_loaded:
        return True  # not yet touched in this process — optimistic, see module docstring
    # Step 20 finding: a synthetic "ping" probe isn't representative of real query
    # length. A real production failure (a rotary-embedding cache mismatch triggered by
    # real traffic) left every real query failing while this same probe kept passing —
    # a short, fixed string happened to avoid whatever sequence length was corrupted.
    # Checked first, before the probe: has a REAL embed call (query or ingest traffic)
    # actually failed recently? See Embedder.recently_failed's docstring for why this
    # is a timestamped window, not a single flag, race-safe under concurrent calls.
    if embedder.recently_failed:
        return False
    # Real traffic hasn't shown a failure — still run the bounded probe as a second,
    # different check: a model that's warm but hasn't been touched by real traffic yet
    # could still be stuck/hung, which failure-tracking alone can't see.
    try:
        await asyncio.wait_for(
            embedder.embed_query("ping"), timeout=_EMBEDDING_CHECK_TIMEOUT_SECONDS
        )
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    postgres_connected = await _postgres_connected()
    qdrant_connected = await _qdrant_connected()
    cache_stale_risk = await _cache_stale_risk()
    audit_backlog = has_backlog()
    audit_poisoned = has_poisoned()
    embedding_model_available = await _embedding_model_available()

    healthy = (
        postgres_connected
        and qdrant_connected
        and not cache_stale_risk
        and not audit_backlog
        and not audit_poisoned
        and embedding_model_available
    )
    return HealthResponse(
        status="healthy" if healthy else "degraded",
        audit_backlog=audit_backlog,
        audit_poisoned=audit_poisoned,
        cache_stale_risk=cache_stale_risk,
        qdrant_connected=qdrant_connected,
        postgres_connected=postgres_connected,
        embedding_model_available=embedding_model_available,
    )
