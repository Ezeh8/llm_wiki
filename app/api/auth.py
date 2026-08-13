"""API key auth (PRD Section 5 "Auth Flow" / Section 6 "Security Decisions").

Single api_keys table (Seam 5 — admin/service/employee all in one table). key_hash =
SHA-256 of the raw key; the raw key itself is never stored or logged (only its first 4
chars, for request logs — middleware.py). All 401s use the identical body regardless
of cause: missing key, invalid key, wrong tier for the route, or a revoked (inactive)
key.

Implemented as a FastAPI dependency (`require_tier(...)`), not raw ASGI middleware —
tier-per-route naturally fits FastAPI's per-route dependency injection, and
reimplementing route-pattern matching inside middleware would be strictly worse for
no benefit. This still satisfies the PRD's "reject unauthorized before processing"
ordering intent: dependencies resolve before the endpoint body runs, and
RequestLoggingMiddleware (middleware.py) wraps the whole call chain so it always logs
the real outcome (including 401s) after the fact. See middleware.py's docstring for
the full ordering reasoning.
"""

import hashlib

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.postgres import PostgresAdapter
from app.api.errors import UnauthorizedError
from app.db.models import ApiKey
from app.db.session import get_session

API_KEY_HEADER = "X-API-Key"

_UNSET = object()


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def _lookup_key(request: Request, session: AsyncSession) -> ApiKey | None:
    # Reuses RateLimitMiddleware's lookup when it already ran (stashed on
    # request.state.api_key) so the same key isn't queried twice per request.
    cached = getattr(request.state, "api_key", _UNSET)
    if cached is not _UNSET:
        return cached
    raw_key = request.headers.get(API_KEY_HEADER)
    if not raw_key:
        request.state.api_key = None
        return None
    key = await PostgresAdapter(session).get_key_by_hash(hash_key(raw_key))
    request.state.api_key = key
    return key


def require_tier(allowed_tiers: set[str]):
    """Dependency factory — `Depends(require_tier({"admin", "service"}))` per route,
    matching the Section 5 Routes table's per-endpoint tier column exactly."""

    async def _dependency(
        request: Request, session: AsyncSession = Depends(get_session)
    ) -> ApiKey:
        key = await _lookup_key(request, session)
        if key is None or not key.active or key.tier not in allowed_tiers:
            raise UnauthorizedError()
        # Stashed for RequestLoggingMiddleware and for endpoints that need the actor
        # for audit writes.
        request.state.actor = key.actor_name
        return key

    return _dependency
