"""Rate limiting + structured request logging (PRD Section 5 "Middleware Order":
1. Rate limiter, 2. Auth, 3. Request logging, 4. Endpoint logic).

Auth is implemented as a FastAPI dependency, not ASGI middleware (see auth.py's
docstring for why). To still realize the PRD's ordering with that split:
- RateLimitMiddleware must be the OUTERMOST layer, so it sees every request first —
  including ones that will later fail auth or 404 — genuinely "rejecting floods before
  doing any work."
- RequestLoggingMiddleware wraps everything downstream of it (auth dependency +
  endpoint), so by the time it logs the response, `request.state.actor` has already
  been set by the auth dependency if auth succeeded (or is absent if it didn't).

Starlette applies the LAST-added middleware OUTERMOST, so app.py must call
`add_middleware(RequestLoggingMiddleware)` BEFORE `add_middleware(RateLimitMiddleware)`
for the actual request flow to be RateLimit -> [Auth dependency, inside routing] ->
Logging -> endpoint. Verified live in Step 12's tests.
"""

import json
import logging
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.adapters.postgres import PostgresAdapter
from app.api.auth import API_KEY_HEADER, hash_key
from app.config import get_settings
from app.db.session import async_session_factory

logger = logging.getLogger("llm_wiki.api")

# Health is public and explicitly unlimited (PRD Section 5 Rate Limits).
_UNLIMITED_PATHS = {"/health"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window limiter, per key_hash, bucketed by tier (30/10/10 per
    minute default, configurable). Single-process, in-memory counters are sufficient
    for this ICL/single-tenant deployment — no Redis needed (D-29 defers
    multi-tenancy/scale-out to v2 anyway).

    Does its own lightweight key lookup and stashes the result on
    `request.state.api_key` so the auth dependency downstream reuses it instead of
    querying twice. A request with a missing/unrecognized key is NOT rejected here
    (that is the auth dependency's job, with its identical-401 body) — it's rate
    limited at the lowest configured tier's bucket instead, so garbage/unauthenticated
    requests still can't flood the database with lookups.
    """

    def __init__(self, app) -> None:
        super().__init__(app)
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _UNLIMITED_PATHS:
            return await call_next(request)

        settings = get_settings()
        raw_key = request.headers.get(API_KEY_HEADER)
        bucket_key = "anonymous"
        limit = min(settings.rate_limit_admin, settings.rate_limit_service, settings.rate_limit_employee)

        if raw_key:
            key_hash = hash_key(raw_key)
            bucket_key = key_hash
            async with async_session_factory() as session:
                api_key = await PostgresAdapter(session).get_key_by_hash(key_hash)
            request.state.api_key = api_key
            if api_key is not None:
                limit = {
                    "admin": settings.rate_limit_admin,
                    "service": settings.rate_limit_service,
                    "employee": settings.rate_limit_employee,
                }.get(api_key.tier, limit)

        now = time.monotonic()
        window = self._buckets[bucket_key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "status": 429,
                    "error": "rate_limited",
                    "message": "Too many requests",
                    "retryable": True,
                },
            )
        window.append(now)

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured JSON per request: method, path, status, duration, actor, key
    prefix. Never logs full API keys (first 4 chars only), file contents, embeddings,
    or raw answers (Section 6)."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        raw_key = request.headers.get(API_KEY_HEADER, "")
        logger.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                    "actor": getattr(request.state, "actor", None),
                    "key_prefix": raw_key[:4] if raw_key else None,
                }
            )
        )
        return response
