"""Tiered timeouts (PRD Section 5). Wraps the awaitable each endpoint runs (a graph
invocation or a DB call) so a slow downstream never holds a request open past its
budget. On timeout, maps to 503 — the PRD's status table doesn't name a dedicated
timeout code, and 503's "Qdrant or Postgres unreachable" framing extends naturally to
"or too slow to respond in budget"; a documented judgment call.
"""

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

from app.api.errors import ServiceUnavailableError

T = TypeVar("T")

QUERY_TIMEOUT_SECONDS = 30
INGESTION_TIMEOUT_SECONDS = 120
DELETE_UPDATE_TIMEOUT_SECONDS = 30
LISTING_TIMEOUT_SECONDS = 10


async def with_timeout(coro: Coroutine[None, None, T], *, seconds: float) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except TimeoutError as exc:
        raise ServiceUnavailableError("Request timed out") from exc
