"""Thin HTTP client for calling the FastAPI backend (PRD Section 4 "System mapping:
Thin wrapper around FastAPI. Zero direct connections to Postgres, Qdrant, or models.
MCP calls FastAPI over HTTP only."). This module is the ONLY place in the mcp_server
package that makes network calls — no app.db/app.adapters/app.graph imports anywhere
in this package, by design.
"""

from typing import Any

import httpx

from app.config import get_settings

_LISTING_TIMEOUT_SECONDS = 10.0  # matches FastAPI's own listing tier (Step 12)


class FastApiResponse:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self.body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def error_code(self) -> str | None:
        return self.body.get("error") if not self.ok else None


async def query(api_key: str, *, question: str, filter: dict | None, session_id: str | None) -> FastApiResponse:
    settings = get_settings()
    payload: dict[str, Any] = {"question": question}
    if filter:
        payload["filter"] = filter
    if session_id:
        payload["session_id"] = session_id

    async with httpx.AsyncClient(
        base_url=settings.mcp_fastapi_base_url, timeout=settings.mcp_query_timeout_seconds
    ) as client:
        response = await client.post("/query", json=payload, headers={"X-API-Key": api_key})
    return FastApiResponse(response.status_code, response.json())


async def list_documents(api_key: str) -> FastApiResponse:
    settings = get_settings()
    async with httpx.AsyncClient(
        base_url=settings.mcp_fastapi_base_url, timeout=_LISTING_TIMEOUT_SECONDS
    ) as client:
        response = await client.get("/documents", headers={"X-API-Key": api_key})
    return FastApiResponse(response.status_code, response.json())
