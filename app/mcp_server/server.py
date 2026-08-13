"""MCP server (PRD Section 4). Thin HTTP wrapper: one tool (llm_wiki_query), one
resource (source_listing), both delegate to FastAPI via client.py — this package never
imports app.db/app.adapters/app.graph, matching "zero direct connections to Postgres,
Qdrant, or models" (System Mapping). Transport is HTTP/SSE (locked, MCP-2).

**Auth design, verified empirically against the installed SDK (mcp 2.0) before
committing to it** — see BUILD_LOG for the exact probes run: static `@server.resource`
handlers cannot receive an injected `Context` (the SDK raises `ValueError` at
registration time: "Context injection for static resources is not supported"), so
per-tool `Context.headers` access (which DOES work) can't be the only mechanism —
the resource needs auth too. A `ServerMiddleware` intercepts every inbound request
(`tools/call`, `resources/read`, everything) uniformly BEFORE it reaches either
handler, reading the real header off `ctx.request` (a Starlette `Request` for the
HTTP/SSE transport) and stashing it in a `ContextVar` — visible to whichever handler
runs next in the same async call stack. MCP does not itself validate the key (that
would mean a second Postgres connection); it just forwards it to FastAPI on every
call and translates whatever FastAPI decides (including a 401) into the MCP error
shape (errors.py) — true auth enforcement stays 100% FastAPI's job, MCP only relays.

Rate limiting (Section 4 Security, "10 requests/minute per API key, configurable"):
an in-memory sliding window per key, same pattern as FastAPI's own limiter (Step 12)
but a single flat rate (MCP-3: "single tier, read-only" — no admin/service/employee
distinction here) — separate from, and enforced in addition to, whatever FastAPI's
own per-tier limiter decides on the forwarded call.

**Return type note, found by testing, not by reading docs:** the tool is annotated
`-> dict[str, Any]`, not bare `-> dict`. The SDK only derives a structured-output JSON
schema (populating `CallToolResult.structured_content`, not just the text fallback)
for a `dict` annotation that carries its value-type argument; a bare `dict` silently
produces text-only output. Confirmed by probing both empirically before settling here
— MCP clients that want typed structured output (rather than parsing the text blob)
need the `dict[str, Any]` form.
"""

import json
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from app.config import get_settings
from app.mcp_server import client
from app.mcp_server.errors import invalid_input, map_listing_error, map_query_error, rate_limited

MAX_QUESTION_LENGTH = 2000

_api_key_var: ContextVar[str | None] = ContextVar("mcp_api_key", default=None)
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _is_rate_limited(key: str) -> bool:
    settings = get_settings()
    now = time.monotonic()
    window = _rate_buckets[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.mcp_rate_limit_per_minute:
        return True
    window.append(now)
    return False


async def _auth_middleware(ctx, call_next):
    request = ctx.request
    headers = getattr(request, "headers", None)
    api_key = headers.get("x-api-key") if headers is not None else None
    token = _api_key_var.set(api_key)
    try:
        return await call_next(ctx)
    finally:
        _api_key_var.reset(token)


def build_server() -> MCPServer:
    settings = get_settings()
    server = MCPServer(name="llm-wiki", version=settings.app_version)
    server.middleware.append(_auth_middleware)

    class QueryFilter(BaseModel):
        source_label: str | None = None
        document_id: str | None = None

    @server.tool(
        name="llm_wiki_query",
        description=(
            "Ask a question against the LLM Wiki knowledge base and get a grounded, "
            "cited answer sourced from the organization's ingested SOPs and policies. "
            "Read-only — does not modify the knowledge base."
        ),
    )
    async def llm_wiki_query(
        question: str, filter: QueryFilter | None = None
    ) -> dict[str, Any]:
        """PRD Section 4 capability contract: input {question, filter?}; output
        {answer, citations, source_chunks, cached, query_id, degraded, session_id} on
        success, or {code, message, retryable} on failure (Section 4 Error Shape)."""
        api_key = _api_key_var.get()
        if not api_key:
            return dict(invalid_input("Missing X-API-Key."))
        if not question or not question.strip():
            return dict(invalid_input("question must not be empty."))
        if len(question) > MAX_QUESTION_LENGTH:
            return dict(
                invalid_input(f"question exceeds max length of {MAX_QUESTION_LENGTH}.")
            )

        if _is_rate_limited(api_key):
            return dict(rate_limited())

        filter_dict = filter.model_dump(exclude_none=True) if filter else None
        response = await client.query(
            api_key, question=question, filter=filter_dict, session_id=None
        )
        if not response.ok:
            return dict(map_query_error(response.error_code))
        return response.body

    @server.resource(
        "llmwiki://source_listing",
        name="source_listing",
        description="List of all documents currently ingested into the LLM Wiki knowledge base.",
        mime_type="application/json",
    )
    async def source_listing() -> str:
        """PRD Section 4: calls GET /documents on FastAPI. Context injection isn't
        available here (static resource — see module docstring), so auth comes from
        the middleware-populated ContextVar instead."""
        api_key = _api_key_var.get()
        if not api_key:
            return json.dumps(dict(invalid_input("Missing X-API-Key.")))

        response = await client.list_documents(api_key)
        if not response.ok:
            return json.dumps(dict(map_listing_error(response.error_code)))
        return json.dumps(response.body)

    return server


server = build_server()
app = server.sse_app()


def main() -> None:
    import asyncio

    settings = get_settings()
    asyncio.run(server.run_sse_async(host=settings.mcp_host, port=settings.mcp_port))


if __name__ == "__main__":
    main()
