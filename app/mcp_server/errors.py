"""MCP error shape (PRD Section 4): {"code": str, "message": str, "retryable": bool}.
MCP maps FastAPI's "error" field to this "code" field; unmapped errors default to
"internal_error", retryable=true.

**Honest coverage note** (documented in BUILD_LOG, not hidden here): 3 of the PRD's 7
listed codes — `insufficient_results`, `generation_error`, `grounding_failed` — are
never actually produced by this system's FastAPI layer. That's not an oversight here;
it's the direct, already-locked consequence of Steps 10/12's design: Quality Gate's
"insufficient" and Generator's "degraded" fallback are both deliberately surfaced as
ordinary 200 responses (informative answer text, `degraded` flag), not HTTP errors —
"LLM never blocks retrieval" (Node 7) extended consistently to "insufficient retrieval
doesn't block a response either." A calling agent gets a clear, honest answer either
way; it just never sees these 3 codes via the {code,message,retryable} error shape.
Reopening that to manufacture these codes would mean re-litigating a decision made
(and tested) two steps ago for the sole benefit of matching an error table more
literally — not worth it. `embedding_error` is similarly only partially reachable: an
Embedder/Retriever node failure that exhausts its own retries currently propagates as
a generic 500 (`internal_error`) rather than a typed error, since Steps 8/10 never
wrapped those node-level failures in a dedicated try/except the way Generator's
LLM-failure path is. Flagged here as a known gap in query-path robustness, not an MCP
concern to fix — the same gap exists at the raw FastAPI layer.
"""

from typing import TypedDict


class McpError(TypedDict):
    code: str
    message: str
    retryable: bool


# Maps a FastAPI error body's "error" field to an MCP code, by call context — the
# SAME FastAPI code (e.g. "service_unavailable") means something different depending
# on which endpoint produced it (POST /query vs GET /documents), so this can't be one
# flat table.
_QUERY_ERROR_MAP: dict[str, str] = {
    "bad_request": "invalid_input",
    "validation_error": "invalid_input",
    "service_unavailable": "retrieval_error",
}

_LISTING_ERROR_MAP: dict[str, str] = {
    "service_unavailable": "listing_error",
}


def _map(fastapi_error_code: str | None, table: dict[str, str]) -> McpError:
    if fastapi_error_code and fastapi_error_code in table:
        code = table[fastapi_error_code]
        retryable = code in ("retrieval_error", "listing_error")
        message = {
            "invalid_input": "The question or filter was invalid.",
            "retrieval_error": "The knowledge base is temporarily unreachable.",
            "listing_error": "The document listing is temporarily unreachable.",
        }[code]
        return McpError(code=code, message=message, retryable=retryable)
    return McpError(
        code="internal_error",
        message="An unexpected error occurred.",
        retryable=True,
    )


def map_query_error(fastapi_error_code: str | None) -> McpError:
    return _map(fastapi_error_code, _QUERY_ERROR_MAP)


def map_listing_error(fastapi_error_code: str | None) -> McpError:
    return _map(fastapi_error_code, _LISTING_ERROR_MAP)


def invalid_input(message: str) -> McpError:
    """For requests MCP itself rejects before ever calling FastAPI (Section 4
    Security: "Input validation: non-empty question, max 2000 chars, valid filter
    values") — output sanitization means never forwarding a raw FastAPI error, but
    MCP's OWN input validation doesn't have a FastAPI error to translate at all."""
    return McpError(code="invalid_input", message=message, retryable=False)


def rate_limited() -> McpError:
    return McpError(
        code="internal_error",  # not one of the 7 listed codes; falls to the documented default
        message="Too many requests.",
        retryable=True,
    )
