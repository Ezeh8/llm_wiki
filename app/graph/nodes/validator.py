"""Node 1 — Validator.

Runs first on every operation path. Rejects malformed input before any expensive work
happens: missing/oversized file, unsupported extension, empty/over-length query,
missing required fields per operation. Built as REAL logic now (not a stub) — these
checks are self-contained and don't depend on any node owned by a later step.

"No extractable text" (the PRD's third rejection reason) is enforced structurally by
the Chunker (Step 9's real logic already exists as of Step 4/5): it wraps parsing in
try/catch and reports the identical client-safe message on failure (D-43). Re-parsing
here to duplicate that check would cost a full parse for no benefit, so Validator does
not attempt it.

Judgment call (file_type): document_metadata is typed by the PRD as
{title, source_label, changelog_id}, with no `file_type`. Since Validator/Chunker both
need the extension, FastAPI (Step 12) is expected to set
`document_metadata["file_type"]` (derived from the uploaded filename) before invoking
the graph — documented here so Step 12 honors it.

Gap-fill (Step 8 judgment call, not a locked-decision change): the PRD's "4 routing
points" describe routing after Validator only by `operation_type`, but a rejected input
has to go somewhere. `route_after_validator` (app/graph/routing.py) checks
`status == "error"` first and sends it straight to Audit Writer — otherwise a bad
upload would silently continue toward the Chunker.
"""

from langchain_core.runnables import RunnableConfig

from app.graph.state import GraphState
from app.parsers import PARSERS

ALLOWED_EXTENSIONS = frozenset(PARSERS.keys())


def _reject(state: GraphState, error: str) -> dict:
    return {
        "status": "error",
        "error": error,
        "thread_id": state.get("thread_id", ""),
    }


def _ok(thread_id: str) -> dict:
    return {"thread_id": thread_id, "status": "success", "error": None}


async def validator(state: GraphState, config: RunnableConfig) -> dict:
    thread_id = state.get("thread_id") or config.get("metadata", {}).get("thread_id", "")
    op = state.get("operation_type")
    configurable = config["configurable"]

    if op == "query":
        query_text = state.get("query_text") or ""
        max_len = configurable.get("max_query_length", 2000)
        if not query_text.strip():
            return _reject(state, "Question cannot be empty")
        if len(query_text) > max_len:
            return _reject(state, f"Question exceeds max length of {max_len} characters")
        return _ok(thread_id)

    if op in ("ingest", "update"):
        metadata = state.get("document_metadata") or {}
        if not metadata.get("title") or not metadata.get("source_label"):
            return _reject(state, "title and source_label are required")

        document_file = state.get("document_file")
        if not document_file:
            return _reject(state, "File is empty")

        file_type = (metadata.get("file_type") or "").lower().lstrip(".")
        if file_type not in ALLOWED_EXTENSIONS:
            return _reject(state, "Unsupported file type")

        max_bytes = configurable.get("max_file_size_mb", 20) * 1024 * 1024
        # document_file is base64; decoded size is ~3/4 of the encoded string length.
        approx_bytes = len(document_file) * 3 // 4
        if approx_bytes > max_bytes:
            return _reject(state, "File exceeds maximum size")

        return _ok(thread_id)

    if op == "delete":
        if not state.get("document_id"):
            return _reject(state, "document_id is required")
        return _ok(thread_id)

    return _reject(state, f"unknown operation_type: {op!r}")
