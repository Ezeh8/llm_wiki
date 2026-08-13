"""The 4 PRD-documented routing points, plus 2 Step 8 gap-fill branches that are
necessary graph wiring (not new decisions — a rejected/duplicate input has to go
somewhere, and the PRD's Node descriptions already specify the rejection itself; see
the docstrings in validator.py and ingest_path.py for the reasoning).
"""

from app.graph.state import GraphState


def route_after_validator(state: GraphState) -> str:
    """PRD routing point 1, plus the gap-fill: a validation failure skips straight to
    Audit Writer instead of continuing toward Duplicate Checker/Chunker/Deleter."""
    if state.get("status") == "error":
        return "audit_writer"
    op = state["operation_type"]
    if op == "query":
        return "cache_checker"
    if op == "ingest":
        return "duplicate_checker"
    if op == "delete":
        return "deleter"
    if op == "update":
        return "chunker"
    raise ValueError(f"unknown operation_type: {op!r}")


def route_after_duplicate_checker(state: GraphState) -> str:
    """Gap-fill: a duplicate must not be chunked/embedded/stored — routes straight to
    Audit Writer instead. Not one of the PRD's literal "4 routing points" but required
    for Node 9's rejection to have an effect."""
    return "audit_writer" if state.get("status") == "error" else "chunker"


def route_after_cache_checker(state: GraphState) -> str:
    """PRD routing point 2."""
    return "audit_writer" if state.get("cache_hit") else "embedder_query"


def route_after_embedder_query(state: GraphState) -> str:
    """Gap-fill (Step 20): an embedding failure must not reach Retriever, which
    unconditionally reads query_embedding — routes straight to Audit Writer instead.
    Required for Node 3's error to have an effect rather than crashing the next node."""
    return "audit_writer" if state.get("status") == "error" else "retriever"


def route_after_retriever(state: GraphState) -> str:
    """Gap-fill (mirrors route_after_embedder_query): a retrieval failure must not
    reach Ranker/Quality Gate, which would otherwise silently score the missing
    retrieved_chunks as "insufficient" instead of surfacing the real infrastructure
    failure — routes straight to Audit Writer instead."""
    return "audit_writer" if state.get("status") == "error" else "ranker"


def route_after_embedding_batcher(state: GraphState) -> str:
    """Gap-fill (Step 20): an embedding failure must not reach Storer, which
    unconditionally reads embeddings — routes straight to Audit Writer instead.
    Required for Node 11's error to have an effect rather than crashing the next node."""
    return "audit_writer" if state.get("status") == "error" else "storer"


def route_after_quality_gate(state: GraphState) -> str:
    """PRD routing point 3."""
    return "generator" if state.get("status") == "sufficient" else "audit_writer"


def route_after_storer(state: GraphState) -> str:
    """PRD routing point 4 (first half): update path deletes the OLD document only
    AFTER the new one is CONFIRMED stored (D-2, fortified order — ingest new first).

    Step 11 bug fix: this originally routed to Deleter for every update regardless of
    Storer's outcome. Found by running the update path end-to-end: when Storer failed
    (a real KeyError bug, since fixed), this still deleted the OLD document — exactly
    the "hole" D-2 exists to prevent (a failed update should leave the old document in
    place, temporarily duplicated with nothing, rather than deleted with nothing to
    replace it) — and Deleter's own `status="success"` then silently overwrote
    Storer's `status="error"` in the final state (all fields overwrite), hiding the
    failure entirely. Now gated on `status != "error"`.
    """
    if state.get("operation_type") == "update" and state.get("status") != "error":
        return "deleter"
    return "audit_writer"
