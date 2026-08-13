"""Delete path node (PRD Section 3, Node 13). REAL as of Step 11.

Two callers reach this node with different targets, per Step 8's routing:
  - A genuine delete (operation_type="delete"): target = state.document_id, the id the
    client asked to remove.
  - The update path's old-document cleanup (operation_type="update", reached only
    after Storer confirmed the NEW document was stored — D-2's fortified order):
    state.document_id has already been overwritten with the NEW document's id by the
    Chunker, so the OLD id to clean up is read from
    document_metadata["previous_document_id"] instead (stashed there by the Chunker —
    see app/graph/nodes/ingest_path.py's chunker_node docstring).

Changelog FK SET NULL (D-4) needs no code here — it's a `ON DELETE SET NULL` foreign
key enforced by Postgres itself (migration 0001), so deleting the documents row
automatically nulls out any changelog entries that referenced it.
"""

from langchain_core.runnables import RunnableConfig

from app.adapters.postgres import PostgresAdapter
from app.adapters.vector_store import build_vector_store
from app.db.session import async_session_factory
from app.graph.nodes._common import call_with_retry, flush_cache_side_effect
from app.graph.state import GraphState


def _target_document_id(state: GraphState) -> str | None:
    if state.get("operation_type") == "update":
        return (state.get("document_metadata") or {}).get("previous_document_id")
    return state.get("document_id")


async def deleter(state: GraphState, config: RunnableConfig) -> dict:
    """Node 13 — REAL. Order matters: Qdrant chunks first, then Postgres metadata —
    naturally idempotent either way (deleting an absent document_id is a no-op), so
    there's no rollback concern symmetric to Storer's. The PRD doesn't state a retry
    count for the delete itself (only for the cache-flush side effect, "same as
    Storer"), so 1 retry is used here too, for consistency with every other
    Postgres/Qdrant-touching node in this build. Side effect: flush cache, same helper
    Storer uses, independent of whether the delete itself succeeds or fails.
    """
    document_id = _target_document_id(state)
    if not document_id:
        # Update path with nothing to clean up (e.g. this node re-run in isolation
        # without the Chunker having stashed a previous_document_id) — nothing to do,
        # but the cache-flush side effect still runs since it's independent of target.
        await flush_cache_side_effect()
        return {"status": "success", "error": None}

    store = build_vector_store()

    async def _delete_attempt() -> None:
        await store.delete_by_document_id(document_id)  # Qdrant first
        async with async_session_factory() as session:
            await PostgresAdapter(session).delete_document(document_id)  # Postgres second
            await session.commit()

    try:
        await call_with_retry(_delete_attempt, retries=1)
        result = {"status": "success", "error": None}
    except Exception as exc:
        result = {"status": "error", "error": f"failed to delete document: {exc}"}

    await flush_cache_side_effect()
    return result
