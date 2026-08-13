"""Node 14 — Audit Writer. Append-only insert; idempotency via the thread_id+event_type
unique constraint already enforced at the Postgres adapter/DB level (Steps 1-2).

3 retries per PRD. Dead-letter on exhaustion (Step 14, app/dead_letter.py): the real
operation (ingest/query/delete/update) has already completed by the time this node
runs, so a failure to WRITE ITS OWN LOG ENTRY must never retroactively fail the
user's already-finished request — it's captured to a persistent-volume file instead,
replayed at the next app startup.
"""

from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from app.adapters.postgres import PostgresAdapter
from app.dead_letter import write_dead_letter
from app.db.session import async_session_factory
from app.graph.nodes._common import call_with_retry
from app.graph.state import GraphState


async def audit_writer(state: GraphState, config: RunnableConfig) -> dict:
    thread_id = state.get("thread_id", "")
    event_type = state.get("operation_type", "unknown")
    idempotency_key = f"{thread_id}:{event_type}"
    retrieved = state.get("retrieved_chunks")
    # Fixed once, not per retry attempt — every attempt (and the dead-letter/replay
    # record, if it comes to that) records the same original event time.
    timestamp = datetime.now(timezone.utc)

    fields = {
        "event_type": event_type,
        "actor": state.get("actor", ""),
        "status": state.get("status", ""),
        "idempotency_key": idempotency_key,
        "document_id": state.get("document_id"),
        "query_text": state.get("query_text"),
        "chunks_retrieved": len(retrieved) if retrieved else None,
        "answer_text": state.get("answer"),
        "error_detail": state.get("error"),
        "timestamp": timestamp,
    }

    async def _write():
        async with async_session_factory() as session:
            await PostgresAdapter(session).write_audit(**fields)
            await session.commit()

    try:
        await call_with_retry(_write, retries=3)
    except Exception:
        try:
            write_dead_letter({**fields, "timestamp": timestamp.isoformat()})
        except Exception:
            pass  # even the dead-letter write failed — nothing further to do here
    return {}
