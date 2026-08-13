"""Query route (PRD Section 5): POST /query. Runs the full query path graph (Steps 8,
10): Validator -> Cache Checker -> [Embedder -> Retriever -> Ranker -> Quality Gate ->
Generator -> Cache Writer] -> Audit Writer.
"""

import uuid

from fastapi import APIRouter, Depends, Request

from app.api.auth import require_tier
from app.api.error_mapping import error_to_app_error
from app.api.graph_runner import run_graph
from app.api.schemas import QueryRequest, QueryResponse
from app.api.timeouts import QUERY_TIMEOUT_SECONDS, with_timeout

router = APIRouter(tags=["query"])

_ADMIN_SERVICE = {"admin", "service"}

_INSUFFICIENT_ANSWER = "No sufficiently relevant documents were found for this question."


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: Request,
    body: QueryRequest,
    key=Depends(require_tier(_ADMIN_SERVICE)),
) -> QueryResponse:
    # PRD: session_id "optional from caller, auto-generated if omitted".
    session_id = body.session_id or str(uuid.uuid4())

    document_metadata: dict = {}
    if body.filter is not None:
        filter_dict = {
            k: v
            for k, v in (
                ("source_label", body.filter.source_label),
                ("document_id", body.filter.document_id),
            )
            if v is not None
        }
        if filter_dict:
            document_metadata["filter"] = filter_dict

    result = await with_timeout(
        run_graph(
            request,
            operation_type="query",
            initial_state={
                "query_text": body.question,
                "session_id": session_id,
                "document_metadata": document_metadata,
            },
            session_id=session_id,
            actor=key.actor_name,
        ),
        seconds=QUERY_TIMEOUT_SECONDS,
    )

    status = result.get("status")
    if status == "error":
        raise error_to_app_error(result.get("error") or "query failed")

    query_id = result.get("thread_id", "")
    if status == "insufficient":
        return QueryResponse(
            answer=_INSUFFICIENT_ANSWER,
            citations=[],
            source_chunks=None,
            cached=False,
            query_id=query_id,
            degraded=False,
            session_id=session_id,
        )

    degraded = status == "degraded"
    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations") or [],
        # PRD Node 7: "populate source_chunks with raw ranked_chunks" on degrade —
        # ranked_chunks already holds exactly that (Step 10), no separate state field.
        source_chunks=result.get("ranked_chunks") if degraded else None,
        cached=bool(result.get("cache_hit")),
        query_id=query_id,
        degraded=degraded,
        session_id=session_id,
    )
