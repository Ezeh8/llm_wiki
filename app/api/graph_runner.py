"""Shared helper for invoking the compiled LangGraph (app.state.graph, built once at
startup — see app.py's lifespan) from HTTP endpoints.
"""

import uuid
from typing import Any

from fastapi import Request

from app.graph.state import build_run_config


def new_thread_id(operation_type: str) -> str:
    # PRD D-20: thread_id = {operation_type}_{uuid4}, generated at the FastAPI boundary.
    return f"{operation_type}_{uuid.uuid4()}"


async def run_graph(
    request: Request,
    *,
    operation_type: str,
    initial_state: dict[str, Any],
    session_id: str,
    actor: str,
) -> dict[str, Any]:
    thread_id = new_thread_id(operation_type)
    config = build_run_config(
        thread_id=thread_id,
        session_id=session_id,
        operation_type=operation_type,
        actor=actor,
    )
    state = {
        **initial_state,
        "operation_type": operation_type,
        "thread_id": thread_id,
        "actor": actor,
    }
    return await request.app.state.graph.ainvoke(state, config=config)
