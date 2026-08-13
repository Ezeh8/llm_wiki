"""FastAPI app factory (PRD Section 5). Registers routers, middleware, and exception
handlers, and builds the checkpointer + compiles the graph once at startup (lifespan),
stored on `app.state.graph` for every request to share.

Middleware order (Section 5: "1. Rate limiter, 2. Auth, 3. Request logging, 4.
Endpoint logic"): Starlette applies the LAST-added middleware OUTERMOST, so
`RequestLoggingMiddleware` is added first (inner) and `RateLimitMiddleware` second
(outer) below. The resulting request flow is: RateLimitMiddleware (outermost, runs
first) -> RequestLoggingMiddleware (starts its timer, then calls onward) -> routing,
where the `require_tier(...)` auth dependency runs -> the endpoint body ->
RequestLoggingMiddleware writes its log line once the response comes back (now with
`request.state.actor` populated if auth succeeded) -> RateLimitMiddleware passes the
response through unchanged. This satisfies the PRD's ordering intent even though Auth
is a FastAPI dependency rather than raw ASGI middleware — see auth.py/middleware.py's
docstrings for why that split was made.

CORS: intentionally NOT configured — v1 has no browser origins (Section 5
Cross-Cutting: "CORS: blocked"). Adding CORSMiddleware is the documented v2 upgrade
path once a browser frontend exists.

Startup bootstrap (Step 17): besides the dead-letter replay routine (Step 14), the
lifespan also idempotently provisions the Qdrant collection (Step 7's
`ensure_collection`) and the checkpointer's own tables + grants (Step 8's
`setup_checkpointer_schema`) — both were explicitly designed as "safe to call on
every startup" from the day they were built, specifically for this purpose. Wiring
them in here means `docker compose up` + `alembic upgrade head` + creating an admin
key (Section 10 Setup Instructions) is the complete bootstrap — no separate manual
`python -m app.qdrant_setup` invocation needed, though it remains available standalone
for CI/manual use. Alembic migrations themselves stay a deliberate manual step (not
run from here): they need the ADMIN role to create the restricted role in the first
place (Step 1), and auto-migrating from application startup code is generally the
wrong call for a system where schema changes should be a reviewed, explicit action.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from app.api.routers import audit, changelog, documents, health, query
from app.config import get_settings
from app.dead_letter import replay_dead_letters
from app.graph.build import compile_graph
from app.graph.checkpointer import build_checkpointer, setup_checkpointer_schema
from app.qdrant_setup import ensure_collection

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # fail fast on startup if required env vars are missing (Section 5)
    # Same fail-fast intent, extended to the dormant-provider model names (Stage 2 Step
    # 2): these have no pydantic default to enforce, so they need an explicit check here
    # rather than at build_llm_adapter() construction time, which only runs per-request.
    if settings.llm_provider == "openai" and not settings.llm_model_name_openai:
        raise ValueError("LLM_MODEL_NAME_OPENAI must be set when LLM_PROVIDER=openai")
    if settings.llm_provider == "llama" and not settings.llm_model_name_llama:
        raise ValueError("LLM_MODEL_NAME_LLAMA must be set when LLM_PROVIDER=llama")
    await ensure_collection()  # Step 7: idempotent Qdrant collection provisioning
    await setup_checkpointer_schema()  # Step 8: idempotent checkpoint tables + grants
    await replay_dead_letters()  # Step 14: startup replay routine
    async with build_checkpointer() as checkpointer:
        app.state.graph = compile_graph(checkpointer=checkpointer)
        yield


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Wiki", version=get_settings().app_version, lifespan=lifespan)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(changelog.router)
    app.include_router(audit.router)

    # See module docstring for why this registration order yields the PRD's stated
    # Rate Limiter -> Auth -> Request Logging -> Endpoint flow.
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    return app


app = create_app()
