"""Document routes (PRD Section 5): POST/GET/DELETE/PUT /documents[/{id}].

Ingest and update both run the LangGraph (Validator -> Duplicate Checker -> Chunker ->
Embedding Batcher -> Storer -> Audit Writer, or the update variant — Steps 8-11).
Delete and listing/detail reads go straight through the Postgres adapter — no graph
involvement needed for a plain metadata read/delete outside the ingest/update pipeline
... except DELETE, which the PRD explicitly routes through the graph's Deleter node
(cache-flush side effect, D-16), so DELETE invokes the graph too.
"""

import base64

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.errors import AdapterValidationError
from app.adapters.postgres import PostgresAdapter
from app.api.auth import require_tier
from app.api.error_mapping import error_to_app_error
from app.api.errors import (
    BadRequestError,
    NotFoundError,
    require_nonblank,
    require_valid_uuid,
)
from app.api.graph_runner import run_graph
from app.api.pagination import clamp_limit, paginate
from app.api.schemas import (
    DocumentDetail,
    DocumentListItem,
    DocumentListResponse,
    DocumentResponse,
)
from app.api.timeouts import (
    DELETE_UPDATE_TIMEOUT_SECONDS,
    INGESTION_TIMEOUT_SECONDS,
    LISTING_TIMEOUT_SECONDS,
    with_timeout,
)
from app.config import get_settings
from app.db.session import get_session

router = APIRouter(tags=["documents"])

_ADMIN = {"admin"}
_ADMIN_SERVICE = {"admin", "service"}


def _file_type_from_filename(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


async def _document_response_from_row(session: AsyncSession, document_id: str) -> DocumentResponse:
    doc = await PostgresAdapter(session).get_document(document_id)
    if doc is None:
        # The graph reported success but the row is gone — should not happen; surface
        # plainly rather than silently returning a wrong/empty response.
        raise NotFoundError("Document not found after write")
    return DocumentResponse(
        document_id=str(doc.document_id),
        title=doc.title,
        source_label=doc.source_label,
        chunk_count=doc.chunk_count,
        ingested_at=doc.ingested_at.isoformat(),
    )


async def _ingest_or_update(
    request: Request,
    *,
    operation_type: str,
    document_id: str | None,
    file: UploadFile,
    title: str,
    source_label: str,
    changelog_id: str | None,
    actor: str,
    session: AsyncSession,
) -> DocumentResponse:
    require_nonblank(title, "title")
    require_nonblank(source_label, "source_label")
    raw = await file.read()
    max_bytes = get_settings().max_file_size_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise error_to_app_error("File exceeds maximum size")

    metadata: dict = {
        "title": title,
        "source_label": source_label,
        "file_type": _file_type_from_filename(file.filename),
    }
    if changelog_id:
        metadata["changelog_id"] = changelog_id

    initial_state = {
        "document_id": document_id,
        "document_file": base64.b64encode(raw).decode(),
        "document_metadata": metadata,
    }
    # Ingest gets the 120s ingestion budget; update gets the 30s delete/update budget
    # (PRD Section 5 Tiered Timeouts — these are NOT the same, easy to conflate since
    # both paths share this helper).
    timeout = (
        INGESTION_TIMEOUT_SECONDS
        if operation_type == "ingest"
        else DELETE_UPDATE_TIMEOUT_SECONDS
    )
    result = await with_timeout(
        run_graph(
            request,
            operation_type=operation_type,
            initial_state=initial_state,
            session_id=f"{operation_type}_{actor}",
            actor=actor,
        ),
        seconds=timeout,
    )
    if result.get("status") == "error":
        raise error_to_app_error(result.get("error") or "ingest failed")
    return await _document_response_from_row(session, result["document_id"])


@router.post("/documents", status_code=201, response_model=DocumentResponse)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(..., max_length=200),
    source_label: str = Form(..., max_length=100),
    changelog_id: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN)),
) -> DocumentResponse:
    return await _ingest_or_update(
        request,
        operation_type="ingest",
        document_id=None,
        file=file,
        title=title,
        source_label=source_label,
        changelog_id=changelog_id,
        actor=key.actor_name,
        session=session,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    cursor: str | None = None,
    limit: int | None = None,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN_SERVICE)),
) -> DocumentListResponse:
    page_size = clamp_limit(limit)
    if cursor is not None:
        require_valid_uuid(cursor, "cursor")

    async def _fetch():
        return await PostgresAdapter(session).list_documents(limit=page_size + 1, cursor=cursor)

    try:
        rows = await with_timeout(_fetch(), seconds=LISTING_TIMEOUT_SECONDS)
    except AdapterValidationError:
        raise BadRequestError(f"cursor not found: {cursor!r}")
    page, next_cursor = paginate(rows, page_size, "document_id")
    return DocumentListResponse(
        items=[
            DocumentListItem(
                document_id=str(d.document_id),
                title=d.title,
                source_label=d.source_label,
                ingested_at=d.ingested_at.isoformat(),
            )
            for d in page
        ],
        next_cursor=next_cursor,
    )


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN_SERVICE)),
) -> DocumentDetail:
    require_valid_uuid(document_id, "document_id")

    async def _fetch():
        return await PostgresAdapter(session).get_document(document_id)

    doc = await with_timeout(_fetch(), seconds=LISTING_TIMEOUT_SECONDS)
    if doc is None:
        raise NotFoundError("Document not found")
    return DocumentDetail(
        document_id=str(doc.document_id),
        title=doc.title,
        source_label=doc.source_label,
        chunk_count=doc.chunk_count,
        ingested_at=doc.ingested_at.isoformat(),
        file_type=doc.file_type,
        file_size_bytes=doc.file_size_bytes,
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    request: Request,
    document_id: str,
    key=Depends(require_tier(_ADMIN)),
) -> Response:
    require_valid_uuid(document_id, "document_id")
    # Deleter (Node 13) is naturally idempotent — deleting an absent document is a
    # no-op success, not a 404 (standard idempotent-DELETE semantics; a judgment call
    # documented in BUILD_LOG since the PRD's 404 entry is written with GET/PUT in mind).
    result = await with_timeout(
        run_graph(
            request,
            operation_type="delete",
            initial_state={"document_id": document_id},
            session_id=f"delete_{key.actor_name}",
            actor=key.actor_name,
        ),
        seconds=DELETE_UPDATE_TIMEOUT_SECONDS,
    )
    if result.get("status") == "error":
        raise error_to_app_error(result.get("error") or "delete failed")
    # 204 responses are body-less per HTTP spec; uvicorn strips any body regardless
    # of what's passed here, so there's no point building one (verified empirically —
    # see BUILD_LOG Step 20).
    return Response(status_code=204)


@router.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    request: Request,
    document_id: str,
    file: UploadFile = File(...),
    title: str = Form(..., max_length=200),
    source_label: str = Form(..., max_length=100),
    changelog_id: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN)),
) -> DocumentResponse:
    require_valid_uuid(document_id, "document_id")
    existing = await PostgresAdapter(session).get_document(document_id)
    if existing is None:
        raise NotFoundError("Document not found")
    return await _ingest_or_update(
        request,
        operation_type="update",
        document_id=document_id,
        file=file,
        title=title,
        source_label=source_label,
        changelog_id=changelog_id,
        actor=key.actor_name,
        session=session,
    )
