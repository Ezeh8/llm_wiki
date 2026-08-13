"""Audit routes (PRD Section 5). Read-only, admin-only."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.errors import AdapterValidationError
from app.adapters.postgres import PostgresAdapter
from app.api.auth import require_tier
from app.api.errors import BadRequestError, NotFoundError, require_valid_uuid
from app.api.pagination import clamp_limit, paginate
from app.api.schemas import AuditDetail, AuditListItem, AuditListResponse
from app.api.timeouts import LISTING_TIMEOUT_SECONDS, with_timeout
from app.db.session import get_session

router = APIRouter(tags=["audit"])

_ADMIN = {"admin"}


@router.get("/audit", response_model=AuditListResponse)
async def list_audit(
    cursor: str | None = None,
    limit: int | None = None,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN)),
) -> AuditListResponse:
    page_size = clamp_limit(limit)
    if cursor is not None:
        require_valid_uuid(cursor, "cursor")

    async def _fetch():
        return await PostgresAdapter(session).list_audit(limit=page_size + 1, cursor=cursor)

    try:
        rows = await with_timeout(_fetch(), seconds=LISTING_TIMEOUT_SECONDS)
    except AdapterValidationError:
        raise BadRequestError(f"cursor not found: {cursor!r}")
    page, next_cursor = paginate(rows, page_size, "event_id")
    return AuditListResponse(
        items=[
            AuditListItem(
                event_id=str(e.event_id),
                event_type=e.event_type,
                document_id=e.document_id,
                actor=e.actor,
                timestamp=e.timestamp,
                status=e.status,
            )
            for e in page
        ],
        next_cursor=next_cursor,
    )


@router.get("/audit/{event_id}", response_model=AuditDetail)
async def get_audit(
    event_id: str,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN)),
) -> AuditDetail:
    require_valid_uuid(event_id, "event_id")

    async def _fetch():
        return await PostgresAdapter(session).get_audit(event_id)

    event = await with_timeout(_fetch(), seconds=LISTING_TIMEOUT_SECONDS)
    if event is None:
        raise NotFoundError("Audit event not found")
    return AuditDetail(
        event_id=str(event.event_id),
        event_type=event.event_type,
        document_id=event.document_id,
        query_text=event.query_text,
        chunks_retrieved=event.chunks_retrieved,
        answer_text=event.answer_text,
        actor=event.actor,
        timestamp=event.timestamp,
        status=event.status,
        error_detail=event.error_detail,
    )
