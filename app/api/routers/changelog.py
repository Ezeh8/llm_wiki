"""Changelog routes (PRD Section 5). Plain CRUD via PostgresAdapter — no graph
involvement, unlike documents/query. Employees can manage changelog entries; admin
sees/manages everything too (Section 5 Routes table: admin, employee tier)."""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.errors import AdapterValidationError
from app.adapters.postgres import PostgresAdapter
from app.api.auth import require_tier
from app.api.errors import BadRequestError, NotFoundError, require_valid_uuid
from app.api.pagination import clamp_limit, paginate
from app.api.schemas import (
    ChangelogCreateRequest,
    ChangelogDetail,
    ChangelogListResponse,
    ChangelogResponse,
    ChangelogUpdateRequest,
)
from app.api.timeouts import LISTING_TIMEOUT_SECONDS, with_timeout
from app.db.session import get_session

router = APIRouter(tags=["changelog"])

_ADMIN_EMPLOYEE = {"admin", "employee"}


def _to_response(entry) -> ChangelogResponse:
    return ChangelogResponse(
        changelog_id=str(entry.changelog_id),
        entry=entry.entry,
        document_id=str(entry.document_id) if entry.document_id else None,
        actor=entry.actor,
        created_at=entry.created_at,
    )


@router.post("/changelog", status_code=201, response_model=ChangelogResponse)
async def create_changelog(
    body: ChangelogCreateRequest,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN_EMPLOYEE)),
) -> ChangelogResponse:
    if body.document_id is not None:
        require_valid_uuid(body.document_id, "document_id")

    async def _create():
        adapter = PostgresAdapter(session)
        if body.document_id is not None and await adapter.get_document(body.document_id) is None:
            raise NotFoundError("Document not found")
        entry = await adapter.create_changelog(
            entry=body.entry, actor=key.actor_name, document_id=body.document_id
        )
        await session.commit()
        return entry

    entry = await with_timeout(_create(), seconds=LISTING_TIMEOUT_SECONDS)
    return _to_response(entry)


@router.get("/changelog", response_model=ChangelogListResponse)
async def list_changelog(
    cursor: str | None = None,
    limit: int | None = None,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN_EMPLOYEE)),
) -> ChangelogListResponse:
    page_size = clamp_limit(limit)
    if cursor is not None:
        require_valid_uuid(cursor, "cursor")

    async def _fetch():
        return await PostgresAdapter(session).list_changelog(limit=page_size + 1, cursor=cursor)

    try:
        rows = await with_timeout(_fetch(), seconds=LISTING_TIMEOUT_SECONDS)
    except AdapterValidationError:
        raise BadRequestError(f"cursor not found: {cursor!r}")
    page, next_cursor = paginate(rows, page_size, "changelog_id")
    return ChangelogListResponse(items=[_to_response(e) for e in page], next_cursor=next_cursor)


@router.get("/changelog/{changelog_id}", response_model=ChangelogDetail)
async def get_changelog(
    changelog_id: str,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN_EMPLOYEE)),
) -> ChangelogDetail:
    require_valid_uuid(changelog_id, "changelog_id")

    async def _fetch():
        return await PostgresAdapter(session).get_changelog(changelog_id)

    entry = await with_timeout(_fetch(), seconds=LISTING_TIMEOUT_SECONDS)
    if entry is None:
        raise NotFoundError("Changelog entry not found")
    return ChangelogDetail(**_to_response(entry).model_dump(), updated_at=entry.updated_at)


@router.put("/changelog/{changelog_id}", response_model=ChangelogDetail)
async def update_changelog(
    changelog_id: str,
    body: ChangelogUpdateRequest,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN_EMPLOYEE)),
) -> ChangelogDetail:
    require_valid_uuid(changelog_id, "changelog_id")
    if body.document_id is not None:
        require_valid_uuid(body.document_id, "document_id")

    async def _update():
        adapter = PostgresAdapter(session)
        if body.document_id is not None and await adapter.get_document(body.document_id) is None:
            raise NotFoundError("Document not found")
        entry = await adapter.update_changelog(
            changelog_id, entry=body.entry, document_id=body.document_id
        )
        if entry is not None:
            await session.commit()
        return entry

    entry = await with_timeout(_update(), seconds=LISTING_TIMEOUT_SECONDS)
    if entry is None:
        raise NotFoundError("Changelog entry not found")
    return ChangelogDetail(**_to_response(entry).model_dump(), updated_at=entry.updated_at)


@router.delete("/changelog/{changelog_id}")
async def delete_changelog(
    changelog_id: str,
    session: AsyncSession = Depends(get_session),
    key=Depends(require_tier(_ADMIN_EMPLOYEE)),
) -> Response:
    require_valid_uuid(changelog_id, "changelog_id")

    async def _delete():
        deleted = await PostgresAdapter(session).delete_changelog(changelog_id)
        if deleted:
            await session.commit()
        return deleted

    deleted = await with_timeout(_delete(), seconds=LISTING_TIMEOUT_SECONDS)
    if not deleted:
        raise NotFoundError("Changelog entry not found")
    # 204 responses are body-less per HTTP spec; uvicorn strips any body regardless
    # of what's passed here, so there's no point building one (verified empirically —
    # see BUILD_LOG Step 20).
    return Response(status_code=204)
