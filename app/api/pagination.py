"""Cursor-based pagination shared by every list endpoint (PRD Section 5): default 20
items per page, max 100 (anything above is silently capped, not rejected)."""

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


def clamp_limit(limit: int | None) -> int:
    if limit is None or limit < 1:
        return DEFAULT_PAGE_SIZE
    return min(limit, MAX_PAGE_SIZE)


def paginate(rows: list, limit: int, id_attr: str) -> tuple[list, str | None]:
    """`rows` must have been fetched with `limit + 1` rows requested — the standard
    keyset "peek ahead" trick that reveals whether another page exists without a
    separate COUNT query. Returns the trimmed page plus the opaque next_cursor (the
    last row's id as a string), or (rows, None) if that was the last page."""
    if len(rows) > limit:
        page = rows[:limit]
        next_cursor = str(getattr(page[-1], id_attr))
        return page, next_cursor
    return rows, None
