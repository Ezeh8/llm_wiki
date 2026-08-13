"""Pure unit tests for the FastAPI layer's small helpers — no DB/Qdrant needed. The
full HTTP surface (auth, rate limiting, all 14 routes) is covered by
tests/test_api.py, gated on RUN_GRAPH_INTEGRATION_TESTS like the rest of the graph
integration suite (Step 12 reuses that pattern rather than inventing a new one).
"""

from app.api.error_mapping import error_to_app_error
from app.api.errors import (
    ConflictError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    UnsupportedMediaTypeError,
    ValidationAppError,
)
from app.api.pagination import clamp_limit, paginate
from app.parsers.base import PARSE_ERROR_MESSAGE


def test_clamp_limit_defaults_and_caps():
    assert clamp_limit(None) == 20
    assert clamp_limit(5) == 5
    assert clamp_limit(1000) == 100
    assert clamp_limit(0) == 20  # non-positive treated as "unset"


def test_paginate_peek_ahead_reveals_next_cursor():
    class Row:
        def __init__(self, id_):
            self.id = id_

    rows = [Row(i) for i in range(6)]  # limit+1 fetched, as the peek-ahead contract requires
    page, next_cursor = paginate(rows, limit=5, id_attr="id")
    assert len(page) == 5
    assert next_cursor == "4"


def test_paginate_last_page_has_no_next_cursor():
    class Row:
        def __init__(self, id_):
            self.id = id_

    rows = [Row(i) for i in range(3)]
    page, next_cursor = paginate(rows, limit=5, id_attr="id")
    assert len(page) == 3
    assert next_cursor is None


def test_error_mapping_known_messages():
    assert isinstance(error_to_app_error("Duplicate document"), ConflictError)
    assert isinstance(error_to_app_error(PARSE_ERROR_MESSAGE), ValidationAppError)
    assert isinstance(error_to_app_error("Unsupported file type"), UnsupportedMediaTypeError)
    assert isinstance(error_to_app_error("File exceeds maximum size"), PayloadTooLargeError)
    assert isinstance(
        error_to_app_error("failed to store document: boom"), ServiceUnavailableError
    )
    assert isinstance(
        error_to_app_error("failed to delete document: boom"), ServiceUnavailableError
    )


def test_error_mapping_unknown_message_defaults_to_bad_request():
    from app.api.errors import BadRequestError

    assert isinstance(error_to_app_error("Question cannot be empty"), BadRequestError)
