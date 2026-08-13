"""Maps a graph node's plain-string `error` (GraphState.error — the schema has no
structured error CODE field, and D-11 locks the schema at 19 fields, so adding one
wasn't worth reopening it) to the right AppError subclass / HTTP status.

The string-matching here is safe despite looking fragile: every message is a FIXED
literal from code written in Steps 8-11 (Validator, Duplicate Checker, Chunker's
parse-failure passthrough), not arbitrary user input or a raw exception's text — with
three exceptions (Storer/Deleter's own failure messages; Embedder's `EmbeddingError`
via embedder_query/embedding_batcher, Step 20; and Retriever's `RetrievalError`, Stage
2 Step 3) that interpolate `str(exc)`, but their prefix is still a fixed, reliably
matchable literal.
"""

from app.api.errors import (
    AppError,
    BadRequestError,
    ConflictError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    UnsupportedMediaTypeError,
    ValidationAppError,
)
from app.parsers.base import PARSE_ERROR_MESSAGE


def error_to_app_error(message: str) -> AppError:
    if message == "Duplicate document":
        return ConflictError(message)
    if message == PARSE_ERROR_MESSAGE:
        return ValidationAppError(message)
    if message == "Unsupported file type":
        return UnsupportedMediaTypeError(message)
    if message == "File exceeds maximum size":
        return PayloadTooLargeError(message)
    if (
        message.startswith("failed to store document:")
        or message.startswith("failed to delete document:")
        or message.startswith("embedding failed")
        or message.startswith("retrieval failed")
    ):
        return ServiceUnavailableError(message)
    return BadRequestError(message)
