"""Consistent error shape across every endpoint (PRD Section 5 "Error Semantics").

Every error response body: {"status": int, "error": str, "message": str, "retryable":
bool}. Stacktraces are logged server-side only (with full context), never returned to
the client. All 401s use the IDENTICAL body regardless of failure reason (missing key,
invalid key, wrong tier, revoked key) — no signal about which check failed (Section 6).
500 bodies are always the generic "Internal server error." string, zero internal detail.
"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("llm_wiki.api")


class AppError(Exception):
    status_code: int = 500
    error_code: str = "internal_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        if retryable is not None:
            self.retryable = retryable


class BadRequestError(AppError):
    status_code = 400
    error_code = "bad_request"
    retryable = False


class UnauthorizedError(AppError):
    status_code = 401
    error_code = "unauthorized"
    retryable = False

    def __init__(self) -> None:
        # Identical body for every 401 cause — no clue about which check failed
        # (missing key / invalid key / wrong tier / revoked key; PRD Auth Flow #7).
        super().__init__("Unauthorized")


class NotFoundError(AppError):
    status_code = 404
    error_code = "not_found"
    retryable = False


class ConflictError(AppError):
    status_code = 409
    error_code = "conflict"
    retryable = False


class PayloadTooLargeError(AppError):
    status_code = 413
    error_code = "payload_too_large"
    retryable = False


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    error_code = "unsupported_media_type"
    retryable = False


class ValidationAppError(AppError):
    status_code = 422
    error_code = "validation_error"
    retryable = False


class RateLimitedError(AppError):
    status_code = 429
    error_code = "rate_limited"
    retryable = True

    def __init__(self) -> None:
        super().__init__("Too many requests")


class ServiceUnavailableError(AppError):
    status_code = 503
    error_code = "service_unavailable"
    retryable = True


def require_valid_uuid(value: str, field_name: str) -> str:
    """Guard for any route accepting a UUID as a string (path param, query param, or
    body field) — raises a clean 400 instead of letting a malformed value reach
    `_as_uuid()` (app/adapters/postgres.py) unguarded, where it raises an unhandled
    ValueError that falls through to the generic 500 handler (Step 20 finding)."""
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise BadRequestError(f"{field_name} is not a valid UUID: {value!r}")
    return value


def require_nonblank(value: str, field_name: str) -> str:
    """Guard for form fields that must contain real content. `max_length` alone
    doesn't stop a whitespace-only value (e.g. a single space) from being accepted
    (Step 20 finding: document titles could be blank in all but name)."""
    if not value.strip():
        raise BadRequestError(f"{field_name} cannot be blank")
    return value


def _error_body(status: int, error: str, message: str, retryable: bool) -> dict:
    return {"status": status, "error": error, "message": message, "retryable": retryable}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.status_code, exc.error_code, exc.message, exc.retryable),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_body(422, "validation_error", "Request failed validation", False),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # Covers FastAPI's own unmatched-route 404 and any bare HTTPException raised
        # somewhere that isn't one of our AppError subclasses.
        code = {404: "not_found", 401: "unauthorized"}.get(exc.status_code, "bad_request")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.status_code, code, message, False),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Full stacktrace/context logged server-side only; client gets the generic body.
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_body(500, "internal_error", "Internal server error.", False),
        )
