"""Mapping failures onto the published error contract (§7.4).

Two rules shape this module.

*One failure, one code.* Consumers branch on ``code``, so the mapping has to be
total and stable. A handler that fell through to a generic 500 would turn "you
lack the deletion scope" into "something broke", and the caller would retry.

*Nothing internal escapes.* An unexpected exception becomes a generic message
plus the trace id. NFR-018 keeps transcript content out of operational logs,
and a stack trace rendered into an HTTP response is the fastest route for it to
leave the building.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from evidence_engine.adapters.inbound.rest.schemas import ErrorBody
from evidence_engine.application.commands.administer_keys import (
    ApplicationNotFound,
    KeyNotFound,
    UnissuableScope,
)
from evidence_engine.application.errors import (
    ApplicationError,
    BackpressureRequired,
    NotAuthenticated,
    NotAuthorized,
    QuotaExceeded,
    ResultNotReady,
    SessionBusy,
    SessionNotFound,
)
from evidence_engine.domain.sessions.capabilities import UnsupportedCapability
from evidence_engine.domain.sessions.consent import ConsentViolation
from evidence_engine.domain.shared.errors import (
    DomainError,
    IllegalSessionTransition,
    ProhibitedLabel,
)

#: application/domain exception -> (HTTP status, stable error code).
#: Order matters for subclasses, so the table is consulted by exact type first
#: and by isinstance afterwards.
_MAPPING: tuple[tuple[type[Exception], int, str], ...] = (
    (NotAuthenticated, status.HTTP_401_UNAUTHORIZED, "not_authenticated"),
    (NotAuthorized, status.HTTP_403_FORBIDDEN, "insufficient_scope"),
    (SessionNotFound, status.HTTP_404_NOT_FOUND, "session_not_found"),
    (ApplicationNotFound, status.HTTP_404_NOT_FOUND, "application_not_found"),
    (KeyNotFound, status.HTTP_404_NOT_FOUND, "api_key_not_found"),
    # 400 rather than 422. The body parses and is well formed; what is refused
    # is the privilege it asks for, and pointing a portal at its serializer
    # would send somebody looking in the wrong place entirely.
    (UnissuableScope, status.HTTP_400_BAD_REQUEST, "unissuable_scope"),
    (ResultNotReady, status.HTTP_409_CONFLICT, "result_not_ready"),
    (SessionBusy, status.HTTP_409_CONFLICT, "session_busy"),
    (QuotaExceeded, status.HTTP_429_TOO_MANY_REQUESTS, "quota_exceeded"),
    (BackpressureRequired, status.HTTP_429_TOO_MANY_REQUESTS, "backpressure_required"),
    (UnsupportedCapability, status.HTTP_422_UNPROCESSABLE_CONTENT, "unsupported_capability"),
    (ConsentViolation, status.HTTP_403_FORBIDDEN, "consent_required"),
    (IllegalSessionTransition, status.HTTP_409_CONFLICT, "illegal_transition"),
    (ProhibitedLabel, status.HTTP_500_INTERNAL_SERVER_ERROR, "prohibited_label"),
)


def resolve(error: Exception) -> tuple[int, str]:
    """Find the status and code for a failure."""
    for exception_type, http_status, code in _MAPPING:
        if type(error) is exception_type:
            return http_status, code
    for exception_type, http_status, code in _MAPPING:
        if isinstance(error, exception_type):
            return http_status, code
    if isinstance(error, ApplicationError):
        return status.HTTP_400_BAD_REQUEST, "invalid_request"
    if isinstance(error, DomainError):
        # A domain error reaching the boundary means the engine was about to
        # publish evidence contradicting its own specification. That is a fault
        # on our side, not a malformed request, and it must read as one.
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "domain_invariant_violated"
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error"


def install(app: FastAPI, schema_version: str) -> None:
    """Register the handlers that render every failure as an ``ErrorBody``."""

    async def _handle(request: Request, error: Exception) -> JSONResponse:
        http_status, code = resolve(error)
        trace_id = request.headers.get("X-Trace-Id")

        # Only our own errors carry a message the caller should see. Anything
        # else is summarized, so an unexpected exception cannot leak internals
        # or evidence content into a response body.
        if isinstance(error, ApplicationError | DomainError):
            message = str(error)
        else:
            message = "the request could not be completed"

        retry_after = getattr(error, "retry_after_seconds", None)
        body = ErrorBody(
            schema_version=schema_version,
            code=code,
            message=message,
            trace_id=trace_id,
            retry_after_seconds=retry_after if retry_after else None,
        )
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        return JSONResponse(status_code=http_status, content=body.model_dump(), headers=headers)

    for exception_type in (ApplicationError, DomainError):
        app.add_exception_handler(exception_type, _as_handler(_handle))
    app.add_exception_handler(Exception, _as_handler(_handle))


def _as_handler(
    handler: Callable[[Request, Exception], Awaitable[JSONResponse]],
) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    """Satisfy Starlette's handler signature without loosening our own typing."""
    return handler
