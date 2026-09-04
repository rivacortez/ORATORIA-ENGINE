"""Request-scoped dependencies: the container, the trace id and the caller.

Authentication is a dependency rather than middleware so that FastAPI records
it in the OpenAPI document and so that a route which forgets it is visibly
different from one that has it - a middleware that authenticates everything
makes the public routes the exception, and exceptions get added by accident.

The trace id is minted here when the client does not supply one. FR-004
requires every request and connection to be attributable to an application and
a trace id, and a trace id created later, deeper in the stack, cannot appear on
the log lines written before it existed.
"""

from __future__ import annotations

import uuid
from typing import Annotated, cast

from fastapi import Depends, Header, HTTPException, status
from starlette.requests import HTTPConnection

from evidence_engine.application.api import EngineApi
from evidence_engine.application.ports.platform import AuthenticatedCaller

#: Header carrying the API key. A bearer scheme rather than a bespoke header
#: so that standard tooling - proxies, SDK generators, log scrubbers - already
#: knows to treat it as a credential.
AUTHORIZATION = "Authorization"
TRACE_HEADER = "X-Trace-Id"
IDEMPOTENCY_HEADER = "Idempotency-Key"


def get_engine(connection: HTTPConnection) -> EngineApi:
    """The engine assembled at startup and stashed on the app state.

    Typed as ``HTTPConnection``, the base of both ``Request`` and
    ``WebSocket``, so one dependency serves the REST routes and the streaming
    endpoint. Asking for a ``Request`` here fails at runtime on the WebSocket
    route, and having two near-identical dependencies is how they drift.

    Cast rather than isinstance-checked: ``EngineApi`` is a data protocol, and
    ``isinstance`` against one raises at runtime. Starlette's ``app.state`` is
    untyped, so the presence check is the real guard and the cast records what
    we know about what was put there.
    """
    engine = getattr(connection.app.state, "container", None)
    if engine is None:  # pragma: no cover - a wiring bug, not a request error
        raise RuntimeError(
            "no engine on app.state; the app was not built by bootstrap.app.create_app"
        )
    return cast(EngineApi, engine)


def get_trace_id(
    incoming: Annotated[str | None, Header(alias=TRACE_HEADER)] = None,
) -> str:
    """Continue the caller's trace, or start one.

    Honouring an inbound id is what makes NFR-018's correlation work across the
    OratorIA integration: a student's session spans two services, and a trace
    that restarts at our boundary cannot be joined up afterwards.
    """
    return incoming or f"tr_{uuid.uuid4().hex}"


async def get_caller(
    engine: Annotated[EngineApi, Depends(get_engine)],
    trace_id: Annotated[str, Depends(get_trace_id)],
    authorization: Annotated[str | None, Header(alias=AUTHORIZATION)] = None,
) -> AuthenticatedCaller:
    """Resolve the API key to a caller, or refuse with 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    secret = authorization[len("bearer ") :].strip()
    caller = await engine.api_keys.authenticate(secret, trace_id)
    if caller is None:
        # The directory already collapses unknown, revoked and expired into one
        # answer; this preserves that. The key itself is never echoed back or
        # logged (NFR-010).
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return caller


def get_idempotency_key(
    key: Annotated[str | None, Header(alias=IDEMPOTENCY_HEADER)] = None,
) -> str | None:
    """US-011: session creation is idempotent under this key."""
    return key


EngineDep = Annotated[EngineApi, Depends(get_engine)]
CallerDep = Annotated[AuthenticatedCaller, Depends(get_caller)]
TraceDep = Annotated[str, Depends(get_trace_id)]
IdempotencyDep = Annotated[str | None, Depends(get_idempotency_key)]
