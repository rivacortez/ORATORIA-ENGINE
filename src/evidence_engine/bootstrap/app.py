"""The ASGI application factory.

A factory rather than a module-level ``app`` object, so the container can be
built with a frozen clock and scripted runtimes for the contract suite without
patching globals. Phase 2's exit criterion runs against a real application
instance; a module-level singleton would force those tests to reach around it.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from evidence_engine.adapters.inbound.rest import administration, errors, health, sessions
from evidence_engine.adapters.inbound.websocket import handler as stream_handler
from evidence_engine.application.ports.runtimes import AudioWindow
from evidence_engine.bootstrap.container import SCHEMA_VERSION, Container, build_container
from evidence_engine.bootstrap.settings import Settings

#: The engine's working decode rate (16 kHz mono 16-bit PCM) and the warm-up
#: window's length. The same shape `sdk.engine._silent_window` uses for
#: `OratoriaEngine.warmup()` - restated here rather than imported, so this
#: composition root does not acquire a dependency on the SDK for one constant.
_WARM_UP_SAMPLE_RATE_HZ = 16_000
_WARM_UP_DURATION_MS = 200


def _silent_warm_up_window() -> AudioWindow:
    frames = _WARM_UP_SAMPLE_RATE_HZ * _WARM_UP_DURATION_MS // 1_000
    return AudioWindow(
        session_position_ms=0,
        duration_ms=_WARM_UP_DURATION_MS,
        sample_rate_hz=_WARM_UP_SAMPLE_RATE_HZ,
        samples=b"\x00\x00" * frames,
    )


async def _warm_up_speech(container: Container) -> None:
    """Decode 200 ms of silence through the wired speech runtime, once.

    A wired recogniser is not evidence that inference works here - the
    weights can be missing, corrupt, or on a driver too old for a compiled
    kernel - and `/health/ready` refusing traffic until a real decode has
    happened is the server-side twin of why `sdk.engine.OratoriaEngine`
    exposes `warmup()` as a call distinct from construction (ADR-011).

    Not wrapped in ``asyncio.to_thread``: unlike the SDK's embedded path,
    there is no *load* step to move off the loop here - `build_container`
    already built and loaded the runtime synchronously, before uvicorn's
    event loop exists to be blocked by it. The decode call is awaited
    directly because the port is already async, and the one adapter that
    talks to a real GPU (`WhisperSpeechRuntime.transcribe`) already runs its
    blocking work through `asyncio.to_thread` internally.

    Left uncaught on purpose. A caught exception would need a third
    `checks["speech:warm"]` state - "tried and failed" - that nothing in this
    change asks for, and starting a service whose only recogniser cannot
    decode is the condition this project's "never fall back to anything"
    rule exists to refuse loudly rather than serve degraded.
    """
    started = time.monotonic()
    await container.speech.transcribe(_silent_warm_up_window())
    container.speech_warm_seconds = time.monotonic() - started


TITLE = "OratorIA Multimodal Evidence Engine"

DESCRIPTION = """
Verbatim transcripts and auditable, timestamped multimodal evidence for
single-speaker presentations.

The engine observes. It does not rank findings, select a top-k, generate
recommendations, or infer emotional or clinical states. `ranking_authority` is
the constant `"none"` on every result.

### What is measured, and what is absent

Every indicator is either a **measurement** or an explicit **unavailability
with a reason**. There is no third state and no substituted zero: a pitch that
could not be estimated over silence comes back as `unavailable`, not as `0.0`,
because a reader cannot tell a real zero from a missing one and the difference
changes the conclusion (FR-025).

Confidence carries a calibration state. An uncalibrated class reports `raw`,
and a raw confidence never clears a publication threshold - so a detector
nobody has evaluated can report a finding as uncertain and cannot report it as
confirmed.

### The WebSocket is not in this schema

OpenAPI 3.0 describes HTTP. The streaming interface - `/v1/stream` - carries
the audio up and the transcript and evidence down, and its message contract is
in §7.2 of the specification rather than here. The handshake appears; the
messages do not.

### Authentication

`Authorization: Bearer <key>`. Use **Authorize** above to call an endpoint from
this page. The health probes take no credential, deliberately: a readiness
check that needed one would fail closed during exactly the incident it exists
to report.
""".strip()

#: Tag descriptions, so the groups in the docs page say what they are for
#: rather than being three words a reader has to infer a meaning from. Declared
#: here rather than on each router because the order of this list is the order
#: they appear, and that order is a reading order: what the engine can do,
#: then how to start, then how to stream, then how to get evidence out and how
#: to destroy it.
OPENAPI_TAGS = [
    {
        "name": "health",
        "description": (
            "Liveness and readiness. `ready` reports the dependencies and the model "
            "runtimes, so a deployment that is up but cannot reach its database or "
            "has no active model version answers honestly rather than accepting a "
            "session it will fail."
        ),
    },
    {
        "name": "sessions",
        "description": (
            "The lifecycle: create, read, retrieve evidence, delete. A session is "
            "one speaker and one presentation (SCOPE.md fixes version 1 at a single "
            "speaker), and every derived record hangs off a processing run rather "
            "than off the session, so a session processed twice keeps both results "
            "distinguishable."
        ),
    },
    {
        "name": "administration",
        "description": (
            "Provisioning: client applications and the API keys that authenticate "
            "them. **Requires the `admin` scope**, which is a platform-operator "
            "credential rather than a customer one - it is a superset of every "
            "other scope, and this API deliberately refuses to mint another one. "
            "The plaintext key appears in exactly one response and is not "
            "recoverable; only a peppered hash is stored (FR-002)."
        ),
    },
    {
        "name": "streaming",
        "description": (
            "The WebSocket. **OpenAPI 3.0 cannot describe a socket**, so what "
            "appears here is the handshake and nothing else; the message contract "
            "is in the description below and in §7.2 of the specification. Audio "
            "and visual features go up, transcript and evidence come down, and a "
            "finalized event is never revised."
        ),
    },
]

#: The `Authorize` button. Without a declared scheme the docs page renders every
#: endpoint and can call none of them, which is worse than no docs page: it
#: looks like the API is open.
#:
#: `bearerFormat` is deliberately not `JWT`. The engine's keys are opaque
#: secrets checked against a peppered hash, and calling them JWTs would invite a
#: consumer to decode one and read claims that are not there.
API_KEY_SCHEME = {
    "OratorIAApiKey": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "opaque",
        "description": (
            "An API key issued to a tenant application, sent as "
            "`Authorization: Bearer <key>`. Keys carry scopes; an endpoint refuses "
            "a key whose scopes do not cover it, and says which scope was missing."
        ),
    }
}


def create_app(container: Container | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the application, wiring a container if one was not supplied."""
    resolved = container or build_container(settings or Settings())  # type: ignore[call-arg]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = resolved
        await _warm_up_speech(resolved)
        try:
            yield
        finally:
            # The persistent backend holds a connection pool and a Redis
            # client. Leaving them open leaks a pool per container, and the
            # leak surfaces much later as PostgreSQL refusing connections. The
            # memory backend holds nothing and this is a no-op for it.
            await resolved.aclose()

    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=str(SCHEMA_VERSION),
        lifespan=lifespan,
        # The engine speaks one major version at a time (NFR-017). Documenting
        # the surface under its version keeps a consumer's generated client
        # pinned to the contract it was generated from.
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
        # ReDoc alongside Swagger. They are read differently: Swagger is for
        # calling an endpoint, ReDoc for reading the contract end to end, and
        # the second is what a consumer integrating against this will want.
        redoc_url="/v1/redoc",
        openapi_tags=OPENAPI_TAGS,
    )
    # Also set outside the lifespan so a TestClient that never enters it still
    # finds the container.
    app.state.container = resolved

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(administration.router)
    app.include_router(stream_handler.router)

    errors.install(app, str(SCHEMA_VERSION))
    _describe_security(app)
    return app


def _describe_security(app: FastAPI) -> None:
    """Attach the bearer scheme to the generated schema.

    Done by wrapping ``app.openapi`` rather than by adding a dependency to every
    route, because the authentication already happens in a dependency and adding
    a second one to make it *visible* would mean two places that can disagree
    about whether a route is protected.

    The health endpoints stay unauthenticated and are listed as such: a
    readiness probe that needed a credential would fail closed during exactly
    the incident it exists to report.
    """
    generate = app.openapi

    def with_security() -> dict[str, Any]:
        schema = generate()
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {}).update(API_KEY_SCHEME)
        # Applied globally, then lifted from the probes, rather than listed on
        # each protected path: a new endpoint is protected by default, and the
        # exceptions are the thing somebody has to write down.
        schema["security"] = [{"OratorIAApiKey": []}]
        for path, operations in schema.get("paths", {}).items():
            if not path.startswith("/health"):
                continue
            for operation in operations.values():
                operation["security"] = []
        return schema

    app.openapi = with_security  # type: ignore[method-assign]
