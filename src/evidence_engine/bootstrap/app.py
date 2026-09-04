"""The ASGI application factory.

A factory rather than a module-level ``app`` object, so the container can be
built with a frozen clock and scripted runtimes for the contract suite without
patching globals. Phase 2's exit criterion runs against a real application
instance; a module-level singleton would force those tests to reach around it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from evidence_engine.adapters.inbound.rest import errors, health, sessions
from evidence_engine.adapters.inbound.websocket import handler as stream_handler
from evidence_engine.bootstrap.container import SCHEMA_VERSION, Container, build_container
from evidence_engine.bootstrap.settings import Settings

TITLE = "OratorIA Multimodal Evidence Engine"

DESCRIPTION = """
Verbatim transcripts and auditable, timestamped multimodal evidence for
single-speaker presentations.

The engine observes. It does not rank findings, select a top-k, generate
recommendations, or infer emotional or clinical states. `ranking_authority` is
the constant `"none"` on every result.
""".strip()


def create_app(container: Container | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the application, wiring a container if one was not supplied."""
    resolved = container or build_container(settings or Settings())  # type: ignore[call-arg]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = resolved
        yield
        # Nothing to tear down for the in-memory backend. The persistent
        # adapters close their pools here, and putting the hook in now means
        # adding them is not also a change to the startup contract.

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
    )
    # Also set outside the lifespan so a TestClient that never enters it still
    # finds the container.
    app.state.container = resolved

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(stream_handler.router)

    errors.install(app, str(SCHEMA_VERSION))
    return app
