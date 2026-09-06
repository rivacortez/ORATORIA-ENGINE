"""Composition root. The only package allowed to know every layer.

Guarded, because this is the boundary where the core stops being embeddable.
Everything under here wires FastAPI, SQLAlchemy, Redis and an object store; the
layers below it wire nothing and import nothing. A consumer who installed the
core to read a taxonomy and reached for a container gets a sentence saying the
server extra exists, rather than a missing-module error naming `fastapi` from
inside a factory.

The check runs on import of the package rather than inside `build_container`,
so it fires before a module-scope import of the absent package does - which is
what makes the message reachable at all.
"""

from __future__ import annotations

from evidence_engine._extras import require

require(
    "server",
    "fastapi",
    "sqlalchemy",
    "redis",
    "aioboto3",
    "structlog",
    "pydantic_settings",
    because=(
        "evidence_engine.bootstrap builds the hosted service - REST, WebSocket, "
        "PostgreSQL, Redis and object storage - and those are an optional extra."
    ),
)
