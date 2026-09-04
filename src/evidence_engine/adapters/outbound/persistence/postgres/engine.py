"""Async engine and session factory.

One decision worth naming: repositories take an ``async_sessionmaker`` and open
their own unit of work per call, rather than sharing a session injected from a
request scope.

The shared-session pattern is more efficient and is the right default for a CRUD
application. It is wrong here because of the streaming path: a WebSocket
connection lives for the length of a presentation, and a database session held
open for those minutes is a transaction held open for those minutes — a
connection pinned, autovacuum blocked behind its snapshot, and a single slow
session capable of exhausting the pool. The cost of opening a session per call
is a pool checkout; the cost of the alternative is an outage under load.

Where a genuine multi-statement transaction is required, the repository opens
one explicitly and says why.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Build the async engine.

    ``pool_pre_ping`` is on because the pilot deployment sits behind a managed
    database that recycles connections; without it the first request after a
    recycle fails with a stale-connection error that looks like an application
    bug.
    """
    return create_async_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
        # Modest: inference, not the database, is the bottleneck, and a large
        # pool mostly buys the ability to overwhelm the database later.
        pool_size=10,
        max_overflow=5,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """A factory each repository call uses to open its own unit of work.

    ``expire_on_commit=False`` because the domain objects returned are frozen
    dataclasses built from rows, not live ORM instances. Leaving expiry on would
    make SQLAlchemy re-fetch attributes off a closed session.
    """
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def unit_of_work(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """One transaction, committed on success and rolled back on failure.

    Explicit rather than relying on the session's implicit behaviour: a
    repository that forgets to commit silently loses writes, and the failure
    surfaces much later as missing evidence.
    """
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
