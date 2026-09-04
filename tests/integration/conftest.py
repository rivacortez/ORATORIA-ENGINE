"""Integration fixtures: the real PostgreSQL, Redis and S3 adapters.

Skipped when the infrastructure is not up, rather than failing. The contract
suite covers behaviour and runs everywhere; these tests answer a narrower
question — do the persistent adapters behave like the in-memory ones the
contracts were written against? — and that question has no answer without a
database, so a red test would be reporting on the developer's laptop rather
than on the code.

``docker compose up -d`` then ``alembic upgrade head`` makes them run.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidence_engine.adapters.outbound.persistence.postgres.engine import (
    create_engine,
    create_session_factory,
)
from evidence_engine.adapters.outbound.persistence.postgres.models import Base

DATABASE_URL = os.environ.get(
    "ENGINE_TEST_DATABASE_URL",
    "postgresql+asyncpg://engine:engine@localhost:5432/engine",
)
REDIS_URL = os.environ.get("ENGINE_TEST_REDIS_URL", "redis://localhost:6379/1")

pytestmark = pytest.mark.integration


#: How long to wait for the infrastructure before deciding it is not there.
#: Short on purpose: without it, the driver spends its own generous default
#: retrying a refused connection, and a developer with no Docker running waits
#: most of a minute to be told the tests were skipped.
PROBE_TIMEOUT_SECONDS = 2.0


#: Probed once per session. Without the cache the two-second timeout is paid
#: by every test in the module, which turns a skip into half a minute.
_database_available: bool | None = None


async def _database_is_up() -> bool:
    global _database_available
    if _database_available is not None:
        return _database_available
    _database_available = await _probe_database()
    return _database_available


async def _probe_database() -> bool:
    engine = create_engine(DATABASE_URL)
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS), engine.connect():
            pass
    except Exception:
        # Any failure means "not available". Distinguishing a refused
        # connection from a missing database would only change which message
        # the skip carries.
        return False
    finally:
        await engine.dispose()
    return True


@pytest_asyncio.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory against a schema created and dropped per test.

    ``create_all`` rather than running the migration: these tests are about the
    adapters, and a migration failure should surface in the migration's own
    test rather than as a confusing error in twenty unrelated ones. The
    migration and the models are kept in step by
    ``test_initial_migration_is_current``.
    """
    if not await _database_is_up():
        pytest.skip(f"no database at {DATABASE_URL}; run: docker compose up -d")

    engine = create_engine(DATABASE_URL)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    yield create_session_factory(engine)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[object]:
    """A Redis client on a scratch database, flushed around each test."""
    from redis.asyncio import Redis

    client: Redis = Redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=PROBE_TIMEOUT_SECONDS
    )
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        pytest.skip(f"no Redis at {REDIS_URL}; run: docker compose up -d")

    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()
