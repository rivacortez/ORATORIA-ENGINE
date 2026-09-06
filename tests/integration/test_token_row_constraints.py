"""`word_token` records a seed or the reason it has none, never both.

Every other provenance-bearing row - speech events, visual events, prosody
readings, configuration snapshots - carried `_seed_xor_reason` in DDL. Tokens
gained their provenance columns last and were the one row without it, and
`seed_from_columns` returns the seed when both are set and drops the reason
without a word. The constraint is what keeps a direct write honest; the mapper
never produces such a row, which is exactly why nothing else would notice.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidence_engine.adapters.outbound.persistence.postgres import models
from evidence_engine.adapters.outbound.persistence.postgres.engine import unit_of_work
from tests.integration.test_postgres_adapters import TENANT, _seed_run

pytestmark = [pytest.mark.integration]


def _token(**overrides: object) -> models.WordTokenRow:
    columns: dict[str, object] = {
        "id": "tok-seeded",
        "run_id": "run-1",
        "tenant_id": TENANT.value,
        "raw_text": "hola",
        "sequence_window_ms": 0,
        "sequence_index": 0,
        "start_ms": 0,
        "end_ms": 300,
        "tolerance_ms": 20,
        "confidence_unavailable_reason": "posterior_not_reported",
        "model_version": "m-1",
        "taxonomy_version": "1.0.0",
        "configuration_id": "c-1",
        "evidence_ref": "e-1",
        "status": "final",
    }
    columns.update(overrides)
    return models.WordTokenRow(**columns)


async def test_a_token_row_cannot_hold_both_a_seed_and_a_reason(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_run(factory)

    with pytest.raises(IntegrityError, match="ck_token_seed_xor_reason"):
        async with unit_of_work(factory) as db:
            db.add(_token(seed=7, seed_reason="deterministic_runtime"))


async def test_a_token_row_cannot_hold_neither(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_run(factory)

    with pytest.raises(IntegrityError, match="ck_token_seed_xor_reason"):
        async with unit_of_work(factory) as db:
            db.add(_token(seed=None, seed_reason=None))


async def test_a_token_row_with_exactly_one_is_accepted(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """The constraint refuses the two bad shapes and nothing else."""
    await _seed_run(factory)

    async with unit_of_work(factory) as db:
        db.add(_token(id="tok-seeded", seed=7, seed_reason=None))
        db.add(_token(id="tok-unseeded", seed=None, seed_reason="deterministic_runtime"))
