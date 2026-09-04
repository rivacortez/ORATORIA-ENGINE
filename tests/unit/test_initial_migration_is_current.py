"""The initial migration cannot drift from the models.

A schema that exists only in the ORM and a migration that has diverged from it
produce the worst kind of failure: everything works in development, where the
tables were created from the models, and the first real deployment builds
something subtly different. The columns are there; a check constraint is not.

This test regenerates the migration and compares. It also asserts the two
properties that make the migration usable at all: it drops in reverse order,
and the DDL still contains the constraints that carry domain rules into the
database.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest
from scripts.render_initial_migration import OUTPUT, render

from evidence_engine.adapters.outbound.persistence.postgres.models import Base


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("initial_migration", OUTPUT)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging bug
        pytest.fail(f"could not load {OUTPUT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_migration_matches_the_models() -> None:
    if not OUTPUT.exists():
        pytest.fail(
            "the initial migration is missing; "
            "run: uv run python scripts/render_initial_migration.py"
        )

    assert OUTPUT.read_text(encoding="utf-8") == render(), (
        "the initial migration is stale. The models changed without it being "
        "regenerated. Run: uv run python scripts/render_initial_migration.py"
    )


def test_every_table_is_created_and_dropped() -> None:
    migration = _load_migration()
    table_names = {table.name for table in Base.metadata.sorted_tables}

    created = " ".join(migration.CREATE_STATEMENTS)
    for name in table_names:
        assert f"CREATE TABLE {name} " in created, f"{name} is never created"

    assert len(migration.DROP_STATEMENTS) == len(table_names)


def test_drops_run_in_reverse_dependency_order() -> None:
    """A table must not be dropped while another still references it."""
    migration = _load_migration()
    creation_order = [table.name for table in Base.metadata.sorted_tables]
    drop_order = [
        statement.split()[4] for statement in migration.DROP_STATEMENTS
    ]  # DROP TABLE IF EXISTS <name> CASCADE

    assert drop_order == list(reversed(creation_order))


def test_the_constraints_that_carry_domain_rules_survive_into_ddl() -> None:
    """The database is the last line, not the only one - but it is a line.

    These three constraints are FR-025, FR-013 and FR-028 expressed in DDL. A
    direct write or a bad migration bypasses every Python check; it does not
    bypass these.
    """
    migration = _load_migration()
    ddl = " ".join(migration.CREATE_STATEMENTS)

    assert "ck_prosody_measured_xor_unavailable" in ddl
    assert "ck_quality_measured_xor_unavailable" in ddl
    assert "ck_availability_reason_required" in ddl
    assert "ck_speech_event_role" in ddl
    assert "ck_cooccurrence_within_window" in ddl


def test_no_causal_inference_column_exists() -> None:
    """FR-028: a column would be somewhere to store a different value."""
    migration = _load_migration()
    ddl = " ".join(migration.CREATE_STATEMENTS).lower()

    assert "causal_inference" not in ddl


def test_the_api_key_table_has_no_column_for_a_secret() -> None:
    """FR-002 and NFR-010: only a hash is persisted, and there is nowhere else."""
    migration = _load_migration()
    api_key_ddl = next(
        statement
        for statement in migration.CREATE_STATEMENTS
        if "CREATE TABLE api_key " in statement
    )

    assert "hashed_secret" in api_key_ddl
    assert "\tsecret " not in api_key_ddl
    assert "plaintext" not in api_key_ddl.lower()
