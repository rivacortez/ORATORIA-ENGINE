"""Give the reproducibility seed a column.

Revision ID: 0002_provenance_seed
Revises: 0001_initial_schema
Create Date: 2026-09-04

Every migration has a symmetric, tested ``downgrade``. A migration that cannot
be reversed is a deployment that cannot be rolled back, and the first time that
matters is the worst time to discover it.

NFR-015 names four things that together reproduce a result: the input, the
artifact, the configuration and the seed. Three of them had columns. This adds
the fourth to the two evidence tables that carry provenance and to the
configuration snapshot that freezes the rules a run used.

**The trap in this repository.** ``0001_initial_schema`` is *rendered from the
models*, not frozen at the moment it was written, and
``tests/unit/test_initial_migration_is_current.py`` fails when the two drift.
So the moment these columns went into ``models.py``, ``0001`` had to be
regenerated and now creates them itself. A delta written as a plain ``ADD
COLUMN`` would therefore succeed against a database built before the columns
existed and fail with ``DuplicateColumn`` against one built from the current
``0001`` - which is every fresh deployment and every CI run. Hence
``IF NOT EXISTS`` throughout: this migration must be a no-op against a database
that already has the columns and a real change against one that does not, and
those are both live cases here. ``ADD CONSTRAINT`` has no ``IF NOT EXISTS``, so
each constraint is dropped first.

**What happens to the rows already there.** They get ``seed_reason =
'not_recorded'``. That is the literal truth about them - they were produced
before anything recorded a seed - and it is not the same statement as
``'deterministic_runtime'``, which would claim those runs are reproducible.
The backfill runs before the check constraint is added, because a row with both
columns null violates it.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_provenance_seed"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The tables that gain the pair, and the name of the constraint tying them
#: together. Kept as data so ``upgrade`` and ``downgrade`` walk the same list:
#: a table added to one and forgotten in the other is the failure the CI
#: round-trip exists to catch, and it is cheaper to make it unrepresentable.
TABLES: tuple[tuple[str, str], ...] = (
    ("speech_event", "ck_speech_event_seed_xor_reason"),
    ("visual_event", "ck_visual_event_seed_xor_reason"),
    ("configuration_snapshot", "ck_configuration_seed_xor_reason"),
)

#: Mirrors ``evidence_engine.domain.shared.provenance.MAX_SEED``. Inlined
#: rather than imported: a migration has to keep running unchanged after the
#: domain moves on, and a constant it imports would silently re-describe a
#: historical schema the day somebody edits the domain.
MAX_SEED = 9007199254740991


def upgrade() -> None:
    for table, constraint in TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS seed BIGINT")
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS seed_reason VARCHAR(40)")
        op.execute(
            f"UPDATE {table} SET seed_reason = 'not_recorded' "
            "WHERE seed IS NULL AND seed_reason IS NULL"
        )
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {constraint} CHECK ("
            f"(seed is not null and seed between 0 and {MAX_SEED} and seed_reason is null) "
            "or (seed is null and seed_reason is not null))"
        )


def downgrade() -> None:
    """Symmetric and tested.

    Reverse order within each table - constraint, then the reason column, then
    the seed - because the constraint names both columns and PostgreSQL refuses
    to drop a column a live constraint depends on unless the drop cascades, and
    a cascading drop of a column is a wider blast radius than this migration
    should ever have.
    """
    for table, constraint in TABLES:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS seed_reason")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS seed")
