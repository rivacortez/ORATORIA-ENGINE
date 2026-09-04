"""Render the initial migration from the models.

Only the *initial* migration is generated. There is no prior schema to diff
against, so hand-writing 17 `CREATE TABLE` statements would be transcription
with a typo budget. Every subsequent migration is a handwritten delta with a
tested ``downgrade``, because a delta encodes intent — which column moved
where, and what happens to the rows already in it — and nothing can infer that.

``tests/unit/test_initial_migration_is_current.py`` fails when the checked-in
migration no longer matches the models, so the two cannot drift apart before
anyone has run it.

    uv run python scripts/render_initial_migration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from evidence_engine.adapters.outbound.persistence.postgres.models import Base

REVISION = "0001_initial_schema"
OUTPUT = Path(__file__).resolve().parents[1] / "alembic" / "versions" / f"{REVISION}.py"

TEMPLATE = '''"""Initial schema.

Revision ID: {revision}
Revises:
Create Date: 2026-09-04

GENERATED FILE - do not edit by hand.

Source: src/evidence_engine/adapters/outbound/persistence/postgres/models.py
Regenerate: uv run python scripts/render_initial_migration.py
Guarded by: tests/unit/test_initial_migration_is_current.py

Only the initial migration is generated. Every later one is a handwritten delta
with a tested downgrade, because a delta encodes intent - which column moved
where, and what happens to the rows already in it - and that cannot be inferred
from a schema diff.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "{revision}"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Rendered from the declarative models for the PostgreSQL dialect, in
#: dependency order so foreign keys resolve.
CREATE_STATEMENTS: tuple[str, ...] = (
{creates}
)

#: Reverse order, so a table is never dropped while another still references it.
DROP_STATEMENTS: tuple[str, ...] = (
{drops}
)


def upgrade() -> None:
    for statement in CREATE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    """Symmetric and tested.

    A migration that cannot be reversed is a deployment that cannot be rolled
    back, and the first time that matters is the worst time to find out.
    """
    for statement in DROP_STATEMENTS:
        op.execute(statement)
'''


def _statements() -> tuple[list[str], list[str]]:
    dialect = postgresql.dialect()
    creates: list[str] = []
    drops: list[str] = []

    for table in Base.metadata.sorted_tables:
        creates.append(str(CreateTable(table).compile(dialect=dialect)).strip())
        for index in sorted(table.indexes, key=lambda i: i.name or ""):
            creates.append(str(CreateIndex(index).compile(dialect=dialect)).strip())

    for table in reversed(Base.metadata.sorted_tables):
        drops.append(f"DROP TABLE IF EXISTS {table.name} CASCADE")

    return creates, drops


def _render_literal(statement: str) -> str:
    body = "\n".join(f"        {line.rstrip()}" for line in statement.splitlines())
    return f'    """\n{body}\n    """,'


def render() -> str:
    creates, drops = _statements()
    return TEMPLATE.format(
        revision=REVISION,
        creates="\n".join(_render_literal(s) for s in creates),
        drops="\n".join(f'    "{s}",' for s in drops),
    )


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
