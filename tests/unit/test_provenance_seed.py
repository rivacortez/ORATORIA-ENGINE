"""NFR-015's fourth term, at every layer that has to carry it.

The seed was named in three docstrings and in no dataclass, column or wire
field, so the reproducibility contract was three-quarters representable. These
tests hold the missing quarter in place, and each one is written against a
*plausible* weaker implementation rather than against the absence of the
feature: an ``int | None`` seed, a single "no seed" case, a wire key that
renders ``null``, a downgrade that forgets a table. Reintroducing any of those
turns one of these red.

The database half is asserted against the declarative metadata rather than a
live server. The check constraint was also exercised against a real PostgreSQL
- it refuses both-null, both-set, negative and above-2**53, and accepts seed 0
- but that needs a container, and a rule this cheap to break should fail in the
unit suite too.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from evidence_engine.adapters.inbound.rest.serialization import render_speech_event
from evidence_engine.adapters.outbound.persistence.postgres import models
from evidence_engine.adapters.outbound.persistence.postgres.mapping import (
    row_to_speech_event,
    seed_from_columns,
    seed_to_columns,
    speech_event_to_row,
)
from evidence_engine.bootstrap.container import default_configuration
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import FabricatedValue
from evidence_engine.domain.shared.identifiers import (
    ConfigurationSnapshotId,
    EventId,
    EvidenceRef,
    ModelVersionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import (
    MAX_SEED,
    Modality,
    ModelRole,
    Provenance,
    ProvenanceViolation,
    Seeded,
    Unseeded,
    UnseededReason,
)
from evidence_engine.domain.shared.taxonomy import TAXONOMY_VERSION, SpeechEventType
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.events import SpeechEvent

MIGRATION = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "0002_provenance_seed.py"

#: The tables that carry the pair. Named here rather than imported from the
#: migration, so a table quietly dropped from the migration's own list is a
#: failure rather than a definition change both sides agree on.
SEEDED_TABLES = ("speech_event", "visual_event", "configuration_snapshot")


def _provenance(seed: Seeded | Unseeded | None = None) -> Provenance:
    fields = {
        "modality": Modality.AUDIO,
        "role": ModelRole.RECOGNISER,
        "model_version": ModelVersionId("asr-baseline-0001"),
        "taxonomy_version": TAXONOMY_VERSION,
        "configuration": ConfigurationSnapshotId("config-0001"),
        "evidence_ref": EvidenceRef("evidence-audio-0001"),
    }
    return Provenance(**fields) if seed is None else Provenance(**fields, seed=seed)


def _event(provenance: Provenance) -> SpeechEvent:
    return SpeechEvent(
        id=EventId("event-0001"),
        type=SpeechEventType.FILLED_PAUSE,
        interval=Interval.of(900, 1_680),
        confidence=Confidence.calibrated(0.93),
        provenance=provenance,
        raw_text="eeeh",
    )


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("seed_migration", MIGRATION)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging bug
        pytest.fail(f"could not load {MIGRATION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The domain type
# ---------------------------------------------------------------------------


def test_seed_zero_is_not_the_same_claim_as_no_seed() -> None:
    """The distinction the whole type exists for.

    Under ``int | None`` these two are ``0`` and ``None``, one ``or`` apart.
    """
    assert Seeded(0) != Unseeded(UnseededReason.NOT_RECORDED)
    assert Seeded(0).is_recorded
    assert not Unseeded(UnseededReason.NOT_RECORDED).is_recorded


def test_an_unseeded_provenance_has_no_seed_to_read() -> None:
    """``seed or 0`` must not be reachable.

    A rerun under a substituted 0 would complete and report a match against a
    result produced under something else, and nothing downstream would see the
    disagreement.
    """
    unseeded = Unseeded(UnseededReason.DETERMINISTIC_RUNTIME)

    with pytest.raises(FabricatedValue, match="NFR-015"):
        _ = unseeded.value


def test_the_two_reasons_for_having_no_seed_do_not_collapse() -> None:
    """ "Nothing to seed" and "nobody wrote it down" are opposite claims.

    The first says the result is reproducible; the second says nobody can tell.
    A single "no seed" case would make the second read like the first.
    """
    deterministic = Unseeded(UnseededReason.DETERMINISTIC_RUNTIME)
    unrecorded = Unseeded(UnseededReason.NOT_RECORDED)

    assert deterministic != unrecorded
    assert deterministic.reason.value != unrecorded.reason.value


@pytest.mark.parametrize("value", [-1, MAX_SEED + 1, 2**63])
def test_a_seed_that_would_not_survive_the_journey_is_refused(value: int) -> None:
    """Refused at construction, which is the last point it is still itself.

    Above 2**53 a JSON consumer parses two distinct seeds as one number; below
    zero a numpy-seeded runtime refuses the value outright. Either way the
    record would describe a reproduction that cannot be run or that would run
    under a different seed and still report a match.
    """
    with pytest.raises(ProvenanceViolation):
        Seeded(value)


@pytest.mark.parametrize("value", [0, 1, 42, MAX_SEED])
def test_the_representable_range_is_accepted_whole(value: int) -> None:
    assert Seeded(value).value == value


def test_a_provenance_built_without_a_seed_says_so_rather_than_claiming_determinism() -> None:
    """The default is the understated claim, not the flattering one.

    ``DETERMINISTIC_RUNTIME`` is true of every runtime shipping today and
    becomes false, silently and for every event, the day a stochastic one
    forgets to pass its seed. ``NOT_RECORDED`` cannot become false: a
    ``Provenance`` built without naming a seed did not record one.
    """
    assert _provenance().seed == Unseeded(UnseededReason.NOT_RECORDED)


def test_the_shipped_configuration_states_its_seed_policy() -> None:
    """Both shipped runtimes replay a script, so there is nothing to seed.

    Stated rather than left empty: the line has to change on the day that stops
    being true, and an empty field would let a reader assume either answer.
    """
    assert default_configuration().seed == Unseeded(UnseededReason.DETERMINISTIC_RUNTIME)


# ---------------------------------------------------------------------------
# The columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_name", SEEDED_TABLES)
def test_every_table_that_carries_a_seed_carries_its_reason_and_the_rule(table_name: str) -> None:
    """A bare nullable ``seed`` column would be the ``int | None`` bug in DDL.

    The constraint is the database-side half: exactly one of the two columns is
    populated, and the upper bound travels with it, so a direct write cannot
    produce a row the domain would refuse to construct.
    """
    table = models.Base.metadata.tables[table_name]

    assert "seed" in table.c
    assert "seed_reason" in table.c

    checks = [str(c.sqltext) for c in table.constraints if hasattr(c, "sqltext")]
    seed_checks = [text for text in checks if "seed" in text]
    assert seed_checks, f"{table_name} has a seed column and no rule tying it to its reason"
    assert f"seed between 0 and {MAX_SEED}" in seed_checks[0]


def test_a_seed_survives_the_row_round_trip() -> None:
    event = _event(_provenance(Seeded(20_260_904)))

    row = speech_event_to_row(event, run_id="run-0001", tenant=TenantId("tenant-0001"))

    assert row.seed == 20_260_904
    assert row.seed_reason is None
    assert row_to_speech_event(row).provenance.seed == Seeded(20_260_904)


def test_the_reason_survives_the_row_round_trip_without_becoming_a_number() -> None:
    """The reason is what a reader needs, and it is the part easiest to lose.

    A mapping that wrote ``seed = 0`` for an unseeded event, or that collapsed
    both reasons on the way back, would produce rows that look right and mean
    something else.
    """
    event = _event(_provenance(Unseeded(UnseededReason.DETERMINISTIC_RUNTIME)))

    row = speech_event_to_row(event, run_id="run-0001", tenant=TenantId("tenant-0001"))

    assert row.seed is None
    assert row.seed_reason == "deterministic_runtime"
    assert row_to_speech_event(row).provenance.seed == Unseeded(
        UnseededReason.DETERMINISTIC_RUNTIME
    )


def test_a_row_written_before_the_columns_existed_reads_as_not_recorded() -> None:
    """Both columns null is what a pre-migration row looks like.

    ``NOT_RECORDED`` is the literal truth about it. Reading it as
    ``DETERMINISTIC_RUNTIME`` would claim those historical runs are
    reproducible, which is the claim nobody can check any more.
    """
    assert seed_from_columns(None, None) == Unseeded(UnseededReason.NOT_RECORDED)


def test_a_reason_the_code_does_not_know_is_refused_rather_than_downgraded() -> None:
    """Folding it into ``NOT_RECORDED`` would report absence about a record.

    A database written by a newer schema than the running code is a deployment
    problem, and reporting "no seed was recorded" about a row that plainly
    recorded one would hide it behind a plausible answer.
    """
    with pytest.raises(ValueError, match="stochastic_undisclosed"):
        seed_from_columns(None, "stochastic_undisclosed")


def test_the_columns_are_written_exclusively() -> None:
    """Never both, never neither - the same rule ``prosody_to_row`` follows."""
    assert seed_to_columns(Seeded(7)) == (7, None)
    assert seed_to_columns(Unseeded(UnseededReason.NOT_RECORDED)) == (None, "not_recorded")


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------


def test_the_wire_carries_the_seed_and_never_a_null_in_its_place() -> None:
    """``"value": null`` is what ``seed ?? 0`` feeds on.

    The unavailable branch has no ``value`` key at all, for the reason
    ``_render_indicator`` has none: a null in a numeric field is what a
    consumer coerces, and 0 is a seed a run could genuinely have used.
    """
    seeded = render_speech_event(_event(_provenance(Seeded(42))))["provenance"]["seed"]
    unseeded = render_speech_event(
        _event(_provenance(Unseeded(UnseededReason.DETERMINISTIC_RUNTIME)))
    )["provenance"]["seed"]

    assert seeded == {"recorded": True, "value": 42}
    assert unseeded == {"recorded": False, "reason": "deterministic_runtime"}
    assert "value" not in unseeded


def test_the_seed_is_always_present_on_the_wire() -> None:
    """Like ``causal_inference``: read, never inferred from a missing key."""
    for provenance in (_provenance(Seeded(0)), _provenance()):
        assert "seed" in render_speech_event(_event(provenance))["provenance"]


# ---------------------------------------------------------------------------
# The migration
# ---------------------------------------------------------------------------


def _emitted(direction: str) -> list[str]:
    """The SQL the migration actually issues, without a database.

    Asserted against the emitted statements rather than the file's text: the
    migration builds its DDL in a loop, so nothing a reader wants to check
    appears literally in the source, and a test that greps the source would
    pass on a docstring and fail on a refactor that changed nothing.
    """
    migration = _load_migration()
    recorded: list[str] = []

    class _Recorder:
        @staticmethod
        def execute(statement: str) -> None:
            recorded.append(" ".join(statement.split()))

    migration.op = _Recorder()
    getattr(migration, direction)()
    return recorded


@pytest.mark.parametrize("table_name", SEEDED_TABLES)
def test_the_migration_adds_the_pair_to_every_table_that_carries_it(table_name: str) -> None:
    emitted = _emitted("upgrade")

    assert f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS seed BIGINT" in emitted
    assert f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS seed_reason VARCHAR(40)" in emitted


@pytest.mark.parametrize("table_name", SEEDED_TABLES)
def test_the_migration_is_reversible_table_for_table(table_name: str) -> None:
    """A downgrade that forgets one table is a rollback that half happens.

    The CI round-trip would still pass - the drop is not the failing step - and
    the next upgrade would meet a table already carrying the constraint.
    """
    emitted = _emitted("downgrade")

    assert f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS seed" in emitted
    assert f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS seed_reason" in emitted


def test_the_constraint_is_dropped_before_the_columns_it_names() -> None:
    """PostgreSQL refuses to drop a column a live constraint depends on.

    Only a cascading drop would get past it, and a cascade here has a far wider
    blast radius than this migration should ever have.
    """
    emitted = _emitted("downgrade")

    for table, constraint in _load_migration().TABLES:
        drop_constraint = emitted.index(
            f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}"
        )
        drop_column = emitted.index(f"ALTER TABLE {table} DROP COLUMN IF EXISTS seed")
        assert drop_constraint < drop_column


def test_the_backfill_does_not_claim_the_historical_runs_were_deterministic() -> None:
    """Existing rows get ``not_recorded``, which is the only checkable claim.

    Backfilling ``deterministic_runtime`` would assert that every result
    already in the ledger is reproducible - about runs nobody can check.
    """
    emitted = _emitted("upgrade")
    updates = [statement for statement in emitted if statement.startswith("UPDATE ")]

    assert len(updates) == len(SEEDED_TABLES)
    for statement in updates:
        assert "SET seed_reason = 'not_recorded'" in statement
        assert "WHERE seed IS NULL AND seed_reason IS NULL" in statement
    assert not [statement for statement in emitted if "deterministic_runtime" in statement]


def test_the_backfill_runs_before_the_rule_that_would_reject_it() -> None:
    """A pre-migration row has both columns null, which the constraint forbids.

    Adding the constraint first would abort the upgrade on any database that
    already holds a row - which is the only kind of database this migration is
    for.
    """
    emitted = _emitted("upgrade")

    for table, constraint in _load_migration().TABLES:
        backfill = next(i for i, s in enumerate(emitted) if s.startswith(f"UPDATE {table} "))
        add_constraint = next(
            i for i, s in enumerate(emitted) if f"ADD CONSTRAINT {constraint}" in s
        )
        assert backfill < add_constraint


def test_the_delta_is_a_no_op_against_a_database_that_already_has_the_columns() -> None:
    """The initial migration is *rendered from the models*, not frozen.

    ``tests/unit/test_initial_migration_is_current.py`` fails when the two
    drift, so ``0001`` had to be regenerated and now creates these columns
    itself. A plain ``ADD COLUMN`` here would fail against every fresh
    deployment and every CI run while succeeding against a database built
    before them, and both are live cases.
    """
    emitted = _emitted("upgrade")
    additions = [statement for statement in emitted if " ADD COLUMN" in statement]

    assert len(additions) == 2 * len(SEEDED_TABLES)
    assert all("ADD COLUMN IF NOT EXISTS" in statement for statement in additions)
    # ADD CONSTRAINT has no IF NOT EXISTS, so it is dropped first instead.
    assert all(
        f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}" in emitted
        for table, constraint in _load_migration().TABLES
    )
