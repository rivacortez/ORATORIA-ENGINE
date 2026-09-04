"""The evidence ledger: append-only, revisable only while provisional.

§5 gives the ledger one responsibility - preserve immutable provenance and
revisions - and §6.1 step 9 draws the line: revise freely until a window is
finalized, never afterwards.

The history is the part worth testing hardest. A ledger that kept only the
current value would satisfy every read in the system and quietly fail the one
thing NFR-022 asks of it, because nobody reads the history until an audit does.
"""

from __future__ import annotations

import pytest

from evidence_engine.domain.evidence.ledger import EvidenceLedger
from evidence_engine.domain.shared.errors import EvidenceLedgerViolation
from evidence_engine.domain.shared.identifiers import EventId, RunId
from evidence_engine.domain.shared.provenance import Provenance
from evidence_engine.domain.shared.timeline import Interval

RUN = RunId("run-0001")


@pytest.fixture
def ledger() -> EvidenceLedger:
    return EvidenceLedger(run_id=RUN)


def _append(
    ledger: EvidenceLedger,
    provenance: Provenance,
    event_id: str,
    start_ms: int,
    end_ms: int,
    *,
    is_final: bool = False,
) -> None:
    ledger.append(
        EventId(event_id),
        Interval.of(start_ms, end_ms),
        provenance,
        is_final=is_final,
        payload={"type": "filled_pause"},
    )


# ---------------------------------------------------------------------------
# Appending
# ---------------------------------------------------------------------------


def test_a_new_event_starts_at_revision_zero(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    entry = ledger.append(EventId("ev-1"), Interval.of(0, 400), audio_provenance)

    assert entry.revision == 0
    assert entry.is_final is False
    assert len(ledger) == 1
    assert EventId("ev-1") in ledger


def test_appending_the_same_event_twice_is_refused(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    """A second append would start a parallel history for one event."""
    _append(ledger, audio_provenance, "ev-1", 0, 400)

    with pytest.raises(EvidenceLedgerViolation, match="use revise"):
        _append(ledger, audio_provenance, "ev-1", 0, 500)


def test_the_payload_is_frozen_against_the_caller(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    """A ledger entry a caller can still mutate is not a record of anything."""
    payload = {"type": "filled_pause"}
    entry = ledger.append(EventId("ev-1"), Interval.of(0, 400), audio_provenance, payload=payload)

    payload["type"] = "tampered"

    assert entry.payload["type"] == "filled_pause"
    with pytest.raises(TypeError):
        entry.payload["type"] = "tampered"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Revising
# ---------------------------------------------------------------------------


def test_revising_appends_a_version_rather_than_overwriting(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    _append(ledger, audio_provenance, "ev-1", 0, 400)

    ledger.revise(EventId("ev-1"), Interval.of(0, 780), audio_provenance)

    history = ledger.history(EventId("ev-1"))
    assert [entry.revision for entry in history] == [0, 1]
    assert history[0].interval.end.ms == 400
    assert history[1].interval.end.ms == 780
    assert ledger.current(EventId("ev-1")).revision == 1


def test_a_finalized_event_cannot_be_revised(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    """§6.1 step 9. A consumer that displayed it is never told it never happened."""
    _append(ledger, audio_provenance, "ev-1", 0, 400)
    ledger.finalize(EventId("ev-1"))

    with pytest.raises(EvidenceLedgerViolation, match="never rewritten"):
        ledger.revise(EventId("ev-1"), Interval.of(0, 900), audio_provenance)


def test_revising_an_unknown_event_is_refused(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    with pytest.raises(EvidenceLedgerViolation, match="use append"):
        ledger.revise(EventId("ghost"), Interval.of(0, 400), audio_provenance)


def test_a_revision_can_finalize_in_the_same_step(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    _append(ledger, audio_provenance, "ev-1", 0, 400)

    entry = ledger.revise(EventId("ev-1"), Interval.of(0, 780), audio_provenance, is_final=True)

    assert entry.is_final
    assert len(ledger.history(EventId("ev-1"))) == 2


# ---------------------------------------------------------------------------
# Finalizing
# ---------------------------------------------------------------------------


def test_finalizing_records_a_version_and_keeps_the_content(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    _append(ledger, audio_provenance, "ev-1", 100, 880)

    entry = ledger.finalize(EventId("ev-1"))

    assert entry.is_final
    assert entry.revision == 1
    assert entry.interval.end.ms == 880


def test_finalizing_twice_is_idempotent(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    """Reconciliation can reach the same window twice; that is not an error."""
    _append(ledger, audio_provenance, "ev-1", 0, 400)
    first = ledger.finalize(EventId("ev-1"))

    second = ledger.finalize(EventId("ev-1"))

    assert second == first
    assert len(ledger.history(EventId("ev-1"))) == 2


def test_finalizing_an_unknown_event_is_refused(ledger: EvidenceLedger) -> None:
    with pytest.raises(EvidenceLedgerViolation, match="not in the ledger"):
        ledger.finalize(EventId("ghost"))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_reading_an_unknown_event_is_refused(ledger: EvidenceLedger) -> None:
    with pytest.raises(EvidenceLedgerViolation, match="not in the ledger"):
        ledger.current(EventId("ghost"))


def test_the_history_of_an_unknown_event_is_empty_rather_than_an_error(
    ledger: EvidenceLedger,
) -> None:
    """Asking what happened to nothing is a fair question with a short answer."""
    assert ledger.history(EventId("ghost")) == ()


def test_finalized_returns_only_the_frozen_events(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    _append(ledger, audio_provenance, "ev-final", 0, 400)
    _append(ledger, audio_provenance, "ev-open", 500, 900)
    ledger.finalize(EventId("ev-final"))

    assert [entry.event_id.value for entry in ledger.finalized()] == ["ev-final"]


def test_all_current_hides_nothing(ledger: EvidenceLedger, audio_provenance: Provenance) -> None:
    """FR-029: low-confidence and provisional evidence still leaves the ledger."""
    _append(ledger, audio_provenance, "ev-final", 0, 400)
    _append(ledger, audio_provenance, "ev-open", 500, 900)
    ledger.finalize(EventId("ev-final"))

    assert len(ledger.all_current()) == 2


def test_reads_are_ordered_by_position_then_id(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    """NFR-015: two runs over the same evidence must agree on the order."""
    _append(ledger, audio_provenance, "ev-c", 900, 1_200)
    _append(ledger, audio_provenance, "ev-a", 100, 400)
    _append(ledger, audio_provenance, "ev-b", 100, 400)

    assert [entry.event_id.value for entry in ledger.all_current()] == [
        "ev-a",
        "ev-b",
        "ev-c",
    ]


def test_a_revision_that_moves_an_event_moves_it_in_the_ordering(
    ledger: EvidenceLedger, audio_provenance: Provenance
) -> None:
    _append(ledger, audio_provenance, "ev-a", 100, 400)
    _append(ledger, audio_provenance, "ev-b", 500, 800)

    ledger.revise(EventId("ev-a"), Interval.of(900, 1_200), audio_provenance)

    assert [entry.event_id.value for entry in ledger.all_current()] == ["ev-b", "ev-a"]


def test_provenance_is_preserved_per_version(
    ledger: EvidenceLedger, audio_provenance: Provenance, video_provenance: Provenance
) -> None:
    """A model promotion mid-session must be legible in the history.

    Not a hypothetical: QA-03 canaries a new version, so two versions answer at
    once and the ledger is where "which one produced revision 1?" is answered.
    """
    _append(ledger, audio_provenance, "ev-1", 0, 400)
    ledger.revise(EventId("ev-1"), Interval.of(0, 400), video_provenance)

    history = ledger.history(EventId("ev-1"))
    assert history[0].provenance.model_version != history[1].provenance.model_version


def test_an_empty_ledger_reads_as_empty(ledger: EvidenceLedger) -> None:
    assert len(ledger) == 0
    assert ledger.all_current() == ()
    assert ledger.finalized() == ()
    assert EventId("anything") not in ledger
