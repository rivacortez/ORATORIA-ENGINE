"""The evidence ledger: append-only, revisable only while provisional.

§5 gives the ledger one responsibility - preserve immutable provenance and
revisions - and one prohibition: it must not select top-k recommendations. Both
are load-bearing and this module implements exactly them and nothing else.

"Revisions" is the interesting word. A streaming session revises constantly:
the recognizer re-decodes its active window, a detector changes its mind as
more right context arrives, a confidence is recalibrated. §6.1 step 9 permits
all of that right up to the moment a window is finalized, and forbids it
afterwards. The ledger therefore keeps a history rather than a current value:
a revision appends a new version and the previous one stays readable, which is
what makes NFR-022's append-only audit trail true of the evidence itself and
not only of administrative actions.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from evidence_engine.domain.shared.errors import EvidenceLedgerViolation
from evidence_engine.domain.shared.identifiers import EventId, RunId
from evidence_engine.domain.shared.provenance import Provenance
from evidence_engine.domain.shared.timeline import Interval


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One version of one event.

    ``revision`` counts from zero. A consumer that saw revision 2 and is handed
    revision 3 knows the finding changed; one that is handed revision 2 again
    knows it did not, which is how §7.4's idempotent redelivery is verifiable
    rather than merely promised.
    """

    event_id: EventId
    revision: int
    interval: Interval
    provenance: Provenance
    is_final: bool
    #: The event's own payload, kept opaque here. The ledger's job is identity,
    #: ordering and provenance; interpreting a payload would make it a second
    #: home for detector logic and give the modality packages a shared
    #: dependency that contract C5 exists to prevent.
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceLedger:
    """Append-only history of every event produced by one processing run.

    Mutable, unlike the rest of the domain, and for the same reason the chunk
    ledger is: it accumulates thousands of entries under a single owner during
    a live session. What it never does is *lose* an entry - append and revise
    both grow the history, and nothing removes from it.
    """

    run_id: RunId
    _history: dict[EventId, list[LedgerEntry]] = field(default_factory=dict)

    def append(
        self,
        event_id: EventId,
        interval: Interval,
        provenance: Provenance,
        *,
        is_final: bool = False,
        payload: Mapping[str, object] | None = None,
    ) -> LedgerEntry:
        """Record the first version of an event."""
        if event_id in self._history:
            raise EvidenceLedgerViolation(
                f"event {event_id} is already in the ledger; use revise() to change it"
            )
        entry = LedgerEntry(
            event_id=event_id,
            revision=0,
            interval=interval,
            provenance=provenance,
            is_final=is_final,
            payload=MappingProxyType(dict(payload or {})),
        )
        self._history[event_id] = [entry]
        return entry

    def revise(
        self,
        event_id: EventId,
        interval: Interval,
        provenance: Provenance,
        *,
        is_final: bool = False,
        payload: Mapping[str, object] | None = None,
    ) -> LedgerEntry:
        """Record a new version of a still-provisional event.

        Refused once the event is final. This is the enforcement point for
        §6.1 step 9: a consumer that has displayed a finalized finding to a
        student will never be told it was withdrawn.
        """
        versions = self._history.get(event_id)
        if versions is None:
            raise EvidenceLedgerViolation(
                f"event {event_id} is not in the ledger; use append() for a new event"
            )
        current = versions[-1]
        if current.is_final:
            raise EvidenceLedgerViolation(
                f"event {event_id} was finalized at revision {current.revision}; "
                "finalized history is never rewritten"
            )
        entry = LedgerEntry(
            event_id=event_id,
            revision=current.revision + 1,
            interval=interval,
            provenance=provenance,
            is_final=is_final,
            payload=MappingProxyType(dict(payload or {})),
        )
        versions.append(entry)
        return entry

    def finalize(self, event_id: EventId) -> LedgerEntry:
        """Freeze an event at its current content."""
        versions = self._history.get(event_id)
        if versions is None:
            raise EvidenceLedgerViolation(f"event {event_id} is not in the ledger")
        current = versions[-1]
        if current.is_final:
            return current
        entry = LedgerEntry(
            event_id=event_id,
            revision=current.revision + 1,
            interval=current.interval,
            provenance=current.provenance,
            is_final=True,
            payload=current.payload,
        )
        versions.append(entry)
        return entry

    # -- reads ------------------------------------------------------------

    def current(self, event_id: EventId) -> LedgerEntry:
        versions = self._history.get(event_id)
        if versions is None:
            raise EvidenceLedgerViolation(f"event {event_id} is not in the ledger")
        return versions[-1]

    def history(self, event_id: EventId) -> tuple[LedgerEntry, ...]:
        """Every version, oldest first. The audit trail for one finding."""
        return tuple(self._history.get(event_id, ()))

    def finalized(self) -> tuple[LedgerEntry, ...]:
        """Current versions of every finalized event, in deterministic order."""
        return tuple(entry for entry in self._ordered_current() if entry.is_final)

    def all_current(self) -> tuple[LedgerEntry, ...]:
        """Current versions of every event, finalized or not.

        FR-029 forbids hiding low-confidence evidence, and §5 forbids the
        result service from doing so either. The complete set is what leaves
        here; deciding what to show a student is the consumer's call.
        """
        return tuple(self._ordered_current())

    def _ordered_current(self) -> Iterator[LedgerEntry]:
        """Ordered by interval, then id - stable across runs (NFR-015)."""
        entries = [versions[-1] for versions in self._history.values()]
        entries.sort(key=lambda e: (e.interval.start.ms, e.interval.end.ms, e.event_id.value))
        return iter(entries)

    def __len__(self) -> int:
        return len(self._history)

    def __contains__(self, event_id: object) -> bool:
        return event_id in self._history
