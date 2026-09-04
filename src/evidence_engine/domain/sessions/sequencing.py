"""Chunk sequencing: detecting loss, duplication and reordering.

FR-008 requires the service to detect missing, duplicated and out-of-order
chunks, and §7.4 requires duplicate client messages to be idempotent. §6.3
adds that a missing chunk must be *detected through* ``chunk_seq`` rather than
inferred from a suspicious silence - which is the point of the whole mechanism.
A gap in the audio and a gap in the network look identical downstream, and
only the sequence numbers tell them apart. Confusing the two would have the
engine report a long silent pause the speaker never took.

The ledger keeps gaps rather than merely counting them. A quality assessment
has to say *which* stretch of the timeline is unreliable, because FR-024 scopes
availability per indicator and per window, not per session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ChunkVerdict(StrEnum):
    """What the ledger decided about an arriving chunk."""

    #: Next in sequence. Process it.
    ACCEPTED = "accepted"
    #: Already seen. Ignore it; §7.4 makes redelivery idempotent, not an error.
    DUPLICATE = "duplicate"
    #: Ahead of the expected number: one or more chunks are missing.
    GAP_DETECTED = "gap_detected"
    #: Behind the highest number seen. Late arrival after a gap was recorded.
    OUT_OF_ORDER = "out_of_order"


@dataclass(frozen=True, slots=True)
class SequenceGap:
    """A contiguous run of sequence numbers that never arrived."""

    first_missing: int
    last_missing: int

    @property
    def count(self) -> int:
        return self.last_missing - self.first_missing + 1


@dataclass(slots=True)
class ChunkLedger:
    """Tracks arrival of a modality's chunks within one session.

    Mutable by design, and the only mutable object in the domain. A streaming
    session receives thousands of chunks per minute, and rebuilding an
    immutable ledger per chunk would allocate for no benefit: the ledger has a
    single owner (the session's streaming coordinator) and no meaningful
    history to preserve beyond the gaps it already records.
    """

    #: Highest sequence number accepted so far; ``-1`` before the first chunk.
    highest_seen: int = -1
    #: Sequence numbers observed out of order and still awaiting their gap fill.
    _seen_ahead: set[int] = field(default_factory=set)
    #: Sequence numbers accepted, kept only while a gap could still be filled.
    _accepted: set[int] = field(default_factory=set)
    gaps: list[SequenceGap] = field(default_factory=list)
    duplicates: int = 0

    def offer(self, sequence: int) -> ChunkVerdict:
        """Record an arriving chunk and say what should happen to it."""
        if sequence < 0:
            raise ValueError(f"chunk sequence must be non-negative, got {sequence}")

        if sequence in self._accepted:
            self.duplicates += 1
            return ChunkVerdict.DUPLICATE

        expected = self.highest_seen + 1

        if sequence == expected:
            self._accept(sequence)
            self._drain_ahead()
            return ChunkVerdict.ACCEPTED

        if sequence > expected:
            # Do not record the gap yet: the missing chunks may still arrive
            # out of order over an unreliable transport. The gap is only
            # confirmed when the session closes (see `finalize`), which is what
            # keeps a reordered network from being reported as data loss.
            self._seen_ahead.add(sequence)
            self._accept(sequence)
            return ChunkVerdict.GAP_DETECTED

        self._accept(sequence)
        return ChunkVerdict.OUT_OF_ORDER

    def _accept(self, sequence: int) -> None:
        self._accepted.add(sequence)
        self.highest_seen = max(self.highest_seen, sequence)

    def _drain_ahead(self) -> None:
        """Advance past chunks that arrived early once their gap is filled."""
        while self.highest_seen + 1 in self._seen_ahead:
            self._seen_ahead.discard(self.highest_seen + 1)
            self.highest_seen += 1

    def finalize(self) -> tuple[SequenceGap, ...]:
        """Confirm which sequence numbers never arrived.

        Called once at session close. Everything before this point is
        provisional because reordering is normal; treating the first
        out-of-order arrival as loss would fill a healthy session with false
        quality warnings.
        """
        self.gaps = list(_collapse(sorted(set(range(self.highest_seen + 1)) - self._accepted)))
        return tuple(self.gaps)

    @property
    def missing_count(self) -> int:
        return sum(gap.count for gap in self.gaps)


def _collapse(missing: list[int]) -> list[SequenceGap]:
    """Turn a sorted list of missing numbers into contiguous runs."""
    if not missing:
        return []
    runs: list[SequenceGap] = []
    start = previous = missing[0]
    for number in missing[1:]:
        if number == previous + 1:
            previous = number
            continue
        runs.append(SequenceGap(start, previous))
        start = previous = number
    runs.append(SequenceGap(start, previous))
    return runs
