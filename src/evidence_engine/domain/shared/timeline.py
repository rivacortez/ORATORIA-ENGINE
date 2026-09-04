"""The shared monotonic timeline.

FR-026 requires speech and visual events to be normalized onto one monotonic
timeline, and NFR-004 requires the engine to disclose how wrong its boundaries
may be. Those two requirements are the whole content of this module.

The unit is milliseconds since the session clock started, as an integer. Not
wall-clock time: a presentation that is paused and resumed (FR-009) keeps
accumulating on the same clock, and wall-clock would jump. Not a float: the
wire format is integer milliseconds (§7.4 ``monotonic_time_ms``) and letting a
float in here would make two runs disagree in the last bits for no gain.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.domain.shared.errors import InvalidInterval

#: Default boundary tolerance when a producer does not state its own.
#: Deliberately not zero: an estimator that declines to say how wrong it might
#: be is not thereby exact, and NFR-004 forbids claiming it is.
DEFAULT_TOLERANCE_MS = 250


@dataclass(frozen=True, slots=True, order=True)
class MonotonicTime:
    """A point on the session clock, in milliseconds since capture started."""

    ms: int

    def __post_init__(self) -> None:
        if self.ms < 0:
            raise InvalidInterval(f"monotonic time cannot be negative: {self.ms}")

    def plus(self, delta_ms: int) -> MonotonicTime:
        return MonotonicTime(self.ms + delta_ms)

    def distance_to(self, other: MonotonicTime) -> int:
        """Absolute distance in milliseconds, used by the fusion window."""
        return abs(self.ms - other.ms)


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open span ``[start, end)`` on the session clock, with tolerance.

    ``tolerance_ms`` is not decoration. Component §5 says the aligner must not
    "claim exact timestamps", and the risk table calls out that timestamp
    claims exceeding model capability make findings unauditable. Carrying the
    tolerance next to the boundary is how a consumer knows whether two events
    2 ms apart are really ordered or just noise.
    """

    start: MonotonicTime
    end: MonotonicTime
    tolerance_ms: int = DEFAULT_TOLERANCE_MS

    def __post_init__(self) -> None:
        if self.end.ms < self.start.ms:
            raise InvalidInterval(
                f"interval ends before it starts: [{self.start.ms}, {self.end.ms})"
            )
        if self.tolerance_ms < 0:
            raise InvalidInterval(f"tolerance cannot be negative: {self.tolerance_ms}")

    @classmethod
    def of(cls, start_ms: int, end_ms: int, tolerance_ms: int = DEFAULT_TOLERANCE_MS) -> Interval:
        """Build from raw milliseconds. Convenience for adapters and tests."""
        return cls(MonotonicTime(start_ms), MonotonicTime(end_ms), tolerance_ms)

    @property
    def duration_ms(self) -> int:
        return self.end.ms - self.start.ms

    @property
    def midpoint(self) -> MonotonicTime:
        return MonotonicTime(self.start.ms + self.duration_ms // 2)

    def overlaps(self, other: Interval) -> bool:
        """True when the two spans share at least one instant.

        Strict overlap, tolerance excluded on purpose: tolerance describes how
        uncertain each boundary is, and folding it in here would silently turn
        "these might overlap" into "these overlap". The fusion engine expresses
        near-misses through its configured window (FR-027) instead, which is a
        versioned decision rather than an implicit one.
        """
        return self.start.ms < other.end.ms and other.start.ms < self.end.ms

    def gap_to(self, other: Interval) -> int:
        """Milliseconds separating two spans; ``0`` when they overlap or touch."""
        if self.overlaps(other):
            return 0
        if self.end.ms <= other.start.ms:
            return other.start.ms - self.end.ms
        return self.start.ms - other.end.ms

    def contains(self, instant: MonotonicTime) -> bool:
        return self.start.ms <= instant.ms < self.end.ms
