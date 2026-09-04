"""The session clock: captured time, not elapsed time.

FR-009 says capture may be paused and resumed "without resetting the monotonic
session clock", and US-003 says pausing and resuming preserves accumulated
duration. Those two sentences rule out both naive implementations.

Wall-clock difference is wrong: a student who pauses for four minutes to fix a
microphone would have those four minutes counted as silence, wrecking every
rate-per-minute indicator and inventing an enormous silent pause that never
happened. A clock that restarts at zero on resume is worse: event intervals
from the second half would collide with intervals from the first, and FR-026's
single shared timeline stops being single.

So the clock advances only while capture is running, and the value it reports
is the sum of captured segments. Every timestamp in the system - transcript
boundaries, visual windows, fusion distances - is expressed on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.domain.shared.errors import IllegalSessionTransition
from evidence_engine.domain.shared.timeline import MonotonicTime


@dataclass(frozen=True, slots=True)
class SessionClock:
    """Captured milliseconds, resilient to pauses.

    Immutable: every operation returns a new clock. The session aggregate owns
    the current one, so there is exactly one writer and no way for a late
    callback to advance a clock that has already been closed.
    """

    #: Captured milliseconds from segments that have already ended.
    accumulated_ms: int = 0
    #: Reference reading from the injected wall clock at which the running
    #: segment began, or ``None`` while capture is suspended. Stored as a raw
    #: reading rather than a datetime: the domain never reads a clock itself
    #: (it is handed the value), and this keeps that rule visible.
    segment_started_at: int | None = None

    def __post_init__(self) -> None:
        if self.accumulated_ms < 0:
            raise IllegalSessionTransition("accumulated capture time cannot be negative")

    @property
    def is_running(self) -> bool:
        return self.segment_started_at is not None

    def start(self, wall_ms: int) -> SessionClock:
        """Begin or resume capture at the given wall-clock reading."""
        if self.is_running:
            return self
        return SessionClock(accumulated_ms=self.accumulated_ms, segment_started_at=wall_ms)

    def pause(self, wall_ms: int) -> SessionClock:
        """Suspend capture, folding the running segment into the total."""
        if self.segment_started_at is None:
            return self
        segment_ms = max(0, wall_ms - self.segment_started_at)
        return SessionClock(
            accumulated_ms=self.accumulated_ms + segment_ms, segment_started_at=None
        )

    def now(self, wall_ms: int) -> MonotonicTime:
        """The current position on the session clock.

        While paused this returns the same value for every wall reading, which
        is the intended behaviour: nothing was captured in between, so nothing
        on the timeline moved.
        """
        if self.segment_started_at is None:
            return MonotonicTime(self.accumulated_ms)
        segment_ms = max(0, wall_ms - self.segment_started_at)
        return MonotonicTime(self.accumulated_ms + segment_ms)

    def captured_ms(self, wall_ms: int) -> int:
        """Total captured duration - the denominator for every per-minute rate.

        Deliberately not "session duration". A words-per-minute figure computed
        over wall-clock time silently punishes anyone who paused, which is the
        error US-003 exists to prevent.
        """
        return self.now(wall_ms).ms
