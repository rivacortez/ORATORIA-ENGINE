"""The clock port.

The domain never reads a clock. Every timestamp it works with is handed to it,
which is what makes QA-05 possible: reprocessing a frozen evaluation item with
the same artifacts, configuration and seed has to produce equivalent output, and
a ``datetime.now()`` buried three calls deep makes that impossible to guarantee
and impossible to test.

Two readings are exposed because they answer different questions. ``wall_ms`` is
what the session clock converts into positions on the timeline. ``epoch_ms`` is
what audit records and retention deadlines are stamped with - those need to
survive a process restart, which a monotonic reading does not.
"""

from __future__ import annotations

from typing import Protocol


class Clock(Protocol):
    """Reads of time, injected rather than taken."""

    def wall_ms(self) -> int:
        """A monotonically non-decreasing reading in milliseconds.

        Used to advance the session clock. Must not jump backwards when the
        host's wall clock is corrected, because a backwards jump would produce
        an interval that ends before it starts and the domain refuses those.
        """
        ...

    def epoch_ms(self) -> int:
        """Milliseconds since the Unix epoch, UTC.

        For audit records, consent receipts and retention deadlines: values a
        human or another system has to interpret later.
        """
        ...
