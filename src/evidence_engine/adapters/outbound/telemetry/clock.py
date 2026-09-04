"""Clock adapters.

``SystemClock`` is what production runs on. ``FrozenClock`` is what tests and
QA-05's reproducibility runs use, and it is here rather than under ``tests/``
because the batch replay path needs it too: reprocessing a frozen evaluation
item has to produce equivalent output, and that is impossible while the
pipeline is reading a live clock.

``wall_ms`` uses ``time.monotonic``, not ``time.time``. A wall clock corrected
by NTP mid-session can jump backwards, and a backwards jump produces an
interval that ends before it starts - which the domain refuses, killing the
session over a clock adjustment.
"""

from __future__ import annotations

import time


class SystemClock:
    """The real clock. One monotonic reading, one epoch reading."""

    def wall_ms(self) -> int:
        return int(time.monotonic() * 1_000)

    def epoch_ms(self) -> int:
        return int(time.time() * 1_000)


class FrozenClock:
    """A clock that only moves when told to.

    Both readings advance together, which keeps a test's session clock and its
    audit timestamps consistent with each other - a subtle source of confusion
    when the two drift.
    """

    def __init__(self, start_ms: int = 0) -> None:
        self._now_ms = start_ms

    def wall_ms(self) -> int:
        return self._now_ms

    def epoch_ms(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Move forward. Refuses to move back: the port promises monotonicity."""
        if delta_ms < 0:
            raise ValueError("a monotonic clock cannot be moved backwards")
        self._now_ms += delta_ms
        return self._now_ms

    def set(self, at_ms: int) -> None:
        if at_ms < self._now_ms:
            raise ValueError("a monotonic clock cannot be moved backwards")
        self._now_ms = at_ms
