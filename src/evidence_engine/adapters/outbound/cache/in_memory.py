"""In-memory ephemeral state, quotas and the outbound event channel.

The Redis adapter's semantics, without Redis. Two behaviours are modelled
carefully because getting them wrong here would let a bug pass CI and fail in
production:

*Leases expire.* A handler that dies holding a session lease must not lock that
session out forever, so the lease carries a deadline and a later acquirer takes
it once the deadline passes. An in-memory lease that never expired would make
the recovery path untested.

*Quota windows roll.* Consumption is counted in a fixed window that resets, not
a counter that only grows, so a long-running process behaves like the Redis one
rather than slowly starving every application.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Protocol

from evidence_engine.application.ports.platform import QuotaDecision
from evidence_engine.application.ports.streaming import (
    OutboundEvent,
    StreamSnapshot,
)
from evidence_engine.domain.shared.identifiers import ApplicationId, SessionId


class ClockReader(Protocol):
    """The slice of ``Clock`` these adapters need.

    Narrower than the full port on purpose (interface segregation): a cache
    does not need the monotonic session reading, and depending on the whole
    port would make every in-memory adapter awkward to construct in a test that
    only cares about expiry.
    """

    def epoch_ms(self) -> int: ...


class InMemoryStreamState:
    """Per-session ephemeral state with expiring leases."""

    def __init__(self, clock_ms: ClockReader) -> None:
        self._clock = clock_ms
        self._snapshots: dict[str, tuple[StreamSnapshot, int]] = {}
        self._leases: dict[str, int] = {}

    async def load(self, session_id: SessionId) -> StreamSnapshot | None:
        entry = self._snapshots.get(session_id.value)
        if entry is None:
            return None
        snapshot, expires_at = entry
        if self._clock.epoch_ms() >= expires_at:
            del self._snapshots[session_id.value]
            return None
        return snapshot

    async def save(self, snapshot: StreamSnapshot, ttl_seconds: int) -> None:
        expires_at = self._clock.epoch_ms() + ttl_seconds * 1_000
        self._snapshots[snapshot.session_id.value] = (snapshot, expires_at)

    async def clear(self, session_id: SessionId) -> None:
        self._snapshots.pop(session_id.value, None)
        self._leases.pop(session_id.value, None)

    async def acquire_lease(self, session_id: SessionId, ttl_seconds: int) -> bool:
        now = self._clock.epoch_ms()
        held_until = self._leases.get(session_id.value)
        if held_until is not None and now < held_until:
            return False
        self._leases[session_id.value] = now + ttl_seconds * 1_000
        return True

    async def release_lease(self, session_id: SessionId) -> None:
        self._leases.pop(session_id.value, None)


class InMemoryQuotaGuard:
    """Fixed-window quotas per application and unit (NFR-023)."""

    def __init__(
        self,
        clock_ms: ClockReader,
        limits: Mapping[str, int] | None = None,
        window_seconds: int = 60,
    ) -> None:
        self._clock = clock_ms
        self._limits = dict(limits or {})
        self._window_ms = window_seconds * 1_000
        self._counts: dict[tuple[str, str, int], int] = defaultdict(int)

    async def check(self, application: ApplicationId, unit: str, amount: int = 1) -> QuotaDecision:
        limit = self._limits.get(unit)
        if limit is None:
            # No configured limit means unlimited. Deliberate: an unconfigured
            # unit should not silently block traffic, and NFR-023 asks for
            # quotas to be configurable rather than mandatory.
            return QuotaDecision(allowed=True, remaining=-1)

        used = self._counts[self._key(application, unit)]
        remaining = limit - used
        if used + amount > limit:
            return QuotaDecision(
                allowed=False,
                remaining=max(remaining, 0),
                retry_after_seconds=self._seconds_to_window_end(),
                reason=f"{unit} quota of {limit} per window exhausted",
            )
        return QuotaDecision(allowed=True, remaining=remaining - amount)

    async def consume(self, application: ApplicationId, unit: str, amount: int = 1) -> None:
        self._counts[self._key(application, unit)] += amount

    def _key(self, application: ApplicationId, unit: str) -> tuple[str, str, int]:
        window = self._clock.epoch_ms() // self._window_ms
        return (application.value, unit, window)

    def _seconds_to_window_end(self) -> int:
        elapsed = self._clock.epoch_ms() % self._window_ms
        return max(1, (self._window_ms - elapsed) // 1_000)


class RecordingEventChannel:
    """Captures published events instead of sending them anywhere.

    Used by the contract tests and by the batch path, which has no socket. It
    keeps ordering, because §7.4 requires events to be ordered and a test that
    only checked membership would miss a reordering regression.
    """

    def __init__(self) -> None:
        self.published: list[OutboundEvent] = []
        self.backpressure_requests: list[tuple[SessionId, int, int]] = []

    async def publish(self, event: OutboundEvent) -> None:
        self.published.append(event)

    async def request_backpressure(
        self, session_id: SessionId, queue_depth: int, chunk_seq: int
    ) -> None:
        self.backpressure_requests.append((session_id, queue_depth, chunk_seq))

    def of_type(self, message_type: str) -> list[OutboundEvent]:
        return [event for event in self.published if event.type.value == message_type]
