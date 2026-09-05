"""The adapters an embedded engine runs on, and the two it needs of its own.

Everything here is either an existing in-memory adapter reused unchanged or a
small piece the hosted service happens to satisfy with infrastructure. Nothing
in this module is a second implementation of a rule: the SDK composes the same
use cases the server does, so a rule enforced once is enforced on both paths.
That is the whole reason the facade goes through `StreamingCoordinator` and
`CreateSession` rather than calling a runtime directly - a second functional
path would be a second place for the state machine, the quota, the consent
check and the ledger to be almost right.

Two pieces are new, and both exist because the server's version needs a
dependency the core does not have.

``SilentTelemetry`` - the structlog adapter would pull the `server` extra into
an embedded install. An SDK writing to somebody else's logging configuration
uninvited is the wrong default anyway; a consumer who wants telemetry passes
their own, and the port is three methods.

``CollectedEvents`` - the WebSocket channel writes to a socket. Here the events
go into a queue the caller drains with ``receive()``, which is what turns
`§7.3`'s server-to-client messages into an ordinary async iteration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

from evidence_engine.application.ports.streaming import OutboundEvent, ServerMessageType
from evidence_engine.domain.shared.identifiers import SessionId


class SilentTelemetry:
    """``Telemetry`` that records nothing.

    Not a stub with a TODO: writing operational telemetry is a deployment's
    job, and an embedded library that emitted counters into a consumer's
    process would be making that decision for them. The hosted service wires
    structlog and OpenTelemetry; a consumer embedding the engine passes
    whatever they already run, or this.
    """

    def counter(self, name: str, value: int = 1, **labels: str) -> None:
        return None

    def histogram(self, name: str, value: float, **labels: str) -> None:
        return None

    def event(self, name: str, trace_id: str, **fields: str | int | float | bool) -> None:
        return None


class CollectedEvents:
    """``EventChannel`` that queues instead of writing to a socket.

    Unbounded on purpose, and the reason is worth stating because "unbounded
    queue" is usually a bug. The bound that matters is upstream: the
    coordinator's ``max_queue_depth`` limits how much *audio* is in flight and
    raises ``BackpressureRequired`` when it is exceeded. This queue holds the
    results of work already done, so bounding it would mean blocking the
    coordinator on a slow reader - turning a consumer that reads late into a
    consumer that loses evidence.
    """

    def __init__(self) -> None:
        self._events: asyncio.Queue[OutboundEvent] = asyncio.Queue()
        self._backpressure = 0

    async def publish(self, event: OutboundEvent) -> None:
        await self._events.put(event)

    async def request_backpressure(self, session_id: SessionId, queue_depth: int) -> None:
        self._backpressure += 1
        await self._events.put(
            OutboundEvent(
                type=ServerMessageType.BACKPRESSURE_REQUESTED,
                session_id=session_id,
                monotonic_time_ms=0,
                payload={"queue_depth": queue_depth},
            )
        )

    @property
    def backpressure_signals(self) -> int:
        """How many times the engine asked the caller to slow down.

        Published because an embedded caller controls its own pacing: a client
        that got five of these and ignored them has a bug in its send loop, and
        nothing else in the SDK would tell it.
        """
        return self._backpressure

    async def next_event(self) -> OutboundEvent:
        return await self._events.get()

    def pending(self) -> int:
        return self._events.qsize()

    def drain(self) -> Iterator[OutboundEvent]:
        """Everything queued right now, without waiting for more."""
        while not self._events.empty():
            yield self._events.get_nowait()
