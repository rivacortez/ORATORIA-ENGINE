"""Ephemeral stream state and the outbound event channel.

Two concerns that only exist while a session is live.

``StreamState`` holds what would be lost if the process handling a socket
restarted mid-session: the chunk ledger, the position of the finalized
frontier, the bounded-queue depth. It is deliberately separate from the session
repository, because these are high-churn values written many times a second and
storing them alongside durable metadata would put a transactional database on
the hot path for data that expires when the session does.

``EventChannel`` is how the application pushes results out without knowing
whether a WebSocket, a webhook or a test double is on the other end. FR-030
offers both delivery routes and the application should not branch on which.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from evidence_engine.domain.sessions.sequencing import SequenceGap
from evidence_engine.domain.shared.identifiers import SessionId
from evidence_engine.domain.shared.provenance import Modality


@dataclass(frozen=True, slots=True)
class StreamSnapshot:
    """What a live session needs in order to survive a handler restart."""

    session_id: SessionId
    highest_audio_seq: int
    highest_video_seq: int
    finalized_through_ms: int
    queue_depth: int
    gaps: tuple[SequenceGap, ...] = ()


class StreamState(Protocol):
    """Ephemeral per-session state, with a lease that outlives no session."""

    async def load(self, session_id: SessionId) -> StreamSnapshot | None: ...

    async def save(self, snapshot: StreamSnapshot, ttl_seconds: int) -> None: ...

    async def clear(self, session_id: SessionId) -> None: ...

    async def acquire_lease(self, session_id: SessionId, ttl_seconds: int) -> bool:
        """Claim exclusive processing of a session, or report it is taken.

        One writer per session. Two handlers advancing the same chunk ledger
        would interleave sequence numbers and manufacture gaps that never
        happened, which FR-008 would then report as data loss.
        """
        ...

    async def release_lease(self, session_id: SessionId) -> None: ...


class ServerMessageType(StrEnum):
    """§7.3: the messages the server sends.

    Kept as an enum so that adding one is a visible change and every emitter
    resolves to a published name. §7.4 requires unknown enum values not to
    crash consumers, so this list can grow within a major version.
    """

    SESSION_ACCEPTED = "session.accepted"
    QUALITY_WARNING = "quality.warning"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    SPEECH_EVENT_PARTIAL = "speech_event.partial"
    SPEECH_EVENT_FINAL = "speech_event.final"
    VISUAL_EVENT_FINAL = "visual_event.final"
    COOCCURRENCE_FINAL = "cooccurrence.final"
    BACKPRESSURE_REQUESTED = "backpressure.requested"
    PROCESSING_DEGRADED = "processing.degraded"
    SESSION_COMPLETED = "session.completed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class OutboundEvent:
    """One server-to-client message, already shaped by §7.4's contract rules."""

    type: ServerMessageType
    session_id: SessionId
    monotonic_time_ms: int
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DegradationNotice:
    """§7.3 ``processing.degraded``: one modality stopped, the session did not.

    Carries the modality and the reason, because QA-02's measure is that no
    visual value is fabricated - which means the client has to be told what to
    stop expecting, not merely that something went wrong.
    """

    modality: Modality
    reason: str
    detail: str = ""


class EventChannel(Protocol):
    """Where partial and final results go. Transport-agnostic by design."""

    async def publish(self, event: OutboundEvent) -> None: ...

    async def request_backpressure(self, session_id: SessionId, queue_depth: int) -> None:
        """FR-010: signal explicitly before memory is exhausted.

        Explicit rather than implicit: US-012 makes backpressure part of the
        contract, so a client that keeps sending is misbehaving rather than
        merely unlucky. Silently dropping chunks instead would show up later as
        an unexplained gap in the evidence.
        """
        ...
