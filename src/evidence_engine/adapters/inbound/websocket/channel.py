"""The event channel over a live WebSocket (§7.3).

An outbound port implemented by an inbound adapter, which looks inverted until
you notice that the socket is the only thing that can reach this client. The
application publishes into ``EventChannel`` without knowing whether a socket, a
webhook or a recorder is behind it (FR-030 offers two delivery routes), and the
transport that happens to own the connection supplies the implementation.

``event_seq`` is assigned here rather than by the application. §7.4 requires it
and US-012 requires events to be ordered, and ordering is a property of the
connection: the same logical event redelivered after a reconnect is a different
position in a different stream.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from evidence_engine.application.ports.streaming import OutboundEvent, ServerMessageType
from evidence_engine.domain.shared.identifiers import SessionId


class SocketSink(Protocol):
    """The slice of a WebSocket this adapter needs.

    Narrower than Starlette's ``WebSocket`` so the channel can be exercised
    without one, and so a change in the framework's surface does not reach into
    the message shaping this module is responsible for.
    """

    async def send_json(self, data: Any) -> None: ...


class WebSocketEventChannel:
    """Serializes outbound events onto one socket, in order."""

    def __init__(self, socket: SocketSink, schema_version: str) -> None:
        self._socket = socket
        self._schema_version = schema_version
        self._event_seq = 0

    async def publish(self, event: OutboundEvent) -> None:
        await self._socket.send_json(self._frame(event.type, event.session_id, event))

    async def request_backpressure(
        self, session_id: SessionId, queue_depth: int, chunk_seq: int
    ) -> None:
        """§7.3 ``backpressure.requested`` (FR-010).

        Explicit and separate from ``publish`` because it is not evidence - it
        is a flow-control instruction, and a client should be able to handle it
        without parsing an evidence payload.

        ``chunk_seq`` is the refused chunk, named so a client can resend that
        exact window rather than guessing which of its unacknowledged sends to
        retry.
        """
        await self._socket.send_json(
            self._frame(
                ServerMessageType.BACKPRESSURE_REQUESTED,
                session_id,
                payload={"queue_depth": queue_depth, "chunk_seq": chunk_seq},
                monotonic_time_ms=0,
            )
        )

    async def send_error(self, session_id: SessionId, code: str, message: str) -> None:
        """§7.3 ``error``: a protocol or business failure that is not fatal."""
        await self._socket.send_json(
            self._frame(
                ServerMessageType.ERROR,
                session_id,
                payload={"code": code, "message": message},
                monotonic_time_ms=0,
            )
        )

    async def send_accepted(self, session_id: SessionId, body: dict[str, Any]) -> None:
        """§7.3 ``session.accepted``: the first message after a successful open."""
        await self._socket.send_json(
            self._frame(
                ServerMessageType.SESSION_ACCEPTED,
                session_id,
                payload=body,
                monotonic_time_ms=0,
            )
        )

    async def send_completed(self, session_id: SessionId, body: dict[str, Any]) -> None:
        """§7.3 ``session.completed``."""
        await self._socket.send_json(
            self._frame(
                ServerMessageType.SESSION_COMPLETED,
                session_id,
                payload=body,
                monotonic_time_ms=0,
            )
        )

    async def send_aborted(self, session_id: SessionId, body: dict[str, Any]) -> None:
        """§7.3 ``session.aborted``: reply to a client ``session.abort``."""
        await self._socket.send_json(
            self._frame(
                ServerMessageType.SESSION_ABORTED,
                session_id,
                payload=body,
                monotonic_time_ms=0,
            )
        )

    # -- framing ----------------------------------------------------------

    def _frame(
        self,
        message_type: ServerMessageType,
        session_id: SessionId,
        event: OutboundEvent | None = None,
        payload: dict[str, Any] | None = None,
        monotonic_time_ms: int | None = None,
    ) -> dict[str, Any]:
        """Wrap a payload in the envelope §7.4 requires on every message."""
        self._event_seq += 1
        return {
            "schema_version": self._schema_version,
            "session_id": session_id.value,
            "message_id": f"msg_{uuid.uuid4().hex}",
            "event_seq": self._event_seq,
            "monotonic_time_ms": (
                event.monotonic_time_ms if event is not None else (monotonic_time_ms or 0)
            ),
            "type": message_type.value,
            "payload": dict(event.payload) if event is not None else (payload or {}),
        }
