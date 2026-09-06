"""S1: ``backpressure.requested`` names the refused chunk, not only the queue.

A caller told only ``queue_depth`` has no way to know which of its
unacknowledged windows to retry - it would have to track its own sends and
guess. Both channel implementations the coordinator can be wired to have to
carry ``chunk_seq`` alongside ``queue_depth`` for the same reason `_admit`
reads it: reverting either channel's payload turns this red without touching
`StreamingCoordinator` at all.
"""

from __future__ import annotations

from typing import Any

from evidence_engine.adapters.inbound.websocket.channel import WebSocketEventChannel
from evidence_engine.domain.shared.identifiers import SessionId
from evidence_engine.sdk._local import CollectedEvents

SESSION_ID = SessionId("session-0001")


class _RecordingSocket:
    """The slice of a WebSocket ``WebSocketEventChannel`` uses, kept in memory."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_json(self, data: Any) -> None:
        self.frames.append(data)


async def test_the_websocket_channel_names_the_refused_chunk() -> None:
    socket = _RecordingSocket()
    channel = WebSocketEventChannel(socket, "1.0.0")

    await channel.request_backpressure(SESSION_ID, queue_depth=3, chunk_seq=7)

    (frame,) = socket.frames
    assert frame["type"] == "backpressure.requested"
    assert frame["payload"] == {"queue_depth": 3, "chunk_seq": 7}


async def test_the_embedded_channel_names_the_refused_chunk() -> None:
    channel = CollectedEvents()

    await channel.request_backpressure(SESSION_ID, queue_depth=2, chunk_seq=4)

    event = await channel.next_event()
    assert event.payload == {"queue_depth": 2, "chunk_seq": 4}
    assert channel.backpressure_signals == 1
