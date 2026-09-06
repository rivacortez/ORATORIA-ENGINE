"""S3: ``_control`` answers ``session.abort`` - exercised without a real socket.

``tests/contract/test_session_abort.py`` drives this end to end over a real
WebSocket and is the test that matters for a consumer. This one exists
because that one has a sharp edge worth naming: if the handler ever stops
answering ``session.abort`` (falls through to the silent-ack branch
``session.configure`` uses), the contract test's ``receive_json()`` blocks
forever waiting for a reply the server will never send - a hang, not a clean
failure. Driving ``_control`` directly, against fakes, turns the same defect
into an ordinary assertion failure with nothing left to time out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from evidence_engine.adapters.inbound.websocket.channel import WebSocketEventChannel
from evidence_engine.adapters.inbound.websocket.handler import _control
from evidence_engine.adapters.inbound.websocket.protocol import (
    ClientMessageType,
    ControlMessage,
    Envelope,
)
from evidence_engine.domain.shared.identifiers import RunId, SessionId

SCHEMA_VERSION = "1.0.0"
SESSION_ID = SessionId("session-abort-0001")
RUN_ID = RunId("run-abort-0001")


class _RecordingSocket:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_json(self, data: Any) -> None:
        self.frames.append(data)


@dataclass
class _FakeCloseRun:
    calls: list[tuple[RunId, bool]] = field(default_factory=list)

    async def execute(self, run_id: RunId, *, succeeded: bool = True) -> None:
        self.calls.append((run_id, succeeded))


@dataclass
class _FakeCaptureControl:
    aborted: list[tuple[object, SessionId]] = field(default_factory=list)

    async def abort(self, caller: object, session_id: SessionId) -> None:
        self.aborted.append((caller, session_id))


@dataclass
class _FakeEngine:
    close_run: _FakeCloseRun
    capture_control: _FakeCaptureControl


@dataclass
class _FakeState:
    run_id: RunId
    captured_audio_ms: int


@dataclass
class _FakeCoordinator:
    state: _FakeState


def _abort_message() -> ControlMessage:
    return ControlMessage(
        envelope=Envelope(
            schema_version=SCHEMA_VERSION,
            session_id=SESSION_ID.value,
            message_id="msg-abort",
            monotonic_time_ms=4_000,
        ),
        type=ClientMessageType.SESSION_ABORT,
    )


async def test_session_abort_closes_the_run_unsuccessful_and_replies_aborted() -> None:
    engine = _FakeEngine(close_run=_FakeCloseRun(), capture_control=_FakeCaptureControl())
    coordinator = _FakeCoordinator(state=_FakeState(run_id=RUN_ID, captured_audio_ms=4_000))
    socket = _RecordingSocket()
    channel = WebSocketEventChannel(socket, SCHEMA_VERSION)

    keep_open = await _control(
        cast(Any, engine),
        cast(Any, coordinator),
        channel,
        cast(Any, None),
        SESSION_ID,
        cast(Any, None),
        _abort_message(),
    )

    assert keep_open is False, "the socket closes right after an abort reply"
    assert engine.close_run.calls == [(RUN_ID, False)]
    assert engine.capture_control.aborted == [(None, SESSION_ID)]

    aborted_frames = [frame for frame in socket.frames if frame["type"] == "session.aborted"]
    assert len(aborted_frames) == 1
    assert aborted_frames[0]["payload"] == {"run_id": RUN_ID.value, "captured_ms": 4_000}
