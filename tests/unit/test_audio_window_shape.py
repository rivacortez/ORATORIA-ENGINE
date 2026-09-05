"""A window cannot lie about how much audio it carries.

The failure this guards is silent by construction. ``StreamingCoordinator``
ends a window at ``session_position_ms + duration_ms`` and hands the payload to
a runtime; if the declaration says a second and the payload holds 160 ms, every
word, event boundary and prosody reading comes back scaled by 6.25 and nothing
raises. The transcript renders, the events carry provenance, the confidence
values are calibrated, and NFR-004's 250 ms boundary target is then measured
against a clock that is lying.

Two halves, because there are two ways to arrive at that state:

*The port refuses a window whose declaration its payload contradicts.* This is
the only place that still holds both halves of the claim.

*The handler refuses a chunk that never made the claim.* ``_parse_audio``
supplies 0 ms and 16 kHz for an absent declaration, and those defaults are not
approximations - they are a scale factor of whatever the client actually sent.
The check therefore reads the wire message, because the parsed ``AudioChunk``
can no longer tell an absent field from a declared one.

The handler half drives ``_dispatch`` rather than a live socket. What is being
checked is which of three things happens to one frame - refused, ingested, or
the connection dropped - and ``_dispatch``'s return value *is* the third of
those: ``_run`` leaves its receive loop when it comes back False.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest

from evidence_engine.adapters.inbound.websocket.channel import WebSocketEventChannel
from evidence_engine.adapters.inbound.websocket.handler import (
    INVALID_AUDIO_SHAPE,
    _dispatch,
)
from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    MisdeclaredAudioWindow,
    VisualFrame,
)
from evidence_engine.domain.shared.identifiers import SessionId

SCHEMA_VERSION = "1.0.0"
SESSION_ID = SessionId("session-audio-shape")

#: 16 kHz mono PCM16 is the negotiated shape of the contract suite's sessions,
#: so the arithmetic below is the arithmetic a real client does.
RATE_HZ = 16_000


def pcm16_mono(duration_ms: int, sample_rate_hz: int = RATE_HZ) -> bytes:
    """Exactly ``duration_ms`` of silence, so the payload is never the variable.

    Every test that lies does so in the declaration and leaves this alone. A
    fixture that got the payload wrong instead would be the very defect under
    test, dressed as a helper.
    """
    return b"\x00\x00" * (duration_ms * sample_rate_hz // 1_000)


# ---------------------------------------------------------------------------
# The port: a declaration checked against its payload
# ---------------------------------------------------------------------------


def test_a_window_whose_declaration_matches_its_payload_is_accepted() -> None:
    window = AudioWindow(
        session_position_ms=0,
        duration_ms=1_000,
        sample_rate_hz=RATE_HZ,
        samples=pcm16_mono(1_000),
    )

    assert window.duration_ms == 1_000
    assert len(window.samples) == 32_000


def test_a_window_declaring_six_times_the_audio_it_carries_is_refused() -> None:
    """The exact shape of the bug: 160 ms of silence announced as a second."""
    with pytest.raises(MisdeclaredAudioWindow) as refused:
        AudioWindow(
            session_position_ms=0,
            duration_ms=1_000,
            sample_rate_hz=RATE_HZ,
            samples=pcm16_mono(160),
        )

    message = str(refused.value)
    # Both numbers, not just the invalid one: a reader needs to see the factor
    # to know how far the timestamps would have moved.
    assert "1000 ms" in message
    assert "160.0 ms" in message


def test_a_window_declaring_a_rate_it_was_not_decoded_at_is_refused() -> None:
    """PCM16 at 16 kHz announced as 48 kHz - a third of the audio it claims."""
    with pytest.raises(MisdeclaredAudioWindow) as refused:
        AudioWindow(
            session_position_ms=0,
            duration_ms=1_000,
            sample_rate_hz=48_000,
            samples=pcm16_mono(1_000),
        )

    assert "333.3 ms" in str(refused.value)


def test_a_duration_of_zero_is_refused() -> None:
    """0 ms is what an undeclared chunk parses into, so it is never tolerated."""
    with pytest.raises(MisdeclaredAudioWindow) as refused:
        AudioWindow(
            session_position_ms=4_000,
            duration_ms=0,
            sample_rate_hz=RATE_HZ,
            samples=b"",
        )

    assert "0 ms" in str(refused.value)


def test_a_negative_duration_is_refused() -> None:
    with pytest.raises(MisdeclaredAudioWindow):
        AudioWindow(
            session_position_ms=0,
            duration_ms=-20,
            sample_rate_hz=RATE_HZ,
            samples=pcm16_mono(20),
        )


def test_a_sample_rate_the_contract_never_negotiates_is_refused() -> None:
    """FR-006 refuses 8 kHz at session creation; a chunk cannot smuggle it back."""
    with pytest.raises(MisdeclaredAudioWindow) as refused:
        AudioWindow(
            session_position_ms=0,
            duration_ms=1_000,
            sample_rate_hz=8_000,
            samples=pcm16_mono(1_000, 8_000),
        )

    message = str(refused.value)
    assert "8000 Hz" in message
    # The accepted set is named, so the client is not left guessing.
    assert "16000" in message and "48000" in message


def test_a_payload_cut_mid_frame_is_refused() -> None:
    """An odd byte count under 16-bit samples means the payload was truncated."""
    with pytest.raises(MisdeclaredAudioWindow) as refused:
        AudioWindow(
            session_position_ms=0,
            duration_ms=1_000,
            sample_rate_hz=RATE_HZ,
            samples=pcm16_mono(1_000) + b"\x00",
        )

    assert "not a whole number of 2-byte frames" in str(refused.value)


def test_a_frame_size_of_zero_is_refused_before_it_divides() -> None:
    with pytest.raises(MisdeclaredAudioWindow):
        AudioWindow(
            session_position_ms=0,
            duration_ms=1_000,
            sample_rate_hz=RATE_HZ,
            samples=pcm16_mono(1_000),
            sample_width_bytes=0,
        )


def test_the_declared_framing_is_used_and_not_assumed() -> None:
    """A stereo payload is accepted as stereo and refused as mono.

    The pair matters more than either half. If ``channel_count`` were assumed
    rather than declared, the same 64 000 bytes would be read as 32 000 frames
    and would agree perfectly with a declaration of twice its real length -
    which is the failure this whole module exists to refuse.
    """
    one_second_of_stereo = b"\x00\x00\x00\x00" * RATE_HZ

    accepted = AudioWindow(
        session_position_ms=0,
        duration_ms=1_000,
        sample_rate_hz=RATE_HZ,
        samples=one_second_of_stereo,
        channel_count=2,
    )
    assert accepted.channel_count == 2

    with pytest.raises(MisdeclaredAudioWindow) as refused:
        AudioWindow(
            session_position_ms=0,
            duration_ms=1_000,
            sample_rate_hz=RATE_HZ,
            samples=one_second_of_stereo,
        )

    assert "2000.0 ms" in str(refused.value)


@pytest.mark.parametrize("frames", [220, 221])
def test_a_rate_with_no_whole_frame_per_millisecond_still_passes(frames: int) -> None:
    """22 050 Hz has 22.05 frames per ms, so a 10 ms chunk cannot be exact.

    Both a client that rounds the duration and one that truncates it land here,
    and refusing either would be refusing arithmetic rather than a lie.
    """
    window = AudioWindow(
        session_position_ms=0,
        duration_ms=10,
        sample_rate_hz=22_050,
        samples=b"\x00\x00" * frames,
    )

    assert window.duration_ms == 10


def test_a_disagreement_larger_than_the_declaration_s_own_resolution_is_refused() -> None:
    """2 ms out at 22 050 Hz is a disagreement, not a rounding difference."""
    with pytest.raises(MisdeclaredAudioWindow):
        AudioWindow(
            session_position_ms=0,
            duration_ms=12,
            sample_rate_hz=22_050,
            samples=b"\x00\x00" * 220,
        )


# ---------------------------------------------------------------------------
# The handler: a chunk that never declared its shape
# ---------------------------------------------------------------------------


class _RecordingSocket:
    """The slice of a WebSocket ``WebSocketEventChannel`` uses, kept in memory."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_json(self, data: Any) -> None:
        self.frames.append(data)

    @property
    def errors(self) -> list[dict[str, Any]]:
        return [frame for frame in self.frames if frame["type"] == "error"]


class _RecordingCoordinator:
    """Records what was handed on, so a refusal can be told from a silence.

    A test that only asserted the error frame would pass against a handler that
    published the error *and* ingested the window anyway, which is the worse of
    the two failures: the client is told and the clock still moves.
    """

    def __init__(self) -> None:
        self.windows: list[tuple[int, AudioWindow]] = []
        self.frames: list[tuple[int, Sequence[VisualFrame]]] = []

    async def ingest_audio(self, sequence: int, window: AudioWindow) -> None:
        self.windows.append((sequence, window))

    async def ingest_video(self, sequence: int, frames: Sequence[VisualFrame]) -> None:
        self.frames.append((sequence, frames))


class _CountingTelemetry:
    """The whole ``Telemetry`` port, so the stub is not narrower than the real one."""

    def __init__(self) -> None:
        self.counters: list[tuple[str, dict[str, str]]] = []

    def counter(self, name: str, value: int = 1, **labels: str) -> None:
        self.counters.append((name, labels))

    def histogram(self, name: str, value: float, **labels: str) -> None:
        return None

    def event(self, name: str, trace_id: str, **fields: str | int | float | bool) -> None:
        return None


@dataclass
class _Exchange:
    """What one dispatched frame produced."""

    keep_open: bool
    socket: _RecordingSocket
    coordinator: _RecordingCoordinator
    telemetry: _CountingTelemetry


async def dispatch_one(raw: dict[str, Any]) -> _Exchange:
    """Run one wire message through the handler's dispatch, in isolation.

    ``caller`` and ``configuration`` are passed as ``None``. Neither is read on
    the media path - only ``_control`` touches them - and building a real
    ``ConfigurationSnapshot`` here would tie this file to a shape that has
    nothing to do with what it checks.
    """
    socket = _RecordingSocket()
    coordinator = _RecordingCoordinator()
    telemetry = _CountingTelemetry()
    engine = type("_StubEngine", (), {"telemetry": telemetry})()

    keep_open = await _dispatch(
        cast(Any, engine),
        cast(Any, coordinator),
        WebSocketEventChannel(socket, SCHEMA_VERSION),
        cast(Any, None),
        SESSION_ID,
        cast(Any, None),
        raw,
    )
    return _Exchange(keep_open, socket, coordinator, telemetry)


def audio_message(**overrides: Any) -> dict[str, Any]:
    """A well-formed ``audio.chunk``: 100 ms of 16 kHz mono PCM16, declared."""
    message: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": SESSION_ID.value,
        "message_id": "msg-1",
        "chunk_seq": 0,
        "monotonic_time_ms": 0,
        "type": "audio.chunk",
        "samples": base64.b64encode(pcm16_mono(100)).decode("ascii"),
        "sample_rate_hz": RATE_HZ,
        "duration_ms": 100,
    }
    message.update(overrides)
    return message


async def test_a_well_formed_chunk_still_reaches_the_coordinator() -> None:
    """The regression guard: the refusal must not cost the ordinary case."""
    exchange = await dispatch_one(audio_message())

    assert exchange.keep_open is True
    assert not exchange.socket.errors
    assert len(exchange.coordinator.windows) == 1
    _, window = exchange.coordinator.windows[0]
    assert window.duration_ms == 100
    assert window.sample_rate_hz == RATE_HZ


async def test_a_chunk_that_omits_its_sample_rate_is_refused_by_name() -> None:
    message = audio_message()
    del message["sample_rate_hz"]

    exchange = await dispatch_one(message)

    assert exchange.socket.errors[0]["payload"]["code"] == INVALID_AUDIO_SHAPE
    # The message names the field, not merely the fact that something was wrong.
    assert "sample_rate_hz" in exchange.socket.errors[0]["payload"]["message"]


async def test_a_chunk_that_omits_its_duration_is_refused_by_name() -> None:
    """Refused as *missing*, which the port's own refusal cannot say.

    A window built from the parser's 0 ms default is refused too, and its
    message happens to contain the string ``duration_ms``. Asserting on the
    opening clause is what separates "you did not declare it" from "what you
    declared is impossible" - two different fixes on the client.
    """
    message = audio_message()
    del message["duration_ms"]

    exchange = await dispatch_one(message)

    refusal = exchange.socket.errors[0]["payload"]["message"]
    assert refusal.startswith("audio.chunk must declare duration_ms")


async def test_a_chunk_that_omits_both_names_both() -> None:
    message = audio_message()
    del message["duration_ms"]
    del message["sample_rate_hz"]

    exchange = await dispatch_one(message)

    refusal = exchange.socket.errors[0]["payload"]["message"]
    assert "duration_ms" in refusal
    assert "sample_rate_hz" in refusal


async def test_an_undeclared_chunk_never_reaches_the_coordinator() -> None:
    """16 000 is not a fallback: a chunk that did not say is not processed."""
    message = audio_message()
    del message["sample_rate_hz"]

    exchange = await dispatch_one(message)

    assert exchange.coordinator.windows == []


async def test_an_undeclared_chunk_does_not_close_the_socket() -> None:
    """§7.3 keeps ``error`` non-fatal: one bad frame, not the presentation."""
    message = audio_message()
    del message["duration_ms"]

    exchange = await dispatch_one(message)

    assert exchange.keep_open is True


async def test_a_declaration_sent_as_a_string_is_refused_rather_than_coerced() -> None:
    """``int("16000")`` succeeds, which is how a typed field stops being one."""
    exchange = await dispatch_one(audio_message(sample_rate_hz="16000"))

    assert exchange.socket.errors[0]["payload"]["code"] == INVALID_AUDIO_SHAPE
    assert exchange.coordinator.windows == []


async def test_a_null_declaration_is_refused_instead_of_raising_in_the_parser() -> None:
    """``int(None)`` would be a ``TypeError`` escaping into the receive loop."""
    exchange = await dispatch_one(audio_message(duration_ms=None))

    assert exchange.socket.errors[0]["payload"]["code"] == INVALID_AUDIO_SHAPE
    assert exchange.keep_open is True


async def test_a_chunk_whose_payload_contradicts_its_declaration_is_refused() -> None:
    """Declared and present, and still wrong: 160 ms announced as a second."""
    exchange = await dispatch_one(
        audio_message(
            duration_ms=1_000,
            samples=base64.b64encode(pcm16_mono(160)).decode("ascii"),
        )
    )

    assert exchange.socket.errors[0]["payload"]["code"] == INVALID_AUDIO_SHAPE
    assert "160.0 ms" in exchange.socket.errors[0]["payload"]["message"]
    assert exchange.coordinator.windows == []
    assert exchange.keep_open is True


async def test_both_refusals_are_counted_separately() -> None:
    """An operator needs to see a client sending bad chunks all session."""
    undeclared = await dispatch_one({**audio_message(), "duration_ms": None})
    misdeclared = await dispatch_one(
        audio_message(
            duration_ms=1_000,
            samples=base64.b64encode(pcm16_mono(160)).decode("ascii"),
        )
    )

    assert undeclared.telemetry.counters == [
        ("stream.audio_shape_refused", {"reason": "undeclared"})
    ]
    assert misdeclared.telemetry.counters == [
        ("stream.audio_shape_refused", {"reason": "misdeclared"})
    ]


async def test_a_video_frame_is_not_asked_to_declare_an_audio_shape() -> None:
    """The rule is about the audio declaration and must not spread to §7.2's rest."""
    exchange = await dispatch_one(
        {
            "schema_version": SCHEMA_VERSION,
            "session_id": SESSION_ID.value,
            "message_id": "msg-2",
            "chunk_seq": 0,
            "monotonic_time_ms": 0,
            "type": "visual.features",
            "landmarks": [0.5, 0.5],
        }
    )

    assert not exchange.socket.errors
    assert exchange.keep_open is True
