"""The streaming protocol (§7.2, §7.3, §7.4).

§7.4 lists five contract rules and each one shows up here as code rather than
as a convention:

- every message carries ``schema_version``, ``session_id``, ``message_id`` and
  either ``chunk_seq`` or ``event_seq``, plus ``monotonic_time_ms``;
- event identifiers are stable across reconciliation (handled upstream, in
  ``application.services.event_identity``);
- duplicate client messages are idempotent (the chunk ledger decides);
- unknown enum values must not crash consumers - hence ``parse_client_message``
  returning an ``UnknownMessage`` rather than raising;
- breaking changes require a new major version.

Audio arrives base64-encoded inside the JSON envelope. A binary frame with a
packed header would be roughly a third smaller and cheaper to parse, and it is
the right optimization for §13 Phase 7 - but it is an optimization, and doing
it now would mean two parsing paths to keep in sync while the contract is still
settling. The envelope rules above apply to every message, and a binary frame
has nowhere to put them without inventing a second, parallel encoding.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ClientMessageType(StrEnum):
    """§7.2: what the client may send."""

    SESSION_CONFIGURE = "session.configure"
    AUDIO_CHUNK = "audio.chunk"
    VIDEO_FRAME = "video.frame"
    #: Privacy-preserving client-side extraction (ADR-009). Geometry only; no
    #: image ever reaches the service on this path.
    VISUAL_FEATURES = "visual.features"
    CAPTURE_PAUSE = "capture.pause"
    CAPTURE_RESUME = "capture.resume"
    SESSION_COMPLETE = "session.complete"
    #: Ends the run as unsuccessful rather than reconciling it. Distinct from
    #: dropping the connection: a dropped socket already survives as a partial
    #: failure (§6.3), but says nothing about whether the client meant to stop
    #: - this says so explicitly, and gets an explicit reply.
    SESSION_ABORT = "session.abort"
    CLIENT_HEARTBEAT = "client.heartbeat"


@dataclass(frozen=True, slots=True)
class Envelope:
    """The header §7.4 requires on every message."""

    schema_version: str
    session_id: str
    message_id: str
    monotonic_time_ms: int
    chunk_seq: int | None = None
    event_seq: int | None = None


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """PCM samples for one window, with their position on the session clock."""

    envelope: Envelope
    sequence: int
    samples: bytes
    sample_rate_hz: int
    duration_ms: int
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class VideoFramePayload:
    """One sampled frame, or the geometry extracted from it client-side."""

    envelope: Envelope
    sequence: int
    session_position_ms: int
    pixels: bytes | None = None
    landmarks: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ControlMessage:
    """Pause, resume, complete, heartbeat, configure - no media payload."""

    envelope: Envelope
    type: ClientMessageType
    body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UnknownMessage:
    """A message this version does not understand.

    Returned rather than raised. §7.4 says unknown enum values must not crash
    consumers, and the same courtesy has to run in this direction: a client
    built against a later minor version will send message types this build has
    never heard of, and dropping the connection over one would make every
    forward-compatible rollout a breaking change.
    """

    raw_type: str
    envelope: Envelope | None
    reason: str


ClientMessage = AudioChunk | VideoFramePayload | ControlMessage | UnknownMessage


class ProtocolError(Exception):
    """The message could not be parsed at all - not merely not understood."""


def parse_client_message(raw: dict[str, Any]) -> ClientMessage:
    """Parse one inbound message, tolerating what it does not recognize."""
    envelope = _parse_envelope(raw)
    raw_type = str(raw.get("type", ""))

    try:
        message_type = ClientMessageType(raw_type)
    except ValueError:
        return UnknownMessage(
            raw_type=raw_type,
            envelope=envelope,
            reason="unrecognized message type for this schema version",
        )

    if message_type is ClientMessageType.AUDIO_CHUNK:
        return _parse_audio(envelope, raw)
    if message_type in (
        ClientMessageType.VIDEO_FRAME,
        ClientMessageType.VISUAL_FEATURES,
    ):
        return _parse_video(envelope, raw, message_type)

    return ControlMessage(
        envelope=envelope,
        type=message_type,
        body={k: v for k, v in raw.items() if k not in _ENVELOPE_KEYS and k != "type"},
    )


_ENVELOPE_KEYS = frozenset(
    {"schema_version", "session_id", "message_id", "monotonic_time_ms", "chunk_seq", "event_seq"}
)


def _parse_envelope(raw: dict[str, Any]) -> Envelope:
    missing = [
        key
        for key in ("schema_version", "session_id", "message_id", "monotonic_time_ms")
        if key not in raw
    ]
    if missing:
        raise ProtocolError(f"message is missing required envelope fields: {missing}")

    monotonic = raw["monotonic_time_ms"]
    if not isinstance(monotonic, int) or monotonic < 0:
        raise ProtocolError(f"monotonic_time_ms must be a non-negative integer, got {monotonic!r}")

    return Envelope(
        schema_version=str(raw["schema_version"]),
        session_id=str(raw["session_id"]),
        message_id=str(raw["message_id"]),
        monotonic_time_ms=monotonic,
        chunk_seq=_optional_int(raw, "chunk_seq"),
        event_seq=_optional_int(raw, "event_seq"),
    )


def _parse_audio(envelope: Envelope, raw: dict[str, Any]) -> AudioChunk:
    if envelope.chunk_seq is None:
        raise ProtocolError(
            "audio.chunk requires chunk_seq; without it FR-008 cannot tell a lost "
            "packet from a pause the speaker actually took"
        )
    try:
        samples = base64.b64decode(str(raw.get("samples", "")), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError("audio.chunk samples are not valid base64") from exc

    return AudioChunk(
        envelope=envelope,
        sequence=envelope.chunk_seq,
        samples=samples,
        sample_rate_hz=int(raw.get("sample_rate_hz", 16_000)),
        duration_ms=int(raw.get("duration_ms", 0)),
        is_final=bool(raw.get("is_final", False)),
    )


def _parse_video(
    envelope: Envelope, raw: dict[str, Any], message_type: ClientMessageType
) -> VideoFramePayload:
    if envelope.chunk_seq is None:
        raise ProtocolError(f"{message_type.value} requires chunk_seq")

    pixels: bytes | None = None
    if message_type is ClientMessageType.VIDEO_FRAME:
        try:
            pixels = base64.b64decode(str(raw.get("pixels", "")), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProtocolError("video.frame pixels are not valid base64") from exc

    landmarks = raw.get("landmarks", [])
    if not isinstance(landmarks, list):
        raise ProtocolError("landmarks must be a flat list of numbers")

    return VideoFramePayload(
        envelope=envelope,
        sequence=envelope.chunk_seq,
        session_position_ms=int(raw.get("session_position_ms", envelope.monotonic_time_ms)),
        pixels=pixels,
        landmarks=tuple(float(value) for value in landmarks),
    )


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{key} must be a non-negative integer, got {value!r}")
    return value
