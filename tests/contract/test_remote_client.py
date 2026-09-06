"""``OratoriaClient`` against a fake transport, and once against the real app.

Two different things are being defended here, and they need different rigs.

**The wire protocol** - what `OratoriaClient` sends and how it reacts to what
comes back - is tested against a fake `Transport` that replays scripted
frames. Fast, and precise about which frame triggers which client behaviour.

**Conformance** - that the remote client and the embedded engine agree on the
evidence for the same session - is tested by actually running the same
scripted presentation through both paths and comparing the results, the same
"redact what differs by construction, compare the rest" approach
`test_sdk_surface.py::test_the_local_engine_is_deterministic_across_runs`
already uses for the embedded engine's own determinism claim. ADR-011 named
this the conformance test it deferred until a remote client existed; this is
that test.

One test drives the real ASGI application end to end, with `uvicorn.Server`
bound to an ephemeral port in a background thread - the memory backend and the
deterministic runtimes mean it needs no infrastructure, so it is not marked
`integration`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import threading
import time
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient

from evidence_engine import EngineConfiguration
from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    DeterministicSpeechRuntime,
    DeterministicVisionRuntime,
    SpeechScript,
    VisualScript,
)
from evidence_engine.adapters.outbound.telemetry.clock import FrozenClock
from evidence_engine.application.ports.streaming import ServerMessageType
from evidence_engine.bootstrap.app import create_app
from evidence_engine.bootstrap.container import Container
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.provenance import ModelRole
from evidence_engine.sdk.client import OratoriaClient, RemoteStreamSession
from evidence_engine.sdk.engine import OratoriaEngine
from evidence_engine.sdk.errors import RemoteEngineUnavailable
from evidence_engine.sdk.results import Evidence, evidence_from_json

pytestmark = pytest.mark.contract

SAMPLE_RATE_HZ = 16_000
WINDOW_MS = 1_000
FRAMES_PER_WINDOW = SAMPLE_RATE_HZ * WINDOW_MS // 1_000
SILENT_CHUNK = b"\x00\x00" * FRAMES_PER_WINDOW

CREATE_BODY = {
    "mode": "realtime",
    "capabilities": {"audio_codec": "pcm16", "sample_rate_hz": SAMPLE_RATE_HZ, "locale": "es-PE"},
    "consent_policy_version": "1.0.0",
}


# ---------------------------------------------------------------------------
# A fake Transport - scripted, synchronous, nothing that can block forever
# ---------------------------------------------------------------------------


class _NoMoreFrames(Exception):
    """The fake connection ran out of scripted frames - ends the reader loop
    the same way a real closed socket would, rather than hanging it."""


class _FakeWsConnection:
    def __init__(self, frames: list[dict[str, object]]) -> None:
        self._frames = list(frames)
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def send_json(self, data: Mapping[str, object]) -> None:
        self.sent.append(dict(data))

    async def receive_json(self) -> dict[str, object]:
        if not self._frames:
            raise _NoMoreFrames
        return self._frames.pop(0)

    async def close(self) -> None:
        self.closed = True


class _FakeTransport:
    """Scripted responses, keyed by the path suffix a real URL would end in."""

    def __init__(
        self,
        *,
        post_response: tuple[int, dict[str, object]] = (201, {}),
        get_responses: dict[str, list[tuple[int, dict[str, object]]]] | None = None,
        ws_frames: list[dict[str, object]] | None = None,
    ) -> None:
        self._post_response = post_response
        self._get_responses = {k: list(v) for k, v in (get_responses or {}).items()}
        self._ws_frames = list(ws_frames or [])
        self.ws_connection: _FakeWsConnection | None = None
        self.posted: list[tuple[str, dict[str, object]]] = []
        self.get_calls: list[str] = []

    async def post_json(
        self, url: str, *, json: Mapping[str, object], headers: Mapping[str, str]
    ) -> tuple[int, dict[str, object]]:
        self.posted.append((url, dict(json)))
        return self._post_response

    async def get_json(
        self, url: str, *, headers: Mapping[str, str]
    ) -> tuple[int, dict[str, object]]:
        self.get_calls.append(url)
        for suffix, responses in self._get_responses.items():
            if url.endswith(suffix):
                return responses.pop(0) if len(responses) > 1 else responses[0]
        raise AssertionError(f"no scripted GET response for {url}")

    async def ws_connect(self, url: str) -> _FakeWsConnection:
        self.ws_connection = _FakeWsConnection(self._ws_frames)
        return self.ws_connection


def _accepted_frame(session_id: str = "sess-0001") -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "session_id": session_id,
        "message_id": "msg-accepted",
        "event_seq": 1,
        "monotonic_time_ms": 0,
        "type": "session.accepted",
        "payload": {"run_id": "run-0001", "instance_id": "gpu-workstation-1"},
    }


def _created_session_response(session_id: str = "sess-0001") -> tuple[int, dict[str, object]]:
    return (
        201,
        {
            "schema_version": "1.0.0",
            "session_id": session_id,
            "state": "created",
            "mode": "realtime",
            "stream_token": "a-stream-token",
            "stream_token_expires_at_ms": 300_000,
            "configuration_id": "config-default-v1",
        },
    )


async def _settle(session: RemoteStreamSession) -> None:
    """Let a still-running reader task reach its natural end before a test
    exits - otherwise it is torn down mid-flight and this repository's
    `filterwarnings = ["error"]` turns the resulting "Task was destroyed but
    it is pending" warning into a failure of a LATER, unrelated test."""
    task = session._state.reader_task
    if task is not None:
        with contextlib.suppress(TimeoutError, _NoMoreFrames):
            await asyncio.wait_for(task, timeout=1)


# ---------------------------------------------------------------------------
# warmup()
# ---------------------------------------------------------------------------


async def test_warmup_refuses_when_the_engine_is_not_ready() -> None:
    transport = _FakeTransport(
        get_responses={
            "/health/ready": [
                (
                    503,
                    {"status": "not_ready", "checks": {"speech:warm": "unavailable (warming up)"}},
                )
            ],
        },
    )
    client = OratoriaClient("http://engine.local:8000", "test-key", transport=transport)

    with pytest.raises(RemoteEngineUnavailable, match=r"engine\.local:8000/health/ready"):
        await client.warmup()


async def test_warmup_refuses_when_capabilities_cannot_be_read() -> None:
    transport = _FakeTransport(
        get_responses={
            "/health/ready": [
                (200, {"status": "ready", "checks": {"speech:warm": "warm (1.0 s)"}})
            ],
            "/v1/capabilities": [(500, {"code": "internal_error", "message": "boom"})],
        },
    )
    client = OratoriaClient("http://engine.local:8000", "test-key", transport=transport)

    with pytest.raises(RemoteEngineUnavailable, match=r"engine\.local:8000/v1/capabilities"):
        await client.warmup()


async def test_warmup_learns_the_contributions_by_role() -> None:
    transport = _FakeTransport(
        get_responses={
            "/health/ready": [
                (200, {"status": "ready", "checks": {"speech:warm": "warm (1.0 s)"}})
            ],
            "/v1/capabilities": [
                (
                    200,
                    {
                        "models": {
                            "recogniser": "deterministic-speech-v1",
                            "visual_estimator": "deterministic-vision-v1",
                        },
                        "instance": {"id": "gpu-1", "hostname": "gpu-1"},
                    },
                )
            ],
        },
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)

    await client.warmup()

    assert client.contributions == {
        ModelRole.RECOGNISER: ModelVersionId("deterministic-speech-v1"),
        ModelRole.VISUAL_ESTIMATOR: ModelVersionId("deterministic-vision-v1"),
    }


# ---------------------------------------------------------------------------
# send_audio()
# ---------------------------------------------------------------------------


async def test_send_audio_sends_the_declared_wire_shape() -> None:
    transport = _FakeTransport(
        post_response=_created_session_response(),
        ws_frames=[_accepted_frame()],
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)
    stream = client.create_stream()

    accepted = await stream.send_audio(SILENT_CHUNK)

    assert accepted is True
    assert transport.ws_connection is not None
    (sent,) = transport.ws_connection.sent
    assert sent["type"] == "audio.chunk"
    assert base64.b64decode(str(sent["samples"])) == SILENT_CHUNK
    assert sent["chunk_seq"] == 0
    assert sent["sample_rate_hz"] == SAMPLE_RATE_HZ
    assert sent["duration_ms"] == WINDOW_MS
    assert sent["monotonic_time_ms"] == 0
    assert sent["is_final"] is False
    await _settle(stream)


async def test_send_audio_advances_chunk_seq_and_position() -> None:
    transport = _FakeTransport(
        post_response=_created_session_response(),
        ws_frames=[_accepted_frame()],
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)
    stream = client.create_stream()

    assert await stream.send_audio(SILENT_CHUNK) is True
    assert await stream.send_audio(SILENT_CHUNK, is_final=True) is True

    assert transport.ws_connection is not None
    first, second = transport.ws_connection.sent
    assert (first["chunk_seq"], first["monotonic_time_ms"]) == (0, 0)
    assert (second["chunk_seq"], second["monotonic_time_ms"]) == (1, WINDOW_MS)
    assert second["is_final"] is True
    await _settle(stream)


async def test_a_refusal_is_resent_under_its_original_chunk_seq_and_returns_false() -> None:
    """§S1: no positive ack exists, so the ONLY safe resend is the last-sent
    chunk's own bytes - never the new window the caller just offered."""
    transport = _FakeTransport(
        post_response=_created_session_response(),
        ws_frames=[
            _accepted_frame(),
            {
                "schema_version": "1.0.0",
                "session_id": "sess-0001",
                "message_id": "msg-bp",
                "event_seq": 2,
                "monotonic_time_ms": 0,
                "type": "backpressure.requested",
                "payload": {"queue_depth": 1, "chunk_seq": 0},
            },
        ],
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)
    stream = client.create_stream()

    assert await stream.send_audio(SILENT_CHUNK) is True
    # Deterministic synchronisation: the reader task sets the refusal flag
    # BEFORE it enqueues the event, so once `receive()` returns it, the flag
    # is guaranteed set - no sleep-based guessing needed.
    refusal = await stream.receive()
    assert refusal.type is ServerMessageType.BACKPRESSURE_REQUESTED
    assert refusal.payload["chunk_seq"] == 0

    a_different_window = b"\x11\x11" * FRAMES_PER_WINDOW
    accepted = await stream.send_audio(a_different_window)

    assert accepted is False, "the caller's new window was not sent - it must offer it again"
    assert transport.ws_connection is not None
    resend = transport.ws_connection.sent[-1]
    assert resend["chunk_seq"] == 0
    assert base64.b64decode(str(resend["samples"])) == SILENT_CHUNK
    await _settle(stream)


# ---------------------------------------------------------------------------
# finish() and abort()
# ---------------------------------------------------------------------------


async def test_finish_drains_polls_409_then_200_and_returns_the_progress_fields() -> None:
    document = {
        "schema_version": "1.0.0",
        "session_id": "sess-0001",
        "run_id": "run-0001",
        "ranking_authority": "none",
        "manifest": {
            "pipeline_version": "0.1.0",
            "schema_version": "1.0.0",
            "taxonomy_version": "1.0.0",
            "configuration_id": "config-default-v1",
            "models": {"recogniser": "deterministic-speech-v1"},
        },
        "transcript": {
            "raw_text": "hola",
            "finalized_through_ms": 300,
            "unaligned_token_count": 0,
            "tokens": [
                {
                    "id": "tok-1",
                    "sequence": [0, 0],
                    "raw_text": "hola",
                    "model_version": "deterministic-speech-v1",
                    "placed": True,
                    "start_ms": 0,
                    "end_ms": 300,
                    "tolerance_ms": 100,
                    "confidence_available": True,
                    "confidence": 0.9,
                    "calibration": "raw",
                    "status": "final",
                }
            ],
        },
        "speech_events": [],
        "visual_events": [],
        "prosody": [],
        "cooccurrences": [],
        "quality": {"assessments": [], "availability": []},
    }
    transport = _FakeTransport(
        post_response=_created_session_response(),
        get_responses={
            "/result": [
                (409, {"code": "result_not_ready", "message": "still completing"}),
                (200, document),
            ],
        },
        ws_frames=[
            _accepted_frame(),
            {
                "schema_version": "1.0.0",
                "session_id": "sess-0001",
                "message_id": "msg-completed",
                "event_seq": 2,
                "monotonic_time_ms": 1_000,
                "type": "session.completed",
                "payload": {
                    "run_id": "run-0001",
                    "speech_events": 0,
                    "visual_events": 0,
                    "cooccurrences": 0,
                    "ranking_authority": "none",
                    "finalized_through_ms": 300,
                    "captured_ms": 1_000,
                },
            },
        ],
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)
    stream = client.create_stream()
    assert await stream.send_audio(SILENT_CHUNK, is_final=True) is True

    result = await stream.finish()

    assert result.evidence == evidence_from_json(document)
    assert result.finalized_through_ms == 300
    assert result.captured_ms == 1_000
    assert result.backpressure_signals == 0
    assert len(transport.get_calls) == 2, "409 then 200 - exactly two polls"
    assert transport.ws_connection is not None
    assert transport.ws_connection.closed


async def test_abort_sends_session_abort_and_is_idempotent() -> None:
    transport = _FakeTransport(
        post_response=_created_session_response(),
        ws_frames=[
            _accepted_frame(),
            {
                "schema_version": "1.0.0",
                "session_id": "sess-0001",
                "message_id": "msg-aborted",
                "event_seq": 2,
                "monotonic_time_ms": 1_000,
                "type": "session.aborted",
                "payload": {"run_id": "run-0001", "captured_ms": 1_000},
            },
        ],
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)
    stream = client.create_stream()
    assert await stream.send_audio(SILENT_CHUNK) is True

    await stream.abort()
    await stream.abort()  # idempotent: must not raise or resend anything

    assert transport.ws_connection is not None
    abort_messages = [f for f in transport.ws_connection.sent if f["type"] == "session.abort"]
    assert len(abort_messages) == 1
    assert transport.ws_connection.closed


async def test_abort_before_any_send_is_a_safe_no_op() -> None:
    client = OratoriaClient("http://engine.local", "test-key", transport=_FakeTransport())
    stream = client.create_stream()

    await stream.abort()
    await stream.abort()


# ---------------------------------------------------------------------------
# Conformance: the remote translation and the embedded translation agree
# ---------------------------------------------------------------------------


def _canonical(evidence: Evidence) -> dict[str, object]:
    """Everything two translations of the SAME session must agree on, with
    the identifiers and stamps that differ by construction removed - the
    same exclusions `test_sdk_surface.py::_canonical` documents and for the
    same reason: comparing them raw would fail on every run."""
    return {
        "raw_text": evidence.transcript.raw_text,
        "unaligned": evidence.transcript.unaligned_count,
        "words": [
            (word.text, word.sequence, word.is_timed, word.status)
            for word in evidence.transcript.words
        ],
        "speech_events": [(e.type, e.start_ms, e.raw_text) for e in evidence.speech_events],
        "visual_events": [e.type for e in evidence.visual_events],
        "ranking_authority": evidence.ranking_authority,
        "manifest": (
            evidence.manifest.taxonomy_version,
            evidence.manifest.schema_version,
            evidence.manifest.pipeline_version,
            evidence.manifest.configuration_id,
        ),
    }


async def _hosted_document(container: Container, auth: dict[str, str]) -> dict[str, Any]:
    """A REAL rendered document - produced by streaming one scripted session
    through the deterministic app in-process, not hand-written."""
    with TestClient(create_app(container)) as client:
        created = client.post("/v1/sessions", json=CREATE_BODY, headers=auth).json()
        session_id = created["session_id"]
        token = created["stream_token"]

        with client.websocket_connect(f"/v1/sessions/{session_id}/stream?token={token}") as socket:
            socket.receive_json()  # session.accepted
            for index in range(3):
                position_ms = index * WINDOW_MS
                socket.send_json(
                    {
                        "schema_version": "1.0.0",
                        "session_id": session_id,
                        "message_id": f"msg-{index}",
                        "chunk_seq": index,
                        "monotonic_time_ms": position_ms,
                        "type": "audio.chunk",
                        "samples": base64.b64encode(SILENT_CHUNK).decode("ascii"),
                        "sample_rate_hz": SAMPLE_RATE_HZ,
                        "duration_ms": WINDOW_MS,
                        "is_final": index == 2,
                    }
                )
            socket.send_json(
                {
                    "schema_version": "1.0.0",
                    "session_id": session_id,
                    "message_id": "msg-complete",
                    "monotonic_time_ms": 3 * WINDOW_MS,
                    "type": "session.complete",
                }
            )
            for _ in range(200):
                message = socket.receive_json()
                if message["type"] == "session.completed":
                    break

        return client.get(f"/v1/sessions/{session_id}/result", headers=auth).json()


async def _embedded_evidence(speech_script: SpeechScript, visual_script: VisualScript) -> Evidence:
    """The same three-window session, run through the embedded engine."""
    engine = OratoriaEngine.local(
        EngineConfiguration(
            speech_runtime=DeterministicSpeechRuntime(speech_script),
            vision_runtime=DeterministicVisionRuntime(visual_script),
            window_seconds=1,
        ),
        clock=FrozenClock(start_ms=1_000_000),
    )
    await engine.warmup()
    async with engine.create_stream() as stream:
        for index in range(3):
            await stream.send_audio(SILENT_CHUNK, is_final=index == 2)
        result = await stream.finish()
    return result.evidence


async def test_evidence_from_json_matches_the_embedded_translation(
    container: Container,
    auth: dict[str, str],
    speech_script: SpeechScript,
    visual_script: VisualScript,
) -> None:
    """ADR-011's deferred conformance test: local and hosted must agree."""
    hosted_document = await _hosted_document(container, auth)
    remote_evidence = evidence_from_json(hosted_document)
    embedded_evidence = await _embedded_evidence(speech_script, visual_script)

    assert _canonical(remote_evidence) == _canonical(embedded_evidence)


# ---------------------------------------------------------------------------
# One test against the real app, over a real socket
# ---------------------------------------------------------------------------


@pytest.fixture
def running_app(container: Container) -> Iterator[str]:
    """The real ASGI application, served by ``uvicorn`` on an ephemeral port
    in a background thread. Memory backend, deterministic runtimes - no
    infrastructure, so this needs no `integration` mark."""
    app = create_app(container)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not report ready within the bound"

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def test_oratoria_client_drives_a_real_session_end_to_end(
    running_app: str, auth: dict[str, str]
) -> None:
    api_key = auth["Authorization"].removeprefix("Bearer ")
    client = OratoriaClient(running_app, api_key)
    try:
        await client.warmup()
        assert ModelRole.RECOGNISER in client.contributions

        stream = client.create_stream()
        for index in range(3):
            accepted = await stream.send_audio(SILENT_CHUNK, is_final=index == 2)
            assert accepted is True
        result = await stream.finish()
    finally:
        await client.aclose()

    assert result.evidence.transcript.raw_text
    assert result.finalized_through_ms is not None
    assert result.finalized_through_ms > 0
    assert result.captured_ms == 3 * WINDOW_MS
