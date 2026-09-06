"""S2: ``session.completed`` states how far the run got, not only that it did.

Before this change the payload carried counts (how many speech events, how
many visual events) and an id - nothing about the timeline. A client whose
connection drops right after ``session.completed`` has no other way to learn
either of two things that are genuinely different questions:

``finalized_through_ms`` - how much of the transcript will not be revised.
Read off the *completed* document, after ``finalize_remaining`` settled
whatever was still provisional, so it agrees with what
``GET /v1/sessions/{id}/result`` renders.

``captured_ms`` - the end of the last audio window the run *ingested*,
independent of whether the speech modality managed to transcribe it. A run
whose recogniser degraded on every window still captured audio, and a caller
needs to tell that apart from a run that captured nothing.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.contract

SCHEMA_VERSION = "1.0.0"
SAMPLE_RATE_HZ = 16_000
WINDOW_MS = 1_000
FRAMES_PER_WINDOW = SAMPLE_RATE_HZ * WINDOW_MS // 1_000
SILENT_CHUNK = base64.b64encode(b"\x00\x00" * FRAMES_PER_WINDOW).decode("ascii")

CREATE_BODY = {
    "mode": "realtime",
    "capabilities": {
        "audio_codec": "pcm16",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "locale": "es-PE",
    },
    "consent_policy_version": "1.0.0",
}


def _envelope(session_id: str, seq: int, position_ms: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "message_id": f"msg-{seq}",
        "chunk_seq": seq,
        "monotonic_time_ms": position_ms,
    }


def _audio(session_id: str, seq: int, position_ms: int, *, final: bool = False) -> dict[str, Any]:
    return {
        **_envelope(session_id, seq, position_ms),
        "type": "audio.chunk",
        "samples": SILENT_CHUNK,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "duration_ms": WINDOW_MS,
        "is_final": final,
    }


def test_session_completed_reports_finalized_and_captured_progress(
    client: TestClient, auth: dict[str, str]
) -> None:
    created = client.post("/v1/sessions", json=CREATE_BODY, headers=auth).json()
    session_id = created["session_id"]
    token = created["stream_token"]

    with client.websocket_connect(f"/v1/sessions/{session_id}/stream?token={token}") as socket:
        assert socket.receive_json()["type"] == "session.accepted"

        for index in range(3):
            socket.send_json(_audio(session_id, index, index * WINDOW_MS, final=index == 2))

        socket.send_json({**_envelope(session_id, 99, 3 * WINDOW_MS), "type": "session.complete"})

        completed: dict[str, Any] | None = None
        for _ in range(200):
            message = socket.receive_json()
            if message["type"] == "session.completed":
                completed = message
                break

    assert completed is not None, "session.completed was never received"
    payload = completed["payload"]

    # Three 1 s windows were ingested; nothing about transcription changes
    # that number.
    assert payload["captured_ms"] == 3 * WINDOW_MS
    # The recogniser produced words in that stretch, so some of it settled.
    assert 0 < payload["finalized_through_ms"] <= payload["captured_ms"]

    # The wire and the stored document agree: this is the same frontier
    # `GET /result` renders, not a second number computed a different way.
    result = client.get(f"/v1/sessions/{session_id}/result", headers=auth).json()
    assert result["transcript"]["finalized_through_ms"] == payload["finalized_through_ms"]
