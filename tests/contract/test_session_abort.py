"""S3: a client can end a session as a deliberate failure, not just vanish.

Before this change the only way to stop early was to drop the connection,
which §6.3 already treats as a partial failure - correct, but silent about
*intent*. ``session.abort`` is the explicit version: the run closes
unsuccessful, the session moves to ``failed``, the client is told with
``session.aborted`` before the socket closes, and the evidence document never
becomes available - a failed run has none to read.
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


def _audio(session_id: str, seq: int, position_ms: int) -> dict[str, Any]:
    return {
        **_envelope(session_id, seq, position_ms),
        "type": "audio.chunk",
        "samples": SILENT_CHUNK,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "duration_ms": WINDOW_MS,
        "is_final": False,
    }


def _open_and_abort(client: TestClient, auth: dict[str, str]) -> tuple[str, dict[str, Any]]:
    created = client.post("/v1/sessions", json=CREATE_BODY, headers=auth).json()
    session_id = created["session_id"]
    token = created["stream_token"]

    with client.websocket_connect(f"/v1/sessions/{session_id}/stream?token={token}") as socket:
        assert socket.receive_json()["type"] == "session.accepted"
        socket.send_json(_audio(session_id, 0, 0))
        socket.send_json({**_envelope(session_id, 1, WINDOW_MS), "type": "session.abort"})

        aborted: dict[str, Any] | None = None
        for _ in range(50):
            message = socket.receive_json()
            if message["type"] == "session.aborted":
                aborted = message
                break

    assert aborted is not None, "session.aborted was never received"
    return session_id, aborted


def test_abort_replies_aborted_and_names_the_run_and_captured_audio(
    client: TestClient, auth: dict[str, str]
) -> None:
    _session_id, aborted = _open_and_abort(client, auth)

    assert aborted["payload"]["run_id"]
    assert aborted["payload"]["captured_ms"] == WINDOW_MS


def test_abort_fails_the_session_rather_than_completing_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    session_id, _aborted = _open_and_abort(client, auth)

    status = client.get(f"/v1/sessions/{session_id}", headers=auth).json()
    assert status["state"] == "failed"


def test_an_aborted_session_has_no_evidence_document_to_read(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A failed run produced no reconciled evidence - `result_not_ready`, not a
    document that happens to be empty."""
    session_id, _aborted = _open_and_abort(client, auth)

    result = client.get(f"/v1/sessions/{session_id}/result", headers=auth)
    assert result.status_code == 409
    assert result.json()["code"] == "result_not_ready"
