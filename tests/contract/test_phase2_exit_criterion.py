"""Phase 2's exit criterion, executed rather than asserted in prose.

§13 Phase 2: "A synthetic session can be streamed, completed, queried and
deleted without model inference."

This module runs that sentence end to end against the real ASGI application:
create over REST, stream over WebSocket, complete, read the evidence document,
delete it, and verify the deletion. Everything below the transports is real
except the two model runtimes, which replay a script.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.contract

SCHEMA_VERSION = "1.0.0"

CREATE_BODY = {
    "mode": "realtime",
    "capabilities": {
        "audio_codec": "pcm16",
        "sample_rate_hz": 16_000,
        "locale": "es-PE",
        "video_format": "landmarks",
        "frame_rate_fps": 30,
    },
    "consent_policy_version": "1.0.0",
    "retention": {"retain_raw_media": False, "raw_media_ttl_seconds": 0},
}

#: 160 ms of PCM16 silence at 16 kHz. The deterministic runtime keys off the
#: window position, not the samples, so real audio would tell us nothing extra.
SILENT_CHUNK = base64.b64encode(b"\x00\x00" * 2_560).decode("ascii")


def _envelope(session_id: str, seq: int, position_ms: int) -> dict[str, Any]:
    """The header §7.4 requires on every client message."""
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
        "sample_rate_hz": 16_000,
        "duration_ms": 1_000,
        "is_final": final,
    }


def _landmarks(session_id: str, seq: int, position_ms: int) -> dict[str, Any]:
    return {
        **_envelope(session_id, seq, position_ms),
        "type": "visual.features",
        "session_position_ms": position_ms,
        "landmarks": [0.5, 0.5, 0.1, 0.2],
    }


def _stream_session(client: TestClient, session_id: str, token: str) -> list[dict[str, Any]]:
    """Run the synthetic presentation and collect every server message."""
    received: list[dict[str, Any]] = []
    url = f"/v1/sessions/{session_id}/stream?token={token}"

    with client.websocket_connect(url) as socket:
        received.append(socket.receive_json())  # session.accepted

        for index in range(8):
            position_ms = index * 1_000
            socket.send_json(_audio(session_id, index, position_ms, final=index == 7))
            socket.send_json(_landmarks(session_id, index, position_ms))

        socket.send_json(
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "message_id": "msg-complete",
                "monotonic_time_ms": 8_000,
                "type": "session.complete",
            }
        )

        # Drain until the server says the session is done. The socket closes
        # right after, so a bounded loop is the honest way to read it.
        for _ in range(400):
            message = socket.receive_json()
            received.append(message)
            if message["type"] == "session.completed":
                break

    return received


def test_a_synthetic_session_streams_completes_is_queried_and_is_deleted(
    client: TestClient, auth: dict[str, str]
) -> None:
    # 1. Create -----------------------------------------------------------
    created = client.post("/v1/sessions", json=CREATE_BODY, headers=auth)
    assert created.status_code == 201, created.text
    body = created.json()

    session_id = body["session_id"]
    token = body["stream_token"]
    assert body["state"] == "created"
    assert body["schema_version"] == SCHEMA_VERSION
    # FR-006: the negotiated shape comes back, not merely an acknowledgement.
    assert body["capabilities"]["sample_rate_hz"] == 16_000
    assert body["capabilities"]["video_format"] == "landmarks"

    # 2. Stream -----------------------------------------------------------
    messages = _stream_session(client, session_id, token)
    types = [message["type"] for message in messages]

    assert types[0] == "session.accepted"
    assert types[-1] == "session.completed"
    assert "transcript.final" in types
    assert "speech_event.final" in types

    # §7.4: every message carries the envelope, and event_seq is monotone.
    sequences = [message["event_seq"] for message in messages]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    for message in messages:
        assert message["schema_version"] == SCHEMA_VERSION
        assert message["session_id"] == session_id
        assert message["message_id"]
        assert message["monotonic_time_ms"] >= 0

    # 3. Query ------------------------------------------------------------
    status = client.get(f"/v1/sessions/{session_id}", headers=auth)
    assert status.status_code == 200
    assert status.json()["state"] == "completed"

    result = client.get(f"/v1/sessions/{session_id}/result", headers=auth)
    assert result.status_code == 200, result.text
    document = result.json()

    assert document["session_id"] == session_id
    assert document["ranking_authority"] == "none"
    assert document["transcript"]["raw_text"].startswith("buenos dias")
    assert document["manifest"]["taxonomy_version"] == "1.0.0"
    assert document["manifest"]["models"]["audio"] == "deterministic-speech-v1"

    # 4. Delete -----------------------------------------------------------
    deleted = client.delete(f"/v1/sessions/{session_id}/evidence", headers=auth)
    assert deleted.status_code == 200, deleted.text
    receipt = deleted.json()
    assert receipt["already_deleted"] is False
    assert receipt["evidence_records_deleted"] >= 1

    # US-014: repeated requests are idempotent.
    again = client.delete(f"/v1/sessions/{session_id}/evidence", headers=auth)
    assert again.status_code == 200
    assert again.json()["already_deleted"] is True

    # The result is gone, and says so distinguishably from "never existed".
    after = client.get(f"/v1/sessions/{session_id}/result", headers=auth)
    assert after.status_code == 409
    assert after.json()["code"] == "result_not_ready"
    assert "deleted" in after.json()["message"]


def test_the_document_carries_the_evidence_the_script_produced(
    client: TestClient, auth: dict[str, str]
) -> None:
    """The pipeline is exercised, not merely traversed."""
    created = client.post("/v1/sessions", json=CREATE_BODY, headers=auth).json()
    _stream_session(client, created["session_id"], created["stream_token"])
    document = client.get(f"/v1/sessions/{created['session_id']}/result", headers=auth).json()

    kinds = {event["type"] for event in document["speech_events"]}
    assert "filled_pause" in kinds
    assert "repetition" in kinds
    assert "lexical_filler" in kinds
    # FR-015: derived by the engine from the transcript under the versioned
    # threshold, never supplied by the runtime.
    assert "silent_pause" in kinds

    # FR-013: the classifier declined, so the role is UNCERTAIN - not guessed.
    ambiguous = next(e for e in document["speech_events"] if e["type"] == "lexical_filler")
    assert ambiguous["context_role"] == "uncertain"
    # FR-017: the raw expression is kept regardless of role.
    assert ambiguous["raw_text"] == "este"
    # ... and an unresolved role does not count against the speaker.
    assert ambiguous["counts_as_disfluency"] is False

    visual_kinds = {event["type"] for event in document["visual_events"]}
    assert "gaze_away_from_camera" in visual_kinds
    assert "posture_deviation" in visual_kinds
    # FR-022: the P1 class scored 0.31 and is dropped, not published weakly.
    assert "self_touch" not in visual_kinds

    # NFR-014: provenance on every derived event.
    for event in document["speech_events"] + document["visual_events"]:
        provenance = event["provenance"]
        assert provenance["model_version"]
        assert provenance["taxonomy_version"] == "1.0.0"
        assert provenance["configuration_id"]
        assert provenance["evidence_ref"]
        # NFR-004: no interval claims an exact boundary.
        assert event["tolerance_ms"] > 0


def test_no_model_inference_is_required(container: Any) -> None:
    """The exit criterion's qualifier, checked rather than assumed."""
    assert container.settings.runtime_mode.value == "deterministic"
    assert container.settings.backend.value == "memory"
