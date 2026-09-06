"""S4: the engine proves it can decode before it claims to be ready, and says
which physical instance answered.

Two gaps this closes, both aimed at the pilot topology - the engine runs on a
GPU workstation and OratorIA's backend consumes it remotely, and a study can
run more than one workstation behind the same consuming application.

*A registered model version is not evidence a decode has run.* The registry
answers `active_for()` from wiring-time state; nothing before this asked the
runtime to actually transcribe anything. The ASGI lifespan now does, once,
before the app starts serving, and `/health/ready` refuses traffic until it
has.

*Nobody could tell which instance produced a session.* `session.accepted`
and `/v1/capabilities` both name it now.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from evidence_engine.bootstrap.app import create_app
from evidence_engine.bootstrap.container import Container

pytestmark = pytest.mark.contract

SCHEMA_VERSION = "1.0.0"
CREATE_BODY = {
    "mode": "realtime",
    "capabilities": {"audio_codec": "pcm16", "sample_rate_hz": 16_000, "locale": "es-PE"},
    "consent_policy_version": "1.0.0",
}


def test_the_lifespan_warms_up_the_wired_speech_runtime_once(container: Container) -> None:
    """Before the app starts serving, nothing has decoded anything yet."""
    assert container.speech_warm_seconds is None

    with TestClient(create_app(container)):
        pass

    assert container.speech_warm_seconds is not None
    assert container.speech_warm_seconds >= 0.0


def test_readiness_reports_warm_once_the_lifespan_has_run(client: TestClient) -> None:
    """``client`` already entered the lifespan (see ``conftest.py``), so by the
    time any test using it runs, the warm-up decode has already happened."""
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["speech:warm"].startswith("warm (")


def test_session_accepted_names_the_instance_that_accepted_it(
    client: TestClient, auth: dict[str, str], container: Container
) -> None:
    created = client.post("/v1/sessions", json=CREATE_BODY, headers=auth).json()

    with client.websocket_connect(
        f"/v1/sessions/{created['session_id']}/stream?token={created['stream_token']}"
    ) as socket:
        accepted: dict[str, Any] = socket.receive_json()

    assert accepted["type"] == "session.accepted"
    assert accepted["payload"]["instance_id"] == container.profile.instance_id
