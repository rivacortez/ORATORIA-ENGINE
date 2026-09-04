"""FR-001..FR-004 and NFR-013: who may call, and what they may see.

The cross-tenant tests here are the ones NFR-013 names as a release gate. They
check a specific and easily-lost property: another tenant's session is not
merely forbidden, it is *invisible*. A 403 tells the caller the session exists;
a 404 tells them nothing. The difference is the whole of tenant isolation from
an attacker's point of view.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from evidence_engine.adapters.outbound.persistence.identity import InMemoryApiKeyDirectory
from evidence_engine.adapters.outbound.telemetry.clock import FrozenClock
from evidence_engine.application.ports.platform import Scope
from evidence_engine.bootstrap.container import Container
from evidence_engine.domain.shared.identifiers import ApplicationId, TenantId

pytestmark = pytest.mark.contract

CREATE_BODY = {
    "mode": "realtime",
    "capabilities": {
        "audio_codec": "pcm16",
        "sample_rate_hz": 16_000,
        "locale": "es-PE",
    },
    "consent_policy_version": "1.0.0",
}


def _create(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post("/v1/sessions", json=CREATE_BODY, headers=headers)
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


# ---------------------------------------------------------------------------
# Authentication (FR-002, NFR-010)
# ---------------------------------------------------------------------------


def test_a_request_without_a_key_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/sessions", json=CREATE_BODY).status_code == 401


def test_an_unknown_key_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/sessions",
        json=CREATE_BODY,
        headers={"Authorization": "Bearer oek_not-a-real-key"},
    )
    assert response.status_code == 401


def test_a_revoked_key_stops_working_immediately(
    client: TestClient, container: Container, tenant: TenantId, frozen_clock: FrozenClock
) -> None:
    """US-006: revocation blocks new calls immediately."""
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, record = directory.issue(
        application=ApplicationId("app-temp"),
        tenant=tenant,
        scopes=frozenset({Scope.SESSIONS_WRITE}),
    )
    headers = {"Authorization": f"Bearer {secret}"}
    assert client.post("/v1/sessions", json=CREATE_BODY, headers=headers).status_code == 201

    directory.revoke(record.id, at_ms=frozen_clock.epoch_ms())

    assert client.post("/v1/sessions", json=CREATE_BODY, headers=headers).status_code == 401


def test_a_key_is_never_echoed_back(client: TestClient, api_key: str) -> None:
    """NFR-010: the raw secret never appears in a response body."""
    response = client.post(
        "/v1/sessions", json=CREATE_BODY, headers={"Authorization": f"Bearer {api_key}"}
    )
    assert api_key not in response.text


def test_only_the_hash_is_stored(container: Container, tenant: TenantId) -> None:
    """FR-002: only a cryptographic hash is persisted."""
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, record = directory.issue(
        application=ApplicationId("app-x"), tenant=tenant, scopes=frozenset()
    )

    assert record.hashed_secret != secret
    assert secret not in record.hashed_secret
    assert len(record.hashed_secret) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# Authorization (FR-003)
# ---------------------------------------------------------------------------


def test_creating_a_session_requires_the_write_scope(
    client: TestClient, container: Container, tenant: TenantId
) -> None:
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, _ = directory.issue(
        application=ApplicationId("app-readonly"),
        tenant=tenant,
        scopes=frozenset({Scope.SESSIONS_READ}),
    )

    response = client.post(
        "/v1/sessions", json=CREATE_BODY, headers={"Authorization": f"Bearer {secret}"}
    )

    assert response.status_code == 403
    assert response.json()["code"] == "insufficient_scope"


def test_deleting_evidence_needs_its_own_scope(
    client: TestClient, container: Container, tenant: TenantId, auth: dict[str, str]
) -> None:
    """A capture credential must not also be able to erase a study's data."""
    session_id = _create(client, auth)

    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, _ = directory.issue(
        application=ApplicationId("app-oratoria"),
        tenant=tenant,
        scopes=frozenset({Scope.SESSIONS_WRITE, Scope.SESSIONS_READ}),
    )

    response = client.delete(
        f"/v1/sessions/{session_id}/evidence",
        headers={"Authorization": f"Bearer {secret}"},
    )

    assert response.status_code == 403
    assert response.json()["code"] == "insufficient_scope"


def test_the_admin_scope_subsumes_the_others(
    client: TestClient, container: Container, tenant: TenantId
) -> None:
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, _ = directory.issue(
        application=ApplicationId("app-admin"),
        tenant=tenant,
        scopes=frozenset({Scope.ADMIN}),
    )

    response = client.post(
        "/v1/sessions", json=CREATE_BODY, headers={"Authorization": f"Bearer {secret}"}
    )
    assert response.status_code == 201


# ---------------------------------------------------------------------------
# Tenant isolation (NFR-013)
# ---------------------------------------------------------------------------


def test_another_tenants_session_is_invisible_not_forbidden(
    client: TestClient, auth: dict[str, str], other_tenant_key: str
) -> None:
    """404, not 403. A 403 confirms the session exists."""
    session_id = _create(client, auth)

    response = client.get(
        f"/v1/sessions/{session_id}", headers={"Authorization": f"Bearer {other_tenant_key}"}
    )

    assert response.status_code == 404
    assert response.json()["code"] == "session_not_found"


def test_another_tenant_cannot_read_a_result(
    client: TestClient, auth: dict[str, str], other_tenant_key: str
) -> None:
    session_id = _create(client, auth)

    response = client.get(
        f"/v1/sessions/{session_id}/result",
        headers={"Authorization": f"Bearer {other_tenant_key}"},
    )

    assert response.status_code == 404


def test_another_tenant_cannot_delete_evidence(
    client: TestClient, auth: dict[str, str], other_tenant_key: str
) -> None:
    """The one that would be catastrophic, so it gets its own test."""
    session_id = _create(client, auth)

    response = client.delete(
        f"/v1/sessions/{session_id}/evidence",
        headers={"Authorization": f"Bearer {other_tenant_key}"},
    )

    assert response.status_code == 404

    # ... and the session is untouched for its real owner.
    assert client.get(f"/v1/sessions/{session_id}", headers=auth).status_code == 200


# ---------------------------------------------------------------------------
# Idempotency (US-011)
# ---------------------------------------------------------------------------


def test_repeating_an_idempotency_key_returns_the_same_session(
    client: TestClient, auth: dict[str, str]
) -> None:
    headers = {**auth, "Idempotency-Key": "preflight-42"}

    first = client.post("/v1/sessions", json=CREATE_BODY, headers=headers)
    second = client.post("/v1/sessions", json=CREATE_BODY, headers=headers)

    assert first.status_code == 201
    # 200, not 201: nothing was created the second time.
    assert second.status_code == 200
    assert first.json()["session_id"] == second.json()["session_id"]


def test_different_idempotency_keys_create_different_sessions(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = client.post("/v1/sessions", json=CREATE_BODY, headers={**auth, "Idempotency-Key": "a"})
    second = client.post("/v1/sessions", json=CREATE_BODY, headers={**auth, "Idempotency-Key": "b"})

    assert first.json()["session_id"] != second.json()["session_id"]


# ---------------------------------------------------------------------------
# Capability negotiation (FR-006, US-011)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("audio_codec", "mp3"),
        ("sample_rate_hz", 8_000),
        ("locale", "es-AR"),
    ],
)
def test_an_unsupported_capability_is_refused_before_capture(
    client: TestClient, auth: dict[str, str], field: str, value: object
) -> None:
    """US-011's criterion: rejected before capture, not after five minutes."""
    body = {**CREATE_BODY, "capabilities": {**CREATE_BODY["capabilities"], field: value}}

    response = client.post("/v1/sessions", json=body, headers=auth)

    assert response.status_code == 422
    assert response.json()["code"] == "unsupported_capability"


def test_an_audio_only_session_is_valid(client: TestClient, auth: dict[str, str]) -> None:
    """QA-02 already requires speech to survive without video (FR-006)."""
    response = client.post("/v1/sessions", json=CREATE_BODY, headers=auth)

    assert response.status_code == 201
    assert response.json()["capabilities"]["video_format"] is None


def test_a_lossy_codec_is_accepted_with_a_warning(client: TestClient, auth: dict[str, str]) -> None:
    """Driver 1: compression can erase the detail P0 classes are decided from."""
    body = {
        **CREATE_BODY,
        "capabilities": {**CREATE_BODY["capabilities"], "audio_codec": "opus"},
    }

    response = client.post("/v1/sessions", json=body, headers=auth)

    assert response.status_code == 201
    assert response.json()["capabilities"]["lossy_audio_warning"] is True


def test_a_typo_in_a_request_field_is_an_error(client: TestClient, auth: dict[str, str]) -> None:
    """A silently dropped field becomes a session negotiated at the wrong rate."""
    body = {**CREATE_BODY, "sampel_rate_hz": 16_000}

    assert client.post("/v1/sessions", json=body, headers=auth).status_code == 422


# ---------------------------------------------------------------------------
# Published surface
# ---------------------------------------------------------------------------


def test_capabilities_publishes_the_taxonomy_and_the_ranking_invariant(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get("/v1/capabilities", headers=auth)

    assert response.status_code == 200
    body = response.json()
    assert body["taxonomy_version"] == "1.0.0"
    assert body["ranking_authority"] == "none"
    assert "filled_pause" in body["speech_event_types"]
    assert "gaze_away_from_camera" in body["visual_event_types"]
    assert body["locales"] == ["es-PE"]


def test_health_endpoints_need_no_credential(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"]["runtime_mode"] == "deterministic"
