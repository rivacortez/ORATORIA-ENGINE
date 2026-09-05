"""Provisioning: the half of key management that did not exist.

The engine could authenticate a key and revoke one, and could not create one.
That meant the published API was uncallable by anyone who had not been handed
a credential nothing could produce - a service with a front door and no way to
cut a key for it.

The tests are ordered by what they defend, most important first.

**Privilege containment.** `Scope.ADMIN` is a superset of every other scope.
The one property that makes a leak of an operator key recoverable is that the
leaked key cannot mint its own successor, so `/v1/admin` refusing to issue
`admin` is asserted before anything else here. If that guard is ever removed,
revoking a compromised operator key stops being sufficient and nothing else in
the suite would notice.

**Secret handling.** The plaintext appears in exactly one response body. Every
other route returns descriptors, and the test asserts the *absence* of a
`secret` field rather than trusting that nobody added one - the same reason the
`ApiKeyRow` model has no column for it.

**Tenant attribution.** A key's tenant comes from its application, never from
the request that created it, because the key's tenant is what every isolation
check in the system reads (NFR-013).
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from evidence_engine.adapters.outbound.persistence.identity import InMemoryApiKeyDirectory
from evidence_engine.application.commands.administer_keys import ISSUABLE_SCOPES
from evidence_engine.application.ports.platform import Scope
from evidence_engine.bootstrap.container import Container
from evidence_engine.domain.shared.identifiers import ApplicationId, TenantId

from .conftest import TEST_PEPPER

A_TENANT = "acme-university"


@pytest.fixture
def operator_key(container: Container) -> str:
    """A key carrying `admin`, minted the only way one legitimately is.

    Registered straight into the directory rather than issued through the API,
    because the API refuses - which is the property under test two functions
    down. In a real deployment this is `evidence-engine bootstrap-admin`.
    """
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, _ = directory.issue(
        application=ApplicationId("platform-operator"),
        tenant=TenantId("operator"),
        scopes=frozenset({Scope.ADMIN}),
    )
    return secret


@pytest.fixture
def operator(operator_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {operator_key}"}


@pytest.fixture
def application_id(client: TestClient, operator: dict[str, str]) -> str:
    response = client.post(
        "/v1/admin/applications",
        headers=operator,
        json={"tenant": A_TENANT, "name": "Practice app (production)"},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["application_id"])


# --------------------------------------------------------------------------
# Privilege containment
# --------------------------------------------------------------------------


def test_the_admin_api_refuses_to_mint_another_operator_key(
    client: TestClient, operator: dict[str, str], application_id: str
) -> None:
    """The single property that bounds the blast radius of a leaked operator key.

    With it, a stolen operator credential can create customer keys - bad, and
    undone by revoking it. Without it, the thief mints a second operator key
    before anybody notices and revoking the first achieves nothing.
    """
    response = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:write", "admin"]},
    )

    assert response.status_code == 400, response.text
    body = response.text
    assert "admin" in body
    assert "bootstrap-admin" in body


def test_a_customer_key_cannot_reach_the_admin_api(
    client: TestClient, auth: dict[str, str], application_id: str
) -> None:
    """`auth` is a full capture credential and still must be refused.

    It carries every session and results scope. If scope checking here ever
    devolves into "is authenticated", this is the test that fails.
    """
    for method, path, body in (
        ("post", "/v1/admin/applications", {"tenant": A_TENANT, "name": "x"}),
        ("post", f"/v1/admin/applications/{application_id}/keys", {"scopes": ["sessions:read"]}),
        ("get", f"/v1/admin/applications/{application_id}/keys", None),
        ("delete", "/v1/admin/keys/whatever", None),
    ):
        call = getattr(client, method)
        response = call(path, headers=auth) if body is None else call(path, headers=auth, json=body)
        assert response.status_code == 403, f"{method.upper()} {path}: {response.text}"


def test_the_admin_api_is_not_open(client: TestClient, application_id: str) -> None:
    """No credential at all is a 401, not a 403 and certainly not a 201."""
    response = client.post(
        f"/v1/admin/applications/{application_id}/keys", json={"scopes": ["sessions:read"]}
    )
    assert response.status_code == 401


def test_every_published_scope_except_admin_is_issuable() -> None:
    """Computed by subtraction, and this proves it stayed that way.

    A hand-written list would be a second declaration of the scope set: a scope
    added to the enum would silently become un-issuable, and the failure would
    surface as a portal unable to grant a permission that exists.
    """
    assert frozenset(Scope) - {Scope.ADMIN} == ISSUABLE_SCOPES
    assert Scope.ADMIN not in ISSUABLE_SCOPES
    assert len(ISSUABLE_SCOPES) == len(Scope) - 1


# --------------------------------------------------------------------------
# Secret handling
# --------------------------------------------------------------------------


def test_the_issued_key_authenticates_and_is_returned_exactly_once(
    client: TestClient, operator: dict[str, str], application_id: str
) -> None:
    """The whole point: a key created through the API opens the API.

    Then it is gone. The listing that follows renders the same key and carries
    no field that could hold it.
    """
    issued = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:write", "sessions:read"]},
    )
    assert issued.status_code == 201, issued.text
    secret = issued.json()["secret"]

    created = client.post(
        "/v1/sessions",
        headers={"Authorization": f"Bearer {secret}"},
        json={
            "mode": "realtime",
            "capabilities": {
                "audio_codec": "pcm16",
                "sample_rate_hz": 16_000,
                "locale": "es-PE",
            },
            "consent_policy_version": "1.0.0",
        },
    )
    assert created.status_code == 201, created.text

    listed = client.get(f"/v1/admin/applications/{application_id}/keys", headers=operator)
    assert listed.status_code == 200
    keys = listed.json()["keys"]
    assert len(keys) == 1
    assert "secret" not in keys[0]
    assert secret not in listed.text


def test_only_the_hash_reaches_storage(
    client: TestClient, operator: dict[str, str], application_id: str, container: Container
) -> None:
    """FR-002 holds on the provisioning path too, not only on the bootstrap one."""
    from evidence_engine.adapters.outbound.persistence.identity import hash_secret

    issued = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:read"]},
    )
    secret = issued.json()["secret"]

    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    stored = list(directory._by_hash)
    assert hash_secret(secret, TEST_PEPPER) in stored
    assert secret not in stored


def test_the_prefix_identifies_a_key_without_being_usable(
    client: TestClient, operator: dict[str, str], application_id: str
) -> None:
    """What a dashboard renders in the list.

    Long enough to pick one key out of ten, short enough that the rest of the
    secret is 26 characters of entropy nobody can derive from it.
    """
    issued = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:read"]},
    )
    body = issued.json()
    secret, prefix = body["secret"], body["key"]["prefix"]

    assert secret.startswith(prefix)
    assert prefix.startswith("oek_")
    assert len(prefix) < len(secret)

    refused = client.get("/v1/capabilities", headers={"Authorization": f"Bearer {prefix}"})
    assert refused.status_code == 401


# --------------------------------------------------------------------------
# Tenant attribution and lifecycle
# --------------------------------------------------------------------------


def test_the_key_takes_its_tenant_from_the_application(
    client: TestClient, operator: dict[str, str], application_id: str
) -> None:
    """Not from the request, which cannot name one.

    A body that could set both would let a caller file a key under one tenant's
    application while stamping another tenant on the key - and the stamp is
    what NFR-013's isolation reads.
    """
    issued = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:read"]},
    )
    assert issued.json()["key"]["tenant"] == A_TENANT

    # The operator's own tenant is `operator`; the key it just minted is not
    # in it. That cross-tenant reach is exactly why `admin` is never issued to
    # a customer.
    assert issued.json()["key"]["tenant"] != "operator"


def test_revocation_blocks_the_key_immediately(
    client: TestClient, operator: dict[str, str], application_id: str
) -> None:
    """US-006. The key works, then it does not, with nothing in between."""
    issued = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:read", "results:read"]},
    )
    secret = issued.json()["secret"]
    key_id = issued.json()["key"]["key_id"]
    customer = {"Authorization": f"Bearer {secret}"}

    assert client.get("/v1/capabilities", headers=customer).status_code == 200

    revoked = client.delete(f"/v1/admin/keys/{key_id}", headers=operator)
    assert revoked.status_code == 200
    assert revoked.json()["revoked_at_ms"] is not None

    assert client.get("/v1/capabilities", headers=customer).status_code == 401


def test_revoking_twice_is_the_same_answer_not_an_error(
    client: TestClient, operator: dict[str, str], application_id: str
) -> None:
    """An operator pressing the button twice is asking for a state.

    Returning 404 the second time reads as "the key is still live", which is
    the opposite of what happened.
    """
    issued = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:read"]},
    )
    key_id = issued.json()["key"]["key_id"]

    first = client.delete(f"/v1/admin/keys/{key_id}", headers=operator)
    second = client.delete(f"/v1/admin/keys/{key_id}", headers=operator)

    assert first.status_code == second.status_code == 200
    # The same instant, not a fresh one: the moment the credential stopped
    # working is what an incident review needs from this field.
    assert first.json()["revoked_at_ms"] == second.json()["revoked_at_ms"]


def test_a_revoked_key_stays_in_the_listing(
    client: TestClient, operator: dict[str, str], application_id: str
) -> None:
    """Revocation stops a credential working; it does not erase that it existed.

    An operator investigating an incident needs to see the key that was used.
    """
    issued = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:read"]},
    )
    key_id = issued.json()["key"]["key_id"]
    client.delete(f"/v1/admin/keys/{key_id}", headers=operator)

    listed = client.get(f"/v1/admin/applications/{application_id}/keys", headers=operator)
    keys = {key["key_id"]: key for key in listed.json()["keys"]}
    assert key_id in keys
    assert keys[key_id]["revoked_at_ms"] is not None


def test_issuing_against_an_unknown_application_is_refused(
    client: TestClient, operator: dict[str, str]
) -> None:
    """Otherwise the FK failure would surface as a 500 from the database."""
    response = client.post(
        "/v1/admin/applications/app_does_not_exist/keys",
        headers=operator,
        json={"scopes": ["sessions:read"]},
    )
    assert response.status_code == 404, response.text


def test_a_key_with_no_scopes_is_refused(
    client: TestClient, operator: dict[str, str], application_id: str
) -> None:
    """It would authenticate and be able to do nothing - a support ticket."""
    response = client.post(
        f"/v1/admin/applications/{application_id}/keys", headers=operator, json={"scopes": []}
    )
    assert response.status_code in (400, 422), response.text


@pytest.mark.parametrize(
    ("expires_at_ms", "why"),
    [
        pytest.param(1, "in the past", id="already-expired"),
        pytest.param(10**15, "more than", id="effectively-unbounded"),
    ],
)
def test_an_unusable_expiry_is_refused(
    client: TestClient,
    operator: dict[str, str],
    application_id: str,
    expires_at_ms: int,
    why: str,
) -> None:
    response = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:read"], "expires_at_ms": expires_at_ms},
    )
    assert response.status_code == 400, response.text
    assert why in response.text


def test_provisioning_lands_in_the_audit_log(
    client: TestClient, operator: dict[str, str], application_id: str, container: Container
) -> None:
    """NFR-022 names key management as one of the trails that must be recorded.

    The record carries the prefix and not the secret: the prefix is in the
    clear by design, and NFR-010 keeps the raw value out of every log.
    """
    issued = client.post(
        f"/v1/admin/applications/{application_id}/keys",
        headers=operator,
        json={"scopes": ["sessions:read"]},
    )
    secret = issued.json()["secret"]
    key_id = issued.json()["key"]["key_id"]

    # Read through the port rather than the adapter's private dict: the trail
    # is only evidence of anything if the published way of reading it shows the
    # record.
    entries = asyncio.run(container.audit.for_resource(key_id))
    issue_records = [entry for entry in entries if entry.action == "api_key.issue"]
    assert len(issue_records) == 1
    record = issue_records[0]
    assert record.resource == key_id
    assert record.tenant.value == A_TENANT
    assert secret not in record.detail
    assert issued.json()["key"]["prefix"] in record.detail
