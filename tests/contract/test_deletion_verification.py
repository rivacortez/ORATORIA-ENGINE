"""QA-04's measure, checked against every store the deletion writes to.

`CONSENT_AND_RETENTION.md` §4 lists five steps and makes verification "a
separate read-only call". The verification used to count media objects and
stop, so a session whose protected evidence survived step 3 came back clean:
the measure passed on the one failure it exists to catch. It also had no
transport, which meant nothing could run it and nothing could notice.

Two properties carry this module.

*A clean answer has to be earned by every store, not by the one that happened
to be checked.* `test_surviving_evidence_records_fail_the_verification` leaves
the media half spotless on purpose - zero objects remaining - so the only
reason the session fails is the evidence, and the assertion states in one line
what the old measure would have answered.

*A clean answer has to mean a deletion happened.* Absence is the default state
of a session that was never touched, so a report built only from absences reads
as a flawless deletion of a session nobody deleted. The session transition and
the audit record are the two fields that cannot be satisfied by doing nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from evidence_engine.adapters.outbound.persistence.identity import InMemoryApiKeyDirectory
from evidence_engine.application.commands.delete_evidence import DeleteEvidence
from evidence_engine.application.ports.platform import Scope
from evidence_engine.application.ports.repositories import EvidenceBundle
from evidence_engine.bootstrap.container import Container
from evidence_engine.domain.evidence.document import EvidenceDocument, ProvenanceManifest
from evidence_engine.domain.quality.assessment import QualityReport
from evidence_engine.domain.shared.identifiers import (
    ApplicationId,
    ConfigurationSnapshotId,
    ModelVersionId,
    RunId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import Modality, SemanticVersion
from evidence_engine.domain.shared.taxonomy import TAXONOMY_VERSION
from evidence_engine.domain.transcript.transcript import Transcript

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


def _verification(client: TestClient, session_id: str, headers: dict[str, str]) -> Any:
    return client.get(f"/v1/sessions/{session_id}/evidence/verification", headers=headers)


def _recomputed(body: dict[str, Any]) -> bool:
    """The summary, rebuilt from the fields published beside it.

    The wire contract promises ``deletion_verified`` is the conjunction of the
    other fields and carries no term of its own. Recomputing it here is what
    makes that promise checkable rather than asserted: a future field that the
    summary consults but does not publish would break this line, which is the
    point at which a summary starts attesting to more than it shows.
    """
    return (
        body["media_objects_remaining"] == 0
        and not body["evidence_document_present"]
        and not body["stream_state_present"]
        and body["session_marked_deleted"]
        and body["audit_record_present"]
    )


def _document_for(session_id: SessionId) -> EvidenceDocument:
    """The smallest document a repository can hand back.

    Contents are irrelevant - the verification asks whether anything is there,
    not what it says - so this carries no tokens and no events. The manifest is
    the only part with a constraint: `ProvenanceManifest` refuses a taxonomy
    version this build is not running.
    """
    return EvidenceDocument(
        session_id=session_id,
        run_id=RunId("run-survivor"),
        manifest=ProvenanceManifest(
            pipeline_version=SemanticVersion(0, 1, 0),
            schema_version=SemanticVersion(1, 0, 0),
            taxonomy_version=TAXONOMY_VERSION,
            configuration=ConfigurationSnapshotId("config-default-v1"),
            models={Modality.AUDIO: ModelVersionId("deterministic-speech-v1")},
        ),
        transcript=Transcript(),
        quality=QualityReport(),
    )


class _EvidenceThatSurvivesDeletion:
    """An evidence repository whose deletion reports a count and removes nothing.

    The failure QA-04 exists to catch, made reachable from a test. Real shapes
    of it: a purge scoped to the wrong tenant, or a transaction rolled back
    after the media store had already committed its half. Both hand back a
    plausible count and leave the rows where they were, which is why the count
    in the receipt cannot be the thing that verifies the deletion.
    """

    def __init__(self, document: EvidenceDocument, tenant: TenantId) -> None:
        self._document = document
        self._tenant = tenant

    async def store(self, bundle: EvidenceBundle) -> None: ...

    async def load(self, tenant: TenantId, run_id: RunId) -> EvidenceBundle | None:
        return None

    async def store_document(self, document: EvidenceDocument, tenant: TenantId) -> None: ...

    async def load_document(
        self, tenant: TenantId, session_id: SessionId
    ) -> EvidenceDocument | None:
        if tenant != self._tenant or session_id != self._document.session_id:
            return None
        return self._document

    async def delete_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        return 412


def _with_surviving_evidence(container: Container, session_id: str, tenant: TenantId) -> None:
    """Rebuild the deletion use case over a repository that never deletes.

    The use case captures its ports at construction, so replacing
    ``container.evidence`` after the container is built would leave the wired
    instance pointing at the original. Replacing the use case is the honest
    way to substitute one port.
    """
    container.delete_evidence = DeleteEvidence(
        sessions=container.sessions,
        evidence=_EvidenceThatSurvivesDeletion(_document_for(SessionId(session_id)), tenant),
        media=container.media,
        stream_state=container.stream_state,
        audit=container.audit,
        clock=container.clock,
    )


# ---------------------------------------------------------------------------
# The measure
# ---------------------------------------------------------------------------


def test_surviving_evidence_records_fail_the_verification(
    client: TestClient, container: Container, auth: dict[str, str], tenant: TenantId
) -> None:
    """QA-04's failure: media gone, protected evidence still there."""
    session_id = _create(client, auth)
    _with_surviving_evidence(container, session_id, tenant)

    deleted = client.delete(f"/v1/sessions/{session_id}/evidence", headers=auth)
    assert deleted.status_code == 200, deleted.text
    # The receipt is confident. It is reporting what the deletion believes it
    # did, which is why it cannot double as the verification.
    assert deleted.json()["evidence_records_deleted"] == 412

    body = _verification(client, session_id, auth).json()

    assert body["deletion_verified"] is False
    assert body["evidence_document_present"] is True
    # Every other store is clean, so a measure that counted media objects alone
    # would answer "no recoverable object reference" about this session.
    assert body["media_objects_remaining"] == 0
    assert body["session_marked_deleted"] is True
    assert body["audit_record_present"] is True
    assert body["deletion_verified"] == _recomputed(body)


def test_a_session_that_was_never_deleted_does_not_verify_as_clean(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Absence is the default state, so absences alone cannot prove a deletion.

    Nothing was ever stored for this session and nothing was ever deleted. A
    report assembled only from "is it gone?" questions would call that a
    completed erasure, and the answer renders identically to a real one once it
    is copied into a compliance record.
    """
    session_id = _create(client, auth)

    body = _verification(client, session_id, auth).json()

    assert body["deletion_verified"] is False
    assert body["media_objects_remaining"] == 0
    assert body["session_marked_deleted"] is False
    assert body["audit_record_present"] is False
    assert body["deletion_verified"] == _recomputed(body)


def test_a_completed_deletion_verifies_clean(client: TestClient, auth: dict[str, str]) -> None:
    session_id = _create(client, auth)
    assert client.delete(f"/v1/sessions/{session_id}/evidence", headers=auth).status_code == 200

    body = _verification(client, session_id, auth).json()

    assert body["deletion_verified"] is True
    assert body["session_marked_deleted"] is True
    assert body["audit_record_present"] is True
    assert body["stream_state_present"] is False
    assert body["deletion_verified"] == _recomputed(body)


def test_the_idempotent_repeat_still_verifies_clean(
    client: TestClient, auth: dict[str, str]
) -> None:
    """US-014: a client retrying after a timeout must not be told it failed."""
    session_id = _create(client, auth)
    client.delete(f"/v1/sessions/{session_id}/evidence", headers=auth)
    repeat = client.delete(f"/v1/sessions/{session_id}/evidence", headers=auth)
    assert repeat.json()["already_deleted"] is True

    assert _verification(client, session_id, auth).json()["deletion_verified"] is True


def test_verifying_twice_gives_the_same_answer(
    client: TestClient, container: Container, auth: dict[str, str], tenant: TenantId
) -> None:
    """§4's reason for keeping verification off the deletion path.

    A verification that removed what it found would pass on its second run and
    destroy the evidence an operator needs after the first. Asserted on a
    session with real residue, because on a clean one the two runs agree for
    the wrong reason.
    """
    session_id = _create(client, auth)
    _with_surviving_evidence(container, session_id, tenant)
    client.delete(f"/v1/sessions/{session_id}/evidence", headers=auth)

    first = _verification(client, session_id, auth).json()
    second = _verification(client, session_id, auth).json()

    assert first == second
    assert second["deletion_verified"] is False


# ---------------------------------------------------------------------------
# Reachability and access (FR-003, NFR-013)
# ---------------------------------------------------------------------------


def test_verifying_an_unknown_session_is_refused(client: TestClient, auth: dict[str, str]) -> None:
    """404, not a clean report about nothing."""
    response = _verification(client, "session-never-existed", auth)

    assert response.status_code == 404
    assert response.json()["code"] == "session_not_found"


def test_verification_needs_the_deletion_scope(
    client: TestClient, container: Container, tenant: TenantId, auth: dict[str, str]
) -> None:
    """The report names what a session still holds, so it is not a read scope."""
    session_id = _create(client, auth)

    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, _ = directory.issue(
        application=ApplicationId("app-oratoria"),
        tenant=tenant,
        scopes=frozenset({Scope.SESSIONS_READ, Scope.RESULTS_READ}),
    )

    response = _verification(client, session_id, {"Authorization": f"Bearer {secret}"})

    assert response.status_code == 403
    assert response.json()["code"] == "insufficient_scope"


def test_another_tenant_cannot_verify_a_deletion(
    client: TestClient, auth: dict[str, str], other_tenant_key: str
) -> None:
    """404, not 403. A 403 would confirm the session exists."""
    session_id = _create(client, auth)
    client.delete(f"/v1/sessions/{session_id}/evidence", headers=auth)

    response = _verification(client, session_id, {"Authorization": f"Bearer {other_tenant_key}"})

    assert response.status_code == 404
    assert response.json()["code"] == "session_not_found"
