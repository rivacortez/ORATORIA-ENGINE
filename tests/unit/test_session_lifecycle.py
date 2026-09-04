"""The session state machine and the consent gate in front of it."""

from __future__ import annotations

from itertools import pairwise

import pytest

from evidence_engine.domain.sessions.consent import (
    ConsentReceipt,
    ConsentViolation,
    RetentionPolicy,
)
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.sessions.state import (
    SessionState,
    can_transition,
    is_terminal,
    require_transition,
)
from evidence_engine.domain.shared.errors import IllegalSessionTransition
from evidence_engine.domain.shared.identifiers import SessionId
from evidence_engine.domain.shared.provenance import SemanticVersion

POLICY = SemanticVersion(1, 0, 0)


# ---------------------------------------------------------------------------
# The machine itself
# ---------------------------------------------------------------------------


def test_the_happy_path_is_legal() -> None:
    path = [
        SessionState.CREATED,
        SessionState.CAPTURING,
        SessionState.PAUSED,
        SessionState.CAPTURING,
        SessionState.COMPLETING,
        SessionState.COMPLETED,
        SessionState.EVIDENCE_DELETED,
    ]

    for source, target in pairwise(path):
        assert can_transition(source, target), f"{source} -> {target} should be legal"


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (SessionState.CREATED, SessionState.COMPLETED),
        (SessionState.CREATED, SessionState.PAUSED),
        (SessionState.COMPLETED, SessionState.CAPTURING),
        (SessionState.COMPLETING, SessionState.PAUSED),
        (SessionState.EVIDENCE_DELETED, SessionState.CAPTURING),
        (SessionState.EVIDENCE_DELETED, SessionState.COMPLETED),
    ],
)
def test_illegal_transitions_are_refused(source: SessionState, target: SessionState) -> None:
    with pytest.raises(IllegalSessionTransition):
        require_transition(source, target)


def test_deletion_is_terminal() -> None:
    """QA-04 promises the caller that deletion is final."""
    assert is_terminal(SessionState.EVIDENCE_DELETED)
    assert not is_terminal(SessionState.COMPLETED)


def test_a_failed_session_can_still_be_deleted() -> None:
    """It holds media and evidence, so FR-032 must still reach it."""
    assert can_transition(SessionState.FAILED, SessionState.EVIDENCE_DELETED)


# ---------------------------------------------------------------------------
# The consent gate (FR-031)
# ---------------------------------------------------------------------------


def test_capture_without_a_consent_receipt_is_refused(
    consented_session: AnalysisSession,
) -> None:
    without_consent = AnalysisSession(
        id=consented_session.id,
        application_id=consented_session.application_id,
        tenant_id=consented_session.tenant_id,
        mode=consented_session.mode,
        locale=consented_session.locale,
        capabilities=consented_session.capabilities,
        configuration=consented_session.configuration,
        created_at_ms=consented_session.created_at_ms,
    )

    with pytest.raises(ConsentViolation, match="FR-031"):
        without_consent.begin_capture(wall_ms=1_000)


def test_capture_after_withdrawal_is_refused(
    consented_session: AnalysisSession, consent: ConsentReceipt
) -> None:
    session = consented_session.with_consent(consent.withdraw(at_ms=2_000))

    with pytest.raises(ConsentViolation, match="withdrawn"):
        session.begin_capture(wall_ms=3_000)


def test_resuming_re_checks_consent(consented_session: AnalysisSession) -> None:
    """A withdrawal mid-session must stop the next resume, not only the start."""
    session = consented_session.begin_capture(wall_ms=0).pause_capture(wall_ms=5_000)
    assert session.consent is not None
    session = session.with_consent(session.consent.withdraw(at_ms=6_000))

    with pytest.raises(ConsentViolation):
        session.resume_capture(wall_ms=7_000)


def test_a_receipt_for_another_session_is_refused(
    consented_session: AnalysisSession,
) -> None:
    foreign = ConsentReceipt(
        session_id=SessionId("some-other-session"),
        policy_version=POLICY,
        retention=RetentionPolicy.ephemeral(POLICY),
        granted_at_ms=0,
    )

    with pytest.raises(ConsentViolation, match="cannot govern"):
        consented_session.with_consent(foreign)


def test_deleting_evidence_withdraws_consent(
    consented_session: AnalysisSession,
) -> None:
    session = (
        consented_session.begin_capture(wall_ms=0)
        .request_completion(wall_ms=10_000)
        .mark_completed(wall_ms=11_000)
        .mark_evidence_deleted(wall_ms=12_000)
    )

    assert session.state is SessionState.EVIDENCE_DELETED
    assert session.consent is not None
    assert session.consent.is_active is False


def test_media_is_only_accepted_while_capturing(
    consented_session: AnalysisSession,
) -> None:
    assert consented_session.accepts_media is False

    capturing = consented_session.begin_capture(wall_ms=0)
    assert capturing.accepts_media is True

    assert capturing.pause_capture(wall_ms=1_000).accepts_media is False


# ---------------------------------------------------------------------------
# Retention policy (NFR-011)
# ---------------------------------------------------------------------------


def test_the_default_policy_is_ephemeral() -> None:
    policy = RetentionPolicy.ephemeral(POLICY)

    assert policy.retain_raw_media is False
    assert policy.raw_media_ttl_seconds == 0
    assert policy.retain_derived_aggregates is True


def test_retaining_raw_media_requires_a_positive_ttl() -> None:
    with pytest.raises(ConsentViolation, match="NFR-011"):
        RetentionPolicy(policy_version=POLICY, retain_raw_media=True, raw_media_ttl_seconds=0)


def test_a_ttl_without_retention_is_refused() -> None:
    with pytest.raises(ConsentViolation, match="cannot declare a TTL"):
        RetentionPolicy(policy_version=POLICY, retain_raw_media=False, raw_media_ttl_seconds=3_600)


def test_a_receipt_cannot_attest_to_a_policy_version_it_did_not_show() -> None:
    with pytest.raises(ConsentViolation, match="never saw"):
        ConsentReceipt(
            session_id=SessionId("s-1"),
            policy_version=SemanticVersion(1, 0, 0),
            retention=RetentionPolicy.ephemeral(SemanticVersion(2, 0, 0)),
            granted_at_ms=0,
        )


def test_withdrawal_is_idempotent() -> None:
    receipt = ConsentReceipt(
        session_id=SessionId("s-1"),
        policy_version=POLICY,
        retention=RetentionPolicy.ephemeral(POLICY),
        granted_at_ms=0,
    )

    once = receipt.withdraw(at_ms=100)
    twice = once.withdraw(at_ms=200)

    assert twice.withdrawn_at_ms == 100
