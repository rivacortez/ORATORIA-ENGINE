"""Consent and retention, recorded before protected media is accepted.

FR-031 puts the ordering plainly: the applicable consent and retention policy
is recorded *before* the service accepts protected media. That ordering is the
whole design. A consent receipt written after the upload describes a decision
the participant had already lost the chance to make, and the risk table's
"consent withdrawal leaves copies" begins exactly there.

``policy_version`` is a version rather than a copy of the text because US-008's
immutability rule applies here too: the participant agreed to a specific
published wording, and a later edit to that wording must not retroactively
become what they agreed to.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from evidence_engine.domain.shared.errors import DomainError
from evidence_engine.domain.shared.identifiers import SessionId
from evidence_engine.domain.shared.provenance import SemanticVersion


class ConsentViolation(DomainError):
    """Protected media was about to be handled without valid consent."""


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How long raw media may be kept, and whether it may be kept at all.

    NFR-011 makes ephemeral processing the default: raw media is discarded once
    the run that needs it finishes, and any longer retention requires an
    explicit versioned policy. ``retain_raw_media=False`` is therefore the
    honest default rather than a convenience.

    Derived, non-identifying aggregates survive deletion of the raw media when
    the policy allows it (§8). That is not a loophole - it is what makes NFR-015
    reproducibility possible after the recording is gone.
    """

    policy_version: SemanticVersion
    retain_raw_media: bool = False
    raw_media_ttl_seconds: int = 0
    retain_derived_aggregates: bool = True

    def __post_init__(self) -> None:
        if self.raw_media_ttl_seconds < 0:
            raise ConsentViolation("retention TTL cannot be negative")
        if self.retain_raw_media and self.raw_media_ttl_seconds == 0:
            raise ConsentViolation(
                "a policy that retains raw media must state a positive TTL; "
                "NFR-011 forbids indefinite retention by default"
            )
        if not self.retain_raw_media and self.raw_media_ttl_seconds != 0:
            raise ConsentViolation("a policy that does not retain raw media cannot declare a TTL")

    @classmethod
    def ephemeral(cls, policy_version: SemanticVersion) -> RetentionPolicy:
        """The default: process, then discard (NFR-011, ADR-008)."""
        return cls(policy_version=policy_version)


@dataclass(frozen=True, slots=True)
class ConsentReceipt:
    """Proof that a participant agreed, and to which published policy.

    Withdrawal is recorded on the same object rather than by deleting it. §14.3
    requires the deletion flow to be demonstrable end to end, and US-010 wants
    audit records that prove execution without retaining the deleted content -
    both of which need the receipt to outlive the evidence it governed.
    """

    session_id: SessionId
    policy_version: SemanticVersion
    retention: RetentionPolicy
    granted_at_ms: int
    withdrawn_at_ms: int | None = None

    def __post_init__(self) -> None:
        if self.granted_at_ms < 0:
            raise ConsentViolation("consent timestamp cannot be negative")
        if self.withdrawn_at_ms is not None and self.withdrawn_at_ms < self.granted_at_ms:
            raise ConsentViolation("consent cannot be withdrawn before it was granted")
        if self.retention.policy_version != self.policy_version:
            raise ConsentViolation(
                "the retention policy version must match the consented policy version; "
                "otherwise the receipt attests to rules the participant never saw"
            )

    @property
    def is_active(self) -> bool:
        return self.withdrawn_at_ms is None

    def withdraw(self, at_ms: int) -> ConsentReceipt:
        """Record a withdrawal.

        US-005: withdrawal requires no justification, so none is accepted here.
        A field for it would invite a UI to demand one.
        """
        if not self.is_active:
            return self
        return replace(self, withdrawn_at_ms=at_ms)

    def require_active(self) -> None:
        """Gate protected-media handling on live consent (FR-031)."""
        if not self.is_active:
            raise ConsentViolation(
                f"consent for session {self.session_id} was withdrawn; "
                "protected media may not be processed or retained"
            )
