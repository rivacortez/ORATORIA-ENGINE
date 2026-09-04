"""The analysis session aggregate.

A session is one capture, from the moment an application asks for it to the
moment its evidence is deleted. It owns the lifecycle (§6.1), the clock
(FR-009) and the consent gate (FR-031), and it owns them together on purpose:
those three rules only hold jointly. Consent checked in an HTTP handler is
consent that a WebSocket path can skip; a clock advanced by a background worker
is a clock that keeps running through a pause.

What the session deliberately does *not* own is evidence. §5 gives the Session
Orchestrator one responsibility - state and lifecycle - and one prohibition:
it must not interpret results. Keeping events out of the aggregate is what
makes that prohibition structural rather than a matter of discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from evidence_engine.domain.sessions.capabilities import NegotiatedCapabilities
from evidence_engine.domain.sessions.clock import SessionClock
from evidence_engine.domain.sessions.consent import ConsentReceipt, ConsentViolation
from evidence_engine.domain.sessions.state import (
    SessionMode,
    SessionState,
    accepts_media,
    require_transition,
)
from evidence_engine.domain.shared.identifiers import (
    ApplicationId,
    ConfigurationSnapshotId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.timeline import MonotonicTime


@dataclass(frozen=True, slots=True)
class AnalysisSession:
    """§8 ``AnalysisSession``, with its lifecycle rules attached.

    Immutable. Every transition returns a new instance, so a caller holding an
    older reference cannot accidentally act on stale state - it has to ask for
    the current one, which is where the persistence boundary already is.
    """

    id: SessionId
    application_id: ApplicationId
    tenant_id: TenantId
    mode: SessionMode
    locale: str
    capabilities: NegotiatedCapabilities
    configuration: ConfigurationSnapshotId
    created_at_ms: int
    state: SessionState = SessionState.CREATED
    clock: SessionClock = field(default_factory=SessionClock)
    consent: ConsentReceipt | None = None
    completed_at_ms: int | None = None

    # -- consent ----------------------------------------------------------

    def with_consent(self, receipt: ConsentReceipt) -> AnalysisSession:
        """Attach the consent receipt that governs this capture."""
        if receipt.session_id != self.id:
            raise ConsentViolation(
                f"consent receipt for {receipt.session_id} cannot govern session {self.id}"
            )
        return replace(self, consent=receipt)

    def require_consent_for_media(self) -> ConsentReceipt:
        """The gate FR-031 puts in front of every byte of protected media."""
        if self.consent is None:
            raise ConsentViolation(
                f"session {self.id} has no consent receipt; "
                "FR-031 requires it to be recorded before protected media is accepted"
            )
        self.consent.require_active()
        return self.consent

    # -- lifecycle --------------------------------------------------------

    def begin_capture(self, wall_ms: int) -> AnalysisSession:
        """Move to capturing and start the clock."""
        self.require_consent_for_media()
        state = require_transition(self.state, SessionState.CAPTURING)
        return replace(self, state=state, clock=self.clock.start(wall_ms))

    def pause_capture(self, wall_ms: int) -> AnalysisSession:
        """Suspend capture. The accumulated duration is preserved (FR-009)."""
        state = require_transition(self.state, SessionState.PAUSED)
        return replace(self, state=state, clock=self.clock.pause(wall_ms))

    def resume_capture(self, wall_ms: int) -> AnalysisSession:
        """Resume from where the clock stopped, not from zero."""
        self.require_consent_for_media()
        state = require_transition(self.state, SessionState.CAPTURING)
        return replace(self, state=state, clock=self.clock.start(wall_ms))

    def request_completion(self, wall_ms: int) -> AnalysisSession:
        """Close capture and enter the final reconciliation pass (§6.1 step 10).

        The clock stops here rather than at ``mark_completed`` because the final
        pass is processing time, not captured time, and folding it in would
        stretch every rate-per-minute denominator by however long inference took.
        """
        state = require_transition(self.state, SessionState.COMPLETING)
        return replace(self, state=state, clock=self.clock.pause(wall_ms))

    def mark_completed(self, wall_ms: int) -> AnalysisSession:
        """The evidence document is ready for retrieval."""
        state = require_transition(self.state, SessionState.COMPLETED)
        return replace(self, state=state, completed_at_ms=wall_ms)

    def fail(self, wall_ms: int) -> AnalysisSession:
        """No valid result could be produced at all.

        Reserved for whole-session failure. §6.3 and QA-02 are explicit that a
        single modality dropping out is not this: speech survives a dead camera,
        and the affected visual indicators are marked unavailable instead.
        """
        state = require_transition(self.state, SessionState.FAILED)
        return replace(self, state=state, clock=self.clock.pause(wall_ms), completed_at_ms=wall_ms)

    def mark_evidence_deleted(self, wall_ms: int) -> AnalysisSession:
        """Terminal. Raw media and protected evidence are gone (FR-032)."""
        state = require_transition(self.state, SessionState.EVIDENCE_DELETED)
        withdrawn = self.consent.withdraw(wall_ms) if self.consent is not None else None
        return replace(self, state=state, consent=withdrawn)

    # -- queries ----------------------------------------------------------

    @property
    def accepts_media(self) -> bool:
        return accepts_media(self.state)

    def position(self, wall_ms: int) -> MonotonicTime:
        """Where the session clock stands right now."""
        return self.clock.now(wall_ms)

    def captured_ms(self, wall_ms: int) -> int:
        """Total captured duration, excluding pauses."""
        return self.clock.captured_ms(wall_ms)
