"""Create an analysis session (FR-005, FR-006, FR-031, US-011).

The ordering in ``execute`` is the requirement, not an implementation detail.

1. Authorize, so an unscoped caller never reaches the rest.
2. Check the idempotency key, so a retried create returns the original session
   instead of a second one (US-011).
3. Negotiate capabilities, so an unsupported format is refused *before* capture
   rather than after five minutes of recording (US-011's acceptance criterion).
4. Freeze the configuration snapshot, so the session's thresholds cannot move
   underneath it mid-presentation (US-008).
5. Record consent, so no protected media is ever accepted without it (FR-031).
6. Audit, so the whole thing is attributable (FR-004).

Consent is recorded here rather than at the first chunk because FR-031 says
"before accepting protected media", and the first chunk is already too late:
by then the browser has captured it.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.application.errors import NotAuthorized, QuotaExceeded
from evidence_engine.application.ports.clock import Clock
from evidence_engine.application.ports.platform import (
    AuthenticatedCaller,
    ConfigurationStore,
    QuotaGuard,
    Scope,
)
from evidence_engine.application.ports.repositories import (
    AuditLog,
    AuditRecord,
    SessionRepository,
)
from evidence_engine.application.ports.tokens import StreamTokenMinter
from evidence_engine.domain.sessions.capabilities import CapabilityRequest, negotiate
from evidence_engine.domain.sessions.consent import ConsentReceipt, RetentionPolicy
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.sessions.state import SessionMode
from evidence_engine.domain.shared.identifiers import SessionId
from evidence_engine.domain.shared.provenance import SemanticVersion


@dataclass(frozen=True, slots=True)
class CreateSessionCommand:
    """What a consuming application asks for."""

    mode: SessionMode
    capabilities: CapabilityRequest
    consent_policy_version: SemanticVersion
    retention: RetentionPolicy
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class CreatedSession:
    """What it gets back.

    ``stream_token`` is short-lived by design (§6.1 step 2). The API key is a
    long-lived server-side credential and must not be handed to a browser to
    open a socket with; this token authorizes exactly one session's stream and
    expires with it.
    """

    session: AnalysisSession
    stream_token: str
    stream_token_expires_at_ms: int
    was_existing: bool = False


#: How long a stream token stays valid. Long enough to survive a slow page load
#: and a preflight, short enough that a leaked token is not a standing grant.
STREAM_TOKEN_TTL_MS = 300_000


class CreateSession:
    """Use case: open a session and negotiate what may be sent into it."""

    def __init__(
        self,
        sessions: SessionRepository,
        configuration: ConfigurationStore,
        quota: QuotaGuard,
        audit: AuditLog,
        clock: Clock,
        token_minter: StreamTokenMinter,
    ) -> None:
        self._sessions = sessions
        self._configuration = configuration
        self._quota = quota
        self._audit = audit
        self._clock = clock
        self._tokens = token_minter

    async def execute(
        self, caller: AuthenticatedCaller, command: CreateSessionCommand
    ) -> CreatedSession:
        if not caller.allows(Scope.SESSIONS_WRITE):
            raise NotAuthorized(f"creating a session requires {Scope.SESSIONS_WRITE.value}")

        existing = await self._replay(caller, command)
        if existing is not None:
            return existing

        decision = await self._quota.check(caller.application, unit="sessions")
        if not decision.allowed:
            raise QuotaExceeded(
                decision.reason or "session quota exhausted", decision.retry_after_seconds
            )

        # Refuses outright on an unsupported codec, sample rate, locale, video
        # format or frame rate. Nothing is persisted before this line.
        capabilities = negotiate(command.capabilities)

        snapshot = await self._configuration.current(caller.tenant)
        now_ms = self._clock.epoch_ms()
        session_id = SessionId.generate()

        session = AnalysisSession(
            id=session_id,
            application_id=caller.application,
            tenant_id=caller.tenant,
            mode=command.mode,
            locale=capabilities.locale,
            capabilities=capabilities,
            configuration=snapshot.id,
            created_at_ms=now_ms,
        ).with_consent(
            ConsentReceipt(
                session_id=session_id,
                policy_version=command.consent_policy_version,
                retention=command.retention,
                granted_at_ms=now_ms,
            )
        )

        if command.idempotency_key is not None:
            claimed = await self._sessions.claim_idempotency_key(
                caller.application, command.idempotency_key, session_id
            )
            if not claimed:
                # Another request won the race between our lookup and here.
                # Return theirs rather than creating a duplicate: US-011
                # promises one session per key, and "usually one" is not that.
                replayed = await self._replay(caller, command)
                if replayed is not None:
                    return replayed

        await self._sessions.add(session)
        await self._configuration.freeze(snapshot, session_id)

        token, expires_at_ms = self._tokens.mint(
            session_id, expires_at_ms=now_ms + STREAM_TOKEN_TTL_MS
        )

        await self._audit.record(
            AuditRecord(
                actor=caller.application.value,
                action="session.create",
                resource=session_id.value,
                timestamp_ms=now_ms,
                trace_id=caller.trace_id,
                outcome="created",
                tenant=caller.tenant,
                detail=f"mode={command.mode.value} locale={capabilities.locale}",
            )
        )
        await self._quota.consume(caller.application, unit="sessions")

        return CreatedSession(
            session=session, stream_token=token, stream_token_expires_at_ms=expires_at_ms
        )

    async def _replay(
        self, caller: AuthenticatedCaller, command: CreateSessionCommand
    ) -> CreatedSession | None:
        """Return the session a repeated idempotency key already produced."""
        if command.idempotency_key is None:
            return None
        found = await self._sessions.find_by_idempotency_key(
            caller.application, command.idempotency_key
        )
        if found is None:
            return None
        token, expires_at_ms = self._tokens.mint(
            found.id, expires_at_ms=self._clock.epoch_ms() + STREAM_TOKEN_TTL_MS
        )
        return CreatedSession(
            session=found,
            stream_token=token,
            stream_token_expires_at_ms=expires_at_ms,
            was_existing=True,
        )
