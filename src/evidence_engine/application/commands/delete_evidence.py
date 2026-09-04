"""Delete session evidence (FR-032, QA-04, US-005, US-014).

The order of operations is the whole design, and it runs from least to most
recoverable.

1. Revoke outstanding signed URLs. QA-04's response includes invalidating them,
   and a URL minted a minute before the request would otherwise keep working
   against a cache or a replica after the bytes are gone.
2. Delete raw media.
3. Delete protected derived evidence.
4. Mark the session deleted, which also withdraws consent.
5. Write the audit record.

The audit record goes last and carries counts. US-010 wants records that prove
execution *without retaining the deleted content*, and "removed 1 media object
and 412 evidence rows" is provable in a way that "deletion requested" is not.
Writing it first would attest to work that had not happened yet.

Repeated requests are idempotent (US-014). A session already in
``EVIDENCE_DELETED`` returns the same success rather than an error: a client
retrying after a timeout must not be told its withdrawal failed.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.application.errors import NotAuthorized, SessionNotFound
from evidence_engine.application.ports.clock import Clock
from evidence_engine.application.ports.platform import AuthenticatedCaller, Scope
from evidence_engine.application.ports.repositories import (
    AuditLog,
    AuditRecord,
    EvidenceRepository,
    SessionRepository,
)
from evidence_engine.application.ports.storage import MediaStore
from evidence_engine.application.ports.streaming import StreamState
from evidence_engine.domain.sessions.state import SessionState
from evidence_engine.domain.shared.identifiers import SessionId


@dataclass(frozen=True, slots=True)
class DeletionReceipt:
    """What was actually removed. The substance of the audit record."""

    session_id: SessionId
    media_objects_deleted: int
    evidence_records_deleted: int
    was_already_deleted: bool = False


class DeleteEvidence:
    """Use case: honour a withdrawal request."""

    def __init__(
        self,
        sessions: SessionRepository,
        evidence: EvidenceRepository,
        media: MediaStore,
        stream_state: StreamState,
        audit: AuditLog,
        clock: Clock,
    ) -> None:
        self._sessions = sessions
        self._evidence = evidence
        self._media = media
        self._stream_state = stream_state
        self._audit = audit
        self._clock = clock

    async def execute(self, caller: AuthenticatedCaller, session_id: SessionId) -> DeletionReceipt:
        # A dedicated scope, not a general write scope. A capture credential
        # that leaked into a client bundle must not also be able to erase a
        # study's data.
        if not caller.allows(Scope.EVIDENCE_DELETE):
            raise NotAuthorized(f"deleting evidence requires {Scope.EVIDENCE_DELETE.value}")

        session = await self._sessions.get(caller.tenant, session_id)
        if session is None:
            raise SessionNotFound(f"session {session_id} not found")

        if session.state is SessionState.EVIDENCE_DELETED:
            return DeletionReceipt(
                session_id=session_id,
                media_objects_deleted=0,
                evidence_records_deleted=0,
                was_already_deleted=True,
            )

        await self._media.revoke_signatures(caller.tenant, session_id)
        media_deleted = await self._media.delete_for_session(caller.tenant, session_id)
        evidence_deleted = await self._evidence.delete_for_session(caller.tenant, session_id)

        # Any live stream state for this session is ephemeral and now
        # meaningless. Clearing it also drops the processing lease, so a
        # handler still holding the socket cannot write more evidence into a
        # session that has just been erased.
        await self._stream_state.clear(session_id)

        now_ms = self._clock.epoch_ms()
        deleted = session.mark_evidence_deleted(now_ms)
        await self._sessions.save(deleted)

        await self._audit.record(
            AuditRecord(
                actor=caller.application.value,
                action="evidence.delete",
                resource=session_id.value,
                timestamp_ms=now_ms,
                trace_id=caller.trace_id,
                outcome="deleted",
                tenant=caller.tenant,
                detail=(f"media_objects={media_deleted} evidence_records={evidence_deleted}"),
            )
        )

        return DeletionReceipt(
            session_id=session_id,
            media_objects_deleted=media_deleted,
            evidence_records_deleted=evidence_deleted,
        )

    async def verify(self, caller: AuthenticatedCaller, session_id: SessionId) -> bool:
        """QA-04's measure: is there any recoverable object reference left?

        A separate call rather than part of ``execute`` because the measure is
        about the state after deletion, and asking the same code path that just
        deleted would only confirm it believes itself.
        """
        if not caller.allows(Scope.EVIDENCE_DELETE):
            raise NotAuthorized(f"verifying deletion requires {Scope.EVIDENCE_DELETE.value}")
        remaining = await self._media.count_for_session(caller.tenant, session_id)
        return remaining == 0
