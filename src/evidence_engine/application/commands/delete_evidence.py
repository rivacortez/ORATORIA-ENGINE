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

Verification is the second half of the module and reads every store the first
half writes to. It used to count media objects and nothing else, so a session
whose evidence rows survived step 3 verified as clean - the measure passed on
exactly the failure it exists to catch. The count is now one field of a report
rather than the whole answer, for the reasons on ``DeletionVerification``.

Two things the report cannot see, named here because a reader will otherwise
assume they are covered.

*Signed-URL revocation.* ``MediaStore`` has no read-only view of outstanding
signatures - ``revoke_signatures`` returns nothing and there is no counterpart
to ask afterwards. ``media_objects_remaining`` is the closest honest proxy: the
S3 adapter revokes by deleting the objects a signature points at, so a zero
count means the links open nothing there, and the in-memory adapter's
signatures address objects that are equally gone. A direct check needs a method
on the port.

*Evidence bundles.* ``EvidenceRepository`` offers ``load_document`` for a
session and ``load`` for a run, and this use case holds no run repository, so
only the published document can be read back. Both adapters purge bundle rows
and the document in one call - PostgreSQL in one transaction - so rows
surviving without their document is not reachable today. It becomes reachable
the moment an adapter separates those writes, and the field is named
``evidence_document_present`` rather than ``evidence_present`` so the summary
never claims more than it read.
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


@dataclass(frozen=True, slots=True)
class DeletionVerification:
    """What each store still holds, one field per store ``execute`` writes to.

    Not a boolean, for two reasons.

    *A boolean cannot name what survived.* QA-04's measure fails in four
    independent ways and each has a different remedy: re-run the deletion,
    repair a session left in its previous state, or find out why the audit
    write did not land. A caller handed ``False`` has to go to the logs to
    learn which, and the logs are the one place the deleted content must not
    be (NFR-018).

    *The fields do not share a polarity.* Media, evidence and stream state must
    be **absent**; the session transition and the audit record must be
    **present**. Folding both directions into one value hides the inversion,
    and the next person to add a term has even odds of getting the sign wrong -
    which is how the original measure came to check only the half that happened
    to be an absence.

    ``is_clean`` is exactly the conjunction of the published fields and adds no
    term of its own, so a reader can recompute it and the summary cannot attest
    to more than was actually read.
    """

    session_id: SessionId
    media_objects_remaining: int
    evidence_document_present: bool
    stream_state_present: bool
    session_marked_deleted: bool
    audit_record_present: bool

    @property
    def is_clean(self) -> bool:
        """Every store this call can read is in the state deletion promised."""
        return (
            self.media_objects_remaining == 0
            and not self.evidence_document_present
            and not self.stream_state_present
            and self.session_marked_deleted
            and self.audit_record_present
        )


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

    async def verify(
        self, caller: AuthenticatedCaller, session_id: SessionId
    ) -> DeletionVerification:
        """QA-04's measure: what, if anything, survived the deletion.

        A separate call rather than part of ``execute`` because the measure is
        about the state deletion left behind, and asking the same code path
        that just deleted would only confirm it believes itself.

        Read-only throughout. Nothing here removes anything, including on the
        paths that find residue: a verification that tidied up as it went would
        report on its own actions and could never fail twice in a row, which is
        precisely the evidence an operator needs after the first failure.
        """
        if not caller.allows(Scope.EVIDENCE_DELETE):
            raise NotAuthorized(f"verifying deletion requires {Scope.EVIDENCE_DELETE.value}")

        # Refused rather than reported. Every "still present" field of a session
        # that does not exist is empty and every count is zero, so the report
        # would render as a flawless deletion of something that was never here -
        # indistinguishable, once copied into a compliance record, from a real
        # one. A session belonging to another tenant arrives here too, which
        # keeps NFR-013's rule that another tenant's session is invisible rather
        # than forbidden.
        session = await self._sessions.get(caller.tenant, session_id)
        if session is None:
            raise SessionNotFound(f"session {session_id} not found")

        return DeletionVerification(
            session_id=session_id,
            media_objects_remaining=await self._media.count_for_session(caller.tenant, session_id),
            evidence_document_present=(
                await self._evidence.load_document(caller.tenant, session_id) is not None
            ),
            stream_state_present=await self._stream_state.load(session_id) is not None,
            session_marked_deleted=session.state is SessionState.EVIDENCE_DELETED,
            audit_record_present=await self._deletion_was_audited(caller, session_id),
        )

    async def _deletion_was_audited(
        self, caller: AuthenticatedCaller, session_id: SessionId
    ) -> bool:
        """Is there a record attesting that deletion ran for this session?

        The one field that must be *present* rather than absent. US-010 makes
        the audit record the proof that deletion executed, so a session whose
        bytes are gone and whose record never landed is not a completed
        deletion - it is an unexplained disappearance, which is worse to hold
        than either.

        Filtered by tenant as well as by resource. ``for_resource`` takes a
        resource string and nothing else, and session ids are unique within a
        tenant rather than globally, so an unfiltered read would let one
        tenant's deletion record vouch for another tenant's session of the same
        name.
        """
        entries = await self._audit.for_resource(session_id.value)
        return any(
            entry.action == "evidence.delete" and entry.tenant == caller.tenant for entry in entries
        )
