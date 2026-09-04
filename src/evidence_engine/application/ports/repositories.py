"""Persistence ports.

Every read is scoped by tenant. NFR-013 requires data access to always be
scoped by tenant or application and cross-tenant tests to pass, and the only
way that survives contact with a growing codebase is to make the unscoped call
impossible to write: there is no ``get(session_id)`` here, only
``get(tenant_id, session_id)``. A developer who forgets the scope gets a type
error rather than another tenant's evidence.

``ProcessingRun`` is separate from the session on purpose (§8). A session can be
processed more than once - a promoted model, a resumed batch job, a replayed
evaluation item - and hanging derived records off the run keeps two results
comparable instead of letting the second overwrite the first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from evidence_engine.domain.evidence.cooccurrence import MultimodalCooccurrence
from evidence_engine.domain.evidence.document import EvidenceDocument
from evidence_engine.domain.quality.assessment import QualityReport
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.shared.identifiers import (
    ApplicationId,
    RunId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading
from evidence_engine.domain.transcript.transcript import Transcript
from evidence_engine.domain.visual_events.events import VisualEvent


class RunState(StrEnum):
    """Where one processing run stands."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProcessingRun:
    """§8 ``ProcessingRun``: one pass of the pipeline over a session's media."""

    id: RunId
    session_id: SessionId
    pipeline_version: SemanticVersion
    state: RunState
    started_at_ms: int
    completed_at_ms: int | None = None
    #: Stages already finished, for NFR-019: an interrupted batch job resumes
    #: from completed idempotent stages instead of restarting the pipeline.
    completed_stages: tuple[str, ...] = field(default_factory=tuple)


class SessionRepository(Protocol):
    """Sessions, always addressed within a tenant."""

    async def add(self, session: AnalysisSession) -> None: ...

    async def get(self, tenant: TenantId, session_id: SessionId) -> AnalysisSession | None:
        """The session, or ``None`` when it does not exist *for this tenant*.

        A session belonging to another tenant returns ``None`` rather than
        raising a distinguishable error: a 404 and a 403 tell an attacker
        different things, and NFR-013 makes tenants invisible to each other.
        """
        ...

    async def save(self, session: AnalysisSession) -> None:
        """Persist a transition. The aggregate is immutable, so this replaces."""
        ...

    async def find_by_idempotency_key(
        self, application: ApplicationId, key: str
    ) -> AnalysisSession | None:
        """US-011: session creation is idempotent under an idempotency key."""
        ...

    async def claim_idempotency_key(
        self, application: ApplicationId, key: str, session_id: SessionId
    ) -> bool:
        """Reserve a key, returning False when another request already holds it.

        Separate from ``add`` so the reservation and the insert can be one
        transaction in the adapter. Two concurrent creates with the same key
        must produce one session, not two with a duplicate-key error surfacing
        to whichever lost the race.
        """
        ...


class RunRepository(Protocol):
    """Processing runs and their resumable stage progress."""

    async def add(self, run: ProcessingRun) -> None: ...

    async def get(self, run_id: RunId) -> ProcessingRun | None: ...

    async def save(self, run: ProcessingRun) -> None: ...

    async def latest_for_session(self, session_id: SessionId) -> ProcessingRun | None: ...


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Everything one run derived, ready to be stored or assembled.

    Kept apart from ``EvidenceDocument``: the document is the published shape
    with its provenance manifest, this is the storage shape. Merging them would
    let a schema change to the public contract force a migration of historical
    rows, which NFR-017's compatibility promise exists to avoid.
    """

    run_id: RunId
    session_id: SessionId
    #: Part of the storage key, not a filter applied afterwards. NFR-013's
    #: scoping rule has to survive the write path as well as the read path.
    tenant: TenantId
    transcript: Transcript
    quality: QualityReport
    speech_events: tuple[SpeechEvent, ...] = field(default_factory=tuple)
    visual_events: tuple[VisualEvent, ...] = field(default_factory=tuple)
    prosody: tuple[ProsodyReading, ...] = field(default_factory=tuple)
    cooccurrences: tuple[MultimodalCooccurrence, ...] = field(default_factory=tuple)


class EvidenceRepository(Protocol):
    """Derived evidence, separated from the raw media it came from (§8)."""

    async def store(self, bundle: EvidenceBundle) -> None: ...

    async def load(self, tenant: TenantId, run_id: RunId) -> EvidenceBundle | None: ...

    async def store_document(self, document: EvidenceDocument, tenant: TenantId) -> None:
        """Persist the published shape of a completed run.

        Separate from ``store`` because the two shapes serve different masters:
        the bundle is storage, the document is the public contract with its
        provenance manifest. Merging them would let a change to the published
        schema force a migration of historical rows, which NFR-017's
        compatibility promise exists to avoid.

        Only the completion path calls this. FR-029's guarantee about what a
        document contains holds because exactly one code path assembles them.
        """
        ...

    async def load_document(
        self, tenant: TenantId, session_id: SessionId
    ) -> EvidenceDocument | None:
        """The published result for a session's latest completed run."""
        ...

    async def delete_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        """Remove protected evidence, returning how many records were deleted.

        The count is what the audit record in FR-032 and US-010 attests to:
        "deletion ran and touched N rows" is provable, "deletion ran" is not.
        Non-identifying aggregates survive when the consent policy allows it
        (§8), so this is not always a full erasure and must not pretend to be.
        """
        ...


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """§8 ``AuditRecord``, append-only (NFR-022)."""

    actor: str
    action: str
    resource: str
    timestamp_ms: int
    trace_id: str
    outcome: str
    tenant: TenantId | None = None
    detail: str = ""


class AuditLog(Protocol):
    """The append-only trail NFR-022 requires.

    No update, no delete. Model promotion, threshold changes, key management,
    export and deletion all land here, and an entry that can be edited is not
    evidence of anything.
    """

    async def record(self, entry: AuditRecord) -> None: ...

    async def for_resource(self, resource: str, limit: int = 100) -> Sequence[AuditRecord]:
        """Read back the trail. Used by deletion verification, not by hot paths."""
        ...
