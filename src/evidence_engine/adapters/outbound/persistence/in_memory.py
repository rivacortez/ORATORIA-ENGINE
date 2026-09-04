"""In-memory persistence.

This is not a test double bolted on afterwards. Phase 2's exit criterion is that
"a synthetic session can be streamed, completed, queried and deleted without
model inference", and that has to be demonstrable without a database, a broker
or a GPU - otherwise the contract suite only runs where the whole stack does,
and it stops being the thing CI checks on every commit.

Tenant scoping is implemented here exactly as the PostgreSQL adapter must
implement it: the tenant is part of the key, not a filter applied afterwards.
NFR-013's cross-tenant tests then check a real property of both adapters rather
than a property of whichever one the test happened to run against.
"""

from __future__ import annotations

from collections.abc import Sequence

from evidence_engine.application.ports.repositories import (
    AuditRecord,
    EvidenceBundle,
    ProcessingRun,
)
from evidence_engine.domain.evidence.document import EvidenceDocument
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.shared.identifiers import (
    ApplicationId,
    RunId,
    SessionId,
    TenantId,
)


class InMemorySessionRepository:
    """Sessions keyed by ``(tenant, session)``."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], AnalysisSession] = {}
        self._idempotency: dict[tuple[str, str], SessionId] = {}

    async def add(self, session: AnalysisSession) -> None:
        self._sessions[(session.tenant_id.value, session.id.value)] = session

    async def get(self, tenant: TenantId, session_id: SessionId) -> AnalysisSession | None:
        return self._sessions.get((tenant.value, session_id.value))

    async def save(self, session: AnalysisSession) -> None:
        self._sessions[(session.tenant_id.value, session.id.value)] = session

    async def find_by_idempotency_key(
        self, application: ApplicationId, key: str
    ) -> AnalysisSession | None:
        session_id = self._idempotency.get((application.value, key))
        if session_id is None:
            return None
        for session in self._sessions.values():
            if session.id == session_id and session.application_id == application:
                return session
        return None

    async def claim_idempotency_key(
        self, application: ApplicationId, key: str, session_id: SessionId
    ) -> bool:
        composite = (application.value, key)
        if composite in self._idempotency:
            return False
        self._idempotency[composite] = session_id
        return True


class InMemoryRunRepository:
    """Processing runs, with their resumable stage progress (NFR-019)."""

    def __init__(self) -> None:
        self._runs: dict[str, ProcessingRun] = {}

    async def add(self, run: ProcessingRun) -> None:
        self._runs[run.id.value] = run

    async def get(self, run_id: RunId) -> ProcessingRun | None:
        return self._runs.get(run_id.value)

    async def save(self, run: ProcessingRun) -> None:
        self._runs[run.id.value] = run

    async def latest_for_session(self, session_id: SessionId) -> ProcessingRun | None:
        candidates = [r for r in self._runs.values() if r.session_id == session_id]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.started_at_ms)


class InMemoryEvidenceRepository:
    """Derived evidence, kept separate from raw media exactly as §8 requires."""

    def __init__(self) -> None:
        self._bundles: dict[tuple[str, str], EvidenceBundle] = {}
        self._documents: dict[tuple[str, str], EvidenceDocument] = {}

    async def store(self, bundle: EvidenceBundle) -> None:
        self._bundles[(bundle.tenant.value, bundle.run_id.value)] = bundle

    async def load(self, tenant: TenantId, run_id: RunId) -> EvidenceBundle | None:
        return self._bundles.get((tenant.value, run_id.value))

    async def load_document(
        self, tenant: TenantId, session_id: SessionId
    ) -> EvidenceDocument | None:
        return self._documents.get((tenant.value, session_id.value))

    async def store_document(self, document: EvidenceDocument, tenant: TenantId) -> None:
        self._documents[(tenant.value, document.session_id.value)] = document

    async def delete_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        removed = 0
        for key in [
            key
            for key, bundle in self._bundles.items()
            if key[0] == tenant.value and bundle.session_id == session_id
        ]:
            del self._bundles[key]
            removed += 1
        if (tenant.value, session_id.value) in self._documents:
            del self._documents[(tenant.value, session_id.value)]
            removed += 1
        return removed


class InMemoryAuditLog:
    """Append-only in fact, not only by convention (NFR-022).

    The list is private and there is no update or delete. A test that wants to
    assert on the trail reads it through ``for_resource``, the same way the
    deletion-verification job does.
    """

    def __init__(self) -> None:
        self._entries: list[AuditRecord] = []

    async def record(self, entry: AuditRecord) -> None:
        self._entries.append(entry)

    async def for_resource(self, resource: str, limit: int = 100) -> Sequence[AuditRecord]:
        matching = [e for e in self._entries if e.resource == resource]
        return tuple(matching[-limit:])

    async def all_entries(self) -> Sequence[AuditRecord]:
        return tuple(self._entries)
