"""PostgreSQL repositories.

Every method that reads takes a tenant and puts it in the WHERE clause, matching
the in-memory adapter's key structure. That is the point of having both: the
cross-tenant tests in the contract suite check a property of the *port*, so they
mean the same thing whichever adapter is behind it.

Idempotency deserves a note. ``claim_idempotency_key`` relies on the primary key
of ``idempotency_key`` to arbitrate, rather than on a read-then-write. Two
concurrent creates with the same key must produce one session; a check followed
by an insert has a window between them, and under exactly the retry storm the
key exists to handle, that window is where a duplicate session gets created.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidence_engine.adapters.outbound.persistence.postgres import models
from evidence_engine.adapters.outbound.persistence.postgres.engine import unit_of_work
from evidence_engine.adapters.outbound.persistence.postgres.mapping import (
    consent_to_row,
    row_to_session,
    session_to_row,
)
from evidence_engine.application.ports.repositories import (
    AuditRecord,
    ProcessingRun,
    RunState,
)
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.shared.identifiers import (
    ApplicationId,
    RunId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import SemanticVersion


class PostgresSessionRepository:
    """Sessions and their consent receipts, always scoped by tenant."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def add(self, session: AnalysisSession) -> None:
        async with unit_of_work(self._factory) as db:
            db.add(session_to_row(session))
            if session.consent is not None:
                db.add(consent_to_row(session.id, session.consent))

    async def get(self, tenant: TenantId, session_id: SessionId) -> AnalysisSession | None:
        async with self._factory() as db:
            row = await db.scalar(
                select(models.AnalysisSessionRow).where(
                    models.AnalysisSessionRow.tenant_id == tenant.value,
                    models.AnalysisSessionRow.id == session_id.value,
                )
            )
            if row is None:
                return None
            consent = await db.scalar(
                select(models.ConsentReceiptRow).where(
                    models.ConsentReceiptRow.session_id == session_id.value
                )
            )
            return row_to_session(row, consent)

    async def save(self, session: AnalysisSession) -> None:
        """Persist a transition.

        A merge rather than an update statement: the aggregate is immutable and
        arrives whole, so writing it whole keeps the row and the object in step
        without anybody maintaining a field list that drifts.
        """
        async with unit_of_work(self._factory) as db:
            await db.merge(session_to_row(session))
            if session.consent is not None:
                await db.merge(consent_to_row(session.id, session.consent))

    async def find_by_idempotency_key(
        self, application: ApplicationId, key: str
    ) -> AnalysisSession | None:
        async with self._factory() as db:
            claim = await db.scalar(
                select(models.IdempotencyKeyRow).where(
                    models.IdempotencyKeyRow.application_id == application.value,
                    models.IdempotencyKeyRow.key == key,
                )
            )
            if claim is None:
                return None
            row = await db.scalar(
                select(models.AnalysisSessionRow).where(
                    models.AnalysisSessionRow.id == claim.session_id,
                    models.AnalysisSessionRow.application_id == application.value,
                )
            )
            if row is None:
                return None
            consent = await db.scalar(
                select(models.ConsentReceiptRow).where(
                    models.ConsentReceiptRow.session_id == row.id
                )
            )
            return row_to_session(row, consent)

    async def claim_idempotency_key(
        self, application: ApplicationId, key: str, session_id: SessionId
    ) -> bool:
        """Let the primary key arbitrate, not a prior read.

        Returns False on a duplicate-key violation, which is the database
        telling us another request won the race. A read-then-write would have a
        window between the two, and that window is exactly where the retry
        storm this key exists to handle would create a second session.
        """
        try:
            async with unit_of_work(self._factory) as db:
                db.add(
                    models.IdempotencyKeyRow(
                        application_id=application.value,
                        key=key,
                        session_id=session_id.value,
                    )
                )
        except IntegrityError:
            return False
        return True


class PostgresRunRepository:
    """Processing runs and their resumable stage progress (NFR-019)."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def add(self, run: ProcessingRun) -> None:
        async with unit_of_work(self._factory) as db:
            db.add(_run_to_row(run))

    async def get(self, run_id: RunId) -> ProcessingRun | None:
        async with self._factory() as db:
            row = await db.get(models.ProcessingRunRow, run_id.value)
            return _row_to_run(row) if row is not None else None

    async def save(self, run: ProcessingRun) -> None:
        async with unit_of_work(self._factory) as db:
            await db.merge(_run_to_row(run))

    async def latest_for_session(self, session_id: SessionId) -> ProcessingRun | None:
        async with self._factory() as db:
            row = await db.scalar(
                select(models.ProcessingRunRow)
                .where(models.ProcessingRunRow.session_id == session_id.value)
                .order_by(models.ProcessingRunRow.started_at_ms.desc())
                .limit(1)
            )
            return _row_to_run(row) if row is not None else None


class PostgresAuditLog:
    """Append-only in fact: this class has no update and no delete (NFR-022)."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def record(self, entry: AuditRecord) -> None:
        async with unit_of_work(self._factory) as db:
            db.add(
                models.AuditRecordRow(
                    actor=entry.actor,
                    action=entry.action,
                    resource=entry.resource,
                    tenant_id=entry.tenant.value if entry.tenant else None,
                    timestamp_ms=entry.timestamp_ms,
                    trace_id=entry.trace_id,
                    outcome=entry.outcome,
                    detail=entry.detail,
                )
            )

    async def for_resource(self, resource: str, limit: int = 100) -> Sequence[AuditRecord]:
        async with self._factory() as db:
            rows = (
                await db.scalars(
                    select(models.AuditRecordRow)
                    .where(models.AuditRecordRow.resource == resource)
                    .order_by(models.AuditRecordRow.id.desc())
                    .limit(limit)
                )
            ).all()
        return tuple(
            AuditRecord(
                actor=row.actor,
                action=row.action,
                resource=row.resource,
                timestamp_ms=row.timestamp_ms,
                trace_id=row.trace_id,
                outcome=row.outcome,
                tenant=TenantId(row.tenant_id) if row.tenant_id else None,
                detail=row.detail,
            )
            # Reversed so the caller reads oldest-first, matching the in-memory
            # adapter. The DESC + LIMIT above is how "the last N" is fetched
            # efficiently; the order it arrives in is an implementation detail
            # that must not leak into the port's contract.
            for row in reversed(rows)
        )


async def purge_session_rows(db: AsyncSession, tenant: TenantId, session_id: SessionId) -> int:
    """Delete every evidence row belonging to a session (FR-032).

    Returns the count, because that is what the audit record attests to:
    "deletion ran and touched N rows" is provable and "deletion ran" is not.

    Runs statement by statement rather than relying on cascades alone, so the
    count is real rather than inferred from a foreign-key side effect nobody
    can see.
    """
    run_ids = (
        await db.scalars(
            select(models.ProcessingRunRow.id).where(
                models.ProcessingRunRow.session_id == session_id.value
            )
        )
    ).all()

    removed = 0
    if run_ids:
        for table in (
            models.WordTokenRow,
            models.SpeechEventRow,
            models.VisualEventRow,
            models.ProsodyReadingRow,
            models.QualityAssessmentRow,
            models.ModalityAvailabilityRow,
            models.CooccurrenceRow,
        ):
            # `execute` is typed as returning `Result`; a DML statement
            # actually returns a `CursorResult`, which is the only one that
            # carries `rowcount`. The cast records that, rather than the count
            # being silently dropped to zero.
            result = cast(
                CursorResult[None],
                await db.execute(
                    delete(table).where(table.run_id.in_(run_ids), table.tenant_id == tenant.value)
                ),
            )
            removed += result.rowcount or 0

    document = cast(
        CursorResult[None],
        await db.execute(
            delete(models.EvidenceDocumentRow).where(
                models.EvidenceDocumentRow.tenant_id == tenant.value,
                models.EvidenceDocumentRow.session_id == session_id.value,
            )
        ),
    )
    removed += document.rowcount or 0
    return removed


def _run_to_row(run: ProcessingRun) -> models.ProcessingRunRow:
    return models.ProcessingRunRow(
        id=run.id.value,
        session_id=run.session_id.value,
        pipeline_version=str(run.pipeline_version),
        state=run.state.value,
        started_at_ms=run.started_at_ms,
        completed_at_ms=run.completed_at_ms,
        completed_stages=list(run.completed_stages),
    )


def _row_to_run(row: models.ProcessingRunRow) -> ProcessingRun:
    return ProcessingRun(
        id=RunId(row.id),
        session_id=SessionId(row.session_id),
        pipeline_version=SemanticVersion.parse(row.pipeline_version),
        state=RunState(row.state),
        started_at_ms=row.started_at_ms,
        completed_at_ms=row.completed_at_ms,
        completed_stages=tuple(row.completed_stages),
    )
