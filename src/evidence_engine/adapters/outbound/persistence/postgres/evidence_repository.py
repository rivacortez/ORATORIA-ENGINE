"""The evidence repository.

Writes the bundle as normalized rows and the published document as a single
JSONB row, for the reasons in ``models.py``: the rows are what a research
export queries and what per-class metrics are computed from, the document is
what ``GET /result`` returns whole.

Both writes happen in one transaction. A committed document with missing rows,
or rows with no document, would each be a state nothing in the system knows how
to repair - and the second one is worse, because the API would keep answering
404 for evidence that exists.
"""

from __future__ import annotations

from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidence_engine.adapters.outbound.persistence.postgres import models
from evidence_engine.adapters.outbound.persistence.postgres.engine import unit_of_work
from evidence_engine.adapters.outbound.persistence.postgres.mapping import (
    prosody_to_row,
    row_to_prosody,
    row_to_speech_event,
    row_to_token,
    row_to_visual_event,
    speech_event_to_row,
    token_to_row,
    visual_event_to_row,
)
from evidence_engine.adapters.outbound.persistence.postgres.repositories import (
    purge_session_rows,
)
from evidence_engine.application.ports.repositories import EvidenceBundle
from evidence_engine.domain.evidence.cooccurrence import (
    FusionWindow,
    MultimodalCooccurrence,
)
from evidence_engine.domain.evidence.document import EvidenceDocument
from evidence_engine.domain.quality.assessment import (
    ModalityAvailability,
    QualityAssessment,
    QualityMetric,
    QualityReport,
)
from evidence_engine.domain.shared.confidence import CalibrationState, Confidence
from evidence_engine.domain.shared.identifiers import (
    ConfigurationSnapshotId,
    EventId,
    RunId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.measurement import (
    Measured,
    UnavailabilityReason,
    Unavailable,
)
from evidence_engine.domain.shared.provenance import Modality
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.transcript import transcript as transcript_module


class DocumentRenderer(Protocol):
    """The published-shape serializer, injected to keep contract C6 intact.

    The serializer lives in the inbound REST adapter, and an outbound
    adapter may not import it. Passing the function in keeps the published
    shape defined in exactly one place without inverting the dependency.
    """

    def __call__(self, document: EvidenceDocument) -> dict[str, Any]: ...


class PostgresEvidenceRepository:
    """Derived evidence, kept apart from the raw media it came from (§8)."""

    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        render_document: DocumentRenderer,
    ) -> None:
        self._factory = factory
        self._render = render_document

    async def store(self, bundle: EvidenceBundle) -> None:
        async with unit_of_work(self._factory) as db:
            self._write_bundle(db, bundle)

    async def store_document(self, document: EvidenceDocument, tenant: TenantId) -> None:
        payload = self._render(document)
        async with unit_of_work(self._factory) as db:
            await db.merge(
                models.EvidenceDocumentRow(
                    tenant_id=tenant.value,
                    session_id=document.session_id.value,
                    run_id=document.run_id.value,
                    schema_version=str(document.manifest.schema_version),
                    document=payload,
                )
            )

    async def load(self, tenant: TenantId, run_id: RunId) -> EvidenceBundle | None:
        async with self._factory() as db:
            run = await db.get(models.ProcessingRunRow, run_id.value)
            if run is None:
                return None
            return await self._read_bundle(db, tenant, run)

    async def load_document(
        self, tenant: TenantId, session_id: SessionId
    ) -> EvidenceDocument | None:
        """Return the stored document, or ``None``.

        The published JSON is returned by the query layer as-is; this method
        exists on the port for symmetry and is not what the REST route calls.
        Rehydrating a full ``EvidenceDocument`` from JSON would re-run every
        domain constructor on data those constructors already approved once,
        which is work with no reader.
        """
        async with self._factory() as db:
            row = await db.get(models.EvidenceDocumentRow, (tenant.value, session_id.value))
            if row is None:
                return None
            bundle = await self._bundle_for_run(db, tenant, RunId(row.run_id))
            if bundle is None:
                return None
            return _document_from(row, bundle)

    async def load_document_payload(
        self, tenant: TenantId, session_id: SessionId
    ) -> dict[str, Any] | None:
        """The stored JSON, for the read path that serves it verbatim."""
        async with self._factory() as db:
            row = await db.get(models.EvidenceDocumentRow, (tenant.value, session_id.value))
            return dict(row.document) if row is not None else None

    async def delete_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        async with unit_of_work(self._factory) as db:
            return await purge_session_rows(db, tenant, session_id)

    # -- writing ----------------------------------------------------------

    def _write_bundle(self, db: AsyncSession, bundle: EvidenceBundle) -> None:
        run_id = bundle.run_id.value
        tenant = bundle.tenant

        for token in bundle.transcript.tokens:
            db.add(token_to_row(token, run_id, tenant))
        for speech_event in bundle.speech_events:
            db.add(speech_event_to_row(speech_event, run_id, tenant))
        for visual_event in bundle.visual_events:
            db.add(visual_event_to_row(visual_event, run_id, tenant))
        for reading in bundle.prosody:
            db.add(prosody_to_row(reading, run_id, tenant))

        for assessment in bundle.quality.assessments:
            db.add(_assessment_to_row(assessment, run_id, tenant))
        for availability in bundle.quality.availability:
            db.add(_availability_to_row(availability, run_id, tenant))

        for pair in bundle.cooccurrences:
            db.add(
                models.CooccurrenceRow(
                    run_id=run_id,
                    tenant_id=tenant.value,
                    speech_event_id=pair.speech_event_id.value,
                    visual_event_id=pair.visual_event_id.value,
                    temporal_distance_ms=pair.temporal_distance_ms,
                    fusion_window_ms=pair.window.width_ms,
                    configuration_id=pair.window.configuration.value,
                )
            )

    # -- reading ----------------------------------------------------------

    async def _bundle_for_run(
        self, db: AsyncSession, tenant: TenantId, run_id: RunId
    ) -> EvidenceBundle | None:
        run = await db.get(models.ProcessingRunRow, run_id.value)
        if run is None:
            return None
        return await self._read_bundle(db, tenant, run)

    async def _read_bundle(
        self, db: AsyncSession, tenant: TenantId, run: models.ProcessingRunRow
    ) -> EvidenceBundle:
        tokens = await self._rows(db, models.WordTokenRow, tenant, run.id)
        speech_rows = await self._rows(db, models.SpeechEventRow, tenant, run.id)
        visual_rows = await self._rows(db, models.VisualEventRow, tenant, run.id)
        prosody_rows = await self._rows(db, models.ProsodyReadingRow, tenant, run.id)
        assessment_rows = await self._rows(db, models.QualityAssessmentRow, tenant, run.id)
        availability_rows = await self._rows(db, models.ModalityAvailabilityRow, tenant, run.id)
        pair_rows = await self._rows(db, models.CooccurrenceRow, tenant, run.id)

        speech_events = tuple(row_to_speech_event(row) for row in speech_rows)
        visual_events = tuple(row_to_visual_event(row) for row in visual_rows)

        # Prosody rows carry no provenance columns of their own - they inherit
        # the run's audio provenance, which every speech event on the same run
        # already records. Duplicating five columns per reading would multiply
        # the largest table in the schema to say something already known.
        audio_provenance = (
            speech_events[0].provenance if speech_events else _synthetic_audio_provenance(run)
        )

        return EvidenceBundle(
            run_id=RunId(run.id),
            session_id=SessionId(run.session_id),
            tenant=tenant,
            transcript=transcript_module.build([row_to_token(row) for row in tokens]),
            quality=QualityReport(
                assessments=tuple(_row_to_assessment(row) for row in assessment_rows),
                availability=tuple(_row_to_availability(row) for row in availability_rows),
            ),
            speech_events=speech_events,
            visual_events=visual_events,
            prosody=tuple(row_to_prosody(row, audio_provenance) for row in prosody_rows),
            cooccurrences=tuple(_row_to_cooccurrence(row) for row in pair_rows),
        )

    async def _rows(self, db: AsyncSession, table: Any, tenant: TenantId, run_id: str) -> list[Any]:
        return list(
            (
                await db.scalars(
                    select(table).where(table.run_id == run_id, table.tenant_id == tenant.value)
                )
            ).all()
        )


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _assessment_to_row(
    assessment: QualityAssessment, run_id: str, tenant: TenantId
) -> models.QualityAssessmentRow:
    row = models.QualityAssessmentRow(
        run_id=run_id,
        tenant_id=tenant.value,
        modality=assessment.modality.value,
        metric=assessment.metric.value,
        start_ms=assessment.window.start.ms,
        end_ms=assessment.window.end.ms,
    )
    if isinstance(assessment.value, Measured):
        row.value = assessment.value.value
        row.unit = assessment.value.unit
        row.detail = ""
    else:
        row.reason = assessment.value.reason.value
        row.detail = assessment.value.detail
    return row


def _row_to_assessment(row: models.QualityAssessmentRow) -> QualityAssessment:
    window = Interval.of(row.start_ms, row.end_ms)
    if row.value is not None:
        value: Measured | Unavailable = Measured(
            value=row.value,
            confidence=Confidence(1.0, CalibrationState.CALIBRATED),
            unit=row.unit or "ratio",
        )
    else:
        value = Unavailable(
            reason=UnavailabilityReason(row.reason or "processing_failed"),
            detail=row.detail,
        )
    return QualityAssessment(
        modality=Modality(row.modality),
        metric=QualityMetric(row.metric),
        window=window,
        value=value,
    )


def _availability_to_row(
    availability: ModalityAvailability, run_id: str, tenant: TenantId
) -> models.ModalityAvailabilityRow:
    return models.ModalityAvailabilityRow(
        run_id=run_id,
        tenant_id=tenant.value,
        modality=availability.modality.value,
        start_ms=availability.window.start.ms,
        end_ms=availability.window.end.ms,
        is_usable=availability.is_usable,
        reason=availability.reason.value if availability.reason else None,
        detail=availability.detail,
    )


def _row_to_availability(row: models.ModalityAvailabilityRow) -> ModalityAvailability:
    return ModalityAvailability(
        modality=Modality(row.modality),
        window=Interval.of(row.start_ms, row.end_ms),
        is_usable=row.is_usable,
        reason=UnavailabilityReason(row.reason) if row.reason else None,
        detail=row.detail,
    )


def _row_to_cooccurrence(row: models.CooccurrenceRow) -> MultimodalCooccurrence:
    return MultimodalCooccurrence(
        speech_event_id=EventId(row.speech_event_id),
        visual_event_id=EventId(row.visual_event_id),
        temporal_distance_ms=row.temporal_distance_ms,
        window=FusionWindow(
            width_ms=row.fusion_window_ms,
            configuration=ConfigurationSnapshotId(row.configuration_id),
        ),
    )


def _document_from(row: models.EvidenceDocumentRow, bundle: EvidenceBundle) -> EvidenceDocument:
    """Rebuild the domain document from its stored JSON plus the bundle.

    The manifest comes from the JSON because it is the record of what produced
    the result; the evidence comes from the rows because they are the queryable
    truth. Rebuilding the manifest from current configuration instead would
    re-stamp a historical document with today's versions, which is the exact
    failure ``ProvenanceManifest`` refuses in its constructor.
    """
    from evidence_engine.domain.evidence.document import ProvenanceManifest
    from evidence_engine.domain.shared.identifiers import ModelVersionId
    from evidence_engine.domain.shared.provenance import SemanticVersion

    manifest_json = row.document["manifest"]
    return EvidenceDocument(
        session_id=SessionId(row.session_id),
        run_id=RunId(row.run_id),
        manifest=ProvenanceManifest(
            pipeline_version=SemanticVersion.parse(manifest_json["pipeline_version"]),
            schema_version=SemanticVersion.parse(manifest_json["schema_version"]),
            taxonomy_version=SemanticVersion.parse(manifest_json["taxonomy_version"]),
            configuration=ConfigurationSnapshotId(manifest_json["configuration_id"]),
            models={
                Modality(modality): ModelVersionId(model)
                for modality, model in manifest_json["models"].items()
            },
        ),
        transcript=bundle.transcript,
        quality=bundle.quality,
        speech_events=bundle.speech_events,
        visual_events=bundle.visual_events,
        prosody=bundle.prosody,
        cooccurrences=bundle.cooccurrences,
    )


def _synthetic_audio_provenance(run: models.ProcessingRunRow) -> Any:
    """Provenance for a run that stored prosody but no speech events.

    Rare and real: a session where the recognizer produced measurements but no
    disfluency crossed a threshold. The run and configuration are known; the
    model version is not recoverable from the prosody rows alone, so it is
    named as unknown rather than guessed. NFR-014 is better served by an
    honest gap than by a plausible-looking wrong value.
    """
    from evidence_engine.domain.shared.identifiers import EvidenceRef, ModelVersionId
    from evidence_engine.domain.shared.provenance import Provenance
    from evidence_engine.domain.shared.taxonomy import TAXONOMY_VERSION

    return Provenance(
        modality=Modality.AUDIO,
        model_version=ModelVersionId("unknown"),
        taxonomy_version=TAXONOMY_VERSION,
        configuration=ConfigurationSnapshotId("unknown"),
        evidence_ref=EvidenceRef(f"audio:{run.id}"),
    )
