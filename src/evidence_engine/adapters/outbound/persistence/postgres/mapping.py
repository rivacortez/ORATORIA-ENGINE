"""Translating between rows and domain objects.

Kept in its own module because it is the seam most likely to be got wrong
quietly. A mapping bug does not raise: it produces a session whose clock is
subtly off, an event whose confidence lost its calibration state, or an
indicator whose reason became an empty string. Each of those reads as a data
problem months later.

Everything here is a pure function of its argument, so the whole seam is
testable without a database.
"""

from __future__ import annotations

from typing import Any

from evidence_engine.adapters.outbound.persistence.postgres import models
from evidence_engine.domain.sessions.capabilities import (
    AudioCodec,
    NegotiatedCapabilities,
    VideoFormat,
)
from evidence_engine.domain.sessions.clock import SessionClock
from evidence_engine.domain.sessions.consent import ConsentReceipt, RetentionPolicy
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.sessions.state import SessionMode, SessionState
from evidence_engine.domain.shared.confidence import CalibrationState, Confidence
from evidence_engine.domain.shared.identifiers import (
    ApplicationId,
    ConfigurationSnapshotId,
    EventId,
    EvidenceRef,
    ModelVersionId,
    SessionId,
    TenantId,
    TokenId,
)
from evidence_engine.domain.shared.measurement import (
    Measured,
    UnavailabilityReason,
    Unavailable,
)
from evidence_engine.domain.shared.provenance import Modality, Provenance, SemanticVersion
from evidence_engine.domain.shared.taxonomy import (
    ContextualRole,
    ProsodicIndicator,
    SpeechEventType,
    VisualEventType,
)
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading
from evidence_engine.domain.transcript.tokens import TokenStatus, WordToken
from evidence_engine.domain.visual_events.events import GazeDirection, VisualEvent

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def session_to_row(session: AnalysisSession) -> models.AnalysisSessionRow:
    return models.AnalysisSessionRow(
        id=session.id.value,
        tenant_id=session.tenant_id.value,
        application_id=session.application_id.value,
        mode=session.mode.value,
        locale=session.locale,
        state=session.state.value,
        configuration_id=session.configuration.value,
        capabilities=_capabilities_to_json(session.capabilities),
        created_at_ms=session.created_at_ms,
        completed_at_ms=session.completed_at_ms,
        clock_accumulated_ms=session.clock.accumulated_ms,
        clock_segment_started_at=session.clock.segment_started_at,
    )


def row_to_session(
    row: models.AnalysisSessionRow, consent: models.ConsentReceiptRow | None
) -> AnalysisSession:
    return AnalysisSession(
        id=SessionId(row.id),
        application_id=ApplicationId(row.application_id),
        tenant_id=TenantId(row.tenant_id),
        mode=SessionMode(row.mode),
        locale=row.locale,
        capabilities=_capabilities_from_json(row.capabilities),
        configuration=ConfigurationSnapshotId(row.configuration_id),
        created_at_ms=row.created_at_ms,
        state=SessionState(row.state),
        clock=SessionClock(
            accumulated_ms=row.clock_accumulated_ms,
            segment_started_at=row.clock_segment_started_at,
        ),
        consent=row_to_consent(consent) if consent is not None else None,
        completed_at_ms=row.completed_at_ms,
    )


def consent_to_row(session_id: SessionId, receipt: ConsentReceipt) -> models.ConsentReceiptRow:
    return models.ConsentReceiptRow(
        session_id=session_id.value,
        policy_version=str(receipt.policy_version),
        retain_raw_media=receipt.retention.retain_raw_media,
        raw_media_ttl_seconds=receipt.retention.raw_media_ttl_seconds,
        retain_derived_aggregates=receipt.retention.retain_derived_aggregates,
        granted_at_ms=receipt.granted_at_ms,
        withdrawn_at_ms=receipt.withdrawn_at_ms,
    )


def row_to_consent(row: models.ConsentReceiptRow) -> ConsentReceipt:
    policy_version = SemanticVersion.parse(row.policy_version)
    return ConsentReceipt(
        session_id=SessionId(row.session_id),
        policy_version=policy_version,
        retention=RetentionPolicy(
            policy_version=policy_version,
            retain_raw_media=row.retain_raw_media,
            raw_media_ttl_seconds=row.raw_media_ttl_seconds,
            retain_derived_aggregates=row.retain_derived_aggregates,
        ),
        granted_at_ms=row.granted_at_ms,
        withdrawn_at_ms=row.withdrawn_at_ms,
    )


def _capabilities_to_json(capabilities: NegotiatedCapabilities) -> dict[str, Any]:
    return {
        "audio_codec": capabilities.audio_codec.value,
        "sample_rate_hz": capabilities.sample_rate_hz,
        "locale": capabilities.locale,
        "video_format": (capabilities.video_format.value if capabilities.video_format else None),
        "frame_rate_fps": capabilities.frame_rate_fps,
    }


def _capabilities_from_json(payload: dict[str, Any]) -> NegotiatedCapabilities:
    video_format = payload.get("video_format")
    return NegotiatedCapabilities(
        audio_codec=AudioCodec(payload["audio_codec"]),
        sample_rate_hz=int(payload["sample_rate_hz"]),
        locale=str(payload["locale"]),
        video_format=VideoFormat(video_format) if video_format else None,
        frame_rate_fps=payload.get("frame_rate_fps"),
    )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def token_to_row(token: WordToken, run_id: str, tenant: TenantId) -> models.WordTokenRow:
    return models.WordTokenRow(
        id=token.id.value,
        run_id=run_id,
        tenant_id=tenant.value,
        raw_text=token.raw_text,
        start_ms=token.interval.start.ms,
        end_ms=token.interval.end.ms,
        tolerance_ms=token.interval.tolerance_ms,
        confidence=token.confidence.value,
        calibration=token.confidence.state.value,
        status=token.status.value,
    )


def row_to_token(row: models.WordTokenRow) -> WordToken:
    return WordToken(
        id=TokenId(row.id),
        raw_text=row.raw_text,
        interval=Interval.of(row.start_ms, row.end_ms, row.tolerance_ms),
        confidence=Confidence(row.confidence, CalibrationState(row.calibration)),
        status=TokenStatus(row.status),
    )


def speech_event_to_row(event: SpeechEvent, run_id: str, tenant: TenantId) -> models.SpeechEventRow:
    return models.SpeechEventRow(
        id=event.id.value,
        run_id=run_id,
        tenant_id=tenant.value,
        type=event.type.value,
        raw_text=event.raw_text,
        context_role=event.context_role.value if event.context_role else None,
        start_ms=event.interval.start.ms,
        end_ms=event.interval.end.ms,
        tolerance_ms=event.interval.tolerance_ms,
        confidence=event.confidence.value,
        calibration=event.confidence.state.value,
        is_final=event.is_final,
        model_version=event.provenance.model_version.value,
        taxonomy_version=str(event.provenance.taxonomy_version),
        configuration_id=event.provenance.configuration.value,
        evidence_ref=event.provenance.evidence_ref.value,
    )


def row_to_speech_event(row: models.SpeechEventRow) -> SpeechEvent:
    return SpeechEvent(
        id=EventId(row.id),
        type=SpeechEventType(row.type),
        interval=Interval.of(row.start_ms, row.end_ms, row.tolerance_ms),
        confidence=Confidence(row.confidence, CalibrationState(row.calibration)),
        provenance=_provenance(row, Modality.AUDIO),
        raw_text=row.raw_text,
        context_role=ContextualRole(row.context_role) if row.context_role else None,
        is_final=row.is_final,
    )


def visual_event_to_row(event: VisualEvent, run_id: str, tenant: TenantId) -> models.VisualEventRow:
    return models.VisualEventRow(
        id=event.id.value,
        run_id=run_id,
        tenant_id=tenant.value,
        type=event.type.value,
        direction=event.direction.value if event.direction else None,
        magnitude=event.magnitude,
        start_ms=event.interval.start.ms,
        end_ms=event.interval.end.ms,
        tolerance_ms=event.interval.tolerance_ms,
        confidence=event.confidence.value,
        calibration=event.confidence.state.value,
        is_final=event.is_final,
        model_version=event.provenance.model_version.value,
        taxonomy_version=str(event.provenance.taxonomy_version),
        configuration_id=event.provenance.configuration.value,
        evidence_ref=event.provenance.evidence_ref.value,
    )


def row_to_visual_event(row: models.VisualEventRow) -> VisualEvent:
    return VisualEvent(
        id=EventId(row.id),
        type=VisualEventType(row.type),
        interval=Interval.of(row.start_ms, row.end_ms, row.tolerance_ms),
        confidence=Confidence(row.confidence, CalibrationState(row.calibration)),
        provenance=_provenance(row, Modality.VIDEO),
        direction=GazeDirection(row.direction) if row.direction else None,
        magnitude=row.magnitude,
        is_final=row.is_final,
    )


def prosody_to_row(
    reading: ProsodyReading, run_id: str, tenant: TenantId
) -> models.ProsodyReadingRow:
    """Map a reading onto the measured-xor-unavailable columns.

    The branch here is the database-side half of FR-025: a measured reading
    writes ``value`` and leaves ``reason`` null, an unavailable one does the
    reverse, and the check constraint refuses anything else. There is no code
    path that writes both or neither.
    """
    row = models.ProsodyReadingRow(
        run_id=run_id,
        tenant_id=tenant.value,
        indicator=reading.indicator.value,
        start_ms=reading.window.start.ms,
        end_ms=reading.window.end.ms,
    )
    if isinstance(reading.value, Measured):
        row.value = reading.value.value
        row.unit = reading.value.unit
        row.confidence = reading.value.confidence.value
        row.calibration = reading.value.confidence.state.value
        row.detail = ""
    else:
        row.reason = reading.value.reason.value
        row.detail = reading.value.detail
    return row


def row_to_prosody(row: models.ProsodyReadingRow, provenance: Provenance) -> ProsodyReading:
    window = Interval.of(row.start_ms, row.end_ms)
    if row.value is not None and row.unit is not None:
        return ProsodyReading(
            indicator=ProsodicIndicator(row.indicator),
            window=window,
            value=Measured(
                value=row.value,
                confidence=Confidence(
                    row.confidence or 0.0, CalibrationState(row.calibration or "raw")
                ),
                unit=row.unit,
            ),
            provenance=provenance,
        )
    return ProsodyReading(
        indicator=ProsodicIndicator(row.indicator),
        window=window,
        value=Unavailable(
            reason=UnavailabilityReason(row.reason or "processing_failed"),
            detail=row.detail,
        ),
        provenance=provenance,
    )


def _provenance(
    row: models.SpeechEventRow | models.VisualEventRow, modality: Modality
) -> Provenance:
    return Provenance(
        modality=modality,
        model_version=ModelVersionId(row.model_version),
        taxonomy_version=SemanticVersion.parse(row.taxonomy_version),
        configuration=ConfigurationSnapshotId(row.configuration_id),
        evidence_ref=EvidenceRef(row.evidence_ref),
    )
