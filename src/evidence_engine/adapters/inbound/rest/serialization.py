"""Rendering the evidence document onto the wire (§7.1, US-013).

Everything observed goes out; nothing is ordered by importance. The two rules
that shape this module both come from FR-029, pulling in opposite directions:
the document must not carry a rank, and it must not hide low-confidence
evidence. So there is no filtering here at all, and no sort by anything but
time.

The other rule is FR-025's. An unavailable indicator is rendered with
``available: false`` and its reason, and there is no numeric field on it - not
``value: null``, which a consumer's chart library will happily coerce to zero,
but no key at all. A reader that wants the number has to notice it is missing.
"""

from __future__ import annotations

from typing import Any

from evidence_engine.domain.evidence.document import EvidenceDocument
from evidence_engine.domain.quality.assessment import QualityReport
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.measurement import Indicator, Measured, Unavailable
from evidence_engine.domain.shared.provenance import Seed, Seeded
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading
from evidence_engine.domain.transcript.tokens import Placement, Timed
from evidence_engine.domain.transcript.transcript import Transcript
from evidence_engine.domain.visual_events.events import VisualEvent


def render_document(document: EvidenceDocument, *, schema_version: str) -> dict[str, Any]:
    """The complete result, as JSON-ready data."""
    return {
        "schema_version": schema_version,
        "session_id": document.session_id.value,
        "run_id": document.run_id.value,
        # Published, not merely enforced: a consumer reading "none" knows the
        # engine will never hand it a ranking and builds its own (§17).
        "ranking_authority": EvidenceDocument.ranking_authority,
        "manifest": {
            "pipeline_version": str(document.manifest.pipeline_version),
            "schema_version": str(document.manifest.schema_version),
            "taxonomy_version": str(document.manifest.taxonomy_version),
            "configuration_id": document.manifest.configuration.value,
            "models": {
                modality.value: model.value for modality, model in document.manifest.models.items()
            },
        },
        "transcript": render_transcript(document.transcript),
        "speech_events": [render_speech_event(e) for e in document.speech_events],
        "visual_events": [render_visual_event(e) for e in document.visual_events],
        "prosody": [render_prosody(r) for r in document.prosody],
        "cooccurrences": [
            {
                "speech_event_id": pair.speech_event_id.value,
                "visual_event_id": pair.visual_event_id.value,
                "temporal_distance_ms": pair.temporal_distance_ms,
                "fusion_window_ms": pair.window.width_ms,
                # FR-028, on the wire. Always present and always false, so a
                # consumer never has to infer it from absence.
                "causal_inference": pair.causal_inference,
            }
            for pair in document.cooccurrences
        ],
        "quality": render_quality(document.quality),
    }


def render_transcript(transcript: Transcript) -> dict[str, Any]:
    """The literal transcript. No cleanup, no punctuation repair (FR-011)."""
    return {
        "raw_text": transcript.raw_text(),
        "finalized_through_ms": transcript.finalized_time_frontier.ms,
        # Published rather than left to be counted, because it is the number
        # that tells a reader how much of the temporal analysis ran on less
        # than the whole transcript. Silent pauses and multimodal fusion read
        # placed tokens only; this says how many they did not see.
        "unaligned_token_count": transcript.unaligned_count,
        "tokens": [
            {
                "id": token.id.value,
                "sequence": list(token.sequence.key),
                "raw_text": token.raw_text,
                **_render_placement(token.placement),
                **_render_token_confidence(token.confidence),
                "status": token.status.value,
            }
            for token in transcript.tokens
        ],
    }


def render_speech_event(event: SpeechEvent) -> dict[str, Any]:
    return {
        "id": event.id.value,
        "type": event.type.value,
        "start_ms": event.interval.start.ms,
        "end_ms": event.interval.end.ms,
        # NFR-004: the engine never claims an exact boundary, so the tolerance
        # travels with every interval rather than living in documentation.
        "tolerance_ms": event.interval.tolerance_ms,
        "confidence": event.confidence.value,
        "calibration": event.confidence.state.value,
        "raw_text": event.raw_text,
        "context_role": event.context_role.value if event.context_role else None,
        "counts_as_disfluency": event.counts_as_disfluency,
        "is_final": event.is_final,
        "provenance": _render_provenance(event),
    }


def render_visual_event(event: VisualEvent) -> dict[str, Any]:
    return {
        "id": event.id.value,
        "type": event.type.value,
        "start_ms": event.interval.start.ms,
        "end_ms": event.interval.end.ms,
        "tolerance_ms": event.interval.tolerance_ms,
        "confidence": event.confidence.value,
        "calibration": event.confidence.state.value,
        "direction": event.direction.value if event.direction else None,
        "magnitude": event.magnitude,
        "describes_capture_quality": event.describes_capture_quality,
        "is_final": event.is_final,
        "provenance": _render_provenance(event),
    }


def render_prosody(reading: ProsodyReading) -> dict[str, Any]:
    body: dict[str, Any] = {
        "indicator": reading.indicator.value,
        "start_ms": reading.window.start.ms,
        "end_ms": reading.window.end.ms,
        "available": reading.is_available,
    }
    body.update(_render_indicator(reading.value))
    return body


def render_quality(report: QualityReport) -> dict[str, Any]:
    return {
        "assessments": [
            {
                "modality": assessment.modality.value,
                "metric": assessment.metric.value,
                "start_ms": assessment.window.start.ms,
                "end_ms": assessment.window.end.ms,
                "available": assessment.is_available,
                **_render_indicator(assessment.value),
            }
            for assessment in report.assessments
        ],
        "availability": [
            {
                "modality": record.modality.value,
                "start_ms": record.window.start.ms,
                "end_ms": record.window.end.ms,
                "usable": record.is_usable,
                "reason": record.reason.value if record.reason else None,
                "detail": record.detail,
            }
            for record in report.availability
        ],
    }


def _render_placement(placement: Placement) -> dict[str, Any]:
    """Where the word sat, or the reason that is unknown.

    Disjoint key sets again. An unplaced word has no ``start_ms`` at all rather
    than ``"start_ms": null`` - a null in a numeric field is what a consumer
    coerces to zero, and zero here would put the word at the start of the
    session, which is both wrong and plausible-looking.
    """
    if isinstance(placement, Timed):
        return {
            "placed": True,
            "start_ms": placement.interval.start.ms,
            "end_ms": placement.interval.end.ms,
            "tolerance_ms": placement.interval.tolerance_ms,
        }
    return {
        "placed": False,
        "placement_unavailable_reason": placement.reason.value,
        "placement_unavailable_detail": placement.detail,
    }


def _render_token_confidence(confidence: Confidence | Unavailable) -> dict[str, Any]:
    """A word's confidence, or the reason there is none.

    Disjoint key sets, for the same reason ``_render_indicator`` uses them: an
    unavailable confidence has no ``confidence`` key at all rather than
    ``"confidence": null``, because a null in a numeric field is what a chart
    coerces to zero - and "this word scored 0.0" is a claim the recogniser
    never made. whisper-large-v3 emits no per-word posterior, so this branch is
    the normal one for the baseline runtime, not an edge case.
    """
    if isinstance(confidence, Unavailable):
        return {
            "confidence_available": False,
            "confidence_unavailable_reason": confidence.reason.value,
            "confidence_unavailable_detail": confidence.detail,
        }
    return {
        "confidence_available": True,
        "confidence": confidence.value,
        "calibration": confidence.state.value,
    }


def _render_indicator(indicator: Indicator) -> dict[str, Any]:
    """Render a measurement, or the reason there is none.

    The two branches produce disjoint key sets. An unavailable indicator has no
    ``value`` key at all - not ``"value": null`` - because a null in a numeric
    field is exactly what a charting library coerces to zero, which is the
    failure FR-025 exists to prevent.
    """
    if isinstance(indicator, Measured):
        return {
            "value": indicator.value,
            "unit": indicator.unit,
            "confidence": indicator.confidence.value,
            "calibration": indicator.confidence.state.value,
        }
    return {"reason": indicator.reason.value, "detail": indicator.detail}


def _render_provenance(event: SpeechEvent | VisualEvent) -> dict[str, Any]:
    """NFR-014's six fields, on every derived event, plus NFR-015's seed."""
    provenance = event.provenance
    return {
        "modality": provenance.modality.value,
        "model_version": provenance.model_version.value,
        "taxonomy_version": str(provenance.taxonomy_version),
        "configuration_id": provenance.configuration.value,
        "evidence_ref": provenance.evidence_ref.value,
        "seed": _render_seed(provenance.seed),
    }


def _render_seed(seed: Seed) -> dict[str, Any]:
    """The seed, or the reason there is none.

    Always present, like ``causal_inference``, so a consumer reads the case
    rather than inferring it from a key that is not there. The two branches
    produce disjoint key sets for the reason ``_render_indicator`` does: an
    unseeded provenance has no ``value`` key at all, because ``seed ?? 0`` is
    what a consumer writes to make a reproduction script run, and 0 is a seed a
    run could genuinely have used. The rerun would then complete and report a
    match against a result produced under something else entirely.
    """
    if isinstance(seed, Seeded):
        return {"recorded": True, "value": seed.value}
    return {"recorded": False, "reason": seed.reason.value}
