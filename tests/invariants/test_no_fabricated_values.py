"""FR-025 / driver 4: an unavailable indicator never becomes a number.

These tests exist because the failure they guard against is silent. Nothing
crashes when a missing measurement is serialized as ``0.0``; the pipeline runs
green, the dashboard renders, and a student is told they scored zero on
something that was never measured. The only place to catch it is here.
"""

from __future__ import annotations

import pytest

from evidence_engine.domain.quality.assessment import (
    ModalityAvailability,
    QualityAssessment,
    QualityMetric,
    QualityReport,
)
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import FabricatedValue
from evidence_engine.domain.shared.measurement import (
    Measured,
    UnavailabilityReason,
    Unavailable,
    value_or_raise,
)
from evidence_engine.domain.shared.provenance import Modality, Provenance
from evidence_engine.domain.shared.taxonomy import ProsodicIndicator
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.prosody import ProsodyReading

pytestmark = pytest.mark.invariant


def test_unavailable_has_no_value_attribute() -> None:
    missing = Unavailable(reason=UnavailabilityReason.SUBJECT_NOT_OBSERVABLE)

    with pytest.raises(FabricatedValue, match="FR-025"):
        _ = missing.value  # type: ignore[attr-defined]


def test_unavailable_has_no_confidence_attribute() -> None:
    missing = Unavailable(reason=UnavailabilityReason.PROCESSING_FAILED)

    with pytest.raises(FabricatedValue, match="FR-025"):
        _ = missing.confidence  # type: ignore[attr-defined]


def test_value_or_raise_refuses_an_unavailable_indicator() -> None:
    missing = Unavailable(
        reason=UnavailabilityReason.QUALITY_BELOW_THRESHOLD, detail="face not detected"
    )

    with pytest.raises(FabricatedValue) as caught:
        value_or_raise(missing)

    # The reason travels in the message: a caller reading a log must be able to
    # tell a covered camera from a crashed runtime without opening a database.
    assert "quality_below_threshold" in str(caught.value)


def test_value_or_raise_returns_a_measured_number() -> None:
    measured = Measured(value=0.62, confidence=Confidence.calibrated(0.8), unit="ratio")

    assert value_or_raise(measured) == pytest.approx(0.62)


def test_unavailable_prosody_reading_is_still_published(
    audio_provenance: Provenance,
) -> None:
    """FR-024 and US-004: the absence is part of the result, not an error."""
    window = Interval.of(0, 5_000)
    reading = ProsodyReading(
        indicator=ProsodicIndicator.PITCH_MEAN_HZ,
        window=window,
        value=Unavailable(reason=UnavailabilityReason.INPUT_GAP, detail="chunks 40-52 lost"),
        provenance=audio_provenance,
    )

    assert reading.is_available is False
    assert reading.value.reason is UnavailabilityReason.INPUT_GAP


def test_quality_summary_omits_unavailable_metrics_instead_of_zeroing_them() -> None:
    """A dashboard must not show a flat zero line during a total outage."""
    window = Interval.of(0, 10_000)
    report = QualityReport(
        assessments=(
            QualityAssessment(
                modality=Modality.AUDIO,
                metric=QualityMetric.SIGNAL_TO_NOISE_DB,
                window=window,
                value=Measured(24.0, Confidence.calibrated(0.9), "dBFS"),
            ),
            QualityAssessment(
                modality=Modality.VIDEO,
                metric=QualityMetric.FACE_VISIBILITY_RATIO,
                window=window,
                value=Unavailable(reason=UnavailabilityReason.MODALITY_NOT_CAPTURED),
            ),
        )
    )

    summary = report.summary()

    assert summary == {"audio.signal_to_noise_db": 24.0}
    assert "video.face_visibility_ratio" not in summary


def test_modality_marked_unusable_must_carry_a_reason() -> None:
    from evidence_engine.domain.quality.assessment import QualityViolation

    with pytest.raises(QualityViolation, match="FR-025"):
        ModalityAvailability(
            modality=Modality.VIDEO,
            window=Interval.of(0, 1_000),
            is_usable=False,
            reason=None,
        )


def test_qa02_video_outage_leaves_speech_untouched() -> None:
    """QA-02: twenty seconds without frames, speech evidence intact."""
    outage = Interval.of(20_000, 40_000)
    report = QualityReport(
        availability=(
            ModalityAvailability.usable(Modality.AUDIO, Interval.of(0, 60_000)),
            ModalityAvailability.usable(Modality.VIDEO, Interval.of(0, 20_000)),
            ModalityAvailability.unusable(
                Modality.VIDEO, outage, UnavailabilityReason.INPUT_GAP, "camera stream dropped"
            ),
        )
    )

    assert report.is_usable(Modality.AUDIO, outage) is True

    missing = report.unavailability_for(Modality.VIDEO, outage)
    assert missing is not None
    assert missing.reason is UnavailabilityReason.INPUT_GAP
    with pytest.raises(FabricatedValue):
        _ = missing.value  # type: ignore[attr-defined]


def test_modality_never_captured_reports_that_reason() -> None:
    report = QualityReport(
        availability=(ModalityAvailability.usable(Modality.AUDIO, Interval.of(0, 60_000)),)
    )

    missing = report.unavailability_for(Modality.VIDEO, Interval.of(0, 1_000))

    assert missing is not None
    assert missing.reason is UnavailabilityReason.MODALITY_NOT_CAPTURED
