"""Turning vision-runtime hypotheses into published visual evidence.

Where the speech assembler publishes everything and lets confidence speak, this
one has a genuine filter, and the asymmetry is required by the spec rather than
chosen.

FR-022 says the optional behavioural classes - self-touch, repetitive hand
movement - "may" be detected "only when confidence and framing thresholds are
satisfied", and §13 Phase 5 calls for a precision-focused evaluation of them. A
precision-focused class that emits its uncertain cases is not precision
focused. So below threshold, nothing is emitted at all: the hypothesis is
dropped, not published as low confidence.

The P0 classes keep the FR-029 rule. They are published with whatever
confidence they have, because hiding weak evidence is itself a selection
decision and this service does not make those.

The other half of this module is the quality gate. §5 forbids the visual
quality gate from fabricating missing observations, and FR-020 requires
detection of visibility loss, poor lighting and insufficient framing. A window
that fails any of those produces an availability record with a reason, which is
what every dependent indicator will carry instead of a number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evidence_engine.application.ports.platform import ConfigurationSnapshot
from evidence_engine.application.ports.runtimes import (
    VisualEventHypothesis,
    VisualQualitySignal,
    VisualResult,
)
from evidence_engine.application.services.calibration import Calibrator
from evidence_engine.application.services.event_identity import derive_event_id
from evidence_engine.domain.quality.assessment import (
    ModalityAvailability,
    QualityAssessment,
    QualityMetric,
)
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.identifiers import EvidenceRef, RunId
from evidence_engine.domain.shared.measurement import Measured, UnavailabilityReason
from evidence_engine.domain.shared.provenance import Modality, Provenance
from evidence_engine.domain.shared.taxonomy import VisualEventType
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.visual_events.events import (
    OPTIONAL_CLASS_MIN_CONFIDENCE,
    VisualEvent,
)

#: FR-022's gated classes. Emitted only above threshold; dropped otherwise.
_GATED_CLASSES: frozenset[VisualEventType] = frozenset(
    {VisualEventType.SELF_TOUCH, VisualEventType.REPETITIVE_HAND_MOVEMENT}
)

#: Below these, the corresponding estimates are not trustworthy and the window
#: is marked unusable. Defaults only - a configuration snapshot overrides them
#: per session (FR-015), which is why they are read from the snapshot below and
#: these constants exist purely as the value a fresh snapshot starts from.
DEFAULT_MIN_FACE_VISIBILITY = 0.5
DEFAULT_MIN_LUMINANCE = 0.15
DEFAULT_MIN_FRAMING_COVERAGE = 0.6


@dataclass(frozen=True, slots=True)
class AssembledVisual:
    """What one batch of frames yielded."""

    events: tuple[VisualEvent, ...]
    assessments: tuple[QualityAssessment, ...]
    availability: tuple[ModalityAvailability, ...]
    #: Hypotheses dropped by the FR-022 gate. Counted, not published: US-009
    #: lets an administrator watch detector behaviour without reading evidence,
    #: and a gate whose rejections are invisible cannot be tuned.
    gated_out: int = 0


class VisualAssembler:
    """Builds visual evidence and quality records from one runtime result."""

    def __init__(
        self,
        run_id: RunId,
        configuration: ConfigurationSnapshot,
        calibrator: Calibrator,
    ) -> None:
        self._run_id = run_id
        self._configuration = configuration
        self._calibrator = calibrator

    def assemble(self, result: VisualResult) -> AssembledVisual:
        provenance = self._provenance(result)

        events: list[VisualEvent] = []
        gated_out = 0
        for hypothesis in result.events:
            built = self._event(hypothesis, provenance)
            if built is None:
                gated_out += 1
                continue
            events.append(built)

        assessments: list[QualityAssessment] = []
        availability: list[ModalityAvailability] = []
        for signal in result.quality:
            assessments.extend(self._assessments(signal))
            availability.append(self._availability(signal))

        return AssembledVisual(
            events=tuple(events),
            assessments=tuple(assessments),
            availability=tuple(availability),
            gated_out=gated_out,
        )

    # -- pieces -----------------------------------------------------------

    def _provenance(self, result: VisualResult) -> Provenance:
        return Provenance(
            modality=Modality.VIDEO,
            model_version=result.model_version,
            taxonomy_version=self._configuration.taxonomy_version,
            configuration=self._configuration.id,
            evidence_ref=EvidenceRef(f"video:{self._run_id.value}"),
        )

    def _event(
        self, hypothesis: VisualEventHypothesis, provenance: Provenance
    ) -> VisualEvent | None:
        """Build the event, or return ``None`` when FR-022's gate rejects it."""
        confidence = self._calibrator.calibrate(hypothesis.type.value, hypothesis.score)

        if hypothesis.type in _GATED_CLASSES:
            threshold = self._configuration.visual_thresholds.get(
                hypothesis.type.value, OPTIONAL_CLASS_MIN_CONFIDENCE
            )
            # `meets` also refuses uncalibrated scores, so an unevaluated
            # detector cannot open this gate by reporting a high raw number.
            if not confidence.meets(threshold):
                return None

        return VisualEvent(
            id=derive_event_id(self._run_id, hypothesis.type.value, hypothesis.start_ms),
            type=hypothesis.type,
            interval=Interval.of(hypothesis.start_ms, hypothesis.end_ms),
            confidence=confidence,
            provenance=provenance,
            direction=hypothesis.direction,
            magnitude=hypothesis.magnitude,
        )

    def _assessments(self, signal: VisualQualitySignal) -> tuple[QualityAssessment, ...]:
        window = Interval.of(signal.start_ms, signal.end_ms)
        # Reported as measured even when they fail the gate: the number is a
        # real observation of the *recording*, and FR-024 asks for the metric.
        # What becomes unavailable is the indicator derived from it, not the
        # measurement of how bad the input was.
        certain = Confidence.calibrated(1.0)
        return (
            QualityAssessment(
                modality=Modality.VIDEO,
                metric=QualityMetric.FACE_VISIBILITY_RATIO,
                window=window,
                value=Measured(signal.face_visibility, certain, "ratio"),
            ),
            QualityAssessment(
                modality=Modality.VIDEO,
                metric=QualityMetric.MEAN_LUMINANCE,
                window=window,
                value=Measured(signal.mean_luminance, certain, "ratio"),
            ),
            QualityAssessment(
                modality=Modality.VIDEO,
                metric=QualityMetric.FRAMING_COVERAGE,
                window=window,
                value=Measured(signal.framing_coverage, certain, "ratio"),
            ),
            QualityAssessment(
                modality=Modality.VIDEO,
                metric=QualityMetric.LANDMARK_CONFIDENCE,
                window=window,
                value=Measured(signal.landmark_confidence, certain, "ratio"),
            ),
        )

    def _availability(self, signal: VisualQualitySignal) -> ModalityAvailability:
        """Decide whether this window can support visual indicators at all.

        Ordered from most to least specific so the reason the consumer sees is
        the actionable one. "Your face was not visible" and "the room is too
        dark" lead to different fixes, and reporting whichever check happened
        to run first would sometimes give the wrong advice.
        """
        window = Interval.of(signal.start_ms, signal.end_ms)

        if signal.face_visibility < DEFAULT_MIN_FACE_VISIBILITY:
            return ModalityAvailability.unusable(
                Modality.VIDEO,
                window,
                UnavailabilityReason.SUBJECT_NOT_OBSERVABLE,
                f"face visible in {signal.face_visibility:.0%} of the window",
            )
        if signal.mean_luminance < DEFAULT_MIN_LUMINANCE:
            return ModalityAvailability.unusable(
                Modality.VIDEO,
                window,
                UnavailabilityReason.QUALITY_BELOW_THRESHOLD,
                "the face region is too dark for reliable geometry",
            )
        if signal.framing_coverage < DEFAULT_MIN_FRAMING_COVERAGE:
            return ModalityAvailability.unusable(
                Modality.VIDEO,
                window,
                UnavailabilityReason.QUALITY_BELOW_THRESHOLD,
                "the framing does not contain the region these indicators need",
            )
        return ModalityAvailability.usable(Modality.VIDEO, window)


def modality_not_captured(
    windows: Sequence[Interval], modality: Modality
) -> tuple[ModalityAvailability, ...]:
    """Mark a modality that was never supplied.

    An audio-only session is valid, not degraded (FR-006 makes video optional),
    but the result still has to say why every visual indicator is missing.
    Silence here would leave a consumer unable to distinguish "no camera" from
    "nothing detected", and US-001 requires the product not to describe a
    missing camera as a performance error.
    """
    return tuple(
        ModalityAvailability.unusable(modality, window, UnavailabilityReason.MODALITY_NOT_CAPTURED)
        for window in windows
    )
