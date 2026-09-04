"""Visual events: what the camera could observe, and only that.

FR-023 is the constraint that defines this module: "No visual result shall
contain emotional or clinical labels." The temptation in a visual pipeline is
enormous, because the landmark geometry that yields a posture deviation is one
short classifier away from yielding an "affect score", and the literature is
full of papers claiming the second follows from the first. §17 calls that
outcome - "multimodal becomes an emotion detector" - a scientific invalidity,
and it is: the geometry is real, the inference from it to an internal state is
not supported by the instrument.

So a visual event names a configuration of the body over an interval. What that
configuration means about the person is not a question this service answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import DomainError
from evidence_engine.domain.shared.identifiers import EventId
from evidence_engine.domain.shared.provenance import Modality, Provenance
from evidence_engine.domain.shared.taxonomy import VisualEventType
from evidence_engine.domain.shared.timeline import Interval


class VisualEventViolation(DomainError):
    """A visual event was built in a shape the taxonomy forbids."""


class GazeDirection(StrEnum):
    """Where an off-camera gaze was pointing, when that can be estimated.

    Coarse buckets rather than angles, because the estimate is derived from
    facial geometry in consumer video and §5 forbids presenting it as
    hardware-grade eye tracking. Four bins are defensible from that signal; a
    number in degrees would imply a precision the method does not have.

    ``UNDETERMINED`` exists so that "off camera, direction unknown" stays
    expressible. Collapsing it into one of the real directions to avoid a null
    would fabricate the very detail the estimator declined to provide.
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UNDETERMINED = "undetermined"


#: Classes that describe where the gaze was pointing, and therefore may carry a
#: direction. Everything else may not: a lighting failure has no direction, and
#: attaching one would be noise dressed as evidence.
_GAZE_CLASSES: frozenset[VisualEventType] = frozenset(
    {VisualEventType.GAZE_TOWARD_CAMERA, VisualEventType.GAZE_AWAY_FROM_CAMERA}
)

#: Classes that report a capture problem rather than a behaviour. FR-020 groups
#: them, and the distinction is load-bearing: US-001 requires the product not to
#: describe a missing camera as a performance error, which is only possible if
#: the two are distinguishable in the data.
_QUALITY_CLASSES: frozenset[VisualEventType] = frozenset(
    {
        VisualEventType.FACE_VISIBILITY_LOSS,
        VisualEventType.INSUFFICIENT_LIGHTING,
        VisualEventType.INSUFFICIENT_FRAMING,
    }
)

#: FR-022 gates the optional behavioural classes behind confidence and framing.
#: Below this, nothing is emitted at all - not a low-confidence event. A
#: precision-focused class that emits its uncertain cases is not precision
#: focused.
OPTIONAL_CLASS_MIN_CONFIDENCE = 0.75

_OPTIONAL_CLASSES: frozenset[VisualEventType] = frozenset(
    {VisualEventType.SELF_TOUCH, VisualEventType.REPETITIVE_HAND_MOVEMENT}
)


@dataclass(frozen=True, slots=True)
class VisualEvent:
    """§8 ``VisualEvent``: one observable visual configuration over a window."""

    id: EventId
    type: VisualEventType
    interval: Interval
    confidence: Confidence
    provenance: Provenance
    #: Set only for gaze classes; ``None`` everywhere else.
    direction: GazeDirection | None = None
    #: Magnitude in the class's own terms - degrees of torso deviation, hertz
    #: of oscillation - when the class has one. Its meaning is documented with
    #: the producing detector, per the same discipline §8 applies to
    #: indicator maps: open, but never inferred from the field name.
    magnitude: float | None = None
    is_final: bool = False

    def __post_init__(self) -> None:
        self._require_video_modality()
        self._require_direction_only_on_gaze()
        self._require_threshold_on_optional_classes()

    def _require_video_modality(self) -> None:
        if self.provenance.modality is not Modality.VIDEO:
            raise VisualEventViolation(
                f"visual event {self.id} claims modality "
                f"'{self.provenance.modality.value}'; visual evidence comes from video"
            )

    def _require_direction_only_on_gaze(self) -> None:
        if self.type in _GAZE_CLASSES:
            if self.direction is None:
                raise VisualEventViolation(
                    f"'{self.type.value}' must state a direction; use "
                    f"'{GazeDirection.UNDETERMINED.value}' when it cannot be estimated"
                )
            return
        if self.direction is not None:
            raise VisualEventViolation(
                f"'{self.type.value}' does not describe a gaze and cannot carry a direction"
            )

    def _require_threshold_on_optional_classes(self) -> None:
        if self.type not in _OPTIONAL_CLASSES:
            return
        if not self.confidence.meets(OPTIONAL_CLASS_MIN_CONFIDENCE):
            raise VisualEventViolation(
                f"'{self.type.value}' is a P1 class gated by FR-022: it may only be "
                f"emitted with calibrated confidence >= {OPTIONAL_CLASS_MIN_CONFIDENCE}, "
                f"got {self.confidence.value} ({self.confidence.state.value})"
            )

    # -- queries ----------------------------------------------------------

    @property
    def describes_capture_quality(self) -> bool:
        """True when this reports a camera problem, not a behaviour (FR-020)."""
        return self.type in _QUALITY_CLASSES

    def finalize(self) -> VisualEvent:
        """Promote to final, keeping the identifier stable (§7.4)."""
        if self.is_final:
            return self
        return VisualEvent(
            id=self.id,
            type=self.type,
            interval=self.interval,
            confidence=self.confidence,
            provenance=self.provenance,
            direction=self.direction,
            magnitude=self.magnitude,
            is_final=True,
        )
