"""Quality and availability, per modality and per indicator.

FR-024 requires quality and availability to be calculated for each modality
*and each indicator*, which is finer-grained than it first appears. A session
where the camera worked for eight of ten minutes is not "video: available" and
not "video: unavailable"; it is a set of windows, some of which support a
posture reading and none of which support a self-touch reading because the
hands were out of frame the whole time. Reporting at session granularity would
force the consumer to choose between discarding good evidence and trusting bad
evidence.

The assessment is also what US-001 and US-004 render: "which indicators will
not be calculated", and "the technical reason". So it is written to be read by
a person, through a consuming application, not only by a monitoring dashboard.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from evidence_engine.domain.shared.errors import DomainError
from evidence_engine.domain.shared.measurement import (
    Indicator,
    Measured,
    UnavailabilityReason,
    Unavailable,
)
from evidence_engine.domain.shared.provenance import Modality
from evidence_engine.domain.shared.timeline import Interval


class QualityViolation(DomainError):
    """A quality assessment was built inconsistently."""


class QualityMetric(StrEnum):
    """What the quality gate measures before letting a modality through.

    Each one is a direct cause of a downstream indicator being unavailable, and
    naming them separately is what lets the consuming application say something
    actionable - "your microphone is clipping" beats "audio quality: low".
    """

    # Audio
    SIGNAL_TO_NOISE_DB = "signal_to_noise_db"
    CLIPPING_RATIO = "clipping_ratio"
    SILENCE_RATIO = "silence_ratio"
    INPUT_LEVEL_DBFS = "input_level_dbfs"
    # Video
    FACE_VISIBILITY_RATIO = "face_visibility_ratio"
    MEAN_LUMINANCE = "mean_luminance"
    FRAMING_COVERAGE = "framing_coverage"
    LANDMARK_CONFIDENCE = "landmark_confidence"
    # Transport and synchronization
    CHUNK_LOSS_RATIO = "chunk_loss_ratio"
    AUDIO_VIDEO_SKEW_MS = "audio_video_skew_ms"
    EFFECTIVE_FRAME_RATE_FPS = "effective_frame_rate_fps"


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """§8 ``QualityAssessment``: one metric, one modality, one window.

    Note that ``value`` is an ``Indicator``, so a quality metric can itself be
    unavailable - and frequently is. If the camera never produced a frame, the
    face-visibility ratio is not zero; there was nothing to compute it over.
    Reporting zero would tell a student their face was never visible, which is
    a claim about them rather than about the camera.
    """

    modality: Modality
    metric: QualityMetric
    window: Interval
    value: Indicator

    @property
    def is_available(self) -> bool:
        return self.value.is_available

    @property
    def reason(self) -> UnavailabilityReason | None:
        return self.value.reason if isinstance(self.value, Unavailable) else None


@dataclass(frozen=True, slots=True)
class ModalityAvailability:
    """Whether a modality can support its indicators over a window.

    This is the object QA-02 is written against: video frames stop for twenty
    seconds, and the response is a quality warning plus affected indicators
    marked unavailable - with the speech side untouched. Holding availability
    per modality rather than per session is what makes "untouched" possible.
    """

    modality: Modality
    window: Interval
    is_usable: bool
    reason: UnavailabilityReason | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.is_usable and self.reason is None:
            raise QualityViolation(
                f"{self.modality.value} was marked unusable without a reason; "
                "FR-025 requires the reason to travel with the absence"
            )
        if self.is_usable and self.reason is not None:
            raise QualityViolation(
                f"{self.modality.value} is usable but carries an unavailability reason"
            )

    @classmethod
    def usable(cls, modality: Modality, window: Interval) -> ModalityAvailability:
        return cls(modality=modality, window=window, is_usable=True)

    @classmethod
    def unusable(
        cls,
        modality: Modality,
        window: Interval,
        reason: UnavailabilityReason,
        detail: str = "",
    ) -> ModalityAvailability:
        return cls(
            modality=modality,
            window=window,
            is_usable=False,
            reason=reason,
            detail=detail,
        )

    def as_unavailable(self) -> Unavailable:
        """Project this into the ``Unavailable`` an indicator would carry."""
        # `reason` is guaranteed non-None by __post_init__ whenever `is_usable`
        # is False. Re-checked rather than asserted because `assert` disappears
        # under `python -O`, and this branch decides whether a number or a
        # reason reaches the published result.
        if self.is_usable or self.reason is None:
            raise QualityViolation(
                f"{self.modality.value} is usable over this window; "
                "there is no unavailability to propagate"
            )
        return Unavailable(reason=self.reason, detail=self.detail)


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Everything the engine knows about how good the input was.

    Assembled per run and published with the result, because NFR-014 makes
    explainability a property of the delivered evidence rather than of an
    internal log. A consumer that cannot see the quality context cannot tell a
    genuinely fluent speaker from a badly recorded one.
    """

    assessments: tuple[QualityAssessment, ...] = field(default_factory=tuple)
    availability: tuple[ModalityAvailability, ...] = field(default_factory=tuple)

    def for_modality(self, modality: Modality) -> tuple[QualityAssessment, ...]:
        return tuple(a for a in self.assessments if a.modality is modality)

    def is_usable(self, modality: Modality, window: Interval) -> bool:
        """Whether every availability record overlapping ``window`` says yes.

        Conservative on purpose: a window that overlaps any unusable stretch is
        treated as unusable. The alternative - averaging usable and unusable
        time - produces an indicator computed over an unknown fraction of the
        window, which is precisely the fabricated value FR-025 forbids.
        """
        overlapping = [
            record
            for record in self.availability
            if record.modality is modality and record.window.overlaps(window)
        ]
        if not overlapping:
            return False
        return all(record.is_usable for record in overlapping)

    def unavailability_for(self, modality: Modality, window: Interval) -> Unavailable | None:
        """The reason a modality cannot serve this window, if it cannot."""
        for record in self.availability:
            if (
                record.modality is modality
                and record.window.overlaps(window)
                and not record.is_usable
            ):
                return record.as_unavailable()
        if not any(r.modality is modality for r in self.availability):
            return Unavailable(reason=UnavailabilityReason.MODALITY_NOT_CAPTURED)
        return None

    def extended(
        self,
        assessments: Iterable[QualityAssessment] = (),
        availability: Iterable[ModalityAvailability] = (),
    ) -> QualityReport:
        return QualityReport(
            assessments=self.assessments + tuple(assessments),
            availability=self.availability + tuple(availability),
        )

    def summary(self) -> Mapping[str, float]:
        """Measured metrics as a flat map, for telemetry and dashboards.

        Unavailable metrics are omitted rather than defaulted. US-009 lets an
        administrator watch operational health without reading transcripts, and
        a dashboard silently plotting zeros for missing metrics would show a
        healthy-looking flat line during a total camera outage.
        """
        return {
            f"{a.modality.value}.{a.metric.value}": a.value.value
            for a in self.assessments
            if isinstance(a.value, Measured)
        }
