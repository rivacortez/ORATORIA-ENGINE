"""Calibrated confidence.

Confidence travels with every derived event (FR-014, NFR-014). It is a value
object rather than a bare float so that two things cannot happen by accident:
a probability outside the unit interval, and an uncalibrated model score being
compared against a threshold as if it were a probability.

The second is the subtle one. A raw softmax maximum is not a probability of
correctness, and thresholding it as though it were is how a detector ends up
reporting 0.95 on classes it gets wrong half the time. ``Confidence`` therefore
records whether it has been through calibration, and the abstention rule
refuses to certify an uncalibrated value as high-confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evidence_engine.domain.shared.errors import InvalidConfidence


class CalibrationState(StrEnum):
    """Whether a score has been mapped onto a calibrated probability scale.

    ``RAW`` is the honest default for a fresh model output. Model-level DoD
    §14.2 requires confidence calibration to be reported before a version can
    be promoted, so a production model should be emitting ``CALIBRATED``; the
    ``RAW`` path exists for baselines and for experiments that have not been
    calibrated yet, and is marked as such rather than hidden.
    """

    RAW = "raw"
    CALIBRATED = "calibrated"


@dataclass(frozen=True, slots=True, order=True)
class Confidence:
    """A score in ``[0, 1]`` that knows whether it has been calibrated."""

    value: float
    state: CalibrationState = CalibrationState.RAW

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise InvalidConfidence(f"confidence must lie in [0, 1], got {self.value}")

    @classmethod
    def calibrated(cls, value: float) -> Confidence:
        return cls(value, CalibrationState.CALIBRATED)

    @classmethod
    def raw(cls, value: float) -> Confidence:
        return cls(value, CalibrationState.RAW)

    @property
    def is_calibrated(self) -> bool:
        return self.state is CalibrationState.CALIBRATED

    def meets(self, threshold: float) -> bool:
        """Whether this confidence clears a publication threshold.

        An uncalibrated score never clears one. That is stricter than the
        arithmetic comparison and it is the point: FR-022 gates optional
        behaviours behind confidence thresholds, and clearing such a gate with
        a number that has no probabilistic meaning would make the gate
        theatre. An uncalibrated detector can still emit its finding - it just
        emits it as uncertain rather than as confirmed.
        """
        return self.is_calibrated and self.value >= threshold
