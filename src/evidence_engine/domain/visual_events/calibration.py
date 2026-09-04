"""Per-speaker visual calibration.

FR-021 measures posture deviation "relative to calibration" and FR-019 estimates
gaze "relative to a calibrated camera reference". Both phrases are doing real
work. There is no universal correct posture, and a camera on a laptop below eye
level makes every speaker look downward; measuring either against a fixed ideal
would produce a metric that tracks furniture rather than behaviour, and §17
warns that varying camera quality is exactly how visual metrics become biased.

Calibration is captured once, during the preflight of US-001, and stored with
the session's configuration snapshot so that a rerun reproduces it (NFR-015).
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.domain.shared.errors import DomainError


class CalibrationViolation(DomainError):
    """A calibration was built with impossible geometry."""


#: Half-angle of the cone counted as "toward the camera", in degrees. A
#: threshold rather than an exact ray because the estimate has real error and
#: FR-019 does not promise otherwise. Configurable and versioned (FR-015): this
#: is the default a snapshot starts from, not a constant of the world.
DEFAULT_GAZE_CONE_DEGREES = 20.0

#: Torso deviation beyond which a posture event is emitted, in degrees from the
#: calibrated baseline.
DEFAULT_POSTURE_DEVIATION_DEGREES = 15.0

#: Minimum landmark visibility for a frame to be usable at all.
DEFAULT_MIN_LANDMARK_VISIBILITY = 0.5


@dataclass(frozen=True, slots=True)
class VisualCalibration:
    """The speaker's own reference frame, captured before capture begins.

    ``baseline_*`` values come from the preflight sample. If preflight was
    skipped or failed, there is no calibration and every relative indicator is
    unavailable with a reason - the alternative, substituting a population
    average, would silently score a speaker against strangers.
    """

    #: Yaw and pitch of the head, in degrees, when the speaker faced the camera
    #: during preflight. Not assumed to be (0, 0): laptop cameras sit low.
    baseline_yaw_degrees: float
    baseline_pitch_degrees: float
    #: Torso inclination at rest, in degrees.
    baseline_torso_degrees: float
    gaze_cone_degrees: float = DEFAULT_GAZE_CONE_DEGREES
    posture_deviation_degrees: float = DEFAULT_POSTURE_DEVIATION_DEGREES
    min_landmark_visibility: float = DEFAULT_MIN_LANDMARK_VISIBILITY

    def __post_init__(self) -> None:
        if not 0.0 < self.gaze_cone_degrees < 90.0:
            raise CalibrationViolation(
                f"gaze cone must lie in (0, 90) degrees, got {self.gaze_cone_degrees}"
            )
        if self.posture_deviation_degrees <= 0.0:
            raise CalibrationViolation(
                f"posture threshold must be positive, got {self.posture_deviation_degrees}"
            )
        if not 0.0 <= self.min_landmark_visibility <= 1.0:
            raise CalibrationViolation(
                f"landmark visibility must lie in [0, 1], got {self.min_landmark_visibility}"
            )

    def is_within_gaze_cone(self, yaw_degrees: float, pitch_degrees: float) -> bool:
        """Whether a head pose falls inside the speaker's camera-facing cone.

        Compared against the speaker's own baseline, so a low camera does not
        turn everyone into someone who never makes eye contact.
        """
        yaw_offset = abs(yaw_degrees - self.baseline_yaw_degrees)
        pitch_offset = abs(pitch_degrees - self.baseline_pitch_degrees)
        return max(yaw_offset, pitch_offset) <= self.gaze_cone_degrees

    def posture_deviation(self, torso_degrees: float) -> float:
        """Degrees away from this speaker's resting posture."""
        return abs(torso_degrees - self.baseline_torso_degrees)

    def exceeds_posture_threshold(self, torso_degrees: float) -> bool:
        return self.posture_deviation(torso_degrees) > self.posture_deviation_degrees
