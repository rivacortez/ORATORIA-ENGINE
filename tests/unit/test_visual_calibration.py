"""Per-speaker visual calibration.

FR-019 estimates gaze "relative to a calibrated camera reference" and FR-021
measures posture deviation "relative to calibration". Both phrases are doing
work, and the tests below are about what happens when they are ignored.

A laptop camera sits below eye level. Measured against an absolute forward
ray, every speaker using one looks downward, and the engine would report a
population-wide gaze problem that is actually a furniture problem — §17's
"camera quality varies" risk, arriving as a systematic bias rather than as
noise. Calibrating against the speaker's own preflight baseline is what removes
it.
"""

from __future__ import annotations

import pytest

from evidence_engine.domain.visual_events.calibration import (
    DEFAULT_GAZE_CONE_DEGREES,
    DEFAULT_MIN_LANDMARK_VISIBILITY,
    DEFAULT_POSTURE_DEVIATION_DEGREES,
    CalibrationViolation,
    VisualCalibration,
)

#: A speaker in front of a laptop: the camera is below eye level, so facing it
#: reads as -12 degrees of pitch. This is the ordinary case, not an edge one.
LAPTOP = VisualCalibration(
    baseline_yaw_degrees=0.0,
    baseline_pitch_degrees=-12.0,
    baseline_torso_degrees=4.0,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_the_defaults_are_the_documented_ones() -> None:
    assert LAPTOP.gaze_cone_degrees == DEFAULT_GAZE_CONE_DEGREES
    assert LAPTOP.posture_deviation_degrees == DEFAULT_POSTURE_DEVIATION_DEGREES
    assert LAPTOP.min_landmark_visibility == DEFAULT_MIN_LANDMARK_VISIBILITY


@pytest.mark.parametrize("cone", [0.0, -5.0, 90.0, 180.0])
def test_a_gaze_cone_outside_zero_to_ninety_is_refused(cone: float) -> None:
    """A zero cone matches nothing; ninety or more matches everything.

    Both make the indicator meaningless while still producing numbers, which is
    the worst failure mode available: it looks like it is working.
    """
    with pytest.raises(CalibrationViolation, match="gaze cone"):
        VisualCalibration(
            baseline_yaw_degrees=0.0,
            baseline_pitch_degrees=0.0,
            baseline_torso_degrees=0.0,
            gaze_cone_degrees=cone,
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0])
def test_a_non_positive_posture_threshold_is_refused(threshold: float) -> None:
    """Zero would make every frame a deviation from itself."""
    with pytest.raises(CalibrationViolation, match="posture threshold"):
        VisualCalibration(
            baseline_yaw_degrees=0.0,
            baseline_pitch_degrees=0.0,
            baseline_torso_degrees=0.0,
            posture_deviation_degrees=threshold,
        )


@pytest.mark.parametrize("visibility", [-0.1, 1.1])
def test_landmark_visibility_stays_in_the_unit_interval(visibility: float) -> None:
    with pytest.raises(CalibrationViolation, match=r"\[0, 1\]"):
        VisualCalibration(
            baseline_yaw_degrees=0.0,
            baseline_pitch_degrees=0.0,
            baseline_torso_degrees=0.0,
            min_landmark_visibility=visibility,
        )


@pytest.mark.parametrize("visibility", [0.0, 1.0])
def test_the_visibility_bounds_themselves_are_allowed(visibility: float) -> None:
    """Closed interval: 0 accepts everything, 1 accepts only perfect detection."""
    calibration = VisualCalibration(
        baseline_yaw_degrees=0.0,
        baseline_pitch_degrees=0.0,
        baseline_torso_degrees=0.0,
        min_landmark_visibility=visibility,
    )

    assert calibration.min_landmark_visibility == pytest.approx(visibility)


# ---------------------------------------------------------------------------
# Gaze
# ---------------------------------------------------------------------------


def test_facing_the_speakers_own_baseline_counts_as_facing_the_camera() -> None:
    assert LAPTOP.is_within_gaze_cone(yaw_degrees=0.0, pitch_degrees=-12.0) is True


def test_a_low_camera_does_not_make_everyone_look_away() -> None:
    """The bug this whole module exists to prevent.

    A pitch of -12 is outside a 20-degree cone measured from zero, and inside
    it once measured from the speaker's own baseline. Without calibration this
    speaker is reported as never facing the camera for the whole session.
    """
    uncalibrated = VisualCalibration(
        baseline_yaw_degrees=0.0, baseline_pitch_degrees=0.0, baseline_torso_degrees=0.0
    )

    assert uncalibrated.is_within_gaze_cone(0.0, -25.0) is False
    assert LAPTOP.is_within_gaze_cone(0.0, -25.0) is True


def test_a_genuine_look_away_is_still_detected() -> None:
    """Calibration must not swallow the signal it exists to measure."""
    assert LAPTOP.is_within_gaze_cone(yaw_degrees=0.0, pitch_degrees=-45.0) is False
    assert LAPTOP.is_within_gaze_cone(yaw_degrees=60.0, pitch_degrees=-12.0) is False


def test_the_cone_boundary_is_inclusive() -> None:
    assert LAPTOP.is_within_gaze_cone(yaw_degrees=20.0, pitch_degrees=-12.0) is True
    assert LAPTOP.is_within_gaze_cone(yaw_degrees=20.001, pitch_degrees=-12.0) is False


def test_the_cone_is_bounded_on_the_worse_of_the_two_axes() -> None:
    """Both inside is inside; either outside is outside.

    A speaker turned 40 degrees sideways is not facing the camera no matter how
    correct their pitch is, so the axes are combined with max, not averaged.
    """
    assert LAPTOP.is_within_gaze_cone(yaw_degrees=19.0, pitch_degrees=5.0) is True
    assert LAPTOP.is_within_gaze_cone(yaw_degrees=40.0, pitch_degrees=-12.0) is False


def test_a_narrower_cone_is_stricter() -> None:
    strict = VisualCalibration(
        baseline_yaw_degrees=0.0,
        baseline_pitch_degrees=-12.0,
        baseline_torso_degrees=0.0,
        gaze_cone_degrees=5.0,
    )

    assert LAPTOP.is_within_gaze_cone(10.0, -12.0) is True
    assert strict.is_within_gaze_cone(10.0, -12.0) is False


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------


def test_deviation_is_measured_from_the_speakers_resting_posture() -> None:
    """Someone who rests at 4 degrees is not deviating by 4 degrees."""
    assert LAPTOP.posture_deviation(4.0) == pytest.approx(0.0)
    assert LAPTOP.posture_deviation(24.0) == pytest.approx(20.0)


def test_deviation_is_unsigned() -> None:
    """Leaning either way is a deviation; the direction is a separate question."""
    assert LAPTOP.posture_deviation(-16.0) == pytest.approx(20.0)
    assert LAPTOP.posture_deviation(24.0) == pytest.approx(20.0)


def test_the_posture_threshold_is_exclusive_at_the_boundary() -> None:
    """Exactly at the threshold is not yet a deviation."""
    at_threshold = 4.0 + DEFAULT_POSTURE_DEVIATION_DEGREES

    assert LAPTOP.exceeds_posture_threshold(at_threshold) is False
    assert LAPTOP.exceeds_posture_threshold(at_threshold + 0.1) is True


def test_a_small_shift_is_not_a_deviation() -> None:
    assert LAPTOP.exceeds_posture_threshold(9.0) is False


def test_a_calibration_is_immutable() -> None:
    """It is bound to a session's configuration snapshot (QA-05).

    A calibration that could be edited mid-session would make the second half
    of a presentation incomparable with the first.
    """
    with pytest.raises((AttributeError, TypeError)):
        LAPTOP.baseline_pitch_degrees = 0.0  # type: ignore[misc]
