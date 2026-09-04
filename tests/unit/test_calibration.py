"""Calibration: mapping raw model scores onto a probability scale.

This module carries a rule that is easy to state and easy to lose: an
uncalibrated score never clears a publication threshold. FR-022 gates the
optional visual classes behind confidence, §14.2 gates model promotion behind
calibration being reported, and both become theatre if a raw softmax maximum
can open the gate by being large.

The curve tests are here for a related reason. A calibration curve that is not
monotone is not a calibration curve — it would map a higher raw score to a
lower probability somewhere, which is incoherent — and a fit that produced one
is a bug in the fitting, not an unusual input to tolerate.
"""

from __future__ import annotations

import pytest

from evidence_engine.application.services.calibration import (
    CalibrationCurve,
    Calibrator,
)
from evidence_engine.domain.shared.confidence import CalibrationState

#: A curve shaped like a real one: the detector is overconfident in the middle
#: of its range, so 0.8 raw maps to 0.6 calibrated.
OVERCONFIDENT = CalibrationCurve(points=((0.0, 0.0), (0.5, 0.2), (0.8, 0.6), (1.0, 0.95)))


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------


def test_a_curve_interpolates_between_its_points() -> None:
    # Halfway between (0.5, 0.2) and (0.8, 0.6).
    assert OVERCONFIDENT.apply(0.65) == pytest.approx(0.4, abs=1e-6)


def test_a_curve_returns_its_endpoints_exactly() -> None:
    assert OVERCONFIDENT.apply(0.0) == pytest.approx(0.0)
    assert OVERCONFIDENT.apply(1.0) == pytest.approx(0.95)


def test_a_curve_clamps_outside_its_fitted_range() -> None:
    """A runtime can return a score slightly outside [0, 1] from accumulation.

    Clamping rather than extrapolating: outside the fitted range there is no
    evidence about what the probability should be, and extrapolating a linear
    segment past the data is inventing one.
    """
    assert OVERCONFIDENT.apply(-0.4) == pytest.approx(0.0)
    assert OVERCONFIDENT.apply(3.0) == pytest.approx(0.95)


def test_a_curve_needs_at_least_two_points() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        CalibrationCurve(points=((0.5, 0.5),))


def test_a_curve_must_be_sorted_by_raw_score() -> None:
    with pytest.raises(ValueError, match="sorted by raw score"):
        CalibrationCurve(points=((0.8, 0.6), (0.2, 0.1)))


def test_a_curve_must_be_monotone() -> None:
    """A higher raw score mapping to a lower probability is incoherent."""
    with pytest.raises(ValueError, match="monotone"):
        CalibrationCurve(points=((0.0, 0.9), (1.0, 0.1)))


def test_calibrated_probabilities_stay_in_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CalibrationCurve(points=((0.0, 0.0), (1.0, 1.4)))


def test_a_flat_segment_is_allowed() -> None:
    """Monotone *non-decreasing*: a detector can be uninformative over a range."""
    flat = CalibrationCurve(points=((0.0, 0.3), (0.5, 0.3), (1.0, 0.9)))

    assert flat.apply(0.25) == pytest.approx(0.3)


def test_a_repeated_raw_score_resolves_without_dividing_by_zero() -> None:
    """A step in the fit. Resolving it either way is arbitrary; crashing is not.

    Two cases, because they take different branches. When the duplicate is an
    interior point the earlier segment brackets the value first; when it is the
    leading point the degenerate segment is reached directly, and that is the
    one that would divide by zero without the guard.
    """
    interior = CalibrationCurve(points=((0.0, 0.1), (0.5, 0.2), (0.5, 0.8), (1.0, 0.9)))
    assert interior.apply(0.5) == pytest.approx(0.2)

    leading = CalibrationCurve(points=((0.5, 0.2), (0.5, 0.8), (1.0, 0.9)))
    assert leading.apply(0.5) == pytest.approx(0.8)
    # ... and the clamp routes anything below the fitted range into it too.
    assert leading.apply(0.1) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# The calibrator
# ---------------------------------------------------------------------------


def test_a_class_with_a_curve_comes_back_calibrated() -> None:
    calibrator = Calibrator(curves={"filled_pause": OVERCONFIDENT})

    confidence = calibrator.calibrate("filled_pause", 0.8)

    assert confidence.state is CalibrationState.CALIBRATED
    assert confidence.value == pytest.approx(0.6)


def test_a_class_without_a_curve_comes_back_raw() -> None:
    """The honest position for a class nobody has evaluated yet."""
    calibrator = Calibrator()

    confidence = calibrator.calibrate("self_touch", 0.93)

    assert confidence.state is CalibrationState.RAW
    assert confidence.value == pytest.approx(0.93)


def test_an_uncalibrated_score_never_clears_a_threshold() -> None:
    """The rule this whole module exists for.

    0.93 is comfortably above any threshold anyone would set, and it still does
    not open the gate, because it is not a probability of correctness - it is a
    number from a detector nobody has measured.
    """
    raw = Calibrator().calibrate("self_touch", 0.93)

    assert raw.value == pytest.approx(0.93)
    assert raw.meets(0.75) is False
    assert raw.meets(0.5) is False
    assert raw.meets(0.0) is False


def test_a_calibrated_score_clears_a_threshold_it_actually_exceeds() -> None:
    calibrated = Calibrator(curves={"self_touch": OVERCONFIDENT}).calibrate("self_touch", 1.0)

    assert calibrated.meets(0.75) is True
    assert calibrated.meets(0.99) is False


def test_calibration_can_close_a_gate_the_raw_score_would_have_opened() -> None:
    """The point of calibrating: 0.8 raw looks confident and means 0.6."""
    calibrator = Calibrator(curves={"self_touch": OVERCONFIDENT})

    assert calibrator.calibrate("self_touch", 0.8).meets(0.75) is False


def test_scores_outside_the_unit_interval_are_clamped_not_refused() -> None:
    """Floating-point accumulation in a vendor runtime is not our bug to raise on."""
    calibrator = Calibrator()

    assert calibrator.calibrate("word", 1.0000001).value == pytest.approx(1.0)
    assert calibrator.calibrate("word", -0.0000001).value == pytest.approx(0.0)


def test_the_calibrator_reports_which_classes_it_can_calibrate() -> None:
    """Used by the model-promotion gate: §14.2 wants calibration *reported*."""
    calibrator = Calibrator(curves={"filled_pause": OVERCONFIDENT})

    assert calibrator.has_curve_for("filled_pause") is True
    assert calibrator.has_curve_for("lexical_filler") is False
