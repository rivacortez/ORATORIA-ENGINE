"""Turning raw model scores into calibrated confidence.

§5 gives confidence calibration one responsibility - calibrate and propagate
uncertainty - and one prohibition: do not turn unavailable values into zero.
This module implements the first; ``domain.shared.measurement`` enforces the
second.

The problem being solved is specific. A softmax maximum is not a probability of
correctness. A detector that outputs 0.95 on a class it gets right 60% of the
time is not lying, it is uncalibrated, and thresholding that number as though
it were a probability is how a precision-focused gate (FR-022) ends up letting
through most of what it was built to stop. §14.2 therefore requires calibration
to be reported before a model version can be promoted.

Calibration itself is fitted offline against a held-out set and arrives here as
a table. What this module does is apply it, and - importantly - refuse to
pretend when there is no table for a class.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise

from evidence_engine.domain.shared.confidence import Confidence


@dataclass(frozen=True, slots=True)
class CalibrationCurve:
    """A monotone piecewise-linear map from raw score to probability.

    Piecewise-linear isotonic regression rather than Platt scaling because
    neural detector scores are rarely sigmoid-shaped, and a two-parameter fit
    that cannot express the shape will look calibrated in aggregate while
    staying wrong at the ends - which is where the thresholds live.

    ``points`` are ``(raw_score, calibrated_probability)`` pairs sorted by raw
    score, taken from the fit on the held-out set.
    """

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("a calibration curve needs at least two points")
        raws = [p[0] for p in self.points]
        if raws != sorted(raws):
            raise ValueError("calibration points must be sorted by raw score")
        probs = [p[1] for p in self.points]
        if probs != sorted(probs):
            raise ValueError("a calibration curve must be monotone non-decreasing")
        if not all(0.0 <= p <= 1.0 for p in probs):
            raise ValueError("calibrated probabilities must lie in [0, 1]")

    def apply(self, raw_score: float) -> float:
        """Map a raw score onto the calibrated scale, clamped at the ends."""
        clamped = min(max(raw_score, self.points[0][0]), self.points[-1][0])
        for (x0, y0), (x1, y1) in pairwise(self.points):
            if x0 <= clamped <= x1:
                if x1 == x0:
                    return y1
                ratio = (clamped - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return self.points[-1][1]


@dataclass(frozen=True, slots=True)
class Calibrator:
    """Applies per-class curves, and says so when it has none.

    A class with no fitted curve yields ``Confidence.raw``. That value still
    travels with the event - FR-014 requires a confidence on everything, and
    withholding it would be worse - but it will not clear a publication
    threshold, because ``Confidence.meets`` refuses uncalibrated scores. The
    effect is that an uncalibrated detector can report findings as uncertain
    and cannot report them as confirmed, which is the honest position for a
    class nobody has evaluated yet.
    """

    curves: Mapping[str, CalibrationCurve] = field(default_factory=dict)

    def calibrate(self, class_identifier: str, raw_score: float) -> Confidence:
        curve = self.curves.get(class_identifier)
        if curve is None:
            return Confidence.raw(_clamp_unit(raw_score))
        return Confidence.calibrated(_clamp_unit(curve.apply(raw_score)))

    def has_curve_for(self, class_identifier: str) -> bool:
        return class_identifier in self.curves


def _clamp_unit(value: float) -> float:
    """Keep a score inside ``[0, 1]``.

    Runtimes occasionally return values slightly outside the unit interval from
    floating-point accumulation. Clamping is right here and only here: the
    domain's ``Confidence`` refuses out-of-range values outright, and this is
    the boundary where a vendor's arithmetic stops being our problem.
    """
    return min(max(value, 0.0), 1.0)
