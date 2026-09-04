"""Agreement coefficients, checked against values computed by hand.

Coefficients are the easiest thing in this repository to get subtly wrong and
the hardest to notice: a kappa implemented with the wrong marginals still
returns a plausible number between 0 and 1, and the mistake surfaces as an
unexpected finding in a thesis rather than as a failure.

So the tests below carry their arithmetic. Where a test asserts 0.6, the
working is in the docstring and can be checked without running anything.
"""

from __future__ import annotations

import pytest

from corpus.agreement.measures import (
    BoundaryError,
    UndefinedUnitError,
    cohens_kappa,
    confusion,
    kappa_over_time_frames,
    krippendorffs_alpha,
)

#: 20 matched events, 16 agreements, marginals balanced at 10/10 on both sides.
#:
#:   p_o = 16/20 = 0.8
#:   p_e = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
#:   kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6
BALANCED = [("a", "a")] * 8 + [("b", "b")] * 8 + [("a", "b")] * 2 + [("b", "a")] * 2


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def test_kappa_matches_the_hand_computed_value() -> None:
    assert cohens_kappa(confusion(BALANCED)) == pytest.approx(0.6)


def test_perfect_agreement_gives_kappa_one() -> None:
    matrix = confusion([("a", "a")] * 10 + [("b", "b")] * 10)

    assert cohens_kappa(matrix) == pytest.approx(1.0)


def test_chance_level_agreement_gives_kappa_zero() -> None:
    """Every cell equal: observed agreement is exactly what chance predicts.

    p_o = 0.5, marginals 0.5/0.5 on both sides, so p_e = 0.5 and kappa = 0.
    """
    matrix = confusion([("a", "a")] * 5 + [("a", "b")] * 5 + [("b", "a")] * 5 + [("b", "b")] * 5)

    assert cohens_kappa(matrix) == pytest.approx(0.0)


def test_systematic_disagreement_goes_negative() -> None:
    """Worse than chance is a real result, not a floor at zero."""
    matrix = confusion([("a", "b")] * 10 + [("b", "a")] * 10)

    kappa = cohens_kappa(matrix)
    assert kappa is not None
    assert kappa < 0


def test_kappa_is_undefined_when_everything_is_one_category() -> None:
    """Two annotators who both said `filled_pause` everywhere agreed perfectly.

    Expected agreement is 1, so the denominator vanishes. Returning 0 there
    would read as "no better than chance", which is the opposite of what
    happened; `None` says the coefficient cannot answer.
    """
    assert cohens_kappa(confusion([("a", "a")] * 30)) is None


def test_kappa_is_undefined_with_a_single_matched_event() -> None:
    assert cohens_kappa(confusion([("a", "a")])) is None


def test_kappa_is_symmetric() -> None:
    swapped = [(b, a) for a, b in BALANCED]

    assert cohens_kappa(confusion(BALANCED)) == pytest.approx(cohens_kappa(confusion(swapped)))


# ---------------------------------------------------------------------------
# Krippendorff's alpha (nominal, over matched events)
# ---------------------------------------------------------------------------


def test_alpha_matches_the_hand_computed_value() -> None:
    """Coincidence matrix from BALANCED, each pair counted in both directions:

    o_aa = 16, o_bb = 16, o_ab = 4, o_ba = 4      n = 40
    marginals n_a = 20, n_b = 20
    observed disagreement  = 4 + 4 = 8
    expected disagreement  = 20*20 + 20*20 = 800
    alpha = 1 - (40 - 1) * 8 / 800 = 0.61
    """
    assert krippendorffs_alpha(confusion(BALANCED)) == pytest.approx(0.61)


def test_perfect_agreement_gives_alpha_one() -> None:
    matrix = confusion([("a", "a")] * 10 + [("b", "b")] * 10)

    assert krippendorffs_alpha(matrix) == pytest.approx(1.0)


def test_alpha_and_kappa_stay_close_on_the_same_data() -> None:
    """They answer nearly the same question and differ by a sample correction.

    A large gap between them means one of the two implementations is wrong, so
    this is a cross-check rather than a property of the data.
    """
    kappa = cohens_kappa(confusion(BALANCED))
    alpha = krippendorffs_alpha(confusion(BALANCED))

    assert kappa is not None
    assert alpha is not None
    assert abs(kappa - alpha) < 0.05


def test_alpha_is_undefined_when_everything_is_one_category() -> None:
    assert krippendorffs_alpha(confusion([("a", "a")] * 30)) is None


def test_alpha_is_symmetric() -> None:
    swapped = [(b, a) for a, b in BALANCED]

    assert krippendorffs_alpha(confusion(BALANCED)) == pytest.approx(
        krippendorffs_alpha(confusion(swapped))
    )


# ---------------------------------------------------------------------------
# The confusion matrix and per-class agreement
# ---------------------------------------------------------------------------


def test_specific_agreement_counts_no_true_negatives() -> None:
    """8 both, 2 left-only, 2 right-only: 2*8 / (2*8 + 2 + 2) = 0.8"""
    matrix = confusion(BALANCED)

    assert matrix.specific_agreement("a") == pytest.approx(0.8)


def test_a_class_nobody_used_has_no_agreement_to_report() -> None:
    """`None`, not 0.0 or 1.0 - either would read as a finding about the class."""
    assert confusion(BALANCED).specific_agreement("prolongation") is None


def test_a_class_only_one_annotator_used_scores_zero() -> None:
    matrix = confusion([("a", "b")] * 4)

    assert matrix.specific_agreement("a") == pytest.approx(0.0)


def test_the_matrix_reports_its_own_totals() -> None:
    matrix = confusion(BALANCED)

    assert matrix.total == 20
    assert matrix.agreements == 16
    assert matrix.observed_agreement() == pytest.approx(0.8)
    assert matrix.labels == ("a", "b")


# ---------------------------------------------------------------------------
# Boundary error
# ---------------------------------------------------------------------------


def test_median_of_an_even_sample_is_the_midpoint() -> None:
    error = BoundaryError(start_errors_ms=(10, 20, 30, 40), end_errors_ms=())

    assert error.median_start_ms() == pytest.approx(25.0)


def test_median_of_an_odd_sample_is_an_observation() -> None:
    error = BoundaryError(start_errors_ms=(10, 20, 90), end_errors_ms=())

    assert error.median_start_ms() == pytest.approx(20.0)


def test_p95_is_nearest_rank_not_interpolated() -> None:
    """On the twenty events a pilot produces, an interpolated p95 invents a
    value between two observations and reads as more precise than the sample
    supports."""
    error = BoundaryError(start_errors_ms=tuple(range(1, 21)), end_errors_ms=())

    # ceil(0.95 * 20) = 19 -> the 19th smallest, which is 19.
    assert error.p95_start_ms() == pytest.approx(19.0)


def test_an_empty_sample_reports_nothing_rather_than_zero() -> None:
    error = BoundaryError()

    assert error.count == 0
    assert error.median_start_ms() is None
    assert error.p95_end_ms() is None


def test_start_and_end_are_measured_separately() -> None:
    """They fail differently: onsets are sharp, offsets trail into silence."""
    error = BoundaryError(start_errors_ms=(10, 12, 14), end_errors_ms=(200, 240, 280))

    assert error.median_start_ms() == pytest.approx(12.0)
    assert error.median_end_ms() == pytest.approx(240.0)


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


def test_kappa_over_time_frames_refuses_to_run() -> None:
    """The trap this module is built around, with somewhere to fail loudly.

    Discretising a recording and computing kappa per frame returns roughly 0.9
    on anything, because the no-event class dominates. A function that quietly
    did it would be the most dangerous thing in the package.
    """
    with pytest.raises(UndefinedUnitError, match="no-event class"):
        kappa_over_time_frames(frame_ms=10)
