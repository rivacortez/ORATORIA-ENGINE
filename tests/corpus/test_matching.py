"""Matching decides the units every later coefficient is computed over.

Every mistake here propagates: a missed match becomes a disagreement, a wrong
match becomes a class confusion that never happened, and both look like the
annotators when they are the algorithm.
"""

from __future__ import annotations

import pytest

from corpus.agreement.matching import (
    DEFAULT_IOU_THRESHOLD,
    MatchCriterion,
    match,
)
from evidence_engine.domain.shared.taxonomy import SpeechEventType
from tests.corpus.conftest import annotation

PAUSE = SpeechEventType.FILLED_PAUSE
CUTOFF = SpeechEventType.CUT_OFF


def test_identical_annotations_match() -> None:
    left = [annotation(PAUSE, 1_000, 1_800, "ana")]
    right = [annotation(PAUSE, 1_000, 1_800, "beto")]

    result = match(left, right)

    assert result.matched_count == 1
    assert result.pairs[0].iou == pytest.approx(1.0)
    assert result.pairs[0].same_class


def test_slightly_offset_annotations_still_match() -> None:
    """Boundary jitter between two humans is normal and is not a disagreement."""
    left = [annotation(PAUSE, 1_000, 1_800, "ana")]
    right = [annotation(PAUSE, 1_080, 1_850, "beto")]

    result = match(left, right)

    assert result.matched_count == 1
    assert result.pairs[0].start_error_ms == 80
    assert result.pairs[0].end_error_ms == 50


def test_barely_overlapping_annotations_do_not_match() -> None:
    """A 200 ms event and a 3 s one are not the same event."""
    left = [annotation(PAUSE, 1_000, 1_200, "ana")]
    right = [annotation(PAUSE, 1_100, 4_000, "beto")]

    result = match(left, right)

    assert result.matched_count == 0
    assert len(result.left_only) == 1
    assert len(result.right_only) == 1


def test_matching_is_class_agnostic() -> None:
    """Finding the same event and labelling it the same are separate questions.

    Collapsing them would make a class disagreement indistinguishable from a
    missed event, and the two have completely different fixes: one is a manual
    problem, the other is a training problem.
    """
    left = [annotation(PAUSE, 1_000, 1_800, "ana")]
    right = [annotation(CUTOFF, 1_000, 1_800, "beto")]

    result = match(left, right)

    assert result.matched_count == 1
    assert result.pairs[0].same_class is False


def test_matching_does_not_cross_in_time() -> None:
    """A match that reorders events in time asserts something absurd.

    Ana's first event overlaps Beto's second slightly and vice versa. A greedy
    or unconstrained matcher could pair them crosswise; a non-crossing one
    cannot, because events happen in an order.
    """
    left = [
        annotation(PAUSE, 1_000, 2_000, "ana"),
        annotation(PAUSE, 3_000, 4_000, "ana"),
    ]
    right = [
        annotation(PAUSE, 1_100, 2_100, "beto"),
        annotation(PAUSE, 3_100, 4_100, "beto"),
    ]

    result = match(left, right)

    assert result.matched_count == 2
    assert result.pairs[0].left.interval.start_ms == 1_000
    assert result.pairs[0].right.interval.start_ms == 1_100
    assert result.pairs[1].left.interval.start_ms == 3_000
    assert result.pairs[1].right.interval.start_ms == 3_100


def test_the_best_overlap_wins_when_two_candidates_compete() -> None:
    """Optimal, not first-come. A greedy matcher would take the earlier one."""
    left = [annotation(PAUSE, 1_000, 2_000, "ana")]
    right = [
        annotation(PAUSE, 900, 1_600, "beto"),  # IoU 0.545
        annotation(PAUSE, 990, 2_010, "beto"),  # IoU 0.980
    ]

    result = match(left, right)

    assert result.matched_count == 1
    assert result.pairs[0].right.interval.start_ms == 990


def test_matching_is_deterministic_regardless_of_input_order() -> None:
    """Two runs over the same data must produce the same report."""
    left = [
        annotation(PAUSE, 3_000, 4_000, "ana"),
        annotation(PAUSE, 1_000, 2_000, "ana"),
    ]
    right = [
        annotation(PAUSE, 3_050, 4_050, "beto"),
        annotation(PAUSE, 1_050, 2_050, "beto"),
    ]

    forward = match(left, right)
    backward = match(list(reversed(left)), list(reversed(right)))

    assert [p.left.interval.start_ms for p in forward.pairs] == [
        p.left.interval.start_ms for p in backward.pairs
    ]


def test_an_unmatched_event_on_each_side_is_reported_separately() -> None:
    left = [annotation(PAUSE, 1_000, 2_000, "ana"), annotation(PAUSE, 8_000, 9_000, "ana")]
    right = [annotation(PAUSE, 1_050, 2_050, "beto"), annotation(PAUSE, 20_000, 21_000, "beto")]

    result = match(left, right)

    assert result.matched_count == 1
    assert result.left_only[0].interval.start_ms == 8_000
    assert result.right_only[0].interval.start_ms == 20_000


# ---------------------------------------------------------------------------
# Positive specific agreement
# ---------------------------------------------------------------------------


def test_perfect_agreement_scores_one() -> None:
    left = [annotation(PAUSE, 1_000, 2_000, "ana")]
    right = [annotation(PAUSE, 1_000, 2_000, "beto")]

    assert match(left, right).positive_specific_agreement() == pytest.approx(1.0)


def test_two_empty_annotations_agree() -> None:
    """Agreeing that a stretch is empty is agreement, not an absence of it."""
    assert match([], []).positive_specific_agreement() == pytest.approx(1.0)


def test_no_overlap_at_all_scores_zero() -> None:
    left = [annotation(PAUSE, 1_000, 2_000, "ana")]
    right = [annotation(PAUSE, 30_000, 31_000, "beto")]

    assert match(left, right).positive_specific_agreement() == pytest.approx(0.0)


def test_agreement_is_symmetric() -> None:
    """Neither annotator is the reference; swapping them cannot change it."""
    left = [annotation(PAUSE, 1_000, 2_000, "ana"), annotation(PAUSE, 5_000, 6_000, "ana")]
    right = [annotation(PAUSE, 1_050, 2_050, "beto")]

    assert match(left, right).positive_specific_agreement() == pytest.approx(
        match(right, left).positive_specific_agreement()
    )


def test_one_match_out_of_three_events() -> None:
    """2*1 / (2*1 + 1 + 1) = 0.5"""
    left = [annotation(PAUSE, 1_000, 2_000, "ana"), annotation(PAUSE, 5_000, 6_000, "ana")]
    right = [annotation(PAUSE, 1_050, 2_050, "beto"), annotation(PAUSE, 40_000, 41_000, "beto")]

    assert match(left, right).positive_specific_agreement() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The criterion is a decision, and the report says which one was used
# ---------------------------------------------------------------------------


def test_the_tolerance_criterion_accepts_what_iou_refuses() -> None:
    """Documented divergence, not a bug: the two rules answer differently.

    A 200 ms annotation centred inside a 3 s one passes a midpoint test and
    fails IoU. Which is right is a methodological choice, so both exist and the
    report prints which was used.
    """
    left = [annotation(PAUSE, 1_400, 1_600, "ana")]
    right = [annotation(PAUSE, 100, 2_900, "beto")]

    assert match(left, right, criterion=MatchCriterion.IOU).matched_count == 0
    assert match(left, right, criterion=MatchCriterion.MIDPOINT_TOLERANCE).matched_count == 1


def test_a_stricter_threshold_matches_less() -> None:
    left = [annotation(PAUSE, 1_000, 2_000, "ana")]
    right = [annotation(PAUSE, 1_400, 2_400, "beto")]  # IoU = 0.6/1.4 = 0.43

    assert match(left, right, iou_threshold=0.4).matched_count == 1
    assert match(left, right, iou_threshold=DEFAULT_IOU_THRESHOLD).matched_count == 0


def test_the_matching_records_the_criterion_it_used() -> None:
    """A coefficient without its threshold is not reproducible."""
    result = match([], [], criterion=MatchCriterion.IOU, iou_threshold=0.7)

    assert result.criterion is MatchCriterion.IOU
    assert result.threshold == pytest.approx(0.7)
