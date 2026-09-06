"""Invariants of the matching, checked over many generated inputs.

The example-based tests in ``test_matching.py`` check behaviour the author
thought of. These check properties that must hold for *every* input, which is
the only way to catch the class of bug the review found: the matching was
order-dependent on ties, and no hand-written example happened to hit one.

Symmetry is the one that matters. `match(a, b)` and `match(b, a)` describe the
same pair of annotators, so a report whose confusion matrix and boundary errors
depend on which file was passed first is a report whose numbers depend on the
order somebody typed two filenames.
"""

from __future__ import annotations

import random

import pytest

from corpus.agreement.matching import (
    InvalidMatchParameters,
    MatchCriterion,
    Matching,
    match,
)
from corpus.schema.records import DisfluencyAnnotation
from evidence_engine.domain.shared.taxonomy import SpeechEventType
from tests.corpus.conftest import annotation

#: Acoustic classes only: they need no raw text or role, so a generator can
#: produce them freely without the schema refusing the combination.
CLASSES = (
    SpeechEventType.FILLED_PAUSE,
    SpeechEventType.CUT_OFF,
    SpeechEventType.PROLONGATION,
    SpeechEventType.SILENT_PAUSE,
)


def _generate(rng: random.Random, annotator: str, count: int) -> list[DisfluencyAnnotation]:
    """Non-overlapping annotations of one annotator, over a 60 s recording.

    Non-overlapping within an annotator because that is what the validator
    enforces for a single class, and generating impossible input would test the
    matching against data it will never see.
    """
    annotations: list[DisfluencyAnnotation] = []
    cursor = 0
    for _ in range(count):
        start = cursor + rng.randint(0, 800)
        duration = rng.randint(120, 1_400)
        if start + duration > 60_000:
            break
        annotations.append(annotation(rng.choice(CLASSES), start, start + duration, annotator))
        cursor = start + duration
    return annotations


#: One matched annotation, identified by content: start, end and class.
Side = tuple[int, int, str]


def _side(annotation: DisfluencyAnnotation) -> Side:
    return (
        annotation.interval.start_ms,
        annotation.interval.end_ms,
        annotation.event_type.value,
    )


def _as_pairs(matching: Matching) -> set[tuple[Side, Side]]:
    """The matching as an unordered set of (annotation, annotation) pairs.

    Compared by content rather than by index, because the indices are relative
    to each side and would differ under a swap even when the matching is
    identical. The class is part of the identity: two annotations can share an
    interval and differ only in label, and that is exactly the pair the old
    tie-break confused.
    """
    return {(_side(pair.left), _side(pair.right)) for pair in matching.pairs}


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(60))
def test_matching_is_symmetric_under_swapping_the_annotators(seed: int) -> None:
    """The bug the review found: ties were broken by "prefer the left side".

    Sixty seeds rather than one, because a tie needs two candidate paths of
    exactly equal total IoU and a single hand-written case will not produce one.
    """
    rng = random.Random(seed)
    left = _generate(rng, "ana", rng.randint(0, 12))
    right = _generate(rng, "beto", rng.randint(0, 12))

    forward = match(left, right)
    backward = match(right, left)

    assert forward.matched_count == backward.matched_count
    # Pairs compared with the sides normalised, so a genuine match reads the
    # same whichever direction it was computed in.
    assert _as_pairs(forward) == {(b, a) for a, b in _as_pairs(backward)}


@pytest.mark.parametrize("seed", range(60))
def test_agreement_is_symmetric_under_swapping(seed: int) -> None:
    rng = random.Random(seed)
    left = _generate(rng, "ana", rng.randint(0, 12))
    right = _generate(rng, "beto", rng.randint(0, 12))

    assert match(left, right).positive_specific_agreement() == pytest.approx(
        match(right, left).positive_specific_agreement()
    )


@pytest.mark.parametrize("seed", range(40))
def test_boundary_errors_are_symmetric_under_swapping(seed: int) -> None:
    """Start and end error feed NFR-004's human ceiling; they must not depend
    on argument order either."""
    rng = random.Random(seed)
    left = _generate(rng, "ana", rng.randint(1, 10))
    right = _generate(rng, "beto", rng.randint(1, 10))

    forward = sorted(p.start_error_ms for p in match(left, right).pairs)
    backward = sorted(p.start_error_ms for p in match(right, left).pairs)

    assert forward == backward


def test_the_known_asymmetric_case() -> None:
    """The exact input the old tie-break got wrong.

    Found by search rather than by intuition, then shrunk to four annotations.
    Worth keeping verbatim: the property tests above pass against the old code,
    because the backtrack is naturally symmetric *except* at an exact tie
    between two different optimal matchings, and independently generated
    annotations essentially never produce one.

    The tie is Beto's two annotations, which have identical IoU (900/1000) with
    Ana's silent pause. Ana's prolongation is not decoration - it is what forces
    the dynamic program through the cell where the tie has to be broken.

    Under the old rule the pair came out as

        silent_pause <-> filled_pause    reading Ana's file first
        silent_pause <-> prolongation    reading Beto's file first

    which is a *class* disagreement that changes identity depending on the
    order two filenames were typed. That is a confusion matrix nobody can
    defend, and it fails silently: both runs look perfectly ordinary.
    """
    ana = [
        annotation(SpeechEventType.PROLONGATION, 500, 1_500, "ana"),
        annotation(SpeechEventType.SILENT_PAUSE, 0, 1_000, "ana"),
    ]
    beto = [
        annotation(SpeechEventType.PROLONGATION, 0, 900, "beto"),
        annotation(SpeechEventType.FILLED_PAUSE, 100, 1_000, "beto"),
    ]

    forward = _as_pairs(match(ana, beto))
    backward = {(b, a) for a, b in _as_pairs(match(beto, ana))}

    assert forward == backward


def test_the_class_confusion_of_the_known_case_does_not_flip() -> None:
    """The same case, stated as the harm rather than as the mechanism."""
    ana = [
        annotation(SpeechEventType.PROLONGATION, 500, 1_500, "ana"),
        annotation(SpeechEventType.SILENT_PAUSE, 0, 1_000, "ana"),
    ]
    beto = [
        annotation(SpeechEventType.PROLONGATION, 0, 900, "beto"),
        annotation(SpeechEventType.FILLED_PAUSE, 100, 1_000, "beto"),
    ]

    forward = {(p.left.event_type.value, p.right.event_type.value) for p in match(ana, beto).pairs}
    backward = {(p.right.event_type.value, p.left.event_type.value) for p in match(beto, ana).pairs}

    assert forward == backward


# The interval grid the counterexample was found on. Small and repetitive on
# purpose: exact ties need candidate matchings of exactly equal total weight,
# which is why the realistic generator above never produces one.
GRID = ((0, 1_000), (0, 2_000), (500, 1_500), (1_000, 2_000), (1_000, 3_000), (2_000, 3_000))


def _generate_on_grid(rng: random.Random, annotator: str, count: int) -> list[DisfluencyAnnotation]:
    """Annotations drawn from a tiny grid, so exact ties are common.

    Not realistic input - annotations repeat and overlap. That is the point:
    this generator exists to stress the tie-breaking, and the tie-break has to
    hold on any input the type system admits, not only on plausible ones.
    """
    return [annotation(rng.choice(CLASSES), *rng.choice(GRID), annotator) for _ in range(count)]


@pytest.mark.parametrize("seed", range(200))
def test_symmetry_holds_on_inputs_engineered_to_produce_ties(seed: int) -> None:
    rng = random.Random(seed)
    left = _generate_on_grid(rng, "ana", rng.randint(1, 5))
    right = _generate_on_grid(rng, "beto", rng.randint(1, 5))

    forward = _as_pairs(match(left, right))
    backward = {(b, a) for a, b in _as_pairs(match(right, left))}

    assert forward == backward


# ---------------------------------------------------------------------------
# Determinism and consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_matching_is_stable_under_shuffling_the_input(seed: int) -> None:
    """A file lists annotations in whatever order the tool wrote them."""
    rng = random.Random(seed)
    left = _generate(rng, "ana", rng.randint(1, 10))
    right = _generate(rng, "beto", rng.randint(1, 10))

    baseline = _as_pairs(match(left, right))

    shuffled_left = list(left)
    shuffled_right = list(right)
    rng.shuffle(shuffled_left)
    rng.shuffle(shuffled_right)

    assert _as_pairs(match(shuffled_left, shuffled_right)) == baseline


@pytest.mark.parametrize("seed", range(40))
def test_every_annotation_is_matched_at_most_once(seed: int) -> None:
    """A one-to-one matching. Two pairs sharing an annotation would count one
    event twice in the confusion matrix."""
    rng = random.Random(seed)
    left = _generate(rng, "ana", rng.randint(1, 12))
    right = _generate(rng, "beto", rng.randint(1, 12))

    matching = match(left, right)

    left_used = [id(pair.left) for pair in matching.pairs]
    right_used = [id(pair.right) for pair in matching.pairs]
    assert len(left_used) == len(set(left_used))
    assert len(right_used) == len(set(right_used))


@pytest.mark.parametrize("seed", range(40))
def test_matched_and_unmatched_account_for_every_annotation(seed: int) -> None:
    """Nothing is silently dropped: an annotation lost here reads as a
    disagreement that never happened."""
    rng = random.Random(seed)
    left = _generate(rng, "ana", rng.randint(0, 12))
    right = _generate(rng, "beto", rng.randint(0, 12))

    matching = match(left, right)

    assert matching.matched_count + len(matching.left_only) == len(left)
    assert matching.matched_count + len(matching.right_only) == len(right)


@pytest.mark.parametrize("seed", range(30))
def test_a_file_matched_against_itself_matches_everything(seed: int) -> None:
    """Not a real comparison, but a property the algorithm must have: an
    annotation overlaps itself perfectly, so nothing can be left over."""
    rng = random.Random(seed)
    events = _generate(rng, "ana", rng.randint(1, 12))

    matching = match(events, events)

    assert matching.matched_count == len(events)
    assert matching.positive_specific_agreement() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0])
def test_an_iou_threshold_outside_the_unit_interval_is_refused(
    threshold: float,
) -> None:
    with pytest.raises(InvalidMatchParameters, match=r"\[0, 1\]"):
        match([], [], iou_threshold=threshold)


def test_an_iou_threshold_of_zero_is_refused() -> None:
    """It matches any pair that touches at all, including a 200 ms annotation
    with a 3 s one - the pair a boundary statistic most needs to separate."""
    with pytest.raises(InvalidMatchParameters, match="deliberately"):
        match([], [], iou_threshold=0.0)


def test_a_negative_tolerance_is_refused() -> None:
    with pytest.raises(InvalidMatchParameters, match="non-negative"):
        match([], [], criterion=MatchCriterion.MIDPOINT_TOLERANCE, tolerance_ms=-1)


def test_a_zero_tolerance_is_allowed() -> None:
    """Exact midpoints only. Strict, but interpretable - unlike a zero IoU."""
    left = [annotation(SpeechEventType.FILLED_PAUSE, 1_000, 2_000, "ana")]
    right = [annotation(SpeechEventType.FILLED_PAUSE, 1_000, 2_000, "beto")]

    result = match(left, right, criterion=MatchCriterion.MIDPOINT_TOLERANCE, tolerance_ms=0)

    assert result.matched_count == 1


# ---------------------------------------------------------------------------
# The tolerance criterion has to admit what it says it admits
# ---------------------------------------------------------------------------


def test_non_overlapping_events_within_tolerance_are_matched() -> None:
    """The second bug the review found.

    Under the tolerance criterion the weight was the IoU, which is 0 for two
    annotations that do not overlap - and a zero-weight pair is never selected
    by the dynamic program. So the criterion silently discarded exactly the
    pairs that distinguish it from IoU matching.

    These two are 200 ms apart at the midpoint and share no instant.
    """
    left = [annotation(SpeechEventType.FILLED_PAUSE, 1_000, 1_100, "ana")]
    right = [annotation(SpeechEventType.FILLED_PAUSE, 1_200, 1_300, "beto")]

    assert left[0].interval.iou(right[0].interval) == pytest.approx(0.0)

    result = match(left, right, criterion=MatchCriterion.MIDPOINT_TOLERANCE, tolerance_ms=250)

    assert result.matched_count == 1
    assert result.pairs[0].start_error_ms == 200


def test_the_closest_candidate_wins_under_tolerance() -> None:
    """Two candidates inside the window, neither overlapping: closeness decides."""
    left = [annotation(SpeechEventType.FILLED_PAUSE, 1_000, 1_100, "ana")]
    right = [
        annotation(SpeechEventType.FILLED_PAUSE, 1_300, 1_400, "beto"),  # 300 away
        annotation(SpeechEventType.FILLED_PAUSE, 1_150, 1_250, "beto"),  # 150 away
    ]

    result = match(left, right, criterion=MatchCriterion.MIDPOINT_TOLERANCE, tolerance_ms=400)

    assert result.matched_count == 1
    assert result.pairs[0].right.interval.start_ms == 1_150


def test_events_beyond_the_tolerance_still_do_not_match() -> None:
    left = [annotation(SpeechEventType.FILLED_PAUSE, 1_000, 1_100, "ana")]
    right = [annotation(SpeechEventType.FILLED_PAUSE, 5_000, 5_100, "beto")]

    result = match(left, right, criterion=MatchCriterion.MIDPOINT_TOLERANCE, tolerance_ms=250)

    assert result.matched_count == 0
