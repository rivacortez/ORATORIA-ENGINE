"""Deciding which of A's events are which of B's.

Every categorical agreement statistic downstream needs *units* to agree about,
and for interval annotation the units are not given — the annotators chose them
independently, including how many there are. So matching comes first, and
everything after it inherits whatever this gets wrong.

Two decisions.

*The criterion is IoU, not midpoint-within-tolerance.* A tolerance rule accepts
a 200 ms annotation and a 3 s one as the same event whenever their centres are
close, which is exactly the pair a boundary-error statistic most needs to
separate. IoU refuses that. Tolerance matching is offered as an alternative
because it is the convention in some of the disfluency literature, and a report
that cannot be compared with prior work is worth less than one that can.

*Matching is optimal under a non-crossing constraint, not greedy.* Greedy is
order-dependent and a reviewer can reasonably ask what it missed. Non-crossing
is the right constraint rather than a convenient one: events sit on a timeline,
so a match where A's third event pairs with B's fifth while A's fourth pairs
with B's second asserts that two events swapped places in time. Under that
constraint the optimum is a dynamic program in O(nm) — the same shape as
sequence alignment — and it is short enough to check by reading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from corpus.schema.records import DisfluencyAnnotation

#: Minimum overlap for two annotations to be the same event. 0.5 is the
#: convention inherited from detection literature and it is a *decision*: at
#: 0.3 the report counts near-misses as agreement, at 0.7 it counts ordinary
#: boundary jitter as two different events. The pilot is where this gets
#: argued with data rather than settled by convention, so it is a parameter
#: and the report prints it.
DEFAULT_IOU_THRESHOLD = 0.5

#: Midpoint distance for the alternative criterion, in milliseconds. Anchored
#: to NFR-004's 250 ms boundary target so that the human ceiling and the model
#: score are measured on the same scale.
DEFAULT_TOLERANCE_MS = 250


class MatchCriterion(StrEnum):
    """How two annotations are decided to be the same event."""

    IOU = "iou"
    MIDPOINT_TOLERANCE = "midpoint_tolerance"


@dataclass(frozen=True, slots=True)
class MatchedPair:
    """One event both annotators found."""

    left: DisfluencyAnnotation
    right: DisfluencyAnnotation
    iou: float

    @property
    def same_class(self) -> bool:
        return self.left.event_type is self.right.event_type

    @property
    def same_role(self) -> bool:
        return self.left.context_role is self.right.context_role

    @property
    def start_error_ms(self) -> int:
        return abs(self.left.interval.start_ms - self.right.interval.start_ms)

    @property
    def end_error_ms(self) -> int:
        return abs(self.left.interval.end_ms - self.right.interval.end_ms)


@dataclass(frozen=True, slots=True)
class Matching:
    """The full result: what paired, and what each annotator found alone."""

    pairs: tuple[MatchedPair, ...]
    left_only: tuple[DisfluencyAnnotation, ...]
    right_only: tuple[DisfluencyAnnotation, ...]
    criterion: MatchCriterion
    threshold: float

    @property
    def matched_count(self) -> int:
        return len(self.pairs)

    def positive_specific_agreement(self) -> float:
        """Agreement on *finding* events, ignoring the empty timeline.

        Equivalently the F1 between the two annotators, and symmetric under
        swapping them. This is the measure that answers "did they find the same
        events?" without the no-event class, which on a ten-minute recording
        would be 95% of any discretised timeline and would put every coefficient
        near 1 regardless of what the annotators did.

        Returns 1.0 when both found nothing: two annotators who agree a stretch
        is empty do agree, and reporting 0 there would penalise a clean
        recording.
        """
        matched = len(self.pairs)
        unmatched = len(self.left_only) + len(self.right_only)
        if matched == 0 and unmatched == 0:
            return 1.0
        return (2 * matched) / (2 * matched + unmatched)


def match(
    left: Sequence[DisfluencyAnnotation],
    right: Sequence[DisfluencyAnnotation],
    *,
    criterion: MatchCriterion = MatchCriterion.IOU,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> Matching:
    """Pair the two annotators' events optimally, without crossing.

    Class-agnostic on purpose. Whether they found the same *event* and whether
    they gave it the same *label* are separate questions, and collapsing them
    would make a class disagreement indistinguishable from a missed event —
    which is the distinction the pilot most needs, because the two have
    completely different fixes.
    """
    ordered_left = _ordered(left)
    ordered_right = _ordered(right)

    weights = _weights(ordered_left, ordered_right, criterion, iou_threshold, tolerance_ms)
    indices = _optimal_non_crossing(weights, len(ordered_left), len(ordered_right))

    pairs = tuple(
        MatchedPair(
            left=ordered_left[i],
            right=ordered_right[j],
            iou=ordered_left[i].interval.iou(ordered_right[j].interval),
        )
        for i, j in indices
    )
    matched_left = {i for i, _ in indices}
    matched_right = {j for _, j in indices}

    return Matching(
        pairs=pairs,
        left_only=tuple(a for i, a in enumerate(ordered_left) if i not in matched_left),
        right_only=tuple(a for j, a in enumerate(ordered_right) if j not in matched_right),
        criterion=criterion,
        threshold=iou_threshold if criterion is MatchCriterion.IOU else float(tolerance_ms),
    )


def _ordered(
    annotations: Sequence[DisfluencyAnnotation],
) -> list[DisfluencyAnnotation]:
    """Temporal order, with deterministic tie-breaking.

    The tie-break matters: two annotations starting at the same millisecond
    would otherwise be ordered by whatever the file happened to list first, and
    the matching would differ between two runs over the same data.
    """
    return sorted(
        annotations,
        key=lambda a: (a.interval.start_ms, a.interval.end_ms, a.event_type.value),
    )


def _weights(
    left: Sequence[DisfluencyAnnotation],
    right: Sequence[DisfluencyAnnotation],
    criterion: MatchCriterion,
    iou_threshold: float,
    tolerance_ms: int,
) -> dict[tuple[int, int], float]:
    """Admissible pairs and their scores. Anything below threshold is absent.

    Sparse rather than a dense matrix: on a real recording almost no pair of
    annotations overlaps, and iterating a dense n by m of zeros to find a
    handful of candidates is work with no reader.
    """
    weights: dict[tuple[int, int], float] = {}
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if criterion is MatchCriterion.IOU:
                score = a.interval.iou(b.interval)
                if score >= iou_threshold:
                    weights[(i, j)] = score
                continue

            midpoint_a = (a.interval.start_ms + a.interval.end_ms) / 2
            midpoint_b = (b.interval.start_ms + b.interval.end_ms) / 2
            if abs(midpoint_a - midpoint_b) <= tolerance_ms:
                # Scored by IoU even under the tolerance criterion, so that
                # when a tolerance window admits two candidates the better
                # overlap wins rather than the earlier one.
                weights[(i, j)] = a.interval.iou(b.interval)
    return weights


def _optimal_non_crossing(
    weights: dict[tuple[int, int], float], n_left: int, n_right: int
) -> list[tuple[int, int]]:
    """Maximum-weight non-crossing matching, by dynamic programming.

    ``best[i][j]`` is the best total score achievable using the first ``i``
    annotations of the left and the first ``j`` of the right. Each cell either
    drops the next left annotation, drops the next right one, or pairs them —
    the same recurrence as sequence alignment, which is exactly what this is:
    two ordered sequences of events, aligned.
    """
    best = [[0.0] * (n_right + 1) for _ in range(n_left + 1)]

    for i in range(1, n_left + 1):
        for j in range(1, n_right + 1):
            skip_left = best[i - 1][j]
            skip_right = best[i][j - 1]
            candidate = skip_left if skip_left >= skip_right else skip_right

            weight = weights.get((i - 1, j - 1))
            if weight is not None:
                paired = best[i - 1][j - 1] + weight
                if paired > candidate:
                    candidate = paired
            best[i][j] = candidate

    pairs: list[tuple[int, int]] = []
    i, j = n_left, n_right
    while i > 0 and j > 0:
        weight = weights.get((i - 1, j - 1))
        if weight is not None and abs(best[i][j] - (best[i - 1][j - 1] + weight)) < 1e-12:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs
