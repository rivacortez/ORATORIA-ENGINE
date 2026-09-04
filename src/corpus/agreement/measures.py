"""Agreement coefficients, computed over units rather than over milliseconds.

The trap this module is built around is worth stating in full, because it is
easy to fall into and produces a number that looks excellent.

Discretise a ten-minute recording into 10 ms frames, label each frame with the
disfluency class covering it or "none", and compute Cohen's kappa. Roughly 95%
of frames are "none", both annotators agree on essentially all of them, and
kappa comes back near 0.9 — while telling you nothing about whether the
annotators agree on the disfluencies, which is the only question. The "none"
class dominates both the observed and the expected agreement, and the
coefficient is measuring the silence.

So: **the unit of analysis is a matched event, not a time frame.** Matching
happens first (``corpus.agreement.matching``); the coefficients here are
computed only over pairs that matching produced, where "none" cannot appear
because a matched pair is by definition two annotations. Unmatched events are
not thrown away — they are reported through positive specific agreement, which
never counts a true negative.

``kappa_over_time_frames`` exists and refuses to run, so that the temptation
has somewhere to fail loudly instead of a plausible number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


class UndefinedUnitError(Exception):
    """A coefficient was requested over a unit that makes it uninterpretable."""


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Who called what, when they matched on the same event."""

    #: ``counts[(left_label, right_label)]``.
    counts: Mapping[tuple[str, str], int]
    labels: tuple[str, ...]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def agreements(self) -> int:
        return sum(count for (a, b), count in self.counts.items() if a == b)

    def observed_agreement(self) -> float:
        return self.agreements / self.total if self.total else 1.0

    def for_label(self, label: str) -> tuple[int, int, int]:
        """``(both, left_only, right_only)`` for one label.

        The three numbers positive specific agreement needs, per class. True
        negatives are deliberately absent: including them is the same mistake
        as counting silent frames.
        """
        both = self.counts.get((label, label), 0)
        left_only = sum(c for (a, b), c in self.counts.items() if a == label and b != label)
        right_only = sum(c for (a, b), c in self.counts.items() if b == label and a != label)
        return both, left_only, right_only

    def specific_agreement(self, label: str) -> float | None:
        """Agreement on one class, or ``None`` when neither annotator used it.

        ``None`` rather than 0.0 or 1.0: a class nobody applied has no
        agreement to report, and either number would be read as a finding about
        the class rather than about the sample. §14.2 asks for per-class
        figures, and "not observed" is one of the legitimate answers.
        """
        both, left_only, right_only = self.for_label(label)
        if both == 0 and left_only == 0 and right_only == 0:
            return None
        return (2 * both) / (2 * both + left_only + right_only)


def confusion(pairs: Sequence[tuple[str, str]]) -> ConfusionMatrix:
    """Build the matrix from matched (left label, right label) pairs."""
    counts: dict[tuple[str, str], int] = {}
    labels: set[str] = set()
    for left, right in pairs:
        counts[(left, right)] = counts.get((left, right), 0) + 1
        labels.add(left)
        labels.add(right)
    return ConfusionMatrix(counts=counts, labels=tuple(sorted(labels)))


def cohens_kappa(matrix: ConfusionMatrix) -> float | None:
    """Cohen's kappa over matched events.

    ``None`` when it is undefined — one matched pair, or every annotation in a
    single category. In the latter case expected agreement is 1 and the
    denominator vanishes: two annotators who both labelled everything
    ``filled_pause`` agree perfectly and the coefficient cannot say how much of
    that is chance. Returning 0 there would read as "no better than chance",
    which is the opposite of what happened.
    """
    total = matrix.total
    if total < 2:
        return None

    observed = matrix.observed_agreement()
    expected = 0.0
    for label in matrix.labels:
        left_marginal = sum(c for (a, _), c in matrix.counts.items() if a == label) / total
        right_marginal = sum(c for (_, b), c in matrix.counts.items() if b == label) / total
        expected += left_marginal * right_marginal

    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def krippendorffs_alpha(matrix: ConfusionMatrix) -> float | None:
    """Krippendorff's alpha for nominal data over matched events.

    Reported alongside kappa because they answer slightly different questions
    and disagree in informative ways: alpha corrects for sample size and
    handles the small pilots this is built for more honestly, while kappa is
    what most readers will expect to see. Where they diverge sharply the sample
    is too small to conclude anything, which is itself the finding.

    This is alpha for *categorising* matched units, not alpha for *unitizing* a
    continuum. The unitizing question — did they segment the recording the same
    way — is answered by positive specific agreement and the boundary-error
    distribution instead. See the note in the module docstring of
    ``corpus.agreement.report``.
    """
    total = matrix.total
    if total < 2:
        return None

    # The coincidence matrix. Each matched pair contributes in both directions,
    # because alpha is defined over unordered pairs of judgements and neither
    # annotator is the reference.
    coincidences: dict[tuple[str, str], int] = {}
    for (left, right), count in matrix.counts.items():
        coincidences[(left, right)] = coincidences.get((left, right), 0) + count
        coincidences[(right, left)] = coincidences.get((right, left), 0) + count

    n = sum(coincidences.values())
    marginals = {
        label: sum(c for (a, _), c in coincidences.items() if a == label) for label in matrix.labels
    }

    observed_disagreement = sum(c for (a, b), c in coincidences.items() if a != b)
    expected_disagreement = sum(
        marginals[a] * marginals[b] for a in matrix.labels for b in matrix.labels if a != b
    )

    if expected_disagreement == 0:
        # Every judgement fell in one category. Same situation as kappa's
        # vanishing denominator, and the same answer: undefined, not zero.
        return None
    return 1.0 - ((n - 1) * observed_disagreement) / expected_disagreement


@dataclass(frozen=True, slots=True)
class BoundaryError:
    """How far apart two annotators put the same event's edges.

    This is the measurement NFR-004 needs and does not have. The engine claims
    a 250 ms boundary tolerance; whether that is generous or optimistic is
    unanswerable until two humans have been measured against each other on the
    same events. If they agree to 40 ms, a model claiming 250 has slack to
    justify. If they disagree by 300, the model's tolerance is not the model's
    problem and NFR-004's target needs revisiting before anything is trained.
    """

    start_errors_ms: tuple[int, ...] = field(default_factory=tuple)
    end_errors_ms: tuple[int, ...] = field(default_factory=tuple)

    @property
    def count(self) -> int:
        return len(self.start_errors_ms)

    def median_start_ms(self) -> float | None:
        return _median(self.start_errors_ms)

    def median_end_ms(self) -> float | None:
        return _median(self.end_errors_ms)

    def p95_start_ms(self) -> float | None:
        return _percentile(self.start_errors_ms, 95)

    def p95_end_ms(self) -> float | None:
        return _percentile(self.end_errors_ms, 95)


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _percentile(values: Sequence[int], percentile: int) -> float | None:
    """Nearest-rank percentile.

    Nearest-rank rather than interpolated: on the twenty or thirty matched
    events a pilot produces, an interpolated p95 invents a value between two
    observations and reads as more precise than the sample supports.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(percentile / 100 * len(ordered))
    return float(ordered[max(rank, 1) - 1])


def kappa_over_time_frames(*_args: object, **_kwargs: object) -> float:
    """Never implemented. Present so the mistake fails loudly.

    Discretising the recording and computing kappa per frame is the obvious
    thing to reach for and it produces a number near 0.9 on any recording,
    because the overwhelming majority of frames contain no event and both
    annotators agree on all of them. The coefficient ends up measuring the
    silence.

    If frame-level agreement is genuinely wanted — for a boundary study, say —
    it needs a declared unit, a stated prevalence for the no-event class, and a
    prevalence-adjusted coefficient reported next to it. Write that
    deliberately; do not get it by calling this.
    """
    raise UndefinedUnitError(
        "kappa over time frames is dominated by the no-event class and would report "
        "roughly 0.9 on any recording. Agreement is computed over matched events "
        "instead: see corpus.agreement.matching, and use positive specific agreement "
        "plus the boundary-error distribution for the segmentation question."
    )
