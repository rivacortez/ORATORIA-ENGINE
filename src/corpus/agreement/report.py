"""The agreement report: two annotators, one recording, one set of numbers.

Structured in three stages because the questions are genuinely separate and
answering them together is how an uninterpretable coefficient gets produced:

1. **Did they find the same events?** Positive specific agreement over the
   matching, plus the counts each annotator found alone. Never counts a true
   negative, so the empty timeline cannot inflate it.
2. **Did they draw the same boundaries?** Median and p95 start and end error
   over matched pairs. This is the human ceiling for NFR-004's 250 ms target,
   and it does not currently exist anywhere.
3. **Did they give it the same label?** Cohen's kappa and nominal
   Krippendorff's alpha over matched pairs only, plus the per-class confusion
   matrix and specific agreement for the classes the manual predicts will be
   hardest.

---

## What is not here: Krippendorff's alpha for unitizing

The brief asked for alpha over temporal units. Two kinds of alpha are relevant
and only one of them is implemented.

Nominal alpha over *matched events* is here (``measures.krippendorffs_alpha``)
and is straightforward. Alpha for *unitizing* a continuum — Krippendorff's
alpha-u, which measures whether two annotators segmented the timeline the same
way without assuming a common set of units — is not, for a reason worth being
explicit about: its difference function over overlapping and non-overlapping
segment pairs is intricate, there is no widely-trusted Python implementation to
check an implementation of it against, and a subtly wrong alpha-u in a thesis
is worse than an absent one. Writing it from memory would be guessing at a
formula that a reviewer can look up.

What stands in its place is the pair the event-detection literature uses for
exactly this question: positive specific agreement for *whether* the same
events were found, and the boundary-error distribution for *how precisely*.
Together they answer what alpha-u answers, in numbers whose computation is
readable in this file.

Two ways to close the gap, if a reviewer asks for alpha-u specifically:
integrate an established implementation and cite it, or have the methodologist
specify the difference function and implement it against worked examples from
Krippendorff's own papers. Both are real work; neither should be improvised.

## Why the matching rule is shared with model evaluation

The same matching used here will be used to score the model against the
held-out set. That is deliberate and it is the most useful property of this
report: it puts the human ceiling and the model's F1 on one scale. If two
trained annotators reach 0.78 positive specific agreement on ``false_start``,
a model reporting 0.80 on that class is at the ceiling, and NFR-002's macro-F1
target of 0.80 has to be read against 0.78 rather than against 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from corpus.agreement.matching import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TOLERANCE_MS,
    MatchCriterion,
    Matching,
    match,
)
from corpus.agreement.measures import (
    BoundaryError,
    ConfusionMatrix,
    cohens_kappa,
    confusion,
    krippendorffs_alpha,
)
from corpus.schema.records import AnnotatedRecording
from evidence_engine.domain.shared.taxonomy import ContextualRole, SpeechEventType

#: The classes the manual predicts will be hardest, reported separately.
#: `false_start` and `self_repair` differ only in whether the abandoned
#: fragment relates to what follows - a judgement, not an observation - and
#: `uncertain` is where an annotator declines to resolve a role. If agreement
#: is going to fail, §17 says it fails here.
WATCHED_CLASSES: tuple[SpeechEventType, ...] = (
    SpeechEventType.FALSE_START,
    SpeechEventType.SELF_REPAIR,
    SpeechEventType.LEXICAL_FILLER,
)


class MismatchedRecordings(Exception):
    """Two files that do not describe the same recording were compared."""


@dataclass(frozen=True, slots=True)
class ClassAgreement:
    """Per-class figures. ``None`` where the class was never used."""

    event_type: str
    both: int
    left_only: int
    right_only: int
    specific_agreement: float | None


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """Everything measurable about one pair of annotations."""

    recording_id: str
    left_annotator: str
    right_annotator: str
    criterion: MatchCriterion
    threshold: float

    # Stage 1 - did they find the same events?
    matched: int
    left_only: int
    right_only: int
    positive_specific_agreement: float

    # Stage 2 - did they draw the same boundaries?
    boundary: BoundaryError

    # Stage 3 - did they label them the same?
    class_kappa: float | None
    class_alpha: float | None
    class_confusion: ConfusionMatrix
    per_class: tuple[ClassAgreement, ...]

    # Contextual roles, measured separately: a disagreement about whether
    # "este" is a filler or a demonstrative is a different problem from a
    # disagreement about whether an event happened at all.
    role_kappa: float | None
    role_confusion: ConfusionMatrix
    role_agreement: tuple[ClassAgreement, ...]

    #: Abstention. §17 prescribes `uncertain`, so using it is following the
    #: manual - but a class that is uncertain half the time has a definition
    #: problem, and that is what the pilot exists to surface.
    left_uncertain: int = 0
    right_uncertain: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_annotations(self) -> int:
        return self.matched * 2 + self.left_only + self.right_only

    def watched(self) -> tuple[ClassAgreement, ...]:
        watched = {event.value for event in WATCHED_CLASSES}
        return tuple(item for item in self.per_class if item.event_type in watched)


def compare(
    left: AnnotatedRecording,
    right: AnnotatedRecording,
    *,
    criterion: MatchCriterion = MatchCriterion.IOU,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> AgreementReport:
    """Measure agreement between two annotations of the same recording."""
    if left.recording_id != right.recording_id:
        raise MismatchedRecordings(
            f"{left.recording_id!r} and {right.recording_id!r} are different recordings"
        )
    if left.annotator_id == right.annotator_id:
        raise MismatchedRecordings(
            f"both files are by {left.annotator_id!r}; agreement is between two people"
        )

    matching = match(
        left.disfluencies,
        right.disfluencies,
        criterion=criterion,
        iou_threshold=iou_threshold,
        tolerance_ms=tolerance_ms,
    )

    class_matrix = confusion(
        [(pair.left.event_type.value, pair.right.event_type.value) for pair in matching.pairs]
    )
    role_matrix = confusion(
        [
            (_role_label(pair.left.context_role), _role_label(pair.right.context_role))
            for pair in matching.pairs
            if pair.left.context_role is not None or pair.right.context_role is not None
        ]
    )

    return AgreementReport(
        recording_id=left.recording_id,
        left_annotator=left.annotator_id,
        right_annotator=right.annotator_id,
        criterion=matching.criterion,
        threshold=matching.threshold,
        matched=matching.matched_count,
        left_only=len(matching.left_only),
        right_only=len(matching.right_only),
        positive_specific_agreement=matching.positive_specific_agreement(),
        boundary=_boundary_error(matching),
        class_kappa=cohens_kappa(class_matrix),
        class_alpha=krippendorffs_alpha(class_matrix),
        class_confusion=class_matrix,
        per_class=_per_class(class_matrix, matching),
        role_kappa=cohens_kappa(role_matrix),
        role_confusion=role_matrix,
        role_agreement=tuple(
            ClassAgreement(
                event_type=label,
                both=role_matrix.for_label(label)[0],
                left_only=role_matrix.for_label(label)[1],
                right_only=role_matrix.for_label(label)[2],
                specific_agreement=role_matrix.specific_agreement(label),
            )
            for label in role_matrix.labels
        ),
        left_uncertain=left.uncertain_count,
        right_uncertain=right.uncertain_count,
        notes=_notes(left, right, matching),
    )


def _boundary_error(matching: Matching) -> BoundaryError:
    return BoundaryError(
        start_errors_ms=tuple(pair.start_error_ms for pair in matching.pairs),
        end_errors_ms=tuple(pair.end_error_ms for pair in matching.pairs),
    )


def _per_class(matrix: ConfusionMatrix, matching: Matching) -> tuple[ClassAgreement, ...]:
    """Per-class agreement, counting unmatched events against their class.

    An event one annotator found and the other missed is a disagreement about
    that class, so it belongs in that class's figures. Counting it only in the
    overall matching statistic would leave a class that one annotator never
    noticed looking perfectly agreed.
    """
    unmatched_left: dict[str, int] = {}
    for annotation in matching.left_only:
        key = annotation.event_type.value
        unmatched_left[key] = unmatched_left.get(key, 0) + 1

    unmatched_right: dict[str, int] = {}
    for annotation in matching.right_only:
        key = annotation.event_type.value
        unmatched_right[key] = unmatched_right.get(key, 0) + 1

    labels = sorted(set(matrix.labels) | set(unmatched_left) | set(unmatched_right))

    results: list[ClassAgreement] = []
    for label in labels:
        both, left_only, right_only = matrix.for_label(label)
        left_total = left_only + unmatched_left.get(label, 0)
        right_total = right_only + unmatched_right.get(label, 0)
        denominator = 2 * both + left_total + right_total
        results.append(
            ClassAgreement(
                event_type=label,
                both=both,
                left_only=left_total,
                right_only=right_total,
                specific_agreement=(2 * both) / denominator if denominator else None,
            )
        )
    return tuple(results)


def _role_label(role: ContextualRole | None) -> str:
    """A missing role is its own label, not an absence.

    One annotator marking an event lexical while the other marks it acoustic
    is a real disagreement, and dropping the pair would hide it.
    """
    return role.value if role is not None else "none"


def _notes(
    left: AnnotatedRecording, right: AnnotatedRecording, matching: Matching
) -> tuple[str, ...]:
    """Conditions that make the numbers above mean less than they appear to."""
    notes: list[str] = []

    if left.taxonomy_version != right.taxonomy_version:
        notes.append(
            f"the annotators worked under different taxonomy versions "
            f"({left.taxonomy_version} and {right.taxonomy_version}); this report "
            "measures the change as well as the annotators"
        )

    if matching.matched_count < 20:
        notes.append(
            f"only {matching.matched_count} matched events: too few for kappa or alpha "
            "to be stable. Treat them as descriptive and widen the taxonomic pilot"
        )

    if matching.matched_count == 0 and (matching.left_only or matching.right_only):
        notes.append(
            "nothing matched at all. Either the threshold is too strict for these "
            "boundaries, or the two files are of different recordings"
        )

    return tuple(notes)
