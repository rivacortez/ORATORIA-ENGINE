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
this question: positive specific agreement for *whether* the same events were
found, and the boundary-error distribution for *how precisely*, in numbers
whose computation is readable in this file.

**This is not the same measurement, and the thesis must not claim it is.** Two
differences a reviewer will find:

*They are not chance-corrected.* Alpha-u is; specific agreement and a boundary
median are raw observed agreement. Two annotators who mark events at random on
a densely annotated recording will show some agreement here and none under
alpha-u.

*They report the segmentation in two numbers rather than one.* Alpha-u gives a
single coefficient over the whole continuum, including the stretches both
annotators left empty, on a scale where 0 is chance and 1 is perfect. The pair
here deliberately never touches the empty timeline (which is why it cannot be
inflated by it) and has no such scale: 0.78 specific agreement is not "0.78 of
the way to perfect unitizing", it is the F1 between two annotators at one
matching threshold, and it moves when the threshold moves.

So: these answer the same *question* — did they segment the timeline the same
way — with different properties, and the report prints the matching rule
precisely because the numbers are only readable alongside it. Where the thesis
needs a chance-corrected unitizing coefficient, it needs alpha-u itself.

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
from enum import StrEnum

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
from corpus.schema.records import (
    AnnotatedRecording,
    AnnotationPass,
    DisfluencyAnnotation,
)
from corpus.schema.validation import validate
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


class InvalidReportParameters(Exception):
    """A reporting parameter would make the report say something false.

    Separate from ``RefusedComparison``: that one is about the *files*, this is
    about how they were asked to be compared. A caller can fix this one by
    typing a different number.
    """


class RefusedComparison(Exception):
    """These two files cannot produce an interpretable agreement figure.

    Refusing rather than reporting with a caveat. Every one of the conditions
    below produces a report that renders perfectly: the coefficients compute,
    the confusion matrix fills in, and the number is about something other than
    what the reader will take it to be about. A note at the bottom of a report
    does not survive being copied into a results table.
    """


class MismatchedRecordings(RefusedComparison):
    """Two files that do not describe the same recording were compared."""


class NotIndependent(RefusedComparison):
    """One of the two files is not an independent opinion.

    Agreement is a measurement of two people annotating the same audio without
    seeing each other's work. An adjudicated file is what they settled on
    *after* seeing it, so comparing it to either original measures the
    resolution process and scores near 1 by construction.
    """


class IncompatibleVersions(RefusedComparison):
    """The two files were written under record shapes that do not compare."""


class UnusableAnnotation(RefusedComparison):
    """A file the validator rejected was submitted for comparison.

    Blocking, unlike a warning. An overlapping pair of same-class annotations
    offers the matching two candidates where the other annotator has one, so
    the disagreement it produces is an artefact of a mis-drag.
    """


class DisagreementKind(StrEnum):
    """Why one position needs adjudicating.

    Separated because the four have completely different fixes. A missed event
    is a detection problem - one annotator did not hear it. A class conflict is
    a taxonomy problem - they both heard it and the manual did not tell them
    what to call it. A role conflict is narrower still. A boundary conflict is
    usually neither: two people agreeing about an event and drawing it
    differently is what NFR-004's 250 ms target is measured against.

    Reporting them as one number is how a manual that needs rewriting gets
    filed as "annotators need more training".
    """

    MISSED_BY_LEFT = "missed_by_left"
    MISSED_BY_RIGHT = "missed_by_right"
    CLASS = "class"
    ROLE = "role"
    BOUNDARY = "boundary"


@dataclass(frozen=True, slots=True)
class Reading:
    """What one annotator recorded at one position."""

    annotator: str
    start_ms: int
    end_ms: int
    event_type: str
    context_role: str | None
    raw_text: str
    note: str

    def __str__(self) -> str:
        role = f"/{self.context_role}" if self.context_role else ""
        text = f" {self.raw_text!r}" if self.raw_text else ""
        return f"{self.event_type}{role} [{self.start_ms}-{self.end_ms}]{text}"


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One position two annotators have to walk through together.

    The protocol's adjudication step asks for the audio position, both
    readings, and what was decided. The first two are here; the third is
    written by the people doing it. Producing this list by hand from a
    confusion matrix is the step where disagreements quietly go missing,
    because a matrix says a `false_start` was read as a `self_repair` three
    times and not *where*.
    """

    kind: DisagreementKind
    at_ms: int
    left: Reading | None
    right: Reading | None
    detail: str

    @property
    def timestamp(self) -> str:
        """``mm:ss.mmm``, to seek to in the annotation tool."""
        minutes, remainder = divmod(self.at_ms, 60_000)
        seconds, milliseconds = divmod(remainder, 1_000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


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

    #: Every position that needs adjudicating, in the order an adjudicator
    #: will walk the recording. The coefficients say how much they disagreed;
    #: this says where, which is the only form the protocol's adjudication step
    #: can act on.
    disagreements: tuple[Disagreement, ...] = field(default_factory=tuple)
    #: The boundary error above which a matched pair is worth walking through.
    #: Anchored to NFR-004's 250 ms so that the pairs flagged here are the
    #: pairs the model will later be scored against.
    boundary_review_ms: int = DEFAULT_TOLERANCE_MS

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
    boundary_review_ms: int = DEFAULT_TOLERANCE_MS,
) -> AgreementReport:
    """Measure agreement between two annotations of the same recording.

    Refuses anything that would produce a number about the wrong thing. See
    ``RefusedComparison`` and its subclasses for what is turned away and why.
    """
    _require_valid_review_threshold(boundary_review_ms)
    _require_comparable(left, right)

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
        disagreements=_disagreements(matching, boundary_review_ms),
        boundary_review_ms=boundary_review_ms,
        notes=_notes(left, right, matching),
    )


def _reading(annotation: DisfluencyAnnotation) -> Reading:
    return Reading(
        annotator=annotation.annotator_id,
        start_ms=annotation.interval.start_ms,
        end_ms=annotation.interval.end_ms,
        event_type=annotation.event_type.value,
        context_role=annotation.context_role.value if annotation.context_role else None,
        raw_text=annotation.raw_text,
        note=annotation.note,
    )


def _disagreements(matching: Matching, boundary_review_ms: int) -> tuple[Disagreement, ...]:
    """Every position worth walking through, in recording order.

    One entry per position, not per category: a pair that differs in class
    *and* in boundary is one thing to discuss, and listing it twice makes the
    adjudication log double-count. Class is reported over role, and role over
    boundary, because that is the order in which resolving one makes the next
    moot - deciding it was a `self_repair` after all settles the role question
    that came with the other reading.
    """
    found: list[Disagreement] = []

    for pair in matching.pairs:
        left, right = _reading(pair.left), _reading(pair.right)
        if not pair.same_class:
            found.append(
                Disagreement(
                    kind=DisagreementKind.CLASS,
                    at_ms=min(left.start_ms, right.start_ms),
                    left=left,
                    right=right,
                    detail=f"{left.event_type} vs {right.event_type}",
                )
            )
            continue
        if not pair.same_role:
            found.append(
                Disagreement(
                    kind=DisagreementKind.ROLE,
                    at_ms=min(left.start_ms, right.start_ms),
                    left=left,
                    right=right,
                    detail=f"{left.context_role} vs {right.context_role}",
                )
            )
            continue
        worst = max(pair.start_error_ms, pair.end_error_ms)
        if worst > boundary_review_ms:
            found.append(
                Disagreement(
                    kind=DisagreementKind.BOUNDARY,
                    at_ms=min(left.start_ms, right.start_ms),
                    left=left,
                    right=right,
                    detail=(
                        f"start off by {pair.start_error_ms} ms, end by "
                        f"{pair.end_error_ms} ms (over the {boundary_review_ms} ms "
                        "NFR-004 target)"
                    ),
                )
            )

    for annotation in matching.left_only:
        reading = _reading(annotation)
        found.append(
            Disagreement(
                kind=DisagreementKind.MISSED_BY_RIGHT,
                at_ms=reading.start_ms,
                left=reading,
                right=None,
                detail=f"only {reading.annotator} marked this",
            )
        )
    for annotation in matching.right_only:
        reading = _reading(annotation)
        found.append(
            Disagreement(
                kind=DisagreementKind.MISSED_BY_LEFT,
                at_ms=reading.start_ms,
                left=None,
                right=reading,
                detail=f"only {reading.annotator} marked this",
            )
        )

    # Recording order, with a total tie-break so two runs over the same files
    # produce the same log and a diff between manual versions is readable.
    found.sort(key=lambda item: (item.at_ms, item.kind.value, item.detail))
    return tuple(found)


def _require_valid_review_threshold(boundary_review_ms: int) -> None:
    """A negative threshold makes perfect agreement look like disagreement.

    ``abs(error) > threshold`` is true for every matched pair once the
    threshold goes below zero, so two annotators who drew identical boundaries
    come back with a worklist reading "start off by 0 ms, end by 0 ms (over the
    -1 ms NFR-004 target)" for every event they agreed on. Nothing crashes and
    nothing warns; the adjudication session just has a hundred items in it that
    are not disagreements.
    """
    if boundary_review_ms < 0:
        raise InvalidReportParameters(
            f"boundary review threshold must be non-negative, got {boundary_review_ms} ms. "
            "Below zero every matched pair is listed as a boundary disagreement, "
            "including the ones where the annotators agreed exactly."
        )


def _require_comparable(left: AnnotatedRecording, right: AnnotatedRecording) -> None:
    """Everything that has to hold before a coefficient means anything.

    Checked here rather than left to the caller. A precondition the caller is
    trusted to remember is a precondition that holds until the first time
    somebody runs the calculator from a notebook.
    """
    if left.recording_id != right.recording_id:
        raise MismatchedRecordings(
            f"{left.recording_id!r} and {right.recording_id!r} are different recordings"
        )
    if left.annotator_id == right.annotator_id:
        # `NotIndependent`, not `MismatchedRecordings`. The two files describe
        # the same recording perfectly well; what they do not describe is two
        # people, which is the same failure as comparing against an adjudicated
        # pass and belongs under the same name.
        raise NotIndependent(
            f"both files are by {left.annotator_id!r}; agreement is between two people"
        )

    for side, recording in (("left", left), ("right", right)):
        if recording.annotation_pass is AnnotationPass.ADJUDICATED:
            raise NotIndependent(
                f"the {side} file ({recording.annotator_id!r}) is an adjudicated pass. "
                "It records what the annotators settled on after seeing each other's "
                "work, so comparing it to either original measures the adjudication "
                "and scores near 1 by construction. Compare the two first passes."
            )

    if not left.schema_version.is_compatible_with(right.schema_version):
        raise IncompatibleVersions(
            f"the files were written under schema {left.schema_version} and "
            f"{right.schema_version}. Across a major version the records are different "
            "shapes, and the report would measure the schema change."
        )

    _require_comparable_taxonomies(left, right)
    _require_valid(left)
    _require_valid(right)


def _require_comparable_taxonomies(left: AnnotatedRecording, right: AnnotatedRecording) -> None:
    """Any taxonomy difference is refused, and so is a missing version.

    Exactly equal, not one-major-version compatible.

    An earlier version of this allowed a minor difference and reported a note,
    reasoning that the class list is additive within a major version. That
    contradicted the protocol, which says *any* taxonomy change requires
    re-running the pilot - and the protocol is right: a definition amended in a
    minor release is a definition the two annotators did not share, and the
    additive-class argument does not cover a class whose *meaning* moved. It
    also matches ``EvidenceDocument``, which refuses a manifest whose taxonomy
    is not exactly this build's.

    A missing version is refused rather than skipped. ``None`` compares unequal
    to nothing, so an early return on absence let a file with no recorded
    manual pass straight through this guard - which is how a check written
    against a value becomes no check at all when the value goes missing.
    """
    for side, recording in (("left", left), ("right", right)):
        if recording.taxonomy_version is None:
            raise IncompatibleVersions(
                f"the {side} file ({recording.annotator_id!r}) records no taxonomy "
                "version, so there is no way to establish that the two annotators "
                "worked from the same manual. Agreement between two manuals is not "
                "agreement between two annotators."
            )

    if left.taxonomy_version != right.taxonomy_version:
        raise IncompatibleVersions(
            f"the annotators worked under taxonomy {left.taxonomy_version} and "
            f"{right.taxonomy_version}. Any difference means they were reading "
            "different manuals, so this would measure the change rather than the "
            "annotators. The protocol's answer is to re-run the pilot under one "
            "version, not to report a number with a caveat."
        )


def _require_valid(recording: AnnotatedRecording) -> None:
    """Refuse a file the validator found errors in.

    Warnings still pass: an agreement report over only the tidy files would
    measure the annotators' tidiness and would drop exactly the hard cases the
    pilot exists to find. Errors do not, because each one breaks a premise the
    matching relies on.
    """
    report = validate(recording)
    if report.is_usable:
        return
    reasons = "; ".join(f"{f.code}: {f.message}" for f in report.errors)
    raise UnusableAnnotation(
        f"{recording.annotator_id}'s annotation of {recording.recording_id} has "
        f"{len(report.errors)} validation error(s) and cannot be compared - {reasons}"
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

    # No note about differing taxonomy versions: `_require_comparable` now
    # refuses any difference at all, so a report that exists was produced under
    # one manual. The note used to cover the minor-version case, and a note is
    # the wrong instrument for it - it survives exactly as long as the person
    # reading the report, and not into the table they paste the number into.

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
