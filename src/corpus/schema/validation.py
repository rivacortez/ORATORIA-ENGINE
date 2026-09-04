"""Validating one annotator's work before it reaches the agreement report.

The schema's constructors already refuse a record that is malformed. This
catches the things that are *well-formed and wrong*, and the distinction
matters: a malformed file fails loudly at import, while a plausible-but-wrong
one silently becomes a disagreement figure that nobody can interpret.

Findings are graded rather than binary. An overlap of the same class is an
error — a speaker cannot produce two filled pauses at the same instant, so one
of them is a mis-drag. A raw expression that never appears in the transcript is
a warning: usually it means the annotator typed a tidied form, which is the
verbatim-fidelity failure this whole corpus exists to avoid, but it can also be
a legitimate transcription of something the word tier missed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from corpus.schema.records import (
    LEXICAL_CLASSES,
    AnnotatedRecording,
    DisfluencyAnnotation,
    Word,
)

#: Below this, the words tier probably stops short of the recording. Not an
#: error - a recording can contain long silences - but a report that does not
#: mention it invites an agreement figure computed over half a session.
LOW_COVERAGE_RATIO = 0.2

#: Characters that do not belong in a verbatim transcription of speech. A
#: comma is not a sound. Their presence is the clearest available tell that
#: somebody typed what the speaker *meant*.
_ORTHOGRAPHIC_MARKS = frozenset(',.;:¿?¡!"“”')


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing worth telling the annotator, with where to look."""

    severity: Severity
    code: str
    message: str
    at_ms: int | None = None

    def __str__(self) -> str:
        where = f" @{self.at_ms}ms" if self.at_ms is not None else ""
        return f"[{self.severity.value}] {self.code}{where}: {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Everything found in one annotated recording."""

    recording_id: str
    annotator_id: str
    findings: tuple[Finding, ...]

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def is_usable(self) -> bool:
        """Whether this file may enter the agreement computation.

        Warnings do not block. An agreement report over only the clean files
        would measure the annotators' tidiness rather than their agreement, and
        would drop exactly the hard cases the pilot exists to find.
        """
        return not self.errors


def validate(recording: AnnotatedRecording) -> ValidationReport:
    """Check one annotator's work."""
    findings: list[Finding] = []
    findings.extend(_same_class_overlaps(recording.disfluencies))
    findings.extend(_word_ordering(recording.words))
    findings.extend(_orthography(recording.words))
    findings.extend(_unattested_expressions(recording))
    findings.extend(_coverage(recording))
    findings.extend(_abstention(recording))
    return ValidationReport(
        recording_id=recording.recording_id,
        annotator_id=recording.annotator_id,
        findings=tuple(findings),
    )


def _same_class_overlaps(
    disfluencies: Sequence[DisfluencyAnnotation],
) -> Iterator[Finding]:
    """Two annotations of one class cannot occupy the same instant.

    Across classes, overlap is legitimate: a filled pause can sit inside a
    false start. Within a class it is a mis-drag, and it breaks the agreement
    matching downstream by offering two candidates where the other annotator
    has one.
    """
    by_class: dict[str, list[DisfluencyAnnotation]] = {}
    for annotation in disfluencies:
        by_class.setdefault(annotation.event_type.value, []).append(annotation)

    for class_name, group in by_class.items():
        ordered = sorted(group, key=lambda a: (a.interval.start_ms, a.interval.end_ms))
        for earlier, later in pairwise(ordered):
            if earlier.interval.overlaps(later.interval):
                yield Finding(
                    Severity.ERROR,
                    "overlapping_same_class",
                    f"two '{class_name}' annotations overlap "
                    f"([{earlier.interval.start_ms}, {earlier.interval.end_ms}) and "
                    f"[{later.interval.start_ms}, {later.interval.end_ms})); "
                    "one of them is probably a mis-drag",
                    at_ms=later.interval.start_ms,
                )


def _word_ordering(words: Sequence[Word]) -> Iterator[Finding]:
    """Words are spoken one after another; their intervals should not overlap."""
    ordered = sorted(words, key=lambda w: (w.interval.start_ms, w.interval.end_ms))
    for earlier, later in pairwise(ordered):
        if earlier.interval.overlaps(later.interval):
            yield Finding(
                Severity.WARNING,
                "overlapping_words",
                f"'{earlier.text}' and '{later.text}' overlap in time; a single speaker "
                "does not say two words at once",
                at_ms=later.interval.start_ms,
            )


def _orthography(words: Sequence[Word]) -> Iterator[Finding]:
    """Punctuation in a verbatim tier means somebody transcribed meaning.

    FR-011 forbids grammatical cleanup, and the corpus is the ground truth
    against which that is later measured. A comma here is not a small
    inaccuracy - it is evidence that the annotator was writing prose.
    """
    for word in words:
        marks = sorted(_ORTHOGRAPHIC_MARKS.intersection(word.text))
        if marks:
            yield Finding(
                Severity.WARNING,
                "orthographic_marks",
                f"'{word.text}' contains {marks}; the words tier is verbatim speech, "
                "and punctuation is not a sound",
                at_ms=word.interval.start_ms,
            )


def _unattested_expressions(recording: AnnotatedRecording) -> Iterator[Finding]:
    """A lexical annotation should quote words that were actually transcribed.

    Checked against the words overlapping the annotation, not the whole
    transcript: an expression that appears somewhere else in the recording is
    not evidence that it appears *here*.
    """
    for annotation in recording.disfluencies:
        if annotation.event_type not in LEXICAL_CLASSES or not annotation.raw_text.strip():
            continue

        spoken = {
            word.text.lower().strip()
            for word in recording.words
            if word.interval.overlaps(annotation.interval)
        }
        quoted = {token.lower() for token in annotation.raw_text.split() if token}
        missing = sorted(quoted - spoken)
        if missing and spoken:
            yield Finding(
                Severity.WARNING,
                "unattested_expression",
                f"'{annotation.raw_text}' quotes {missing}, which the words tier does "
                "not contain here; check it is not a tidied form",
                at_ms=annotation.interval.start_ms,
            )


def _coverage(recording: AnnotatedRecording) -> Iterator[Finding]:
    """How much of the recording the words tier accounts for."""
    covered = sum(word.interval.duration_ms for word in recording.words)
    ratio = covered / recording.duration_ms if recording.duration_ms else 0.0

    if not recording.words:
        yield Finding(
            Severity.ERROR,
            "no_transcription",
            "the words tier is empty; lexical classes cannot be checked and the "
            "silent-pause derivation has nothing to work from",
        )
        return

    if ratio < LOW_COVERAGE_RATIO:
        yield Finding(
            Severity.WARNING,
            "low_coverage",
            f"words cover {ratio:.0%} of the recording; if the annotator stopped "
            "early, agreement would be computed over a stretch only one of them "
            "annotated",
        )


def _abstention(recording: AnnotatedRecording) -> Iterator[Finding]:
    """Report the abstention rate; never treat it as a fault.

    §17 prescribes `uncertain` as the mitigation for ambiguous lexical fillers,
    so an annotator using it is following the manual. It is reported because
    the rate is a finding about the *taxonomy*: a class that is uncertain half
    the time has a definition problem, which is exactly what the pilot is for.
    """
    lexical = [a for a in recording.disfluencies if a.event_type in LEXICAL_CLASSES]
    if not lexical:
        return

    rate = recording.uncertain_count / len(lexical)
    yield Finding(
        Severity.INFO,
        "abstention_rate",
        f"{recording.uncertain_count} of {len(lexical)} lexical annotations are "
        f"'uncertain' ({rate:.0%})",
    )
