"""The canonical annotation record, version 1.0.0.

One file, one shape, one version. Everything downstream - the ELAN importer,
the validator, the agreement calculator, the eventual training pipeline - reads
this and nothing else, so there is exactly one place where "what an annotation
is" can change and exactly one version string that changes with it.

Three decisions worth stating.

*Classes resolve against the engine's taxonomy, not a copy.* An annotation
schema with its own list of classes drifts from the published allowlist, and
the drift shows up weeks later as inter-annotator disagreement that looks like
a hard boundary case and is really a documentation bug. ``require_known_*``
from ``evidence_engine.domain.shared.taxonomy`` is the single source.

*Speaker identity is pseudonymous by construction.* There is no field for a
name, an email or a student code. NFR-011 and the consent policy are easier to
honour when the identifying field does not exist than when it exists and is
supposed to stay empty.

*An annotation records who made it and how sure they were.* Agreement is
computed between annotators, so the annotator is part of the datum, not
metadata about the file. And an annotator who is unsure needs somewhere to say
so that is not the ``uncertain`` contextual role - the two mean different
things: ``uncertain`` is a decision about the *utterance*, low confidence is a
statement about the *annotator*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import (
    ContextualRole,
    SpeechEventType,
    require_known_speech_event,
)

#: Bumped whenever the shape below changes. An annotation file carries the
#: version it was written under, so a corpus assembled across a schema change
#: stays readable rather than becoming a migration problem.
SCHEMA_VERSION = SemanticVersion(1, 0, 0)


class SchemaViolation(Exception):
    """An annotation record was built in a shape the schema forbids."""


class AnnotationPass(StrEnum):
    """Which pass of the double-annotation workflow produced this.

    ``ADJUDICATED`` is not a third opinion. It is the record of what the two
    annotators settled on after seeing their disagreement, and it is excluded
    from agreement computation by construction - including it would measure the
    resolution process rather than the annotators.
    """

    FIRST = "first"
    SECOND = "second"
    ADJUDICATED = "adjudicated"


@dataclass(frozen=True, slots=True)
class Interval:
    """A span in milliseconds, relative to the start of the recording.

    Milliseconds and integers, matching the engine's timeline. Annotation tools
    work in seconds as floats; the importer converts once, at the boundary, so
    that no two parts of this system disagree about what 1.2340000001 means.
    """

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise SchemaViolation(f"interval starts before the recording: {self.start_ms}")
        if self.end_ms <= self.start_ms:
            raise SchemaViolation(f"interval has no duration: [{self.start_ms}, {self.end_ms})")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def overlaps(self, other: Interval) -> bool:
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms

    def intersection_ms(self, other: Interval) -> int:
        return max(0, min(self.end_ms, other.end_ms) - max(self.start_ms, other.start_ms))

    def union_ms(self, other: Interval) -> int:
        return self.duration_ms + other.duration_ms - self.intersection_ms(other)

    def iou(self, other: Interval) -> float:
        """Intersection over union. The matching criterion, in one number."""
        union = self.union_ms(other)
        return self.intersection_ms(other) / union if union else 0.0


@dataclass(frozen=True, slots=True)
class Word:
    """One literal word, as spoken.

    ``text`` is verbatim (FR-011). An annotator who "corrects" a word here has
    destroyed the evidence the corpus exists to capture, so the annotation
    manual says so and the validator flags the tell-tale patterns.
    """

    text: str
    interval: Interval

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise SchemaViolation("a word cannot be empty; mark unintelligible speech as an event")


@dataclass(frozen=True, slots=True)
class DisfluencyAnnotation:
    """One annotated disfluency.

    ``context_role`` is required for the lexical classes and forbidden for the
    rest, mirroring the engine's own rule (FR-013, FR-017): a filled pause has
    no word to assign a role to, and forcing one invites an annotator to invent
    a distinction they did not make.
    """

    event_type: SpeechEventType
    interval: Interval
    annotator_id: str
    #: The literal expression this covers. Required for lexical classes and
    #: kept regardless of role - FR-017 keeps the raw expression even when the
    #: role turns out to be semantic, because precision is uncomputable
    #: without the true negatives.
    raw_text: str = ""
    context_role: ContextualRole | None = None
    #: How sure the *annotator* was. Distinct from ``ContextualRole.UNCERTAIN``,
    #: which is a decision about the utterance. An annotator can be certain
    #: that the context does not decide.
    annotator_confidence: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if not self.annotator_id.strip():
            raise SchemaViolation("every annotation names its annotator; agreement needs it")
        if not 0.0 <= self.annotator_confidence <= 1.0:
            raise SchemaViolation(
                f"annotator confidence must lie in [0, 1], got {self.annotator_confidence}"
            )
        # Resolves against the engine's published allowlist. A class that is
        # not in the taxonomy cannot be annotated, which is the same rule the
        # service enforces on what it emits.
        require_known_speech_event(self.event_type.value)
        self._require_role_rules()

    def _require_role_rules(self) -> None:
        if self.event_type in LEXICAL_CLASSES:
            if not self.raw_text.strip():
                raise SchemaViolation(
                    f"'{self.event_type.value}' is decided from a word, so raw_text is "
                    "required (FR-017)"
                )
            if self.context_role is None:
                raise SchemaViolation(
                    f"'{self.event_type.value}' requires a contextual role (FR-013); use "
                    f"'{ContextualRole.UNCERTAIN.value}' when the context does not decide it"
                )
            return
        if self.context_role is not None:
            raise SchemaViolation(
                f"'{self.event_type.value}' has no word to attach a role to; setting one "
                "asserts a distinction the annotator did not make"
            )

    @property
    def is_uncertain(self) -> bool:
        """Whether the annotator declined to resolve the role.

        Reported separately in the agreement report. An engine that answers
        `uncertain` everywhere would score well on precision if these were
        dropped, so the abstention rate travels with the precision figure.
        """
        return self.context_role is ContextualRole.UNCERTAIN


#: Mirrors the engine's own set. Duplicated rather than imported because the
#: engine keeps it private to its constructor check, and the two serve
#: different purposes: the engine refuses a bad event, this decides what an
#: annotator is required to supply.
LEXICAL_CLASSES: frozenset[SpeechEventType] = frozenset(
    {
        SpeechEventType.LEXICAL_FILLER,
        SpeechEventType.REPETITION,
        SpeechEventType.FALSE_START,
        SpeechEventType.SELF_REPAIR,
    }
)


@dataclass(frozen=True, slots=True)
class Speaker:
    """A participant, pseudonymously.

    There is no name field, no email field, no student code. §14.4 requires
    research exports to be pseudonymized and the consent policy promises it;
    both are far easier to honour when the identifying column does not exist
    than when it exists and is supposed to stay empty.

    The attributes that *are* here exist because §14.2 asks for error analysis
    by dialect, audio quality and relevant demographic subgroups. They are
    coarse on purpose: fine-grained demographics on a thesis-scale corpus
    identify people.
    """

    pseudonym: str
    #: Coarse regional variety, for the dialect analysis §14.2 asks for.
    variety: str = "es-PE"
    #: Free-text, coarse. Empty when not collected.
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.pseudonym.strip():
            raise SchemaViolation("a speaker needs a pseudonym")
        lowered = self.pseudonym.lower()
        if "@" in lowered or any(part.isdigit() and len(part) > 6 for part in lowered.split()):
            raise SchemaViolation(
                f"pseudonym {self.pseudonym!r} looks like an identifier rather than a "
                "pseudonym; §14.4 requires research data to be pseudonymized"
            )


@dataclass(frozen=True, slots=True)
class AnnotatedRecording:
    """Everything one annotator produced for one recording.

    The unit of agreement computation. Two of these over the same recording,
    from different annotators, are what the agreement calculator compares.
    """

    recording_id: str
    speaker: Speaker
    annotator_id: str
    annotation_pass: AnnotationPass
    duration_ms: int
    words: tuple[Word, ...] = field(default_factory=tuple)
    disfluencies: tuple[DisfluencyAnnotation, ...] = field(default_factory=tuple)
    schema_version: SemanticVersion = SCHEMA_VERSION
    #: The taxonomy version the annotator worked under. A pilot that changes
    #: the manual produces annotations under two versions, and comparing them
    #: without noticing would measure the change rather than the annotators.
    taxonomy_version: SemanticVersion | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.recording_id.strip():
            raise SchemaViolation("a recording needs an id")
        if self.duration_ms <= 0:
            raise SchemaViolation(f"a recording has positive duration, got {self.duration_ms}")
        self._require_parsed_versions()
        for annotation in self.disfluencies:
            if annotation.annotator_id != self.annotator_id:
                raise SchemaViolation(
                    f"annotation by {annotation.annotator_id!r} in a file attributed to "
                    f"{self.annotator_id!r}; agreement would compare the wrong people"
                )
            if annotation.interval.end_ms > self.duration_ms:
                raise SchemaViolation(
                    f"annotation ends at {annotation.interval.end_ms} ms, past the "
                    f"recording's {self.duration_ms} ms"
                )

    def _require_parsed_versions(self) -> None:
        """A version is a ``SemanticVersion``, not a string that looks like one.

        The ELAN reader parses both versions and refuses what it cannot read,
        so a record built from a file is safe. A record built in code is not:
        Python does not enforce the annotation, and ``taxonomy_version="v1"``
        used to be accepted here and then sail through the agreement guard,
        because two records carrying the *same* unparseable string compare
        equal. The comparison would have been between two manuals nobody can
        identify, and it would have looked like agreement.

        Enforced in the constructor rather than at the comparison, so the
        failure surfaces where the bad value was introduced. Every path into
        the corpus - the reader, a notebook, a future importer - goes through
        here.
        """
        for field_name in ("schema_version", "taxonomy_version"):
            value = getattr(self, field_name)
            if value is None and field_name == "taxonomy_version":
                # Permitted at construction, refused at comparison. A record
                # can legitimately not know its manual - one written before the
                # property existed - and the refusal belongs where the number
                # would be produced.
                continue
            if not isinstance(value, SemanticVersion):
                raise SchemaViolation(
                    f"{field_name} must be a SemanticVersion, got {value!r}. A version "
                    "that is not parsed is a version that cannot be compared, and two "
                    "records carrying the same unreadable string compare equal."
                )

    def of_type(self, event_type: SpeechEventType) -> tuple[DisfluencyAnnotation, ...]:
        return tuple(a for a in self.disfluencies if a.event_type is event_type)

    @property
    def uncertain_count(self) -> int:
        return sum(1 for a in self.disfluencies if a.is_uncertain)
