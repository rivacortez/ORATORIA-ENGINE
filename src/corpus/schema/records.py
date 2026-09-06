"""The canonical annotation record, version 2.0.0.

One file, one shape, one version. Everything downstream - the ELAN importer,
the validator, the agreement calculator, the eventual training pipeline - reads
this and nothing else, so there is exactly one place where "what an annotation
is" can change and exactly one version string that changes with it.

Five decisions worth stating.

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

*A recording carries the consent it was collected under.* The engine models
consent thoroughly for a live session (``domain/sessions/consent.py``), and
until now the research tooling - the half that will actually hold forty
participants' speech - modelled none of it. `BASELINES.md` §4 requires the
dataset card to record the consent basis, and a basis reconstructed from memory
after the fact is not a basis.

*A recording carries the chain that captured it.* `REFERENCE_ENVIRONMENT.md`
lists four audio confirmations that gate Pilot A, and a confirmation that lives
only in a document is confirmed once for a corpus rather than once per
recording. The consequence of getting this wrong is not a lost field: NVIDIA
Broadcast attenuates exactly the acoustic evidence for ``cut_off`` and
``prolongation``, so a corpus recorded through it encodes the enhancer's
decisions as ground truth and no later analysis can tell.

**Why 2.0.0 and not 1.1.0.** The three additions are required, not additive:
the ELAN reader turns away a file that does not carry them, so a 1.0.0 file is
not a 2.0.0 file with fields missing - it is a file this tool cannot read. A
minor bump would let ``is_compatible_with`` pass and the failure would surface
as "no 'consent_basis' property", which reads as *you deleted something* rather
than *your file predates this shape*. The bump costs nothing today because
nothing has been recorded yet, and it is unpayable in four weeks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date
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
SCHEMA_VERSION = SemanticVersion(2, 0, 0)


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


class ConsentBasis(StrEnum):
    """How the participant's agreement was obtained.

    Two values, because this project has two ways of obtaining it: a signed
    form, or a spoken statement captured at the head of the recording itself.

    Rejected: a free-text field. `BASELINES.md` §4 asks the dataset card to
    record the consent basis *per dataset*, which is a count, and free text
    cannot be counted - forty spellings of "signed consent form" answer the
    question in a way that no audit can total.
    """

    WRITTEN_INFORMED = "written_informed"
    RECORDED_VERBAL = "recorded_verbal"


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    """What the participant agreed to, and under which published wording.

    Modelled on ``domain/sessions/consent.py`` and deliberately narrower. Three
    things are carried over because the reasoning behind them is the same here:

    *The policy version, not the policy text.* US-008's immutability rule
    applies to a corpus as much as to a session - the participant agreed to a
    specific published wording, and a later edit to `CONSENT_AND_RETENTION.md`
    must not retroactively become what they agreed to.

    *Withdrawal is recorded, not enacted by deletion.* US-005 grants withdrawal
    without justification, so there is no field for a reason. The policy's
    deletion flow ends with the annotation file gone; this field covers the
    window between the participant saying so and the file being removed, and
    every corpus-level command refuses a recording that carries it. A withdrawal
    that could only be expressed by deleting the file would be invisible for
    exactly as long as it takes somebody to get round to deleting it.

    *Video is opt-in.* The participant-facing text says audio is captured and
    video only "si lo autorizas", so ``covers_video`` defaults to ``False`` -
    the honest default rather than the convenient one, mirroring
    ``RetentionPolicy.retain_raw_media``. Phase 5 annotates visual events over
    this same corpus, and "did this person agree to be filmed" is not a question
    that can be answered later.

    What is *not* carried over is retention. The engine retains media; this
    tooling holds annotation files, and a retention policy stated here would
    describe something nothing in this tree enforces.

    Dates rather than the engine's epoch milliseconds. ``ConsentReceipt``
    timestamps a click inside a session; a form is signed on a day. Storing that
    day as midnight-UTC milliseconds would assert a time nobody recorded, and a
    fabricated precision is read as a real one.
    """

    basis: ConsentBasis
    #: The published version of `CONSENT_AND_RETENTION.md` the participant read.
    policy_version: SemanticVersion
    granted_on: date
    covers_video: bool = False
    withdrawn_on: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.basis, ConsentBasis):
            raise SchemaViolation(
                f"consent basis must be a ConsentBasis, got {self.basis!r}. A basis that "
                "is not one of the published values cannot be counted, and §4 asks for a "
                "count"
            )
        # Same trap as the version fields below: a string that looks exactly
        # like a version compares equal to another copy of itself, so a check
        # written against the value passes while nobody can say what the value
        # means.
        if not isinstance(self.policy_version, SemanticVersion):
            raise SchemaViolation(
                f"consent policy version must be a SemanticVersion, got "
                f"{self.policy_version!r}. An unparsed version cannot answer which "
                "wording the participant actually read"
            )
        if not isinstance(self.granted_on, date):
            raise SchemaViolation(f"consent must be granted on a date, got {self.granted_on!r}")
        if self.withdrawn_on is not None:
            if not isinstance(self.withdrawn_on, date):
                raise SchemaViolation(
                    f"consent must be withdrawn on a date, got {self.withdrawn_on!r}"
                )
            if self.withdrawn_on < self.granted_on:
                raise SchemaViolation(
                    f"consent was withdrawn on {self.withdrawn_on} and granted on "
                    f"{self.granted_on}; it cannot be withdrawn before it was given"
                )

    @property
    def is_active(self) -> bool:
        return self.withdrawn_on is None

    def withdraw(self, on: date) -> ConsentRecord:
        """Record a withdrawal. No justification is accepted, because none is required."""
        return self if not self.is_active else replace(self, withdrawn_on=on)


@dataclass(frozen=True, slots=True)
class RecordingConditions:
    """The capture chain, per recording rather than per corpus.

    `REFERENCE_ENVIRONMENT.md` lists four things an operator confirms before
    Pilot A - the named physical microphone, mono PCM, a declared sample rate
    and bit depth, and NVIDIA Broadcast and Voicemeeter out of the path - and
    says they are confirmed by inspecting the recorded file rather than a
    settings dialog, because a dialog reports what was requested and the file
    reports what happened. Those four live here so they are confirmed once per
    recording; a document records them once for a corpus and cannot tell which
    session drifted.

    The fields are facts, not judgements: this record refuses only values that
    are impossible. Whether 48 kHz stereo through Voicemeeter belongs in a
    frozen corpus is a question about the corpus, and it is answered where the
    corpus is assembled - see ``partition.freeze``. Refusing it here would make
    a botched session unrepresentable, and an unrepresentable fact is one
    nobody can act on.
    """

    #: The named physical device, as it enumerates. Not "laptop mic": the whole
    #: point of the confirmation is that a virtual endpoint and a physical one
    #: are indistinguishable once the recording exists.
    microphone: str
    sample_rate_hz: int
    bit_depth: int
    channels: int
    #: NVIDIA Broadcast and Voicemeeter out of the recording path. An assertion
    #: the operator makes, which is why nothing defaults it.
    virtual_audio_bypassed: bool
    #: Free text. Room, distance, anything an error analysis by audio quality
    #: would want and no field anticipates.
    room_notes: str = ""

    def __post_init__(self) -> None:
        if not self.microphone.strip():
            raise SchemaViolation(
                "a recording names the device it was captured from; an unnamed device "
                "confirms nothing, and naming the device is the confirmation"
            )
        for name in ("sample_rate_hz", "bit_depth", "channels"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SchemaViolation(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(self.virtual_audio_bypassed, bool):
            raise SchemaViolation(
                f"virtual_audio_bypassed must be a bool, got "
                f"{self.virtual_audio_bypassed!r}. A truthy string makes the answer 'yes' "
                "whatever it says"
            )


#: A coarse regional variety, as a BCP-47-shaped tag: ``es-PE``, ``es-419``,
#: ``qu``. Checked because the alternative failure is silent and expensive - one
#: operator typing ``es-PE`` and another ``es_PE`` splits the dialect analysis
#: §14.2 asks for into two dialects that are one, and nothing downstream can
#: tell that apart from a genuine second variety.
_VARIETY_TAG = re.compile(r"^[a-z]{2,3}(-([A-Z]{2}|[0-9]{3}))?$")


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

    ``variety`` no longer defaults to ``es-PE``. That default was why nobody
    noticed the field was unreachable: it always rendered a plausible value, so
    a dialect analysis over a corpus where no one had ever stated a variety
    would have come back with one dialect and looked correct. ``None`` is the
    honest absence, and it is refused at both boundaries that matter - the ELAN
    reader will not read a file that omits the property, and the freeze will not
    put a recording in a manifest without one.
    """

    pseudonym: str
    #: Coarse regional variety, for the dialect analysis §14.2 asks for.
    #: ``None`` means nobody stated one, which is not the same as ``es-PE``.
    variety: str | None = None
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
        if self.variety is not None and not _VARIETY_TAG.match(self.variety):
            raise SchemaViolation(
                f"variety {self.variety!r} is not a language tag such as 'es-PE' or "
                "'es-419'. Two spellings of one variety split the dialect analysis §14.2 "
                "asks for into two dialects that are one, and nothing downstream can tell "
                "that apart from a genuine second variety"
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
    #: What the participant agreed to. Optional here, required by the reader and
    #: by the freeze - see ``_require_declared_provenance``.
    consent: ConsentRecord | None = None
    #: The chain that captured the audio. Same rule as ``consent``.
    conditions: RecordingConditions | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.recording_id.strip():
            raise SchemaViolation("a recording needs an id")
        if self.duration_ms <= 0:
            raise SchemaViolation(f"a recording has positive duration, got {self.duration_ms}")
        self._require_parsed_versions()
        self._require_declared_provenance()
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

    def _require_declared_provenance(self) -> None:
        """Consent and capture conditions are typed if present. Presence is not checked here.

        **The decision, and the defence.** Both are optional at construction and
        required at every boundary a real recording crosses: the ELAN reader
        refuses a file that omits the properties, and ``partition.freeze``
        refuses a corpus whose recordings do not carry them. Making them
        mandatory constructor arguments was the obvious alternative and it is
        the wrong one.

        A record built in code by a test is not a record written by the
        template. There is no participant behind a fixture that exists to place
        two intervals 80 ms apart, so a required field would be satisfied by
        whatever value the first test author typed, copied into every fixture
        after it, and thereafter present in every record in the tree. A field
        that is always filled because the type system demanded it is not
        evidence that anybody asked a participant anything - and "there is a
        consent basis on every record" is exactly the claim an audit would be
        making. Optional-and-refused-at-the-boundary keeps the field's presence
        meaning what it says.

        This is the shape ``taxonomy_version`` already has, and it is the shape
        that survived the review round where the *other* half of it was the bug:
        the guard downstream compared values and returned early on ``None``, so
        an absent version sailed through a check written to compare versions.
        The lesson was not "make it required here"; it was "make every guard
        test for absence explicitly". So they do - and each one names the
        recording, because a refusal that says "some recording lacks consent"
        sends somebody through forty files by hand.

        What *is* checked here is the type, for the reason the versions are:
        ``consent="written_informed"`` is falsy-adjacent, passes an ``is not
        None`` test, and fails much later with an ``AttributeError`` about
        ``str`` having no ``basis``.
        """
        if self.consent is not None and not isinstance(self.consent, ConsentRecord):
            raise SchemaViolation(
                f"consent must be a ConsentRecord, got {self.consent!r}. Anything else "
                "passes an 'is not None' check and then means nothing"
            )
        if self.conditions is not None and not isinstance(self.conditions, RecordingConditions):
            raise SchemaViolation(
                f"conditions must be a RecordingConditions, got {self.conditions!r}"
            )

    @property
    def consent_is_active(self) -> bool:
        """Whether this recording may be counted at all.

        ``False`` for an unstated consent as well as a withdrawn one. A property
        that answered ``True`` when nobody had recorded a basis would make the
        absence of consent look like the presence of it, which is the failure
        the whole field exists to prevent.
        """
        return self.consent is not None and self.consent.is_active

    def of_type(self, event_type: SpeechEventType) -> tuple[DisfluencyAnnotation, ...]:
        return tuple(a for a in self.disfluencies if a.event_type is event_type)

    @property
    def uncertain_count(self) -> int:
        return sum(1 for a in self.disfluencies if a.is_uncertain)
