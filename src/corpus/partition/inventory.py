"""What the corpus actually contains, per class and per speaker.

The question Phase 1 has to answer before it stops recording is not "how many
minutes do we have" but "is there enough of each class, from enough different
people, to compute a per-class figure that means something". Those are
different questions and only the second one gates the phase.

**Counting instances is not enough, and this is the whole content of the
module.** Forty `cut_off` instances from one speaker is one speaker's habit,
not evidence about `cut_off`. The pilot protocol already says as much about
agreement - *"A single speaker's disfluency profile is idiosyncratic - some
people never produce `prolongation` at all - and agreement measured on one
speaker is agreement about that speaker"* - and the same is true of a
per-class F1. So adequacy is a pair: enough instances, spread over enough
speakers, and a class failing either test fails.

**The dialect axis is counted here too, and for the same reason.** §14.2 asks
for error analysis by dialect, which is a stratification question with exactly
the shape of the one above: a corpus of forty speakers of whom thirty-nine are
``es-PE`` and one is ``es-MX`` cannot be sliced by dialect, and the only moment
that fact is actionable is while there is still time to recruit. A field that
exists and is never counted is a field nobody finds out is empty.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from corpus.schema.records import AnnotatedRecording
from evidence_engine.domain.shared.taxonomy import SpeechEventType, p0_speech_events

#: The floor a P0 class has to clear to carry a per-class figure. Deliberately
#: a parameter with a default rather than a constant: the right number depends
#: on the corpus and on what the figure is used for, and the pilot is where it
#: gets argued with data. Printed by every report that uses it, so nobody has
#: to guess which floor a verdict was reached under.
DEFAULT_MINIMUM_INSTANCES = 30

#: And from at least this many different speakers. Five is the smallest number
#: at which "not one person's habit" is arguable; it is not a number with a
#: literature behind it and the report says so by printing it.
DEFAULT_MINIMUM_SPEAKERS = 5

#: The bucket for speakers nobody recorded a variety for. Parenthesised so it
#: cannot collide with a real one: ``Speaker`` rejects anything that is not a
#: language tag, and a language tag cannot contain brackets - so this label is
#: unforgeable rather than merely unlikely.
UNRECORDED_VARIETY = "(unrecorded)"


class InventoryError(Exception):
    """The recordings cannot be inventoried as one corpus."""


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    """One speaker's whole contribution.

    Keyed by pseudonym, which is the only identifier the schema has - §14.4
    requires research exports to be pseudonymized and `Speaker` has no field
    for anything else.
    """

    pseudonym: str
    #: ``None`` where no recording of theirs stated one. Reported rather than
    #: defaulted: a corpus in which nobody stated a variety and one in which
    #: everybody is Peruvian look identical once the absence is filled in.
    variety: str | None
    recording_ids: tuple[str, ...]
    duration_ms: int
    class_counts: Mapping[str, int]

    @property
    def event_count(self) -> int:
        return sum(self.class_counts.values())

    def has(self, event_type: SpeechEventType) -> bool:
        return self.class_counts.get(event_type.value, 0) > 0


@dataclass(frozen=True, slots=True)
class ClassAdequacy:
    """Whether one class can carry a per-class figure, and why not if not."""

    event_type: str
    instances: int
    speakers: int
    minimum_instances: int
    minimum_speakers: int

    @property
    def has_enough_instances(self) -> bool:
        return self.instances >= self.minimum_instances

    @property
    def has_enough_speakers(self) -> bool:
        return self.speakers >= self.minimum_speakers

    @property
    def is_adequate(self) -> bool:
        return self.has_enough_instances and self.has_enough_speakers

    @property
    def shortfall(self) -> str:
        """What is missing, in words, or an empty string when nothing is.

        Both halves are reported even when both fail. "Needs 12 more instances"
        sends somebody to record more of the same speaker; "and 3 more speakers"
        is the half that changes what they record.
        """
        if self.is_adequate:
            return ""
        missing = []
        if not self.has_enough_instances:
            missing.append(f"{self.minimum_instances - self.instances} more instances")
        if not self.has_enough_speakers:
            missing.append(f"{self.minimum_speakers - self.speakers} more speakers")
        return " and ".join(missing)


@dataclass(frozen=True, slots=True)
class CorpusInventory:
    """Everything countable about a set of annotated recordings."""

    recording_count: int
    total_duration_ms: int
    speakers: tuple[SpeakerProfile, ...]
    class_counts: Mapping[str, int]
    #: How many distinct speakers produced each class. The number that stops a
    #: rare class from being one person's habit.
    class_speaker_counts: Mapping[str, int]
    #: How many distinct speakers were recorded as speaking each variety, with
    #: ``UNRECORDED_VARIETY`` for those nobody stated one for. The answer to "can
    #: this corpus be sliced by dialect at all", which §14.2 assumes it can.
    variety_speaker_counts: Mapping[str, int]

    @property
    def speaker_count(self) -> int:
        return len(self.speakers)

    @property
    def event_count(self) -> int:
        return sum(self.class_counts.values())

    def adequacy(
        self,
        *,
        minimum_instances: int = DEFAULT_MINIMUM_INSTANCES,
        minimum_speakers: int = DEFAULT_MINIMUM_SPEAKERS,
    ) -> tuple[ClassAdequacy, ...]:
        """One verdict per P0 class, in taxonomy order.

        Every P0 class is reported, including the ones with zero instances.
        Iterating what was found would silently omit exactly the classes the
        corpus is short of, which are the ones the answer is about.
        """
        return tuple(
            ClassAdequacy(
                event_type=event.value,
                instances=self.class_counts.get(event.value, 0),
                speakers=self.class_speaker_counts.get(event.value, 0),
                minimum_instances=minimum_instances,
                minimum_speakers=minimum_speakers,
            )
            for event in sorted(p0_speech_events(), key=lambda e: e.value)
        )

    def is_adequate(
        self,
        *,
        minimum_instances: int = DEFAULT_MINIMUM_INSTANCES,
        minimum_speakers: int = DEFAULT_MINIMUM_SPEAKERS,
    ) -> bool:
        return all(
            item.is_adequate
            for item in self.adequacy(
                minimum_instances=minimum_instances, minimum_speakers=minimum_speakers
            )
        )


def inventory(recordings: Sequence[AnnotatedRecording]) -> CorpusInventory:
    """Count a corpus.

    Takes one annotation per recording. Two annotators over the same recording
    are an agreement question, not a corpus question, and counting both would
    double every figure here - so a repeated recording id is refused rather
    than silently summed.
    """
    _require_one_annotation_per_recording(recordings)
    _require_live_consent(recordings)

    by_speaker: dict[str, list[AnnotatedRecording]] = {}
    for recording in recordings:
        by_speaker.setdefault(recording.speaker.pseudonym, []).append(recording)

    profiles = tuple(_profile(pseudonym, group) for pseudonym, group in sorted(by_speaker.items()))

    class_counts: Counter[str] = Counter()
    class_speakers: dict[str, set[str]] = {}
    variety_speakers: Counter[str] = Counter()
    for profile in profiles:
        variety_speakers[profile.variety or UNRECORDED_VARIETY] += 1
        for event_type, count in profile.class_counts.items():
            class_counts[event_type] += count
            class_speakers.setdefault(event_type, set()).add(profile.pseudonym)

    return CorpusInventory(
        recording_count=len(recordings),
        total_duration_ms=sum(r.duration_ms for r in recordings),
        speakers=profiles,
        class_counts=MappingProxyType(dict(sorted(class_counts.items()))),
        class_speaker_counts=MappingProxyType(
            {event: len(speakers) for event, speakers in sorted(class_speakers.items())}
        ),
        variety_speaker_counts=MappingProxyType(dict(sorted(variety_speakers.items()))),
    )


def _require_live_consent(recordings: Sequence[AnnotatedRecording]) -> None:
    """A withdrawn recording is refused here, not quietly dropped.

    Withdrawal is the one consent state where the correct action is to delete
    the file, and the inventory is the command a recording schedule is driven
    by: it says "keep going" or "stop" and a methodologist acts on it weekly.
    Counting a withdrawn participant inflates every figure in that answer - the
    corpus reads as forty speakers when thirty-nine remain - and *excluding*
    them silently is worse, because the counts then change between two runs over
    what looks like the same directory.

    Refused rather than warned for the reason every other refusal in this tree
    exists: the report renders perfectly either way, and the adequacy verdict is
    the number somebody stops recording on.

    Note what is *not* refused: a recording with no consent record at all. That
    is caught at the two boundaries a real recording crosses - the ELAN reader,
    which will not read a file without the properties, and the freeze, which
    will not put one in a manifest. Refusing it here as well would break every
    in-code fixture for no gain, since none of them can reach a manifest.
    """
    for recording in recordings:
        if recording.consent is not None and not recording.consent.is_active:
            raise InventoryError(
                f"consent for {recording.recording_id!r} (speaker "
                f"{recording.speaker.pseudonym!r}) was withdrawn on "
                f"{recording.consent.withdrawn_on}. It cannot be counted: the adequacy "
                "verdict is what somebody stops recording on, and this participant's "
                "data has to be deleted rather than tallied. Remove the file."
            )


def _require_one_annotation_per_recording(
    recordings: Sequence[AnnotatedRecording],
) -> None:
    seen: dict[str, str] = {}
    for recording in recordings:
        previous = seen.get(recording.recording_id)
        if previous is not None:
            raise InventoryError(
                f"recording {recording.recording_id!r} appears twice, annotated by "
                f"{previous!r} and {recording.annotator_id!r}. An inventory counts a "
                "corpus once; two annotations of one recording are an agreement "
                "question, and counting both doubles every figure here. Pass the "
                "adjudicated pass, or one annotator's."
            )
        seen[recording.recording_id] = recording.annotator_id


def _profile(pseudonym: str, group: Iterable[AnnotatedRecording]) -> SpeakerProfile:
    recordings = sorted(group, key=lambda r: r.recording_id)
    counts: Counter[str] = Counter()
    for recording in recordings:
        for annotation in recording.disfluencies:
            counts[annotation.event_type.value] += 1

    varieties = {r.speaker.variety for r in recordings}
    if len(varieties) > 1:
        # Labelled before sorting. `variety` is now nullable, and `sorted` over
        # a set holding both `None` and a string raises TypeError - so the
        # refusal that exists to name a metadata problem would instead crash
        # with a message about '<' not being supported.
        named = sorted(v if v is not None else UNRECORDED_VARIETY for v in varieties)
        raise InventoryError(
            f"speaker {pseudonym!r} is recorded with more than one variety "
            f"({named}). Either the pseudonym is reused for two people - "
            "which breaks the speaker-independence guarantee the partitions rest on - "
            "or one of the recordings has the wrong metadata."
        )

    return SpeakerProfile(
        pseudonym=pseudonym,
        variety=next(iter(varieties)),
        recording_ids=tuple(r.recording_id for r in recordings),
        duration_ms=sum(r.duration_ms for r in recordings),
        class_counts=MappingProxyType(dict(sorted(counts.items()))),
    )
