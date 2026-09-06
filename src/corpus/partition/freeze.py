"""Freezing the partitions, so that "frozen" is checkable rather than intended.

`BASELINES.md` §4 says the held-out set is frozen at the end of Phase 1 and not
touched during tuning, and §14.4 makes that a scientific gate. A gate enforced
by everyone remembering is not a gate: the failure it exists to prevent -
someone regenerating the held-out set after seeing a disappointing number -
looks exactly like ordinary work while it is happening, and is undetectable
afterwards.

So the freeze writes a manifest that names every recording, its partition, its
speaker, and the SHA-256 of the annotation file it came from, and carries a
digest over all of it. Three questions become answerable months later by
running one command:

- Is this the held-out set the baseline was scored on?
- Has any file in it changed since?
- Was any recording added, removed, or moved between partitions?

The manifest is the machine-readable half of the dataset card §4 asks for, and
it now carries the machine-readable *parts* of what §4 lists: the consent basis
and policy version per recording, the capture chain per recording, and the
speaker's variety. Those used to be described here as "the half a human writes",
which was true of *why these speakers* and false of the rest - a consent basis
is an enumerated value and a sample rate is an integer, and a prose dataset card
carrying them cannot be checked against the corpus it describes. What stays
prose is the reasoning: why these speakers, why this room, what the recruitment
missed.

**Why the freeze refuses more than it used to.** §4 makes this manifest the
thing a result is defended with, so every value in it has to mean what a reader
will take it to mean. Four refusals follow from that and each one names a number
that would otherwise render perfectly and be about something else:

- **No consent record.** A manifest entry with a blank consent basis is read as
  "not applicable" and cannot be told from "nobody asked".
- **Withdrawn consent.** US-005 grants withdrawal without justification and the
  policy's deletion flow removes the file; freezing the participant into a
  held-out set makes the withdrawal undoable in practice.
- **No variety.** §14.2 asks for error analysis by dialect. A null variety in
  one row of forty either drops that speaker from every slice or lands them in a
  bucket nobody named.
- **A capture chain that was not bypassed.** `REFERENCE_ENVIRONMENT.md` is
  explicit: NVIDIA Broadcast suppresses breath, creak and the trailing energy of
  a cut-off word, which is the acoustic evidence for three of the nine classes.
  A per-class F1 over enhanced audio is a figure about the enhancer.

What is deliberately *not* refused is a corpus that mixes sample rates or
consent policy versions. Both are analysable rather than fatal: §14.2 asks for
error analysis by audio quality, which needs the variation recorded rather than
excluded, and a policy amended mid-recruitment leaves two validly consented
cohorts. Refusing them would be the tool making a methodological decision that
belongs to the methodologist.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from corpus.partition.inventory import inventory
from corpus.partition.split import Partition, PartitionPlan
from corpus.schema.records import (
    AnnotatedRecording,
    ConsentBasis,
    ConsentRecord,
    RecordingConditions,
    SchemaViolation,
)
from corpus.schema.validation import Finding, Severity
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import p0_speech_events

#: Bumped when the manifest's shape changes. Separate from the annotation
#: schema version: a manifest can gain a field without any annotation record
#: changing, and a reader has to be able to tell the two apart.
#:
#: 2.0.0 rather than 1.1.0 for the reason ``SCHEMA_VERSION`` gives: the new
#: per-recording fields are required, so a 1.0.0 manifest is not this shape with
#: fields missing. A minor bump would pass ``is_compatible_with`` and then fail
#: in ``from_json`` with a ``KeyError`` reported as "missing or malformed",
#: which points at a corrupt file rather than at an old one.
MANIFEST_VERSION = SemanticVersion(2, 0, 0)


class FreezeError(Exception):
    """The partitions cannot be frozen, or a frozen manifest does not verify."""


@dataclass(frozen=True, slots=True)
class FrozenRecording:
    """One recording, as it was when the corpus was frozen."""

    recording_id: str
    speaker_pseudonym: str
    #: The dialect axis, carried per recording so a held-out slice by variety is
    #: computable from the manifest alone.
    speaker_variety: str
    partition: str
    source_path: str
    #: SHA-256 of the annotation file's bytes. The bytes, not the parsed
    #: record: a re-export that changes formatting without changing meaning
    #: should still be visible, because "nothing meaningful changed" is a
    #: judgement somebody has to make rather than one the tool should make for
    #: them.
    sha256: str
    duration_ms: int
    #: The domain records themselves rather than eleven flattened columns. A
    #: second definition of what consent is would drift from the first, and it
    #: would drift silently: the manifest would keep serialising whatever fields
    #: were listed here on the day somebody added one to ``ConsentRecord``.
    consent: ConsentRecord
    conditions: RecordingConditions
    class_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class FrozenCorpus:
    """The manifest. Written once, read forever."""

    manifest_version: str
    schema_version: str
    taxonomy_version: str
    seed: int
    shares: Mapping[str, float]
    recordings: tuple[FrozenRecording, ...]
    #: Digest over the canonical content below, not over the file's formatting.
    #: A manifest re-serialised with different indentation is the same manifest.
    digest: str

    def of(self, partition: Partition) -> tuple[FrozenRecording, ...]:
        return tuple(r for r in self.recordings if r.partition == partition.value)

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(sorted({r.speaker_pseudonym for r in self.recordings}))


def freeze(
    plan: PartitionPlan,
    recordings: Sequence[AnnotatedRecording],
    sources: Mapping[str, Path],
) -> FrozenCorpus:
    """Turn a usable plan into a manifest.

    Refuses an unusable plan. Freezing a split with a P0 class missing from
    held-out would give the missing measurement the appearance of a decision:
    the manifest would be committed, cited, and nobody reading it later would
    know the class was absent rather than deliberately excluded.
    """
    if not plan.is_usable:
        reasons = "; ".join(f"{f.code}: {f.message}" for f in plan.errors)
        raise FreezeError(
            f"the split has {len(plan.errors)} error(s) and must not be frozen - {reasons}"
        )

    _require_a_scorable_corpus(recordings)

    by_id = {r.recording_id: r for r in recordings}
    entries: list[FrozenRecording] = []

    for contents in plan.partitions:
        for recording_id in contents.recording_ids:
            recording = by_id.get(recording_id)
            if recording is None:
                raise FreezeError(
                    f"the plan places {recording_id!r} but no annotation was given "
                    "for it; the manifest would name a recording nobody can check"
                )
            source = sources.get(recording_id)
            if source is None:
                raise FreezeError(
                    f"no source file for {recording_id!r}. A manifest entry with no "
                    "digest is an entry that cannot answer whether the file changed, "
                    "which is the only question a freeze exists to answer."
                )
            entries.append(
                FrozenRecording(
                    recording_id=recording_id,
                    speaker_pseudonym=recording.speaker.pseudonym,
                    speaker_variety=_required_variety(recording),
                    partition=contents.partition.value,
                    source_path=source.name,
                    sha256=digest_of(source),
                    duration_ms=recording.duration_ms,
                    consent=_freezable_consent(recording),
                    conditions=_freezable_conditions(recording),
                    class_counts=_class_counts(recording),
                )
            )

    entries.sort(key=lambda e: (e.partition, e.recording_id))
    versions = _one_version_each(recordings)

    corpus = FrozenCorpus(
        manifest_version=str(MANIFEST_VERSION),
        schema_version=versions[0],
        taxonomy_version=versions[1],
        seed=plan.seed,
        shares=dict(plan.shares),
        recordings=tuple(entries),
        digest="",
    )
    return _with_digest(corpus)


def _required_variety(recording: AnnotatedRecording) -> str:
    """The speaker's variety, or a refusal naming the recording.

    Named, because "a recording has no variety" sends somebody through forty
    files by hand and the tool already knows which one.
    """
    variety = recording.speaker.variety
    if variety is None:
        raise FreezeError(
            f"{recording.recording_id!r} (speaker {recording.speaker.pseudonym!r}) has no "
            "variety. §14.2 asks for error analysis by dialect, and a null variety in one "
            "row of forty either drops that speaker from every slice or lands them in a "
            "bucket nobody named. It is recorded at recruitment or not at all."
        )
    return variety


def _freezable_consent(recording: AnnotatedRecording) -> ConsentRecord:
    """The consent this recording was collected under, if it may be frozen at all.

    Two refusals, and they are different failures. An absent record means nobody
    wrote down what the participant agreed to, and the manifest would carry a
    blank that reads as "not applicable". A withdrawn one means they did agree
    and then said stop - freezing them into a held-out set makes the withdrawal
    undoable in practice, because the held-out set is the thing every reported
    number is defended with and re-freezing it is the failure §14.4 exists to
    prevent.
    """
    consent = recording.consent
    if consent is None:
        raise FreezeError(
            f"{recording.recording_id!r} declares no consent. `BASELINES.md` §4 requires "
            "the dataset card to record the consent basis, and a manifest entry with a "
            "blank one cannot be told from a recording nobody asked about."
        )
    if not consent.is_active:
        raise FreezeError(
            f"consent for {recording.recording_id!r} was withdrawn on "
            f"{consent.withdrawn_on}. Freezing it into the held-out set makes the "
            "withdrawal undoable in practice: every reported number would be defended "
            "with a manifest naming this recording. Remove the file - the policy's "
            "deletion flow gives 24 hours - and re-split."
        )
    return consent


def _freezable_conditions(recording: AnnotatedRecording) -> RecordingConditions:
    """The capture chain, if it is one a corpus figure can be computed over.

    ``REFERENCE_ENVIRONMENT.md`` states both refusals as requirements and states
    the harm behind them. They are enforced here rather than in
    ``RecordingConditions`` because a botched session is a fact that has to be
    representable in order to be acted on; what must not happen is that it
    reaches a manifest.
    """
    conditions = recording.conditions
    if conditions is None:
        raise FreezeError(
            f"{recording.recording_id!r} declares no recording conditions. "
            "`REFERENCE_ENVIRONMENT.md` gates Pilot A on four of them, and a chain "
            "recorded once in a document cannot say which session drifted."
        )
    if not conditions.virtual_audio_bypassed:
        raise FreezeError(
            f"{recording.recording_id!r} was captured with a virtual audio chain in the "
            "path. NVIDIA Broadcast suppresses breath, creak and the trailing energy of a "
            "cut-off word - the acoustic evidence for cut_off, prolongation and "
            "silent_pause boundaries - and Voicemeeter resamples and mixes. A per-class "
            "figure over that audio is a figure about the enhancer."
        )
    if conditions.channels != 1:
        raise FreezeError(
            f"{recording.recording_id!r} was captured on {conditions.channels} channels. "
            "`REFERENCE_ENVIRONMENT.md` requires mono PCM: both baselines' feature "
            "extractors take one channel, which one is not recorded anywhere, and a "
            "corpus scored half on channel 0 and half on a downmix has no single "
            "signal behind its numbers."
        )
    return conditions


def _require_a_scorable_corpus(recordings: Sequence[AnnotatedRecording]) -> None:
    """Refuse a freeze over a corpus no per-class figure can be computed from.

    Found by running the tooling over real audio. OpenSLR SLR73 is Peruvian and
    read, and its collection protocol re-recorded any take containing
    stuttering - so an honest annotation of it has an empty disfluency tier.
    ``corpus inventory`` said so and exited non-zero. ``corpus split --freeze``
    then froze a held-out set over it and exited zero.

    Both were locally correct and together they were wrong: the split's own
    coverage check deliberately ignores a class absent from the whole corpus,
    on the grounds that it is an inventory problem rather than a split problem,
    and nothing downstream re-asked. An operator who freezes without
    inventorying first gets a manifest, a digest and a committed artifact over
    a corpus that cannot answer anything.

    The threshold here is zero, not ``DEFAULT_MINIMUM_INSTANCES``. How many
    instances a class needs to carry a figure is a parameter the pilot argues
    about with data, and a freeze that refused on somebody's default would be
    enforcing a number nobody has settled. Zero is not that: a per-class figure
    over no instances is undefined at every threshold, and an undefined figure
    reads as a low score.
    """
    counts = inventory(recordings).class_counts
    empty = sorted(event.value for event in p0_speech_events() if not counts.get(event.value, 0))
    if not empty:
        return

    raise FreezeError(
        f"{len(empty)} P0 class(es) have no instances anywhere in this corpus: "
        f"{', '.join(empty)}. Their per-class figures would be undefined, and an "
        "undefined figure is a missing measurement that reads as a low score. Run "
        "`corpus inventory` to see what is short before freezing."
    )


def verify(manifest: FrozenCorpus, sources: Mapping[str, Path]) -> tuple[Finding, ...]:
    """Check a frozen corpus against the files on disk.

    Every discrepancy is an error. There is no such thing as a held-out set
    that has changed a little: the number it produced was produced over the
    bytes named here, and any difference means the number and the data no
    longer refer to each other.
    """
    findings: list[Finding] = []

    recomputed = _with_digest(
        FrozenCorpus(
            manifest_version=manifest.manifest_version,
            schema_version=manifest.schema_version,
            taxonomy_version=manifest.taxonomy_version,
            seed=manifest.seed,
            shares=manifest.shares,
            recordings=manifest.recordings,
            digest="",
        )
    )
    if recomputed.digest != manifest.digest:
        findings.append(
            Finding(
                Severity.ERROR,
                "manifest_digest_mismatch",
                f"the manifest's own digest does not match its contents "
                f"(recorded {manifest.digest[:16]}..., computed "
                f"{recomputed.digest[:16]}...). Somebody edited the manifest",
            )
        )

    for entry in manifest.recordings:
        source = sources.get(entry.recording_id)
        if source is None:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "frozen_recording_missing",
                    f"{entry.recording_id!r} is in the {entry.partition} partition and "
                    "was not supplied. A held-out set with a missing file is a "
                    "different held-out set",
                )
            )
            continue
        if not source.exists():
            findings.append(
                Finding(
                    Severity.ERROR,
                    "frozen_recording_missing",
                    f"{entry.recording_id!r} points at {source}, which does not exist",
                )
            )
            continue
        actual = digest_of(source)
        if actual != entry.sha256:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "frozen_recording_changed",
                    f"{entry.recording_id!r} has changed since the freeze "
                    f"(recorded {entry.sha256[:16]}..., now {actual[:16]}...). Any "
                    "number computed over the frozen set no longer refers to this file",
                )
            )

    frozen_ids = {e.recording_id for e in manifest.recordings}
    for recording_id in sorted(set(sources) - frozen_ids):
        findings.append(
            Finding(
                Severity.ERROR,
                "recording_not_in_manifest",
                f"{recording_id!r} was supplied but is not in the manifest. It joined "
                "the corpus after the freeze, and including it in an evaluation would "
                "score a model on data the freeze does not cover",
            )
        )

    return tuple(findings)


def digest_of(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks.

    Chunked because an annotation file is small today and a media manifest is
    not, and a function that only works on small files is a function somebody
    discovers the limits of at the worst time.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65_536):
            hasher.update(chunk)
    return hasher.hexdigest()


def to_json(manifest: FrozenCorpus) -> str:
    """The manifest as a committable file."""
    return json.dumps(_canonical(manifest), indent=2, ensure_ascii=False) + "\n"


def from_json(text: str) -> FrozenCorpus:
    """Read a manifest back. Refuses one this tool cannot interpret."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise FreezeError(f"the manifest is not valid JSON: {error}") from error

    try:
        version = SemanticVersion.parse(str(payload["manifest_version"]))
    except Exception as error:
        raise FreezeError(
            f"the manifest declares no readable version: {payload.get('manifest_version')!r}"
        ) from error

    if not version.is_compatible_with(MANIFEST_VERSION):
        raise FreezeError(
            f"the manifest was written under version {version}; this tool reads "
            f"{MANIFEST_VERSION}. Reading it anyway would compare two different "
            "manifest shapes and report the difference as a changed corpus."
        )

    try:
        return FrozenCorpus(
            manifest_version=str(version),
            schema_version=str(payload["schema_version"]),
            taxonomy_version=str(payload["taxonomy_version"]),
            seed=int(payload["seed"]),
            shares={str(k): float(v) for k, v in payload["shares"].items()},
            recordings=tuple(
                FrozenRecording(
                    recording_id=str(item["recording_id"]),
                    speaker_pseudonym=str(item["speaker_pseudonym"]),
                    speaker_variety=str(item["speaker_variety"]),
                    partition=str(item["partition"]),
                    source_path=str(item["source_path"]),
                    sha256=str(item["sha256"]),
                    duration_ms=int(item["duration_ms"]),
                    consent=_consent_from(item["consent"]),
                    conditions=_conditions_from(item["conditions"]),
                    class_counts={str(k): int(v) for k, v in item["class_counts"].items()},
                )
                for item in payload["recordings"]
            ),
            digest=str(payload["digest"]),
        )
    except (KeyError, TypeError, ValueError, SchemaViolation) as error:
        raise FreezeError(f"the manifest is missing or malformed: {error}") from error


def _consent_from(payload: Mapping[str, object]) -> ConsentRecord:
    """Rebuild the record rather than trusting the file.

    Through the constructor, so a hand-edited manifest gets the same refusals a
    hand-edited annotation does. Reading these back as plain strings would let a
    manifest declare a consent basis that is not a published one, and the digest
    check would happily confirm that nobody had altered it since.
    """
    return ConsentRecord(
        basis=ConsentBasis(str(payload["basis"])),
        policy_version=SemanticVersion.parse(str(payload["policy_version"])),
        granted_on=date.fromisoformat(str(payload["granted_on"])),
        covers_video=_strict_bool(payload["covers_video"], "covers_video"),
    )


def _conditions_from(payload: Mapping[str, object]) -> RecordingConditions:
    return RecordingConditions(
        microphone=str(payload["microphone"]),
        sample_rate_hz=_strict_int(payload["sample_rate_hz"], "sample_rate_hz"),
        bit_depth=_strict_int(payload["bit_depth"], "bit_depth"),
        channels=_strict_int(payload["channels"], "channels"),
        virtual_audio_bypassed=_strict_bool(
            payload["virtual_audio_bypassed"], "virtual_audio_bypassed"
        ),
        room_notes=str(payload["room_notes"]),
    )


def _strict_bool(value: object, name: str) -> bool:
    """A JSON boolean, and nothing that merely behaves like one.

    ``bool("false")`` is ``True``, and so is ``bool("no")``. A manifest edited
    by hand into ``"virtual_audio_bypassed": "false"`` would then verify as
    bypassed, which is the exact reading the field exists to make impossible.
    """
    if not isinstance(value, bool):
        raise FreezeError(f"{name} must be a JSON boolean, got {value!r}")
    return value


def _strict_int(value: object, name: str) -> int:
    """A JSON integer. ``bool`` is not one, whatever Python thinks.

    ``isinstance(True, int)`` holds, so a manifest reading ``"channels": true``
    would coerce to 1 and pass the mono check on the strength of a typo.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise FreezeError(f"{name} must be a JSON integer, got {value!r}")
    return value


def _canonical(manifest: FrozenCorpus) -> dict[str, object]:
    """The manifest's content, in one fixed shape.

    The digest is computed over this, so field order and float formatting are
    part of the contract. Sorted keys and a fixed key order everywhere, so that
    two manifests with the same content have the same digest regardless of
    which Python wrote them.
    """
    return {
        "manifest_version": manifest.manifest_version,
        "schema_version": manifest.schema_version,
        "taxonomy_version": manifest.taxonomy_version,
        "seed": manifest.seed,
        "shares": dict(sorted(manifest.shares.items())),
        "recordings": [
            {
                "recording_id": entry.recording_id,
                "speaker_pseudonym": entry.speaker_pseudonym,
                "speaker_variety": entry.speaker_variety,
                "partition": entry.partition,
                "source_path": entry.source_path,
                "sha256": entry.sha256,
                "duration_ms": entry.duration_ms,
                # No `withdrawn_on`. The freeze refuses a withdrawn recording, so
                # the key would be null in every row ever written, and a field
                # that is always null tells a reader that withdrawal is tracked
                # here. It is not: a withdrawal after the freeze is enacted by
                # deleting the file, and `verify` then reports the held-out set
                # as changed - which it is.
                "consent": {
                    "basis": entry.consent.basis.value,
                    "policy_version": str(entry.consent.policy_version),
                    "granted_on": entry.consent.granted_on.isoformat(),
                    "covers_video": entry.consent.covers_video,
                },
                "conditions": {
                    "microphone": entry.conditions.microphone,
                    "sample_rate_hz": entry.conditions.sample_rate_hz,
                    "bit_depth": entry.conditions.bit_depth,
                    "channels": entry.conditions.channels,
                    "virtual_audio_bypassed": entry.conditions.virtual_audio_bypassed,
                    "room_notes": entry.conditions.room_notes,
                },
                "class_counts": dict(sorted(entry.class_counts.items())),
            }
            for entry in manifest.recordings
        ],
        "digest": manifest.digest,
    }


def _with_digest(manifest: FrozenCorpus) -> FrozenCorpus:
    content = _canonical(manifest)
    # The digest cannot cover itself.
    content.pop("digest")
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return FrozenCorpus(
        manifest_version=manifest.manifest_version,
        schema_version=manifest.schema_version,
        taxonomy_version=manifest.taxonomy_version,
        seed=manifest.seed,
        shares=manifest.shares,
        recordings=manifest.recordings,
        digest=hashlib.sha256(encoded).hexdigest(),
    )


def _class_counts(recording: AnnotatedRecording) -> dict[str, int]:
    counts: dict[str, int] = {}
    for annotation in recording.disfluencies:
        counts[annotation.event_type.value] = counts.get(annotation.event_type.value, 0) + 1
    return dict(sorted(counts.items()))


def _one_version_each(recordings: Sequence[AnnotatedRecording]) -> tuple[str, str]:
    """Both versions, and a refusal if the corpus does not agree on them.

    A corpus assembled across a taxonomy change is a corpus whose classes mean
    two things, and freezing it would put one version string on a manifest
    describing two.
    """
    schemas = {str(r.schema_version) for r in recordings}
    taxonomies = {str(r.taxonomy_version) for r in recordings if r.taxonomy_version}

    if len(schemas) != 1:
        raise FreezeError(
            f"the recordings were written under schema versions {sorted(schemas)}. "
            "One manifest cannot describe two record shapes."
        )
    if len(taxonomies) != 1:
        raise FreezeError(
            f"the recordings were annotated under taxonomy versions "
            f"{sorted(taxonomies) or 'none'}. A corpus assembled across a taxonomy "
            "change is a corpus whose classes mean two things."
        )
    return next(iter(schemas)), next(iter(taxonomies))
