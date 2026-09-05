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

The manifest is the machine-readable half of the dataset card §4 asks for. The
half a human writes - consent basis, recording conditions, why these speakers -
is prose and belongs beside it, not in it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from corpus.partition.split import Partition, PartitionPlan
from corpus.schema.records import AnnotatedRecording
from corpus.schema.validation import Finding, Severity
from evidence_engine.domain.shared.provenance import SemanticVersion

#: Bumped when the manifest's shape changes. Separate from the annotation
#: schema version: a manifest can gain a field without any annotation record
#: changing, and a reader has to be able to tell the two apart.
MANIFEST_VERSION = SemanticVersion(1, 0, 0)


class FreezeError(Exception):
    """The partitions cannot be frozen, or a frozen manifest does not verify."""


@dataclass(frozen=True, slots=True)
class FrozenRecording:
    """One recording, as it was when the corpus was frozen."""

    recording_id: str
    speaker_pseudonym: str
    partition: str
    source_path: str
    #: SHA-256 of the annotation file's bytes. The bytes, not the parsed
    #: record: a re-export that changes formatting without changing meaning
    #: should still be visible, because "nothing meaningful changed" is a
    #: judgement somebody has to make rather than one the tool should make for
    #: them.
    sha256: str
    duration_ms: int
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
                    partition=contents.partition.value,
                    source_path=source.name,
                    sha256=digest_of(source),
                    duration_ms=recording.duration_ms,
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
                    partition=str(item["partition"]),
                    source_path=str(item["source_path"]),
                    sha256=str(item["sha256"]),
                    duration_ms=int(item["duration_ms"]),
                    class_counts={str(k): int(v) for k, v in item["class_counts"].items()},
                )
                for item in payload["recordings"]
            ),
            digest=str(payload["digest"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FreezeError(f"the manifest is missing or malformed: {error}") from error


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
                "partition": entry.partition,
                "source_path": entry.source_path,
                "sha256": entry.sha256,
                "duration_ms": entry.duration_ms,
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
