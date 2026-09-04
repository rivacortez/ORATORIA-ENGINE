"""Provenance: what produced this, under which rules.

NFR-014 requires every derived event to expose its source modality, interval,
confidence, model version, taxonomy version and evidence reference. NFR-015
requires that the same input, artifact, configuration and seed reproduce an
equivalent result. Neither is achievable if the versions live in a config file
somewhere and the result only carries numbers, because six months later nobody
can tell whether a metric moved because the student improved or because a
threshold was retuned - the exact failure the risk table names as "model
updates break longitudinal comparison".

So provenance is attached to the evidence itself, not to the deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evidence_engine.domain.shared.errors import InvalidIdentifier
from evidence_engine.domain.shared.identifiers import (
    ConfigurationSnapshotId,
    EvidenceRef,
    ModelVersionId,
)


class Modality(StrEnum):
    """Which channel of observation produced a piece of evidence.

    Recorded on every event because QA-02 requires one modality to fail
    without invalidating the other, and that is only reportable if each finding
    says where it came from. A missing camera must be legible as a missing
    camera, never as a change in performance.
    """

    AUDIO = "audio"
    VIDEO = "video"
    #: Derived from two or more modalities - only co-occurrences (FR-027).
    MULTIMODAL = "multimodal"


@dataclass(frozen=True, slots=True, order=True)
class SemanticVersion:
    """A published, immutable version of a taxonomy, schema or threshold set.

    US-008 requires published versions to be immutable and historical results
    to be unaffected by later default changes. Comparison is defined so that
    the compatibility question NFR-017 asks - "is this consumer still within
    one major version?" - has an answer in code rather than in a changelog.
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise InvalidIdentifier(f"version components must be non-negative: {self}")

    @classmethod
    def parse(cls, text: str) -> SemanticVersion:
        parts = text.strip().split(".")
        if len(parts) != 3:
            raise InvalidIdentifier(f"expected 'major.minor.patch', got {text!r}")
        try:
            major, minor, patch = (int(p) for p in parts)
        except ValueError as exc:
            raise InvalidIdentifier(f"non-numeric version component in {text!r}") from exc
        return cls(major, minor, patch)

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """Backward compatibility within one major version (NFR-017)."""
        return self.major == other.major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True, slots=True)
class Provenance:
    """The full answer to "where did this finding come from?".

    Every field is required. An optional model version would be filled in as
    ``None`` on exactly the code paths where it matters most - a fallback
    adapter, a degraded run - and the result would claim provenance it does not
    have. Component §5 puts it plainly for the Evidence Ledger: preserve
    immutable provenance.
    """

    modality: Modality
    model_version: ModelVersionId
    taxonomy_version: SemanticVersion
    configuration: ConfigurationSnapshotId
    evidence_ref: EvidenceRef
