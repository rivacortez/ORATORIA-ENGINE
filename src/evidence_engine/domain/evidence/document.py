"""The evidence document: everything observed, ranked by nobody.

This is what §7.1's ``GET /v1/sessions/{id}/result`` returns and what US-013
describes: a complete evidence document a consumer's own decision engine can
process. Two requirements define its shape and they pull in opposite
directions, which is why both are stated here rather than left to the
serializer.

FR-029 forbids the document from carrying a score, a ranking or a top-k, and
§17 makes the mitigation explicit: the contract invariant is
``ranking_authority=none``. The pressure to break this is real and comes from
the most reasonable place imaginable - a consumer asks for "just a severity
field, so we do not each reimplement it" - and granting it would move the
ranking authority into a service that, by §3 driver 8, only observes.

FR-029 also forbids the opposite failure: the result "must not hide
low-confidence evidence". A document that silently dropped uncertain findings
would be making a selection decision too, just in the other direction, and it
would corrupt the recall denominator of every evaluation built on it.

So: everything observed goes in, nothing is ordered by importance, and the
consumer decides what matters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final

from evidence_engine.domain.evidence.cooccurrence import MultimodalCooccurrence
from evidence_engine.domain.quality.assessment import QualityReport
from evidence_engine.domain.shared.errors import DomainError
from evidence_engine.domain.shared.identifiers import (
    ConfigurationSnapshotId,
    ModelVersionId,
    RunId,
    SessionId,
)
from evidence_engine.domain.shared.provenance import Modality, SemanticVersion
from evidence_engine.domain.shared.taxonomy import TAXONOMY_VERSION
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading
from evidence_engine.domain.transcript.transcript import Transcript
from evidence_engine.domain.visual_events.events import VisualEvent


class ResultViolation(DomainError):
    """The result document was about to carry something FR-029 forbids."""


#: Field names that would make this service a ranking authority. Checked
#: against the assembled payload before publication. A denylist is the right
#: shape here - unlike the taxonomy, where the space of prohibited labels is
#: unbounded, the space of ranking vocabulary is small and stable, and the
#: check exists to catch an accidental passthrough rather than a novel concept.
FORBIDDEN_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "rank",
        "ranking",
        "ranked",
        "score",
        "scores",
        "priority",
        "priority_score",
        "severity",
        "top_k",
        "topk",
        "selected",
        "recommendation",
        "recommendations",
        "intervention",
        "interventions",
        "exercise",
        "exercises",
    }
)


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """Which versions produced this document (NFR-014, NFR-015, US-008).

    Every model that contributed is listed by modality, so a reader can tell
    which half of a result changed when a single runtime is promoted. Without
    that split, a canary on the visual model would appear to have moved the
    speech metrics too.
    """

    pipeline_version: SemanticVersion
    schema_version: SemanticVersion
    taxonomy_version: SemanticVersion
    configuration: ConfigurationSnapshotId
    models: Mapping[Modality, ModelVersionId]

    def __post_init__(self) -> None:
        if self.taxonomy_version != TAXONOMY_VERSION:
            # A document produced under a different taxonomy is legitimate -
            # historical results keep their own version. What is not legitimate
            # is this process claiming a version it is not running, so the
            # mismatch is surfaced rather than normalized away.
            raise ResultViolation(
                f"manifest declares taxonomy {self.taxonomy_version} but this build "
                f"runs {TAXONOMY_VERSION}; historical documents must be read with "
                "their own manifest, not re-stamped with the current one"
            )


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    """The complete, unranked result for one processing run."""

    #: §17's contract invariant, structurally. A `ClassVar`, so it cannot be
    #: constructed with any other value, and a consumer reading it knows the
    #: absence of ranking is guaranteed rather than merely observed today.
    ranking_authority: ClassVar[str] = "none"

    session_id: SessionId
    run_id: RunId
    manifest: ProvenanceManifest
    transcript: Transcript
    quality: QualityReport
    speech_events: tuple[SpeechEvent, ...] = field(default_factory=tuple)
    visual_events: tuple[VisualEvent, ...] = field(default_factory=tuple)
    prosody: tuple[ProsodyReading, ...] = field(default_factory=tuple)
    cooccurrences: tuple[MultimodalCooccurrence, ...] = field(default_factory=tuple)

    # -- queries ----------------------------------------------------------

    @property
    def has_speech_evidence(self) -> bool:
        return bool(self.transcript.tokens) or bool(self.speech_events)

    @property
    def has_visual_evidence(self) -> bool:
        return bool(self.visual_events)

    def unavailable_indicators(self) -> tuple[ProsodyReading, ...]:
        """Readings that carry a reason instead of a number.

        Part of the published result, not an error list. US-004 requires
        unavailable indicators to show their technical reason, which is only
        possible if they survive into the document at all.
        """
        return tuple(reading for reading in self.prosody if not reading.is_available)

    def events_from(self, modality: Modality) -> int:
        if modality is Modality.AUDIO:
            return len(self.speech_events)
        if modality is Modality.VIDEO:
            return len(self.visual_events)
        return len(self.cooccurrences)


def assert_carries_no_ranking(payload: Mapping[str, object]) -> None:
    """Refuse a serialized result that has grown a ranking field (FR-029).

    Called on the assembled payload just before it leaves the service. Walks
    nested structures because the field that breaks this will not be added at
    the top level - it will arrive inside an event, forwarded from a model
    runtime that happened to include its own confidence ordering.
    """
    offenders = sorted(_forbidden_keys_in(payload))
    if offenders:
        raise ResultViolation(
            f"result payload carries ranking fields {offenders}; FR-029 keeps this "
            "service an observer - selecting and ordering findings belongs to the "
            "consuming application"
        )


def _forbidden_keys_in(node: object, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(node, Mapping):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key).lower() in FORBIDDEN_RESULT_KEYS:
                found.add(here)
            found |= _forbidden_keys_in(value, here)
    elif isinstance(node, Sequence) and not isinstance(node, str | bytes):
        for index, value in enumerate(node):
            found |= _forbidden_keys_in(value, f"{path}[{index}]")
    return found
