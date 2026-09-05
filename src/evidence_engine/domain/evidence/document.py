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

Two guards enforce that, because a ranking can arrive in two shapes. A *field*
- ``overall_score``, ``percentile``, ``severity`` - is caught by walking the
serialized payload against ``RANKING_VOCABULARY``. An *order* - the same fields,
sorted by confidence descending - carries no new key at all and is caught in
this class's constructor, which refuses any sequence not in time order. The
second is the one that would survive review: it adds nothing to the schema, so
it does not show up as a schema change.
"""

from __future__ import annotations

import re
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


#: Vocabulary that turns a published field into a ranking claim (FR-029).
#:
#: Matched against the *word tokens* of every key in the serialized payload,
#: not against whole key names. The first version of this check compared whole
#: keys, so ``score`` was refused and ``overall_score`` was published - and
#: ``overall_score`` is the shape an accidental passthrough actually takes,
#: because a runtime that forwards its own ordering names it after what it
#: ordered. Whole-key matching only ever caught a field somebody would have
#: had to add on purpose.
#:
#: A denylist is the right instrument here, unlike in the taxonomy: the space
#: of prohibited *labels* is unbounded, so the taxonomy uses an allowlist and
#: keeps ``PROHIBITED_CONCEPTS`` as a tripwire behind it. The vocabulary of
#: ranking is small and stable, and there is no allowlist of published keys to
#: sit behind - the payload's shape is whatever the serializer writes.
#:
#: Deliberately blunt. A false positive costs a rename and a short argument; a
#: false negative costs the separation of authority that §3 driver 8 exists to
#: protect, and it costs it silently.
RANKING_VOCABULARY: Final[frozenset[str]] = frozenset(
    {
        # Ordinal position, however it is spelled.
        "rank",
        "ranking",
        "ranked",
        "ordinal",
        "percentile",
        "quartile",
        "decile",
        "leaderboard",
        # A magnitude standing in for importance.
        "score",
        "scores",
        "scored",
        "scoring",
        "rating",
        "rated",
        "priority",
        "severity",
        "severe",
        # Comparison against another observation, or another speaker.
        "better",
        "worse",
        "best",
        "worst",
        # Selection: publishing a subset is the other half of ranking, and
        # FR-029 forbids hiding low-confidence evidence just as firmly as it
        # forbids ordering it.
        "top",
        "topk",
        "selected",
        # An order asserted as meaningful. `order`/`sort` are broad on purpose:
        # no key in the published payload needs either word, so the cost of
        # keeping them is zero until somebody adds a field that does.
        "order",
        "ordered",
        "ordering",
        "sort",
        "sorted",
        # Acting on the observation, which belongs to the consumer's engine.
        "recommendation",
        "recommendations",
        "recommended",
        "intervention",
        "interventions",
        "exercise",
        "exercises",
    }
)

#: Published keys that tokenise into the denylist and must survive it.
#:
#: The same exemption the taxonomy tripwire needs and for the same reason: the
#: field that publishes the guarantee is named after the thing it forbids.
#: Listed with the reason, because an unexplained exemption is how a guard gets
#: hollowed out one commit at a time.
_EXEMPT_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        # §17's contract invariant, on the wire, always the string "none". This
        # is the one key whose presence proves the rule rather than breaking it.
        "ranking_authority",
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

    def __post_init__(self) -> None:
        """Refuse a document whose evidence is ordered by anything but time.

        The denylist below catches a ranking that arrives as a *field*. This
        catches the one that arrives as an *order*, which no denylist can see:
        sort the events by confidence descending, publish no new key at all,
        and the consumer reads a ranked list. Position 0 means "first" to a
        reader, and "first" becomes "worst" the moment the order is anything
        other than the clock.

        Checked here, in the constructor, rather than at completion or before
        the database write, because those two are not the only ways a document
        comes into existence. It is also rebuilt from stored rows on every
        ``GET /result``, and that path renders straight to the wire without
        passing through either. The constructor is the only chokepoint all of
        them share, and it costs one pass over sequences that are already
        materialised.

        ``cooccurrences`` is not checked. A pair carries two event ids and a
        distance, not intervals, so this object cannot tell what time order
        would even be without re-reading the events. ``correlate`` is proven
        deterministic separately, by ``test_correlation_is_deterministic``.
        """
        _require_time_ordered(
            "speech_events",
            [(e.interval.start.ms, e.interval.end.ms, e.id.value) for e in self.speech_events],
        )
        _require_time_ordered(
            "visual_events",
            [(e.interval.start.ms, e.interval.end.ms, e.id.value) for e in self.visual_events],
        )
        _require_time_ordered(
            "prosody",
            [(r.window.start.ms, r.window.end.ms, r.indicator.value) for r in self.prosody],
        )
        # **Timed tokens only.** A word whose alignment failed has no interval,
        # and asking one for a boundary raises `FabricatedValue` by design - so
        # this used to abort document construction on any transcript containing
        # an unplaced word, which is exactly the case the placement union was
        # added to support. Nothing caught it because no test had an unplaced
        # token reach an `EvidenceDocument`.
        #
        # Ordering the placed tokens is still the right check: they are what a
        # consumer reads against the clock, and their relative order is what
        # a non-deterministic assembly would disturb. The unplaced ones carry a
        # lexical sequence instead, which `Transcript` orders and this object
        # has no clock-based claim to make about.
        _require_time_ordered(
            "transcript.tokens",
            [
                (t.interval.start.ms, t.interval.end.ms, t.id.value)
                for t in self.transcript.tokens
                if t.is_timed
            ],
        )

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

    Takes the real published payload - the dict the serializer produced - and
    not a summary assembled for the occasion. An earlier version of this call
    site built a two-key stub of its own and walked that; the stub could not
    fail, because the code building it only ever wrote ``id`` and ``type``. A
    check that cannot fail is worse than no check, because it makes the claim
    in the README look enforced.

    Walks nested structures because the field that breaks this will not appear
    at the top level. It will arrive inside an event, forwarded from a model
    runtime that included its own confidence ordering.

    Only keys are inspected, not values. A value naming a prohibited class is
    already refused by the taxonomy allowlist and by the domain constructors it
    had to pass; a *key* is constrained by nothing at all except this.
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
            if _is_ranking_key(str(key)):
                found.add(here)
            found |= _forbidden_keys_in(value, here)
    elif isinstance(node, Sequence) and not isinstance(node, str | bytes):
        for index, value in enumerate(node):
            found |= _forbidden_keys_in(value, f"{path}[{index}]")
    return found


def _is_ranking_key(key: str) -> bool:
    if key.lower() in _EXEMPT_RESULT_KEYS:
        return False
    return bool(_tokens(key) & RANKING_VOCABULARY)


def _tokens(identifier: str) -> set[str]:
    """Split an identifier into lowercase word tokens.

    ``snake_case``, ``camelCase`` and ``SCREAMING_CASE`` reduce to the same
    set, so ``overallScore``, ``overall_score`` and ``OVERALL_SCORE`` are
    equally caught.

    Underscores are separated first, and only then are case boundaries. Doing
    it the other way round - inserting a space before every capital, then
    stripping underscores - shreds ``OVERALL_SCORE`` into eleven single
    letters, and a set of single letters intersects no denylist. The taxonomy
    tripwire in ``tests/invariants/test_observable_only.py`` splits in that
    order and its docstring claims ``ANXIETY_SCORE`` is caught; it is not.
    That tripwire survives because it tests enum *values* as well as names and
    the values are ``snake_case``, so it has never had to rely on the broken
    half. This one has no second string to fall back on, so it splits
    correctly and the divergence is recorded here rather than copied.

    The second alternative separates an acronym from the word after it
    (``HTTPScore``); the third separates a trailing number (``top3``).
    """
    spaced = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[0-9])",
        " ",
        identifier.replace("_", " "),
    )
    return {token.lower() for token in spaced.split() if token}


def _require_time_ordered(field_name: str, keys: Sequence[tuple[int, int, str]]) -> None:
    """Refuse a sequence not in the canonical published order (FR-029).

    The key is ``(start_ms, end_ms, identifier)`` - the same one
    ``transcript.build`` already imposes on word tokens, reused rather than
    reinvented so that two orderings cannot drift apart. The tie-break on the
    identifier is not cosmetic: two events can share a start at a window seam,
    and NFR-015 wants the same input to publish the same order on the rerun.

    One pass, rather than a comparison against ``sorted(keys)``. Building a
    second sorted copy of a ten-thousand-token transcript to answer a yes/no
    question is work with no reader, and this runs on the completion path and
    on every result read.
    """
    for index in range(1, len(keys)):
        if keys[index] < keys[index - 1]:
            previous, current = keys[index - 1], keys[index]
            raise ResultViolation(
                f"{field_name} is out of published time order at position {index}: "
                f"{current} follows {previous}. FR-029 gives this service no ranking "
                "authority, and a list order is a claim whether or not it is labelled "
                "one - a consumer reads position 0 as 'first', and 'first' reads as "
                "'worst' the moment the order is anything but the clock"
            )
