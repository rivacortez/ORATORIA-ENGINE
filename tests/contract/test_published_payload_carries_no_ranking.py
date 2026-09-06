"""FR-029, checked against the payload rather than against the class list.

The engine's flagship claim is that it has no ranking authority. Until this
file existed, four things asserted it - README line 26, ADR-007, the
``ranking_authority`` ``ClassVar`` and a reflection tripwire over
``PROHIBITED_CONCEPTS`` - and none of them looked at what the service actually
emits. The ``ClassVar`` proves one field's value. The tripwire walks the domain
*package* by reflection, so it sees dataclass fields and enum members; it does
not see the dict the serializer builds, and the serializer is a different file
with different names in it.

So this walks the real thing: ``render_document`` run over a fully populated
document, which is the dict ``GET /v1/sessions/{id}/result`` returns and the
dict stored as the JSONB row. Every branch of the serializer is exercised, and
that is checked rather than assumed - a payload with no unavailable indicator
in it would never reach ``_render_indicator``'s second branch, and the walk
would report a clean result about half the code.

Two shapes of ranking are tested, because they fail differently:

*A field.* ``overall_score``, ``percentile``, ``better_than``. Caught by the
denylist. The interesting case is not ``score`` - nobody adds that by accident
- but ``overall_score``, which the first version of the denylist published,
because it compared whole key names and only ever caught the field somebody
would have had to add deliberately.

*An order.* The same events, sorted by confidence descending. No new key, no
schema change, nothing for a denylist to find, and a consumer reading position
0 as "the first thing that happened" now reads it as "the worst thing that
happened". The fixture below is built so that this is detectable: confidence
rises with time, so any sort by confidence puts the payload out of clock order
and the ordering sweep fails. That property is itself asserted, because a
fixture where the two orders coincide would make the ordering tests pass while
proving nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Any, cast

import pytest

from evidence_engine.adapters.inbound.rest.serialization import render_document
from evidence_engine.domain.evidence.document import (
    RANKING_VOCABULARY,
    EvidenceDocument,
    ResultViolation,
    assert_carries_no_ranking,
)
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.identifiers import EventId, TenantId
from evidence_engine.domain.shared.provenance import Provenance
from evidence_engine.domain.shared.taxonomy import PROHIBITED_CONCEPTS, SpeechEventType
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.events import SpeechEvent

pytestmark = pytest.mark.contract

SCHEMA_VERSION = "1.0.0"

# The fully populated `EvidenceDocument` this whole file walks is the `audio`,
# `video` and `document` fixtures in `tests/contract/conftest.py` - shared
# with `test_remote_client.py`'s round-trip conformance test rather than
# redefined here.


@pytest.fixture
def payload(document: EvidenceDocument) -> dict[str, Any]:
    """The dict the REST adapter publishes. Not a description of it."""
    return render_document(document, schema_version=SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# Walking the payload
# ---------------------------------------------------------------------------


def _mappings_in(node: object) -> Iterator[dict[str, Any]]:
    """Every mapping in the tree, root first, in a stable order."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _mappings_in(value)
    elif isinstance(node, list):
        for value in node:
            yield from _mappings_in(value)


def _lists_of_intervals(node: object, path: str = "") -> Iterator[tuple[str, list[Any]]]:
    """Every list in the payload whose elements carry a time window.

    Discovered by shape rather than named one by one, so a list added to the
    serializer tomorrow is swept without anyone remembering to add it here.
    That is the whole point: the failure this guards against arrives with a
    new field, and a test that enumerates today's fields cannot see it.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _lists_of_intervals(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        if node and all(isinstance(item, dict) and "start_ms" in item for item in node):
            yield path, node
        for index, value in enumerate(node):
            yield from _lists_of_intervals(value, f"{path}[{index}]")


def test_the_published_payload_carries_no_ranking_field(payload: dict[str, Any]) -> None:
    assert_carries_no_ranking(payload)


def test_the_published_payload_names_no_prohibited_concept(payload: dict[str, Any]) -> None:
    """The taxonomy tripwire's vocabulary, applied to the emitted document.

    ``test_observable_only.py`` walks the domain package by reflection, which
    covers dataclass fields and enum members. The serializer is neither: it
    writes literal key names that no dataclass has to declare. Reusing
    ``PROHIBITED_CONCEPTS`` here rather than restating it keeps NFR-024 to one
    list - two that can disagree would be worse than one, because each would
    look like a guard while the gap sat between them.
    """
    offenders = [
        key
        for mapping in _mappings_in(payload)
        for key in mapping
        if set(key.lower().replace("_", " ").split()) & PROHIBITED_CONCEPTS
    ]

    assert not offenders, f"NFR-024 forbids these published keys: {sorted(offenders)}"


def test_the_walk_reaches_every_mapping_in_the_payload(payload: dict[str, Any]) -> None:
    """A clean result means nothing until the walk is known to reach everything.

    Without this, a walker that stopped at the second level would pass every
    other test in this file. Each mapping in the real payload gets a ranking
    key planted in it in turn, and the walk has to find it - which also proves
    the walk descends through lists, where the realistic offender lives.
    """
    total = len(list(_mappings_in(payload)))
    assert total > 10, "the fixture is not populated enough to be a real walk"

    for index in range(total):
        probe = deepcopy(payload)
        list(_mappings_in(probe))[index]["overall_score"] = 0.7

        with pytest.raises(ResultViolation, match="FR-029"):
            assert_carries_no_ranking(probe)


@pytest.mark.parametrize(
    "planted",
    [
        # The keys the whole-key denylist published. Each is a plausible name
        # for a field a runtime forwards, and none is an exact denylist member.
        "overall_score",
        "severity_rank",
        "priority_order",
        "percentile",
        "better_than",
        "confidence_ranking",
        "sorted_by",
        "topScore",
        "OVERALL_SEVERITY",
    ],
)
def test_a_qualified_ranking_key_is_refused(payload: dict[str, Any], planted: str) -> None:
    probe = deepcopy(payload)
    probe["speech_events"][0][planted] = 1

    with pytest.raises(ResultViolation, match="FR-029"):
        assert_carries_no_ranking(probe)


def test_the_published_invariant_field_survives_its_own_denylist(
    payload: dict[str, Any],
) -> None:
    """``ranking_authority`` tokenises to a denied word and must be published.

    The exemption is the point of the field: §17 wants the guarantee on the
    wire so a consumer can read it, not merely enforced where nobody can see
    it. A guard that deleted the statement of the rule would be obeying the
    letter of FR-029 and destroying its purpose.
    """
    assert payload["ranking_authority"] == "none"
    assert_carries_no_ranking(payload)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_every_published_sequence_is_in_clock_order(payload: dict[str, Any]) -> None:
    swept = dict(_lists_of_intervals(payload))
    assert set(swept) >= {
        "transcript.tokens",
        "speech_events",
        "visual_events",
        "prosody",
        "quality.assessments",
        "quality.availability",
    }, f"the sweep missed a published sequence: found {sorted(swept)}"

    for path, items in swept.items():
        starts = [item["start_ms"] for item in items]
        assert starts == sorted(starts), (
            f"{path} is not in clock order: {starts}. A consumer reads position 0 as "
            "the first thing that happened; any other order is a ranking the engine "
            "has no authority to assert (FR-029)"
        )


def test_the_fixture_can_tell_clock_order_from_confidence_order(
    payload: dict[str, Any],
) -> None:
    """Proof that the sweep above is not vacuous.

    If the document happened to be in the same order under both keys, every
    ordering assertion here would hold against a serializer that sorted by
    severity, and the suite would report a guarantee it never checked.
    """
    for path in ("transcript.tokens", "speech_events", "visual_events"):
        items = dict(_lists_of_intervals(payload))[path]
        by_clock = [item["start_ms"] for item in items]
        by_confidence = [
            item["start_ms"] for item in sorted(items, key=lambda i: i["confidence"], reverse=True)
        ]

        assert by_clock != by_confidence, f"{path} cannot distinguish the two orders"


def test_a_document_ordered_by_confidence_is_refused(document: EvidenceDocument) -> None:
    """The realistic regression, at the only chokepoint every path shares.

    A consumer asks for "just the important ones first"; the change is one
    ``sorted`` call and it adds no field, so a denylist over the payload sees
    nothing and a schema diff shows nothing.
    """
    ranked = sorted(document.speech_events, key=lambda e: e.confidence.value, reverse=True)

    with pytest.raises(ResultViolation, match="out of published time order"):
        EvidenceDocument(
            session_id=document.session_id,
            run_id=document.run_id,
            manifest=document.manifest,
            transcript=document.transcript,
            quality=document.quality,
            speech_events=tuple(ranked),
            visual_events=document.visual_events,
            prosody=document.prosody,
            cooccurrences=document.cooccurrences,
        )


def test_events_sharing_a_start_are_ordered_by_identifier(
    document: EvidenceDocument, audio: Provenance
) -> None:
    """Two events can start together at a window seam.

    Without the identifier tie-break the published order of those two would
    follow whatever order they were assembled in, which NFR-015 requires to be
    reproducible and which arrival timing decides.
    """
    first = SpeechEvent(
        id=EventId("speech-a"),
        type=SpeechEventType.FILLED_PAUSE,
        interval=Interval.of(400, 900),
        confidence=Confidence.calibrated(0.7),
        provenance=audio,
        is_final=True,
    )
    second = SpeechEvent(
        id=EventId("speech-b"),
        type=SpeechEventType.SILENT_PAUSE,
        interval=Interval.of(400, 900),
        confidence=Confidence.calibrated(0.7),
        provenance=audio,
        is_final=True,
    )

    with pytest.raises(ResultViolation, match="out of published time order"):
        EvidenceDocument(
            session_id=document.session_id,
            run_id=document.run_id,
            manifest=document.manifest,
            transcript=document.transcript,
            quality=document.quality,
            speech_events=(second, first),
        )


# ---------------------------------------------------------------------------
# The two denylists
# ---------------------------------------------------------------------------


def test_the_two_denylists_answer_different_requirements(payload: dict[str, Any]) -> None:
    """They must not be merged, and they must not overlap.

    ``PROHIBITED_CONCEPTS`` is NFR-024: no inferred inner state. This one is
    FR-029: no ranking. A shared term would mean one requirement's exemption
    list silently weakened the other's guard - the exemption that lets
    ``ranking_authority`` be published must never become an exemption that lets
    an emotion field through.
    """
    assert not (RANKING_VOCABULARY & PROHIBITED_CONCEPTS)


# ---------------------------------------------------------------------------
# Where the walk runs in production
# ---------------------------------------------------------------------------


async def test_the_payload_is_walked_before_it_is_written(
    document: EvidenceDocument,
) -> None:
    """The write path refuses a ranking payload without reaching the database.

    The session factory is ``None`` on purpose. If the walk ever moves to after
    the write - or is removed - this test stops raising ``ResultViolation`` and
    starts raising whatever ``None`` does when a transaction is opened on it,
    which is the failure this asserts against.

    The repository, rather than the use case, is where this can live at all:
    the serializer belongs to the inbound REST adapter, contract C6 forbids the
    application importing it, and the repository already receives it as an
    injected ``DocumentRenderer``. It is the one place in production holding
    the published dict rather than a description of one.
    """
    from evidence_engine.adapters.outbound.persistence.postgres.evidence_repository import (
        PostgresEvidenceRepository,
    )

    def render_with_a_ranking(doc: EvidenceDocument) -> dict[str, Any]:
        published = render_document(doc, schema_version=SCHEMA_VERSION)
        published["speech_events"][0]["overall_score"] = 0.91
        return published

    repository = PostgresEvidenceRepository(cast(Any, None), render_document=render_with_a_ranking)

    with pytest.raises(ResultViolation, match="FR-029"):
        await repository.store_document(document, TenantId("tenant-oratoria"))
