"""FR-028 and FR-029: the engine correlates and reports. It never explains or ranks.

Two separations of authority are guarded here, and both are named in §3 driver
8: "detectors observe; the RPP ranks; the language model explains."

FR-028 keeps the fusion engine from asserting causality. FR-029 keeps the
result document from carrying a rank, a score or a top-k. The risk table calls
the second one out by name - "RPP logic leaks into the evidence service" - and
prescribes exactly this test: a contract invariant of ``ranking_authority=none``
with negative tests for rank and top-k fields.
"""

from __future__ import annotations

import dataclasses

import pytest

from evidence_engine.domain.evidence.cooccurrence import (
    FusionViolation,
    FusionWindow,
    MultimodalCooccurrence,
    TimedEvent,
    correlate,
)
from evidence_engine.domain.evidence.document import (
    EvidenceDocument,
    ResultViolation,
    assert_carries_no_ranking,
)
from evidence_engine.domain.shared.identifiers import ConfigurationSnapshotId, EventId
from evidence_engine.domain.shared.timeline import Interval

pytestmark = pytest.mark.invariant


@pytest.fixture
def window(configuration: ConfigurationSnapshotId) -> FusionWindow:
    return FusionWindow(width_ms=500, configuration=configuration)


# ---------------------------------------------------------------------------
# FR-028 - no causal inference
# ---------------------------------------------------------------------------


def test_cooccurrence_always_declares_no_causal_inference(window: FusionWindow) -> None:
    pair = MultimodalCooccurrence(
        speech_event_id=EventId("speech-1"),
        visual_event_id=EventId("visual-1"),
        temporal_distance_ms=120,
        window=window,
    )

    assert pair.causal_inference is False


def test_causal_inference_is_not_a_constructor_parameter(window: FusionWindow) -> None:
    """The value cannot be supplied, so it cannot be supplied wrongly.

    A boolean field defaulting to False would satisfy the letter of FR-028 and
    fail its purpose: any call site could pass True. Making it a ClassVar means
    asserting causality requires editing the domain, which is reviewable.
    """
    field_names = {f.name for f in dataclasses.fields(MultimodalCooccurrence)}

    assert "causal_inference" not in field_names

    with pytest.raises(TypeError):
        MultimodalCooccurrence(  # type: ignore[call-arg]
            speech_event_id=EventId("speech-1"),
            visual_event_id=EventId("visual-1"),
            temporal_distance_ms=120,
            window=window,
            causal_inference=True,
        )


def test_cooccurrence_outside_the_window_is_refused(window: FusionWindow) -> None:
    with pytest.raises(FusionViolation, match="fusion window"):
        MultimodalCooccurrence(
            speech_event_id=EventId("speech-1"),
            visual_event_id=EventId("visual-1"),
            temporal_distance_ms=900,
            window=window,
        )


def test_correlation_is_deterministic(window: FusionWindow) -> None:
    """NFR-015: same input, same window, same pairs in the same order."""
    speech = [
        TimedEvent(EventId("s-2"), Interval.of(3_000, 3_400)),
        TimedEvent(EventId("s-1"), Interval.of(1_000, 1_200)),
    ]
    visual = [
        TimedEvent(EventId("v-2"), Interval.of(3_200, 3_600)),
        TimedEvent(EventId("v-1"), Interval.of(1_100, 1_500)),
    ]

    first = correlate(speech, visual, window)
    second = correlate(list(reversed(speech)), list(reversed(visual)), window)

    assert first == second
    assert [p.speech_event_id.value for p in first] == ["s-1", "s-2"]


def test_correlation_emits_every_admissible_pair_and_selects_none(
    window: FusionWindow,
) -> None:
    """Picking a 'best' match would be a ranking decision (§5)."""
    speech = [TimedEvent(EventId("s-1"), Interval.of(1_000, 1_200))]
    visual = [
        TimedEvent(EventId("v-1"), Interval.of(1_150, 1_300)),
        TimedEvent(EventId("v-2"), Interval.of(1_400, 1_600)),
    ]

    pairs = correlate(speech, visual, window)

    assert len(pairs) == 2


def test_correlation_drops_pairs_beyond_the_window(window: FusionWindow) -> None:
    speech = [TimedEvent(EventId("s-1"), Interval.of(1_000, 1_200))]
    visual = [TimedEvent(EventId("v-far"), Interval.of(5_000, 5_200))]

    assert correlate(speech, visual, window) == ()


# ---------------------------------------------------------------------------
# FR-029 - no ranking authority
# ---------------------------------------------------------------------------


def test_document_declares_ranking_authority_none() -> None:
    assert EvidenceDocument.ranking_authority == "none"


def test_ranking_authority_is_not_a_constructor_parameter() -> None:
    field_names = {f.name for f in dataclasses.fields(EvidenceDocument)}

    assert "ranking_authority" not in field_names


@pytest.mark.parametrize(
    "forbidden",
    ["rank", "score", "priority", "severity", "top_k", "recommendations", "interventions"],
)
def test_top_level_ranking_field_is_refused(forbidden: str) -> None:
    payload = {"session_id": "s-1", forbidden: 3}

    with pytest.raises(ResultViolation, match="FR-029"):
        assert_carries_no_ranking(payload)


def test_ranking_field_nested_inside_an_event_is_refused() -> None:
    """The realistic failure: a runtime forwards its own ordering."""
    payload = {
        "speech_events": [
            {"id": "e-1", "type": "filled_pause"},
            {"id": "e-2", "type": "lexical_filler", "priority": 0.8},
        ]
    }

    with pytest.raises(ResultViolation) as caught:
        assert_carries_no_ranking(payload)

    assert "speech_events[1].priority" in str(caught.value)


def test_a_clean_payload_passes() -> None:
    payload = {
        "session_id": "s-1",
        "ranking_authority": "none",
        "speech_events": [
            {"id": "e-1", "type": "filled_pause", "confidence": 0.9, "start_ms": 100}
        ],
        "quality": {"audio": {"signal_to_noise_db": 24.0}},
    }

    assert_carries_no_ranking(payload)
