"""Temporal fusion: pairing speech and visual events that happened together.

The algorithm lives in the domain (``domain.evidence.cooccurrence``); this
module is the adapter between it and the two modality packages, which contract
C5 keeps from knowing about each other. Projecting both sides down to
``TimedEvent`` is what lets the correlation run without either package
importing the other - and it also keeps the algorithm honest, because a
function that only sees identities and intervals cannot be tempted to weight
pairs by what the events mean.

FR-026 is a precondition, not a step here: both sides must already be on the
shared monotonic timeline. They are, because every interval in this system is
built from the session clock and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence

from evidence_engine.domain.evidence.cooccurrence import (
    FusionWindow,
    MultimodalCooccurrence,
    TimedEvent,
    correlate,
)
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.visual_events.events import VisualEvent


def fuse(
    speech_events: Sequence[SpeechEvent],
    visual_events: Sequence[VisualEvent],
    window: FusionWindow,
    *,
    finalized_only: bool = True,
) -> tuple[MultimodalCooccurrence, ...]:
    """Correlate the two modalities over the configured window.

    ``finalized_only`` defaults to true because §7.3 publishes co-occurrences
    only as ``cooccurrence.final``. Pairing provisional events would produce
    correlations that dissolve when one side is revised, and a consumer that
    already displayed one has no way to learn it never happened - §6.1 step 9
    reserves that guarantee for finalized history.

    A modality with nothing to contribute yields no pairs and no error. QA-02
    requires a dead camera to leave speech evidence intact, and fusion is the
    one place where "intact" could quietly mean "empty result".
    """
    speech = _project_speech(speech_events, finalized_only)
    visual = _project_visual(visual_events, finalized_only)
    if not speech or not visual:
        return ()
    return correlate(speech, visual, window)


def _project_speech(events: Sequence[SpeechEvent], finalized_only: bool) -> list[TimedEvent]:
    return [
        TimedEvent(id=event.id, interval=event.interval)
        for event in events
        if event.is_final or not finalized_only
    ]


def _project_visual(events: Sequence[VisualEvent], finalized_only: bool) -> list[TimedEvent]:
    return [
        TimedEvent(id=event.id, interval=event.interval)
        for event in events
        if event.is_final or not finalized_only
    ]
