"""Builders for annotation records.

Terse on purpose. These tests are about agreement mathematics, and a test whose
setup takes fifteen lines to place two intervals hides the arithmetic it exists
to check.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from corpus.schema.records import (
    AnnotatedRecording,
    AnnotationPass,
    DisfluencyAnnotation,
    Interval,
    Speaker,
    Word,
)
from evidence_engine.domain.shared.taxonomy import ContextualRole, SpeechEventType


def annotation(
    event_type: SpeechEventType,
    start_ms: int,
    end_ms: int,
    annotator: str = "ana",
    *,
    raw_text: str = "",
    role: ContextualRole | None = None,
    confidence: float = 1.0,
) -> DisfluencyAnnotation:
    """One annotation, with the lexical fields filled in when the class needs them."""
    from corpus.schema.records import LEXICAL_CLASSES

    if event_type in LEXICAL_CLASSES:
        raw_text = raw_text or "este"
        role = role or ContextualRole.FILLER
    return DisfluencyAnnotation(
        event_type=event_type,
        interval=Interval(start_ms, end_ms),
        annotator_id=annotator,
        raw_text=raw_text,
        context_role=role,
        annotator_confidence=confidence,
    )


def recording(
    annotator: str,
    annotations: Sequence[DisfluencyAnnotation],
    *,
    recording_id: str = "pilot-001",
    words: Sequence[Word] = (),
    duration_ms: int = 60_000,
) -> AnnotatedRecording:
    return AnnotatedRecording(
        recording_id=recording_id,
        speaker=Speaker(pseudonym="P-001"),
        annotator_id=annotator,
        annotation_pass=AnnotationPass.FIRST,
        duration_ms=duration_ms,
        words=tuple(words),
        disfluencies=tuple(annotations),
    )


def word(text: str, start_ms: int, end_ms: int) -> Word:
    return Word(text=text, interval=Interval(start_ms, end_ms))


@pytest.fixture
def filled_pause() -> SpeechEventType:
    return SpeechEventType.FILLED_PAUSE
