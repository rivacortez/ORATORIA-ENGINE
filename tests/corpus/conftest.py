"""Builders for annotation records.

Terse on purpose. These tests are about agreement mathematics, and a test whose
setup takes fifteen lines to place two intervals hides the arithmetic it exists
to check.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from corpus.schema.records import (
    SCHEMA_VERSION,
    AnnotatedRecording,
    AnnotationPass,
    DisfluencyAnnotation,
    Interval,
    Speaker,
    Word,
)
from evidence_engine.domain.shared.provenance import SemanticVersion
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
    words: Sequence[Word] | None = None,
    duration_ms: int | None = None,
    annotation_pass: AnnotationPass = AnnotationPass.FIRST,
    schema_version: SemanticVersion = SCHEMA_VERSION,
    taxonomy_version: SemanticVersion | None = None,
) -> AnnotatedRecording:
    """A record the validator accepts, unless a test deliberately breaks it.

    The words tier is synthesized rather than left empty. An empty transcript
    is a validation *error* - lexical classes cannot be checked against
    anything - so a fixture without one is a file `compare` now refuses, and
    every agreement test would be exercising the refusal instead of the
    mathematics.
    """
    resolved_words = tuple(words) if words is not None else _implied_words(annotations)
    ends = [w.interval.end_ms for w in resolved_words]
    ends.extend(a.interval.end_ms for a in annotations)
    return AnnotatedRecording(
        recording_id=recording_id,
        speaker=Speaker(pseudonym="P-001"),
        annotator_id=annotator,
        annotation_pass=annotation_pass,
        duration_ms=duration_ms if duration_ms is not None else max(ends, default=0) + 1_000,
        words=resolved_words,
        disfluencies=tuple(annotations),
        schema_version=schema_version,
        taxonomy_version=taxonomy_version,
    )


def _implied_words(annotations: Sequence[DisfluencyAnnotation]) -> tuple[Word, ...]:
    """A transcript that attests whatever the annotations quote.

    One word per annotation, spanning it and carrying its raw expression, so a
    lexical annotation of "este" has an "este" beneath it rather than a
    `unattested_expression` warning. Plus one word at the start, because a
    recording with no annotations still needs a transcript to be a valid file.
    """
    words = [Word(text="buenos", interval=Interval(0, 500))]
    words.extend(
        Word(
            text=annotation.raw_text.split()[0] if annotation.raw_text.strip() else "palabra",
            interval=annotation.interval,
        )
        for annotation in annotations
        if annotation.interval.start_ms >= 500
    )
    return tuple(words)


def word(text: str, start_ms: int, end_ms: int) -> Word:
    return Word(text=text, interval=Interval(start_ms, end_ms))


@pytest.fixture
def filled_pause() -> SpeechEventType:
    return SpeechEventType.FILLED_PAUSE
