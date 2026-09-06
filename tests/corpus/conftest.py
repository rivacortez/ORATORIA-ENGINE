"""Builders for annotation records.

Terse on purpose. These tests are about agreement mathematics, and a test whose
setup takes fifteen lines to place two intervals hides the arithmetic it exists
to check.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from corpus.schema.records import (
    SCHEMA_VERSION,
    AnnotatedRecording,
    AnnotationPass,
    ConsentBasis,
    ConsentRecord,
    DisfluencyAnnotation,
    Interval,
    RecordingConditions,
    Speaker,
    Word,
)
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import (
    TAXONOMY_VERSION,
    ContextualRole,
    SpeechEventType,
)

#: A consent record and a capture chain that a freeze would accept.
#:
#: Not defaults on the record itself. `AnnotatedRecording` leaves both `None`,
#: because a fixture placing two intervals 80 ms apart has no participant behind
#: it and a field that is always filled because the constructor demanded it
#: proves nothing about whether anybody was asked. These exist so that the tests
#: which *are* about the corpus - inventory, split, freeze - can say so in one
#: word instead of eight lines.
CONSENT = ConsentRecord(
    basis=ConsentBasis.WRITTEN_INFORMED,
    policy_version=SemanticVersion(1, 0, 0),
    granted_on=date(2026, 9, 1),
)

CONDITIONS = RecordingConditions(
    microphone="Realtek(R) Audio - onboard array",
    sample_rate_hz=16_000,
    bit_depth=16,
    channels=1,
    virtual_audio_bypassed=True,
    room_notes="",
)


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
    taxonomy_version: SemanticVersion | None = TAXONOMY_VERSION,
    consent: ConsentRecord | None = None,
    conditions: RecordingConditions | None = None,
) -> AnnotatedRecording:
    """A record the validator accepts, unless a test deliberately breaks it.

    Two defaults exist so that a test about agreement mathematics is about
    agreement mathematics.

    `consent` and `conditions` are *not* among them: they default to `None`,
    which is what an in-code record honestly has. A helper that supplied them
    would put a consent basis on every record in the suite and thereby make the
    field's presence prove nothing - the tests that care pass `CONSENT` and
    `CONDITIONS` explicitly, and the ones that do not are about arithmetic.

    The words tier is synthesized rather than left empty. An empty transcript
    is a validation *error* - lexical classes cannot be checked against
    anything - so a fixture without one is a file `compare` refuses.

    `taxonomy_version` defaults to the published one rather than to `None`,
    for the same reason: `compare` now requires both files to record a manual
    and to record the same one. `None` is still reachable, and a test passes it
    deliberately to check that the refusal fires.
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
        consent=consent,
        conditions=conditions,
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


def template(path, **overrides):  # type: ignore[no-untyped-def]
    """Write a template with the consent and capture chain a real one needs.

    `write_template` requires the speaker's variety, a consent record and the
    capture conditions, and requires them for a reason worth restating here:
    all three are captured at recruitment or they are unreconstructable, so a
    default would quietly produce a corpus that cannot be sliced by dialect and
    cannot support a consent audit.

    That reasoning is right and it makes every call eight arguments long. This
    helper carries the ones a test is not about, so a test about overwrite
    protection reads as a test about overwrite protection. Any of them can be
    overridden by keyword.
    """
    from corpus.io.elan import write_template

    arguments = {
        "recording_id": "pilot-001",
        "speaker_pseudonym": "P-001",
        "speaker_variety": "es-PE",
        "annotator_id": "ana",
        "media_url": "pilot-001.wav",
        "consent": CONSENT,
        "conditions": CONDITIONS,
    }
    arguments.update(overrides)
    write_template(path, **arguments)  # type: ignore[arg-type]


#: The CLI flags that match `CONSENT` and `CONDITIONS` above, for tests that
#: drive `corpus template` through `main()` rather than calling the writer.
TEMPLATE_FLAGS = [
    "--variety",
    "es-PE",
    "--consent-basis",
    "written_informed",
    "--consent-policy",
    "1.0.0",
    "--consent-granted",
    "2026-09-01",
    "--microphone",
    "Realtek(R) Audio - onboard array",
    "--sample-rate-hz",
    "16000",
    "--bit-depth",
    "16",
    "--channels",
    "1",
    "--virtual-audio-bypassed",
    "yes",
]
