"""Driver 1: what the speaker said survives the pipeline.

"Losing a disfluency prevents subsequent measurement and feedback." Every rule
tested here follows from that one sentence, and each guards a place where a
well-meaning change would quietly delete evidence:

- FR-011: no grammatical cleanup, so a repeated word stays repeated.
- FR-017: the raw expression is kept even when its role turns out to be
  semantic, because precision is uncomputable without the true negatives.
- FR-013: an undecidable case is reported as `uncertain`, not guessed.
- §6.1 step 9: finalized history is never rewritten.
"""

from __future__ import annotations

import pytest

from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.identifiers import EventId, TokenId
from evidence_engine.domain.shared.provenance import Provenance
from evidence_engine.domain.shared.taxonomy import ContextualRole, SpeechEventType
from evidence_engine.domain.shared.timeline import Interval, MonotonicTime
from evidence_engine.domain.speech_events.events import SpeechEvent, SpeechEventViolation
from evidence_engine.domain.transcript import transcript as transcript_module
from evidence_engine.domain.transcript.tokens import TokenStatus, TranscriptViolation, WordToken

pytestmark = pytest.mark.invariant


def _token(text: str, start: int, end: int, index: int = 0) -> WordToken:
    return WordToken(
        id=TokenId(f"tok-{index}"),
        raw_text=text,
        interval=Interval.of(start, end),
        confidence=Confidence.calibrated(0.9),
    )


# ---------------------------------------------------------------------------
# FR-011 - the transcript is literal
# ---------------------------------------------------------------------------


def test_repeated_words_are_not_collapsed() -> None:
    """A collapsed repetition is a REPETITION event that can never be detected."""
    transcript = transcript_module.build(
        [
            _token("vamos", 0, 400, 0),
            _token("a", 400, 500, 1),
            _token("a", 500, 620, 2),
            _token("analizar", 620, 1_200, 3),
        ]
    )

    assert transcript.raw_text() == "vamos a a analizar"


def test_an_empty_token_is_refused() -> None:
    """An unintelligible stretch is an event, not a blank word."""
    with pytest.raises(TranscriptViolation, match="UNINTELLIGIBLE"):
        WordToken(
            id=TokenId("tok-0"),
            raw_text="",
            interval=Interval.of(0, 100),
            confidence=Confidence.calibrated(0.5),
        )


# ---------------------------------------------------------------------------
# FR-016 and section 6.1 step 9 - reconciliation never rewrites the past
# ---------------------------------------------------------------------------


def test_finalized_tokens_survive_a_new_provisional_hypothesis() -> None:
    transcript = transcript_module.build([_token("hola", 0, 400, 0)])
    transcript = transcript.finalize_through(MonotonicTime(400))

    transcript = transcript.with_provisional([_token("mundo", 400, 900, 1)])

    assert [t.raw_text for t in transcript.tokens] == ["hola", "mundo"]
    assert transcript.tokens[0].status is TokenStatus.FINAL
    assert transcript.tokens[1].status is TokenStatus.PROVISIONAL


def test_a_provisional_hypothesis_behind_the_frontier_is_refused() -> None:
    transcript = transcript_module.build([_token("hola", 0, 400, 0)])
    transcript = transcript.finalize_through(MonotonicTime(400))

    with pytest.raises(TranscriptViolation, match="finalized frontier"):
        transcript.with_provisional([_token("adios", 200, 600, 9)])


def test_the_frontier_never_moves_backwards() -> None:
    transcript = transcript_module.build([_token("hola", 0, 400, 0)])
    transcript = transcript.finalize_through(MonotonicTime(400))

    with pytest.raises(TranscriptViolation, match="backwards"):
        transcript.finalize_through(MonotonicTime(200))


def test_a_finalized_token_cannot_be_revised() -> None:
    token = _token("treinta", 0, 500, 0).finalize()

    with pytest.raises(TranscriptViolation, match="never rewritten"):
        token.revise("trescientos", Interval.of(0, 700), Confidence.calibrated(0.95))


def test_a_provisional_token_keeps_its_identity_when_revised() -> None:
    """Section 7.4: ids are stable across provisional-to-final reconciliation."""
    token = _token("treinta", 0, 500, 0)

    revised = token.revise("trescientos", Interval.of(0, 700), Confidence.calibrated(0.95))

    assert revised.id == token.id
    assert revised.raw_text == "trescientos"


def test_the_transcript_orders_deterministically_regardless_of_input_order() -> None:
    forward = transcript_module.build([_token("a", 0, 100, 0), _token("b", 100, 200, 1)])
    backward = transcript_module.build([_token("b", 100, 200, 1), _token("a", 0, 100, 0)])

    assert forward == backward


# ---------------------------------------------------------------------------
# FR-013 and FR-017 - contextual role, and keeping the raw expression
# ---------------------------------------------------------------------------


def _lexical_event(role: ContextualRole, provenance: Provenance, text: str = "este") -> SpeechEvent:
    return SpeechEvent(
        id=EventId("ev-1"),
        type=SpeechEventType.LEXICAL_FILLER,
        interval=Interval.of(1_000, 1_400),
        confidence=Confidence.calibrated(0.88),
        provenance=provenance,
        raw_text=text,
        context_role=role,
    )


def test_raw_expression_is_kept_when_the_role_is_semantic(
    audio_provenance: Provenance,
) -> None:
    """FR-017. Dropping it would destroy the precision denominator (NFR-003)."""
    event = _lexical_event(ContextualRole.SEMANTIC, audio_provenance)

    assert event.raw_text == "este"
    assert event.counts_as_disfluency is False


def test_only_the_filler_role_counts_as_a_disfluency(
    audio_provenance: Provenance,
) -> None:
    """Section 17's mitigation for ambiguous lexical fillers, in one assertion."""
    counted = {
        role: _lexical_event(role, audio_provenance).counts_as_disfluency for role in ContextualRole
    }

    assert counted == {
        ContextualRole.FILLER: True,
        ContextualRole.SEMANTIC: False,
        ContextualRole.DISCOURSE_MARKER: False,
        ContextualRole.UNCERTAIN: False,
    }


def test_a_lexical_class_without_a_role_is_refused(audio_provenance: Provenance) -> None:
    with pytest.raises(SpeechEventViolation, match="uncertain"):
        SpeechEvent(
            id=EventId("ev-1"),
            type=SpeechEventType.LEXICAL_FILLER,
            interval=Interval.of(1_000, 1_400),
            confidence=Confidence.calibrated(0.88),
            provenance=audio_provenance,
            raw_text="o sea",
            context_role=None,
        )


def test_a_lexical_class_without_raw_text_is_refused(audio_provenance: Provenance) -> None:
    with pytest.raises(SpeechEventViolation, match="FR-017"):
        SpeechEvent(
            id=EventId("ev-1"),
            type=SpeechEventType.REPETITION,
            interval=Interval.of(1_000, 1_400),
            confidence=Confidence.calibrated(0.88),
            provenance=audio_provenance,
            raw_text="   ",
            context_role=ContextualRole.FILLER,
        )


def test_an_acoustic_class_cannot_carry_a_lexical_role(
    audio_provenance: Provenance,
) -> None:
    """A filled pause has no word to assign a role to; inventing one is a claim."""
    with pytest.raises(SpeechEventViolation, match="acoustic"):
        SpeechEvent(
            id=EventId("ev-1"),
            type=SpeechEventType.FILLED_PAUSE,
            interval=Interval.of(1_000, 1_780),
            confidence=Confidence.calibrated(0.91),
            provenance=audio_provenance,
            context_role=ContextualRole.FILLER,
        )


def test_a_silent_pause_is_evidence_but_not_a_disfluency(
    audio_provenance: Provenance,
) -> None:
    event = SpeechEvent(
        id=EventId("ev-1"),
        type=SpeechEventType.SILENT_PAUSE,
        interval=Interval.of(2_000, 3_800),
        confidence=Confidence.calibrated(0.99),
        provenance=audio_provenance,
    )

    assert event.counts_as_disfluency is False


def test_an_unintelligible_stretch_is_not_charged_to_the_speaker(
    audio_provenance: Provenance,
) -> None:
    """It is a failure of the recording, not of the presentation."""
    event = SpeechEvent(
        id=EventId("ev-1"),
        type=SpeechEventType.UNINTELLIGIBLE,
        interval=Interval.of(4_000, 4_900),
        confidence=Confidence.calibrated(0.4),
        provenance=audio_provenance,
    )

    assert event.counts_as_disfluency is False
