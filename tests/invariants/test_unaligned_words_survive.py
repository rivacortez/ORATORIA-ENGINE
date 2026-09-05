"""A word the aligner could not place is still a word (driver 1, FR-011).

The transcript is the authoritative verbatim record. Before this, a word whose
alignment failed was deleted from it - not because of anything the speaker did,
but because the transcript could only hold what it could sort by time. Nothing
downstream could detect the loss: the text rendered, it was one word shorter,
and no count anywhere disagreed.

The fix separates two things that had been one. Lexical order is always known;
temporal placement sometimes is not. So the tests below come in two halves:
what must include unplaced words (the verbatim text, the classifier's context,
finalization) and what must exclude them and say so (windows, the time
frontier, anything that reads the clock).

The rule that took the most argument is the last group's. An unplaced word is
finalized by an explicit sequence frontier, never by its neighbour. Finalizing
it because the word next to it is settled would assert that this word will not
be revised on the strength of evidence about a different word - which is a
guess wearing the costume of a decision.
"""

from __future__ import annotations

import pytest

from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import FabricatedValue
from evidence_engine.domain.shared.identifiers import TokenId
from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable
from evidence_engine.domain.shared.timeline import Interval, MonotonicTime
from evidence_engine.domain.transcript import transcript as transcript_module
from evidence_engine.domain.transcript.tokens import (
    AlignmentUnavailable,
    Timed,
    TokenSequence,
    TokenStatus,
    TranscriptViolation,
    WordToken,
)

pytestmark = pytest.mark.invariant


def placed(text: str, start: int, end: int, index: int) -> WordToken:
    return WordToken(
        id=TokenId(f"tok-{index}"),
        sequence=TokenSequence(window_position_ms=0, index=index),
        raw_text=text,
        placement=Timed(Interval.of(start, end)),
        confidence=Confidence.calibrated(0.9),
    )


def unplaced(text: str, index: int) -> WordToken:
    return WordToken(
        id=TokenId(f"tok-{index}"),
        sequence=TokenSequence(window_position_ms=0, index=index),
        raw_text=text,
        placement=AlignmentUnavailable(detail="the alignment heads returned no interval"),
        confidence=Unavailable(reason=UnavailabilityReason.POSTERIOR_NOT_REPORTED),
    )


#: "vamos a analizar esto" with the third word unplaceable.
def a_mixed_transcript() -> transcript_module.Transcript:
    return transcript_module.build(
        [
            placed("vamos", 0, 300, 0),
            placed("a", 300, 400, 1),
            unplaced("analizar", 2),
            placed("esto", 900, 1_200, 3),
        ]
    )


# ---------------------------------------------------------------------------
# Lexical: everything that reads what was said
# ---------------------------------------------------------------------------


def test_the_verbatim_text_contains_the_unplaced_word() -> None:
    """The defect, stated as an assertion.

    If this fails, a word the model genuinely recognised has been deleted from
    the authoritative record, and the transcript reads as a fluent sentence
    that the speaker did not say.
    """
    assert a_mixed_transcript().raw_text() == "vamos a analizar esto"


def test_the_unplaced_word_keeps_its_lexical_position() -> None:
    """Third word in, not appended at the end.

    Sorting unplaced tokens after the placed ones would be the cheap fix and
    would produce "vamos a esto analizar" - a different sentence, and a
    REPETITION or FALSE_START detector reading it would be reading fiction.
    """
    assert [t.raw_text for t in a_mixed_transcript().tokens] == [
        "vamos",
        "a",
        "analizar",
        "esto",
    ]


def test_the_classifier_context_includes_unplaced_neighbours() -> None:
    """FR-013 decides a filler's role from what surrounds it.

    What surrounds a word is a matter of what was said, not of what could be
    timestamped. A context window with silent holes in it, and no indication
    the holes are there, is worse than a short one.
    """
    transcript = a_mixed_transcript()
    context = transcript.context_around_sequence(
        TokenSequence(window_position_ms=0, index=1), before=1, after=1
    )
    assert [t.raw_text for t in context] == ["vamos", "a", "analizar"]


def test_the_transcript_reports_how_many_words_it_could_not_place() -> None:
    """The number that tells a reader how much of the temporal analysis is partial."""
    assert a_mixed_transcript().unaligned_count == 1
    assert transcript_module.build([placed("hola", 0, 100, 0)]).unaligned_count == 0


# ---------------------------------------------------------------------------
# Temporal: everything that reads the clock excludes them, deliberately
# ---------------------------------------------------------------------------


def test_a_time_window_returns_only_placed_words() -> None:
    """Silent pauses and fusion read this, and must not see a word with no position.

    Including it would mean choosing a window for it. Excluding it silently
    would be the old bug; excluding it while publishing `unaligned_count` is
    the honest version.
    """
    transcript = a_mixed_transcript()
    within = transcript.within_time(MonotonicTime(0), MonotonicTime(2_000))

    assert [t.raw_text for t in within] == ["vamos", "a", "esto"]
    assert all(t.is_timed for t in within)


def test_an_unplaced_word_cannot_be_asked_where_it_sat() -> None:
    """Reaching for a boundary that does not exist raises, rather than returning zero.

    A zero start would put the word at the beginning of the session: wrong,
    and entirely plausible-looking in a chart.
    """
    token = unplaced("analizar", 0)
    assert token.is_timed is False

    with pytest.raises(FabricatedValue, match="no interval"):
        _ = token.interval
    with pytest.raises(FabricatedValue, match="no alignment"):
        _ = token.placement.interval  # type: ignore[union-attr]


def test_an_unplaced_word_does_not_move_the_time_frontier() -> None:
    """It has no end to contribute.

    Letting one move the frontier would freeze a stretch of audio on the
    strength of a word that was never located inside it.
    """
    transcript = transcript_module.build([placed("hola", 0, 400, 0), unplaced("mundo", 1)])
    transcript = transcript.finalize_through_time(MonotonicTime(400))

    assert transcript.finalized_time_frontier.ms == 400
    transcript = transcript.finalize_through_sequence(TokenSequence(0, 1))
    assert transcript.finalized_time_frontier.ms == 400


# ---------------------------------------------------------------------------
# Finalization: by an explicit sequence frontier, never by a neighbour
# ---------------------------------------------------------------------------


def test_a_time_boundary_alone_leaves_the_unplaced_word_provisional() -> None:
    """This is the point of having two frontiers.

    The time boundary cannot reach a token with no interval. If finalization
    ran on time alone, an unplaced word would stay provisional for the whole
    session and never reach a consumer as settled text.
    """
    transcript = transcript_module.build([placed("hola", 0, 400, 0), unplaced("mundo", 1)])
    transcript = transcript.finalize_through_time(MonotonicTime(400))

    assert transcript.tokens[0].status is TokenStatus.FINAL
    assert transcript.tokens[1].status is TokenStatus.PROVISIONAL


def test_the_sequence_frontier_is_what_finalizes_it() -> None:
    """Stated by the coordinator, which knows how much audio it committed."""
    transcript = transcript_module.build([placed("hola", 0, 400, 0), unplaced("mundo", 1)])
    transcript = transcript.finalize_through_sequence(TokenSequence(0, 1))

    assert all(t.is_final for t in transcript.tokens)


def test_the_sequence_frontier_does_not_reach_past_itself() -> None:
    """A frontier at index 1 settles 0 and 1, and leaves 2 alone.

    The guard against the rejected design: finalizing by proximity would have
    swept the unplaced token in because its neighbour was settled, which
    asserts stability from evidence about a different word.
    """
    transcript = transcript_module.build(
        [placed("hola", 0, 400, 0), unplaced("mundo", 1), unplaced("otra", 2)]
    )
    transcript = transcript.finalize_through_sequence(TokenSequence(0, 1))

    assert [t.is_final for t in transcript.tokens] == [True, True, False]


def test_session_close_finalizes_whatever_is_left() -> None:
    """No more audio, no more passes: a provisional tail is settled text.

    Without this a session that ended on an unplaced word would publish it as
    provisional forever, which a consumer is entitled to keep re-rendering.
    """
    transcript = a_mixed_transcript().finalize_remaining()

    assert all(t.is_final for t in transcript.tokens)
    assert transcript.raw_text() == "vamos a analizar esto"


# ---------------------------------------------------------------------------
# The past stays frozen, in both orders
# ---------------------------------------------------------------------------


def test_a_word_behind_the_lexical_frontier_is_refused() -> None:
    """An unplaced token has no start, so only the lexical check can protect it.

    Without this rule the words the aligner already struggled with would be the
    only ones permanently rewritable - a hole in §6.1 step 9 that opens exactly
    where the evidence is weakest.
    """
    transcript = transcript_module.build([unplaced("hola", 0)])
    transcript = transcript.finalize_through_sequence(TokenSequence(0, 0))

    with pytest.raises(TranscriptViolation, match="lexical frontier"):
        transcript.with_provisional([unplaced("adios", 0)])


def test_a_placed_word_behind_the_time_frontier_is_still_refused() -> None:
    """The lexical check does not subsume the temporal one.

    A token with a fresh sequence can still assert a word inside audio that is
    already frozen, and that contradicts the frozen stretch regardless of where
    it sits in the word order.
    """
    transcript = transcript_module.build([placed("hola", 0, 400, 0)])
    transcript = transcript.finalize_through_time(MonotonicTime(400))

    with pytest.raises(TranscriptViolation, match="finalized frontier"):
        transcript.with_provisional([placed("adios", 200, 600, 9)])
