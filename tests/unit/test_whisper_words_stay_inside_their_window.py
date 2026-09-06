"""A whisper word never reaches past the audio it was decoded from.

The pipeline pads every window to 30 s of silence, and the alignment heads can
close the last word - or open a hallucinated one - inside that padding. The
runtime declares the whole window stable through its end, so a word ending
past it was a word the coordinator could neither settle nor keep: the time
rule left it provisional, the next window's hypothesis replaced the tail, and
before that a frontier that swept it in refused the next window's first word
as rewriting frozen audio. A review found the second; this is the fix at the
source, with the coordinator's own guard beside it.
"""

from __future__ import annotations

from evidence_engine.adapters.outbound.model_runtime.whisper import WhisperSpeechRuntime
from evidence_engine.application.ports.runtimes import TimedWordHypothesis, UntimedWordHypothesis


def _output(*chunks: tuple[str, tuple[float | None, float | None]]) -> dict[str, object]:
    return {"chunks": [{"text": text, "timestamp": stamp} for text, stamp in chunks]}


def test_a_word_closing_past_the_window_is_clamped_to_its_end() -> None:
    (hola, bien) = WhisperSpeechRuntime.words_from(
        _output((" hola", (0.0, 0.3)), (" bien", (0.8, 1.05))),
        session_position_ms=6_000,
        window_duration_ms=1_000,
    )

    assert isinstance(bien, TimedWordHypothesis)
    assert (bien.start_ms, bien.end_ms) == (6_800, 7_000)
    assert isinstance(hola, TimedWordHypothesis)
    assert (hola.start_ms, hola.end_ms) == (6_000, 6_300)


def test_a_word_opening_past_the_window_is_kept_without_a_placement() -> None:
    """Placed in silence that was never audio: not deleted, not left there."""
    (hola, fantasma) = WhisperSpeechRuntime.words_from(
        _output((" hola", (0.0, 0.3)), (" fantasma", (1.2, 1.6))),
        session_position_ms=0,
        window_duration_ms=1_000,
    )

    assert isinstance(hola, TimedWordHypothesis)
    assert isinstance(fantasma, UntimedWordHypothesis)
    assert fantasma.raw_text == "fantasma"
    assert "past the end of the window" in fantasma.detail


def test_a_word_inside_the_window_is_untouched() -> None:
    (word,) = WhisperSpeechRuntime.words_from(
        _output((" dentro", (0.4, 0.9))), session_position_ms=2_000, window_duration_ms=1_000
    )

    assert isinstance(word, TimedWordHypothesis)
    assert (word.start_ms, word.end_ms) == (2_400, 2_900)


def test_without_a_duration_nothing_is_clamped() -> None:
    """The other judgement calls are tested without a window; they keep working."""
    (word,) = WhisperSpeechRuntime.words_from(
        _output((" largo", (0.8, 1.05))), session_position_ms=0
    )

    assert isinstance(word, TimedWordHypothesis)
    assert (word.start_ms, word.end_ms) == (800, 1_050)
