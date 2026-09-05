"""Direct tests for the one adapter that hears real audio.

Until this file existed there were none. Eleven hundred tests ran green against
the deterministic runtime — which replays a script and is bit-exact by
construction — while the component doing the actual recognition, carrying three
`type: ignore`s and two decisions about how to represent absence, had no test
at all. Both of the semantic defects fixed alongside this file survived four
adversarial review rounds for exactly that reason.

**Why a fake pipeline rather than the real one.** `WhisperSpeechRuntime.load()`
downloads 3 GB and needs a GPU; `__init__` takes the built pipeline as a
parameter, so the adapter's own logic — how it reads `chunks`, where it places
words on the session clock, what it does with a word it cannot place — is
reachable with a callable that returns a dict. That is the part with the
decisions in it. Whether transformers works is transformers' problem.

**Why the file is split.** `_to_float32` needs numpy, which is in the `managed`
extra and not a dependency of the service, so a test that always went through
`transcribe()` would be skipped in CI — on exactly the defect surface that most
needs covering. The chunk reading is a static method over a plain dict, so it
runs everywhere; the end-to-end pass is marked and runs where the extra is
installed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evidence_engine.adapters.outbound.model_runtime.whisper import (
    PINNED_DECODING,
    PINNED_MODEL,
    PINNED_REVISION,
    WhisperSpeechRuntime,
)
from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    TimedWordHypothesis,
    UntimedWordHypothesis,
)
from evidence_engine.domain.shared.errors import FabricatedValue
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable

A_MODEL = ModelVersionId("whisper-large-v3@test")


class FakePipeline:
    """Stands in for the transformers ASR pipeline.

    Records the call so the pinned decoding configuration can be asserted:
    `condition_on_prev_tokens=False` is what stops the decoder biasing toward
    fluent continuations of what it already produced, which is the grammatical
    cleanup FR-011 exists to measure the absence of. Silently losing it would
    change every transcript and nothing would fail.
    """

    def __init__(self, output: dict[str, Any]) -> None:
        self._output = output
        self.calls: list[dict[str, Any]] = []

    def __call__(self, audio: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"audio": audio, **kwargs})
        return self._output


def chunk(text: str, start: float | None, end: float | None) -> dict[str, Any]:
    return {"text": text, "timestamp": (start, end)}


def a_window(position_ms: int = 0, duration_ms: int = 1_000) -> AudioWindow:
    frames = int(16_000 * duration_ms / 1000)
    return AudioWindow(
        session_position_ms=position_ms,
        duration_ms=duration_ms,
        sample_rate_hz=16_000,
        samples=b"\x00\x00" * frames,
    )


def transcribe(output: dict[str, Any], window: AudioWindow | None = None) -> Any:
    pytest.importorskip("numpy", reason="the managed extra is not installed")
    runtime = WhisperSpeechRuntime(FakePipeline(output), A_MODEL)
    return asyncio.run(runtime.transcribe(window or a_window()))


# ---------------------------------------------------------------------------
# The defect this file was written for: words were being deleted
# ---------------------------------------------------------------------------


def test_a_word_with_no_timestamps_is_kept_not_dropped() -> None:
    """Driver 1 is verbatim fidelity, and this used to violate it silently.

    The alignment heads can fail on a fragment and return `None` for a bound.
    The adapter dropped the whole chunk — text included — so a word the model
    genuinely recognised vanished from the transcript. Nothing downstream could
    notice: the transcript renders, it is one word shorter, and no count
    anywhere disagrees.

    Not guessing the boundary was right. Deleting the word was not: what is
    unknown is *where* it was, not *whether* it was said.
    """
    words = WhisperSpeechRuntime.words_from(
        {"chunks": [chunk("hola", 0.0, 0.4), chunk("mundo", None, None)]},
        session_position_ms=0,
    )

    assert [word.raw_text for word in words] == ["hola", "mundo"]
    assert isinstance(words[0], TimedWordHypothesis)
    assert isinstance(words[1], UntimedWordHypothesis)


def test_an_untimed_word_cannot_be_asked_where_it_was() -> None:
    """The union makes the illegal state unrepresentable, not merely undesirable.

    `start_ms: int | None` on each end would admit "start known, end unknown",
    which is not a state the aligner can produce and not one any consumer
    should have to handle. And a caller reaching for a boundary that does not
    exist gets a domain error naming the rule rather than a `None` it might
    coerce to zero.
    """
    (word,) = WhisperSpeechRuntime.words_from(
        {"chunks": [chunk("mundo", None, None)]}, session_position_ms=0
    )
    assert isinstance(word, UntimedWordHypothesis)
    assert word.reason is UnavailabilityReason.ALIGNMENT_UNAVAILABLE

    with pytest.raises(FabricatedValue):
        _ = word.start_ms  # type: ignore[attr-defined]
    with pytest.raises(FabricatedValue):
        _ = word.end_ms  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(None, 0.4, id="start-missing"),
        pytest.param(0.0, None, id="end-missing"),
    ],
)
def test_a_half_aligned_word_is_untimed_rather_than_half_placed(
    start: float | None, end: float | None
) -> None:
    """One bound is not an interval.

    Keeping the half that arrived would put a word on the clock with an
    invented other end, which is the same fabrication as guessing both.
    """
    (word,) = WhisperSpeechRuntime.words_from(
        {"chunks": [chunk("mundo", start, end)]}, session_position_ms=0
    )
    assert isinstance(word, UntimedWordHypothesis)
    assert word.raw_text == "mundo"


def test_an_empty_chunk_is_dropped_and_a_blank_one_is_not_a_word() -> None:
    """A chunk with no text is not a word missing its timing.

    `WordToken.__post_init__` already refuses empty text — "an unintelligible
    stretch is an UNINTELLIGIBLE speech event, not a blank word" — so admitting
    one here would only move the failure later.
    """
    words = WhisperSpeechRuntime.words_from(
        {"chunks": [chunk("   ", 0.0, 0.4), chunk("", None, None), chunk("hola", 0.1, 0.3)]},
        session_position_ms=0,
    )
    assert [word.raw_text for word in words] == ["hola"]


# ---------------------------------------------------------------------------
# The other defect: an invented posterior
# ---------------------------------------------------------------------------


def test_the_score_says_no_posterior_was_reported_rather_than_0_5() -> None:
    """Whisper exposes no per-word confidence, so the adapter must not invent one.

    It used to report `0.5`, defended in a comment as "the neutral value". A
    consumer reading a number cannot tell a neutral placeholder from a genuine
    50% posterior, and FR-025's whole point is that the two must not look
    alike.
    """
    (word,) = WhisperSpeechRuntime.words_from(
        {"chunks": [chunk("hola", 0.0, 0.4)]}, session_position_ms=0
    )

    assert isinstance(word.score, Unavailable)
    assert word.score.reason is UnavailabilityReason.POSTERIOR_NOT_REPORTED

    with pytest.raises(FabricatedValue):
        _ = word.score.value


# ---------------------------------------------------------------------------
# The behaviour that was already right, now covered
# ---------------------------------------------------------------------------


def test_word_offsets_are_placed_on_the_session_clock() -> None:
    """The pipeline reports seconds within the window; sessions run for minutes.

    A window at 60 s carrying a word at 0.4 s is a word at 60.4 s of the
    session. Getting this wrong shifts every interval and every co-occurrence
    by the window position, and the transcript still reads correctly.
    """
    words = WhisperSpeechRuntime.words_from(
        {"chunks": [chunk("hola", 0.4, 0.9)]}, session_position_ms=60_000
    )

    assert isinstance(words[0], TimedWordHypothesis)
    assert words[0].start_ms == 60_400
    assert words[0].end_ms == 60_900


def test_output_with_no_chunks_yields_no_words_rather_than_raising() -> None:
    """Silence is a legitimate window. It is not a failure and not a gap."""
    assert WhisperSpeechRuntime.words_from({"text": ""}, session_position_ms=0) == ()


def test_the_runtime_invents_no_events_and_no_prosody() -> None:
    """Whisper is a recogniser. It does not detect disfluencies or measure pitch.

    Empty tuples rather than fabricated entries is what keeps FR-025's path
    honest: a runtime that invented a filled pause would make the unavailable
    handling look exercised while it was being bypassed.
    """
    result = transcribe({"chunks": [chunk("hola", 0.0, 0.4)]})
    assert result.events == ()
    assert result.prosody == ()


def test_the_whole_window_is_reported_stable() -> None:
    """Whisper decodes a window in one pass and has no revisable tail.

    Unlike a streaming transducer, everything it returns is as stable as it
    will ever be, so the coordinator may finalize through the window end.
    """
    result = transcribe({"chunks": []}, a_window(position_ms=5_000, duration_ms=2_000))
    assert result.stable_through_ms == 7_000


def test_the_pinned_decoding_configuration_reaches_the_model() -> None:
    """The pins and the code must not drift apart silently.

    `condition_on_prev_tokens=False` is the load-bearing one: it stops the
    decoder biasing toward fluent continuations of its own output, which is the
    grammatical cleanup FR-011 measures the absence of. Losing it would change
    every transcript and break no test that did not check for it.
    """
    pytest.importorskip("numpy", reason="the managed extra is not installed")
    pipeline = FakePipeline({"chunks": []})
    runtime = WhisperSpeechRuntime(pipeline, A_MODEL)
    asyncio.run(runtime.transcribe(a_window()))

    (call,) = pipeline.calls
    assert call["return_timestamps"] == "word"
    assert call["generate_kwargs"] == PINNED_DECODING
    assert PINNED_DECODING["condition_on_prev_tokens"] is False
    assert PINNED_DECODING["language"] == "es"


def test_the_reported_model_version_names_the_revision_not_the_model() -> None:
    """Two people running "whisper-large-v3" a year apart are not running the same weights."""
    assert PINNED_MODEL == "openai/whisper-large-v3"
    assert len(PINNED_REVISION) == 40
    assert PINNED_REVISION[:12] in "whisper-large-v3@" + PINNED_REVISION[:12]


def test_the_environment_is_declared_unpinned() -> None:
    """Phase 3 freezes the backend, container digest and Torch/CUDA build.

    Until then any figure this runtime produces is reproducible only by whoever
    ran it, and the runtime says so rather than staying silent about it.
    """
    assert WhisperSpeechRuntime.environment_is_pinned is False
