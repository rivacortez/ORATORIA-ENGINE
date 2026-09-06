"""The frontier never reports audio as frozen that the runtime did not certify.

The first fix for the erased unplaced word took the sequence frontier as a
maximum over committed windows. A review found what that swept in: a *placed*
word whose interval reaches past ``stable_through_ms`` - whisper closing the
last word of a window a few tens of milliseconds after the window's end. The
time rule had left it provisional, correctly; the maximum finalized it anyway;
the time frontier then stood at the word's end, past the stable point; and the
next window's first word, starting exactly at the window boundary, was refused
as rewriting frozen audio. Ordinary audio ended the session.

The frontier now walks lexical order and stops at the first token that is not
settled. Two smaller findings from the same review are guarded beside it: the
event clock is the latest end among the placed words rather than the lexically
last one's, and a result filed under a window position that was never handed
out is refused rather than left to lose its unplaced words.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from evidence_engine import EngineConfiguration, OratoriaEngine
from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    MisdeclaredAudioWindow,
    SpeechResult,
    TimedWordHypothesis,
)
from evidence_engine.application.ports.streaming import OutboundEvent, ServerMessageType
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable
from evidence_engine.domain.shared.provenance import ModelRole
from evidence_engine.domain.shared.taxonomy import ProsodicIndicator, SpeechEventType
from evidence_engine.domain.transcript.tokens import TokenStatus
from evidence_engine.sdk.stream import StreamSession

RECOGNISER = ModelVersionId("streaming-shaped-v1")
ONE_SECOND = b"\x00\x00" * 16_000
ABSENT = Unavailable(reason=UnavailabilityReason.POSTERIOR_NOT_REPORTED)


def _word(text: str, start_ms: int, end_ms: int, index: int) -> TimedWordHypothesis:
    return TimedWordHypothesis(
        raw_text=text, start_ms=start_ms, end_ms=end_ms, score=ABSENT, index=index
    )


class _Scripted:
    """A runtime replaying one result per window position."""

    emitted_speech_events: frozenset[SpeechEventType] = frozenset()
    emitted_prosody: frozenset[ProsodicIndicator] = frozenset()
    capability_detail = "scripted"

    def __init__(self, script: Mapping[int, SpeechResult]) -> None:
        self._script = script

    @property
    def contributions(self) -> Mapping[ModelRole, ModelVersionId]:
        return {ModelRole.RECOGNISER: RECOGNISER}

    async def transcribe(self, window: AudioWindow) -> SpeechResult:
        position = window.session_position_ms
        return self._script.get(
            position, SpeechResult(contributions=self.contributions, window_position_ms=position)
        )


def _result(position: int, words: tuple[TimedWordHypothesis, ...], stable: int) -> SpeechResult:
    return SpeechResult(
        contributions={ModelRole.RECOGNISER: RECOGNISER},
        window_position_ms=position,
        words=words,
        stable_through_ms=stable,
    )


async def _stream(script: Mapping[int, SpeechResult]) -> StreamSession:
    engine = OratoriaEngine.local(
        EngineConfiguration(speech_runtime=_Scripted(script), window_seconds=1)
    )
    await engine.warmup()
    return engine.create_stream()


async def _drain(stream: StreamSession, type_: ServerMessageType) -> list[OutboundEvent]:
    events: list[OutboundEvent] = []
    while stream.pending():
        event = await asyncio.wait_for(stream.receive(), timeout=5)
        if event.type is type_:
            events.append(event)
    return events


async def test_a_word_reaching_past_the_stable_point_is_not_swept_final() -> None:
    """Window 0 ends at 1 000 ms; `bien` ends at 1 050 ms; the runtime re-emits it.

    A streaming runtime that certifies a window through its end and closes a
    word past it has left that word revisable, and re-emits it with the next
    window - the contract `with_provisional` documents. The coordinator must
    keep the word provisional until the time rule settles it, and must keep
    the time frontier at 300 ms meanwhile, or the re-emission and the next
    window's first word are both refused as rewriting frozen audio.
    """
    stream = await _stream(
        {
            0: _result(0, (_word("hola", 0, 300, 0), _word("bien", 800, 1_050, 1)), stable=1_000),
            1_000: _result(
                1_000, (_word("bien", 800, 1_050, 0), _word("luego", 1_000, 1_300, 1)), stable=2_000
            ),
        }
    )

    assert await stream.send_audio(ONE_SECOND)
    transcript = stream._coordinator.state.transcript
    assert [t.status for t in transcript.tokens] == [TokenStatus.FINAL, TokenStatus.PROVISIONAL]
    assert transcript.finalized_time_frontier.ms == 300

    # This is the call that raised before: `luego` starts at 1 000 ms, and a
    # frontier that had swept `bien` in stood at 1 050 ms.
    assert await stream.send_audio(ONE_SECOND, is_final=True)
    finals = await _drain(stream, ServerMessageType.TRANSCRIPT_FINAL)
    clocks = [event.monotonic_time_ms for event in finals]
    assert clocks == sorted(clocks), "the timeline must not run backwards"

    result = await stream.finish()
    assert result.transcript.raw_text == "hola bien luego"
    assert all(word.status == "final" for word in result.transcript.words)


async def test_the_event_clock_is_the_latest_end_not_the_last_words() -> None:
    """Two overlapping words: the lexically last ends first."""
    stream = await _stream(
        {
            0: _result(
                0,
                (
                    _word("hola", 0, 300, 0),
                    _word("laaargo", 400, 1_500, 1),
                    _word("corto", 1_200, 1_400, 2),
                ),
                stable=2_000,
            ),
        }
    )

    assert await stream.send_audio(ONE_SECOND)
    (final,) = await _drain(stream, ServerMessageType.TRANSCRIPT_FINAL)

    assert final.monotonic_time_ms == 1_500
    await stream.abort()


async def test_a_result_filed_under_the_wrong_window_is_refused() -> None:
    """The runtime reports 1 ms for the window handed to it at 0 ms.

    Its tokens would be sequenced under a window that was never ingested, and
    an unplaced one among them would wait forever for that window to commit.
    """
    stream = await _stream({0: _result(1, (_word("hola", 0, 300, 0),), stable=1_000)})

    with pytest.raises(MisdeclaredAudioWindow, match="reported window position 1 ms"):
        await stream.send_audio(ONE_SECOND)
