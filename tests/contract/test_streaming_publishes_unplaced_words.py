"""A live session keeps, finalizes and publishes a word the aligner could not place.

§3.14 admitted the unplaced word into the transcript. §4.14 found the document
invariant reading `t.interval` unconditionally. The live path had two more
copies of the same assumption, and nothing drove an unplaced word through it:

*The frontier deleted it.* `_finalize_through` took the sequence boundary from
the *placed* tokens the time frontier had just settled. A window ending in an
unplaced word - whisper's alignment heads failing on the last word, which is
the ordinary case - left that word provisional, and the next window's
hypothesis replaced the provisional tail wholesale. The word was gone, and no
count anywhere disagreed.

*The publisher crashed on it.* `_publish_transcript` read `token.interval` on
every token. The first `transcript.final` carrying an unplaced word raised
`FabricatedValue` out of the coordinator and ended the session.

The runtime here does what whisper does: each window is decoded once, emits
only its own words, and is declared stable through its end. The second
window is what used to erase the first one's tail.

`transcript.partial` is not exercised because the engine never emits it: the
`is_final=False` branch of the publisher has no caller. Left as it is, and
noted, rather than activated in passing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from evidence_engine import EngineConfiguration, OratoriaEngine
from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    SpeechResult,
    TimedWordHypothesis,
    UntimedWordHypothesis,
)
from evidence_engine.application.ports.streaming import OutboundEvent, ServerMessageType
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable
from evidence_engine.domain.shared.provenance import ModelRole
from evidence_engine.domain.shared.taxonomy import ProsodicIndicator, SpeechEventType
from evidence_engine.sdk.stream import StreamSession

RECOGNISER = ModelVersionId("aligner-fails-v1")
ONE_SECOND = b"\x00\x00" * 16_000


class _WhisperShaped:
    """Window 0: `hola` placed, `mundo` unplaced at the tail. Window 1: `adios`.

    Every window is stable through its own end - whisper has no revisable
    tail - and emits only the words it decoded, as whisper does.
    """

    emitted_speech_events: frozenset[SpeechEventType] = frozenset()
    emitted_prosody: frozenset[ProsodicIndicator] = frozenset()
    capability_detail = "one unplaced word at the end of a window"

    @property
    def contributions(self) -> Mapping[ModelRole, ModelVersionId]:
        return {ModelRole.RECOGNISER: RECOGNISER}

    async def transcribe(self, window: AudioWindow) -> SpeechResult:
        absent = Unavailable(reason=UnavailabilityReason.POSTERIOR_NOT_REPORTED)
        position = window.session_position_ms
        end = position + window.duration_ms
        if position == 0:
            words: tuple[TimedWordHypothesis | UntimedWordHypothesis, ...] = (
                TimedWordHypothesis(raw_text="hola", start_ms=0, end_ms=300, score=absent, index=0),
                UntimedWordHypothesis(raw_text="mundo", score=absent, index=1),
            )
        elif position == 1_000:
            words = (
                TimedWordHypothesis(
                    raw_text="adios", start_ms=1_000, end_ms=1_300, score=absent, index=0
                ),
            )
        else:
            words = ()
        return SpeechResult(
            contributions=self.contributions,
            window_position_ms=position,
            words=words,
            stable_through_ms=end,
        )


async def _next(stream: StreamSession, type_: ServerMessageType) -> OutboundEvent:
    while True:
        event = await asyncio.wait_for(stream.receive(), timeout=5)
        if event.type is type_:
            return event


async def _warm_stream() -> StreamSession:
    engine = OratoriaEngine.local(
        EngineConfiguration(speech_runtime=_WhisperShaped(), window_seconds=1)
    )
    await engine.warmup()
    return engine.create_stream()


async def test_a_window_committed_whole_settles_its_unplaced_tail() -> None:
    """The frontier: `mundo` is final after window 0, because window 0 is.

    The runtime declared the whole window stable. The coordinator committed
    that audio, so every word it produced is settled - the unplaced one
    included. Under the placed-tokens rule it stayed provisional here and was
    erased by the next window.
    """
    stream = await _warm_stream()

    assert await stream.send_audio(ONE_SECOND)
    final = await _next(stream, ServerMessageType.TRANSCRIPT_FINAL)

    tokens = final.payload["tokens"]
    assert isinstance(tokens, list)
    assert [t["raw_text"] for t in tokens] == ["hola", "mundo"]

    hola, mundo = tokens
    assert hola["placed"] is True and (hola["start_ms"], hola["end_ms"]) == (0, 300)
    assert mundo["placed"] is False
    assert "start_ms" not in mundo and "end_ms" not in mundo and "tolerance_ms" not in mundo
    assert mundo["placement_unavailable_reason"] == "alignment_unavailable"
    # The recogniser is named on every word the wire carries (NFR-014), on the
    # live channel as on REST.
    assert {t["model_version"] for t in tokens} == {RECOGNISER.value}
    # The event's clock reading is the last *placed* word's end, not a number
    # invented for the unplaced one.
    assert final.monotonic_time_ms == 300

    await stream.abort()


async def test_the_next_window_does_not_erase_the_unplaced_tail() -> None:
    """The deletion, stated as an assertion."""
    stream = await _warm_stream()
    assert await stream.send_audio(ONE_SECOND)
    assert await stream.send_audio(ONE_SECOND, is_final=True)

    result = await stream.finish()

    assert result.transcript.raw_text == "hola mundo adios"
    assert result.transcript.unaligned_count == 1
    assert [w.is_timed for w in result.transcript.words] == [True, False, True]
    assert all(w.status == "final" for w in result.transcript.words)
