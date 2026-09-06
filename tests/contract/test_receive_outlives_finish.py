"""``receive()`` keeps draining once ``finish()`` has closed the stream - both paths.

Found live, not in a test. OratorIA's adapter runs a receiver task beside the
feeder that calls ``finish()``. Against the hosted engine, the final window's
``transcript.final`` and ``session.completed`` arrive AFTER ``finish()`` has
marked the stream closed; the receiver's next ``receive()`` hit
``_require_open`` and the whole session failed as incomplete - with the last
2.4 s of audio transcribed on the server and thrown away on the client. The
embedded path never showed it, only because its ``_finish`` publishes nothing
through the channel that a receiver could wake up for.

The contract now, in the same words for both sessions: ``receive()`` answers
until nothing more can arrive - the stream settled and the queue is empty -
and only then raises ``StreamAlreadyClosed``.
"""

from __future__ import annotations

import pytest

from evidence_engine import EngineConfiguration
from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    DeterministicSpeechRuntime,
    ScriptedWord,
    SpeechScript,
)
from evidence_engine.application.ports.streaming import ServerMessageType
from evidence_engine.sdk.client import OratoriaClient
from evidence_engine.sdk.engine import OratoriaEngine
from evidence_engine.sdk.errors import StreamAlreadyClosed

from .test_remote_client import (
    SILENT_CHUNK,
    _accepted_frame,
    _created_session_response,
    _FakeTransport,
)

pytestmark = pytest.mark.contract


def _frame(event_seq: int, type_: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "session_id": "sess-0001",
        "message_id": f"msg-{event_seq}",
        "event_seq": event_seq,
        "monotonic_time_ms": 1_000,
        "type": type_,
        "payload": payload,
    }


_MINIMAL_DOCUMENT: dict[str, object] = {
    "schema_version": "1.0.0",
    "session_id": "sess-0001",
    "run_id": "run-0001",
    "ranking_authority": "none",
    "manifest": {
        "pipeline_version": "0.1.0",
        "schema_version": "1.0.0",
        "taxonomy_version": "1.0.0",
        "configuration_id": "config-default-v1",
        "models": {"recogniser": "deterministic-speech-v1"},
    },
    "transcript": {
        "raw_text": "",
        "finalized_through_ms": 1_000,
        "unaligned_token_count": 0,
        "tokens": [],
    },
    "speech_events": [],
    "visual_events": [],
    "prosody": [],
    "cooccurrences": [],
    "quality": {"assessments": [], "availability": []},
}


async def test_the_remote_session_hands_over_what_arrived_after_finish_closed_it() -> None:
    """The final window's ``transcript.final`` and ``session.completed`` land
    while ``finish()`` is in flight; a concurrent receiver gets both, and is
    refused only after the last one."""
    transport = _FakeTransport(
        post_response=_created_session_response(),
        get_responses={"/result": [(200, _MINIMAL_DOCUMENT)]},
        ws_frames=[
            _accepted_frame(),
            _frame(2, "transcript.final", {"tokens": []}),
            _frame(
                3,
                "session.completed",
                {"run_id": "run-0001", "finalized_through_ms": 1_000, "captured_ms": 1_000},
            ),
        ],
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)
    stream = client.create_stream()
    assert await stream.send_audio(SILENT_CHUNK, is_final=True) is True

    await stream.finish()  # closes the stream for sending; the two frames are queued

    first = await stream.receive()
    second = await stream.receive()
    assert (first.type, second.type) == (
        ServerMessageType.TRANSCRIPT_FINAL,
        ServerMessageType.SESSION_COMPLETED,
    )
    with pytest.raises(StreamAlreadyClosed):
        await stream.receive()


async def test_the_embedded_session_hands_over_what_it_published_before_finish() -> None:
    """Events the coordinator published for earlier windows stay receivable
    after ``finish()`` returned; the refusal comes only with an empty queue."""
    script = SpeechScript(words=(ScriptedWord("hola", 0, 300),))
    engine = OratoriaEngine.local(
        EngineConfiguration(speech_runtime=DeterministicSpeechRuntime(script), window_seconds=1)
    )
    await engine.warmup()
    stream = engine.create_stream()
    try:
        await stream.send_audio(SILENT_CHUNK)
        await stream.send_audio(SILENT_CHUNK, is_final=True)
        queued = stream.pending()
        assert queued > 0, "the scripted runtime must have published something to drain"

        await stream.finish()

        drained = 0
        while stream.pending() > 0:
            await stream.receive()
            drained += 1
        assert drained == queued
        with pytest.raises(StreamAlreadyClosed):
            await stream.receive()
    finally:
        await engine.aclose()
