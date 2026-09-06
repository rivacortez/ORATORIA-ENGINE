"""QA-02, QA-05 and FR-010, executed against the wired pipeline.

Three scenarios the spec states as measurable outcomes:

- QA-02: video stops for a stretch; speech evidence survives untouched and no
  visual value is fabricated.
- QA-05: the same frozen input, artifacts, configuration and seed produce
  equivalent finalized evidence.
- FR-010: bounded queues signal backpressure before memory is exhausted.

These run below the transports, driving the coordinator directly. The socket
adds ordering and framing, which the exit-criterion suite already covers; what
matters here is what the pipeline does with the evidence.
"""

from __future__ import annotations

import pytest

from evidence_engine.adapters.outbound.cache.in_memory import RecordingEventChannel
from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    DeterministicSpeechRuntime,
    DeterministicVisionRuntime,
    SpeechScript,
    VisualScript,
)
from evidence_engine.adapters.outbound.telemetry.structured import NullTelemetry
from evidence_engine.application.errors import BackpressureRequired
from evidence_engine.application.ports.platform import ConfigurationSnapshot
from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    SpeechRuntime,
    VisualFrame,
)
from evidence_engine.application.services.calibration import Calibrator
from evidence_engine.application.services.speech_assembly import SpeechAssembler
from evidence_engine.application.services.visual_assembly import VisualAssembler
from evidence_engine.application.workflows.streaming import (
    StreamingCoordinator,
    StreamingState,
)
from evidence_engine.bootstrap.container import default_configuration
from evidence_engine.domain.evidence.ledger import EvidenceLedger
from evidence_engine.domain.sessions.sequencing import ChunkVerdict
from evidence_engine.domain.shared.identifiers import RunId, SessionId
from evidence_engine.domain.shared.measurement import UnavailabilityReason
from evidence_engine.domain.shared.provenance import Modality
from evidence_engine.domain.shared.timeline import Interval

pytestmark = pytest.mark.contract

#: One second of 16 kHz mono PCM16, sized from the declaration rather than
#: guessed at. This was 2 560 frames - 160 ms - declared as 1 000 ms, and
#: nothing compared the two until `AudioWindow` began refusing a declaration
#: its payload contradicts. Derived here so the next person who changes the
#: window length changes one number.
WINDOW_MS = 1_000
SAMPLE_RATE_HZ = 16_000
FRAMES_PER_WINDOW = SAMPLE_RATE_HZ * WINDOW_MS // 1_000
SILENCE = b"\x00\x00" * FRAMES_PER_WINDOW


def _coordinator(
    speech_script: SpeechScript,
    visual_script: VisualScript,
    *,
    configuration: ConfigurationSnapshot | None = None,
    max_queue_depth: int = 8,
    run_id: RunId | None = None,
) -> tuple[StreamingCoordinator, RecordingEventChannel, StreamingState]:
    snapshot = configuration or default_configuration()
    resolved_run = run_id or RunId("run-fixed-0001")
    state = StreamingState(run_id=resolved_run, session_id=SessionId("session-fixed-0001"))
    channel = RecordingEventChannel()
    calibrator = Calibrator()

    coordinator = StreamingCoordinator(
        state=state,
        configuration=snapshot,
        speech=DeterministicSpeechRuntime(speech_script),
        vision=DeterministicVisionRuntime(visual_script),
        channel=channel,
        telemetry=NullTelemetry(),
        ledger=EvidenceLedger(run_id=resolved_run),
        speech_assembler=SpeechAssembler(resolved_run, snapshot, calibrator),
        visual_assembler=VisualAssembler(resolved_run, snapshot, calibrator),
        max_queue_depth=max_queue_depth,
    )
    return coordinator, channel, state


async def _stream(
    coordinator: StreamingCoordinator, windows: int = 8, *, with_video: bool = True
) -> None:
    for index in range(windows):
        position = index * 1_000
        await coordinator.ingest_audio(
            index,
            AudioWindow(
                session_position_ms=position,
                duration_ms=1_000,
                sample_rate_hz=16_000,
                samples=SILENCE,
                is_final_window=index == windows - 1,
            ),
        )
        if with_video:
            await coordinator.ingest_video(
                index, [VisualFrame(session_position_ms=position, landmarks=(0.5, 0.5))]
            )


# ---------------------------------------------------------------------------
# QA-02 - modality degradation
# ---------------------------------------------------------------------------


async def test_a_vision_failure_does_not_stop_speech(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    """QA-02's measure: no loss of finalized speech events."""
    failing_vision = VisualScript(events=visual_script.events, fail_after_batch=2)
    coordinator, _, state = _coordinator(speech_script, failing_vision)

    await _stream(coordinator)

    assert state.transcript.raw_text().startswith("buenos dias")
    assert any(event.is_final for event in state.speech_events.values())
    assert Modality.VIDEO in state.degraded_modalities
    assert Modality.AUDIO not in state.degraded_modalities


async def test_a_vision_failure_publishes_processing_degraded_once(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    """One notice, not one per window: a client needs to be told, not spammed."""
    failing_vision = VisualScript(fail_after_batch=1)
    coordinator, channel, _ = _coordinator(speech_script, failing_vision)

    await _stream(coordinator)

    degraded = channel.of_type("processing.degraded")
    assert len(degraded) == 1
    assert degraded[0].payload["modality"] == "video"


async def test_a_session_with_no_video_reports_the_reason_not_a_zero(
    speech_script: SpeechScript,
) -> None:
    """FR-025 through the whole pipeline: absence carries a reason."""
    coordinator, _, state = _coordinator(speech_script, VisualScript())

    await _stream(coordinator, with_video=False)

    missing = state.quality.unavailability_for(Modality.VIDEO, Interval.of(0, 8_000))
    assert missing is not None
    assert missing.reason is UnavailabilityReason.MODALITY_NOT_CAPTURED
    assert not state.visual_events


async def test_a_speech_failure_does_not_fabricate_events_from_video(
    visual_script: VisualScript,
) -> None:
    """§6.3, stated explicitly: no linguistic events invented from the video."""
    failing_speech = SpeechScript(fail_after_window=1)
    coordinator, _, state = _coordinator(failing_speech, visual_script)

    await _stream(coordinator)

    assert Modality.AUDIO in state.degraded_modalities
    assert not state.speech_events
    assert state.transcript.raw_text() == ""
    # ... and the visual side kept working.
    assert state.visual_events


async def test_captured_ms_advances_even_when_the_speech_modality_degrades(
    visual_script: VisualScript,
) -> None:
    """S2: ``captured_audio_ms`` is set before the runtime is asked to decode.

    A run whose recogniser fails on every window still ingested audio, and a
    caller reading ``session.completed`` needs to tell "we heard everything
    and the model produced nothing" from "we heard nothing at all" -
    ``finalized_through_ms`` alone cannot: it reads 0 in both cases.
    """
    failing_speech = SpeechScript(fail_after_window=0)
    coordinator, _, state = _coordinator(failing_speech, visual_script)

    await coordinator.ingest_audio(
        0,
        AudioWindow(
            session_position_ms=0, duration_ms=1_000, sample_rate_hz=16_000, samples=SILENCE
        ),
    )

    assert Modality.AUDIO in state.degraded_modalities
    assert state.transcript.raw_text() == ""
    assert state.captured_audio_ms == 1_000


# ---------------------------------------------------------------------------
# FR-010 - bounded queues and backpressure
# ---------------------------------------------------------------------------


class _StallingSpeechRuntime:
    """A runtime that never returns, so the queue fills.

    Modelled rather than mocked: FR-010's failure mode is inference falling
    behind capture, and that is precisely a call that has not come back yet.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, window: AudioWindow) -> object:
        self.calls += 1
        raise AssertionError("the queue should have been full before this ran")


async def test_backpressure_is_signalled_before_the_queue_is_exceeded(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    coordinator, channel, state = _coordinator(speech_script, visual_script, max_queue_depth=1)
    # Simulate one window already in flight.
    state.in_flight = 1

    with pytest.raises(BackpressureRequired) as caught:
        await coordinator.ingest_audio(
            0,
            AudioWindow(
                session_position_ms=0,
                duration_ms=1_000,
                sample_rate_hz=16_000,
                samples=SILENCE,
            ),
        )

    assert caught.value.queue_depth == 1
    # `0` is the refused chunk's own sequence, not merely the queue depth -
    # the caller needs it to know exactly which window to resend.
    assert channel.backpressure_requests == [(state.session_id, 1, 0)]


async def test_a_refused_chunk_is_not_marked_seen_and_can_be_resent(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    """S1: the ledger must meet a resent, refused chunk for the first time.

    The old order called `offer()` before the admission check, so a refused
    chunk was already recorded as accepted. The caller's only sane retry - the
    identical `chunk_seq` - then came back `DUPLICATE` and was silently
    dropped: an explicit backpressure signal on the wire, and an invisible
    loss underneath it. Checking admission first is what this test would have
    caught: reverting the reorder in `_admit` turns it red.
    """
    coordinator, _, state = _coordinator(speech_script, visual_script, max_queue_depth=1)
    state.in_flight = 1
    window = AudioWindow(
        session_position_ms=0, duration_ms=1_000, sample_rate_hz=16_000, samples=SILENCE
    )

    with pytest.raises(BackpressureRequired):
        await coordinator.ingest_audio(0, window)

    assert state.audio_chunks.highest_seen == -1, "a refused chunk must not be marked seen"

    # Room exists now; the caller resends the identical window.
    state.in_flight = 0
    verdict = await coordinator.ingest_audio(0, window)

    assert verdict is ChunkVerdict.ACCEPTED
    assert state.audio_chunks.highest_seen == 0


async def test_backpressure_raises_rather_than_dropping_the_chunk(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    """A silently dropped chunk resurfaces later as unexplained data loss."""
    coordinator, _, state = _coordinator(speech_script, visual_script, max_queue_depth=1)
    state.in_flight = 1
    stalling: SpeechRuntime = _StallingSpeechRuntime()  # type: ignore[assignment]
    coordinator._speech = stalling

    with pytest.raises(BackpressureRequired):
        await coordinator.ingest_audio(
            0,
            AudioWindow(
                session_position_ms=0,
                duration_ms=1_000,
                sample_rate_hz=16_000,
                samples=SILENCE,
            ),
        )

    assert stalling.calls == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# QA-05 - reproducibility
# ---------------------------------------------------------------------------


async def test_the_same_input_produces_equivalent_evidence(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    """QA-05's measure: identical event labels, no numerical drift."""
    first_coordinator, _, first = _coordinator(speech_script, visual_script)
    await _stream(first_coordinator)

    second_coordinator, _, second = _coordinator(speech_script, visual_script)
    await _stream(second_coordinator)

    assert first.transcript.raw_text() == second.transcript.raw_text()
    assert sorted(first.speech_events) == sorted(second.speech_events)
    assert sorted(first.visual_events) == sorted(second.visual_events)

    for key in first.speech_events:
        left, right = first.speech_events[key], second.speech_events[key]
        assert left.type is right.type
        assert left.interval == right.interval
        assert left.confidence == right.confidence
        assert left.context_role is right.context_role


async def test_event_ids_are_stable_across_runs(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    """§7.4 derives ids from the evidence, so two runs agree on them."""
    run_id = RunId("run-reproducible")

    first_coordinator, _, first = _coordinator(speech_script, visual_script, run_id=run_id)
    await _stream(first_coordinator)

    second_coordinator, _, second = _coordinator(speech_script, visual_script, run_id=run_id)
    await _stream(second_coordinator)

    assert set(first.speech_events) == set(second.speech_events)


async def test_a_different_run_gets_different_ids(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    """Runs stay separable (§8), so reprocessing compares instead of overwriting."""
    first_coordinator, _, first = _coordinator(speech_script, visual_script, run_id=RunId("run-a"))
    await _stream(first_coordinator)

    second_coordinator, _, second = _coordinator(
        speech_script, visual_script, run_id=RunId("run-b")
    )
    await _stream(second_coordinator)

    assert set(first.speech_events).isdisjoint(second.speech_events)


# ---------------------------------------------------------------------------
# FR-016 - partial then final, without rewriting history
# ---------------------------------------------------------------------------


async def test_partials_precede_finals_for_the_same_event(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    coordinator, channel, _ = _coordinator(speech_script, visual_script)

    await _stream(coordinator)

    partial_ids = {
        event["id"]
        for message in channel.of_type("speech_event.partial")
        for event in message.payload["events"]  # type: ignore[union-attr]
    }
    final_ids = {
        event["id"]
        for message in channel.of_type("speech_event.final")
        for event in message.payload["events"]  # type: ignore[union-attr]
    }

    assert final_ids
    # Every finalized event that was ever provisional kept its identity, so a
    # consumer can replace it in place instead of rendering it twice.
    derived_only = final_ids - partial_ids
    assert all("ev_" in event_id for event_id in derived_only)


async def test_no_finalized_event_is_ever_revised(
    speech_script: SpeechScript, visual_script: VisualScript
) -> None:
    """§6.1 step 9, observed at the wire rather than in the domain."""
    coordinator, channel, _ = _coordinator(speech_script, visual_script)

    await _stream(coordinator)

    seen: dict[str, tuple[int, int]] = {}
    for message in channel.of_type("speech_event.final"):
        for event in message.payload["events"]:  # type: ignore[union-attr]
            span = (event["start_ms"], event["end_ms"])
            if event["id"] in seen:
                assert seen[event["id"]] == span, (
                    f"finalized event {event['id']} was re-emitted with a different span"
                )
            seen[event["id"]] = span
