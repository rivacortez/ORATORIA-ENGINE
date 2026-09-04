"""The streaming coordinator: chunks in, partial and final evidence out.

This is §6.1 as executable code, and the three rules it enforces are the ones
that make real-time behaviour survivable.

*Bounded queues (FR-010, §11.3 iteration 3).* Inference is slower than capture
during a burst. Unbounded buffering turns that into memory exhaustion; dropping
chunks turns it into evidence that silently disappears. So the queue has a
ceiling and hitting it raises ``BackpressureRequired``, which the transport
turns into §7.3's explicit ``backpressure.requested``. US-012 makes that part
of the contract: a client that keeps sending is misbehaving, not unlucky.

*Partial/final reconciliation (FR-016, §6.1 step 9).* The runtime says how far
its output is stable and the coordinator finalizes only to that point.
Finalizing on a fixed lag would be a guess, and finalization is irreversible.

*Modality independence (QA-02, §6.3).* A vision failure marks the visual
indicators unavailable and publishes ``processing.degraded``. It does not fail
the session, and it never causes the engine to fabricate visual evidence from
the audio - §6.3 forbids the reverse case explicitly and the same reasoning
applies here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from evidence_engine.application.errors import BackpressureRequired
from evidence_engine.application.ports.platform import ConfigurationSnapshot, Telemetry
from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    SpeechRuntime,
    VisionRuntime,
    VisualFrame,
)
from evidence_engine.application.ports.streaming import (
    DegradationNotice,
    EventChannel,
    OutboundEvent,
    ServerMessageType,
)
from evidence_engine.application.services.speech_assembly import SpeechAssembler, silent_pauses
from evidence_engine.application.services.visual_assembly import VisualAssembler
from evidence_engine.domain.evidence.ledger import EvidenceLedger
from evidence_engine.domain.quality.assessment import ModalityAvailability, QualityReport
from evidence_engine.domain.sessions.sequencing import ChunkLedger, ChunkVerdict
from evidence_engine.domain.shared.identifiers import RunId, SessionId
from evidence_engine.domain.shared.provenance import Modality, Provenance
from evidence_engine.domain.shared.timeline import MonotonicTime
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading
from evidence_engine.domain.transcript.transcript import Transcript
from evidence_engine.domain.visual_events.events import VisualEvent

#: Windows allowed to be in flight before the client is told to slow down.
#: Small on purpose: NFR-005 targets a p95 partial latency of 1.5 s, and a deep
#: queue meets a throughput target by spending exactly the latency budget the
#: user experiences. Backpressure early is the honest failure.
DEFAULT_MAX_QUEUE_DEPTH = 8


@dataclass(slots=True)
class StreamingState:
    """Everything accumulated for one live session.

    Mutable and single-owner. The session lease in ``StreamState`` guarantees
    the second half of that: two handlers advancing one chunk ledger would
    interleave sequence numbers and manufacture gaps FR-008 would then report
    as data loss.
    """

    run_id: RunId
    session_id: SessionId
    transcript: Transcript = field(default_factory=Transcript)
    audio_chunks: ChunkLedger = field(default_factory=ChunkLedger)
    video_chunks: ChunkLedger = field(default_factory=ChunkLedger)
    speech_events: dict[str, SpeechEvent] = field(default_factory=dict)
    visual_events: dict[str, VisualEvent] = field(default_factory=dict)
    prosody: list[ProsodyReading] = field(default_factory=list)
    quality: QualityReport = field(default_factory=QualityReport)
    in_flight: int = 0
    #: Set once a modality has failed, so the degradation notice is published
    #: on the first failure rather than on every subsequent window.
    degraded_modalities: set[Modality] = field(default_factory=set)

    def ledger_for(self, modality: Modality) -> ChunkLedger:
        return self.audio_chunks if modality is Modality.AUDIO else self.video_chunks


class StreamingCoordinator:
    """Drives one live session from first chunk to final reconciliation."""

    def __init__(
        self,
        state: StreamingState,
        configuration: ConfigurationSnapshot,
        speech: SpeechRuntime,
        vision: VisionRuntime,
        channel: EventChannel,
        telemetry: Telemetry,
        ledger: EvidenceLedger,
        speech_assembler: SpeechAssembler,
        visual_assembler: VisualAssembler,
        max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
    ) -> None:
        self._state = state
        self._configuration = configuration
        self._speech = speech
        self._vision = vision
        self._channel = channel
        self._telemetry = telemetry
        self._ledger = ledger
        self._speech_assembler = speech_assembler
        self._visual_assembler = visual_assembler
        self._max_queue_depth = max_queue_depth

    @property
    def state(self) -> StreamingState:
        """Everything accumulated so far.

        Exposed for the completion path, which assembles the document from it.
        Read-only by convention rather than by copy: the state holds thousands
        of events by the end of a session and duplicating it per read would
        cost more than the encapsulation buys.
        """
        return self._state

    # -- audio ------------------------------------------------------------

    async def ingest_audio(self, sequence: int, window: AudioWindow) -> ChunkVerdict:
        """Process one audio window, or say why it was not processed."""
        verdict = self._state.audio_chunks.offer(sequence)
        if verdict is ChunkVerdict.DUPLICATE:
            # §7.4: redelivery is idempotent. Counting it would double every
            # rate on a flaky connection.
            self._telemetry.counter("chunks.duplicate", modality="audio")
            return verdict

        await self._admit()
        self._state.in_flight += 1
        try:
            result = await self._speech.transcribe(window)
        except Exception as failure:
            # Deliberately broad. A runtime is a foreign process behind a port
            # and may fail in ways this layer cannot enumerate; QA-02 requires
            # the session to survive whatever it does. The failure is recorded
            # and published, never swallowed.
            await self._degrade(Modality.AUDIO, failure)
            return ChunkVerdict.ACCEPTED
        finally:
            self._state.in_flight -= 1

        assembled = self._speech_assembler.assemble(result)

        self._state.transcript = self._state.transcript.with_provisional(assembled.tokens)
        for event in assembled.events:
            self._state.speech_events[event.id.value] = event
        self._state.prosody.extend(assembled.prosody)

        await self._publish_speech(assembled.events, is_final=False)
        await self._finalize_through(assembled.stable_through_ms)
        return verdict

    async def _finalize_through(self, stable_through_ms: int) -> None:
        """Freeze everything the runtime considers settled (§6.1 step 9)."""
        if stable_through_ms <= self._state.transcript.finalized_frontier.ms:
            return

        boundary = MonotonicTime(stable_through_ms)
        self._state.transcript = self._state.transcript.finalize_through(boundary)

        newly_final: list[SpeechEvent] = []
        for key, event in list(self._state.speech_events.items()):
            if event.is_final or event.interval.end.ms > stable_through_ms:
                continue
            finalized = event.finalize()
            self._state.speech_events[key] = finalized
            newly_final.append(finalized)
            self._record(finalized)

        # Silent pauses are derived from finalized tokens only. Deriving them
        # from provisional ones would publish a pause that vanishes when the
        # recognizer fills the gap with a word it had not yet decoded.
        pauses = silent_pauses(
            self._state.transcript.final_tokens,
            self._configuration.silence_threshold_ms,
            self._speech_provenance(newly_final),
            self._state.run_id,
        )
        for pause in pauses:
            if pause.id.value in self._state.speech_events:
                continue
            final_pause = pause.finalize()
            self._state.speech_events[pause.id.value] = final_pause
            newly_final.append(final_pause)
            self._record(final_pause)

        await self._publish_transcript(is_final=True)
        await self._publish_speech(tuple(newly_final), is_final=True)

    # -- video ------------------------------------------------------------

    async def ingest_video(self, sequence: int, frames: Sequence[VisualFrame]) -> ChunkVerdict:
        """Process a batch of frames. A failure here never fails the session."""
        verdict = self._state.video_chunks.offer(sequence)
        if verdict is ChunkVerdict.DUPLICATE:
            self._telemetry.counter("chunks.duplicate", modality="video")
            return verdict

        await self._admit()
        self._state.in_flight += 1
        try:
            result = await self._vision.observe(frames)
        except Exception as failure:
            await self._degrade(Modality.VIDEO, failure)
            return ChunkVerdict.ACCEPTED
        finally:
            self._state.in_flight -= 1

        assembled = self._visual_assembler.assemble(result)
        for event in assembled.events:
            self._state.visual_events[event.id.value] = event
        self._state.quality = self._state.quality.extended(
            assembled.assessments, assembled.availability
        )
        if assembled.gated_out:
            self._telemetry.counter("visual_events.gated_out", assembled.gated_out)

        await self._publish_quality_warnings(assembled.availability)
        return verdict

    # -- shared -----------------------------------------------------------

    async def _admit(self) -> None:
        """Refuse work once the bounded queue is full (FR-010)."""
        if self._state.in_flight < self._max_queue_depth:
            return
        await self._channel.request_backpressure(self._state.session_id, self._state.in_flight)
        self._telemetry.counter("backpressure.requested")
        raise BackpressureRequired(
            f"{self._state.in_flight} windows already in flight; slow down",
            queue_depth=self._state.in_flight,
        )

    async def _degrade(self, modality: Modality, failure: Exception) -> None:
        """Mark a modality down and tell the client, without failing the session."""
        first_time = modality not in self._state.degraded_modalities
        self._state.degraded_modalities.add(modality)
        self._telemetry.counter("modality.degraded", modality=modality.value)

        if not first_time:
            return

        notice = DegradationNotice(
            modality=modality,
            reason=type(failure).__name__,
            detail=(
                f"{modality.value} processing failed; its indicators will be reported "
                "as unavailable for the affected windows"
            ),
        )
        await self._channel.publish(
            OutboundEvent(
                type=ServerMessageType.PROCESSING_DEGRADED,
                session_id=self._state.session_id,
                monotonic_time_ms=self._state.transcript.finalized_frontier.ms,
                payload={
                    "modality": notice.modality.value,
                    "reason": notice.reason,
                    "detail": notice.detail,
                },
            )
        )

    def _record(self, event: SpeechEvent) -> None:
        """Append a finalized event to the immutable ledger."""
        if event.id in self._ledger:
            self._ledger.finalize(event.id)
            return
        self._ledger.append(
            event.id,
            event.interval,
            event.provenance,
            is_final=True,
            payload={"type": event.type.value, "raw_text": event.raw_text},
        )

    def _speech_provenance(self, events: Sequence[SpeechEvent]) -> Provenance:
        """Provenance for events this layer derives rather than receives.

        Borrowed from a real event of the same run so the derived pause carries
        the same model and configuration versions as the tokens it was computed
        from. Falling back to a synthetic provenance would make NFR-014's
        traceability a half-truth for exactly the events nobody inspects.
        """
        if events:
            return events[0].provenance
        for event in self._state.speech_events.values():
            return event.provenance
        raise RuntimeError(
            "no speech provenance available; a derived event cannot be built before "
            "the runtime has produced at least one observation"
        )

    # -- publishing -------------------------------------------------------

    async def _publish_transcript(self, *, is_final: bool) -> None:
        tokens = (
            self._state.transcript.final_tokens
            if is_final
            else self._state.transcript.provisional_tokens
        )
        if not tokens:
            return
        await self._channel.publish(
            OutboundEvent(
                type=(
                    ServerMessageType.TRANSCRIPT_FINAL
                    if is_final
                    else ServerMessageType.TRANSCRIPT_PARTIAL
                ),
                session_id=self._state.session_id,
                monotonic_time_ms=tokens[-1].interval.end.ms,
                payload={
                    "tokens": [
                        {
                            "id": token.id.value,
                            "raw_text": token.raw_text,
                            "start_ms": token.interval.start.ms,
                            "end_ms": token.interval.end.ms,
                            "tolerance_ms": token.interval.tolerance_ms,
                            "confidence": token.confidence.value,
                            "calibration": token.confidence.state.value,
                        }
                        for token in tokens
                    ]
                },
            )
        )

    async def _publish_speech(self, events: Sequence[SpeechEvent], *, is_final: bool) -> None:
        if not events:
            return
        await self._channel.publish(
            OutboundEvent(
                type=(
                    ServerMessageType.SPEECH_EVENT_FINAL
                    if is_final
                    else ServerMessageType.SPEECH_EVENT_PARTIAL
                ),
                session_id=self._state.session_id,
                monotonic_time_ms=max(e.interval.end.ms for e in events),
                payload={"events": [_speech_payload(e) for e in events]},
            )
        )

    async def _publish_quality_warnings(self, availability: Sequence[ModalityAvailability]) -> None:
        """Tell the client which windows will have no numbers, and why.

        Published as they are found rather than only at completion, because
        US-001 wants the student told which indicators will not be calculated
        while there is still time to uncover the camera.
        """
        unusable = [record for record in availability if not record.is_usable]
        if not unusable:
            return
        await self._channel.publish(
            OutboundEvent(
                type=ServerMessageType.QUALITY_WARNING,
                session_id=self._state.session_id,
                monotonic_time_ms=self._state.transcript.finalized_frontier.ms,
                payload={
                    "windows": [
                        {
                            "modality": record.modality.value,
                            # `is_usable` is False here, so the domain guarantees
                            # a reason is present - it refuses to construct the
                            # record otherwise.
                            "reason": record.reason.value if record.reason else "",
                            "detail": record.detail,
                            "start_ms": record.window.start.ms,
                            "end_ms": record.window.end.ms,
                        }
                        for record in unusable
                    ]
                },
            )
        )


def _speech_payload(event: SpeechEvent) -> dict[str, object]:
    """Wire shape for one speech event.

    Carries no rank, score or severity - FR-029 keeps selection with the
    consumer - and every field here is either an observation or the provenance
    NFR-014 requires alongside it.
    """
    return {
        "id": event.id.value,
        "type": event.type.value,
        "start_ms": event.interval.start.ms,
        "end_ms": event.interval.end.ms,
        "tolerance_ms": event.interval.tolerance_ms,
        "confidence": event.confidence.value,
        "calibration": event.confidence.state.value,
        "raw_text": event.raw_text,
        "context_role": event.context_role.value if event.context_role else None,
        "modality": event.provenance.modality.value,
        "model_version": event.provenance.model_version.value,
        "taxonomy_version": str(event.provenance.taxonomy_version),
        "evidence_ref": event.provenance.evidence_ref.value,
    }
