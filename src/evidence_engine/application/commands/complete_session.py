"""Complete a session and assemble its evidence document (§6.1 step 10).

The final pass is where the last three things happen that cannot happen during
streaming.

*Chunk gaps are confirmed.* Reordering is normal on an unreliable transport, so
a chunk that has not arrived yet is not lost yet. Only at close does the ledger
decide, and only then can FR-024 report which windows were unreliable.

*Co-occurrences are computed.* FR-027 pairs finalized events, and there are no
finalized events to pair until finalization has run. §7.3 has no partial
co-occurrence message for the same reason: a correlation between two events one
of which is later revised would be a claim the engine has to withdraw, which
§6.1 step 9 does not allow.

*The document is assembled and checked.* ``assert_carries_no_ranking`` walks the
serialized payload before it is stored. That check is redundant with the
domain's ``ClassVar`` in the normal case and exists for the abnormal one: a
model runtime that forwards its own ordering inside an event payload.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.application.errors import NotAuthorized, SessionNotFound
from evidence_engine.application.ports.clock import Clock
from evidence_engine.application.ports.platform import (
    AuthenticatedCaller,
    ConfigurationSnapshot,
    Scope,
)
from evidence_engine.application.ports.repositories import (
    AuditLog,
    AuditRecord,
    EvidenceBundle,
    EvidenceRepository,
    SessionRepository,
)
from evidence_engine.application.services.fusion import fuse
from evidence_engine.application.workflows.streaming import StreamingState
from evidence_engine.domain.evidence.document import (
    EvidenceDocument,
    ProvenanceManifest,
    assert_carries_no_ranking,
)
from evidence_engine.domain.quality.assessment import (
    QualityAssessment,
    QualityMetric,
    QualityReport,
)
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.identifiers import ModelVersionId, SessionId
from evidence_engine.domain.shared.measurement import Measured
from evidence_engine.domain.shared.provenance import Modality
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.visual_events.events import VisualEvent


@dataclass(frozen=True, slots=True)
class CompletedSession:
    """The session in its final state, with the document it produced."""

    session: AnalysisSession
    document: EvidenceDocument


class CompleteSession:
    """Use case: close capture, reconcile, assemble and store the result."""

    def __init__(
        self,
        sessions: SessionRepository,
        evidence: EvidenceRepository,
        audit: AuditLog,
        clock: Clock,
    ) -> None:
        self._sessions = sessions
        self._evidence = evidence
        self._audit = audit
        self._clock = clock

    async def execute(
        self,
        caller: AuthenticatedCaller,
        session_id: SessionId,
        state: StreamingState,
        configuration: ConfigurationSnapshot,
    ) -> CompletedSession:
        if not caller.allows(Scope.SESSIONS_WRITE):
            raise NotAuthorized(f"completing a session requires {Scope.SESSIONS_WRITE.value}")

        session = await self._sessions.get(caller.tenant, session_id)
        if session is None:
            raise SessionNotFound(f"session {session_id} not found")

        wall_ms = self._clock.wall_ms()
        session = session.request_completion(wall_ms)
        await self._sessions.save(session)

        quality = self._with_transport_quality(state, session, wall_ms)

        speech_events = tuple(event.finalize() for event in state.speech_events.values())
        visual_events = tuple(event.finalize() for event in state.visual_events.values())

        cooccurrences = fuse(speech_events, visual_events, configuration.fusion_window)

        bundle = EvidenceBundle(
            run_id=state.run_id,
            session_id=session_id,
            tenant=caller.tenant,
            transcript=state.transcript,
            quality=quality,
            speech_events=speech_events,
            visual_events=visual_events,
            prosody=tuple(state.prosody),
            cooccurrences=cooccurrences,
        )

        document = EvidenceDocument(
            session_id=session_id,
            run_id=state.run_id,
            manifest=ProvenanceManifest(
                pipeline_version=configuration.pipeline_version,
                schema_version=configuration.schema_version,
                taxonomy_version=configuration.taxonomy_version,
                configuration=configuration.id,
                models=_models_used(speech_events, visual_events),
            ),
            transcript=bundle.transcript,
            quality=bundle.quality,
            speech_events=bundle.speech_events,
            visual_events=bundle.visual_events,
            prosody=bundle.prosody,
            cooccurrences=bundle.cooccurrences,
        )

        # Belt and braces before anything leaves this process (FR-029).
        assert_carries_no_ranking(
            {
                "speech_events": [{"id": e.id.value, "type": e.type.value} for e in speech_events],
                "visual_events": [{"id": e.id.value, "type": e.type.value} for e in visual_events],
            }
        )

        await self._evidence.store(bundle)
        await self._evidence.store_document(document, caller.tenant)

        now_ms = self._clock.epoch_ms()
        session = session.mark_completed(now_ms)
        await self._sessions.save(session)

        await self._audit.record(
            AuditRecord(
                actor=caller.application.value,
                action="session.complete",
                resource=session_id.value,
                timestamp_ms=now_ms,
                trace_id=caller.trace_id,
                outcome="completed",
                tenant=caller.tenant,
                detail=(
                    f"speech_events={len(speech_events)} "
                    f"visual_events={len(visual_events)} "
                    f"cooccurrences={len(cooccurrences)}"
                ),
            )
        )

        return CompletedSession(session=session, document=document)

    def _with_transport_quality(
        self, state: StreamingState, session: AnalysisSession, wall_ms: int
    ) -> QualityReport:
        """Fold confirmed chunk loss into the quality report (FR-008, FR-024).

        Confirmed here and nowhere earlier: a gap during streaming is a chunk
        that has not arrived *yet*. Reporting it live would fill a healthy
        session with warnings that a late packet immediately contradicts.
        """
        captured = Interval.of(0, max(session.captured_ms(wall_ms), 0))
        certain = Confidence.calibrated(1.0)

        # Called for effect: `finalize` is what turns provisional gaps into
        # confirmed ones and populates `missing_count`.
        state.audio_chunks.finalize()
        state.video_chunks.finalize()

        assessments = [
            QualityAssessment(
                modality=Modality.AUDIO,
                metric=QualityMetric.CHUNK_LOSS_RATIO,
                window=captured,
                value=Measured(
                    _loss_ratio(state.audio_chunks.missing_count, state.audio_chunks.highest_seen),
                    certain,
                    "ratio",
                ),
            )
        ]
        if state.video_chunks.highest_seen >= 0:
            assessments.append(
                QualityAssessment(
                    modality=Modality.VIDEO,
                    metric=QualityMetric.CHUNK_LOSS_RATIO,
                    window=captured,
                    value=Measured(
                        _loss_ratio(
                            state.video_chunks.missing_count, state.video_chunks.highest_seen
                        ),
                        certain,
                        "ratio",
                    ),
                )
            )

        return state.quality.extended(assessments)


def _loss_ratio(missing: int, highest_seen: int) -> float:
    """Fraction of expected chunks that never arrived."""
    expected = highest_seen + 1
    if expected <= 0:
        return 0.0
    return missing / expected


def _models_used(
    speech_events: tuple[SpeechEvent, ...], visual_events: tuple[VisualEvent, ...]
) -> dict[Modality, ModelVersionId]:
    """Collect the model version each modality actually contributed under.

    Read off the events rather than from the registry, because during a canary
    the registry's answer and the answer for *these* events differ - and
    NFR-014 is about these events.
    """
    # Iterated separately rather than as one concatenated sequence: the two
    # event types share no base beyond `object`, so a merged loop would erase
    # the very `provenance` attribute this function reads.
    models: dict[Modality, ModelVersionId] = {}
    for speech_event in speech_events:
        models.setdefault(speech_event.provenance.modality, speech_event.provenance.model_version)
    for visual_event in visual_events:
        models.setdefault(visual_event.provenance.modality, visual_event.provenance.model_version)
    return models
