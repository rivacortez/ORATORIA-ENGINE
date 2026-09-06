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

*The document is assembled in canonical time order.* Events arrive keyed by id
and a streaming runtime may revise one long after a later one arrived, so
``state.speech_events.values()`` is in arrival order, not clock order. FR-029
makes that difference matter: a consumer reads position 0 as "first", so the
sequence order is a claim, and the only order that claims nothing is the
recording's own. Sorting here rather than in the serializer means the stored
rows, the streamed messages and the published document cannot disagree.

This function does *not* walk the serialized payload, and an earlier version
that claimed to did not either - it built a two-key stub of its own and walked
that, which could not fail, because the code building it only ever wrote ``id``
and ``type``. The application layer cannot walk the real payload: the
serializer lives in the inbound REST adapter and contract C6 forbids importing
it from here. The walk therefore lives where the payload actually exists,
in ``PostgresEvidenceRepository.store_document``, and the ordering half lives in
``EvidenceDocument.__post_init__`` where every construction path meets.
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
from evidence_engine.domain.shared.provenance import (
    Modality,
    ModelRole,
    Provenance,
    ProvenanceViolation,
)
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading
from evidence_engine.domain.transcript.transcript import Transcript
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

        speech_events = tuple(
            sorted((event.finalize() for event in state.speech_events.values()), key=_time_order)
        )
        visual_events = tuple(
            sorted((event.finalize() for event in state.visual_events.values()), key=_time_order)
        )

        cooccurrences = fuse(speech_events, visual_events, configuration.fusion_window)

        # No more audio and no more passes: whatever is still provisional is
        # final text. `finalize_remaining` said so in its docstring and had no
        # caller, so a session that ended mid-window published its tail as
        # revisable - which a consumer is entitled to keep re-rendering as
        # such. The events above get the same treatment two statements up.
        transcript = state.transcript.finalize_remaining()

        bundle = EvidenceBundle(
            run_id=state.run_id,
            session_id=session_id,
            tenant=caller.tenant,
            transcript=transcript,
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
                models=_models_used(transcript, speech_events, visual_events, tuple(state.prosody)),
            ),
            transcript=bundle.transcript,
            quality=bundle.quality,
            speech_events=bundle.speech_events,
            visual_events=bundle.visual_events,
            prosody=bundle.prosody,
            cooccurrences=bundle.cooccurrences,
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


def _time_order(event: SpeechEvent | VisualEvent) -> tuple[int, int, str]:
    """The canonical published order for a sequence of events.

    The same key ``transcript.build`` imposes on word tokens, reused rather
    than reinvented: two orderings that can disagree are worse than one, and
    the disagreement would surface as a transcript and an event list that tell
    different stories about which came first.

    The tie-break on the id matters. Two events can share a start - a window
    seam produces overlapping hypotheses - and without it the published order
    of those two would follow dictionary insertion, which NFR-015 requires to
    be reproducible and which arrival timing decides.
    """
    return event.interval.start.ms, event.interval.end.ms, event.id.value


def _loss_ratio(missing: int, highest_seen: int) -> float:
    """Fraction of expected chunks that never arrived."""
    expected = highest_seen + 1
    if expected <= 0:
        return 0.0
    return missing / expected


def _models_used(
    transcript: Transcript,
    speech_events: tuple[SpeechEvent, ...],
    visual_events: tuple[VisualEvent, ...],
    prosody: tuple[ProsodyReading, ...],
) -> dict[ModelRole, ModelVersionId]:
    """Collect the model version each *role* actually contributed under.

    Read off the evidence rather than from the registry, because during a
    canary the registry's answer and the answer for *this* evidence differ -
    and NFR-014 is about this evidence.

    Three things this used to get wrong, each of which produced a manifest
    that read as complete.

    *It read events only.* Tokens carried no provenance and prosody was
    never consulted, so a run that produced a transcript and no disfluency
    recorded no model at all: `models: {}` on five recognised words.

    *It was keyed by modality.* One audio model could be recorded. The
    contextual classifier QA-03 requires to be canaried independently of
    the recogniser had nowhere to go.

    *It used `setdefault`.* The second model in a modality was dropped
    without a trace. Now two versions for one role in one run is refused:
    a canary is *between* runs, and within a run it is a bug that would
    otherwise be recorded as whichever happened to come first.
    """
    models: dict[ModelRole, ModelVersionId] = {}

    def note(provenance: Provenance) -> None:
        seen = models.get(provenance.role)
        if seen is not None and seen != provenance.model_version:
            raise ProvenanceViolation(
                f"two versions of {provenance.role.value} in one run: {seen.value} and "
                f"{provenance.model_version.value}. A canary runs between sessions, "
                "not inside one; this document cannot say which model produced what"
            )
        models[provenance.role] = provenance.model_version

    # Iterated separately rather than as one concatenated sequence: the
    # types share no base beyond `object`, so a merged loop would erase the
    # very `provenance` attribute this function reads.
    for token in transcript.tokens:
        note(token.provenance)
    for speech_event in speech_events:
        note(speech_event.provenance)
    for visual_event in visual_events:
        note(visual_event.provenance)
    for reading in prosody:
        note(reading.provenance)
    return models
