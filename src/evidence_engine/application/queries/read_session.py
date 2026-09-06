"""Read-side use cases: session status, result document, capabilities.

Separated from the commands because they answer to different rules. A command
changes state and must audit; a query does neither, and giving it an audit
write would put a database insert on the polling path of every client waiting
for a batch job.

The scope split matters too. ``sessions:read`` gets lifecycle and status;
``results:read`` gets the evidence. An integration that only needs to show a
progress bar should not thereby be able to read every transcript.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.application.errors import (
    NotAuthorized,
    ResultNotReady,
    SessionNotFound,
)
from evidence_engine.application.ports.platform import AuthenticatedCaller, Scope
from evidence_engine.application.ports.repositories import (
    EvidenceRepository,
    SessionRepository,
)
from evidence_engine.application.ports.runtimes import SpeechRuntime, VisionRuntime
from evidence_engine.domain.evidence.document import EvidenceDocument
from evidence_engine.domain.sessions.capabilities import (
    MAX_FRAME_RATE_FPS,
    MIN_FRAME_RATE_FPS,
    SUPPORTED_LOCALES,
    SUPPORTED_SAMPLE_RATES_HZ,
    AudioCodec,
    VideoFormat,
)
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.sessions.state import SessionState
from evidence_engine.domain.shared.identifiers import SessionId
from evidence_engine.domain.shared.measurement import UnavailabilityReason
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import (
    TAXONOMY_VERSION,
    ProsodicIndicator,
    SpeechEventType,
    VisualEventType,
)


class ReadSession:
    """Query: a session's lifecycle and processing status (§7.1)."""

    def __init__(self, sessions: SessionRepository) -> None:
        self._sessions = sessions

    async def execute(self, caller: AuthenticatedCaller, session_id: SessionId) -> AnalysisSession:
        if not caller.allows(Scope.SESSIONS_READ):
            raise NotAuthorized(f"reading a session requires {Scope.SESSIONS_READ.value}")
        session = await self._sessions.get(caller.tenant, session_id)
        if session is None:
            raise SessionNotFound(f"session {session_id} not found")
        return session


class ReadResult:
    """Query: the complete evidence document (§7.1, US-013)."""

    def __init__(self, sessions: SessionRepository, evidence: EvidenceRepository) -> None:
        self._sessions = sessions
        self._evidence = evidence

    async def execute(self, caller: AuthenticatedCaller, session_id: SessionId) -> EvidenceDocument:
        if not caller.allows(Scope.RESULTS_READ):
            raise NotAuthorized(f"reading a result requires {Scope.RESULTS_READ.value}")

        session = await self._sessions.get(caller.tenant, session_id)
        if session is None:
            raise SessionNotFound(f"session {session_id} not found")

        if session.state is SessionState.EVIDENCE_DELETED:
            # Distinguishable from "not found" on purpose, and only here: the
            # caller already proved they can see this session, and US-005
            # promises a confirmation of deletion rather than a resource that
            # silently stops existing.
            raise ResultNotReady(
                f"evidence for session {session_id} was deleted under the retention policy"
            )

        if session.state is not SessionState.COMPLETED:
            raise ResultNotReady(
                f"session {session_id} is '{session.state.value}'; "
                "the evidence document is available once it completes"
            )

        document = await self._evidence.load_document(caller.tenant, session_id)
        if document is None:
            raise ResultNotReady(f"no evidence document stored for session {session_id}")
        return document


@dataclass(frozen=True, slots=True)
class UnavailableCapability:
    """A taxonomy class the contract defines and this deployment cannot emit."""

    kind: str
    name: str
    reason: UnavailabilityReason
    detail: str


def _absent(
    speech: frozenset[SpeechEventType],
    vision: frozenset[VisualEventType],
    prosody: frozenset[ProsodicIndicator],
    detail: str,
) -> tuple[UnavailableCapability, ...]:
    """The catalogue minus what is wired, each entry carrying its reason.

    Computed by subtraction rather than listed, so a class added to the
    taxonomy shows up here automatically until something is built that can
    emit it. A hand-written list would be a second declaration of the same
    fact, and it would be the one that goes stale.
    """
    absent: list[UnavailableCapability] = []
    for kind, members, emitted in (
        ("speech_event", SpeechEventType, speech),
        ("visual_event", VisualEventType, vision),
        ("prosodic_indicator", ProsodicIndicator, prosody),
    ):
        absent.extend(
            UnavailableCapability(
                kind=kind,
                name=member.value,
                reason=UnavailabilityReason.DETECTOR_NOT_DEPLOYED,
                detail=detail,
            )
            for member in sorted(set(members) - emitted, key=lambda m: m.value)
        )
    return tuple(absent)


@dataclass(frozen=True, slots=True)
class Capabilities:
    """§7.1 ``GET /v1/capabilities``: what this deployment accepts and emits.

    Published so a consuming application can check compatibility before it
    integrates rather than by trial. It includes the taxonomy and schema
    versions because NFR-017 promises compatibility only within one major
    version, and a client has to be able to see which one it is talking to.
    """

    schema_version: SemanticVersion
    taxonomy_version: SemanticVersion
    audio_codecs: tuple[str, ...]
    sample_rates_hz: tuple[int, ...]
    video_formats: tuple[str, ...]
    frame_rate_range_fps: tuple[int, int]
    locales: tuple[str, ...]

    #: **The taxonomy catalogue**: every class the published contract defines.
    #: A consumer reads this to know what the schema can carry, and must not
    #: read it to know what this deployment will send.
    speech_event_types: tuple[str, ...]
    visual_event_types: tuple[str, ...]

    #: **What the wired runtimes can actually produce.** This is the list a
    #: consumer builds against. It used to be absent, and the catalogue above
    #: stood in for it - so a deployment running a recogniser and no detector
    #: advertised nine disfluency classes it could not detect.
    emitted_speech_event_types: tuple[str, ...] = ()
    emitted_visual_event_types: tuple[str, ...] = ()
    emitted_prosodic_indicators: tuple[str, ...] = ()

    #: **What the contract defines and this deployment cannot produce**, each
    #: with a reason. Published as its own list rather than left as a
    #: subtraction the consumer performs, because the difference between "no
    #: findings" and "no detector" is the difference between a clean delivery
    #: and an unmeasured one.
    unavailable_capabilities: tuple[UnavailableCapability, ...] = ()
    #: §17's contract invariant, published rather than merely enforced. A
    #: consumer reading this knows the engine will never hand it a ranking, so
    #: it does not build one expecting the field to appear later.
    ranking_authority: str = EvidenceDocument.ranking_authority


class ReadCapabilities:
    """Query: the negotiation surface. Requires no scope beyond a valid key."""

    def __init__(
        self,
        schema_version: SemanticVersion,
        speech: SpeechRuntime | None = None,
        vision: VisionRuntime | None = None,
    ) -> None:
        self._schema_version = schema_version
        # Optional so a transport-only test can build this without wiring two
        # model runtimes. A container always supplies them; when neither is
        # present the query reports that nothing is emitted, which is the
        # honest answer for an engine with no runtimes rather than a fallback
        # to the catalogue.
        self._speech = speech
        self._vision = vision

    def execute(self) -> Capabilities:
        speech_emitted = self._speech.emitted_speech_events if self._speech else frozenset()
        vision_emitted = self._vision.emitted_visual_events if self._vision else frozenset()
        prosody_emitted = self._speech.emitted_prosody if self._speech else frozenset()
        detail = " ".join(
            runtime.capability_detail
            for runtime in (self._speech, self._vision)
            if runtime is not None
        )

        return Capabilities(
            emitted_speech_event_types=tuple(sorted(t.value for t in speech_emitted)),
            emitted_visual_event_types=tuple(sorted(t.value for t in vision_emitted)),
            emitted_prosodic_indicators=tuple(sorted(t.value for t in prosody_emitted)),
            unavailable_capabilities=_absent(
                speech_emitted, vision_emitted, prosody_emitted, detail
            ),
            schema_version=self._schema_version,
            taxonomy_version=TAXONOMY_VERSION,
            audio_codecs=tuple(sorted(codec.value for codec in AudioCodec)),
            sample_rates_hz=tuple(sorted(SUPPORTED_SAMPLE_RATES_HZ)),
            video_formats=tuple(sorted(fmt.value for fmt in VideoFormat)),
            frame_rate_range_fps=(MIN_FRAME_RATE_FPS, MAX_FRAME_RATE_FPS),
            locales=tuple(sorted(SUPPORTED_LOCALES)),
            speech_event_types=tuple(sorted(t.value for t in SpeechEventType)),
            visual_event_types=tuple(sorted(t.value for t in VisualEventType)),
        )
