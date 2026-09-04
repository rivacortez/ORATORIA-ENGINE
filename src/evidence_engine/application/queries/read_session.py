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
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import (
    TAXONOMY_VERSION,
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
    speech_event_types: tuple[str, ...]
    visual_event_types: tuple[str, ...]
    #: §17's contract invariant, published rather than merely enforced. A
    #: consumer reading this knows the engine will never hand it a ranking, so
    #: it does not build one expecting the field to appear later.
    ranking_authority: str = EvidenceDocument.ranking_authority


class ReadCapabilities:
    """Query: the negotiation surface. Requires no scope beyond a valid key."""

    def __init__(self, schema_version: SemanticVersion) -> None:
        self._schema_version = schema_version

    def execute(self) -> Capabilities:
        return Capabilities(
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
