"""``OratoriaEngine`` - the embedded engine, composed without a server.

The facade a consumer of the SDK holds. It runs the **same** use cases the
hosted service runs, over in-memory adapters, in the caller's own process.

**Why it composes use cases instead of calling a runtime.** The short version
of this class would take an audio file, hand it to `WhisperSpeechRuntime` and
return words. That would be a second functional path, and the state machine,
the consent check, the quota, the evidence ledger, the co-occurrence window and
the calibration gate would all have to be re-implemented on it or silently
skipped. A rule enforced in one path and absent from the other is worse than a
rule enforced nowhere, because the two paths produce results that look alike.

So `analyze_file` opens a session, opens a processing run, streams windows
through `StreamingCoordinator` and completes the session - the same sequence
`adapters/inbound/websocket/handler.py` drives. What differs is the transport
and the adapters underneath, which is the only thing that should differ.

**Why it never imports bootstrap.** `bootstrap` is the composition root of the
*service*: it wires FastAPI, SQLAlchemy, Redis and an object store. Importing
it here would drag the `server` extra into every embedded install and undo
C9. The layering contract enforces this rather than trusting the comment:
`evidence_engine.sdk` sits below `evidence_engine.bootstrap` in C1, so the
import fails CI, not review.

**Async-first, and there is no sync wrapper yet.** The ports are async because
the streaming path is; a synchronous facade would either block an event loop or
own one, and picking which is a decision for whoever has a real use for it.
"""

from __future__ import annotations

import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from evidence_engine.adapters.outbound.cache.in_memory import (
    InMemoryQuotaGuard,
    InMemoryStreamState,
)
from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    DeterministicVisionRuntime,
    VisualScript,
)
from evidence_engine.adapters.outbound.object_storage.in_memory import InMemoryMediaStore
from evidence_engine.adapters.outbound.persistence.configuration import (
    InMemoryConfigurationStore,
    InMemoryModelRegistry,
)
from evidence_engine.adapters.outbound.persistence.in_memory import (
    InMemoryAuditLog,
    InMemoryEvidenceRepository,
    InMemoryRunRepository,
    InMemorySessionRepository,
)
from evidence_engine.adapters.outbound.persistence.tokens import HmacStreamTokenMinter
from evidence_engine.adapters.outbound.telemetry.clock import SystemClock
from evidence_engine.application.commands.capture_control import CaptureControl
from evidence_engine.application.commands.complete_session import CompleteSession
from evidence_engine.application.commands.create_session import (
    CreateSession,
    CreateSessionCommand,
)
from evidence_engine.application.commands.open_run import (
    CloseProcessingRun,
    OpenProcessingRun,
)
from evidence_engine.application.ports.clock import Clock
from evidence_engine.application.ports.platform import (
    AuthenticatedCaller,
    ConfigurationSnapshot,
    Scope,
)
from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    SpeechRuntime,
    VisionRuntime,
)
from evidence_engine.application.services.calibration import Calibrator
from evidence_engine.application.services.speech_assembly import SpeechAssembler
from evidence_engine.application.services.visual_assembly import VisualAssembler
from evidence_engine.application.workflows.streaming import (
    StreamingCoordinator,
    StreamingState,
)
from evidence_engine.domain.evidence.document import EvidenceDocument
from evidence_engine.domain.evidence.ledger import EvidenceLedger
from evidence_engine.domain.sessions.capabilities import CapabilityRequest
from evidence_engine.domain.sessions.consent import RetentionPolicy
from evidence_engine.domain.sessions.state import SessionMode
from evidence_engine.domain.shared.identifiers import (
    ApiKeyId,
    ApplicationId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.sdk._local import CollectedEvents, SilentTelemetry
from evidence_engine.sdk.configuration import EngineConfiguration, SessionConfiguration
from evidence_engine.sdk.errors import AudioNotUsable, LocalInferenceUnavailable
from evidence_engine.sdk.preflight import HardwareReport, hardware_preflight
from evidence_engine.sdk.stream import StreamSession

#: The caller an embedded engine runs as.
#:
#: There is no API key here and there must not be: authentication answers "may
#: this request reach the engine", and an embedded engine is already inside the
#: consumer's process - a credential would be the process authenticating to
#: itself. The scopes are full because the process boundary is the authorization
#: boundary; the tenant is fixed and named `embedded` so that a document
#: produced locally is identifiable as such in an audit trail that also holds
#: hosted runs.
EMBEDDED_TENANT = TenantId("embedded")
EMBEDDED_APPLICATION = ApplicationId("oratoria-engine-sdk")


def _embedded_caller(trace_id: str) -> AuthenticatedCaller:
    return AuthenticatedCaller(
        application=EMBEDDED_APPLICATION,
        tenant=EMBEDDED_TENANT,
        key_id=ApiKeyId("embedded"),
        scopes=frozenset(Scope),
        trace_id=trace_id,
    )


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """What one completed analysis produced.

    The published contract, not an internal entity: `EvidenceDocument` is what
    `GET /v1/sessions/{id}/result` serialises, so a consumer reading this
    locally and a consumer reading the hosted API are reading the same thing.
    """

    document: EvidenceDocument
    #: How many times the engine asked the caller to slow down. Zero on
    #: `analyze_file`, which paces itself; meaningful on a live stream.
    backpressure_signals: int = 0

    @property
    def transcript(self) -> object:
        return self.document.transcript

    @property
    def speech_events(self) -> object:
        return self.document.speech_events

    @property
    def prosody(self) -> object:
        return self.document.prosody


class OratoriaEngine:
    """The engine, running in this process."""

    def __init__(
        self,
        configuration: EngineConfiguration,
        speech: SpeechRuntime,
        vision: VisionRuntime,
        clock: Clock | None = None,
    ) -> None:
        self._configuration = configuration
        self._speech = speech
        self._vision = vision
        self._clock = clock or SystemClock()
        self._warmed = False

        snapshot = configuration.snapshot()
        self._snapshot = snapshot
        self._sessions = InMemorySessionRepository()
        self._runs = InMemoryRunRepository()
        self._evidence = InMemoryEvidenceRepository()
        self._audit = InMemoryAuditLog()
        self._media = InMemoryMediaStore(self._clock)
        self._stream_state = InMemoryStreamState(self._clock)
        self._quota = InMemoryQuotaGuard(self._clock)
        self._store = InMemoryConfigurationStore(snapshot)
        self._registry = InMemoryModelRegistry()
        self._telemetry = configuration.telemetry or SilentTelemetry()
        self._calibrator = Calibrator()

        self._create_session = CreateSession(
            sessions=self._sessions,
            configuration=self._store,
            quota=self._quota,
            audit=self._audit,
            clock=self._clock,
            token_minter=HmacStreamTokenMinter(configuration.stream_signing_key),
        )
        self._capture = CaptureControl(sessions=self._sessions, clock=self._clock)
        self._open_run = OpenProcessingRun(
            sessions=self._sessions,
            runs=self._runs,
            clock=self._clock,
            pipeline_version=configuration.pipeline_version,
        )
        self._close_run = CloseProcessingRun(runs=self._runs, clock=self._clock)
        self._complete = CompleteSession(
            sessions=self._sessions,
            evidence=self._evidence,
            audit=self._audit,
            clock=self._clock,
        )

    # -- construction -----------------------------------------------------

    @classmethod
    def local(
        cls,
        configuration: EngineConfiguration | None = None,
        clock: Clock | None = None,
    ) -> OratoriaEngine:
        """Build an engine that runs entirely in this process.

        Constructing this does **not** load a model. `hardware_preflight()`
        reports what the machine has and `warmup()` loads the weights, and
        keeping construction cheap is what lets a consumer ask both questions
        before committing to a 3 GB download.
        """
        resolved = configuration or EngineConfiguration()
        speech = resolved.build_speech_runtime()
        vision = resolved.vision_runtime or DeterministicVisionRuntime(VisualScript())
        return cls(resolved, speech, vision, clock)

    # -- capability -------------------------------------------------------

    def hardware_preflight(self) -> HardwareReport:
        """What this machine has. Says nothing about whether the model runs."""
        return hardware_preflight(self._configuration.device)

    async def warmup(self) -> None:
        """Load the weights and decode a short window. This is the real check.

        Idempotent: calling it twice is a no-op, so a caller that warms up
        defensively before every analysis pays once.

        The window is 200 ms of silence at the engine's decode rate. Silence
        rather than a tone because the point is to exercise load, allocation
        and one decode pass - not to assert anything about what came back, which
        would be a measurement this is not entitled to make.
        """
        if self._warmed:
            return
        try:
            await self._speech.transcribe(_silent_window(self._configuration.sample_rate_hz))
        except Exception as error:
            raise LocalInferenceUnavailable(
                f"the model could not complete a warm-up decode: {error}. "
                "hardware_preflight() reports what the machine has; this is what "
                "reports whether the model runs on it."
            ) from error
        self._warmed = True

    # -- batch ------------------------------------------------------------

    async def analyze_file(
        self,
        path: str | Path,
        session: SessionConfiguration | None = None,
    ) -> AnalysisResult:
        """Run a recording end to end and return the evidence document.

        This is the batch path the hosted service does not have: `mode: batch`
        is accepted by the API and no worker implements it. Here it is the same
        pipeline the streaming path uses, driven by a loop over windows instead
        of by a socket, which is why it produces the same contract rather than
        a parallel one.
        """
        audio = _read_wav(Path(path), self._configuration.sample_rate_hz)
        async with self.create_stream(session) as stream:
            for offset in range(0, len(audio), self._configuration.window_bytes):
                await stream.send_audio(audio[offset : offset + self._configuration.window_bytes])
            return await stream.finish()

    # -- streaming --------------------------------------------------------

    def create_stream(self, session: SessionConfiguration | None = None) -> StreamSession:
        """Open a live session with explicit control.

        Returns a session object rather than an async generator. A generator
        can only be advanced; a presentation has to be paused when the speaker
        stops, resumed, finished, and abandoned when something goes wrong - and
        `pause` is not decoration here, because §6.1's session clock excludes
        paused stretches and a rate computed across one would be wrong.
        """
        return StreamSession(self, session or SessionConfiguration())

    # -- lifecycle --------------------------------------------------------

    async def aclose(self) -> None:
        """Release what the engine holds.

        In-memory today, so there is nothing to release and this is a no-op.
        Published anyway: a consumer writing `await engine.aclose()` should not
        have to change their code when a future adapter holds a file handle or
        a device context, and adding the call later would be a breaking change
        to a published surface.
        """
        return None

    async def __aenter__(self) -> OratoriaEngine:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    # -- internals the stream session drives ------------------------------

    async def _open(
        self, session: SessionConfiguration
    ) -> tuple[SessionId, StreamingCoordinator, CollectedEvents, AuthenticatedCaller]:
        caller = _embedded_caller(f"sdk_{uuid.uuid4().hex}")
        created = await self._create_session.execute(
            caller,
            CreateSessionCommand(
                mode=SessionMode.REALTIME,
                capabilities=CapabilityRequest(
                    audio_codec=session.audio_codec,
                    sample_rate_hz=self._configuration.sample_rate_hz,
                    locale=session.locale,
                ),
                consent_policy_version=SemanticVersion.parse(session.consent_policy_version),
                retention=RetentionPolicy(
                    policy_version=SemanticVersion.parse(session.consent_policy_version),
                    retain_raw_media=session.retain_raw_media,
                    raw_media_ttl_seconds=session.raw_media_ttl_seconds,
                ),
            ),
        )
        session_id = created.session.id

        await self._capture.begin(caller, session_id)
        run = await self._open_run.execute(caller, session_id)
        channel = CollectedEvents()

        coordinator = StreamingCoordinator(
            state=StreamingState(run_id=run.id, session_id=session_id),
            configuration=self._snapshot,
            speech=self._speech,
            vision=self._vision,
            channel=channel,
            telemetry=self._telemetry,
            ledger=EvidenceLedger(run_id=run.id),
            speech_assembler=SpeechAssembler(run.id, self._snapshot, self._calibrator),
            visual_assembler=VisualAssembler(run.id, self._snapshot, self._calibrator),
            max_queue_depth=self._configuration.max_queue_depth,
        )
        return session_id, coordinator, channel, caller

    async def _finish(
        self,
        caller: AuthenticatedCaller,
        session_id: SessionId,
        coordinator: StreamingCoordinator,
        channel: CollectedEvents,
    ) -> AnalysisResult:
        completed = await self._complete.execute(
            caller, session_id, coordinator.state, self._snapshot
        )
        await self._close_run.execute(coordinator.state.run_id)
        return AnalysisResult(
            document=completed.document,
            backpressure_signals=channel.backpressure_signals,
        )

    async def _abort(self, coordinator: StreamingCoordinator) -> None:
        await self._close_run.execute(coordinator.state.run_id, succeeded=False)

    @property
    def configuration(self) -> ConfigurationSnapshot:
        """The snapshot every result from this engine is bound to (US-008)."""
        return self._snapshot


def _silent_window(sample_rate_hz: int) -> AudioWindow:
    duration_ms = 200
    frames = sample_rate_hz * duration_ms // 1000
    return AudioWindow(
        session_position_ms=0,
        duration_ms=duration_ms,
        sample_rate_hz=sample_rate_hz,
        samples=b"\x00\x00" * frames,
    )


def _read_wav(path: Path, expected_rate_hz: int) -> bytes:
    """Read a recording, refusing the shapes the engine cannot time correctly.

    Refused rather than resampled. A silent conversion is how a recording ends
    up on a clock nobody chose: the declared rate is what turns a byte count
    into a duration, so audio decoded at one rate and timed at another scales
    every boundary by the ratio between them - and the transcript still reads
    correctly.
    """
    if not path.exists():
        raise AudioNotUsable(f"{path} does not exist")

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        samples = handle.readframes(handle.getnframes())

    if channels != 1:
        raise AudioNotUsable(
            f"{path.name} has {channels} channels and the engine decodes mono. "
            "Downmixing here would average two microphones into one signal."
        )
    if width != 2:
        raise AudioNotUsable(f"{path.name} is {width * 8}-bit; the engine decodes 16-bit PCM")
    if rate != expected_rate_hz:
        raise AudioNotUsable(
            f"{path.name} is {rate} Hz and this engine is configured for "
            f"{expected_rate_hz} Hz. Resample it deliberately, with a tool that "
            "records what it did, rather than having it happen here."
        )
    return samples
