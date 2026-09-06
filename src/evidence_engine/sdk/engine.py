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
from evidence_engine.sdk.errors import (
    AudioNotUsable,
    EngineNotWarmed,
    LocalInferenceUnavailable,
)
from evidence_engine.sdk.preflight import HardwareReport, hardware_preflight
from evidence_engine.sdk.results import (
    Evidence,
    ProsodyReading,
    SpeechEvent,
    Transcript,
    evidence_from,
)
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
    """What one completed analysis produced, in the SDK's own types.

    This used to carry an ``EvidenceDocument`` - a domain entity - under a
    docstring calling it "the published contract, not an internal entity". It
    was an internal entity, and the surface test passed on the letter while
    every consumer touching `result.document.transcript.tokens[0].placement`
    was coupled to the domain's shape. A leak through a field is still a leak.

    ``evidence`` is now the public shape, mirroring the wire rather than the
    domain, so the day `OratoriaClient` exists it returns these unchanged.
    """

    evidence: Evidence
    #: How many times the engine asked the caller to slow down. Zero on
    #: `analyze_file`, which paces itself; meaningful on a live stream.
    backpressure_signals: int = 0

    @property
    def transcript(self) -> Transcript:
        return self.evidence.transcript

    @property
    def speech_events(self) -> tuple[SpeechEvent, ...]:
        return self.evidence.speech_events

    @property
    def prosody(self) -> tuple[ProsodyReading, ...]:
        return self.evidence.prosody


class OratoriaEngine:
    """The engine, running in this process."""

    def __init__(
        self,
        configuration: EngineConfiguration,
        speech: SpeechRuntime | None,
        vision: VisionRuntime,
        clock: Clock | None = None,
    ) -> None:
        self._configuration = configuration
        #: `None` until `warmup()`. The speech runtime is built lazily because
        #: building the Whisper one downloads and loads 3 GB of weights, and
        #: doing that in the constructor made `hardware_preflight()` - the
        #: method whose entire purpose is to be asked *before* committing to
        #: that - unreachable until after it had happened.
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
        # `contributions` is completed at warmup, once the speech runtime
        # exists; until then the run would record only the vision model.
        self._open_run = OpenProcessingRun(
            sessions=self._sessions,
            runs=self._runs,
            clock=self._clock,
            pipeline_version=configuration.pipeline_version,
            contributions=dict(vision.contributions),
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

        Constructing this loads **no** model, and that is now true rather than
        merely written down: it used to call `build_speech_runtime()` here,
        which for the Whisper baseline downloads and loads 3 GB before
        returning - so `hardware_preflight()`, whose entire purpose is to be
        asked *before* committing to that, could only be reached afterwards.
        The runtime is built by `warmup()`.
        """
        resolved = configuration or EngineConfiguration()
        # No speech runtime yet. A caller supplying their own gets it wired
        # immediately - there is nothing to defer - but the configured one is
        # built by `warmup()`, so construction stays free.
        vision = resolved.vision_runtime or DeterministicVisionRuntime(VisualScript())
        return cls(resolved, resolved.speech_runtime, vision, clock)

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

        **Required before analysing anything.** `analyze_file` and
        `create_stream` refuse until this has run. Loading the model implicitly
        on first use would put a 3 GB download inside what a caller timed as a
        transcription, and would make `hardware_preflight()` advisory.
        """
        if self._warmed:
            return
        if self._speech is None:
            self._speech = self._configuration.build_speech_runtime()
        # The run records what is wired, by role, from the moment it opens -
        # and the speech runtime is only known now.
        self._open_run = OpenProcessingRun(
            sessions=self._sessions,
            runs=self._runs,
            clock=self._clock,
            pipeline_version=self._configuration.pipeline_version,
            contributions={**self._speech.contributions, **self._vision.contributions},
        )
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
        self._require_warm("analyze_file")
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
        self._require_warm("create_stream")
        return StreamSession(self, session or SessionConfiguration())

    # -- lifecycle --------------------------------------------------------

    async def aclose(self) -> None:
        """Release what the engine holds, including the device.

        This was a no-op with a docstring saying there was nothing to release.
        That was true of the in-memory adapters and false of the model: a
        warmed `baseline_whisper` engine holds a CUDA context and about 4.2 GiB
        of it, so a consumer who closed one and built another on an 8 GB card
        ran out of memory on a machine that should have fitted both.

        Dropping the reference is not enough on its own - CUDA caches freed
        blocks in its own allocator, so the memory stays reserved from the
        driver's point of view until the cache is emptied. Both steps run, in
        that order, and the torch import is guarded because an engine on the
        deterministic runtime has no torch to import.

        Idempotent, and safe to call on an engine that was never warmed.
        """
        self._speech = None
        self._warmed = False
        _release_device_memory()

    def __del__(self) -> None:  # pragma: no cover - a safety net, not a design
        # A consumer who forgot `aclose()` should not hold a GPU until the
        # process exits. This is a net rather than the mechanism: `__del__`
        # runs at a time nobody controls, so the explicit call remains the
        # supported way and this only shortens the leak.
        if getattr(self, "_speech", None) is not None:
            self._speech = None

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
        # `_require_warm` has already run in `create_stream`; this states it
        # for the type checker and would fire loudly rather than building a
        # coordinator around `None` if that guard were ever removed.
        assert self._speech is not None
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
            evidence=evidence_from(completed.document),
            backpressure_signals=channel.backpressure_signals,
        )

    async def _abort(self, coordinator: StreamingCoordinator) -> None:
        await self._close_run.execute(coordinator.state.run_id, succeeded=False)

    def _require_warm(self, doing: str) -> None:
        """Refuse to work with a model that has not been loaded and proved.

        Loading implicitly on first use would put a 3 GB download inside a call
        a consumer timed as a transcription, and would leave
        `hardware_preflight()` - which exists to be asked first - as advice
        nobody had to take.
        """
        if not self._warmed:
            raise EngineNotWarmed(
                f"{doing} needs a loaded model. Call `await engine.warmup()` first: it "
                "loads the weights and decodes a window, and it is the only thing that "
                "can say the model runs on this machine. `hardware_preflight()` reports "
                "what the hardware has and deliberately does not answer that."
            )

    @property
    def configuration(self) -> ConfigurationSnapshot:
        """The snapshot every result from this engine is bound to (US-008)."""
        return self._snapshot


def _release_device_memory() -> None:
    """Empty CUDA's cached allocator, if there is one to empty.

    Guarded on the import rather than on a flag: a deterministic engine never
    imported torch and must not start now, and an engine whose warmup failed
    part-way may still be holding an allocation.
    """
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():  # pragma: no cover - needs a device
        torch.cuda.empty_cache()


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
