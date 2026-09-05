"""The composition root.

The only place that knows every layer at once. Contract C1 puts ``bootstrap``
above ``adapters``, which is what lets this module import both a port and its
implementation without making that a general licence: nothing else in the
codebase may do it.

Wiring is explicit rather than reflective. A container that discovers adapters
by scanning is shorter to write and much worse to operate - a missing binding
shows up as an ``AttributeError`` on the first request in production rather
than as a name error at import time, and "which adapter is actually serving
this port?" stops having an answer you can read.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial

from redis.asyncio import Redis

from evidence_engine.adapters.inbound.rest.serialization import render_document
from evidence_engine.adapters.outbound.cache.in_memory import (
    InMemoryQuotaGuard,
    InMemoryStreamState,
)
from evidence_engine.adapters.outbound.cache.redis_state import (
    RedisQuotaGuard,
    RedisStreamState,
)
from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    DeterministicSpeechRuntime,
    DeterministicVisionRuntime,
    SpeechScript,
    VisualScript,
)
from evidence_engine.adapters.outbound.object_storage.in_memory import InMemoryMediaStore
from evidence_engine.adapters.outbound.object_storage.s3 import S3MediaStore
from evidence_engine.adapters.outbound.persistence.configuration import (
    InMemoryConfigurationStore,
    InMemoryModelRegistry,
)
from evidence_engine.adapters.outbound.persistence.identity import InMemoryApiKeyDirectory
from evidence_engine.adapters.outbound.persistence.in_memory import (
    InMemoryAuditLog,
    InMemoryEvidenceRepository,
    InMemoryRunRepository,
    InMemorySessionRepository,
)
from evidence_engine.adapters.outbound.persistence.postgres.control_plane import (
    PostgresApiKeyDirectory,
    PostgresConfigurationStore,
    PostgresModelRegistry,
)
from evidence_engine.adapters.outbound.persistence.postgres.engine import (
    create_engine,
    create_session_factory,
)
from evidence_engine.adapters.outbound.persistence.postgres.evidence_repository import (
    PostgresEvidenceRepository,
)
from evidence_engine.adapters.outbound.persistence.postgres.repositories import (
    PostgresAuditLog,
    PostgresRunRepository,
    PostgresSessionRepository,
)
from evidence_engine.adapters.outbound.persistence.tokens import HmacStreamTokenMinter
from evidence_engine.adapters.outbound.telemetry.clock import SystemClock
from evidence_engine.adapters.outbound.telemetry.structured import StructlogTelemetry
from evidence_engine.application.api import RuntimeProfile
from evidence_engine.application.commands.capture_control import CaptureControl
from evidence_engine.application.commands.complete_session import CompleteSession
from evidence_engine.application.commands.create_session import CreateSession
from evidence_engine.application.commands.delete_evidence import DeleteEvidence
from evidence_engine.application.commands.open_run import (
    CloseProcessingRun,
    OpenProcessingRun,
)
from evidence_engine.application.ports.clock import Clock
from evidence_engine.application.ports.platform import (
    ApiKeyDirectory,
    ApprovalState,
    ConfigurationSnapshot,
    ConfigurationStore,
    ModelRegistry,
    ModelVersion,
    QuotaGuard,
    Telemetry,
)
from evidence_engine.application.ports.repositories import (
    AuditLog,
    EvidenceRepository,
    RunRepository,
    SessionRepository,
)
from evidence_engine.application.ports.runtimes import SpeechRuntime, VisionRuntime
from evidence_engine.application.ports.storage import MediaStore
from evidence_engine.application.ports.streaming import StreamState
from evidence_engine.application.ports.tokens import StreamTokenMinter
from evidence_engine.application.queries.read_session import (
    ReadCapabilities,
    ReadResult,
    ReadSession,
)
from evidence_engine.application.services.calibration import Calibrator
from evidence_engine.bootstrap.settings import Backend, RuntimeMode, Settings
from evidence_engine.domain.evidence.cooccurrence import (
    DEFAULT_FUSION_WINDOW_MS,
    FusionWindow,
)
from evidence_engine.domain.shared.identifiers import ConfigurationSnapshotId, ModelVersionId
from evidence_engine.domain.shared.provenance import (
    Modality,
    SemanticVersion,
    Unseeded,
    UnseededReason,
)
from evidence_engine.domain.shared.taxonomy import TAXONOMY_VERSION

#: The published contract version this build speaks. Bumped by hand, because
#: NFR-017's compatibility promise is a decision, not a build artifact.
SCHEMA_VERSION = SemanticVersion(1, 0, 0)
PIPELINE_VERSION = SemanticVersion(0, 1, 0)


@dataclass(slots=True)
class Container:
    """Every wired dependency, and the use cases assembled from them."""

    settings: Settings
    #: The slice of `settings` the transports are allowed to see. Keeping the
    #: projection here rather than letting an adapter read `settings` directly
    #: is what lets `Container` satisfy `EngineApi` without the transports
    #: acquiring a dependency on pydantic or on environment-variable names.
    profile: RuntimeProfile

    # Ports
    clock: Clock
    telemetry: Telemetry
    sessions: SessionRepository
    runs: RunRepository
    evidence: EvidenceRepository
    audit: AuditLog
    media: MediaStore
    stream_state: StreamState
    quota: QuotaGuard
    configuration: ConfigurationStore
    registry: ModelRegistry
    api_keys: ApiKeyDirectory
    tokens: StreamTokenMinter
    speech: SpeechRuntime
    vision: VisionRuntime
    calibrator: Calibrator

    # Use cases
    create_session: CreateSession
    capture_control: CaptureControl
    open_run: OpenProcessingRun
    close_run: CloseProcessingRun
    complete_session: CompleteSession
    delete_evidence: DeleteEvidence
    read_session: ReadSession
    read_result: ReadResult
    read_capabilities: ReadCapabilities

    #: Async callables that release the infrastructure this container holds -
    #: the database connection pool, the Redis client. Empty for the memory
    #: backend, which holds nothing.
    closers: tuple[Callable[[], Awaitable[None]], ...] = field(default_factory=tuple)

    async def aclose(self) -> None:
        """Release every held resource.

        Called from the ASGI lifespan. Without it a process that outlives one
        container - a test harness, a worker that rebuilds its wiring - leaks a
        connection pool per container, and the leak surfaces as PostgreSQL
        refusing new connections long after the code that caused it ran.
        """
        for close in self.closers:
            await close()


def default_configuration() -> ConfigurationSnapshot:
    """The snapshot a fresh deployment starts from.

    Every threshold here is a *starting point*, not a calibrated value. §10 is
    explicit that its numbers are engineering acceptance targets to be
    calibrated against reference hardware and the annotated corpus before being
    declared achieved, and Phase 1 has not frozen a held-out set yet. The empty
    threshold maps say so honestly: with no fitted curve, ``Confidence.meets``
    refuses every gate, so nothing is published as confirmed on the strength of
    a number nobody measured.

    The seed says ``DETERMINISTIC_RUNTIME`` because that is checkable today:
    both shipped runtimes replay a script, so the same session replayed twice
    produces byte-identical evidence and there is nothing to seed. The moment
    a runtime that makes a random choice is wired in, this line becomes false
    and has to change with it - which is the point of stating the claim here
    rather than leaving the field empty and letting a reader assume either
    answer.
    """
    snapshot_id = ConfigurationSnapshotId("config-default-v1")
    return ConfigurationSnapshot(
        id=snapshot_id,
        taxonomy_version=TAXONOMY_VERSION,
        pipeline_version=PIPELINE_VERSION,
        schema_version=SCHEMA_VERSION,
        fusion_window=FusionWindow(width_ms=DEFAULT_FUSION_WINDOW_MS, configuration=snapshot_id),
        seed=Unseeded(UnseededReason.DETERMINISTIC_RUNTIME),
        speech_thresholds={},
        visual_thresholds={},
        silence_threshold_ms=700,
        calibration=None,
    )


def build_container(
    settings: Settings,
    *,
    clock: Clock | None = None,
    speech_script: SpeechScript | None = None,
    visual_script: VisualScript | None = None,
) -> Container:
    """Wire everything. Raises rather than degrading when a backend is missing."""
    settings.require_infrastructure()

    resolved_clock: Clock = clock or SystemClock()
    telemetry = StructlogTelemetry()
    snapshot = default_configuration()
    quota_limits = (
        {"sessions": settings.sessions_per_minute} if settings.sessions_per_minute > 0 else {}
    )

    sessions: SessionRepository
    runs: RunRepository
    evidence: EvidenceRepository
    audit: AuditLog
    media: MediaStore
    stream_state: StreamState
    quota: QuotaGuard
    configuration: ConfigurationStore
    registry: ModelRegistry
    api_keys: ApiKeyDirectory

    closers: tuple[Callable[[], Awaitable[None]], ...] = ()

    if settings.backend is Backend.POSTGRES:
        database_engine = create_engine(settings.database_url)
        factory = create_session_factory(database_engine)
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        closers = (database_engine.dispose, redis_client.aclose)

        sessions = PostgresSessionRepository(factory)
        runs = PostgresRunRepository(factory)
        # `render_document` comes from the inbound REST adapter. The composition
        # root is the one place allowed to know both (contract C1), which is
        # what lets the published shape stay defined once without an outbound
        # adapter importing an inbound one. The schema version is bound here so
        # the repository stores exactly the bytes the API would have served.
        evidence = PostgresEvidenceRepository(
            factory, partial(render_document, schema_version=str(SCHEMA_VERSION))
        )
        audit = PostgresAuditLog(factory)
        media = S3MediaStore(
            bucket=settings.object_storage_bucket,
            endpoint_url=settings.object_storage_endpoint,
            clock_epoch_ms=resolved_clock.epoch_ms,
            access_key=settings.object_storage_access_key,
            secret_key=settings.object_storage_secret_key,
        )
        stream_state = RedisStreamState(redis_client)
        quota = RedisQuotaGuard(redis_client, limits=quota_limits)
        configuration = PostgresConfigurationStore(factory, snapshot)
        registry = PostgresModelRegistry(factory)
        api_keys = PostgresApiKeyDirectory(
            factory, settings.api_key_pepper, resolved_clock.epoch_ms
        )
    else:
        sessions = InMemorySessionRepository()
        runs = InMemoryRunRepository()
        evidence = InMemoryEvidenceRepository()
        audit = InMemoryAuditLog()
        media = InMemoryMediaStore(resolved_clock)
        stream_state = InMemoryStreamState(resolved_clock)
        quota = InMemoryQuotaGuard(resolved_clock, limits=quota_limits)
        configuration = InMemoryConfigurationStore(snapshot)
        registry = InMemoryModelRegistry()
        api_keys = InMemoryApiKeyDirectory(settings.api_key_pepper, resolved_clock)

    tokens = HmacStreamTokenMinter(settings.stream_token_signing_key)

    speech, vision = _build_runtimes(settings, registry, speech_script, visual_script)

    # An empty calibrator, deliberately. §14.2 requires calibration to be
    # reported before a model version is promoted, and none has been fitted
    # yet - so every confidence comes back RAW and no threshold gate opens.
    calibrator = Calibrator()

    return Container(
        settings=settings,
        profile=RuntimeProfile(
            backend=settings.backend.value,
            runtime_mode=settings.runtime_mode.value,
            max_queue_depth=settings.max_queue_depth,
            stream_lease_ttl_seconds=settings.stream_lease_ttl_seconds,
        ),
        clock=resolved_clock,
        telemetry=telemetry,
        sessions=sessions,
        runs=runs,
        evidence=evidence,
        audit=audit,
        media=media,
        stream_state=stream_state,
        quota=quota,
        configuration=configuration,
        registry=registry,
        api_keys=api_keys,
        tokens=tokens,
        speech=speech,
        vision=vision,
        calibrator=calibrator,
        create_session=CreateSession(
            sessions=sessions,
            configuration=configuration,
            quota=quota,
            audit=audit,
            clock=resolved_clock,
            token_minter=tokens,
        ),
        capture_control=CaptureControl(sessions=sessions, clock=resolved_clock),
        open_run=OpenProcessingRun(
            sessions=sessions,
            runs=runs,
            clock=resolved_clock,
            pipeline_version=PIPELINE_VERSION,
        ),
        close_run=CloseProcessingRun(runs=runs, clock=resolved_clock),
        complete_session=CompleteSession(
            sessions=sessions, evidence=evidence, audit=audit, clock=resolved_clock
        ),
        delete_evidence=DeleteEvidence(
            sessions=sessions,
            evidence=evidence,
            media=media,
            stream_state=stream_state,
            audit=audit,
            clock=resolved_clock,
        ),
        read_session=ReadSession(sessions=sessions),
        read_result=ReadResult(sessions=sessions, evidence=evidence),
        read_capabilities=ReadCapabilities(SCHEMA_VERSION),
        closers=closers,
    )


def _build_runtimes(
    settings: Settings,
    registry: ModelRegistry,
    speech_script: SpeechScript | None,
    visual_script: VisualScript | None,
) -> tuple[SpeechRuntime, VisionRuntime]:
    """Select the model runtimes and register the versions they will report."""
    if settings.runtime_mode is RuntimeMode.MANAGED:
        raise NotImplementedError(
            "managed ASR and vision runtimes land in phases 3 and 5; the "
            "deterministic runtimes are what phase 2's exit criterion is defined on"
        )

    speech = DeterministicSpeechRuntime(speech_script or SpeechScript())
    vision = DeterministicVisionRuntime(visual_script or VisualScript())

    # Registered so that NFR-014's provenance resolves even in this mode. A
    # deterministic runtime is still a version that produced evidence, and a
    # result that could not name it would be untraceable in exactly the runs
    # that are supposed to be the most reproducible.
    if not isinstance(registry, InMemoryModelRegistry):
        # The persistent registry is seeded by a migration or an administrative
        # call, not by process startup. Registering on boot would let a replica
        # silently reintroduce a version an administrator had just disabled.
        return speech, vision

    for modality, model_id in (
        (Modality.AUDIO, ModelVersionId("deterministic-speech-v1")),
        (Modality.VIDEO, ModelVersionId("deterministic-vision-v1")),
    ):
        registry.register(
            ModelVersion(
                id=model_id,
                modality=modality,
                artifact_digest=f"sha256:deterministic-{modality.value}",
                dataset_version="none",
                approval=ApprovalState.EVALUATED,
                metrics={},
            ),
            make_active=True,
        )

    return speech, vision
