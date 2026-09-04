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

from dataclasses import dataclass

from evidence_engine.adapters.outbound.cache.in_memory import (
    InMemoryQuotaGuard,
    InMemoryStreamState,
)
from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    DeterministicSpeechRuntime,
    DeterministicVisionRuntime,
    SpeechScript,
    VisualScript,
)
from evidence_engine.adapters.outbound.object_storage.in_memory import InMemoryMediaStore
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
from evidence_engine.adapters.outbound.persistence.tokens import HmacStreamTokenMinter
from evidence_engine.adapters.outbound.telemetry.clock import SystemClock
from evidence_engine.adapters.outbound.telemetry.structured import StructlogTelemetry
from evidence_engine.application.api import RuntimeProfile
from evidence_engine.application.commands.capture_control import CaptureControl
from evidence_engine.application.commands.complete_session import CompleteSession
from evidence_engine.application.commands.create_session import CreateSession
from evidence_engine.application.commands.delete_evidence import DeleteEvidence
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
from evidence_engine.domain.shared.provenance import Modality, SemanticVersion
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
    complete_session: CompleteSession
    delete_evidence: DeleteEvidence
    read_session: ReadSession
    read_result: ReadResult
    read_capabilities: ReadCapabilities


def default_configuration() -> ConfigurationSnapshot:
    """The snapshot a fresh deployment starts from.

    Every threshold here is a *starting point*, not a calibrated value. §10 is
    explicit that its numbers are engineering acceptance targets to be
    calibrated against reference hardware and the annotated corpus before being
    declared achieved, and Phase 1 has not frozen a held-out set yet. The empty
    threshold maps say so honestly: with no fitted curve, ``Confidence.meets``
    refuses every gate, so nothing is published as confirmed on the strength of
    a number nobody measured.
    """
    snapshot_id = ConfigurationSnapshotId("config-default-v1")
    return ConfigurationSnapshot(
        id=snapshot_id,
        taxonomy_version=TAXONOMY_VERSION,
        pipeline_version=PIPELINE_VERSION,
        schema_version=SCHEMA_VERSION,
        fusion_window=FusionWindow(width_ms=DEFAULT_FUSION_WINDOW_MS, configuration=snapshot_id),
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

    if settings.backend is Backend.POSTGRES:
        raise NotImplementedError(
            "the PostgreSQL backend lands with the persistence adapters in phase 2's "
            "second half; set ENGINE_BACKEND=memory to run the contract suite today"
        )

    resolved_clock: Clock = clock or SystemClock()
    telemetry = StructlogTelemetry()

    sessions = InMemorySessionRepository()
    runs = InMemoryRunRepository()
    evidence = InMemoryEvidenceRepository()
    audit = InMemoryAuditLog()
    media = InMemoryMediaStore(resolved_clock)
    stream_state = InMemoryStreamState(resolved_clock)
    quota = InMemoryQuotaGuard(
        resolved_clock,
        limits=(
            {"sessions": settings.sessions_per_minute} if settings.sessions_per_minute > 0 else {}
        ),
    )

    snapshot = default_configuration()
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
    )


def _build_runtimes(
    settings: Settings,
    registry: InMemoryModelRegistry,
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
