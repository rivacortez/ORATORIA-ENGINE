"""The driving port: what a transport is allowed to call.

This module exists because import-linter caught a real problem. The REST and
WebSocket adapters needed the assembled use cases, and the obvious way to get
them - import the composition root's ``Container`` - made every inbound adapter
transitively depend on every outbound one, through ``bootstrap``. Contract C6
broke, and it was right to: an inbound adapter that can see a concrete
PostgreSQL repository can eventually call it, and the application layer's
control over transactions, quotas and the state machine becomes advisory.

So the transports depend on this Protocol instead. ``bootstrap.Container``
satisfies it structurally without either side importing the other, which
restores the dependency rule and, incidentally, makes the transports testable
against a hand-built stub rather than a fully wired container.

The surface is deliberately narrow: use cases, the ports a transport genuinely
needs to construct a per-connection pipeline, and a profile of the few settings
that change transport behaviour. Anything else a transport wants belongs behind
a use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evidence_engine.application.commands.administer_keys import AdministerApiKeys
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
    ConfigurationStore,
    ModelRegistry,
    Telemetry,
)
from evidence_engine.application.ports.runtimes import SpeechRuntime, VisionRuntime
from evidence_engine.application.ports.streaming import StreamState
from evidence_engine.application.ports.tokens import StreamTokenMinter
from evidence_engine.application.queries.read_session import (
    ReadCapabilities,
    ReadResult,
    ReadSession,
)
from evidence_engine.application.services.calibration import Calibrator


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """The settings that change how a transport behaves.

    A projection of the deployment's configuration rather than the settings
    object itself, so the transports do not acquire a dependency on pydantic or
    on the environment-variable names. ``backend`` and ``runtime_mode`` are
    strings for the same reason: the readiness endpoint reports them, and it
    should not need the enum that defines which ones exist.
    """

    backend: str
    runtime_mode: str
    max_queue_depth: int
    stream_lease_ttl_seconds: int
    #: This process's identity, published on `/v1/capabilities` and on every
    #: `session.accepted`. A pilot running the engine on more than one GPU
    #: workstation needs to attribute a result to the machine that produced
    #: it - `instance_id` names it, `hostname` is the operating-system answer
    #: regardless of what `instance_id` was configured to.
    instance_id: str
    hostname: str


class EngineApi(Protocol):
    """Everything a transport may reach.

    Note what is absent: the session, evidence and audit repositories, the
    media store, the quota guard. A transport has no business touching those
    directly - every legitimate use of them runs through a use case that also
    applies scope checks, consent and the state machine.
    """

    profile: RuntimeProfile
    #: Seconds the startup warm-up decode took, or `None` before it has
    #: completed. `/health/ready` refuses traffic while this is `None`: a
    #: wired recogniser that has never actually decoded is not yet evidence
    #: that inference works here, by the same reasoning `sdk.engine.warmup()`
    #: is a separate method from `hardware_preflight()` (ADR-011).
    speech_warm_seconds: float | None

    # Ports a transport needs to build a per-connection pipeline or to
    # authenticate before any use case is reachable.
    clock: Clock
    telemetry: Telemetry
    tokens: StreamTokenMinter
    api_keys: ApiKeyDirectory
    stream_state: StreamState
    configuration: ConfigurationStore
    registry: ModelRegistry
    speech: SpeechRuntime
    vision: VisionRuntime
    calibrator: Calibrator

    # Use cases.
    create_session: CreateSession
    capture_control: CaptureControl
    open_run: OpenProcessingRun
    close_run: CloseProcessingRun
    complete_session: CompleteSession
    delete_evidence: DeleteEvidence
    read_session: ReadSession
    read_result: ReadResult
    read_capabilities: ReadCapabilities
    #: Provisioning. Reachable only with `Scope.ADMIN`, which the use case
    #: checks - the transport does not, deliberately, so the rule holds on
    #: every entry point rather than on the one that remembered it.
    administer_keys: AdministerApiKeys
