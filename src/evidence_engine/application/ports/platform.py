"""Control-plane ports: identity, quota, configuration, model registry, telemetry.

§11.3 iteration 6 separates the control plane from the inference data plane, and
these are the control plane's edges. They share a property worth stating: none
of them ever sees transcript content or raw media. US-009 requires an
administrator to monitor latency, failures, resources and drift *without*
reading student transcripts, and NFR-018 requires operational logs to carry no
transcript content at all. A port that accepted the text would make both
promises depend on nobody passing it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from evidence_engine.domain.evidence.cooccurrence import FusionWindow
from evidence_engine.domain.shared.identifiers import (
    ApiKeyId,
    ApplicationId,
    ConfigurationSnapshotId,
    ModelVersionId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import Modality, Seed, SemanticVersion
from evidence_engine.domain.visual_events.calibration import VisualCalibration

# ---------------------------------------------------------------------------
# Identity and authorization (§9.1)
# ---------------------------------------------------------------------------


class Scope(StrEnum):
    """What a key is allowed to do (FR-003).

    Deletion is its own scope rather than part of a write scope. FR-032 and
    US-014 make evidence deletion irreversible, and a capture credential leaked
    into a client bundle should not also be able to erase a study's data.
    """

    SESSIONS_WRITE = "sessions:write"
    SESSIONS_READ = "sessions:read"
    RESULTS_READ = "results:read"
    EVIDENCE_DELETE = "evidence:delete"
    JOBS_WRITE = "jobs:write"
    ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class AuthenticatedCaller:
    """The resolved identity behind a request.

    Every request and connection is attributable to an application and a trace
    id (FR-004), so both travel together from the moment authentication
    succeeds - an audit record assembled later from ambient state would be
    guessing.
    """

    application: ApplicationId
    tenant: TenantId
    key_id: ApiKeyId
    scopes: frozenset[Scope]
    trace_id: str

    def allows(self, scope: Scope) -> bool:
        return scope in self.scopes or Scope.ADMIN in self.scopes


class ApiKeyDirectory(Protocol):
    """Resolves a presented secret to a caller, or refuses.

    The port takes the raw secret and never returns it. FR-002 and NFR-010
    require only a cryptographic hash to be persisted and the raw value never
    to appear in logs or dumps, so hashing happens inside the adapter and the
    plaintext never becomes a field on anything the application holds.
    """

    async def authenticate(
        self, presented_secret: str, trace_id: str
    ) -> AuthenticatedCaller | None:
        """Resolve a caller, or ``None`` for unknown, revoked or expired keys.

        One return value for all three failures on purpose: distinguishing
        "revoked" from "never existed" tells a probing caller which keys once
        worked.
        """
        ...


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    """Whether a call may proceed, and what to tell the caller if not."""

    allowed: bool
    remaining: int
    retry_after_seconds: int = 0
    reason: str = ""


class QuotaGuard(Protocol):
    """Per-application limits (FR-003, NFR-023).

    Cost control is the point, not just abuse prevention: NFR-023 requires
    per-minute inference cost to be measurable by modality and model version,
    with configurable quotas preventing unbounded consumption.
    """

    async def check(
        self, application: ApplicationId, unit: str, amount: int = 1
    ) -> QuotaDecision: ...

    async def consume(self, application: ApplicationId, unit: str, amount: int = 1) -> None: ...


# ---------------------------------------------------------------------------
# Configuration (FR-015, US-008)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    """§8 ``ConfigurationSnapshot``: the frozen rules one session ran under.

    US-008: published versions are immutable and a session stores its complete
    snapshot, so updating a default never alters a historical result. The whole
    object is captured at session creation rather than read per window - a
    threshold changed mid-session would make the first half and the second half
    of one presentation incomparable.

    ``seed`` has no default, unlike every other tuning knob here. "Complete
    snapshot" is the whole of US-008's promise, and a defaulted seed would let
    a snapshot that never stated its seed policy be indistinguishable from one
    that deliberately chose the same value. NFR-015 names the seed alongside
    the input, the artifact and the configuration; the other three cannot be
    omitted either.
    """

    id: ConfigurationSnapshotId
    taxonomy_version: SemanticVersion
    pipeline_version: SemanticVersion
    schema_version: SemanticVersion
    fusion_window: FusionWindow
    #: The seed the run is configured to use, or the reason there is none. Not
    #: the same claim as ``Provenance.seed``, which is what a runtime actually
    #: consumed: a deterministic runtime handed a configured seed ignores it,
    #: and reporting the configured value as provenance would describe a
    #: reproduction path that was never taken.
    seed: Seed
    #: Publication threshold per speech class, keyed by the class identifier.
    speech_thresholds: Mapping[str, float] = field(default_factory=dict)
    #: Publication threshold per visual class.
    visual_thresholds: Mapping[str, float] = field(default_factory=dict)
    #: Silence above this counts as a silent pause (FR-015).
    silence_threshold_ms: int = 700
    #: Per-speaker visual reference, absent when preflight was skipped.
    calibration: VisualCalibration | None = None


class ConfigurationStore(Protocol):
    """Immutable, versioned configuration snapshots."""

    async def current(self, tenant: TenantId) -> ConfigurationSnapshot:
        """The snapshot a new session should freeze."""
        ...

    async def get(self, snapshot_id: ConfigurationSnapshotId) -> ConfigurationSnapshot | None:
        """The snapshot a historical run was produced under."""
        ...

    async def freeze(self, snapshot: ConfigurationSnapshot, session_id: SessionId) -> None:
        """Bind a snapshot to a session so the run can be reproduced (QA-05)."""
        ...


# ---------------------------------------------------------------------------
# Model registry (FR-033, QA-03, §14.2)
# ---------------------------------------------------------------------------


class ApprovalState(StrEnum):
    """§14.2 gates promotion behind evaluation, so state is explicit."""

    DRAFT = "draft"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """§8 ``ModelVersion``: an artifact and everything needed to trust it."""

    id: ModelVersionId
    modality: Modality
    artifact_digest: str
    dataset_version: str
    approval: ApprovalState
    #: Reported metrics. Empty for a draft; §14.2 requires WER, vWER, per-class
    #: precision/recall/F1, timestamp error and calibration before promotion.
    metrics: Mapping[str, float] = field(default_factory=dict)
    #: The version to fall back to. QA-03 measures rollback in under ten
    #: minutes, which is only possible if the target was decided in advance.
    rollback_to: ModelVersionId | None = None


class ModelRegistry(Protocol):
    """Lineage, approval and traffic routing for model artifacts."""

    async def active_for(self, modality: Modality) -> ModelVersion:
        """The version that should serve the next request for this modality."""
        ...

    async def get(self, model_id: ModelVersionId) -> ModelVersion | None: ...

    async def promote(self, model_id: ModelVersionId, canary_percent: int) -> ModelVersion:
        """Route a share of traffic to a new version (US-007).

        Refuses anything that is not ``EVALUATED`` or ``APPROVED``: §5 says the
        registry must not allow unapproved models in production, and US-007's
        criterion is that only evaluated artifacts can be promoted.
        """
        ...

    async def rollback(self, modality: Modality) -> ModelVersion:
        """Return to the recorded fallback without changing any schema."""
        ...

    async def list_versions(self, modality: Modality) -> Sequence[ModelVersion]: ...


# ---------------------------------------------------------------------------
# Telemetry (NFR-018)
# ---------------------------------------------------------------------------


class Telemetry(Protocol):
    """Traces, metrics and structured logs, correlated by trace id.

    No method accepts transcript text or media. NFR-018 requires operational
    logs to carry none, and the cheapest way to keep that true is to give the
    port nowhere to put it.
    """

    def counter(self, name: str, value: int = 1, **labels: str) -> None: ...

    def histogram(self, name: str, value: float, **labels: str) -> None: ...

    def event(self, name: str, trace_id: str, **fields: str | int | float | bool) -> None:
        """A structured operational log line.

        ``fields`` takes scalars only. A ``dict`` or an arbitrary object is how
        a payload containing a transcript ends up in a log aggregator.
        """
        ...
