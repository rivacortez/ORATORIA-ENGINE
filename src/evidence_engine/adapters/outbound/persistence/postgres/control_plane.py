"""Control-plane repositories: API keys, configuration snapshots, model registry.

§11.3 iteration 6 separates the control plane from the inference data plane.
These three tables are that plane's persistent state, and they share a property
worth stating: none of them ever holds evidence. An administrator reading any of
them learns about the system, never about a student.

The key lookup is by hash, indexed. The in-memory adapter walks every record
with a constant-time compare to make the property visible; here the index does
the finding and ``hmac.compare_digest`` still does the comparing, because the
value being compared is derived from attacker-supplied input either way.
"""

from __future__ import annotations

import hmac
from collections.abc import Sequence
from typing import Any, Protocol, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidence_engine.adapters.outbound.persistence.identity import hash_secret
from evidence_engine.adapters.outbound.persistence.postgres import models
from evidence_engine.adapters.outbound.persistence.postgres.engine import unit_of_work
from evidence_engine.adapters.outbound.persistence.postgres.mapping import (
    seed_from_columns,
    seed_to_columns,
)
from evidence_engine.application.ports.platform import (
    ApprovalState,
    AuthenticatedCaller,
    ConfigurationSnapshot,
    ModelVersion,
    Scope,
)
from evidence_engine.domain.evidence.cooccurrence import FusionWindow
from evidence_engine.domain.shared.identifiers import (
    ApiKeyId,
    ApplicationId,
    ConfigurationSnapshotId,
    ModelVersionId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import Modality, SemanticVersion
from evidence_engine.domain.visual_events.calibration import VisualCalibration


class ClockRead(Protocol):
    """A zero-argument epoch-millisecond read.

    Narrower than the ``Clock`` port on purpose: the key directory needs to
    know whether a key has expired and nothing else, and depending on the
    whole port would make it awkward to construct in a test that only cares
    about expiry.
    """

    def __call__(self) -> int: ...


class ControlPlaneError(Exception):
    """A control-plane rule was violated."""


class PostgresApiKeyDirectory:
    """Resolves presented secrets to callers (FR-002, NFR-010)."""

    def __init__(
        self, factory: async_sessionmaker[AsyncSession], pepper: str, clock_epoch_ms: ClockRead
    ) -> None:
        self._factory = factory
        self._pepper = pepper
        self._now = clock_epoch_ms

    async def authenticate(
        self, presented_secret: str, trace_id: str
    ) -> AuthenticatedCaller | None:
        candidate = hash_secret(presented_secret, self._pepper)
        async with self._factory() as db:
            row = await db.scalar(
                select(models.ApiKeyRow).where(models.ApiKeyRow.hashed_secret == candidate)
            )

        if row is None or not hmac.compare_digest(row.hashed_secret, candidate):
            return None

        now_ms = self._now()
        if row.revoked_at_ms is not None:
            return None
        if row.expires_at_ms is not None and now_ms >= row.expires_at_ms:
            return None

        return AuthenticatedCaller(
            application=ApplicationId(row.application_id),
            tenant=TenantId(row.tenant_id),
            key_id=ApiKeyId(row.id),
            scopes=frozenset(Scope(value) for value in row.scopes),
            trace_id=trace_id,
        )

    async def revoke(self, key_id: ApiKeyId, at_ms: int) -> bool:
        """US-006: revocation blocks new calls immediately."""
        async with unit_of_work(self._factory) as db:
            # `execute` is typed as returning `Result`; a DML statement
            # returns a `CursorResult`, which is the one that carries
            # `rowcount`. Without the cast the revocation would report failure
            # on every successful revoke.
            result = cast(
                CursorResult[None],
                await db.execute(
                    update(models.ApiKeyRow)
                    .where(models.ApiKeyRow.id == key_id.value)
                    .values(revoked_at_ms=at_ms)
                ),
            )
        return bool(result.rowcount)


class PostgresConfigurationStore:
    """Immutable, versioned snapshots bound to sessions (US-008)."""

    def __init__(
        self, factory: async_sessionmaker[AsyncSession], default: ConfigurationSnapshot
    ) -> None:
        self._factory = factory
        self._default = default

    async def current(self, tenant: TenantId) -> ConfigurationSnapshot:
        """The snapshot a new session should freeze.

        Falls back to the build's default when nothing has been published for
        this tenant. A tenant with no configuration is a new tenant, not an
        error - and refusing to start their first session over it would be an
        odd way to say so.
        """
        async with self._factory() as db:
            row = await db.scalar(
                select(models.ConfigurationSnapshotRow)
                .order_by(models.ConfigurationSnapshotRow.created_at.desc())
                .limit(1)
            )
        return _row_to_snapshot(row) if row is not None else self._default

    async def get(self, snapshot_id: ConfigurationSnapshotId) -> ConfigurationSnapshot | None:
        async with self._factory() as db:
            row = await db.get(models.ConfigurationSnapshotRow, snapshot_id.value)
        if row is not None:
            return _row_to_snapshot(row)
        return self._default if snapshot_id == self._default.id else None

    async def freeze(self, snapshot: ConfigurationSnapshot, session_id: SessionId) -> None:
        """Publish the snapshot if new, and refuse to redefine an existing one.

        US-008 makes published versions immutable. Silently accepting an
        overwrite would make every historical result unverifiable, because the
        id a run recorded would no longer resolve to what it ran under.
        """
        async with unit_of_work(self._factory) as db:
            existing = await db.get(models.ConfigurationSnapshotRow, snapshot.id.value)
            payload = _snapshot_to_json(snapshot)
            seed, seed_reason = seed_to_columns(snapshot.seed)
            if existing is None:
                db.add(
                    models.ConfigurationSnapshotRow(
                        id=snapshot.id.value,
                        taxonomy_version=str(snapshot.taxonomy_version),
                        pipeline_version=str(snapshot.pipeline_version),
                        schema_version=str(snapshot.schema_version),
                        payload=payload,
                        seed=seed,
                        seed_reason=seed_reason,
                    )
                )
            # The seed is compared alongside the payload rather than left out
            # of the immutability check. It lives in its own columns, so a
            # payload comparison on its own would let one published id mean two
            # different reproduction claims - which is the exact thing US-008
            # forbids, and the harder half of it to notice.
            elif (existing.payload, existing.seed, existing.seed_reason) != (
                payload,
                seed,
                seed_reason,
            ):
                raise ControlPlaneError(
                    f"configuration {snapshot.id} is already published with different "
                    "content; published versions are immutable (US-008)"
                )


class PostgresModelRegistry:
    """Model lineage, approval gates and rollback targets (FR-033, QA-03)."""

    _PROMOTABLE = frozenset({ApprovalState.EVALUATED, ApprovalState.APPROVED, ApprovalState.CANARY})

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def active_for(self, modality: Modality) -> ModelVersion:
        async with self._factory() as db:
            row = await db.scalar(
                select(models.ModelVersionRow).where(
                    models.ModelVersionRow.modality == modality.value,
                    models.ModelVersionRow.is_active.is_(True),
                )
            )
        if row is None:
            raise ControlPlaneError(
                f"no active model for {modality.value}; the pipeline cannot run a "
                "modality whose version it could not record (NFR-014)"
            )
        return _row_to_model(row)

    async def get(self, model_id: ModelVersionId) -> ModelVersion | None:
        async with self._factory() as db:
            row = await db.get(models.ModelVersionRow, model_id.value)
        return _row_to_model(row) if row is not None else None

    async def promote(self, model_id: ModelVersionId, canary_percent: int) -> ModelVersion:
        if not 0 <= canary_percent <= 100:
            raise ControlPlaneError(f"canary percentage out of range: {canary_percent}")

        async with unit_of_work(self._factory) as db:
            row = await db.get(models.ModelVersionRow, model_id.value)
            if row is None:
                raise ControlPlaneError(f"unknown model version {model_id}")
            if ApprovalState(row.approval) not in self._PROMOTABLE:
                raise ControlPlaneError(
                    f"model {model_id} is '{row.approval}'; only evaluated or approved "
                    "artifacts may be promoted (US-007)"
                )

            previous = await db.scalar(
                select(models.ModelVersionRow).where(
                    models.ModelVersionRow.modality == row.modality,
                    models.ModelVersionRow.is_active.is_(True),
                )
            )
            if previous is not None and previous.id != row.id:
                previous.is_active = False
                # Recorded at promotion, not chosen during an incident. That is
                # what makes QA-03's ten-minute rollback achievable.
                row.rollback_to = previous.id
                # Flushed before the new version is activated. The partial
                # unique index allows one active row per modality and is not
                # deferrable, so leaving both updates to a single flush lets
                # the driver order them the wrong way round and violate it.
                # Found by running this against a real PostgreSQL; SQLite and
                # the in-memory adapter both accept the unflushed version.
                await db.flush()

            row.is_active = True
            row.approval = (
                ApprovalState.PRODUCTION.value
                if canary_percent == 100
                else ApprovalState.CANARY.value
            )
            promoted = _row_to_model(row)
        return promoted

    async def rollback(self, modality: Modality) -> ModelVersion:
        async with unit_of_work(self._factory) as db:
            current = await db.scalar(
                select(models.ModelVersionRow).where(
                    models.ModelVersionRow.modality == modality.value,
                    models.ModelVersionRow.is_active.is_(True),
                )
            )
            if current is None:
                raise ControlPlaneError(f"no active model for {modality.value}")
            if current.rollback_to is None:
                raise ControlPlaneError(
                    f"no rollback target recorded for {modality.value}; "
                    "a promotion without one cannot be undone"
                )
            target = await db.get(models.ModelVersionRow, current.rollback_to)
            if target is None:
                raise ControlPlaneError(
                    f"rollback target {current.rollback_to} is missing from the registry"
                )
            current.is_active = False
            current.approval = ApprovalState.ROLLED_BACK.value
            # Same ordering constraint as `promote`.
            await db.flush()
            target.is_active = True
            restored = _row_to_model(target)
        return restored

    async def list_versions(self, modality: Modality) -> Sequence[ModelVersion]:
        async with self._factory() as db:
            rows = (
                await db.scalars(
                    select(models.ModelVersionRow).where(
                        models.ModelVersionRow.modality == modality.value
                    )
                )
            ).all()
        return tuple(_row_to_model(row) for row in rows)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def _snapshot_to_json(snapshot: ConfigurationSnapshot) -> dict[str, Any]:
    calibration = snapshot.calibration
    return {
        "fusion_window_ms": snapshot.fusion_window.width_ms,
        "speech_thresholds": dict(snapshot.speech_thresholds),
        "visual_thresholds": dict(snapshot.visual_thresholds),
        "silence_threshold_ms": snapshot.silence_threshold_ms,
        "calibration": (
            {
                "baseline_yaw_degrees": calibration.baseline_yaw_degrees,
                "baseline_pitch_degrees": calibration.baseline_pitch_degrees,
                "baseline_torso_degrees": calibration.baseline_torso_degrees,
                "gaze_cone_degrees": calibration.gaze_cone_degrees,
                "posture_deviation_degrees": calibration.posture_deviation_degrees,
                "min_landmark_visibility": calibration.min_landmark_visibility,
            }
            if calibration is not None
            else None
        ),
    }


def _row_to_snapshot(row: models.ConfigurationSnapshotRow) -> ConfigurationSnapshot:
    payload = row.payload
    snapshot_id = ConfigurationSnapshotId(row.id)
    calibration = payload.get("calibration")
    return ConfigurationSnapshot(
        id=snapshot_id,
        taxonomy_version=SemanticVersion.parse(row.taxonomy_version),
        pipeline_version=SemanticVersion.parse(row.pipeline_version),
        schema_version=SemanticVersion.parse(row.schema_version),
        fusion_window=FusionWindow(
            width_ms=int(payload["fusion_window_ms"]), configuration=snapshot_id
        ),
        seed=seed_from_columns(row.seed, row.seed_reason),
        speech_thresholds=dict(payload.get("speech_thresholds", {})),
        visual_thresholds=dict(payload.get("visual_thresholds", {})),
        silence_threshold_ms=int(payload.get("silence_threshold_ms", 700)),
        calibration=VisualCalibration(**calibration) if calibration else None,
    )


def _row_to_model(row: models.ModelVersionRow) -> ModelVersion:
    return ModelVersion(
        id=ModelVersionId(row.id),
        modality=Modality(row.modality),
        artifact_digest=row.artifact_digest,
        dataset_version=row.dataset_version,
        approval=ApprovalState(row.approval),
        metrics=dict(row.metrics),
        rollback_to=ModelVersionId(row.rollback_to) if row.rollback_to else None,
    )
