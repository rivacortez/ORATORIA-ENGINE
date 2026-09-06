"""Configuration snapshots and the model registry.

Both enforce the same rule from two directions: what a run executed under is
frozen and knowable afterwards.

``InMemoryConfigurationStore`` keeps published snapshots immutable (US-008) and
binds one to each session, so a later default change cannot reach back into a
finished result. ``InMemoryModelRegistry`` refuses to promote anything that has
not been evaluated (§5, US-007) and keeps a recorded rollback target, because
QA-03 measures rollback in under ten minutes and that is only achievable if the
target was chosen in advance rather than reasoned about during an incident.
"""

from __future__ import annotations

from collections.abc import Sequence

from evidence_engine.application.ports.platform import (
    ApprovalState,
    ConfigurationSnapshot,
    ModelVersion,
)
from evidence_engine.domain.shared.identifiers import (
    ConfigurationSnapshotId,
    ModelVersionId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import ModelRole


class ConfigurationError(Exception):
    """A configuration or promotion rule was violated."""


class InMemoryConfigurationStore:
    """Immutable, versioned snapshots bound to sessions."""

    def __init__(self, default: ConfigurationSnapshot) -> None:
        self._published: dict[str, ConfigurationSnapshot] = {default.id.value: default}
        self._current_by_tenant: dict[str, ConfigurationSnapshot] = {}
        self._default = default
        self._by_session: dict[str, ConfigurationSnapshotId] = {}

    def publish(self, snapshot: ConfigurationSnapshot, tenant: TenantId | None = None) -> None:
        """Publish a snapshot. Re-publishing an id with different content fails.

        US-008: "published versions are immutable". Silently accepting an
        overwrite would make every historical result unverifiable, because the
        id a run recorded would no longer resolve to what it ran under.
        """
        existing = self._published.get(snapshot.id.value)
        if existing is not None and existing != snapshot:
            raise ConfigurationError(
                f"configuration {snapshot.id} is already published with different "
                "content; published versions are immutable (US-008)"
            )
        self._published[snapshot.id.value] = snapshot
        if tenant is not None:
            self._current_by_tenant[tenant.value] = snapshot

    async def current(self, tenant: TenantId) -> ConfigurationSnapshot:
        return self._current_by_tenant.get(tenant.value, self._default)

    async def get(self, snapshot_id: ConfigurationSnapshotId) -> ConfigurationSnapshot | None:
        return self._published.get(snapshot_id.value)

    async def freeze(self, snapshot: ConfigurationSnapshot, session_id: SessionId) -> None:
        self._published.setdefault(snapshot.id.value, snapshot)
        self._by_session[session_id.value] = snapshot.id

    async def for_session(self, session_id: SessionId) -> ConfigurationSnapshot | None:
        """What this session actually ran under. The basis of QA-05's replay."""
        snapshot_id = self._by_session.get(session_id.value)
        if snapshot_id is None:
            return None
        return self._published.get(snapshot_id.value)


#: States a version may be promoted from. §5: the registry must not allow
#: unapproved models in production, and US-007 says only evaluated artifacts
#: can be promoted.
_PROMOTABLE: frozenset[ApprovalState] = frozenset(
    {ApprovalState.EVALUATED, ApprovalState.APPROVED, ApprovalState.CANARY}
)


class InMemoryModelRegistry:
    """Model lineage, approval gates and rollback targets."""

    def __init__(self) -> None:
        self._versions: dict[str, ModelVersion] = {}
        self._active: dict[ModelRole, ModelVersionId] = {}

    def register(self, version: ModelVersion, *, make_active: bool = False) -> None:
        self._versions[version.id.value] = version
        if make_active:
            self._active[version.role] = version.id

    async def active_for(self, role: ModelRole) -> ModelVersion:
        model_id = self._active.get(role)
        if model_id is None:
            raise ConfigurationError(
                f"no active model for {role.value}; the pipeline cannot run a "
                "component whose version it could not record (NFR-014)"
            )
        return self._versions[model_id.value]

    async def get(self, model_id: ModelVersionId) -> ModelVersion | None:
        return self._versions.get(model_id.value)

    async def promote(self, model_id: ModelVersionId, canary_percent: int) -> ModelVersion:
        version = self._versions.get(model_id.value)
        if version is None:
            raise ConfigurationError(f"unknown model version {model_id}")
        if version.approval not in _PROMOTABLE:
            raise ConfigurationError(
                f"model {model_id} is '{version.approval.value}'; only evaluated or "
                "approved artifacts may be promoted (US-007)"
            )
        if not 0 <= canary_percent <= 100:
            raise ConfigurationError(f"canary percentage out of range: {canary_percent}")

        promoted = ModelVersion(
            id=version.id,
            role=version.role,
            artifact_digest=version.artifact_digest,
            dataset_version=version.dataset_version,
            approval=(ApprovalState.PRODUCTION if canary_percent == 100 else ApprovalState.CANARY),
            metrics=version.metrics,
            # The version being replaced becomes the rollback target. Recorded
            # at promotion rather than looked up during an incident, which is
            # what makes QA-03's ten-minute rollback achievable.
            rollback_to=self._active.get(version.role),
        )
        self._versions[promoted.id.value] = promoted
        self._active[promoted.role] = promoted.id
        return promoted

    async def rollback(self, role: ModelRole) -> ModelVersion:
        current = await self.active_for(role)
        if current.rollback_to is None:
            raise ConfigurationError(
                f"no rollback target recorded for {role.value}; "
                "a promotion without one cannot be undone"
            )
        target = self._versions[current.rollback_to.value]
        self._versions[current.id.value] = ModelVersion(
            id=current.id,
            role=current.role,
            artifact_digest=current.artifact_digest,
            dataset_version=current.dataset_version,
            approval=ApprovalState.ROLLED_BACK,
            metrics=current.metrics,
            rollback_to=current.rollback_to,
        )
        self._active[role] = target.id
        return target

    async def list_versions(self, role: ModelRole) -> Sequence[ModelVersion]:
        return tuple(v for v in self._versions.values() if v.role is role)
