"""The persistent adapters behave like the in-memory ones.

Each test here has a counterpart in the contract suite that runs against the
memory backend. The pairing is the point: a rule the contracts prove is only
true in production if the persistent adapter obeys it too, and the two most
likely places for it to stop being true are the tenant scoping and the
constraint that FR-025 leans on.

The check constraints get their own tests. They are the last line - a direct
write or a bad migration bypasses every Python check and does not bypass these
- and a constraint nobody has ever seen reject anything is a constraint that
might not be doing what its name says.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidence_engine.adapters.outbound.persistence.identity import hash_secret
from evidence_engine.adapters.outbound.persistence.postgres import models
from evidence_engine.adapters.outbound.persistence.postgres.control_plane import (
    ControlPlaneError,
    PostgresApiKeyDirectory,
    PostgresModelRegistry,
)
from evidence_engine.adapters.outbound.persistence.postgres.engine import unit_of_work
from evidence_engine.adapters.outbound.persistence.postgres.repositories import (
    PostgresAuditLog,
    PostgresSessionRepository,
)
from evidence_engine.application.ports.platform import ApprovalState, Scope
from evidence_engine.application.ports.repositories import AuditRecord
from evidence_engine.domain.sessions.capabilities import (
    AudioCodec,
    NegotiatedCapabilities,
)
from evidence_engine.domain.sessions.consent import ConsentReceipt, RetentionPolicy
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.sessions.state import SessionMode, SessionState
from evidence_engine.domain.shared.identifiers import (
    ApiKeyId,
    ApplicationId,
    ConfigurationSnapshotId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import ModelRole, SemanticVersion

pytestmark = [pytest.mark.integration]

POLICY = SemanticVersion(1, 0, 0)
TENANT = TenantId("tenant-a")
OTHER_TENANT = TenantId("tenant-b")


def _session(session_id: str, tenant: TenantId) -> AnalysisSession:
    identifier = SessionId(session_id)
    return AnalysisSession(
        id=identifier,
        application_id=ApplicationId("app-1"),
        tenant_id=tenant,
        mode=SessionMode.REALTIME,
        locale="es-PE",
        capabilities=NegotiatedCapabilities(
            audio_codec=AudioCodec.PCM16,
            sample_rate_hz=16_000,
            locale="es-PE",
            video_format=None,
            frame_rate_fps=None,
        ),
        configuration=ConfigurationSnapshotId("config-default-v1"),
        created_at_ms=1_000,
    ).with_consent(
        ConsentReceipt(
            session_id=identifier,
            policy_version=POLICY,
            retention=RetentionPolicy.ephemeral(POLICY),
            granted_at_ms=1_000,
        )
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


async def test_a_session_round_trips_with_its_clock_and_consent(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = PostgresSessionRepository(factory)
    original = _session("s-1", TENANT).begin_capture(wall_ms=5_000)

    await repository.add(original)
    loaded = await repository.get(TENANT, original.id)

    assert loaded is not None
    assert loaded.state is SessionState.CAPTURING
    assert loaded.clock.segment_started_at == 5_000
    assert loaded.consent is not None
    assert loaded.consent.is_active


async def test_another_tenant_cannot_read_the_session(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """NFR-013, at the adapter rather than at the API."""
    repository = PostgresSessionRepository(factory)
    await repository.add(_session("s-1", TENANT))

    assert await repository.get(OTHER_TENANT, SessionId("s-1")) is None


async def test_the_pause_and_resume_cycle_survives_a_round_trip(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """FR-009 across a process restart, which is when the clock is at risk."""
    repository = PostgresSessionRepository(factory)
    session = _session("s-1", TENANT).begin_capture(wall_ms=0).pause_capture(wall_ms=10_000)
    await repository.add(session)

    reloaded = await repository.get(TENANT, session.id)
    assert reloaded is not None
    assert reloaded.captured_ms(wall_ms=999_999) == 10_000

    resumed = reloaded.resume_capture(wall_ms=100_000)
    await repository.save(resumed)

    final = await repository.get(TENANT, session.id)
    assert final is not None
    assert final.captured_ms(wall_ms=102_000) == 12_000


async def test_an_idempotency_key_can_only_be_claimed_once(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = PostgresSessionRepository(factory)
    application = ApplicationId("app-1")

    assert await repository.claim_idempotency_key(application, "k", SessionId("s-1"))
    assert not await repository.claim_idempotency_key(application, "k", SessionId("s-2"))


# ---------------------------------------------------------------------------
# Check constraints
# ---------------------------------------------------------------------------


async def test_a_prosody_row_cannot_hold_both_a_value_and_a_reason(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """FR-025 in DDL. A row is a measurement or an absence, never both."""
    await _seed_run(factory)

    with pytest.raises(IntegrityError, match="ck_prosody_measured_xor_unavailable"):
        async with unit_of_work(factory) as db:
            db.add(
                models.ProsodyReadingRow(
                    run_id="run-1",
                    tenant_id=TENANT.value,
                    indicator="pitch_mean_hz",
                    start_ms=0,
                    end_ms=1_000,
                    value=132.0,
                    unit="Hz",
                    reason="input_gap",
                    role="prosody_estimator",
                    model_version="m-1",
                    taxonomy_version="1.0.0",
                    configuration_id="c-1",
                    evidence_ref="e-1",
                )
            )


async def test_a_prosody_row_cannot_hold_neither(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_run(factory)

    with pytest.raises(IntegrityError, match="ck_prosody_measured_xor_unavailable"):
        async with unit_of_work(factory) as db:
            db.add(
                models.ProsodyReadingRow(
                    run_id="run-1",
                    tenant_id=TENANT.value,
                    indicator="pitch_mean_hz",
                    start_ms=0,
                    end_ms=1_000,
                    role="prosody_estimator",
                    model_version="m-1",
                    taxonomy_version="1.0.0",
                    configuration_id="c-1",
                    evidence_ref="e-1",
                )
            )


async def test_an_unusable_window_must_state_a_reason(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_run(factory)

    with pytest.raises(IntegrityError, match="ck_availability_reason_required"):
        async with unit_of_work(factory) as db:
            db.add(
                models.ModalityAvailabilityRow(
                    run_id="run-1",
                    tenant_id=TENANT.value,
                    modality="video",
                    start_ms=0,
                    end_ms=1_000,
                    is_usable=False,
                    reason=None,
                )
            )


async def test_a_speech_event_cannot_carry_an_invented_role(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """FR-013's vocabulary is closed, and the database knows it."""
    await _seed_run(factory)

    with pytest.raises(IntegrityError, match="ck_speech_event_role"):
        async with unit_of_work(factory) as db:
            db.add(
                models.SpeechEventRow(
                    id="ev-1",
                    run_id="run-1",
                    tenant_id=TENANT.value,
                    type="lexical_filler",
                    raw_text="este",
                    context_role="nervous",
                    start_ms=0,
                    end_ms=400,
                    tolerance_ms=250,
                    confidence=0.9,
                    calibration="calibrated",
                    role="disfluency_detector",
                    model_version="m-1",
                    taxonomy_version="1.0.0",
                    configuration_id="c-1",
                    evidence_ref="e-1",
                )
            )


async def test_a_cooccurrence_cannot_exceed_its_own_window(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_run(factory)

    with pytest.raises(IntegrityError, match="ck_cooccurrence_within_window"):
        async with unit_of_work(factory) as db:
            db.add(
                models.CooccurrenceRow(
                    run_id="run-1",
                    tenant_id=TENANT.value,
                    speech_event_id="s-1",
                    visual_event_id="v-1",
                    temporal_distance_ms=900,
                    fusion_window_ms=500,
                    configuration_id="c-1",
                )
            )


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------


async def test_a_revoked_key_stops_authenticating(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    pepper = "integration-pepper-at-least-32-characters"
    secret = "oek_integration-test-secret"

    # Two transactions, not one: `api_key` has a foreign key to
    # `client_application`, and inserting both in a single flush leaves the
    # order to SQLAlchemy. In production the application row is created by a
    # separate administrative call, so this matches reality as well as working.
    async with unit_of_work(factory) as db:
        db.add(
            models.ClientApplicationRow(
                id="app-1", tenant_id=TENANT.value, name="test", status="active"
            )
        )

    async with unit_of_work(factory) as db:
        db.add(
            models.ApiKeyRow(
                id="key-1",
                application_id="app-1",
                tenant_id=TENANT.value,
                hashed_secret=hash_secret(secret, pepper),
                prefix="oek_int",
                scopes=[Scope.SESSIONS_WRITE.value],
            )
        )

    directory = PostgresApiKeyDirectory(factory, pepper, lambda: 10_000)

    caller = await directory.authenticate(secret, "trace-1")
    assert caller is not None
    assert caller.tenant == TENANT

    await directory.revoke(ApiKeyId("key-1"), at_ms=11_000)
    assert await directory.authenticate(secret, "trace-2") is None


async def test_only_one_model_per_role_can_be_active(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """Two would make provenance ambiguous for every event produced (NFR-014).

    Per *role*, not per modality: a recogniser and a disfluency detector are
    both audio and both active at once, which the modality-keyed index
    forbade - the exact configuration QA-03 requires for a canary.
    """
    async with unit_of_work(factory) as db:
        for identifier in ("m-1", "m-2"):
            db.add(
                models.ModelVersionRow(
                    id=identifier,
                    role="recogniser",
                    artifact_digest=f"sha256:{identifier}",
                    dataset_version="v1",
                    approval=ApprovalState.EVALUATED.value,
                    metrics={},
                    is_active=identifier == "m-1",
                )
            )

    with pytest.raises(IntegrityError):
        async with unit_of_work(factory) as db:
            await db.execute(text("update model_version set is_active = true where id = 'm-2'"))


async def test_promotion_records_its_rollback_target(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """QA-03: the target is chosen at promotion, not during an incident."""
    async with unit_of_work(factory) as db:
        db.add(
            models.ModelVersionRow(
                id="m-1",
                role="recogniser",
                artifact_digest="sha256:m-1",
                dataset_version="v1",
                approval=ApprovalState.PRODUCTION.value,
                metrics={},
                is_active=True,
            )
        )
        db.add(
            models.ModelVersionRow(
                id="m-2",
                role="recogniser",
                artifact_digest="sha256:m-2",
                dataset_version="v2",
                approval=ApprovalState.EVALUATED.value,
                metrics={"macro_f1": 0.81},
                is_active=False,
            )
        )

    registry = PostgresModelRegistry(factory)
    from evidence_engine.domain.shared.identifiers import ModelVersionId

    promoted = await registry.promote(ModelVersionId("m-2"), canary_percent=100)
    assert promoted.rollback_to == ModelVersionId("m-1")

    restored = await registry.rollback(ModelRole.RECOGNISER)
    assert restored.id == ModelVersionId("m-1")


async def test_an_unevaluated_model_cannot_be_promoted(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """§5 and US-007: only evaluated artifacts reach production."""
    async with unit_of_work(factory) as db:
        db.add(
            models.ModelVersionRow(
                id="draft-1",
                role="recogniser",
                artifact_digest="sha256:draft",
                dataset_version="v0",
                approval=ApprovalState.DRAFT.value,
                metrics={},
                is_active=False,
            )
        )

    from evidence_engine.domain.shared.identifiers import ModelVersionId

    with pytest.raises(ControlPlaneError, match="US-007"):
        await PostgresModelRegistry(factory).promote(ModelVersionId("draft-1"), canary_percent=10)


async def test_the_audit_log_reads_back_oldest_first(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """The port's contract, not the query's convenience."""
    log = PostgresAuditLog(factory)
    for index in range(3):
        await log.record(
            AuditRecord(
                actor="app-1",
                action="session.create",
                resource="s-1",
                timestamp_ms=1_000 + index,
                trace_id=f"t-{index}",
                outcome="created",
                tenant=TENANT,
            )
        )

    entries = await log.for_resource("s-1")

    assert [entry.trace_id for entry in entries] == ["t-0", "t-1", "t-2"]


async def _seed_run(factory: async_sessionmaker[AsyncSession]) -> None:
    """A session and a run for the evidence rows to hang off."""
    await PostgresSessionRepository(factory).add(_session("s-1", TENANT))
    async with unit_of_work(factory) as db:
        db.add(
            models.ProcessingRunRow(
                id="run-1",
                session_id="s-1",
                pipeline_version="0.1.0",
                state="running",
                started_at_ms=1_000,
                completed_stages=[],
            )
        )
