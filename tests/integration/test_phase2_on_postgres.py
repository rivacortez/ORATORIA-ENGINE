"""Phase 2's exit criterion, on the persistent backend.

The contract suite proves the sentence against the memory backend, which is
what makes it runnable on every commit. This module proves the same sentence
against PostgreSQL, Redis and S3 — because a rule that holds only in memory
holds nowhere that matters, and the gaps between the two adapter sets are
exactly where it would stop holding: transaction boundaries, JSONB round-trips,
Redis lease semantics, presigned-URL revocation.

Same session, same script, same assertions. Only the wiring differs.
"""

from __future__ import annotations

import base64
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidence_engine.adapters.outbound.persistence.identity import (
    generate_secret,
    hash_secret,
)
from evidence_engine.adapters.outbound.persistence.postgres import models
from evidence_engine.adapters.outbound.persistence.postgres.engine import unit_of_work
from evidence_engine.adapters.outbound.telemetry.clock import FrozenClock
from evidence_engine.application.ports.platform import ApprovalState, Scope
from evidence_engine.bootstrap.app import create_app
from evidence_engine.bootstrap.container import Container, build_container
from evidence_engine.bootstrap.settings import Backend, RuntimeMode, Settings
from tests.contract.conftest import TEST_PEPPER, TEST_SIGNING_KEY
from tests.integration.conftest import DATABASE_URL, REDIS_URL

pytestmark = pytest.mark.integration

MINIO_ENDPOINT = os.environ.get("ENGINE_TEST_S3_ENDPOINT", "http://localhost:9000")
# Matches docker-compose.yml. Test-only credentials for a local container;
# real ones come from the secret manager (spec 15.2).
MINIO_ACCESS_KEY = os.environ.get("ENGINE_TEST_S3_ACCESS_KEY", "engine")
MINIO_SECRET_KEY = os.environ.get("ENGINE_TEST_S3_SECRET_KEY", "engine-secret")

CREATE_BODY = {
    "mode": "realtime",
    "capabilities": {
        "audio_codec": "pcm16",
        "sample_rate_hz": 16_000,
        "locale": "es-PE",
        "video_format": "landmarks",
        "frame_rate_fps": 30,
    },
    "consent_policy_version": "1.0.0",
}

SILENT_CHUNK = base64.b64encode(b"\x00\x00" * 2_560).decode("ascii")


@pytest.fixture
def postgres_settings() -> Settings:
    return Settings(
        environment="test",
        backend=Backend.POSTGRES,
        runtime_mode=RuntimeMode.DETERMINISTIC,
        api_key_pepper=TEST_PEPPER,
        stream_token_signing_key=TEST_SIGNING_KEY,
        database_url=DATABASE_URL,
        redis_url=REDIS_URL,
        object_storage_endpoint=MINIO_ENDPOINT,
        object_storage_access_key=MINIO_ACCESS_KEY,
        object_storage_secret_key=MINIO_SECRET_KEY,
        max_queue_depth=4,
    )


@pytest_asyncio.fixture
async def seeded(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[str, str]]:
    """A tenant, an application and a full-scope key, in the real database."""
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    application = f"app-{uuid.uuid4().hex[:8]}"
    secret = generate_secret()

    async with unit_of_work(factory) as db:
        db.add(
            models.ClientApplicationRow(
                id=application, tenant_id=tenant, name="integration", status="active"
            )
        )

    async with unit_of_work(factory) as db:
        db.add(
            models.ApiKeyRow(
                id=f"key-{uuid.uuid4().hex[:8]}",
                application_id=application,
                tenant_id=tenant,
                hashed_secret=hash_secret(secret, TEST_PEPPER),
                prefix=secret[:10],
                scopes=[
                    Scope.SESSIONS_WRITE.value,
                    Scope.SESSIONS_READ.value,
                    Scope.RESULTS_READ.value,
                    Scope.EVIDENCE_DELETE.value,
                ],
            )
        )
        # The registry is seeded here rather than at process startup: a replica
        # that registered models on boot could silently reintroduce a version an
        # administrator had just disabled.
        for identifier, modality in (
            ("deterministic-speech-v1", "audio"),
            ("deterministic-vision-v1", "video"),
        ):
            db.add(
                models.ModelVersionRow(
                    id=identifier,
                    modality=modality,
                    artifact_digest=f"sha256:deterministic-{modality}",
                    dataset_version="none",
                    approval=ApprovalState.EVALUATED.value,
                    metrics={},
                    is_active=True,
                )
            )

    yield tenant, secret


@pytest.fixture
def postgres_container(
    postgres_settings: Settings,
    speech_script: Any,
    visual_script: Any,
    seeded: tuple[str, str],
) -> Container:
    return build_container(
        postgres_settings,
        clock=FrozenClock(start_ms=1_000_000),
        speech_script=speech_script,
        visual_script=visual_script,
    )


@pytest.fixture
def postgres_client(postgres_container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(postgres_container)) as client:
        yield client


def _stream(client: TestClient, session_id: str, token: str) -> list[dict[str, Any]]:
    received: list[dict[str, Any]] = []
    with client.websocket_connect(f"/v1/sessions/{session_id}/stream?token={token}") as socket:
        received.append(socket.receive_json())

        for index in range(8):
            socket.send_json(
                {
                    "schema_version": "1.0.0",
                    "session_id": session_id,
                    "message_id": f"m-{index}",
                    "chunk_seq": index,
                    "monotonic_time_ms": index * 1_000,
                    "type": "audio.chunk",
                    "samples": SILENT_CHUNK,
                    "sample_rate_hz": 16_000,
                    "duration_ms": 1_000,
                    "is_final": index == 7,
                }
            )
            socket.send_json(
                {
                    "schema_version": "1.0.0",
                    "session_id": session_id,
                    "message_id": f"v-{index}",
                    "chunk_seq": index,
                    "monotonic_time_ms": index * 1_000,
                    "type": "visual.features",
                    "session_position_ms": index * 1_000,
                    "landmarks": [0.5, 0.5, 0.1, 0.2],
                }
            )

        socket.send_json(
            {
                "schema_version": "1.0.0",
                "session_id": session_id,
                "message_id": "m-complete",
                "monotonic_time_ms": 8_000,
                "type": "session.complete",
            }
        )
        for _ in range(400):
            message = socket.receive_json()
            received.append(message)
            if message["type"] == "session.completed":
                break
    return received


def test_the_whole_flow_works_on_postgres_redis_and_s3(
    postgres_client: TestClient, seeded: tuple[str, str]
) -> None:
    _tenant, secret = seeded
    auth = {"Authorization": f"Bearer {secret}"}

    created = postgres_client.post("/v1/sessions", json=CREATE_BODY, headers=auth)
    assert created.status_code == 201, created.text
    body = created.json()

    messages = _stream(postgres_client, body["session_id"], body["stream_token"])
    assert messages[-1]["type"] == "session.completed"

    result = postgres_client.get(f"/v1/sessions/{body['session_id']}/result", headers=auth)
    assert result.status_code == 200, result.text
    document = result.json()

    assert document["ranking_authority"] == "none"
    assert document["transcript"]["raw_text"].startswith("buenos dias")
    assert {e["type"] for e in document["speech_events"]} >= {
        "filled_pause",
        "repetition",
        "silent_pause",
    }

    deleted = postgres_client.delete(f"/v1/sessions/{body['session_id']}/evidence", headers=auth)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["evidence_records_deleted"] >= 1

    gone = postgres_client.get(f"/v1/sessions/{body['session_id']}/result", headers=auth)
    assert gone.status_code == 409


async def test_deletion_actually_removes_the_rows(
    postgres_client: TestClient,
    seeded: tuple[str, str],
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """FR-032: the count in the audit record has to be true of the database."""
    from sqlalchemy import func, select

    _tenant, secret = seeded
    auth = {"Authorization": f"Bearer {secret}"}

    created = postgres_client.post("/v1/sessions", json=CREATE_BODY, headers=auth).json()
    _stream(postgres_client, created["session_id"], created["stream_token"])

    async with factory() as db:
        before = await db.scalar(select(func.count()).select_from(models.SpeechEventRow))
    assert before and before > 0

    postgres_client.delete(f"/v1/sessions/{created['session_id']}/evidence", headers=auth)

    async with factory() as db:
        after = await db.scalar(select(func.count()).select_from(models.SpeechEventRow))
        documents = await db.scalar(select(func.count()).select_from(models.EvidenceDocumentRow))
    assert after == 0
    assert documents == 0

    # US-010: the audit record proves execution without retaining the content.
    async with factory() as db:
        audit_rows = list(
            (
                await db.scalars(
                    select(models.AuditRecordRow).where(
                        models.AuditRecordRow.action == "evidence.delete"
                    )
                )
            ).all()
        )
    assert audit_rows
    assert "evidence_records=" in audit_rows[-1].detail
