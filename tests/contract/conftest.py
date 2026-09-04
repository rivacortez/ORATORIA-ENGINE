"""A fully wired engine, with nothing non-deterministic in it.

The contract suite runs against a real ASGI application - real routing, real
dependency injection, real serialization - with a frozen clock and the scripted
runtimes from the root conftest underneath. That combination is what Phase 2's
exit criterion asks for: "a synthetic session can be streamed, completed,
queried and deleted without model inference".
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    SpeechScript,
    VisualScript,
)
from evidence_engine.adapters.outbound.persistence.identity import InMemoryApiKeyDirectory
from evidence_engine.adapters.outbound.telemetry.clock import FrozenClock
from evidence_engine.application.ports.platform import Scope
from evidence_engine.bootstrap.app import create_app
from evidence_engine.bootstrap.container import Container, build_container
from evidence_engine.bootstrap.settings import Backend, RuntimeMode, Settings
from evidence_engine.domain.shared.identifiers import ApplicationId, TenantId

TEST_PEPPER = "test-pepper-value-at-least-32-chars-long"
TEST_SIGNING_KEY = "test-signing-key-at-least-32-characters"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        backend=Backend.MEMORY,
        runtime_mode=RuntimeMode.DETERMINISTIC,
        api_key_pepper=TEST_PEPPER,
        stream_token_signing_key=TEST_SIGNING_KEY,
        max_queue_depth=4,
    )


@pytest.fixture
def frozen_clock() -> FrozenClock:
    return FrozenClock(start_ms=1_000_000)


@pytest.fixture
def container(
    settings: Settings,
    frozen_clock: FrozenClock,
    speech_script: SpeechScript,
    visual_script: VisualScript,
) -> Container:
    return build_container(
        settings,
        clock=frozen_clock,
        speech_script=speech_script,
        visual_script=visual_script,
    )


@pytest.fixture
def tenant() -> TenantId:
    return TenantId("tenant-oratoria")


@pytest.fixture
def api_key(container: Container, tenant: TenantId) -> str:
    """A full-scope key. Individual tests narrow it when they test a gate."""
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, _ = directory.issue(
        application=ApplicationId("app-oratoria"),
        tenant=tenant,
        scopes=frozenset(
            {
                Scope.SESSIONS_WRITE,
                Scope.SESSIONS_READ,
                Scope.RESULTS_READ,
                Scope.EVIDENCE_DELETE,
            }
        ),
    )
    return secret


@pytest.fixture
def other_tenant_key(container: Container) -> str:
    """A key for a different tenant, for the NFR-013 isolation checks."""
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)
    secret, _ = directory.issue(
        application=ApplicationId("app-intruder"),
        tenant=TenantId("tenant-intruder"),
        scopes=frozenset({Scope.SESSIONS_READ, Scope.RESULTS_READ, Scope.EVIDENCE_DELETE}),
    )
    return secret


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as test_client:
        yield test_client


@pytest.fixture
def auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}
