"""A fully wired engine, with nothing non-deterministic in it.

The contract suite runs against a real ASGI application - real routing, real
dependency injection, real serialization - with a frozen clock and scripted
runtimes underneath. That combination is what Phase 2's exit criterion asks
for: "a synthetic session can be streamed, completed, queried and deleted
without model inference".

The scripted presentation is deliberately awkward. It contains a filled pause,
a repetition, an ambiguous "este" the classifier declined to resolve, a gap
long enough to become a silent pause, and a stretch of visual evidence to fuse
against. A script of clean speech would let half the pipeline pass by doing
nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    ScriptedSpeechEvent,
    ScriptedVisualEvent,
    ScriptedWord,
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
from evidence_engine.domain.shared.taxonomy import (
    ContextualRole,
    SpeechEventType,
    VisualEventType,
)
from evidence_engine.domain.visual_events.events import GazeDirection

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
def speech_script() -> SpeechScript:
    """One awkward minute of synthetic Peruvian Spanish.

    Positions are chosen so that the 2 400 ms gap between "resultados" and
    "muestran" exceeds the default 700 ms silence threshold and becomes a
    SILENT_PAUSE - derived by the engine from the transcript, not supplied by
    the script, which is what FR-015 requires.
    """
    return SpeechScript(
        words=(
            ScriptedWord("buenos", 0, 400),
            ScriptedWord("dias", 400, 800),
            ScriptedWord("eeeh", 900, 1_680, score=0.71),
            ScriptedWord("los", 1_800, 2_000),
            ScriptedWord("los", 2_000, 2_200),
            ScriptedWord("resultados", 2_200, 3_000),
            # A 2 400 ms hole: long enough to become a silent pause.
            ScriptedWord("muestran", 5_400, 6_000),
            ScriptedWord("este", 6_100, 6_500),
            ScriptedWord("una", 6_600, 6_800),
            ScriptedWord("mejora", 6_800, 7_400),
        ),
        events=(
            ScriptedSpeechEvent(SpeechEventType.FILLED_PAUSE, 900, 1_680, score=0.93),
            ScriptedSpeechEvent(
                SpeechEventType.REPETITION,
                1_800,
                2_200,
                score=0.88,
                raw_text="los los",
                context_role=ContextualRole.FILLER,
            ),
            # Role deliberately absent: the classifier did not decide, so the
            # assembler must resolve it to UNCERTAIN rather than guessing.
            ScriptedSpeechEvent(
                SpeechEventType.LEXICAL_FILLER, 6_100, 6_500, score=0.64, raw_text="este"
            ),
        ),
    )


@pytest.fixture
def visual_script() -> VisualScript:
    return VisualScript(
        events=(
            ScriptedVisualEvent(
                VisualEventType.GAZE_AWAY_FROM_CAMERA,
                1_000,
                1_900,
                score=0.9,
                direction=GazeDirection.DOWN,
            ),
            ScriptedVisualEvent(
                VisualEventType.POSTURE_DEVIATION, 6_000, 6_900, score=0.81, magnitude=22.0
            ),
            # A P1 class below FR-022's gate. Must be dropped, not published
            # with low confidence.
            ScriptedVisualEvent(VisualEventType.SELF_TOUCH, 6_200, 6_600, score=0.31),
        )
    )


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
