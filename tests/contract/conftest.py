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
from evidence_engine.domain.evidence.cooccurrence import FusionWindow, MultimodalCooccurrence
from evidence_engine.domain.evidence.document import EvidenceDocument, ProvenanceManifest
from evidence_engine.domain.quality.assessment import (
    ModalityAvailability,
    QualityAssessment,
    QualityMetric,
    QualityReport,
)
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.identifiers import (
    ApplicationId,
    ConfigurationSnapshotId,
    EventId,
    EvidenceRef,
    ModelVersionId,
    RunId,
    SessionId,
    TenantId,
    TokenId,
)
from evidence_engine.domain.shared.measurement import Measured, UnavailabilityReason, Unavailable
from evidence_engine.domain.shared.provenance import (
    Modality,
    ModelRole,
    Provenance,
    SemanticVersion,
)
from evidence_engine.domain.shared.taxonomy import (
    TAXONOMY_VERSION,
    ContextualRole,
    ProsodicIndicator,
    SpeechEventType,
    VisualEventType,
)
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading
from evidence_engine.domain.transcript import transcript as transcript_module
from evidence_engine.domain.transcript.tokens import Timed, TokenSequence, TokenStatus, WordToken
from evidence_engine.domain.visual_events.events import GazeDirection, VisualEvent

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


# ---------------------------------------------------------------------------
# A fully populated `EvidenceDocument` - every branch of the serializer, in
# one fixture, shared by every conformance-shaped test under `tests/contract`.
# Originally local to `test_published_payload_carries_no_ranking.py`; moved
# here so `test_remote_client.py`'s round-trip conformance test can build the
# same document rather than a second, narrower one of its own.
# ---------------------------------------------------------------------------


@pytest.fixture
def audio() -> Provenance:
    return Provenance(
        modality=Modality.AUDIO,
        role=ModelRole.RECOGNISER,
        model_version=ModelVersionId("asr-baseline-0001"),
        taxonomy_version=TAXONOMY_VERSION,
        configuration=ConfigurationSnapshotId("config-0001"),
        evidence_ref=EvidenceRef("evidence-audio-0001"),
    )


@pytest.fixture
def video() -> Provenance:
    return Provenance(
        modality=Modality.VIDEO,
        role=ModelRole.VISUAL_ESTIMATOR,
        model_version=ModelVersionId("vision-baseline-0001"),
        taxonomy_version=TAXONOMY_VERSION,
        configuration=ConfigurationSnapshotId("config-0001"),
        evidence_ref=EvidenceRef("evidence-video-0001"),
    )


@pytest.fixture
def document(audio: Provenance, video: Provenance) -> EvidenceDocument:
    """Every branch of the serializer, in one document.

    Confidence rises with time throughout - tokens, speech events, visual
    events - so clock order and confidence order are opposites of each other
    rather than the same list twice. A document where they agree cannot tell a
    time-ordered payload from a severity-ordered one, and every ordering
    assertion in `test_published_payload_carries_no_ranking.py` would hold
    against code that ranked.

    Measured and unavailable both appear, in prosody and in quality, because
    ``_render_indicator`` produces two disjoint key sets and a payload built
    only from measured values leaves the other one unwalked.
    """
    tokens = [
        WordToken(
            id=TokenId("t-1"),
            sequence=TokenSequence(window_position_ms=0, index=0),
            raw_text="buenos",
            placement=Timed(Interval.of(0, 400)),
            confidence=Confidence.calibrated(0.55),
            provenance=audio,
            status=TokenStatus.FINAL,
        ),
        WordToken(
            id=TokenId("t-2"),
            sequence=TokenSequence(window_position_ms=0, index=1),
            raw_text="dias",
            placement=Timed(Interval.of(400, 800)),
            confidence=Confidence.calibrated(0.71),
            provenance=audio,
            status=TokenStatus.FINAL,
        ),
        WordToken(
            id=TokenId("t-3"),
            sequence=TokenSequence(window_position_ms=0, index=2),
            raw_text="este",
            placement=Timed(Interval.of(6_100, 6_500)),
            confidence=Confidence.calibrated(0.94),
            provenance=audio,
            status=TokenStatus.FINAL,
        ),
    ]

    speech_events = (
        # Acoustic class: no lexical role, no raw text.
        SpeechEvent(
            id=EventId("speech-1"),
            type=SpeechEventType.FILLED_PAUSE,
            interval=Interval.of(900, 1_680),
            confidence=Confidence.calibrated(0.61),
            provenance=audio,
            is_final=True,
        ),
        SpeechEvent(
            id=EventId("speech-2"),
            type=SpeechEventType.REPETITION,
            interval=Interval.of(1_800, 2_200),
            confidence=Confidence.calibrated(0.78),
            provenance=audio,
            raw_text="los los",
            context_role=ContextualRole.FILLER,
            is_final=True,
        ),
        SpeechEvent(
            id=EventId("speech-3"),
            type=SpeechEventType.LEXICAL_FILLER,
            interval=Interval.of(6_100, 6_500),
            confidence=Confidence.calibrated(0.93),
            provenance=audio,
            raw_text="este",
            context_role=ContextualRole.UNCERTAIN,
            is_final=True,
        ),
    )

    visual_events = (
        # A gaze class, so `direction` is populated rather than None.
        VisualEvent(
            id=EventId("visual-1"),
            type=VisualEventType.GAZE_AWAY_FROM_CAMERA,
            interval=Interval.of(1_000, 1_900),
            confidence=Confidence.calibrated(0.66),
            provenance=video,
            direction=GazeDirection.DOWN,
            is_final=True,
        ),
        # A capture-quality class, so `describes_capture_quality` is true here
        # and false elsewhere: FR-020's distinction has to survive to the wire.
        VisualEvent(
            id=EventId("visual-2"),
            type=VisualEventType.INSUFFICIENT_LIGHTING,
            interval=Interval.of(3_000, 3_800),
            confidence=Confidence.calibrated(0.74),
            provenance=video,
            is_final=True,
        ),
        VisualEvent(
            id=EventId("visual-3"),
            type=VisualEventType.POSTURE_DEVIATION,
            interval=Interval.of(6_000, 6_900),
            confidence=Confidence.calibrated(0.88),
            provenance=video,
            magnitude=22.0,
            is_final=True,
        ),
    )

    prosody = (
        ProsodyReading.measured(
            indicator=ProsodicIndicator.PITCH_MEAN_HZ,
            window=Interval.of(0, 5_000),
            value=182.4,
            confidence=Confidence.calibrated(0.81),
            provenance=audio,
        ),
        ProsodyReading.unavailable(
            indicator=ProsodicIndicator.SPEAKING_RATE_WPM,
            window=Interval.of(5_000, 10_000),
            unavailable=Unavailable(
                reason=UnavailabilityReason.INPUT_GAP,
                detail="3 of 50 audio chunks never arrived",
            ),
            provenance=audio,
        ),
    )

    quality = QualityReport(
        assessments=(
            QualityAssessment(
                modality=Modality.AUDIO,
                metric=QualityMetric.SIGNAL_TO_NOISE_DB,
                window=Interval.of(0, 10_000),
                value=Measured(value=24.0, confidence=Confidence.calibrated(1.0), unit="dB"),
            ),
            QualityAssessment(
                modality=Modality.VIDEO,
                metric=QualityMetric.FACE_VISIBILITY_RATIO,
                window=Interval.of(0, 10_000),
                value=Unavailable(
                    reason=UnavailabilityReason.MODALITY_NOT_CAPTURED,
                    detail="audio-only session",
                ),
            ),
        ),
        availability=(
            ModalityAvailability(
                modality=Modality.AUDIO,
                window=Interval.of(0, 10_000),
                is_usable=True,
            ),
            ModalityAvailability(
                modality=Modality.VIDEO,
                window=Interval.of(0, 10_000),
                is_usable=False,
                reason=UnavailabilityReason.MODALITY_NOT_CAPTURED,
                detail="the session was created without video",
            ),
        ),
    )

    window = FusionWindow(width_ms=500, configuration=ConfigurationSnapshotId("config-0001"))

    return EvidenceDocument(
        session_id=SessionId("session-0001"),
        run_id=RunId("run-0001"),
        manifest=ProvenanceManifest(
            pipeline_version=SemanticVersion(1, 0, 0),
            schema_version=SemanticVersion(1, 0, 0),
            taxonomy_version=TAXONOMY_VERSION,
            configuration=ConfigurationSnapshotId("config-0001"),
            models={
                Modality.AUDIO: ModelVersionId("asr-baseline-0001"),
                Modality.VIDEO: ModelVersionId("vision-baseline-0001"),
            },
        ),
        transcript=transcript_module.build(tokens),
        quality=quality,
        speech_events=speech_events,
        visual_events=visual_events,
        prosody=prosody,
        cooccurrences=(
            MultimodalCooccurrence(
                speech_event_id=EventId("speech-1"),
                visual_event_id=EventId("visual-1"),
                temporal_distance_ms=100,
                window=window,
            ),
        ),
    )
