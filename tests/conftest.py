"""Shared fixtures.

Everything here is deterministic. NFR-015 requires the same input, artifact,
configuration and seed to produce equivalent output, and a test suite that
seeds itself from ``uuid4`` or the wall clock cannot tell a real reproducibility
failure from its own noise.
"""

from __future__ import annotations

import pytest

from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    ScriptedSpeechEvent,
    ScriptedVisualEvent,
    ScriptedWord,
    SpeechScript,
    VisualScript,
)
from evidence_engine.domain.sessions.capabilities import (
    AudioCodec,
    NegotiatedCapabilities,
    VideoFormat,
)
from evidence_engine.domain.sessions.consent import ConsentReceipt, RetentionPolicy
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.sessions.state import SessionMode
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.identifiers import (
    ApplicationId,
    ConfigurationSnapshotId,
    EvidenceRef,
    ModelVersionId,
    SessionId,
    TenantId,
)
from evidence_engine.domain.shared.provenance import Modality, Provenance, SemanticVersion
from evidence_engine.domain.shared.taxonomy import (
    TAXONOMY_VERSION,
    ContextualRole,
    SpeechEventType,
    VisualEventType,
)
from evidence_engine.domain.visual_events.events import GazeDirection

POLICY_VERSION = SemanticVersion(1, 0, 0)


@pytest.fixture
def session_id() -> SessionId:
    return SessionId("session-0001")


@pytest.fixture
def configuration() -> ConfigurationSnapshotId:
    return ConfigurationSnapshotId("config-0001")


@pytest.fixture
def audio_provenance(configuration: ConfigurationSnapshotId) -> Provenance:
    return Provenance(
        modality=Modality.AUDIO,
        model_version=ModelVersionId("asr-baseline-0001"),
        taxonomy_version=TAXONOMY_VERSION,
        configuration=configuration,
        evidence_ref=EvidenceRef("evidence-audio-0001"),
    )


@pytest.fixture
def video_provenance(configuration: ConfigurationSnapshotId) -> Provenance:
    return Provenance(
        modality=Modality.VIDEO,
        model_version=ModelVersionId("vision-baseline-0001"),
        taxonomy_version=TAXONOMY_VERSION,
        configuration=configuration,
        evidence_ref=EvidenceRef("evidence-video-0001"),
    )


@pytest.fixture
def high_confidence() -> Confidence:
    return Confidence.calibrated(0.92)


@pytest.fixture
def capabilities() -> NegotiatedCapabilities:
    return NegotiatedCapabilities(
        audio_codec=AudioCodec.PCM16,
        sample_rate_hz=16_000,
        locale="es-PE",
        video_format=VideoFormat.LANDMARKS,
        frame_rate_fps=30,
    )


@pytest.fixture
def consent(session_id: SessionId) -> ConsentReceipt:
    return ConsentReceipt(
        session_id=session_id,
        policy_version=POLICY_VERSION,
        retention=RetentionPolicy.ephemeral(POLICY_VERSION),
        granted_at_ms=1_000,
    )


@pytest.fixture
def consented_session(
    session_id: SessionId,
    capabilities: NegotiatedCapabilities,
    configuration: ConfigurationSnapshotId,
    consent: ConsentReceipt,
) -> AnalysisSession:
    """A session that has cleared the FR-031 gate and can accept media."""
    return AnalysisSession(
        id=session_id,
        application_id=ApplicationId("app-0001"),
        tenant_id=TenantId("tenant-0001"),
        mode=SessionMode.REALTIME,
        locale="es-PE",
        capabilities=capabilities,
        configuration=configuration,
        created_at_ms=1_000,
    ).with_consent(consent)


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
