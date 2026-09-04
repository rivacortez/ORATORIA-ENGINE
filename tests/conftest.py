"""Shared fixtures.

Everything here is deterministic. NFR-015 requires the same input, artifact,
configuration and seed to produce equivalent output, and a test suite that
seeds itself from ``uuid4`` or the wall clock cannot tell a real reproducibility
failure from its own noise.
"""

from __future__ import annotations

import pytest

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
from evidence_engine.domain.shared.taxonomy import TAXONOMY_VERSION

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
