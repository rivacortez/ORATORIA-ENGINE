"""OratorIA Multimodal Evidence Engine - the public SDK surface.

Verbatim transcripts and auditable, timestamped multimodal evidence for
single-speaker presentations, embeddable in a consumer's own process.

    from evidence_engine import EngineConfiguration, OratoriaEngine

    engine = OratoriaEngine.local(EngineConfiguration(runtime="baseline_whisper"))
    print(engine.hardware_preflight().render())
    await engine.warmup()

    result = await engine.analyze_file("presentation.wav")
    print(result.document.transcript.raw_text())

    await engine.aclose()

**What this exports, and what it will not.** The list below is a published
contract: once a consumer imports a name from here, changing it is a breaking
change to every integration. So it holds the facade, the configuration a caller
supplies, the result contract they read, the errors they catch, and the
version - and nothing else.

It does **not** export `Container`, any repository, any adapter, any concrete
runtime, or the domain entities the engine builds internally. Those are how the
engine is put together this month. A consumer who imported `InMemoryEvidence-
Repository` or `WhisperSpeechRuntime` would be coupled to a composition
decision, and the next one would break them - which is exactly the coupling
ADR-001 separated this engine from the monolith to avoid.

The domain types remain importable by their full path for anybody who needs
them, and doing so is opting out of this promise rather than being handed a
loaded gun by an `__init__`.

**Importing this module pulls in nothing.** The core has no dependencies at all
(C9); FastAPI, SQLAlchemy and torch arrive only with the extra that needs them.
"""

from __future__ import annotations

from evidence_engine.sdk.client import OratoriaClient
from evidence_engine.sdk.configuration import (
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WINDOW_SECONDS,
    EngineConfiguration,
    SessionConfiguration,
)
from evidence_engine.sdk.engine import AnalysisResult, OratoriaEngine
from evidence_engine.sdk.errors import (
    AudioNotUsable,
    EngineNotWarmed,
    LocalInferenceUnavailable,
    OratoriaError,
    RemoteChunkLost,
    RemoteEngineUnavailable,
    StreamAlreadyClosed,
)
from evidence_engine.sdk.preflight import HardwareReport
from evidence_engine.sdk.results import (
    AlignmentUnavailable,
    Confidence,
    ConfidenceUnavailable,
    Evidence,
    Manifest,
    ProsodyReading,
    SpeechEvent,
    TimedPlacement,
    Transcript,
    Value,
    ValueUnavailable,
    VisualEvent,
    Word,
)
from evidence_engine.sdk.stream import StreamSession

#: The SDK's own version, which is the engine's. Published because a consumer
#: reporting a bug needs to name what they ran, and because §7.4's schema
#: version answers a different question - what the contract looks like, not
#: what produced it.
__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SAMPLE_RATE_HZ",
    "DEFAULT_WINDOW_SECONDS",
    # Absence, in the two shapes it takes. Exported because a consumer has to
    # narrow on them: a word is placed or it is not, a confidence exists or it
    # does not, and `isinstance` is how they find out.
    "AlignmentUnavailable",
    "AnalysisResult",
    "AudioNotUsable",
    "Confidence",
    "ConfidenceUnavailable",
    # Configuration
    "EngineConfiguration",
    "EngineNotWarmed",
    # Results, in the SDK's own types
    "Evidence",
    "HardwareReport",
    "LocalInferenceUnavailable",
    "Manifest",
    # The facade, and its remote twin (ADR-011's deferred client)
    "OratoriaClient",
    "OratoriaEngine",
    # Errors
    "OratoriaError",
    "ProsodyReading",
    "RemoteChunkLost",
    "RemoteEngineUnavailable",
    "SessionConfiguration",
    "SpeechEvent",
    "StreamAlreadyClosed",
    "StreamSession",
    "TimedPlacement",
    "Transcript",
    "Value",
    "ValueUnavailable",
    "VisualEvent",
    "Word",
    "__version__",
]
