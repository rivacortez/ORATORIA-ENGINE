"""What a consumer configures, and what they deliberately cannot.

Two objects, split by lifetime. ``EngineConfiguration`` is per process - which
runtime, which device, how big a window - and ``SessionConfiguration`` is per
presentation: locale, codec, the consent policy the speaker agreed to.

**What is absent is the design.** There is no threshold here, no taxonomy
version, no fusion window. Those live in the `ConfigurationSnapshot` that every
result is bound to (US-008), and letting a caller pass them per call would mean
two runs of the same audio could differ without either saying so. A consumer
who needs different thresholds supplies a whole snapshot, which travels with
the evidence and can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    DeterministicSpeechRuntime,
    SpeechScript,
)
from evidence_engine.application.ports.platform import ConfigurationSnapshot, Telemetry
from evidence_engine.application.ports.runtimes import SpeechRuntime, VisionRuntime
from evidence_engine.domain.evidence.cooccurrence import (
    DEFAULT_FUSION_WINDOW_MS,
    FusionWindow,
)
from evidence_engine.domain.shared.identifiers import ConfigurationSnapshotId
from evidence_engine.domain.shared.provenance import (
    SemanticVersion,
    Unseeded,
    UnseededReason,
)
from evidence_engine.domain.shared.taxonomy import TAXONOMY_VERSION
from evidence_engine.sdk.errors import LocalInferenceUnavailable

#: How the engine decodes. 16 kHz mono 16-bit is the working rate; a recording
#: at another rate is refused rather than resampled.
DEFAULT_SAMPLE_RATE_HZ = 16_000
#: How much audio goes to the model at once. Whisper decodes a 30 s context
#: regardless, so a longer window is not cheaper per second - but a shorter one
#: bounds what is lost when a window fails.
DEFAULT_WINDOW_SECONDS = 25

#: The same versions the hosted service publishes. Bound here so a document
#: produced by an embedded engine and one produced by the service carry the
#: same manifest and can be compared at all.
PIPELINE_VERSION = SemanticVersion(0, 1, 0)
SCHEMA_VERSION = SemanticVersion(1, 0, 0)


@dataclass(frozen=True, slots=True)
class EngineConfiguration:
    """Per-process settings for an embedded engine."""

    #: `"deterministic"` replays a script and needs nothing installed;
    #: `"baseline_whisper"` loads the pinned research baseline and needs the
    #: `local` extra plus torch. The same two names the hosted service uses,
    #: because a figure produced under one of them has to be comparable across
    #: both deployments.
    runtime: str = "deterministic"
    device: str = "cuda"
    dtype: str = "float16"
    cache_dir: str | None = None

    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    max_queue_depth: int = 8

    #: Signs the stream tokens the session machinery mints. Present because the
    #: use case requires one, and irrelevant in-process: nothing crosses a
    #: trust boundary here, so it defaults to a fixed local value rather than
    #: asking a consumer for a secret that protects nothing.
    stream_signing_key: str = "embedded-engine-local-signing-key-32ch"

    #: Module constants rather than dataclass defaults: `SemanticVersion` is
    #: frozen, so a shared instance is safe, and RUF009 is right that a call
    #: in a default is a trap in the general case.
    pipeline_version: SemanticVersion = PIPELINE_VERSION
    schema_version: SemanticVersion = SCHEMA_VERSION

    #: A consumer's own telemetry, or `None` for silence. An SDK that wrote
    #: into somebody's logging configuration uninvited would be making a
    #: deployment decision on their behalf.
    telemetry: Telemetry | None = None
    #: Override the wired runtimes. Present for tests and for a consumer with
    #: their own model; the ports are the extension point NFR-016 promises.
    speech_runtime: SpeechRuntime | None = None
    vision_runtime: VisionRuntime | None = None

    @property
    def window_bytes(self) -> int:
        """One window of 16-bit mono samples."""
        return self.sample_rate_hz * self.window_seconds * 2

    def snapshot(self) -> ConfigurationSnapshot:
        """The snapshot every result from this engine is bound to.

        Empty thresholds and no calibration, exactly as the hosted service
        starts: §14.2 requires calibration to be reported before a version is
        promoted, none has been fitted, so every confidence comes back `RAW`
        and no publication gate opens. An SDK that shipped different defaults
        would produce results a hosted run could not be compared against.
        """
        return ConfigurationSnapshot(
            id=ConfigurationSnapshotId("config-default-v1"),
            taxonomy_version=TAXONOMY_VERSION,
            pipeline_version=self.pipeline_version,
            schema_version=self.schema_version,
            fusion_window=FusionWindow(
                width_ms=DEFAULT_FUSION_WINDOW_MS,
                configuration=ConfigurationSnapshotId("config-default-v1"),
            ),
            # Field for field the same as the service's `default_configuration()`,
            # including the empty threshold maps and the absent calibration.
            # Conformance depends on this: two documents produced from the same
            # audio can only be compared if they were produced under the same
            # snapshot, and a snapshot that differed in one threshold would make
            # every difference downstream unattributable.
            seed=Unseeded(UnseededReason.DETERMINISTIC_RUNTIME),
            speech_thresholds={},
            visual_thresholds={},
            silence_threshold_ms=700,
            calibration=None,
        )

    def build_speech_runtime(self) -> SpeechRuntime:
        """Resolve the speech runtime this configuration names.

        The Whisper import happens here and not at module scope, so an
        embedded install with no `local` extra can construct an engine, run a
        preflight and read the report explaining what to install.
        """
        if self.speech_runtime is not None:
            return self.speech_runtime
        if self.runtime == "deterministic":
            return DeterministicSpeechRuntime(SpeechScript())
        if self.runtime == "baseline_whisper":
            from evidence_engine.adapters.outbound.model_runtime.whisper import (
                WhisperRuntimeUnavailable,
                WhisperSettings,
                WhisperSpeechRuntime,
            )

            try:
                return WhisperSpeechRuntime.load(
                    WhisperSettings(device=self.device, dtype=self.dtype, cache_dir=self.cache_dir)
                )
            except WhisperRuntimeUnavailable as error:
                raise LocalInferenceUnavailable(str(error)) from error
        raise LocalInferenceUnavailable(
            f"unknown runtime {self.runtime!r}; use 'deterministic' or 'baseline_whisper'. "
            "The project model is not built - see ADR-003."
        )


@dataclass(frozen=True, slots=True)
class SessionConfiguration:
    """Per-presentation settings."""

    locale: str = "es-PE"
    #: `pcm16` is the only lossless option and the one the engine decodes. A
    #: lossy codec is accepted by the hosted API because browsers produce them,
    #: and carries a standing quality warning; embedded callers control their
    #: own capture and should not start there.
    audio_codec: str = "pcm16"
    #: The consent policy version the speaker agreed to. Recorded on the
    #: session so a later export can prove which terms the recording was made
    #: under (§14.4).
    consent_policy_version: str = "1.0.0"
    retain_raw_media: bool = False
    raw_media_ttl_seconds: int = 0

    def duration_ms_for(self, samples: bytes, sample_rate_hz: int) -> int:
        """How long a window of 16-bit mono samples lasts.

        Computed from the payload rather than declared by the caller: the
        window refuses a declaration its own bytes contradict, and an SDK that
        made the caller supply both would be handing them a way to get it
        wrong for no benefit.
        """
        return round(len(samples) / 2 / sample_rate_hz * 1000)
