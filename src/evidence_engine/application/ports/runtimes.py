"""Model runtime ports: speech and vision, each replaceable on its own.

NFR-016 requires speech, vision and fusion runtimes to implement independent
outbound ports and be replaceable without changing domain entities or public
contracts, and QA-03 requires a new contextual classifier to be canaried
without touching a schema. Those two requirements are what this module exists
to make true.

The ports return *hypotheses*, not events. A runtime says "there is a filled
pause here, my score is 0.83, my artifact digest is X"; deciding whether that
becomes a published event - after calibration, after the quality gate, under
this session's thresholds - is the application's job. Letting a runtime hand
back a finished ``SpeechEvent`` would put threshold and taxonomy decisions
inside a vendor adapter, and the next model swap would silently change them.

Every hypothesis carries the artifact digest that produced it. NFR-014 requires
the model version on every derived event, and collecting it here rather than
from deployment config is what keeps it true during a canary, when two versions
are answering at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.taxonomy import (
    ContextualRole,
    ProsodicIndicator,
    SpeechEventType,
    VisualEventType,
)
from evidence_engine.domain.visual_events.events import GazeDirection


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """A decoded, resampled slice of audio handed to a speech runtime.

    Carries its own position on the session clock. A runtime that had to infer
    where it was from call order would produce wrong timestamps the first time
    a window was retried after a timeout (§6.3), and retries are expected.
    """

    session_position_ms: int
    duration_ms: int
    sample_rate_hz: int
    samples: bytes
    is_final_window: bool = False


@dataclass(frozen=True, slots=True)
class WordHypothesis:
    """One literal word the recognizer heard, with its estimated boundaries."""

    raw_text: str
    start_ms: int
    end_ms: int
    score: float


@dataclass(frozen=True, slots=True)
class SpeechEventHypothesis:
    """A candidate disfluency, before calibration and thresholding.

    ``context_role`` is optional because the acoustic detector cannot supply it
    and the contextual classifier runs later. ``None`` here means "not decided
    yet", which the application resolves to a role or to
    ``ContextualRole.UNCERTAIN`` - never by guessing.
    """

    type: SpeechEventType
    start_ms: int
    end_ms: int
    score: float
    raw_text: str = ""
    context_role: ContextualRole | None = None


@dataclass(frozen=True, slots=True)
class ProsodyHypothesis:
    """One prosodic measurement over a window, or an explicit failure.

    ``value`` may be ``None``, and that is the runtime's way of saying it could
    not measure. The application turns it into an ``Unavailable`` with a reason
    rather than into a zero - FR-025 lives at that boundary, so the port is
    shaped to make the absence impossible to overlook.
    """

    indicator: ProsodicIndicator
    start_ms: int
    end_ms: int
    value: float | None
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class SpeechResult:
    """Everything one speech runtime produced for one window."""

    model_version: ModelVersionId
    words: tuple[WordHypothesis, ...] = field(default_factory=tuple)
    events: tuple[SpeechEventHypothesis, ...] = field(default_factory=tuple)
    prosody: tuple[ProsodyHypothesis, ...] = field(default_factory=tuple)
    #: Position up to which the runtime considers its output stable. The
    #: application finalizes the transcript through this point and no further:
    #: §6.1 step 9 makes finalization irreversible, so the runtime - not a
    #: fixed lag constant - decides when it is safe.
    stable_through_ms: int = 0


class SpeechRuntime(Protocol):
    """Verbatim recognition, acoustic detection and prosody over audio."""

    async def transcribe(self, window: AudioWindow) -> SpeechResult:
        """Produce hypotheses for one window.

        Must be idempotent over the same window: §6.3 retries idempotent
        windows after a model timeout, and a runtime that accumulated state
        across calls would double-count the retried audio.
        """
        ...


@dataclass(frozen=True, slots=True)
class VisualFrame:
    """A sampled frame, or the geometry extracted from one.

    ``pixels`` is ``None`` when the client did its own extraction and sent
    landmarks instead (ADR-009, §7.2 ``visual.features``). That path is the
    privacy-preserving one and the pipeline treats it as ordinary input rather
    than as a degraded mode: the estimators downstream consume geometry either
    way, so nothing after this point needs to know which arrived.
    """

    session_position_ms: int
    pixels: bytes | None = None
    landmarks: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class VisualEventHypothesis:
    """A candidate visual observation, before thresholding."""

    type: VisualEventType
    start_ms: int
    end_ms: int
    score: float
    direction: GazeDirection | None = None
    magnitude: float | None = None


@dataclass(frozen=True, slots=True)
class VisualQualitySignal:
    """What the quality gate saw, whether or not any event followed.

    Reported even when everything was fine, because FR-024 asks for quality per
    window and an absent record is indistinguishable from an unprocessed one.
    """

    start_ms: int
    end_ms: int
    face_visibility: float
    mean_luminance: float
    framing_coverage: float
    landmark_confidence: float


@dataclass(frozen=True, slots=True)
class VisualResult:
    """Everything one vision runtime produced for a batch of frames."""

    model_version: ModelVersionId
    events: tuple[VisualEventHypothesis, ...] = field(default_factory=tuple)
    quality: tuple[VisualQualitySignal, ...] = field(default_factory=tuple)


class VisionRuntime(Protocol):
    """Observable visual estimation over sampled frames or client geometry."""

    async def observe(self, frames: Sequence[VisualFrame]) -> VisualResult:
        """Produce hypotheses and quality signals for a batch of frames.

        Batched rather than per-frame because FR-021 measures periodicity, and
        a single frame cannot exhibit any. Passing frames one at a time would
        force the adapter to keep the window itself, which is state the
        application already owns.
        """
        ...
