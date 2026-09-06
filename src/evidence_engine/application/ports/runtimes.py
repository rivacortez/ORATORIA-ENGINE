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

``AudioWindow`` also refuses to be built from a declaration its own payload
contradicts. The port is the last place that still holds both halves - the
declared shape and the bytes - and once the window is handed on, a duration
that is six times too long is indistinguishable from a speaker who paused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from evidence_engine.domain.sessions.capabilities import SUPPORTED_SAMPLE_RATES_HZ
from evidence_engine.domain.shared.errors import FabricatedValue
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable
from evidence_engine.domain.shared.provenance import ModelRole
from evidence_engine.domain.shared.taxonomy import (
    ContextualRole,
    ProsodicIndicator,
    SpeechEventType,
    VisualEventType,
)
from evidence_engine.domain.visual_events.events import GazeDirection

#: The shape a decoded window is in unless it says otherwise: one channel of
#: 16-bit samples. These are the engine's own decode target, not a guess about
#: what the client sent - §7.2 gives the wire no field for either, so a window
#: carrying anything else has to declare it here rather than be inferred.
DEFAULT_CHANNEL_COUNT = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2


class MisdeclaredAudioWindow(Exception):
    """A window's declared shape does not describe the audio it carries."""


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """A decoded, resampled slice of audio handed to a speech runtime.

    Carries its own position on the session clock. A runtime that had to infer
    where it was from call order would produce wrong timestamps the first time
    a window was retried after a timeout (§6.3), and retries are expected.

    The declared shape is checked against ``samples`` on construction, because
    nothing downstream can do it. ``StreamingCoordinator`` ends the window at
    ``session_position_ms + duration_ms`` and the runtime hears whatever bytes
    arrived; when those two disagree, the transcript still renders, the events
    still carry provenance, and every boundary is scaled by the ratio between
    them. NFR-004's 250 ms target would then be measured against a clock that
    is lying, which is worse than having no target at all.
    """

    session_position_ms: int
    duration_ms: int
    sample_rate_hz: int
    samples: bytes
    is_final_window: bool = False
    #: The framing ``samples`` uses. Declared rather than assumed because the
    #: consistency check divides by it: a stereo payload checked as mono counts
    #: twice the frames it has, and so agrees with a declaration of twice its
    #: real length - the exact class of error the check exists to catch.
    channel_count: int = DEFAULT_CHANNEL_COUNT
    sample_width_bytes: int = DEFAULT_SAMPLE_WIDTH_BYTES

    def __post_init__(self) -> None:
        _require_declaration_matches_payload(self)


def _require_declaration_matches_payload(window: AudioWindow) -> None:
    """Refuse a window whose declaration and payload describe different audio.

    Every failure below renders. The runtime returns hypotheses, the assembler
    calibrates them, the coordinator finalizes through the declared end of the
    window, and the client receives a transcript at times that look entirely
    plausible - they are simply about a different stretch of the recording than
    the one the speaker produced. A note at the bottom of a report does not
    survive being copied into a results table, so the window is refused here.

    The checks run in this order because each one is the next one's premise:
    the sample rate is the divisor that turns frames into milliseconds, and the
    framing is the divisor that turns bytes into frames.
    """
    if window.duration_ms <= 0:
        raise MisdeclaredAudioWindow(
            f"a window at {window.session_position_ms} ms declares "
            f"{window.duration_ms} ms of audio. The window ends at "
            "session_position_ms + duration_ms, so a non-positive duration ends it at "
            "or before its own start: nothing it carries is ever reached by the stable "
            "frontier, and the audio drops out of the timeline without a gap being "
            "reported. Zero is also what an undeclared chunk defaults to on the wire, "
            "which is why it is refused rather than tolerated as an empty window."
        )

    if window.sample_rate_hz not in SUPPORTED_SAMPLE_RATES_HZ:
        raise MisdeclaredAudioWindow(
            f"a window declares {window.sample_rate_hz} Hz, which FR-006 never "
            f"negotiates; accepted: {sorted(SUPPORTED_SAMPLE_RATES_HZ)}. The declared "
            "rate is what converts a byte count into a duration, so a window timed at "
            "a rate it was not decoded at scales every boundary by the ratio between "
            "the two: 48 kHz read as 16 kHz stretches a 250 ms event to 750 ms, and "
            "nothing about the result looks wrong."
        )

    if window.channel_count < 1 or window.sample_width_bytes < 1:
        raise MisdeclaredAudioWindow(
            f"a window declares {window.channel_count} channel(s) of "
            f"{window.sample_width_bytes}-byte samples. Neither can be below one; a "
            "zero makes the frame size zero and leaves the duration undefined rather "
            "than wrong, which is the one failure that would reach a reader as a "
            "crash instead of as a number."
        )

    frame_bytes = window.channel_count * window.sample_width_bytes
    leftover = len(window.samples) % frame_bytes
    if leftover:
        raise MisdeclaredAudioWindow(
            f"a window carries {len(window.samples)} bytes, which is not a whole "
            f"number of {frame_bytes}-byte frames ({leftover} left over). Either the "
            "payload was truncated in transit or it is not in the framing it declares. "
            "Both mean the frame count - and every timestamp derived from it - would "
            "be computed from a sample boundary that is not there."
        )

    # Integer form of ``abs(frames / sample_rate_hz * 1000 - duration_ms) <= 1``,
    # multiplied through by the rate so that nothing here is decided by a float.
    # One millisecond is the resolution the declaration itself is written in
    # (§7.4 ``duration_ms``), so a client that rounds and a client that truncates
    # both pass, and anything larger is a disagreement rather than a rounding
    # difference.
    frames = len(window.samples) // frame_bytes
    if abs(frames * 1_000 - window.duration_ms * window.sample_rate_hz) > window.sample_rate_hz:
        carried_ms = frames * 1_000 / window.sample_rate_hz
        raise MisdeclaredAudioWindow(
            f"a window at {window.session_position_ms} ms declares "
            f"{window.duration_ms} ms but carries {carried_ms:.1f} ms - {frames} frames "
            f"of {frame_bytes} bytes at {window.sample_rate_hz} Hz. The coordinator "
            "times this window by the declaration and the runtime hears the payload, "
            "so every word, event boundary and prosody reading it produces would be "
            "placed at a moment the speaker was somewhere else, and each later window "
            "inherits the same offset."
        )


#: What a recogniser reports about its own certainty for one item, or an
#: explicit statement that it reports nothing.
#:
#: Not ``float | None``. ``measurement.py`` argues that encoding out at length
#: and the argument holds here: somebody writes ``score or 0.0`` to make a
#: chart render, and a word the model never scored becomes a word it scored
#: zero. ``Unavailable`` raises on ``.value``, so the same line fails loudly.
type RecogniserScore = float | Unavailable


@dataclass(frozen=True, slots=True)
class TimedWordHypothesis:
    """One literal word the recogniser heard, and placed on the clock."""

    raw_text: str
    start_ms: int
    end_ms: int
    score: RecogniserScore
    #: Position in the order this runtime emitted words for this window. The
    #: assembler pairs it with the window position to build a session-global
    #: `TokenSequence`; a runtime that renumbered between passes would churn
    #: every token id, so it is the emission order and nothing else.
    index: int = 0


@dataclass(frozen=True, slots=True)
class UntimedWordHypothesis:
    """One literal word the recogniser heard and could **not** place.

    The alignment heads fail on fragments. When they do, what is unknown is
    *where* the word was - not *whether* it was said, and not what it was. This
    adapter used to drop such a chunk entirely, deleting a recognised word from
    a verbatim transcript with nothing downstream able to notice: the text
    renders, it is one word shorter, and no count anywhere disagrees. Driver 1
    is verbatim fidelity, so the word survives and its position does not get
    invented.

    There is no ``start_ms`` and no ``end_ms``, by design - the same design as
    ``Unavailable``. Reaching for one raises instead of returning a ``None``
    that a caller might coerce to zero.
    """

    raw_text: str
    score: RecogniserScore
    index: int = 0
    reason: UnavailabilityReason = UnavailabilityReason.ALIGNMENT_UNAVAILABLE
    detail: str = ""

    def __getattr__(self, name: str) -> object:
        if name in {"start_ms", "end_ms", "interval"}:
            raise FabricatedValue(
                f"cannot read '{name}' from a word with no timing "
                f"(reason={self.reason.value}); FR-025 forbids substituting a boundary. "
                "The word is in the transcript; its position on the session clock is not."
            )
        raise AttributeError(name)


#: A word is exactly one of the two cases, so a consumer has to narrow before
#: reading a boundary. ``start_ms: int | None`` on each end would instead admit
#: "start known, end unknown" - a state no aligner produces and every consumer
#: would have to handle.
type WordHypothesis = TimedWordHypothesis | UntimedWordHypothesis


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
    #: Which component produced this hypothesis. The assembler looks the
    #: version up in `SpeechResult.contributions` by this key and refuses a
    #: role the result did not declare - attributing a detector's event to
    #: the recogniser because the detector forgot to say its name is the
    #: silent substitution the whole provenance model exists to prevent.
    #:
    #: No default, on purpose. With `DISFLUENCY_DETECTOR` as the fallback an
    #: event carrying a `context_role` - the classifier's decision, when a
    #: classifier exists - was attributed to the detector by a runtime that
    #: never said so, and nothing could tell a declared role from an omitted
    #: one. The emitter states the role; a wrong role is then a visible
    #: decision at one line rather than a default nobody reads.
    role: ModelRole
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
    role: ModelRole
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class SpeechResult:
    """Everything one speech runtime produced for one window.

    ``contributions`` names every component that produced part of this
    result, by role. A pure recogniser declares one entry; a runtime that
    also runs a detector and a prosody estimator declares three. It replaced
    a single ``model_version`` because one version per result could only
    attribute one component, and a manifest keyed by modality then dropped
    the others with a ``setdefault``. Words are always the recogniser's;
    every other hypothesis names its role and the assembler refuses one
    that was not declared here.
    """

    contributions: Mapping[ModelRole, ModelVersionId]
    #: Where the window this describes began on the session clock. Carried so a
    #: stateless assembler can build a session-global token sequence from a
    #: per-window emission index without being told the window separately -
    #: two arguments that could disagree are two arguments that will.
    #:
    #: Always the position of the window the runtime was *handed*, including
    #: when the result re-emits words from an earlier window as part of its
    #: active region. The coordinator files each ingested window's end under
    #: this key to decide when the words of that window - the unplaced ones
    #: in particular - are settled, and refuses a result reported under any
    #: other position rather than let those words wait for a window that
    #: never arrives.
    window_position_ms: int = 0
    words: tuple[WordHypothesis, ...] = field(default_factory=tuple)
    events: tuple[SpeechEventHypothesis, ...] = field(default_factory=tuple)
    prosody: tuple[ProsodyHypothesis, ...] = field(default_factory=tuple)
    #: Position up to which the runtime considers its output stable. The
    #: application finalizes the transcript through this point and no further:
    #: §6.1 step 9 makes finalization irreversible, so the runtime - not a
    #: fixed lag constant - decides when it is safe.
    stable_through_ms: int = 0

    def __post_init__(self) -> None:
        if ModelRole.RECOGNISER not in self.contributions:
            raise ValueError(
                "a speech result must name its recogniser in `contributions`; "
                "words with no attributable model are the gap NFR-014 forbids"
            )

    @property
    def model_version(self) -> ModelVersionId:
        """The recogniser's version - what every word is attributed to."""
        return self.contributions[ModelRole.RECOGNISER]


class SpeechRuntime(Protocol):
    """Verbatim recognition, acoustic detection and prosody over audio.

    A runtime states what it can produce. `/v1/capabilities` used to publish
    the whole taxonomy regardless of what was wired, so a deployment running a
    recogniser and no disfluency detector advertised every disfluency class it
    could not detect - and a consumer integrating against that would build a
    view for findings that were never coming.
    """

    #: Taxonomy classes this runtime can actually emit. Empty is a legitimate
    #: and common answer: a pure recogniser produces words, not events.
    emitted_speech_events: frozenset[SpeechEventType]
    #: Prosodic indicators this runtime can actually measure.
    emitted_prosody: frozenset[ProsodicIndicator]
    #: One sentence a consumer can read about why the rest is absent.
    capability_detail: str

    #: The versions this runtime will stamp on its results, by role. Declared
    #: on the runtime and not only on each result so the processing run can
    #: record what was wired *before* the first window: a run that fails on
    #: window one still has to say which model it was running.
    @property
    def contributions(self) -> Mapping[ModelRole, ModelVersionId]: ...

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

    contributions: Mapping[ModelRole, ModelVersionId]
    events: tuple[VisualEventHypothesis, ...] = field(default_factory=tuple)
    quality: tuple[VisualQualitySignal, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if ModelRole.VISUAL_ESTIMATOR not in self.contributions:
            raise ValueError("a visual result must name its estimator in `contributions`")

    @property
    def model_version(self) -> ModelVersionId:
        return self.contributions[ModelRole.VISUAL_ESTIMATOR]


class VisionRuntime(Protocol):
    """Observable visual estimation over sampled frames or client geometry."""

    #: Visual classes this runtime can actually emit.
    emitted_visual_events: frozenset[VisualEventType]
    #: One sentence a consumer can read about why the rest is absent.
    capability_detail: str

    @property
    def contributions(self) -> Mapping[ModelRole, ModelVersionId]: ...

    async def observe(self, frames: Sequence[VisualFrame]) -> VisualResult:
        """Produce hypotheses and quality signals for a batch of frames.

        Batched rather than per-frame because FR-021 measures periodicity, and
        a single frame cannot exhibit any. Passing frames one at a time would
        force the adapter to keep the window itself, which is state the
        application already owns.
        """
        ...
