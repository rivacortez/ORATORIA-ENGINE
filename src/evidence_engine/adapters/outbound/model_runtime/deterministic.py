"""Deterministic runtimes: the pipeline without a model.

Phase 2's exit criterion is that a synthetic session can be streamed,
completed, queried and deleted *without model inference*, and these are what
make that possible. They are not stubs that return empty results - that would
exercise none of the reconciliation, fusion or quality logic the phase exists
to validate. They replay a script.

A script is a list of utterances with positions, and the runtime returns
whatever falls inside the window it is handed. The consequence is the property
Phase 2 needs and QA-05 needs later: the same session replayed twice produces
byte-identical evidence, so a reproducibility failure in CI is a real
regression rather than a reseeded random number.

They also model the two behaviours that break naive pipelines. Output near the
right edge of a window is reported as *not yet stable*, because a real
streaming recognizer revises its tail as it hears more; and a scripted failure
point raises, so the degradation path of QA-02 is exercised by something other
than an outage in production.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    ProsodyHypothesis,
    SpeechEventHypothesis,
    SpeechResult,
    TimedWordHypothesis,
    VisualEventHypothesis,
    VisualFrame,
    VisualQualitySignal,
    VisualResult,
)
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.provenance import ModelRole
from evidence_engine.domain.shared.taxonomy import (
    ContextualRole,
    ProsodicIndicator,
    SpeechEventType,
    VisualEventType,
)
from evidence_engine.domain.visual_events.events import GazeDirection

#: How far back from the end of a window output is considered still revisable.
#: A real streaming recognizer needs right context before it commits, and a
#: runtime that finalized to the window edge would let the coordinator freeze
#: text the model would have changed - making section 6.1 step 9 untestable.
DEFAULT_INSTABILITY_TAIL_MS = 400


@dataclass(frozen=True, slots=True)
class ScriptedWord:
    """One word the synthetic speaker says, at a fixed position."""

    text: str
    start_ms: int
    end_ms: int
    score: float = 0.92


@dataclass(frozen=True, slots=True)
class ScriptedSpeechEvent:
    """One disfluency the synthetic speaker produces."""

    type: SpeechEventType
    start_ms: int
    end_ms: int
    score: float = 0.85
    raw_text: str = ""
    context_role: ContextualRole | None = None


@dataclass(frozen=True, slots=True)
class SpeechScript:
    """A whole synthetic presentation.

    ``fail_after_window`` makes the QA-02 degradation path reachable from a
    test rather than only from a real outage.
    """

    words: tuple[ScriptedWord, ...] = field(default_factory=tuple)
    events: tuple[ScriptedSpeechEvent, ...] = field(default_factory=tuple)
    model_version: ModelVersionId = field(
        default_factory=lambda: ModelVersionId("deterministic-speech-v1")
    )
    instability_tail_ms: int = DEFAULT_INSTABILITY_TAIL_MS
    fail_after_window: int | None = None


class DeterministicSpeechRuntime:
    """Replays a speech script, window by window.

    Emits its whole *active region* on every call - everything from the last
    point it declared stable up to the end of the current window - not just
    what is new. That is what a streaming recognizer does, and getting it wrong
    is not a detail: ``Transcript.with_provisional`` replaces the revisable
    tail wholesale, so a runtime that reported only new words would silently
    drop every word that had not yet been finalized. The first end-to-end run
    of this pipeline produced "buenos los este" for exactly that reason.
    """

    #: The whole taxonomy. A scripted runtime emits whatever its script says,
    #: so its capability is genuinely the full set - and that is the point of
    #: declaring it per runtime rather than publishing the taxonomy for every
    #: deployment: the contract suite exercises every class, and the managed
    #: baseline advertises none, and both statements are true.
    emitted_speech_events: frozenset[SpeechEventType] = frozenset(SpeechEventType)
    emitted_prosody: frozenset[ProsodicIndicator] = frozenset(ProsodicIndicator)
    capability_detail = "a scripted runtime emits whatever its script declares"

    @property
    def contributions(self) -> Mapping[ModelRole, ModelVersionId]:
        """One script plays every audio role.

        Declared explicitly rather than inferred, because a scripted event
        with a role the runtime had not declared would be refused by the
        assembler - and a script is the one place where playing three
        components under one version is the truth rather than a shortcut.
        """
        # Three components, three versions - even though one script backs
        # them all. `ModelVersion` has one role, and the registry keys stored
        # versions by id: one id under three roles would overwrite the role
        # twice and leave two components registered as something they are
        # not. Deriving the ids from the script's version keeps a scripted
        # run reproducible and keeps the recogniser's id exactly what every
        # existing test and manifest expects.
        version = self._script.model_version
        return {
            ModelRole.RECOGNISER: version,
            ModelRole.DISFLUENCY_DETECTOR: ModelVersionId(f"{version.value}-detector"),
            ModelRole.PROSODY_ESTIMATOR: ModelVersionId(f"{version.value}-prosody"),
        }

    def __init__(self, script: SpeechScript) -> None:
        self._script = script
        self._windows_seen = 0
        #: How far this runtime has already declared stable. Everything after
        #: it is still revisable and gets re-emitted.
        self._stable_ms = 0

    async def transcribe(self, window: AudioWindow) -> SpeechResult:
        self._windows_seen += 1
        if (
            self._script.fail_after_window is not None
            and self._windows_seen > self._script.fail_after_window
        ):
            raise RuntimeError("scripted speech runtime failure")

        window_end_ms = window.session_position_ms + window.duration_ms
        active_from_ms = self._stable_ms

        words = tuple(
            # Always timed. A scripted runtime asserts what it hears, and a
            # script that omitted a boundary would be describing an aligner
            # failure rather than a speaker - the untimed case belongs to the
            # adapters that wrap a real recogniser.
            TimedWordHypothesis(
                raw_text=word.text,
                start_ms=word.start_ms,
                end_ms=word.end_ms,
                score=word.score,
                index=index,
            )
            for index, word in enumerate(
                word
                for word in self._script.words
                if word.end_ms > active_from_ms and word.start_ms < window_end_ms
            )
        )
        events = tuple(
            SpeechEventHypothesis(
                type=event.type,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                score=event.score,
                raw_text=event.raw_text,
                context_role=event.context_role,
            )
            for event in self._script.events
            if event.end_ms > active_from_ms and event.start_ms < window_end_ms
        )

        stable_through_ms = self._stable_through(window_end_ms, window.is_final_window)
        self._stable_ms = max(self._stable_ms, stable_through_ms)

        return SpeechResult(
            contributions=self.contributions,
            window_position_ms=window.session_position_ms,
            words=words,
            events=events,
            prosody=self._prosody(window.session_position_ms, window_end_ms),
            stable_through_ms=stable_through_ms,
        )

    def _stable_through(self, window_end_ms: int, is_final: bool) -> int:
        """Everything up to the instability tail, or all of it on the last window."""
        if is_final:
            return window_end_ms
        return max(0, window_end_ms - self._script.instability_tail_ms)

    def _prosody(self, start_ms: int, end_ms: int) -> tuple[ProsodyHypothesis, ...]:
        """Two measured indicators and one deliberately unmeasurable.

        Pitch comes back as ``None`` on windows with no scripted words, which
        is what a real f0 estimator does over silence. That single ``None`` is
        the input FR-025's whole mechanism is built to handle, so the synthetic
        pipeline produces one by default rather than only when a test asks.
        """
        has_speech = any(start_ms <= word.start_ms < end_ms for word in self._script.words)
        return (
            ProsodyHypothesis(
                indicator=ProsodicIndicator.MEAN_INTENSITY_DB,
                start_ms=start_ms,
                end_ms=end_ms,
                value=-22.5 if has_speech else -60.0,
                score=0.9,
            ),
            ProsodyHypothesis(
                indicator=ProsodicIndicator.PITCH_MEAN_HZ,
                start_ms=start_ms,
                end_ms=end_ms,
                value=132.0 if has_speech else None,
                score=0.8 if has_speech else 0.0,
            ),
            ProsodyHypothesis(
                indicator=ProsodicIndicator.VOICED_RATIO,
                start_ms=start_ms,
                end_ms=end_ms,
                value=0.74 if has_speech else 0.0,
                score=0.95,
            ),
        )


@dataclass(frozen=True, slots=True)
class ScriptedVisualEvent:
    """One visual observation the synthetic speaker produces."""

    type: VisualEventType
    start_ms: int
    end_ms: int
    score: float = 0.88
    direction: GazeDirection | None = None
    magnitude: float | None = None


@dataclass(frozen=True, slots=True)
class VisualScript:
    """A synthetic visual track, with its own quality profile."""

    events: tuple[ScriptedVisualEvent, ...] = field(default_factory=tuple)
    model_version: ModelVersionId = field(
        default_factory=lambda: ModelVersionId("deterministic-vision-v1")
    )
    face_visibility: float = 0.95
    mean_luminance: float = 0.55
    framing_coverage: float = 0.9
    landmark_confidence: float = 0.87
    fail_after_batch: int | None = None


class DeterministicVisionRuntime:
    """Replays a visual script over batches of frames."""

    emitted_visual_events: frozenset[VisualEventType] = frozenset(VisualEventType)
    capability_detail = "a scripted runtime emits whatever its script declares"

    @property
    def contributions(self) -> Mapping[ModelRole, ModelVersionId]:
        return {ModelRole.VISUAL_ESTIMATOR: self._script.model_version}

    def __init__(self, script: VisualScript) -> None:
        self._script = script
        self._batches_seen = 0

    async def observe(self, frames: Sequence[VisualFrame]) -> VisualResult:
        self._batches_seen += 1
        if (
            self._script.fail_after_batch is not None
            and self._batches_seen > self._script.fail_after_batch
        ):
            raise RuntimeError("scripted vision runtime failure")

        if not frames:
            return VisualResult(contributions=self.contributions)

        start_ms = min(frame.session_position_ms for frame in frames)
        end_ms = max(frame.session_position_ms for frame in frames) + 1

        events = tuple(
            VisualEventHypothesis(
                type=event.type,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                score=event.score,
                direction=event.direction,
                magnitude=event.magnitude,
            )
            for event in self._script.events
            if start_ms <= event.start_ms < end_ms
        )

        return VisualResult(
            contributions=self.contributions,
            events=events,
            quality=(
                VisualQualitySignal(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    face_visibility=self._script.face_visibility,
                    mean_luminance=self._script.mean_luminance,
                    framing_coverage=self._script.framing_coverage,
                    landmark_confidence=self._script.landmark_confidence,
                ),
            ),
        )
