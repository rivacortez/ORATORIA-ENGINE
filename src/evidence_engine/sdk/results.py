"""The result types a consumer reads. Public, versioned, and not the domain.

``AnalysisResult`` used to carry an ``EvidenceDocument`` - a domain entity -
with a docstring calling it "the published contract, not an internal entity".
It is an internal entity. The surface test passed on the letter, because
``EvidenceDocument`` is not exported from the package root, while every
consumer reaching ``result.document.transcript.tokens[0].placement`` was
coupled to the domain's shape anyway. A leak through a field is still a leak.

So the SDK returns its own types, and they mirror **the wire** rather than the
domain. That choice is what makes the local and hosted paths comparable at all:
a consumer reading `result.transcript.words[0].placement` locally and a consumer
parsing `GET /v1/sessions/{id}/result` are looking at the same shape, and the
day `OratoriaClient` exists it can return these unchanged.

Two encodings are preserved rather than flattened, because flattening them is
the failure FR-025 exists to prevent.

*A word is placed or it is not.* ``TimedPlacement | AlignmentUnavailable``,
never ``start_ms: int | None`` on each end - that would admit "start known, end
unknown", a state no aligner produces.

*A confidence exists or it does not.* whisper-large-v3 reports no per-word
posterior, so most words carry ``ConfidenceUnavailable``. A ``float | None``
here would become ``0.0`` in the first chart somebody drew.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.domain.evidence.document import EvidenceDocument
from evidence_engine.domain.shared.confidence import Confidence as _Confidence
from evidence_engine.domain.shared.measurement import Measured, Unavailable
from evidence_engine.domain.speech_events.events import SpeechEvent as _SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading as _ProsodyReading
from evidence_engine.domain.transcript.tokens import Timed, WordToken
from evidence_engine.domain.transcript.transcript import Transcript as _Transcript
from evidence_engine.domain.visual_events.events import VisualEvent as _VisualEvent

# ---------------------------------------------------------------------------
# The two absences
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimedPlacement:
    """The word was placed on the session clock."""

    start_ms: int
    end_ms: int
    #: NFR-004: the engine never claims an exact boundary, so the tolerance
    #: travels with every interval rather than living in documentation.
    tolerance_ms: int


@dataclass(frozen=True, slots=True)
class AlignmentUnavailable:
    """The word was heard and could not be placed. It is still in the transcript.

    No ``start_ms`` field, deliberately: a zero would put the word at the start
    of the session, which is wrong and entirely plausible-looking in a chart.
    """

    reason: str
    detail: str = ""


type WordPlacement = TimedPlacement | AlignmentUnavailable


@dataclass(frozen=True, slots=True)
class Confidence:
    """A score the model reported, and whether it has been calibrated."""

    value: float
    #: ``raw`` or ``calibrated``. An uncalibrated score never clears a
    #: publication threshold, which is stricter than the arithmetic comparison
    #: and is the point: a number with no probabilistic meaning must not open a
    #: gate.
    calibration: str


@dataclass(frozen=True, slots=True)
class ConfidenceUnavailable:
    """The model reports no confidence for this item, and why.

    Not low confidence and not a failure: the architecture never emitted one.
    This is the normal case for the Whisper baseline, which exposes no per-word
    posterior.
    """

    reason: str
    detail: str = ""


type WordConfidence = Confidence | ConfidenceUnavailable


@dataclass(frozen=True, slots=True)
class Value:
    """A measurement that was actually taken."""

    value: float
    unit: str
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class ValueUnavailable:
    """A measurement that could not be taken, and why (FR-025).

    No ``value`` attribute. Anything needing a number has to narrow first, and
    a consumer that forgot gets an attribute error rather than a zero.
    """

    reason: str
    detail: str = ""


type Indicator = Value | ValueUnavailable


# ---------------------------------------------------------------------------
# The transcript
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Word:
    """One literal word, in lexical order, placed or not."""

    id: str
    #: Lexical position, always present: ``(window_position_ms, index)``. This
    #: is what orders the transcript, including words with no placement.
    sequence: tuple[int, int]
    text: str
    placement: WordPlacement
    confidence: WordConfidence
    #: Which recogniser produced this word (NFR-014).
    model_version: str
    #: ``provisional`` or ``final``. A final word will not be revised.
    status: str

    @property
    def is_timed(self) -> bool:
        return isinstance(self.placement, TimedPlacement)


@dataclass(frozen=True, slots=True)
class Transcript:
    """The verbatim transcript. No cleanup, no punctuation repair (FR-011)."""

    raw_text: str
    words: tuple[Word, ...]
    finalized_through_ms: int
    #: How many words are in the transcript with no position on the clock.
    #: Published rather than derivable, because it says how much of the
    #: temporal analysis ran on less than the whole transcript.
    unaligned_count: int


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    """One detected disfluency, after calibration and thresholding."""

    id: str
    type: str
    start_ms: int
    end_ms: int
    tolerance_ms: int
    confidence: Confidence
    raw_text: str
    context_role: str | None
    counts_as_disfluency: bool


@dataclass(frozen=True, slots=True)
class VisualEvent:
    id: str
    type: str
    start_ms: int
    end_ms: int
    tolerance_ms: int
    confidence: Confidence
    direction: str | None
    describes_capture_quality: bool


@dataclass(frozen=True, slots=True)
class ProsodyReading:
    """One prosodic measurement over a window, or the reason there is none."""

    indicator: str
    start_ms: int
    end_ms: int
    value: Indicator


@dataclass(frozen=True, slots=True)
class Manifest:
    """What produced this result. NFR-014's provenance, on the public surface.

    ``models`` is keyed by **role** - ``recogniser``, ``disfluency_detector``,
    ``context_classifier``, ``prosody_estimator``, ``visual_estimator`` - so a
    consumer can see which component changed between two runs. A map keyed
    by modality held one audio model and could not show a detector being
    canaried beside a fixed recogniser.
    """

    pipeline_version: str
    schema_version: str
    taxonomy_version: str
    configuration_id: str
    models: dict[str, str]


@dataclass(frozen=True, slots=True)
class Evidence:
    """Everything one analysis produced.

    ``ranking_authority`` is the constant ``"none"`` and is published rather
    than merely enforced (§17): a consumer reading it knows the engine will
    never hand it a ranking, and builds its own instead of waiting for a field
    that is not coming.
    """

    session_id: str
    run_id: str
    ranking_authority: str
    manifest: Manifest
    transcript: Transcript
    speech_events: tuple[SpeechEvent, ...]
    visual_events: tuple[VisualEvent, ...]
    prosody: tuple[ProsodyReading, ...]


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def _confidence(value: _Confidence | Unavailable) -> WordConfidence:
    if isinstance(value, Unavailable):
        return ConfidenceUnavailable(reason=value.reason.value, detail=value.detail)
    return Confidence(value=value.value, calibration=value.state.value)


def _required_confidence(value: _Confidence) -> Confidence:
    return Confidence(value=value.value, calibration=value.state.value)


def _indicator(value: Measured | Unavailable) -> Indicator:
    if isinstance(value, Unavailable):
        return ValueUnavailable(reason=value.reason.value, detail=value.detail)
    return Value(
        value=value.value, unit=value.unit, confidence=_required_confidence(value.confidence)
    )


def _word(token: WordToken) -> Word:
    placement: WordPlacement
    if isinstance(token.placement, Timed):
        placement = TimedPlacement(
            start_ms=token.placement.interval.start.ms,
            end_ms=token.placement.interval.end.ms,
            tolerance_ms=token.placement.interval.tolerance_ms,
        )
    else:
        placement = AlignmentUnavailable(
            reason=token.placement.reason.value, detail=token.placement.detail
        )
    return Word(
        id=token.id.value,
        sequence=token.sequence.key,
        text=token.raw_text,
        placement=placement,
        confidence=_confidence(token.confidence),
        model_version=token.provenance.model_version.value,
        status=token.status.value,
    )


def _transcript(transcript: _Transcript) -> Transcript:
    return Transcript(
        raw_text=transcript.raw_text(),
        words=tuple(_word(token) for token in transcript.tokens),
        finalized_through_ms=transcript.finalized_time_frontier.ms,
        unaligned_count=transcript.unaligned_count,
    )


def _speech_event(event: _SpeechEvent) -> SpeechEvent:
    return SpeechEvent(
        id=event.id.value,
        type=event.type.value,
        start_ms=event.interval.start.ms,
        end_ms=event.interval.end.ms,
        tolerance_ms=event.interval.tolerance_ms,
        confidence=_required_confidence(event.confidence),
        raw_text=event.raw_text,
        context_role=event.context_role.value if event.context_role else None,
        counts_as_disfluency=event.counts_as_disfluency,
    )


def _visual_event(event: _VisualEvent) -> VisualEvent:
    return VisualEvent(
        id=event.id.value,
        type=event.type.value,
        start_ms=event.interval.start.ms,
        end_ms=event.interval.end.ms,
        tolerance_ms=event.interval.tolerance_ms,
        confidence=_required_confidence(event.confidence),
        direction=event.direction.value if event.direction else None,
        describes_capture_quality=event.describes_capture_quality,
    )


def _prosody(reading: _ProsodyReading) -> ProsodyReading:
    return ProsodyReading(
        indicator=reading.indicator.value,
        start_ms=reading.window.start.ms,
        end_ms=reading.window.end.ms,
        value=_indicator(reading.value),
    )


def evidence_from(document: EvidenceDocument) -> Evidence:
    """Translate the domain's document into the public shape.

    One direction only. There is no `Evidence -> EvidenceDocument`, and there
    should not be: the domain builds its own entities through constructors that
    enforce invariants, and a translation *into* it would be a second way to
    construct evidence that skipped them.
    """
    return Evidence(
        session_id=document.session_id.value,
        run_id=document.run_id.value,
        ranking_authority=EvidenceDocument.ranking_authority,
        manifest=Manifest(
            pipeline_version=str(document.manifest.pipeline_version),
            schema_version=str(document.manifest.schema_version),
            taxonomy_version=str(document.manifest.taxonomy_version),
            configuration_id=document.manifest.configuration.value,
            models={role.value: model.value for role, model in document.manifest.models.items()},
        ),
        transcript=_transcript(document.transcript),
        speech_events=tuple(_speech_event(e) for e in document.speech_events),
        visual_events=tuple(_visual_event(e) for e in document.visual_events),
        prosody=tuple(_prosody(r) for r in document.prosody),
    )
