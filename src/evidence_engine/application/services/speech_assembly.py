"""Turning speech-runtime hypotheses into published domain evidence.

This is the boundary where a vendor's output becomes this engine's claim, and
four spec rules are applied here rather than anywhere else.

*FR-025.* A prosody hypothesis with ``value=None`` becomes an ``Unavailable``
carrying a reason. Nothing downstream ever sees a substituted zero, because the
substitution has no place to happen: the port hands over ``None``, and the only
thing this module can build from ``None`` is an explicit absence.

*FR-013 and FR-017.* A lexical class whose role the classifier did not decide
becomes ``UNCERTAIN`` - not a guess, and not a dropped event. Its raw text is
kept regardless of role, which is what leaves NFR-003's per-class precision
computable.

*FR-029.* Low-confidence P0 evidence is published with its low confidence
attached. The result "must not hide low-confidence evidence", so thresholds do
not filter here; they inform the consumer through the confidence value.

*NFR-014.* Provenance is attached to every event as it is built, from the model
version the runtime reported. Reading it from deployment config instead would
be wrong during a canary, when two versions are answering at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from evidence_engine.application.ports.platform import ConfigurationSnapshot
from evidence_engine.application.ports.runtimes import (
    ProsodyHypothesis,
    SpeechEventHypothesis,
    SpeechResult,
    TimedWordHypothesis,
    WordHypothesis,
)
from evidence_engine.application.services.calibration import Calibrator
from evidence_engine.application.services.event_identity import derive_event_id, derive_token_id
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.identifiers import (
    EvidenceRef,
    RunId,
    TokenId,
)
from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable
from evidence_engine.domain.shared.provenance import Modality, Provenance
from evidence_engine.domain.shared.taxonomy import ContextualRole, SpeechEventType
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.speech_events.prosody import ProsodyReading
from evidence_engine.domain.transcript.tokens import (
    AlignmentUnavailable,
    Placement,
    Timed,
    TokenSequence,
    WordToken,
)

#: Classes decided from a recognized word, which therefore need a role. Mirrors
#: the domain's own set; duplicated here rather than imported because the
#: domain keeps it private to its constructor check, and the two serve
#: different purposes - the domain refuses a bad event, this decides what to
#: supply so no bad event is built in the first place.
_LEXICAL_CLASSES: frozenset[SpeechEventType] = frozenset(
    {
        SpeechEventType.LEXICAL_FILLER,
        SpeechEventType.REPETITION,
        SpeechEventType.FALSE_START,
        SpeechEventType.SELF_REPAIR,
    }
)


@dataclass(frozen=True, slots=True)
class AssembledSpeech:
    """What one window of audio yielded, ready for the ledger and the wire."""

    tokens: tuple[WordToken, ...]
    events: tuple[SpeechEvent, ...]
    prosody: tuple[ProsodyReading, ...]
    stable_through_ms: int


class SpeechAssembler:
    """Builds domain evidence from one speech runtime result.

    Stateless. Every call is a pure function of its arguments, which is what
    lets §6.3 retry an idempotent window after a timeout without the second
    attempt differing from the first.
    """

    def __init__(
        self,
        run_id: RunId,
        configuration: ConfigurationSnapshot,
        calibrator: Calibrator,
    ) -> None:
        self._run_id = run_id
        self._configuration = configuration
        self._calibrator = calibrator

    def assemble(self, result: SpeechResult) -> AssembledSpeech:
        provenance = self._provenance(result)
        return AssembledSpeech(
            tokens=tuple(self._token(word, result.window_position_ms) for word in result.words),
            events=tuple(self._event(hypothesis, provenance) for hypothesis in result.events),
            prosody=tuple(self._prosody(hypothesis, provenance) for hypothesis in result.prosody),
            stable_through_ms=result.stable_through_ms,
        )

    # -- pieces -----------------------------------------------------------

    def _provenance(self, result: SpeechResult) -> Provenance:
        return Provenance(
            modality=Modality.AUDIO,
            model_version=result.model_version,
            taxonomy_version=self._configuration.taxonomy_version,
            configuration=self._configuration.id,
            evidence_ref=EvidenceRef(f"audio:{self._run_id.value}"),
        )

    def _token(self, word: WordHypothesis, window_position_ms: int) -> WordToken:
        """One hypothesis becomes one token, whether or not it could be placed.

        Both branches produce a token. The untimed branch used to produce
        nothing at all, which deleted a recognised word from the authoritative
        verbatim record for a reason that had nothing to do with what the
        speaker said.
        """
        sequence = TokenSequence(window_position_ms=window_position_ms, index=word.index)
        placement: Placement = (
            Timed(Interval.of(word.start_ms, word.end_ms))
            if isinstance(word, TimedWordHypothesis)
            else AlignmentUnavailable(reason=word.reason, detail=word.detail)
        )
        return WordToken(
            id=TokenId(derive_token_id(self._run_id, sequence, word.raw_text)),
            sequence=sequence,
            raw_text=word.raw_text,
            placement=placement,
            confidence=self._calibrator.calibrate("word", word.score),
        )

    def _event(self, hypothesis: SpeechEventHypothesis, provenance: Provenance) -> SpeechEvent:
        role = self._role_for(hypothesis)
        return SpeechEvent(
            id=derive_event_id(
                self._run_id,
                hypothesis.type.value,
                hypothesis.start_ms,
                discriminator=hypothesis.raw_text,
            ),
            type=hypothesis.type,
            interval=Interval.of(hypothesis.start_ms, hypothesis.end_ms),
            confidence=self._calibrator.calibrate(hypothesis.type.value, hypothesis.score),
            provenance=provenance,
            raw_text=hypothesis.raw_text,
            context_role=role,
        )

    def _role_for(self, hypothesis: SpeechEventHypothesis) -> ContextualRole | None:
        """Resolve the contextual role, never by guessing.

        A lexical class arriving without a decision is ``UNCERTAIN``: §17
        prescribes exactly that for ambiguous fillers, and an event labelled
        uncertain costs a consumer far less than a confident wrong one. A
        non-lexical class keeps ``None``, because the domain refuses a role on
        a class that has no word to attach it to.
        """
        if hypothesis.type not in _LEXICAL_CLASSES:
            return None
        return hypothesis.context_role or ContextualRole.UNCERTAIN

    def _prosody(self, hypothesis: ProsodyHypothesis, provenance: Provenance) -> ProsodyReading:
        window = Interval.of(hypothesis.start_ms, hypothesis.end_ms)
        if hypothesis.value is None:
            return ProsodyReading.unavailable(
                indicator=hypothesis.indicator,
                window=window,
                unavailable=Unavailable(
                    reason=UnavailabilityReason.INSUFFICIENT_OBSERVATION,
                    detail=(
                        f"the runtime could not measure {hypothesis.indicator.value} "
                        f"over [{hypothesis.start_ms}, {hypothesis.end_ms}) ms"
                    ),
                ),
                provenance=provenance,
            )
        return ProsodyReading.measured(
            indicator=hypothesis.indicator,
            window=window,
            value=hypothesis.value,
            confidence=self._calibrator.calibrate(hypothesis.indicator.value, hypothesis.score),
            provenance=provenance,
        )


def silent_pauses(
    tokens: Sequence[WordToken],
    threshold_ms: int,
    provenance: Provenance,
    run_id: RunId,
) -> tuple[SpeechEvent, ...]:
    """Derive silent pauses from the gaps between words (FR-015).

    Computed from the transcript rather than asked of the runtime, because the
    threshold is versioned configuration and must be reproducible: the same
    audio under a different snapshot has to yield a different set of pauses,
    which is impossible if the recognizer decided them internally.

    Only interior gaps count. The silence before the first word and after the
    last are not pauses in the presentation - they are the operator finding the
    stop button - and the taxonomy says so explicitly.
    """
    ordered = sorted(tokens, key=lambda t: t.interval.start.ms)
    pauses: list[SpeechEvent] = []
    for previous, following in pairwise(ordered):
        gap_ms = following.interval.start.ms - previous.interval.end.ms
        if gap_ms < threshold_ms:
            continue
        start_ms = previous.interval.end.ms
        pauses.append(
            SpeechEvent(
                id=derive_event_id(run_id, SpeechEventType.SILENT_PAUSE.value, start_ms),
                type=SpeechEventType.SILENT_PAUSE,
                interval=Interval.of(start_ms, following.interval.start.ms),
                # A gap between two timestamps is arithmetic, not estimation.
                # The uncertainty lives in the word boundaries, which already
                # carry their own tolerance, so the pause itself is certain
                # given them.
                confidence=Confidence.calibrated(1.0),
                provenance=provenance,
            )
        )
    return tuple(pauses)
