"""Speech events: typed, timestamped, attributable disfluency evidence.

Two rules from §9.3 shape this module and both are enforced in the constructor
rather than left to callers.

FR-017 requires the raw expression to be retained *even when the contextual
role is semantic*. The temptation is obvious - if "este" turned out to be a
demonstrative, why keep the event at all? Because discarding it destroys the
denominator. NFR-003 requires per-class precision and recall on ambiguous
lexical fillers, and precision is uncomputable if the true negatives were
dropped before anyone counted them. The event stays; its role says it is not a
filler; nothing downstream counts it as one.

FR-013's ``UNCERTAIN`` role is a real outcome, not a placeholder. An engine that
guesses between "filler" and "semantic" to avoid saying "I don't know" is
manufacturing the false positives §17 lists as a top risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import DomainError
from evidence_engine.domain.shared.identifiers import EventId
from evidence_engine.domain.shared.provenance import Modality, Provenance
from evidence_engine.domain.shared.taxonomy import (
    ContextualRole,
    DetectionBasis,
    SpeechEventType,
    definition_of,
)
from evidence_engine.domain.shared.timeline import Interval


class SpeechEventViolation(DomainError):
    """A speech event was constructed in a shape the taxonomy forbids."""


#: Classes decided from a recognized word, which therefore must carry the word
#: and must state which role it played. The acoustic and temporal classes are
#: absent on purpose: a filled pause has no lexical identity to assign a role
#: to, and forcing one would invite a classifier to invent it.
_LEXICAL_CLASSES: frozenset[SpeechEventType] = frozenset(
    {
        SpeechEventType.LEXICAL_FILLER,
        SpeechEventType.REPETITION,
        SpeechEventType.FALSE_START,
        SpeechEventType.SELF_REPAIR,
    }
)

#: Classes that count toward a disfluency rate. Membership is a property of the
#: class, not of a downstream consumer's opinion, so it lives with the taxonomy.
#: A silent pause is evidence but not a disfluency; an unintelligible stretch is
#: a failure of the recording, not of the speaker.
_DISFLUENT_CLASSES: frozenset[SpeechEventType] = frozenset(
    {
        SpeechEventType.FILLED_PAUSE,
        SpeechEventType.LEXICAL_FILLER,
        SpeechEventType.REPETITION,
        SpeechEventType.FALSE_START,
        SpeechEventType.SELF_REPAIR,
        SpeechEventType.CUT_OFF,
        SpeechEventType.PROLONGATION,
    }
)


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    """§8 ``SpeechEvent``: one observation about how something was said.

    The event describes the utterance, never the speaker. "A filled pause
    occurred here, 780 ms long, confidence 0.83" is the whole content. Any
    statement about what that implies belongs to the consuming application's
    own engine, which is the separation §3 driver 8 exists to protect.
    """

    id: EventId
    type: SpeechEventType
    interval: Interval
    confidence: Confidence
    provenance: Provenance
    #: The literal text this event covers. Required for lexical classes,
    #: optional (and usually empty) for purely acoustic or temporal ones.
    raw_text: str = ""
    #: Which role the expression played. Required for lexical classes.
    context_role: ContextualRole | None = None
    is_final: bool = False

    def __post_init__(self) -> None:
        self._require_audio_modality()
        self._require_lexical_fields()
        self._require_no_role_on_acoustic_classes()

    def _require_audio_modality(self) -> None:
        if self.provenance.modality is not Modality.AUDIO:
            raise SpeechEventViolation(
                f"speech event {self.id} claims modality "
                f"'{self.provenance.modality.value}'; speech evidence comes from audio"
            )

    def _require_lexical_fields(self) -> None:
        if self.type not in _LEXICAL_CLASSES:
            return
        if not self.raw_text.strip():
            raise SpeechEventViolation(
                f"'{self.type.value}' is decided from a recognized word, so raw_text is "
                "required; FR-017 keeps the raw expression regardless of its role"
            )
        if self.context_role is None:
            raise SpeechEventViolation(
                f"'{self.type.value}' requires a contextual role (FR-013); "
                f"use '{ContextualRole.UNCERTAIN.value}' when the context does not decide it"
            )

    def _require_no_role_on_acoustic_classes(self) -> None:
        if self.type in _LEXICAL_CLASSES or self.context_role is None:
            return
        raise SpeechEventViolation(
            f"'{self.type.value}' is decided from "
            f"{definition_of(self.type).basis.value} evidence and has no lexical role to "
            "assign; setting one would assert a distinction the detector never made"
        )

    # -- queries ----------------------------------------------------------

    @property
    def counts_as_disfluency(self) -> bool:
        """Whether this event contributes to a disfluency rate.

        A lexical class only counts when its role is ``FILLER``. This single
        check is the mitigation §17 prescribes for ambiguous lexical fillers:
        a speaker who uses "o sea" as a genuine reformulation marker is not
        penalised for it, and one who uses it to hesitate is.
        """
        if self.type not in _DISFLUENT_CLASSES:
            return False
        if self.type in _LEXICAL_CLASSES:
            return self.context_role is ContextualRole.FILLER
        return True

    @property
    def basis(self) -> DetectionBasis:
        return definition_of(self.type).basis

    def finalize(self) -> SpeechEvent:
        """Promote a provisional event to final, keeping its identity (§7.4)."""
        if self.is_final:
            return self
        return SpeechEvent(
            id=self.id,
            type=self.type,
            interval=self.interval,
            confidence=self.confidence,
            provenance=self.provenance,
            raw_text=self.raw_text,
            context_role=self.context_role,
            is_final=True,
        )
