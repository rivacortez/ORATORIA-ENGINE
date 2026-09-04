"""Measured or unavailable - never a fabricated number.

This is the smallest module in the domain and the one that carries the most
weight. Architectural driver 4 states that unavailable or uncertain evidence
must never be represented as zero or as a confirmed finding, FR-025 turns that
into a requirement, and QA-02 turns it into a scenario: when the camera drops
for twenty seconds, the visual indicators for that stretch must say why they
are missing, not report ``0.0``.

The obvious encoding, ``float | None``, does not survive contact with real
code. Somebody writes ``value or 0.0`` to make a chart render, or a serializer
maps ``None`` to ``0`` because the schema said the field was numeric, and by
the time the number reaches a student it reads as "you scored zero on eye
contact" rather than "the camera was covered". So the two cases are separate
types, and the unavailable one has no numeric attribute to read at all:
reaching for a value that does not exist raises instead of defaulting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import FabricatedValue


class UnavailabilityReason(StrEnum):
    """Why an indicator could not be produced.

    FR-025 requires a reason, and a free-text one would be unusable: the
    consuming application has to decide whether to tell the student "your
    camera was covered" or "we could not process this stretch", and it cannot
    branch on prose. The enum is closed and each member maps to a distinct
    user-facing situation. §7.4 also requires that unknown enum values must not
    crash consumers, so adding a member later is a compatible change.
    """

    #: The modality was never supplied - video disabled, audio-only session.
    MODALITY_NOT_CAPTURED = "modality_not_captured"
    #: Captured, but the quality gate rejected it (§5 Visual Quality Gate).
    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    #: The subject was not visible or audible for enough of the window.
    SUBJECT_NOT_OBSERVABLE = "subject_not_observable"
    #: Frames or chunks were lost in transport (FR-008).
    INPUT_GAP = "input_gap"
    #: The runtime failed or timed out for this window (§6.3).
    PROCESSING_FAILED = "processing_failed"
    #: Produced, but below the confidence required to publish (FR-022).
    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"
    #: The window was shorter than the indicator needs to be meaningful.
    INSUFFICIENT_OBSERVATION = "insufficient_observation"
    #: Raw evidence was deleted under the retention policy (FR-032).
    EVIDENCE_DELETED = "evidence_deleted"


@dataclass(frozen=True, slots=True)
class Measured:
    """An indicator that was actually observed."""

    value: float
    confidence: Confidence
    unit: str

    @property
    def is_available(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class Unavailable:
    """An indicator that could not be observed, and why.

    There is no ``value`` attribute by design. Anything that needs a number
    from an indicator has to ask, and asking the wrong one raises rather than
    quietly producing a zero.
    """

    reason: UnavailabilityReason
    detail: str = ""

    @property
    def is_available(self) -> bool:
        return False

    def __getattr__(self, name: str) -> object:
        # Only consulted for attributes that do not exist. The two that matter
        # get a domain error naming the rule instead of an AttributeError that
        # a caller might reasonably decide to swallow.
        if name in {"value", "confidence"}:
            raise FabricatedValue(
                f"cannot read '{name}' from an unavailable indicator "
                f"(reason={self.reason.value}); FR-025 forbids substituting a number"
            )
        raise AttributeError(name)


#: An indicator is exactly one of the two cases. Consumers must narrow before
#: reading, which is precisely the check that gets skipped with ``float|None``.
type Indicator = Measured | Unavailable


def value_or_raise(indicator: Indicator) -> float:
    """Read the number, or fail loudly.

    Use this at the point where a number is genuinely required and its absence
    is a bug. Everywhere else, branch on ``is_available`` and carry the reason
    forward: FR-024 makes availability part of the published result, so it is
    information to be reported, not an error to be handled.
    """
    if isinstance(indicator, Unavailable):
        raise FabricatedValue(
            f"indicator is unavailable (reason={indicator.reason.value}); "
            "FR-025 forbids substituting a number"
        )
    return indicator.value
