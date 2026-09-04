"""Word tokens: what was said, literally.

FR-011 forbids intentional grammatical cleanup and driver 1 explains why it
matters more than it sounds: "losing a disfluency prevents subsequent
measurement and feedback". Every commercial ASR is tuned in the opposite
direction - readable output is what its customers want - so a token that
reaches this module has already survived a system designed to delete it. From
here on it is immutable.

``raw_text`` is therefore the only authoritative text. There is no ``clean_text``
field and there will not be one: a second, tidier string is exactly the thing a
downstream consumer would end up rendering, and the disfluency would vanish
from the report while remaining in the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import DomainError
from evidence_engine.domain.shared.identifiers import TokenId
from evidence_engine.domain.shared.timeline import Interval


class TranscriptViolation(DomainError):
    """A rule about transcript content or ordering was broken."""


class TokenStatus(StrEnum):
    """Whether this hypothesis can still change.

    FR-016 and §6.1: the engine emits provisional results while the window is
    still moving, then finalizes. ``FINAL`` is a promise to the consumer that
    the text and boundaries will not be revised, which is what lets a UI stop
    re-rendering that stretch.
    """

    PROVISIONAL = "provisional"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class WordToken:
    """§8 ``WordToken``: one literal word hypothesis with its boundaries.

    The identifier is stable across reconciliation (§7.4). A consumer that
    displayed this token as provisional replaces it in place when the final
    arrives; without a stable id it would have to guess by position, and a
    single inserted word would shift everything after it.
    """

    id: TokenId
    raw_text: str
    interval: Interval
    confidence: Confidence
    status: TokenStatus = TokenStatus.PROVISIONAL

    def __post_init__(self) -> None:
        if not self.raw_text:
            raise TranscriptViolation(
                "a word token cannot be empty; an unintelligible stretch is an "
                "UNINTELLIGIBLE speech event, not a blank word"
            )

    @property
    def is_final(self) -> bool:
        return self.status is TokenStatus.FINAL

    def finalize(self) -> WordToken:
        """Promote to final. Text and boundaries are already fixed."""
        if self.is_final:
            return self
        return WordToken(
            id=self.id,
            raw_text=self.raw_text,
            interval=self.interval,
            confidence=self.confidence,
            status=TokenStatus.FINAL,
        )

    def revise(self, raw_text: str, interval: Interval, confidence: Confidence) -> WordToken:
        """Replace the hypothesis while it is still provisional.

        Revising a finalized token is refused rather than allowed-with-a-warning.
        §6.1 step 9 says finalized history is not rewritten, and a consumer that
        has already shown the student "you said X" cannot be quietly corrected
        to "you said Y" - it would make every displayed result provisional in
        practice while claiming to be final.
        """
        if self.is_final:
            raise TranscriptViolation(
                f"token {self.id} is final; finalized history is never rewritten"
            )
        return WordToken(
            id=self.id,
            raw_text=raw_text,
            interval=interval,
            confidence=confidence,
            status=TokenStatus.PROVISIONAL,
        )
