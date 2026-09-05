"""Word tokens: what was said, literally, and where it sat if that is known.

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

**Lexical order and temporal placement are separate things.** A token always
has a position in the sequence the recogniser produced; it does not always have
a position on the clock. The alignment heads fail on fragments, and when they
do, what is unknown is *where* the word was - not whether it was said, and not
what it was. The adapter used to drop such a word entirely, which produced a
transcript one word shorter that nothing downstream could detect.

So the two are modelled separately. ``sequence`` orders the transcript and is
always present; ``placement`` is either a ``Timed`` interval or an explicit
``AlignmentUnavailable``. Everything lexical - the verbatim text, the context a
classifier reads - runs on sequence. Everything temporal - windows, silent
pauses, multimodal fusion - runs on placement and skips what has none, saying
so rather than quietly shortening its input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import DomainError, FabricatedValue
from evidence_engine.domain.shared.identifiers import TokenId
from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable
from evidence_engine.domain.shared.timeline import Interval


class TranscriptViolation(DomainError):
    """A rule about transcript content or ordering was broken."""


@dataclass(frozen=True, slots=True, order=True)
class TokenSequence:
    """Where a word sits in the order the recogniser produced it.

    Two fields rather than one counter, because the assembler is stateless -
    §6.3 retries an idempotent window and a second attempt must produce the
    same ids as the first, which a running counter could not guarantee. The
    window's own position supplies the outer ordinal and the index within the
    window supplies the inner one, so the pair is derivable from the evidence
    alone and re-derives identically on every pass.

    Ordering is lexicographic and agrees with time order for timed tokens: a
    later window always carries a larger ``window_position_ms``, and within one
    window the recogniser emits words in the order it heard them.
    """

    window_position_ms: int
    index: int

    def __post_init__(self) -> None:
        if self.window_position_ms < 0 or self.index < 0:
            raise TranscriptViolation(
                f"a token sequence cannot be negative, got "
                f"({self.window_position_ms}, {self.index})"
            )

    @property
    def key(self) -> tuple[int, int]:
        return self.window_position_ms, self.index


@dataclass(frozen=True, slots=True)
class Timed:
    """The word was placed on the session clock."""

    interval: Interval

    @property
    def is_timed(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class AlignmentUnavailable:
    """The word was heard and could not be placed on the session clock.

    There is no ``interval`` attribute, by the same design as ``Unavailable``:
    reaching for a boundary that does not exist raises a domain error rather
    than returning a ``None`` a caller might coerce to zero, and a zero here
    would put the word at the start of the session.
    """

    reason: UnavailabilityReason = UnavailabilityReason.ALIGNMENT_UNAVAILABLE
    detail: str = ""

    @property
    def is_timed(self) -> bool:
        return False

    def __getattr__(self, name: str) -> object:
        if name in {"interval", "start", "end"}:
            raise FabricatedValue(
                f"cannot read '{name}' from a word with no alignment "
                f"(reason={self.reason.value}); FR-025 forbids substituting a boundary. "
                "The word is in the transcript in its lexical position; where it "
                "sat on the clock is what is missing."
            )
        raise AttributeError(name)


#: A word is placed or it is not. A consumer has to narrow before reading an
#: interval, which is precisely the check that gets skipped when the field is
#: an ``Interval | None``.
type Placement = Timed | AlignmentUnavailable


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
    """§8 ``WordToken``: one literal word hypothesis, ordered and maybe placed.

    The identifier is stable across reconciliation (§7.4). A consumer that
    displayed this token as provisional replaces it in place when the final
    arrives; without a stable id it would have to guess by position, and a
    single inserted word would shift everything after it.
    """

    id: TokenId
    sequence: TokenSequence
    raw_text: str
    placement: Placement
    #: ``Unavailable`` when the recogniser reports no per-word posterior, which
    #: whisper-large-v3 does not. The alternative was a placeholder number, and
    #: a consumer reading one cannot tell it from a score the model actually
    #: emitted - which is the substitution FR-025 exists to prevent. An
    #: unavailable confidence clears no publication threshold, for the same
    #: reason an uncalibrated one does not: there is nothing to compare.
    confidence: Confidence | Unavailable
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

    @property
    def is_timed(self) -> bool:
        return self.placement.is_timed

    @property
    def interval(self) -> Interval:
        """The interval, or a domain error naming the rule.

        A convenience for the temporal paths, which have already narrowed. Code
        that has not narrowed gets ``FabricatedValue`` rather than an
        ``AttributeError`` it might reasonably decide to swallow.
        """
        if isinstance(self.placement, Timed):
            return self.placement.interval
        raise FabricatedValue(
            f"token {self.id.value} has no interval "
            f"(reason={self.placement.reason.value}); narrow on `is_timed` first"
        )

    def finalize(self) -> WordToken:
        """Promote to final. Text and placement are already fixed."""
        if self.is_final:
            return self
        return self._with(status=TokenStatus.FINAL)

    def revise(
        self, raw_text: str, placement: Placement, confidence: Confidence | Unavailable
    ) -> WordToken:
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
        return self._with(raw_text=raw_text, placement=placement, confidence=confidence)

    def _with(
        self,
        *,
        raw_text: str | None = None,
        placement: Placement | None = None,
        confidence: Confidence | Unavailable | None = None,
        status: TokenStatus | None = None,
    ) -> WordToken:
        return WordToken(
            id=self.id,
            sequence=self.sequence,
            raw_text=self.raw_text if raw_text is None else raw_text,
            placement=self.placement if placement is None else placement,
            confidence=self.confidence if confidence is None else confidence,
            status=self.status if status is None else status,
        )
