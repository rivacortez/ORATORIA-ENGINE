"""The verbatim transcript and its partial/final reconciliation.

This is where FR-016 ("expose provisional and finalized transcript events")
meets §6.1 step 9 ("stable windows are reconciled and emitted as final without
rewriting already finalized history"). The two together define a specific
discipline: the tail of the transcript may churn freely while the model is
still hearing context, and everything behind the finalized frontier is frozen.

The frontier is what makes the guarantee checkable. Rather than trusting each
call site to look before it writes, the transcript tracks the position past
which nothing may change and refuses any revision that reaches behind it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from evidence_engine.domain.shared.timeline import MonotonicTime
from evidence_engine.domain.transcript.tokens import TokenStatus, TranscriptViolation, WordToken


@dataclass(frozen=True, slots=True)
class Transcript:
    """An ordered, immutable set of word hypotheses with a finalized frontier.

    Ordering is by interval start. Two tokens may share a start - overlapping
    hypotheses happen at window seams - so ordering is stable rather than
    strict, and the token id breaks ties deterministically. Determinism here is
    not cosmetic: NFR-015 requires equivalent output from equivalent input, and
    a set iteration order would break that on the first rerun.
    """

    tokens: tuple[WordToken, ...] = field(default_factory=tuple)

    @property
    def finalized_frontier(self) -> MonotonicTime:
        """The point up to which the transcript is frozen.

        Zero when nothing has been finalized yet, which is the correct answer
        for a session that has just started: everything is still revisable.
        """
        finalized_ends = [t.interval.end.ms for t in self.tokens if t.is_final]
        return MonotonicTime(max(finalized_ends, default=0))

    @property
    def final_tokens(self) -> tuple[WordToken, ...]:
        return tuple(t for t in self.tokens if t.is_final)

    @property
    def provisional_tokens(self) -> tuple[WordToken, ...]:
        return tuple(t for t in self.tokens if not t.is_final)

    def raw_text(self) -> str:
        """The literal transcript, finalized and provisional, in order.

        Joined with single spaces and nothing else. No capitalization, no
        punctuation repair, no collapsing of repeated words: FR-011 forbids
        grammatical cleanup, and a repeated word collapsed here is a REPETITION
        event that can never be detected downstream.
        """
        return " ".join(token.raw_text for token in self.tokens)

    def with_provisional(self, incoming: Iterable[WordToken]) -> Transcript:
        """Replace the revisable tail with a new hypothesis.

        Tokens at or after the frontier are discarded and replaced wholesale,
        because a streaming recognizer re-decodes its whole active window rather
        than editing individual words; merging token by token would invent an
        alignment the model never asserted.
        """
        frontier = self.finalized_frontier
        kept = tuple(t for t in self.tokens if t.is_final)
        fresh = tuple(sorted(incoming, key=_ordering))

        for token in fresh:
            if token.interval.start.ms < frontier.ms:
                raise TranscriptViolation(
                    f"provisional token {token.id} starts at {token.interval.start.ms} ms, "
                    f"behind the finalized frontier at {frontier.ms} ms"
                )
        return Transcript(tuple(sorted(kept + fresh, key=_ordering)))

    def finalize_through(self, boundary: MonotonicTime) -> Transcript:
        """Freeze every token that ends at or before ``boundary``.

        Called when the streaming coordinator decides a window is stable. The
        boundary is a session-clock position rather than a token count because
        stability is a property of the audio - how much right context the model
        has heard - not of how many words happened to fall in it.
        """
        if boundary.ms < self.finalized_frontier.ms:
            raise TranscriptViolation(
                f"cannot move the finalized frontier backwards: "
                f"{self.finalized_frontier.ms} ms -> {boundary.ms} ms"
            )
        promoted = tuple(
            token.finalize() if token.interval.end.ms <= boundary.ms else token
            for token in self.tokens
        )
        return Transcript(promoted)

    def within(self, start: MonotonicTime, end: MonotonicTime) -> tuple[WordToken, ...]:
        """Tokens whose interval starts inside ``[start, end)``.

        The window for contextual classification: FR-013 decides a lexical
        filler's role from what surrounds it, and this is how the classifier
        asks for that surrounding.
        """
        return tuple(token for token in self.tokens if start.ms <= token.interval.start.ms < end.ms)

    def count(self, status: TokenStatus | None = None) -> int:
        if status is None:
            return len(self.tokens)
        return sum(1 for token in self.tokens if token.status is status)


def _ordering(token: WordToken) -> tuple[int, int, str]:
    return token.interval.start.ms, token.interval.end.ms, token.id.value


def build(tokens: Sequence[WordToken]) -> Transcript:
    """Build a transcript from a sequence, imposing the canonical order."""
    return Transcript(tuple(sorted(tokens, key=_ordering)))
