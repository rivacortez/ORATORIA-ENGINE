"""The verbatim transcript and its partial/final reconciliation.

This is where FR-016 ("expose provisional and finalized transcript events")
meets §6.1 step 9 ("stable windows are reconciled and emitted as final without
rewriting already finalized history"). The two together define a specific
discipline: the tail of the transcript may churn freely while the model is
still hearing context, and everything behind the finalized frontier is frozen.

**There are two frontiers, because there are two orders.**

Ordering is lexical - by ``TokenSequence`` - and every token has one. Placement
is temporal, and a token whose alignment failed has none. Folding the two
together is what forced the old adapter to delete unplaceable words: the
transcript could only hold what it could sort by time.

So the temporal frontier (``finalized_time_frontier``) is computed from timed
tokens alone and governs everything that reads the clock, and the lexical
frontier (``stable_through_sequence``) governs promotion to final. An unplaced
token is finalized when the sequence frontier passes it - never by borrowing
the stability of a neighbour, which would assert that this word is settled
because a *different* word is.

The split runs through the query surface too. ``raw_text`` and
``context_around_sequence`` are lexical and return everything. ``within_time``
is temporal and returns only what it can honestly place; callers that measure
silence or fuse across modalities use it and declare the limitation rather than
quietly working on a shorter transcript.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from evidence_engine.domain.shared.timeline import MonotonicTime
from evidence_engine.domain.transcript.tokens import (
    TokenSequence,
    TokenStatus,
    TranscriptViolation,
    WordToken,
)


@dataclass(frozen=True, slots=True)
class Transcript:
    """An ordered, immutable set of word hypotheses with two frontiers.

    Ordering is by ``TokenSequence``, and the token id breaks ties
    deterministically. Determinism here is not cosmetic: NFR-015 requires
    equivalent output from equivalent input, and a set iteration order would
    break that on the first rerun.
    """

    tokens: tuple[WordToken, ...] = field(default_factory=tuple)

    # -- lexical: every token, placed or not ------------------------------

    def raw_text(self) -> str:
        """The literal transcript, finalized and provisional, in lexical order.

        Joined with single spaces and nothing else. No capitalization, no
        punctuation repair, no collapsing of repeated words: FR-011 forbids
        grammatical cleanup, and a repeated word collapsed here is a REPETITION
        event that can never be detected downstream.

        **Includes tokens whose alignment failed.** This is the authoritative
        verbatim source, and a word omitted from it because the aligner could
        not place it would be deleted from the record for a reason that has
        nothing to do with what the speaker said.
        """
        return " ".join(token.raw_text for token in self.tokens)

    def context_around_sequence(
        self, centre: TokenSequence, *, before: int = 5, after: int = 5
    ) -> tuple[WordToken, ...]:
        """The lexical neighbourhood of a token, unplaced words included.

        FR-013 decides a lexical filler's role from what surrounds it, and what
        surrounds it is a matter of what was *said*, not of what could be
        timestamped. A context window that skipped unplaced words would hand
        the classifier a sentence with holes in it and no indication that the
        holes were there.
        """
        ordered = self.tokens
        for position, token in enumerate(ordered):
            if token.sequence == centre:
                return ordered[max(0, position - before) : position + after + 1]
        return ()

    @property
    def unaligned_count(self) -> int:
        """How many words are in the transcript without a position on the clock.

        Published rather than derivable-if-you-look, because it is the number
        that tells a reader how much of the temporal analysis is running on
        less than the whole transcript.
        """
        return sum(1 for token in self.tokens if not token.is_timed)

    # -- temporal: timed tokens only --------------------------------------

    def timed_tokens(self) -> tuple[WordToken, ...]:
        """Only the tokens that can honestly be placed on the clock."""
        return tuple(token for token in self.tokens if token.is_timed)

    def within_time(self, start: MonotonicTime, end: MonotonicTime) -> tuple[WordToken, ...]:
        """Timed tokens whose interval starts inside ``[start, end)``.

        Unplaced tokens are absent and that is deliberate: they belong to no
        window, and putting them in one would be choosing a window for them.
        Silent-pause derivation and multimodal fusion read this and must say
        that they ran on ``len(tokens) - unaligned_count`` words.
        """
        return tuple(
            token for token in self.timed_tokens() if start.ms <= token.interval.start.ms < end.ms
        )

    @property
    def finalized_time_frontier(self) -> MonotonicTime:
        """The point on the clock up to which the transcript is frozen.

        Computed from finalized **timed** tokens only. An unplaced token
        contributes nothing here - it has no end to contribute - and letting
        one move the frontier would freeze a stretch of audio on the strength
        of a word that was never located in it.

        Zero when nothing timed has been finalized, which is the correct answer
        for a session that has just started: everything is still revisable.
        """
        ends = [t.interval.end.ms for t in self.tokens if t.is_final and t.is_timed]
        return MonotonicTime(max(ends, default=0))

    # -- status -----------------------------------------------------------

    @property
    def final_tokens(self) -> tuple[WordToken, ...]:
        return tuple(t for t in self.tokens if t.is_final)

    @property
    def provisional_tokens(self) -> tuple[WordToken, ...]:
        return tuple(t for t in self.tokens if not t.is_final)

    def with_provisional(self, incoming: Iterable[WordToken]) -> Transcript:
        """Replace the revisable tail with a new hypothesis.

        Tokens at or after the frontier are discarded and replaced wholesale,
        because a streaming recognizer re-decodes its whole active window rather
        than editing individual words; merging token by token would invent an
        alignment the model never asserted.

        **Two refusals, because there are two ways to rewrite the past.**

        A token at or behind the finalized *lexical* frontier is trying to
        replace a word already frozen. A *timed* token starting behind the
        finalized *time* frontier is trying to re-describe audio already
        frozen, even when its sequence is new - the recogniser asserting a word
        at 200 ms while 0-400 ms is settled contradicts the settled stretch.

        Neither check subsumes the other. Time alone leaves every unplaced
        token permanently rewritable, a hole that opens only on the words the
        aligner already struggled with. Sequence alone lets a new word
        contradict frozen audio. So both run.
        """
        kept = tuple(t for t in self.tokens if t.is_final)
        lexical = _last_final_sequence(kept)
        temporal = self.finalized_time_frontier
        fresh = tuple(sorted(incoming, key=_ordering))

        for token in fresh:
            if lexical is not None and token.sequence <= lexical:
                raise TranscriptViolation(
                    f"provisional token {token.id} at sequence {token.sequence.key} is at "
                    f"or behind the finalized lexical frontier {lexical.key}"
                )
            if token.is_timed and token.interval.start.ms < temporal.ms:
                raise TranscriptViolation(
                    f"provisional token {token.id} starts at {token.interval.start.ms} ms, "
                    f"behind the finalized frontier at {temporal.ms} ms"
                )
        return Transcript(tuple(sorted(kept + fresh, key=_ordering)))

    def finalize_through_time(self, boundary: MonotonicTime) -> Transcript:
        """Freeze every **timed** token that ends at or before ``boundary``.

        Called when the streaming coordinator decides a window is stable. The
        boundary is a session-clock position rather than a token count because
        stability is a property of the audio - how much right context the model
        has heard - not of how many words happened to fall in it.

        Unplaced tokens are untouched here. They are finalized by
        ``finalize_through_sequence``, which the coordinator calls with the
        sequence frontier the same stability decision implies.
        """
        if boundary.ms < self.finalized_time_frontier.ms:
            raise TranscriptViolation(
                f"cannot move the finalized time frontier backwards: "
                f"{self.finalized_time_frontier.ms} ms -> {boundary.ms} ms"
            )
        return Transcript(
            tuple(
                token.finalize()
                if token.is_timed and token.interval.end.ms <= boundary.ms
                else token
                for token in self.tokens
            )
        )

    def finalize_through_sequence(self, boundary: TokenSequence) -> Transcript:
        """Freeze every token at or before ``boundary`` in lexical order.

        This is how an unplaced word becomes final. The boundary is stated by
        the coordinator, which knows how much audio it has committed; it is
        never inferred from a neighbour. Inferring it would mean asserting that
        an unplaced word is settled because the *next* word is settled, and the
        next word being stable says nothing about whether the recogniser will
        revise this one on its next pass.
        """
        return Transcript(
            tuple(
                token.finalize() if token.sequence <= boundary else token for token in self.tokens
            )
        )

    def finalize_remaining(self) -> Transcript:
        """Freeze everything left, at session close.

        A word still provisional when the session ends is not going to be
        revised - there is no more audio and no more passes. It is finalized as
        recognised text, keeping whatever placement it has, so a session that
        ended with an unplaced tail publishes that tail rather than dropping it
        for never having reached a frontier.
        """
        return Transcript(tuple(token.finalize() for token in self.tokens))

    def count(self, status: TokenStatus | None = None) -> int:
        if status is None:
            return len(self.tokens)
        return sum(1 for token in self.tokens if token.status is status)


def _ordering(token: WordToken) -> tuple[int, int, str]:
    return token.sequence.window_position_ms, token.sequence.index, token.id.value


def _last_final_sequence(finalized: Sequence[WordToken]) -> TokenSequence | None:
    if not finalized:
        return None
    return max(token.sequence for token in finalized)


def build(tokens: Sequence[WordToken]) -> Transcript:
    """Build a transcript from a sequence, imposing the canonical order."""
    return Transcript(tuple(sorted(tokens, key=_ordering)))
