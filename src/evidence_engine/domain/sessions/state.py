"""The session lifecycle, as an explicit state machine.

FR-009 and §6.1 describe a capture that can be paused, resumed and completed,
and §6.3 describes failures that must not take the whole session with them.
Written as scattered boolean flags - ``is_started``, ``is_paused``, ``is_done``
- those rules produce states nobody designed: paused and completed at once,
completed twice, a delete that half-applies. The machine below makes the
illegal combinations unrepresentable and gives every transition one place to be
audited.

Terminal means terminal. ``EVIDENCE_DELETED`` accepts no outgoing transition
because FR-032 and QA-04 promise a caller that deletion is final; a state that
could be left would make that promise conditional.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from evidence_engine.domain.shared.errors import IllegalSessionTransition


class SessionMode(StrEnum):
    """Real-time streaming (§6.1) or batch upload (§6.2)."""

    REALTIME = "realtime"
    BATCH = "batch"


class SessionState(StrEnum):
    """Where a session is in its life."""

    #: Created through the API; capabilities negotiated, nothing captured yet.
    CREATED = "created"
    #: Media is arriving and being processed.
    CAPTURING = "capturing"
    #: Capture suspended by the client. The session clock keeps its accumulated
    #: duration; §6.1 and US-003 both require resuming not to reset it.
    PAUSED = "paused"
    #: Capture is closed and the final reconciliation pass is running.
    COMPLETING = "completing"
    #: The evidence document is available for retrieval.
    COMPLETED = "completed"
    #: Processing could not produce a valid result at all. Note that a single
    #: modality failing does NOT land here - QA-02 requires the session to
    #: continue on the surviving modality.
    FAILED = "failed"
    #: Protected evidence and raw media removed under FR-032. Terminal.
    EVIDENCE_DELETED = "evidence_deleted"


_ALLOWED: Mapping[SessionState, frozenset[SessionState]] = MappingProxyType(
    {
        SessionState.CREATED: frozenset(
            {SessionState.CAPTURING, SessionState.FAILED, SessionState.EVIDENCE_DELETED}
        ),
        SessionState.CAPTURING: frozenset(
            {SessionState.PAUSED, SessionState.COMPLETING, SessionState.FAILED}
        ),
        SessionState.PAUSED: frozenset(
            {SessionState.CAPTURING, SessionState.COMPLETING, SessionState.FAILED}
        ),
        SessionState.COMPLETING: frozenset({SessionState.COMPLETED, SessionState.FAILED}),
        # A completed or failed session still holds media and evidence, so both
        # remain deletable. Nothing else follows them.
        SessionState.COMPLETED: frozenset({SessionState.EVIDENCE_DELETED}),
        SessionState.FAILED: frozenset({SessionState.EVIDENCE_DELETED}),
        SessionState.EVIDENCE_DELETED: frozenset(),
    }
)


def can_transition(source: SessionState, target: SessionState) -> bool:
    return target in _ALLOWED[source]


def require_transition(source: SessionState, target: SessionState) -> SessionState:
    """Return ``target`` when the move is legal, otherwise refuse.

    Refusing rather than ignoring is deliberate: a silently dropped transition
    leaves the caller believing capture resumed when it did not, and the next
    thing they hear is an empty result.
    """
    if not can_transition(source, target):
        raise IllegalSessionTransition(
            f"a session in '{source.value}' cannot move to '{target.value}'"
        )
    return target


def is_terminal(state: SessionState) -> bool:
    return not _ALLOWED[state]


def accepts_media(state: SessionState) -> bool:
    """Whether media sent right now would be processed rather than discarded.

    Used by the streaming coordinator to reject chunks with an explicit error
    instead of accepting them into a session that will never emit them.
    """
    return state is SessionState.CAPTURING
