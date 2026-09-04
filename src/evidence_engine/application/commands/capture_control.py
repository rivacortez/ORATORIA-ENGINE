"""Capture lifecycle: begin, pause, resume (FR-009, §7.2).

Thin by design. Each of these loads the session, asks the aggregate to make the
transition and saves the result; the rules about which transitions are legal,
whether consent is still live and what happens to the clock all live in the
domain. That split is deliberate: §7.2 lets a client send ``capture.pause`` over
the socket and §7.1 has a REST path reaching the same session, and any rule
implemented in one handler would be missing from the other.

The one thing this layer adds is the wall-clock reading. The domain never takes
one itself (QA-05), so somebody has to hand it in, and that somebody is here.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.application.errors import NotAuthorized, SessionNotFound
from evidence_engine.application.ports.clock import Clock
from evidence_engine.application.ports.platform import AuthenticatedCaller, Scope
from evidence_engine.application.ports.repositories import SessionRepository
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.shared.identifiers import SessionId


@dataclass(frozen=True, slots=True)
class _CaptureContext:
    """Resolved session plus the clock reading its transition will use."""

    session: AnalysisSession
    wall_ms: int


class CaptureControl:
    """Use cases: start, suspend and resume media capture for a session."""

    def __init__(self, sessions: SessionRepository, clock: Clock) -> None:
        self._sessions = sessions
        self._clock = clock

    async def begin(self, caller: AuthenticatedCaller, session_id: SessionId) -> AnalysisSession:
        context = await self._resolve(caller, session_id)
        # `begin_capture` re-checks consent before moving. FR-031's gate is on
        # the aggregate rather than here so the WebSocket path cannot bypass it.
        updated = context.session.begin_capture(context.wall_ms)
        await self._sessions.save(updated)
        return updated

    async def pause(self, caller: AuthenticatedCaller, session_id: SessionId) -> AnalysisSession:
        context = await self._resolve(caller, session_id)
        updated = context.session.pause_capture(context.wall_ms)
        await self._sessions.save(updated)
        return updated

    async def resume(self, caller: AuthenticatedCaller, session_id: SessionId) -> AnalysisSession:
        context = await self._resolve(caller, session_id)
        # Consent is checked again here, not only at `begin`. A participant who
        # withdraws mid-session must not have capture resumed on them, and the
        # resume path is the one an interrupted presentation actually takes.
        updated = context.session.resume_capture(context.wall_ms)
        await self._sessions.save(updated)
        return updated

    async def _resolve(self, caller: AuthenticatedCaller, session_id: SessionId) -> _CaptureContext:
        if not caller.allows(Scope.SESSIONS_WRITE):
            raise NotAuthorized(f"capture control requires {Scope.SESSIONS_WRITE.value}")
        session = await self._sessions.get(caller.tenant, session_id)
        if session is None:
            raise SessionNotFound(f"session {session_id} not found")
        return _CaptureContext(session=session, wall_ms=self._clock.wall_ms())
