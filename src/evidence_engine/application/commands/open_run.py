"""Open a processing run (§8 ``ProcessingRun``).

This use case exists because the real database found its absence. Evidence rows
hang off a run, the streaming coordinator was minting a ``RunId`` and stamping
it onto every event, and nothing ever wrote the run itself. The in-memory
adapter has no foreign keys, so the whole contract suite passed while
production would have inserted evidence pointing at a row that did not exist.

The lesson is worth recording next to the fix: an in-memory adapter that omits
referential integrity omits a class of bug, and the integration suite is what
catches it. That is what the integration suite is *for*, and why the two
adapter sets are not redundant.

A run is opened when processing starts and closed when it finishes, so §8's
separation holds: a session reprocessed under a promoted model produces a
second run whose evidence is comparable with the first rather than overwriting
it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from evidence_engine.application.errors import NotAuthorized, SessionNotFound
from evidence_engine.application.ports.clock import Clock
from evidence_engine.application.ports.platform import AuthenticatedCaller, Scope
from evidence_engine.application.ports.repositories import (
    ProcessingRun,
    RunRepository,
    RunState,
    SessionRepository,
)
from evidence_engine.domain.shared.identifiers import (
    ModelVersionId,
    RunId,
    SessionId,
)
from evidence_engine.domain.shared.provenance import (
    ModelRole,
    SemanticVersion,
)


class OpenProcessingRun:
    """Use case: begin a pass of the pipeline over a session's media."""

    def __init__(
        self,
        sessions: SessionRepository,
        runs: RunRepository,
        clock: Clock,
        pipeline_version: SemanticVersion,
        contributions: Mapping[ModelRole, ModelVersionId] | None = None,
    ) -> None:
        self._sessions = sessions
        self._runs = runs
        self._clock = clock
        self._pipeline_version = pipeline_version
        #: What is wired, by role, recorded on every run at open. The
        #: document's manifest says what *contributed*; this says what was
        #: *running*, so a run that fails on its first window still names
        #: the model it attempted (NFR-019 cannot resume a run it cannot
        #: describe).
        self._contributions: dict[ModelRole, ModelVersionId] = dict(contributions or {})

    async def execute(self, caller: AuthenticatedCaller, session_id: SessionId) -> ProcessingRun:
        if not caller.allows(Scope.SESSIONS_WRITE):
            raise NotAuthorized(f"opening a run requires {Scope.SESSIONS_WRITE.value}")

        # The session is checked first, and within the caller's tenant, so a run
        # cannot be opened against a session the caller cannot see.
        session = await self._sessions.get(caller.tenant, session_id)
        if session is None:
            raise SessionNotFound(f"session {session_id} not found")

        run = ProcessingRun(
            id=RunId.generate(),
            session_id=session_id,
            pipeline_version=self._pipeline_version,
            state=RunState.RUNNING,
            started_at_ms=self._clock.epoch_ms(),
            models=self._contributions,
        )
        await self._runs.add(run)
        return run


class CloseProcessingRun:
    """Use case: mark a run finished, successfully or not.

    Separate from ``CompleteSession`` because a run can fail while its session
    survives - §6.3 keeps a partial failure from invalidating a session, and a
    run that ended in ``FAILED`` still has to be closed so the next one is
    distinguishable from it.
    """

    def __init__(self, runs: RunRepository, clock: Clock) -> None:
        self._runs = runs
        self._clock = clock

    async def execute(self, run_id: RunId, *, succeeded: bool = True) -> None:
        run = await self._runs.get(run_id)
        if run is None:
            # Nothing to close. Not an error: the run may have been deleted
            # along with its session's evidence between the two calls.
            return

        await self._runs.save(
            replace(
                run,
                state=RunState.COMPLETED if succeeded else RunState.FAILED,
                completed_at_ms=self._clock.epoch_ms(),
            )
        )
