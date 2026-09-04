"""FR-009 and US-003: pausing does not reset or inflate the session clock.

Both naive implementations of a session clock fail a real presentation, and
both fail silently. A wall-clock difference counts a four-minute microphone fix
as four minutes of silence, inventing an enormous silent pause and destroying
every per-minute rate. A clock that restarts at zero on resume makes the second
half of the timeline collide with the first, so FR-026's single shared timeline
stops being single and the fusion engine starts pairing events that were
minutes apart.

Neither raises an exception. Only these assertions catch them.
"""

from __future__ import annotations

import pytest

from evidence_engine.domain.sessions.clock import SessionClock
from evidence_engine.domain.sessions.session import AnalysisSession
from evidence_engine.domain.sessions.state import SessionState

pytestmark = pytest.mark.invariant


def test_a_pause_does_not_advance_the_clock() -> None:
    clock = SessionClock().start(wall_ms=1_000)

    paused = clock.pause(wall_ms=6_000)

    assert paused.now(wall_ms=6_000).ms == 5_000
    # Four more minutes of real time pass while the microphone is fixed.
    assert paused.now(wall_ms=246_000).ms == 5_000


def test_resuming_continues_from_the_accumulated_position() -> None:
    clock = SessionClock().start(wall_ms=1_000).pause(wall_ms=6_000)

    resumed = clock.start(wall_ms=246_000)

    assert resumed.now(wall_ms=248_000).ms == 7_000


def test_captured_duration_excludes_every_pause() -> None:
    clock = (
        SessionClock()
        .start(wall_ms=0)
        .pause(wall_ms=10_000)
        .start(wall_ms=100_000)
        .pause(wall_ms=115_000)
    )

    assert clock.captured_ms(wall_ms=500_000) == 25_000


def test_the_session_clock_never_goes_backwards_across_a_pause_cycle(
    consented_session: AnalysisSession,
) -> None:
    session = consented_session.begin_capture(wall_ms=1_000)
    before_pause = session.position(wall_ms=11_000).ms

    session = session.pause_capture(wall_ms=11_000)
    during_pause = session.position(wall_ms=300_000).ms

    session = session.resume_capture(wall_ms=300_000)
    after_resume = session.position(wall_ms=302_000).ms

    assert before_pause == 10_000
    assert during_pause == before_pause
    assert after_resume == 12_000


def test_completion_stops_the_clock_before_the_final_pass(
    consented_session: AnalysisSession,
) -> None:
    """Inference time is not captured time, so it must not stretch the denominator."""
    session = consented_session.begin_capture(wall_ms=0).request_completion(wall_ms=60_000)

    assert session.state is SessionState.COMPLETING
    # Eight seconds of final reconciliation later, the duration is unchanged.
    assert session.captured_ms(wall_ms=68_000) == 60_000
