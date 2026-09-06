"""``finish()`` under a station that is slow or gone - both found live.

Two behaviours, one session each:

* The stream died before ``session.completed``. The server never received
  ``session.complete``, so there is no result and there will not be one;
  polling ``/result`` for the whole bound would only collect 409s. ``finish()``
  says so at once instead.
* Reconciliation is still running and the 409 names ``retry_after_seconds``.
  The server's estimate is believed over this client's own backoff.
"""

from __future__ import annotations

import pytest

from evidence_engine.sdk import client as client_module
from evidence_engine.sdk.client import OratoriaClient
from evidence_engine.sdk.errors import RemoteEngineUnavailable

from .test_receive_outlives_finish import _MINIMAL_DOCUMENT, _frame
from .test_remote_client import (
    SILENT_CHUNK,
    _accepted_frame,
    _created_session_response,
    _FakeTransport,
)

pytestmark = pytest.mark.contract


async def test_finish_refuses_to_poll_when_the_stream_died_before_completion() -> None:
    transport = _FakeTransport(
        post_response=_created_session_response(),
        get_responses={"/result": [(409, {"code": "result_not_ready"})]},
        ws_frames=[_accepted_frame()],  # then the socket is gone: no completion ever arrives
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)
    stream = client.create_stream()
    assert await stream.send_audio(SILENT_CHUNK, is_final=True) is True

    with pytest.raises(RemoteEngineUnavailable, match="before the engine confirmed"):
        await stream.finish()

    assert transport.get_calls == [], "nothing to fetch: the server never got session.complete"


async def test_finish_honours_retry_after_seconds_from_a_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def _record_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(client_module.asyncio, "sleep", _record_sleep)
    transport = _FakeTransport(
        post_response=_created_session_response(),
        get_responses={
            "/result": [
                (409, {"code": "result_not_ready", "retry_after_seconds": 3}),
                (200, _MINIMAL_DOCUMENT),
            ]
        },
        ws_frames=[
            _accepted_frame(),
            _frame(
                2,
                "session.completed",
                {"run_id": "run-0001", "finalized_through_ms": 1_000, "captured_ms": 1_000},
            ),
        ],
    )
    client = OratoriaClient("http://engine.local", "test-key", transport=transport)
    stream = client.create_stream()
    assert await stream.send_audio(SILENT_CHUNK, is_final=True) is True

    result = await stream.finish()

    assert result.finalized_through_ms == 1_000
    assert slept == [3.0], "the server's estimate, not the client's 0.25 s backoff"
    assert len(transport.get_calls) == 2
