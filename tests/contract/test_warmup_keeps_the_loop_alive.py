"""`warmup()` loads the model without freezing the event loop it runs on.

The first consumer to embed the engine in a server found the defect the hard
way: `warmup()` called `build_speech_runtime()` inline, the baseline takes
~20 s of blocking CPU and GPU work to load, and the WebSocket the server was
holding open missed every keepalive in the meantime. The peer closed it with
"keepalive ping timeout" before the first window was ever decoded. An
`async def` that blocks for twenty seconds is a synchronous function with a
misleading signature, and the SDK's whole surface is documented as
async-first.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from evidence_engine import EngineConfiguration, OratoriaEngine
from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    DeterministicSpeechRuntime,
    SpeechScript,
)
from evidence_engine.application.ports.runtimes import SpeechRuntime


async def test_a_slow_model_load_does_not_stop_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A builder that blocks for 300 ms; a ticker that must keep ticking."""

    def slow_builder(self: EngineConfiguration) -> SpeechRuntime:
        time.sleep(0.3)  # what loading 3 GB of weights looks like, shortened
        return DeterministicSpeechRuntime(SpeechScript())

    monkeypatch.setattr(EngineConfiguration, "build_speech_runtime", slow_builder)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    heartbeat = asyncio.create_task(ticker())
    try:
        await OratoriaEngine.local(EngineConfiguration()).warmup()
    finally:
        heartbeat.cancel()

    # Inline, the loop is frozen for the whole load and the ticker gets at
    # most one turn. Off the loop, it keeps its 20 ms cadence throughout.
    assert ticks >= 5, ticks
