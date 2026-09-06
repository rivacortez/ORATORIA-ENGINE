"""A live session, with the controls a presentation actually needs.

Returned from ``OratoriaEngine.create_stream()`` rather than an async
generator, and the difference is not stylistic. A generator can be advanced and
closed. A presentation has to be **paused** when the speaker stops to answer a
question, **resumed**, **finished** when it ends, and **abandoned** when
something goes wrong - and pause in particular is load-bearing rather than
decorative: §6.1's session clock excludes paused stretches, so a words-per-
minute figure computed across one would be diluted by however long the speaker
was not speaking.

The session is an async context manager. Leaving the block without calling
``finish()`` aborts, because a session that fell out of scope produced no
result and marking its processing run successful would put a half-finished run
in the trail as a completed one.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from evidence_engine.application.errors import BackpressureRequired
from evidence_engine.application.ports.runtimes import AudioWindow, VisualFrame
from evidence_engine.application.ports.streaming import OutboundEvent
from evidence_engine.sdk.errors import StreamAlreadyClosed

if TYPE_CHECKING:  # pragma: no cover - import cycle only at type-check time
    from evidence_engine.sdk.configuration import SessionConfiguration
    from evidence_engine.sdk.engine import AnalysisResult, OratoriaEngine


class StreamSession:
    """One live session against an embedded engine."""

    def __init__(self, engine: OratoriaEngine, configuration: SessionConfiguration) -> None:
        self._engine = engine
        self._configuration = configuration
        self._opened = False
        self._closed = False
        #: Nothing more will ever be published on this stream: `finish()`
        #: returned (or failed) or `abort()` ran. Distinct from `_closed`,
        #: which flips the moment `finish()` STARTS and only governs sending.
        self._settled = False
        self._position_ms = 0
        self._sequence = 0

    async def __aenter__(self) -> StreamSession:
        await self._open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._closed:
            # Left the block without finishing - an exception, a `break`, a
            # `return`. Either way no result was produced, and closing the run
            # as successful would record a half-finished run as a complete one.
            await self.abort()

    async def _open(self) -> None:
        if self._opened:
            return
        (
            self._session_id,
            self._coordinator,
            self._channel,
            self._caller,
        ) = await self._engine._open(self._configuration)
        self._opened = True

    # -- sending ----------------------------------------------------------

    async def send_audio(self, samples: bytes, *, is_final: bool = False) -> bool:
        """Hand one window of PCM to the engine.

        Returns whether the engine accepted it. ``False`` means backpressure:
        the queue is full and the caller is sending faster than the model
        decodes. Returned rather than raised because US-012 makes backpressure
        part of the contract - a client that keeps sending is misbehaving, not
        unlucky - and an exception would push a routine pacing signal onto the
        error path.

        The window's position on the session clock is tracked here, from the
        bytes actually accepted. A caller that had to supply it would have to
        know about pauses.
        """
        await self._open()
        self._require_open()

        duration_ms = self._configuration.duration_ms_for(samples, self._engine_rate())
        window = AudioWindow(
            session_position_ms=self._position_ms,
            duration_ms=duration_ms,
            sample_rate_hz=self._engine_rate(),
            samples=samples,
            is_final_window=is_final,
        )
        try:
            await self._coordinator.ingest_audio(self._sequence, window)
        except BackpressureRequired:
            # The coordinator has already published `backpressure.requested`
            # through the channel, so the caller can see it in `receive()` too.
            return False

        self._sequence += 1
        self._position_ms += duration_ms
        return True

    async def send_visual_features(self, landmarks: tuple[float, ...]) -> bool:
        """Geometry extracted by the caller (ADR-009), never an image.

        The privacy-preserving path is the first-class one: a consumer running
        MediaPipe in a browser or on their own frames sends coordinates, and no
        likeness reaches the engine at all.
        """
        await self._open()
        self._require_open()
        frame = VisualFrame(session_position_ms=self._position_ms, landmarks=landmarks)
        try:
            await self._coordinator.ingest_video(self._sequence, [frame])
        except BackpressureRequired:
            return False
        return True

    # -- receiving --------------------------------------------------------

    async def receive(self) -> OutboundEvent:
        """The next server-to-client message (§7.3), waiting if none is queued.

        Keeps answering after ``finish()`` has started. Finishing can publish
        the last window's events through this same channel, and a consumer
        that reads concurrently with the caller that finishes (OratorIA's
        adapter runs a receiver task beside its feeder) has to be able to drain
        them rather than be thrown out of a session that is still speaking to
        it. It raises only once nothing more can arrive: the stream settled
        (``finish()`` returned or ``abort()`` ran) and the queue is empty.
        ``RemoteStreamSession.receive`` keeps the same words.
        """
        await self._open()
        if self._settled and self._channel.pending() == 0:
            self._raise_closed()
        return await self._channel.next_event()

    def pending(self) -> int:
        """How many events are queued right now, without waiting."""
        return self._channel.pending() if self._opened else 0

    # -- control ----------------------------------------------------------

    async def pause(self) -> None:
        """Stop the session clock. Not the same as sending nothing.

        §6.1 excludes paused stretches from the session clock, so a speaker who
        stops for two minutes to answer a question does not have those two
        minutes counted as silence in their speaking rate.
        """
        await self._open()
        self._require_open()
        await self._engine._capture.pause(self._caller, self._session_id)

    async def resume(self) -> None:
        await self._open()
        self._require_open()
        await self._engine._capture.resume(self._caller, self._session_id)

    async def finish(self) -> AnalysisResult:
        """Close capture, reconcile, assemble, and return the evidence document."""
        await self._open()
        self._require_open()
        self._closed = True
        try:
            return await self._engine._finish(
                self._caller, self._session_id, self._coordinator, self._channel
            )
        finally:
            self._settled = True

    async def abort(self) -> None:
        """Abandon the session without producing a result.

        Idempotent, so a caller that aborts in a `finally` after an explicit
        abort does not raise on the way out of an error path.

        The processing run is closed as **unsuccessful**. The evidence gathered
        so far stays valid - §6.3 keeps a partial failure from invalidating a
        session - but the run says it did not complete, which is what stops a
        half-finished run being read as a whole one.
        """
        if self._closed or not self._opened:
            self._closed = True
            self._settled = True
            return
        self._closed = True
        try:
            await self._engine._abort(self._coordinator)
        finally:
            self._settled = True

    # -- internals --------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            self._raise_closed()

    @staticmethod
    def _raise_closed() -> None:
        raise StreamAlreadyClosed(
            "this stream has been finished or aborted. A session is not "
            "reusable: §6.1 step 9 makes finalized history immutable, so a "
            "second pass would either rewrite it or silently start a new one."
        )

    def _engine_rate(self) -> int:
        return self._engine._configuration.sample_rate_hz
