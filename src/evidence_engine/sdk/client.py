"""``OratoriaClient`` - the engine, reached over the network instead of run in-process.

The pilot topology this exists for: the engine runs as a private service on a
GPU workstation, and OratorIA's own backend runs on CPU infrastructure and
consumes it remotely, over HTTP and WebSocket. ADR-011 named this client
"deferred" and said what it would have to satisfy when it existed - the same
two structural protocols the embedded path satisfies - and that is the whole
design constraint here. OratorIA's adapter depends on an engine with
``warmup``, ``create_stream``, ``aclose`` and ``contributions``, and a session
with ``send_audio``, ``receive``, ``pending``, ``finish`` and ``abort``.
``OratoriaClient`` and ``RemoteStreamSession`` satisfy exactly that surface, so
the same adapter that drives an embedded ``OratoriaEngine`` drives this
unchanged - the conformance test in ``tests/contract/test_remote_client.py``
is what proves it.

**Why the transport is injectable.** Nothing here should have to open a real
socket to be tested, and a consumer who already runs httpx or websockets under
another wrapper should not be forced to carry a second copy for this one
client. ``Transport`` is the seam: three methods, none of them opinionated
about retries, connection pooling or TLS configuration - that is the
consumer's business, not this module's.

**The backpressure-without-loss rule, restated for a client with no positive
ack.** The wire only ever tells you a chunk was *refused*
(``backpressure.requested``, naming the ``chunk_seq``); nothing ever confirms
one was *accepted*, and a refusal can name ANY chunk still outstanding, not
only the one most recently sent - the reader task and ``send_audio`` run
concurrently, so a refusal for an earlier chunk can still be in flight after
later ones were already sent. So this keeps every not-yet-confirmed chunk's
bytes, keyed by its own ``chunk_seq``, bounded to the server's own advertised
window (``session.accepted``'s ``max_queue_depth``, or
``DEFAULT_IN_FLIGHT_WINDOW`` if an older server did not name one). A refusal
resends exactly the bytes for the ``chunk_seq`` it named, under that same
sequence number, and returns ``False`` - the caller retries the same window,
precisely as the embedded engine's own contract already promises. If the
named ``chunk_seq`` has already aged out of the retained window, this raises
``RemoteChunkLost`` rather than resending different bytes under a sequence
number that no longer describes them - a wrong resend would look like
backpressure handled correctly while quietly corrupting the transcript's
timing.

**The residual risk this leaves, named rather than hidden.** A refusal for the
very last chunk can arrive *after* the caller has already moved on to
``finish()``. ``finish()`` resends whatever is still marked refused before it
sends ``session.complete``, which closes that window in the common case - but
if the refusal itself is still in flight and has not yet been read off the
socket, it will be missed. This is a client that cannot outrun the network
it runs on; a positive per-chunk acknowledgement would close the gap and does
not exist on this wire today.

**What this deliberately does not carry.** ``Evidence`` has no
``cooccurrences`` and no ``quality`` field. The wire renders both, and adding
them now would be inventing scope this change was not asked to cover; a
future change that wants them extends ``evidence_from_json`` and ``Evidence``
together; see ``sdk.results`` for exactly where.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast

from evidence_engine._extras import require
from evidence_engine.application.ports.streaming import OutboundEvent, ServerMessageType
from evidence_engine.domain.shared.identifiers import ModelVersionId, SessionId
from evidence_engine.domain.shared.provenance import ModelRole
from evidence_engine.sdk.configuration import EngineConfiguration, SessionConfiguration
from evidence_engine.sdk.engine import AnalysisResult
from evidence_engine.sdk.errors import RemoteChunkLost, RemoteEngineUnavailable, StreamAlreadyClosed
from evidence_engine.sdk.results import evidence_from_json

#: How long `finish()` polls `GET /result` for before giving up. §6.1 step 10
#: runs a reconciliation pass after `session.complete`; this is how long a
#: client waits for it to land, not a guess at how long it usually takes.
RESULT_POLL_BOUND_SECONDS = 10.0
#: How long `abort()` waits for `session.aborted` before closing anyway. An
#: abort that never gets acknowledged should not hang a caller who is already
#: trying to leave.
ABORT_WAIT_BOUND_SECONDS = 10.0
#: How many not-yet-confirmed chunks this client keeps bytes for when
#: `session.accepted` did not name a `max_queue_depth` - an older server, or a
#: transport substituting its own accepted frame in a test. Matches
#: `Settings.max_queue_depth`'s own default (`bootstrap/settings.py`); a
#: same-version server names its own bound instead, and that value governs
#: once `session.accepted` has been read.
DEFAULT_IN_FLIGHT_WINDOW = 8


class WsConnection(Protocol):
    """One open WebSocket connection, as far as this client needs to see it."""

    async def send_json(self, data: Mapping[str, object]) -> None: ...

    async def receive_json(self) -> dict[str, object]: ...

    async def close(self) -> None: ...


class Transport(Protocol):
    """What ``OratoriaClient`` needs from an HTTP and WebSocket library.

    Narrow on purpose. Retries, pooling, timeouts and TLS are transport
    concerns a consumer already has an opinion about; this seam exists so
    ``OratoriaClient`` never has one of its own.
    """

    async def post_json(
        self, url: str, *, json: Mapping[str, object], headers: Mapping[str, str]
    ) -> tuple[int, dict[str, object]]:
        """POST a JSON body, returning the status code and the parsed response."""
        ...

    async def get_json(
        self, url: str, *, headers: Mapping[str, str]
    ) -> tuple[int, dict[str, object]]:
        """GET a resource, returning the status code and the parsed response."""
        ...

    async def ws_connect(self, url: str) -> WsConnection:
        """Open a WebSocket connection to ``url``."""
        ...


class _WebsocketsConnection:
    """Adapts a ``websockets`` client connection to ``WsConnection``."""

    def __init__(self, connection: object) -> None:
        self._connection = connection

    async def send_json(self, data: Mapping[str, object]) -> None:
        import json

        await self._connection.send(json.dumps(dict(data)))  # type: ignore[attr-defined]

    async def receive_json(self) -> dict[str, object]:
        import json

        raw = await self._connection.recv()  # type: ignore[attr-defined]
        parsed = json.loads(raw)
        return cast(dict[str, object], parsed)

    async def close(self) -> None:
        await self._connection.close()  # type: ignore[attr-defined]


class HttpxTransport:
    """The default transport: ``httpx`` for REST, ``websockets`` for the stream.

    Both imports are lazy, behind ``evidence_engine._extras.require``, so
    importing this module - and therefore ``evidence_engine.sdk.client`` and
    ``OratoriaClient`` - costs nothing on a core-only install. Only
    *constructing* one of these (the default a consumer gets when they do not
    pass their own ``transport``) needs the ``client`` extra installed.
    """

    def __init__(self) -> None:
        require(
            "client",
            "httpx",
            "websockets",
            because="OratoriaClient's default transport needs an HTTP and a WebSocket client",
        )
        import httpx

        self._client = httpx.AsyncClient()

    async def post_json(
        self, url: str, *, json: Mapping[str, object], headers: Mapping[str, str]
    ) -> tuple[int, dict[str, object]]:
        response = await self._client.post(url, json=dict(json), headers=dict(headers))
        return response.status_code, cast(dict[str, object], response.json())

    async def get_json(
        self, url: str, *, headers: Mapping[str, str]
    ) -> tuple[int, dict[str, object]]:
        response = await self._client.get(url, headers=dict(headers))
        return response.status_code, cast(dict[str, object], response.json())

    async def ws_connect(self, url: str) -> WsConnection:
        import websockets

        connection = await websockets.connect(url)
        return _WebsocketsConnection(connection)

    async def aclose(self) -> None:
        await self._client.aclose()


def _as_ws_url(base_url: str, session_id: str, token: str) -> str:
    """Translate the client's http(s) base URL into the stream's ws(s) one."""
    scheme, _, rest = base_url.partition("://")
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}://{rest}/v1/sessions/{session_id}/stream?token={token}"


def _event_from_frame(frame: Mapping[str, object]) -> OutboundEvent:
    """The live-wire twin of ``sdk.results.evidence_from_json``: one frame in."""
    payload = frame.get("payload", {})
    return OutboundEvent(
        type=ServerMessageType(str(frame["type"])),
        session_id=SessionId(str(frame["session_id"])),
        monotonic_time_ms=int(cast(int, frame["monotonic_time_ms"])),
        payload=cast(Mapping[str, object], payload if isinstance(payload, Mapping) else {}),
    )


class OratoriaClient:
    """Speaks to a hosted engine over HTTP and WebSocket.

    Constructing one does not reach the network. ``warmup()`` is the one
    thing that does, for the same reason ``OratoriaEngine.warmup()`` is a
    separate call from construction (ADR-011): a client nobody has confirmed
    can reach a ready engine is not something a caller should be able to
    mistake for one that can.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        configuration: EngineConfiguration | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._configuration = configuration or EngineConfiguration()
        self._transport = transport or HttpxTransport()
        self._contributions: dict[ModelRole, ModelVersionId] = {}
        self._instance: Mapping[str, object] | None = None
        self._warmed = False

    @property
    def contributions(self) -> Mapping[ModelRole, ModelVersionId]:
        """Models by role, learned at ``warmup()`` from ``/v1/capabilities``."""
        return self._contributions

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def warmup(self) -> None:
        """Confirm the remote engine is ready. NEVER falls back to anything.

        Two checks against the already-running service - there is no *load*
        step here the way there is in ``OratoriaEngine.warmup()``: the model
        was loaded and warmed by the server's own startup (``bootstrap.app``'s
        lifespan) before this process ever connected. This confirms that,
        rather than repeating it.

        Both ``models`` and ``instance`` must be present in ``/v1/capabilities``,
        not merely well-shaped when present. A response missing either is what
        an older or misconfigured service looks like - the same one this check
        exists to keep a caller from mistaking for ready during a rolling
        upgrade of the pilot's GPU workstations.
        """
        ready_url = f"{self._base_url}/health/ready"
        status, body = await self._transport.get_json(ready_url, headers=self._headers())
        if status != 200 or body.get("status") != "ready":
            raise RemoteEngineUnavailable(
                f"{ready_url} is not ready: HTTP {status}, status={body.get('status')!r}, "
                f"checks={body.get('checks')!r}"
            )

        capabilities_url = f"{self._base_url}/v1/capabilities"
        status, capabilities = await self._transport.get_json(
            capabilities_url, headers=self._headers()
        )
        if status != 200:
            raise RemoteEngineUnavailable(
                f"{capabilities_url} returned HTTP {status}: {capabilities!r}"
            )

        models = capabilities.get("models")
        if not isinstance(models, Mapping):
            raise RemoteEngineUnavailable(
                f"{capabilities_url} did not name its models: {capabilities!r}. An engine "
                "able to confirm readiness always names what it loaded; its absence means "
                "this response came from an older or misconfigured service, not one this "
                "client can treat as ready."
            )
        instance = capabilities.get("instance")
        if not isinstance(instance, Mapping):
            raise RemoteEngineUnavailable(
                f"{capabilities_url} did not name its instance: {capabilities!r}. A pilot "
                "can run more than one GPU workstation behind the same consuming "
                "application, and a client that cannot tell them apart must not be told "
                "it is ready to proceed."
            )

        self._contributions = {
            ModelRole(str(role)): ModelVersionId(str(version)) for role, version in models.items()
        }
        self._instance = instance
        self._warmed = True

    def create_stream(
        self, configuration: SessionConfiguration | None = None
    ) -> RemoteStreamSession:
        """Open a live session against the remote engine.

        Returns a session, not a generator, for the same reason
        ``OratoriaEngine.create_stream()`` does (ADR-011): a presentation is
        paused and resumed, not merely advanced and closed.
        """
        return RemoteStreamSession(self, configuration or SessionConfiguration())

    async def aclose(self) -> None:
        """Release the transport, if it is one this client built itself.

        A ``transport`` the caller supplied is theirs to close; closing it
        here would take a resource away from someone still holding a
        reference to it.
        """
        aclose = getattr(self._transport, "aclose", None)
        if aclose is not None:
            await aclose()


@dataclass(frozen=True, slots=True)
class _SentChunk:
    """Everything needed to resend one ``audio.chunk`` byte-for-byte."""

    chunk_seq: int
    samples_b64: str
    sample_rate_hz: int
    duration_ms: int
    monotonic_time_ms: int
    is_final: bool


@dataclass(slots=True)
class _StreamState:
    """Mutable per-session bookkeeping, kept off ``RemoteStreamSession`` itself
    only to keep that class's method bodies short enough to read in one pass.
    """

    opened: bool = False
    closed: bool = False
    sequence: int = 0
    position_ms: int = 0
    session_id: str | None = None
    schema_version: str = "1.0.0"
    #: Every not-yet-confirmed chunk's bytes, keyed by its own `chunk_seq` -
    #: see the module docstring's "backpressure-without-loss rule, restated
    #: for a client with no positive ack". A plain `dict` preserves insertion
    #: order, which is what makes evicting `next(iter(...))` remove the
    #: OLDEST entry rather than an arbitrary one.
    in_flight: dict[int, _SentChunk] = field(default_factory=dict)
    #: `chunk_seq`s refused by the server, oldest first - the order
    #: `backpressure.requested` named them, and the order a resend answers
    #: them in. Appended to by the reader task; drained by
    #: `send_audio`/`finish` one at a time.
    refused_chunk_seqs: deque[int] = field(default_factory=deque)
    #: The server's own advertised bound on outstanding chunks, learned from
    #: `session.accepted`'s `max_queue_depth`. `None` until accepted, in which
    #: case `DEFAULT_IN_FLIGHT_WINDOW` stands in for it.
    accepted_window: int | None = None
    #: Which physical instance accepted this session (`session.accepted`'s
    #: `instance_id`). `None` until accepted.
    instance_id: str | None = None
    backpressure_signals: int = 0
    completion_payload: dict[str, object] | None = None
    events: asyncio.Queue[OutboundEvent] = field(default_factory=asyncio.Queue)
    reader_task: asyncio.Task[None] | None = None


class RemoteStreamSession:
    """One live session against a hosted engine, over HTTP and WebSocket.

    Opens lazily on the first ``send_audio`` - a caller that builds one and
    never sends anything should not have created a session on the server for
    nothing, the same reasoning the embedded ``StreamSession`` already applies.
    """

    def __init__(self, client: OratoriaClient, configuration: SessionConfiguration) -> None:
        self._client = client
        self._configuration = configuration
        self._state = _StreamState()

    async def _open(self) -> None:
        state = self._state
        if state.opened:
            return

        body: dict[str, object] = {
            "mode": "realtime",
            "capabilities": {
                "audio_codec": self._configuration.audio_codec,
                "sample_rate_hz": self._client._configuration.sample_rate_hz,
                "locale": self._configuration.locale,
            },
            "consent_policy_version": self._configuration.consent_policy_version,
            "retention": {
                "retain_raw_media": self._configuration.retain_raw_media,
                "raw_media_ttl_seconds": self._configuration.raw_media_ttl_seconds,
            },
        }
        status, created = await self._client._transport.post_json(
            f"{self._client._base_url}/v1/sessions", json=body, headers=self._client._headers()
        )
        if status not in (200, 201):
            raise RemoteEngineUnavailable(
                f"POST {self._client._base_url}/v1/sessions returned HTTP {status}: {created}"
            )

        state.session_id = str(created["session_id"])
        state.schema_version = str(created["schema_version"])
        token = str(created["stream_token"])

        ws = await self._client._transport.ws_connect(
            _as_ws_url(self._client._base_url, state.session_id, token)
        )
        self._ws = ws
        # `session.accepted` is consumed here, not surfaced through `receive()`
        # - the embedded session has no equivalent message at all, and parity
        # with it is the point (see the module docstring). Its payload names
        # two things this client remembers for the rest of the session: which
        # instance accepted it, and how large a window of chunks it may keep
        # sent bytes for.
        accepted = await ws.receive_json()
        accepted_payload = accepted.get("payload", {})
        if isinstance(accepted_payload, Mapping):
            instance_id = accepted_payload.get("instance_id")
            state.instance_id = instance_id if isinstance(instance_id, str) else None
            max_queue_depth = accepted_payload.get("max_queue_depth")
            state.accepted_window = max_queue_depth if isinstance(max_queue_depth, int) else None
        state.reader_task = asyncio.create_task(self._read_loop())
        state.opened = True

    async def _read_loop(self) -> None:
        state = self._state
        while True:
            try:
                frame = await self._ws.receive_json()
            except Exception:
                return

            frame_type = str(frame.get("type", ""))
            if frame_type == ServerMessageType.BACKPRESSURE_REQUESTED.value:
                payload = frame.get("payload", {})
                chunk_seq = payload.get("chunk_seq") if isinstance(payload, Mapping) else None
                if isinstance(chunk_seq, int):
                    state.refused_chunk_seqs.append(chunk_seq)
                    state.backpressure_signals += 1

            try:
                event = _event_from_frame(frame)
            except ValueError:
                # §7.4: an unrecognized server message type must not crash a
                # consumer - a client built against an older minor version
                # will eventually meet one it has never heard of.
                continue
            await state.events.put(event)

            if frame_type in (
                ServerMessageType.SESSION_COMPLETED.value,
                ServerMessageType.SESSION_ABORTED.value,
            ):
                payload = frame.get("payload")
                state.completion_payload = dict(payload) if isinstance(payload, Mapping) else {}
                return

    def _require_open(self) -> None:
        if self._state.closed:
            raise StreamAlreadyClosed(
                "this stream has been finished or aborted. A session is not "
                "reusable: §6.1 step 9 makes finalized history immutable, so a "
                "second pass would either rewrite it or silently start a new one."
            )

    def _envelope(
        self, *, monotonic_time_ms: int, chunk_seq: int | None = None
    ) -> dict[str, object]:
        envelope: dict[str, object] = {
            "schema_version": self._state.schema_version,
            "session_id": self._state.session_id,
            "message_id": f"msg_{uuid.uuid4().hex}",
            "monotonic_time_ms": monotonic_time_ms,
        }
        if chunk_seq is not None:
            envelope["chunk_seq"] = chunk_seq
        return envelope

    async def _send_wire_chunk(self, chunk: _SentChunk) -> None:
        await self._ws.send_json(
            {
                **self._envelope(
                    monotonic_time_ms=chunk.monotonic_time_ms, chunk_seq=chunk.chunk_seq
                ),
                "type": "audio.chunk",
                "samples": chunk.samples_b64,
                "sample_rate_hz": chunk.sample_rate_hz,
                "duration_ms": chunk.duration_ms,
                "is_final": chunk.is_final,
            }
        )

    async def _resend_refused(self) -> None:
        """Resend the oldest still-refused chunk under its own ``chunk_seq``.

        Raises ``RemoteChunkLost`` if this client no longer holds its bytes -
        the retained window (``accepted_window``, or
        ``DEFAULT_IN_FLIGHT_WINDOW`` before ``session.accepted`` has named one)
        was exceeded since the refusal arrived. Resending different bytes
        under that sequence number would corrupt the transcript's timing
        rather than admit the chunk is gone, which is why this raises instead.
        """
        state = self._state
        refused_seq = state.refused_chunk_seqs.popleft()
        resend = state.in_flight.get(refused_seq)
        if resend is None:
            window = state.accepted_window or DEFAULT_IN_FLIGHT_WINDOW
            raise RemoteChunkLost(
                f"chunk_seq={refused_seq} was refused by the server but this client no "
                f"longer retains its bytes (retained window is {window} chunks). Its "
                "audio was produced and is now lost."
            )
        await self._send_wire_chunk(resend)

    def _remember_sent(self, chunk: _SentChunk) -> None:
        """Keep ``chunk``'s bytes, evicting the oldest once the window is full."""
        state = self._state
        state.in_flight[chunk.chunk_seq] = chunk
        window = state.accepted_window or DEFAULT_IN_FLIGHT_WINDOW
        while len(state.in_flight) > window:
            oldest_seq = next(iter(state.in_flight))
            del state.in_flight[oldest_seq]

    # -- sending ------------------------------------------------------------

    async def send_audio(self, samples: bytes, *, is_final: bool = False) -> bool:
        """Hand one window of PCM to the remote engine.

        If a previous chunk was refused, this call spends its round trip
        resending *that* chunk under its original ``chunk_seq`` and returns
        ``False`` without touching ``samples`` at all - the caller's window is
        untouched and has to be offered again on the next call, exactly as a
        ``False`` from the embedded engine's ``send_audio`` means.
        """
        await self._open()
        self._require_open()
        state = self._state

        if state.refused_chunk_seqs:
            await self._resend_refused()
            return False

        duration_ms = self._configuration.duration_ms_for(
            samples, self._client._configuration.sample_rate_hz
        )
        chunk = _SentChunk(
            chunk_seq=state.sequence,
            samples_b64=base64.b64encode(samples).decode("ascii"),
            sample_rate_hz=self._client._configuration.sample_rate_hz,
            duration_ms=duration_ms,
            monotonic_time_ms=state.position_ms,
            is_final=is_final,
        )
        await self._send_wire_chunk(chunk)
        self._remember_sent(chunk)
        state.sequence += 1
        state.position_ms += duration_ms
        return True

    # -- receiving ------------------------------------------------------------

    async def receive(self) -> OutboundEvent:
        """The next server-to-client message, waiting if none is queued yet.

        Keeps answering after ``finish()`` has closed the stream for sending.
        The server publishes the final window's events and then
        ``session.completed`` while ``finish()`` is in flight, and a consumer
        draining them concurrently (OratorIA's adapter runs a receiver task
        beside its feeder) must not be thrown out of a session the server is
        still speaking to - found live, with the last window transcribed on the
        server and discarded here. It raises only once nothing more can
        arrive: the reader task has ended (completion, abort, or a dead
        socket) and every queued event has been handed over. Same words as
        the embedded ``StreamSession.receive``.
        """
        await self._open()
        state = self._state
        reader = state.reader_task
        if reader is not None and reader.done() and state.events.empty():
            self._require_open()  # raises when closed ...
            raise StreamAlreadyClosed(  # ... and when the socket died first
                "nothing more can arrive on this stream: the connection to the engine "
                "has ended and every queued event has already been handed over."
            )
        return await state.events.get()

    def pending(self) -> int:
        """How many events are queued right now, without waiting."""
        return self._state.events.qsize() if self._state.opened else 0

    @property
    def instance_id(self) -> str | None:
        """Which physical instance accepted this session (`session.accepted`).

        ``None`` until the session has opened - a caller that reads it before
        the first ``send_audio`` sees the same "not yet known" the wire itself
        has not reported yet, rather than a stale value.
        """
        return self._state.instance_id

    # -- finishing ------------------------------------------------------------

    async def finish(self) -> AnalysisResult:
        """Close capture, reconcile on the server, and fetch the result.

        Order matters. Any still-refused chunk is resent *before*
        ``session.complete`` - the last chance to land it while the socket is
        still open for audio - then the reader task is awaited so
        ``session.completed`` has already been recorded before this reads it,
        then the result is polled: §6.1 step 10's reconciliation pass runs
        after completion is signalled on the socket, not before, so
        ``GET /result`` answers 409 until it has finished.
        """
        await self._open()
        self._require_open()
        state = self._state
        state.closed = True

        while state.refused_chunk_seqs:
            await self._resend_refused()

        await self._ws.send_json(
            {**self._envelope(monotonic_time_ms=state.position_ms), "type": "session.complete"}
        )

        if state.reader_task is not None:
            await state.reader_task
        await self._ws.close()

        document = await self._poll_result()
        completion = state.completion_payload or {}
        return AnalysisResult(
            evidence=evidence_from_json(document),
            backpressure_signals=state.backpressure_signals,
            finalized_through_ms=_optional_int(completion, "finalized_through_ms"),
            captured_ms=_optional_int(completion, "captured_ms"),
        )

    async def _poll_result(self) -> dict[str, object]:
        assert self._state.session_id is not None  # `_open()` set it; states the invariant
        url = f"{self._client._base_url}/v1/sessions/{self._state.session_id}/result"
        deadline = time.monotonic() + RESULT_POLL_BOUND_SECONDS
        backoff_seconds = 0.25
        while True:
            status, body = await self._client._transport.get_json(
                url, headers=self._client._headers()
            )
            if status == 200:
                return body
            if status != 409 or time.monotonic() >= deadline:
                raise RemoteEngineUnavailable(f"{url} returned HTTP {status}: {body}")
            await asyncio.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 1.0)

    # -- aborting ------------------------------------------------------------

    async def abort(self) -> None:
        """Abandon the session without waiting for a result. Idempotent.

        Bounded: an abort that never gets ``session.aborted`` back should not
        hang a caller who is already trying to leave.
        """
        state = self._state
        if state.closed or not state.opened:
            state.closed = True
            return
        state.closed = True

        await self._ws.send_json(
            {**self._envelope(monotonic_time_ms=state.position_ms), "type": "session.abort"}
        )
        if state.reader_task is not None:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(state.reader_task, timeout=ABORT_WAIT_BOUND_SECONDS)
        await self._ws.close()


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) else None
