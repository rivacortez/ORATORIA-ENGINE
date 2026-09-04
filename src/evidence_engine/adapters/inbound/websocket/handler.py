"""The streaming endpoint: ``WS /v1/sessions/{session_id}/stream`` (§6.1).

The handler owns the socket and nothing else. It verifies the stream grant,
takes the session lease, translates wire messages into use-case calls and
translates results back. Every rule about what may happen next lives behind
those calls, because §7.1 exposes the same session over REST and a rule
enforced here would be missing there.

Three decisions worth stating.

*The grant is the identity.* It arrives signed, carrying the tenant and a
reduced scope set, so this handler never looks a session up without a tenant.
The alternative - resolving a bare session id - would put the one unscoped
query in the system on the path that handles unauthenticated input.

*The lease is released in a ``finally``.* Two handlers advancing one chunk
ledger would interleave sequence numbers and manufacture gaps FR-008 would then
report as data loss. A lease leaked on an exception would lock the session out
for its whole TTL, which to a student mid-presentation is indistinguishable
from an outage.

*A protocol error does not close the socket.* §7.3 lists ``error`` as
non-fatal. One malformed frame from a buggy client should cost that frame, not
the presentation being given.
"""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from evidence_engine.adapters.inbound.rest.dependencies import EngineDep
from evidence_engine.adapters.inbound.websocket.channel import WebSocketEventChannel
from evidence_engine.adapters.inbound.websocket.protocol import (
    AudioChunk,
    ClientMessageType,
    ControlMessage,
    ProtocolError,
    UnknownMessage,
    VideoFramePayload,
    parse_client_message,
)
from evidence_engine.application.api import EngineApi
from evidence_engine.application.errors import ApplicationError, BackpressureRequired
from evidence_engine.application.ports.platform import (
    AuthenticatedCaller,
    ConfigurationSnapshot,
)
from evidence_engine.application.ports.runtimes import AudioWindow, VisualFrame
from evidence_engine.application.ports.tokens import StreamGrant
from evidence_engine.application.services.speech_assembly import SpeechAssembler
from evidence_engine.application.services.visual_assembly import VisualAssembler
from evidence_engine.application.workflows.streaming import (
    StreamingCoordinator,
    StreamingState,
)
from evidence_engine.domain.evidence.ledger import EvidenceLedger
from evidence_engine.domain.shared.identifiers import RunId, SessionId

router = APIRouter(tags=["stream"])

#: WebSocket "policy violation". The socket opened and the credential was then
#: refused, which is what this code means - not a transport failure.
CLOSE_POLICY_VIOLATION = 1008


@router.websocket("/v1/sessions/{session_id}/stream")
async def stream(
    socket: WebSocket,
    engine: EngineDep,
    session_id: str,
    token: str = Query(..., description="short-lived stream token from POST /v1/sessions"),
) -> None:
    schema_version = str(engine.read_capabilities.execute().schema_version)
    await socket.accept()
    channel = WebSocketEventChannel(socket, schema_version)

    grant = engine.tokens.verify(token, engine.clock.epoch_ms())
    if grant is None or grant.session_id.value != session_id:
        # One response for expired, forged and mismatched: telling them apart
        # tells a prober which half of their guess was right.
        await channel.send_error(
            SessionId(session_id), "invalid_stream_token", "the stream token is not valid"
        )
        await socket.close(code=CLOSE_POLICY_VIOLATION)
        return

    caller = _caller_from(grant)

    if not await engine.stream_state.acquire_lease(
        grant.session_id, engine.profile.stream_lease_ttl_seconds
    ):
        await channel.send_error(
            grant.session_id,
            "session_busy",
            "another connection is already processing this session",
        )
        await socket.close(code=CLOSE_POLICY_VIOLATION)
        return

    try:
        await _run(engine, socket, channel, caller, grant.session_id)
    finally:
        await engine.stream_state.release_lease(grant.session_id)


def _caller_from(grant: StreamGrant) -> AuthenticatedCaller:
    """The identity the grant stands for.

    Reconstructed from signed claims rather than looked up, so FR-004's
    attribution survives without the API key ever reaching the browser.
    """
    return AuthenticatedCaller(
        application=grant.application,
        tenant=grant.tenant,
        key_id=grant.key_id,
        scopes=grant.scopes,
        trace_id=f"ws_{grant.session_id.value}",
    )


async def _run(
    engine: EngineApi,
    socket: WebSocket,
    channel: WebSocketEventChannel,
    caller: AuthenticatedCaller,
    session_id: SessionId,
) -> None:
    configuration = await engine.configuration.current(caller.tenant)
    run_id = RunId.generate()

    coordinator = StreamingCoordinator(
        state=StreamingState(run_id=run_id, session_id=session_id),
        configuration=configuration,
        speech=engine.speech,
        vision=engine.vision,
        channel=channel,
        telemetry=engine.telemetry,
        ledger=EvidenceLedger(run_id=run_id),
        speech_assembler=SpeechAssembler(run_id, configuration, engine.calibrator),
        visual_assembler=VisualAssembler(run_id, configuration, engine.calibrator),
        max_queue_depth=engine.profile.max_queue_depth,
    )

    try:
        await engine.capture_control.begin(caller, session_id)
    except ApplicationError as error:
        # Consent withdrawn, session already completed, wrong tenant: all
        # legitimate refusals, none of them a reason to drop the socket without
        # saying why.
        await channel.send_error(session_id, "cannot_begin_capture", str(error))
        await socket.close(code=CLOSE_POLICY_VIOLATION)
        return

    await channel.send_accepted(
        session_id,
        {
            "run_id": run_id.value,
            "configuration_id": configuration.id.value,
            "taxonomy_version": str(configuration.taxonomy_version),
            "max_queue_depth": engine.profile.max_queue_depth,
        },
    )

    try:
        while True:
            raw = await socket.receive_json()
            keep_open = await _dispatch(
                engine, coordinator, channel, caller, session_id, configuration, raw
            )
            if not keep_open:
                break
    except WebSocketDisconnect:
        # A dropped connection is not a failed session. §6.3 keeps a partial
        # failure from invalidating a session, and the transport is no
        # different: the evidence gathered so far stays valid.
        engine.telemetry.counter("stream.disconnected")
    finally:
        with contextlib.suppress(RuntimeError):
            await socket.close()


async def _dispatch(
    engine: EngineApi,
    coordinator: StreamingCoordinator,
    channel: WebSocketEventChannel,
    caller: AuthenticatedCaller,
    session_id: SessionId,
    configuration: ConfigurationSnapshot,
    raw: dict[str, Any],
) -> bool:
    """Handle one message. Returns False when the session should close."""
    try:
        message = parse_client_message(raw)
    except ProtocolError as error:
        await channel.send_error(session_id, "protocol_error", str(error))
        return True

    if isinstance(message, UnknownMessage):
        # §7.4: unknown values must not crash a consumer, and a client built
        # against a later minor version is not an error. Dropping the socket
        # here would make every forward-compatible rollout a breaking change.
        engine.telemetry.counter("stream.unknown_message", type=message.raw_type)
        return True

    if isinstance(message, AudioChunk):
        await _ingest_audio(coordinator, message)
        return True

    if isinstance(message, VideoFramePayload):
        await _ingest_video(coordinator, message)
        return True

    return await _control(engine, coordinator, channel, caller, session_id, configuration, message)


async def _ingest_audio(coordinator: StreamingCoordinator, message: AudioChunk) -> None:
    window = AudioWindow(
        session_position_ms=message.envelope.monotonic_time_ms,
        duration_ms=message.duration_ms,
        sample_rate_hz=message.sample_rate_hz,
        samples=message.samples,
        is_final_window=message.is_final,
    )
    with contextlib.suppress(BackpressureRequired):
        # The coordinator has already published backpressure.requested through
        # the channel, so there is nothing further to tell the client here.
        await coordinator.ingest_audio(message.sequence, window)


async def _ingest_video(coordinator: StreamingCoordinator, message: VideoFramePayload) -> None:
    frame = VisualFrame(
        session_position_ms=message.session_position_ms,
        pixels=message.pixels,
        landmarks=message.landmarks,
    )
    with contextlib.suppress(BackpressureRequired):
        await coordinator.ingest_video(message.sequence, [frame])


async def _control(
    engine: EngineApi,
    coordinator: StreamingCoordinator,
    channel: WebSocketEventChannel,
    caller: AuthenticatedCaller,
    session_id: SessionId,
    configuration: ConfigurationSnapshot,
    message: ControlMessage,
) -> bool:
    if message.type is ClientMessageType.CLIENT_HEARTBEAT:
        return True

    if message.type is ClientMessageType.CAPTURE_PAUSE:
        await engine.capture_control.pause(caller, session_id)
        return True

    if message.type is ClientMessageType.CAPTURE_RESUME:
        await engine.capture_control.resume(caller, session_id)
        return True

    if message.type is ClientMessageType.SESSION_COMPLETE:
        completed = await engine.complete_session.execute(
            caller, session_id, coordinator.state, configuration
        )
        await channel.send_completed(
            session_id,
            {
                "run_id": completed.document.run_id.value,
                "speech_events": len(completed.document.speech_events),
                "visual_events": len(completed.document.visual_events),
                "cooccurrences": len(completed.document.cooccurrences),
                # Restated at the close of every session (§17): the consumer
                # never has to infer that no ranking is coming.
                "ranking_authority": completed.document.ranking_authority,
            },
        )
        return False

    # `session.configure` is accepted and acknowledged by silence: capabilities
    # were already negotiated and frozen at session creation (FR-006), and
    # honouring a mid-stream change would let a client alter the terms the
    # evidence is being produced under.
    return True
