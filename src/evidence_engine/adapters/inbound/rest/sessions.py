"""The session REST surface (§7.1).

Thin. Each route parses a body, calls one use case and renders the result;
scope checks, consent, idempotency and the state machine all live behind the
use case. That is not tidiness for its own sake - §7.2 lets a client drive the
same session over a WebSocket, and any rule enforced in a route handler would
be absent from the socket path.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Path, status
from fastapi.responses import JSONResponse

from evidence_engine.adapters.inbound.rest.dependencies import (
    CallerDep,
    EngineDep,
    IdempotencyDep,
)
from evidence_engine.adapters.inbound.rest.schemas import (
    CapabilitiesBody,
    CreatedSessionBody,
    CreateSessionBody,
    DeletionReceiptBody,
    DeletionVerificationBody,
    NegotiatedCapabilitiesBody,
    SessionStatusBody,
    UnavailableCapabilityBody,
)
from evidence_engine.adapters.inbound.rest.serialization import render_document
from evidence_engine.application.api import EngineApi
from evidence_engine.application.commands.create_session import (
    CreatedSession,
    CreateSessionCommand,
)
from evidence_engine.domain.sessions.capabilities import (
    CapabilityRequest,
    NegotiatedCapabilities,
)
from evidence_engine.domain.sessions.consent import RetentionPolicy
from evidence_engine.domain.sessions.state import SessionMode
from evidence_engine.domain.shared.identifiers import SessionId
from evidence_engine.domain.shared.provenance import SemanticVersion

router = APIRouter(prefix="/v1", tags=["sessions"])

SessionIdPath = Annotated[str, Path(min_length=1, max_length=128)]


@router.post(
    "/sessions",
    status_code=status.HTTP_201_CREATED,
    response_model=CreatedSessionBody,
    summary="Create a streaming or batch session",
)
async def create_session(
    engine: EngineDep,
    caller: CallerDep,
    idempotency_key: IdempotencyDep,
    body: Annotated[CreateSessionBody, Body()],
) -> JSONResponse:
    policy_version = SemanticVersion.parse(body.consent_policy_version)
    command = CreateSessionCommand(
        mode=SessionMode(body.mode),
        capabilities=CapabilityRequest(
            audio_codec=body.capabilities.audio_codec,
            sample_rate_hz=body.capabilities.sample_rate_hz,
            locale=body.capabilities.locale,
            video_format=body.capabilities.video_format,
            frame_rate_fps=body.capabilities.frame_rate_fps,
        ),
        consent_policy_version=policy_version,
        retention=RetentionPolicy(
            policy_version=policy_version,
            retain_raw_media=body.retention.retain_raw_media,
            raw_media_ttl_seconds=body.retention.raw_media_ttl_seconds,
            retain_derived_aggregates=body.retention.retain_derived_aggregates,
        ),
        idempotency_key=idempotency_key,
    )

    created = await engine.create_session.execute(caller, command)
    payload = _render_created(created, _schema_version(engine))

    # A replayed idempotency key returns 200, not 201: nothing was created this
    # time, and US-011's promise is one session per key rather than one
    # response shape.
    return JSONResponse(
        status_code=status.HTTP_200_OK if created.was_existing else status.HTTP_201_CREATED,
        content=payload,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionStatusBody,
    summary="Read lifecycle and processing status",
)
async def read_session(
    engine: EngineDep, caller: CallerDep, session_id: SessionIdPath
) -> SessionStatusBody:
    session = await engine.read_session.execute(caller, SessionId(session_id))
    return SessionStatusBody(
        schema_version=_schema_version(engine),
        session_id=session.id.value,
        state=session.state.value,
        mode=session.mode.value,
        locale=session.locale,
        created_at_ms=session.created_at_ms,
        completed_at_ms=session.completed_at_ms,
        captured_ms=session.captured_ms(engine.clock.wall_ms()),
        consent_active=session.consent is not None and session.consent.is_active,
    )


@router.get(
    "/sessions/{session_id}/result",
    summary="Retrieve the complete evidence document",
)
async def read_result(
    engine: EngineDep, caller: CallerDep, session_id: SessionIdPath
) -> dict[str, Any]:
    document = await engine.read_result.execute(caller, SessionId(session_id))
    return render_document(document, schema_version=_schema_version(engine))


@router.delete(
    "/sessions/{session_id}/evidence",
    response_model=DeletionReceiptBody,
    summary="Delete protected evidence and raw media",
)
async def delete_evidence(
    engine: EngineDep, caller: CallerDep, session_id: SessionIdPath
) -> DeletionReceiptBody:
    receipt = await engine.delete_evidence.execute(caller, SessionId(session_id))
    return DeletionReceiptBody(
        schema_version=_schema_version(engine),
        session_id=receipt.session_id.value,
        media_objects_deleted=receipt.media_objects_deleted,
        evidence_records_deleted=receipt.evidence_records_deleted,
        already_deleted=receipt.was_already_deleted,
    )


@router.get(
    "/sessions/{session_id}/evidence/verification",
    response_model=DeletionVerificationBody,
    summary="Check what, if anything, survived a deletion",
)
async def verify_deletion(
    engine: EngineDep, caller: CallerDep, session_id: SessionIdPath
) -> DeletionVerificationBody:
    """The read-only counterpart to ``DELETE .../evidence`` (QA-04).

    A separate route, not a field on the deletion response. The consent policy
    makes verification a call that does not share a code path with deletion,
    because a check performed by the code that just deleted reports on its own
    actions; a route the deletion handler cannot reach is how that separation
    survives the next person to add a convenience.

    Residue is a 200 carrying ``deletion_verified: false``, not a 409. The
    status describes what happened to the request, and the request succeeded:
    it asked a question and got an answer. A 409 would make a monitor read "the
    verification ran and found surviving evidence" as "the verification failed"
    - opposite conclusions - and put a client into a retry loop against a
    finding that retrying cannot change.
    """
    verification = await engine.delete_evidence.verify(caller, SessionId(session_id))
    return DeletionVerificationBody(
        schema_version=_schema_version(engine),
        session_id=verification.session_id.value,
        deletion_verified=verification.is_clean,
        media_objects_remaining=verification.media_objects_remaining,
        evidence_document_present=verification.evidence_document_present,
        stream_state_present=verification.stream_state_present,
        session_marked_deleted=verification.session_marked_deleted,
        audit_record_present=verification.audit_record_present,
    )


@router.get(
    "/capabilities",
    response_model=CapabilitiesBody,
    summary="List codecs, languages, taxonomies and active schema versions",
)
async def read_capabilities(engine: EngineDep, caller: CallerDep) -> CapabilitiesBody:
    # Authenticated but unscoped: any valid key may ask what the engine
    # accepts. Requiring a scope here would make integration testing need
    # production-shaped credentials for a question with no data in the answer.
    del caller
    capabilities = engine.read_capabilities.execute()
    return CapabilitiesBody(
        schema_version=str(capabilities.schema_version),
        taxonomy_version=str(capabilities.taxonomy_version),
        audio_codecs=list(capabilities.audio_codecs),
        sample_rates_hz=list(capabilities.sample_rates_hz),
        video_formats=list(capabilities.video_formats),
        frame_rate_range_fps=capabilities.frame_rate_range_fps,
        locales=list(capabilities.locales),
        speech_event_types=list(capabilities.speech_event_types),
        visual_event_types=list(capabilities.visual_event_types),
        emitted_speech_event_types=list(capabilities.emitted_speech_event_types),
        emitted_visual_event_types=list(capabilities.emitted_visual_event_types),
        emitted_prosodic_indicators=list(capabilities.emitted_prosodic_indicators),
        unavailable_capabilities=[
            UnavailableCapabilityBody(
                schema_version=str(capabilities.schema_version),
                kind=absent.kind,
                name=absent.name,
                reason=absent.reason.value,
                detail=absent.detail,
            )
            for absent in capabilities.unavailable_capabilities
        ],
    )


def _render_created(created: CreatedSession, schema_version: str) -> dict[str, Any]:
    body = CreatedSessionBody(
        schema_version=schema_version,
        session_id=created.session.id.value,
        state=created.session.state.value,
        mode=created.session.mode.value,
        capabilities=_render_capabilities(created.session.capabilities, schema_version),
        configuration_id=created.session.configuration.value,
        stream_token=created.stream_token,
        stream_token_expires_at_ms=created.stream_token_expires_at_ms,
    )
    return body.model_dump()


def _render_capabilities(
    capabilities: NegotiatedCapabilities, schema_version: str
) -> NegotiatedCapabilitiesBody:
    return NegotiatedCapabilitiesBody(
        schema_version=schema_version,
        audio_codec=capabilities.audio_codec.value,
        sample_rate_hz=capabilities.sample_rate_hz,
        locale=capabilities.locale,
        video_format=capabilities.video_format.value if capabilities.video_format else None,
        frame_rate_fps=capabilities.frame_rate_fps,
        lossy_audio_warning=capabilities.requires_quality_warning,
    )


def _schema_version(engine: EngineApi) -> str:
    return str(engine.read_capabilities.execute().schema_version)
