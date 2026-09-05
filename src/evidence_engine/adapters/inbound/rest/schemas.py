"""Wire schemas for the REST surface (§7.1).

Pydantic lives here and nowhere deeper. The domain expresses its invariants in
its own constructors (contract C2), so these models validate *shape* - is this
field an integer, is this enum a member - and hand the result to a domain
constructor that decides whether the value is legal. Duplicating a domain rule
in a validator would create two places for it to be true, and they would drift.

Every response carries ``schema_version``. §7.4 requires it, and NFR-017 makes
it the only way a consumer can tell whether it is still inside the major
version it was written against.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evidence_engine.domain.evidence.document import EvidenceDocument
from evidence_engine.domain.sessions.capabilities import SUPPORTED_LOCALES, AudioCodec


class WireModel(BaseModel):
    """Base for everything on the wire.

    ``extra="forbid"`` on requests: a typo in a field name must be an error,
    not a silently ignored setting. A misspelled ``sample_rate_hz`` that is
    quietly dropped becomes a session negotiated at the wrong rate, discovered
    only when the disfluency metrics look strange.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class CapabilityRequestBody(WireModel):
    """What the client says it can send (FR-006)."""

    audio_codec: str = Field(
        description=(
            "One of "
            + ", ".join(f"`{codec.value}`" for codec in AudioCodec)
            + ". `pcm16` is the only lossless option; the others are accepted "
            "because browsers produce them, and a session using one carries a "
            "standing quality warning - lossy compression discards the "
            "high-frequency detail some disfluency classes are decided from."
        ),
    )
    sample_rate_hz: int = Field(
        gt=0,
        description=(
            "16 kHz is the working rate; higher rates are resampled down. Below "
            "16 kHz is refused rather than upsampled, because upsampling cannot "
            "restore detail that was never captured."
        ),
    )
    locale: str = Field(
        description=(
            "Only " + ", ".join(f"`{tag}`" for tag in sorted(SUPPORTED_LOCALES)) + ". "
            "Other locales are refused rather than served by a model whose error "
            "rates on them have not been measured (§11.1)."
        ),
    )
    video_format: str | None = Field(
        default=None,
        description=(
            "Omit for audio only. `landmarks` sends geometry the client extracted "
            "locally, so no image ever reaches the service - the privacy-preserving "
            "option of ADR-009, and a first-class format rather than a fallback."
        ),
    )
    frame_rate_fps: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Required when a video format is set. Below 5 fps the periodicity of "
            "repetitive movement cannot be estimated, so it is refused."
        ),
    )


class RetentionBody(WireModel):
    """The retention policy this capture runs under (FR-031, NFR-011)."""

    retain_raw_media: bool = False
    raw_media_ttl_seconds: int = Field(default=0, ge=0)
    retain_derived_aggregates: bool = True


#: What the docs page pre-fills when a reader presses "Try it out".
#:
#: Derived from the domain enums rather than typed out. A hand-written example
#: is a second place the accepted values are declared, and it is the one that
#: goes stale - while being the first thing anybody actually sends. Without it
#: Swagger offers `"audio_codec": "string"`, the call comes back 400
#: `unsupported_capability`, and a reader's first impression of the API is that
#: it rejects its own documentation.
#:
#: `pcm16` and 16 kHz rather than a browser-friendly pair: it is the only
#: lossless codec and the rate the ASR front end works at, so the example is
#: also the configuration that carries no standing quality warning.
CREATE_SESSION_EXAMPLE: dict[str, Any] = {
    "mode": "realtime",
    "capabilities": {
        "audio_codec": AudioCodec.PCM16.value,
        "sample_rate_hz": 16_000,
        "locale": sorted(SUPPORTED_LOCALES)[0],
    },
    "consent_policy_version": "1.0.0",
}


class CreateSessionBody(WireModel):
    """``POST /v1/sessions``."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"examples": [CREATE_SESSION_EXAMPLE]},
    )

    mode: str = Field(
        pattern="^(realtime|batch)$",
        description=(
            "`realtime` streams over the WebSocket while the presentation happens; "
            "`batch` processes an upload after it. The evidence is the same shape; "
            "only when it arrives differs."
        ),
    )
    capabilities: CapabilityRequestBody
    consent_policy_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description=(
            "The consent policy version the speaker agreed to. Recorded on the "
            "session so a later export can prove which terms the recording was "
            "made under (§14.4); it is not validated against a policy registry."
        ),
    )
    retention: RetentionBody = RetentionBody()


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class Envelope(BaseModel):
    """Common header on every response (§7.4)."""

    model_config = ConfigDict(frozen=True)

    schema_version: str


class NegotiatedCapabilitiesBody(Envelope):
    """What the engine agreed to accept."""

    audio_codec: str
    sample_rate_hz: int
    locale: str
    video_format: str | None
    frame_rate_fps: int | None
    #: Set when the agreed audio path can erase the acoustic detail some P0
    #: classes are decided from. Surfaced rather than logged, because the client
    #: is the only party that can still change the codec.
    lossy_audio_warning: bool = False


class CreatedSessionBody(Envelope):
    """``POST /v1/sessions`` response."""

    session_id: str
    state: str
    mode: str
    capabilities: NegotiatedCapabilitiesBody
    configuration_id: str
    stream_token: str
    stream_token_expires_at_ms: int


class SessionStatusBody(Envelope):
    """``GET /v1/sessions/{id}``."""

    session_id: str
    state: str
    mode: str
    locale: str
    created_at_ms: int
    completed_at_ms: int | None
    captured_ms: int
    consent_active: bool


class DeletionReceiptBody(Envelope):
    """``DELETE /v1/sessions/{id}/evidence``.

    Returns counts because US-010 wants audit records that prove execution
    without retaining what was deleted. ``already_deleted`` makes the
    idempotent repeat (US-014) visible rather than indistinguishable from a
    first success.
    """

    session_id: str
    media_objects_deleted: int
    evidence_records_deleted: int
    already_deleted: bool


class DeletionVerificationBody(Envelope):
    """``GET /v1/sessions/{id}/evidence/verification``.

    One field per store the deletion writes to, and ``deletion_verified`` is
    exactly their conjunction - nothing else feeds it. A consumer can therefore
    recompute the summary from the fields beside it, which is the property that
    stops the summary attesting to more than was read.

    Published as fields rather than as a bare boolean because the four ways
    QA-04 fails have four different remedies, and an operator told only
    ``false`` would go looking in the logs - the one place NFR-018 keeps this
    content out of.

    ``media_objects_remaining`` and ``evidence_document_present`` say what was
    counted, not what was concluded. Outstanding signed URLs and evidence
    bundles behind an unpublished run have no read-only port to ask; see
    ``application.commands.delete_evidence`` for which of those the object
    count covers by proxy and which it does not.
    """

    session_id: str
    deletion_verified: bool
    media_objects_remaining: int
    evidence_document_present: bool
    stream_state_present: bool
    session_marked_deleted: bool
    audit_record_present: bool


class CapabilitiesBody(Envelope):
    """``GET /v1/capabilities``."""

    taxonomy_version: str
    audio_codecs: list[str]
    sample_rates_hz: list[int]
    video_formats: list[str]
    frame_rate_range_fps: tuple[int, int]
    locales: list[str]
    speech_event_types: list[str]
    visual_event_types: list[str]
    #: §17's contract invariant, published. A consumer reading `"none"` knows
    #: no ranking will ever appear here and builds its own.
    ranking_authority: str = EvidenceDocument.ranking_authority


class ErrorBody(Envelope):
    """A stable error shape (§7.4).

    ``code`` is machine-readable and stable across versions; ``message`` is for
    a human reading a log. Consumers branch on the code, which is why adding a
    new one is a compatible change and renaming an old one is not.
    """

    code: str
    message: str
    trace_id: str | None = None
    retry_after_seconds: int | None = None


class HealthBody(BaseModel):
    """``/health/live`` and ``/health/ready``."""

    model_config = ConfigDict(frozen=True)

    status: str
    checks: dict[str, str] = Field(default_factory=dict)
