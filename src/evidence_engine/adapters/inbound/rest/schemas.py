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

from pydantic import BaseModel, ConfigDict, Field

from evidence_engine.domain.evidence.document import EvidenceDocument


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

    audio_codec: str
    sample_rate_hz: int = Field(gt=0)
    locale: str
    video_format: str | None = None
    frame_rate_fps: int | None = Field(default=None, gt=0)


class RetentionBody(WireModel):
    """The retention policy this capture runs under (FR-031, NFR-011)."""

    retain_raw_media: bool = False
    raw_media_ttl_seconds: int = Field(default=0, ge=0)
    retain_derived_aggregates: bool = True


class CreateSessionBody(WireModel):
    """``POST /v1/sessions``."""

    mode: str = Field(pattern="^(realtime|batch)$")
    capabilities: CapabilityRequestBody
    consent_policy_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
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
