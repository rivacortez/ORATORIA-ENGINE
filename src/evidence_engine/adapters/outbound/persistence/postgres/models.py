"""The relational schema.

Two storage shapes on purpose, and the reason is worth stating because it looks
like duplication.

*Normalized rows* hold the evidence bundle: one row per token, event, reading
and assessment. That is what a research export queries, what a per-class metric
is computed from, and what an analyst joins against annotations. A JSON blob
cannot serve any of those without being parsed row by row in Python.

*A JSONB column* holds the published document. It is the public contract with
its provenance manifest, read whole, and read far more often than it is
written. Reassembling it from thirty joins on every ``GET /result`` would spend
a query budget on a value that never changes after completion, and - worse - a
change to the published schema would then require migrating historical rows,
which is exactly what NFR-017's compatibility promise exists to avoid.

Tenant scoping is in the primary keys and the indexes, not in a WHERE clause
applied by convention. NFR-013 requires cross-tenant access to be impossible
rather than merely unusual.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from evidence_engine.domain.shared.provenance import MAX_SEED


def _seed_xor_reason(name: str) -> CheckConstraint:
    """A row records a seed or the reason it has none, never both or neither.

    The same shape as ``ck_prosody_measured_xor_unavailable`` and for the same
    reason: a nullable ``seed`` on its own is a column a query can read as zero
    or as absent depending on who wrote the query, and NFR-015 turns on those
    being different answers. The upper bound is folded in rather than left to
    the domain constructor, because a direct write or a bad migration bypasses
    every Python check and this is the one line it does not bypass.
    """
    return CheckConstraint(
        f"(seed is not null and seed between 0 and {MAX_SEED} and seed_reason is null) "
        "or (seed is null and seed_reason is not null)",
        name=name,
    )


class Base(DeclarativeBase):
    """Declarative base for every table in the engine."""


class ClientApplicationRow(Base):
    """§8 ``ClientApplication``."""

    __tablename__ = "client_application"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("status in ('active','disabled')", name="ck_application_status"),
    )


class ApiKeyRow(Base):
    """§8 ``ApiKey``.

    There is no column for the secret. FR-002 issues it once and persists only
    a hash; a nullable ``secret`` column would be filled in by somebody
    debugging, and NFR-010 would be false from that moment on.
    """

    __tablename__ = "api_key"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    application_id: Mapped[str] = mapped_column(
        ForeignKey("client_application.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    hashed_secret: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    prefix: Mapped[str] = mapped_column(String(32), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    expires_at_ms: Mapped[int | None] = mapped_column(BigInteger)
    revoked_at_ms: Mapped[int | None] = mapped_column(BigInteger)

    __table_args__ = (Index("ix_api_key_lookup", "hashed_secret"),)


class AnalysisSessionRow(Base):
    """§8 ``AnalysisSession``, with its clock and negotiated capabilities."""

    __tablename__ = "analysis_session"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    application_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    locale: Mapped[str] = mapped_column(String(16), nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration_id: Mapped[str] = mapped_column(String(64), nullable=False)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    completed_at_ms: Mapped[int | None] = mapped_column(BigInteger)
    #: Session-clock state. Persisted so a handler restart resumes the same
    #: timeline rather than starting a second one (FR-009).
    clock_accumulated_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    clock_segment_started_at: Mapped[int | None] = mapped_column(BigInteger)

    __table_args__ = (
        # The composite index is the scoping mechanism, not an optimization:
        # every read is (tenant, session), so the unscoped query has no index
        # to ride on and shows up immediately in a plan.
        Index("ix_session_tenant", "tenant_id", "id"),
        CheckConstraint("mode in ('realtime','batch')", name="ck_session_mode"),
    )


class ConsentReceiptRow(Base):
    """§8 ``ConsentReceipt``.

    Outlives the evidence it governed. US-010 wants records that prove deletion
    happened without retaining what was deleted, and that needs the receipt to
    survive.
    """

    __tablename__ = "consent_receipt"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_session.id", ondelete="CASCADE"), primary_key=True
    )
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    retain_raw_media: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw_media_ttl_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retain_derived_aggregates: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    granted_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    withdrawn_at_ms: Mapped[int | None] = mapped_column(BigInteger)


class IdempotencyKeyRow(Base):
    """US-011: one session per (application, key)."""

    __tablename__ = "idempotency_key"

    application_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ProcessingRunRow(Base):
    """§8 ``ProcessingRun``."""

    __tablename__ = "processing_run"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_session.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    completed_at_ms: Mapped[int | None] = mapped_column(BigInteger)
    #: NFR-019: an interrupted batch job resumes from completed stages.
    completed_stages: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    #: Role -> model version wired when the run opened. What was *running*,
    #: as opposed to the document manifest's what *contributed*.
    models: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False, default=dict)


class WordTokenRow(Base):
    """§8 ``WordToken``. ``raw_text`` is the only text column, by design."""

    __tablename__ = "word_token"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: Lexical order, always present. Two columns rather than one so the pair
    #: sorts the way the recogniser emitted words: window first, then the index
    #: within it.
    sequence_window_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sequence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Temporal placement, null together when the aligner could not place the
    #: word. Null rather than zero: a zero start is a word at the beginning of
    #: the session, which is wrong and looks entirely plausible in a dump.
    start_ms: Mapped[int | None] = mapped_column(BigInteger)
    end_ms: Mapped[int | None] = mapped_column(BigInteger)
    tolerance_ms: Mapped[int | None] = mapped_column(Integer)
    placement_unavailable_reason: Mapped[str | None] = mapped_column(String(64))
    placement_unavailable_detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Null when the recogniser reports no per-word posterior. Nullable
    #: rather than defaulted, and paired with a reason: a 0.0 here would be
    #: indistinguishable in a dump from a word the model actually scored
    #: zero, which is the substitution FR-025 forbids.
    confidence: Mapped[float | None] = mapped_column(Float)
    calibration: Mapped[str | None] = mapped_column(String(16))
    confidence_unavailable_reason: Mapped[str | None] = mapped_column(String(64))
    confidence_unavailable_detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: The recogniser's provenance, on every word. Tokens were the only
    #: evidence with no provenance columns, which is why a document with
    #: five recognised words and no disfluency recorded no model at all.
    #: No `role` column: a token is the recogniser's by construction and
    #: the domain constructor refuses any other.
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration_id: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    seed_reason: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        Index("ix_token_run_sequence", "run_id", "sequence_window_ms", "sequence_index"),
        Index("ix_token_run_position", "run_id", "start_ms"),
        CheckConstraint("start_ms is null or end_ms >= start_ms", name="ck_token_interval"),
        # Placement is one state or the other, enforced here rather than only
        # in the mapper. A row with a start and a reason is uninterpretable; a
        # row with neither has silently lost the word's position.
        CheckConstraint(
            "(start_ms is not null and end_ms is not null and tolerance_ms is not null "
            "and placement_unavailable_reason is null) or "
            "(start_ms is null and end_ms is null and tolerance_ms is null "
            "and placement_unavailable_reason is not null)",
            name="ck_token_placement_exactly_one_state",
        ),
        # The only provenance-bearing row that lacked this. `seed_from_columns`
        # returns the seed when both are set and drops the reason without a
        # word, so the constraint is what keeps a direct write honest.
        _seed_xor_reason("ck_token_seed_xor_reason"),
        CheckConstraint(
            "confidence is null or confidence between 0 and 1",
            name="ck_token_confidence",
        ),
        # Exactly one of the two states, enforced by the database rather
        # than by the mapper. A row carrying both a score and a reason is a
        # row nobody can interpret, and one carrying neither has silently
        # lost the confidence it was written with.
        CheckConstraint(
            "(confidence is not null and calibration is not null "
            "and confidence_unavailable_reason is null) or "
            "(confidence is null and calibration is null "
            "and confidence_unavailable_reason is not null)",
            name="ck_token_confidence_exactly_one_state",
        ),
    )


class SpeechEventRow(Base):
    """§8 ``SpeechEvent``."""

    __tablename__ = "speech_event"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    context_role: Mapped[str | None] = mapped_column(String(24))
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tolerance_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    calibration: Mapped[str] = mapped_column(String(16), nullable=False)
    is_final: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration_id: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    #: NFR-015's fourth term, denormalized onto the event row for the same
    #: reason the other provenance fields are: the seed a run consumed is a
    #: property of the evidence, and reaching it through a join to the
    #: configuration would answer a different question - what the run was
    #: configured to use, not what it used.
    seed: Mapped[int | None] = mapped_column(BigInteger)
    seed_reason: Mapped[str | None] = mapped_column(String(40))

    __table_args__ = (
        Index("ix_speech_event_run_position", "run_id", "start_ms"),
        _seed_xor_reason("ck_speech_event_seed_xor_reason"),
        # The role vocabulary is closed (FR-013). A constraint here means a bad
        # migration or a direct write cannot introduce a role the domain would
        # refuse, which is the one path that bypasses the domain entirely.
        CheckConstraint(
            "context_role is null or context_role in "
            "('filler','semantic','discourse_marker','uncertain')",
            name="ck_speech_event_role",
        ),
        CheckConstraint("confidence between 0 and 1", name="ck_speech_event_confidence"),
    )


class VisualEventRow(Base):
    """§8 ``VisualEvent``."""

    __tablename__ = "visual_event"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(40), nullable=False)
    direction: Mapped[str | None] = mapped_column(String(16))
    magnitude: Mapped[float | None] = mapped_column(Float)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tolerance_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    calibration: Mapped[str] = mapped_column(String(16), nullable=False)
    is_final: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration_id: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Same pair, same rule, as ``SpeechEventRow``.
    seed: Mapped[int | None] = mapped_column(BigInteger)
    seed_reason: Mapped[str | None] = mapped_column(String(40))

    __table_args__ = (
        Index("ix_visual_event_run_position", "run_id", "start_ms"),
        CheckConstraint("confidence between 0 and 1", name="ck_visual_event_confidence"),
        _seed_xor_reason("ck_visual_event_seed_xor_reason"),
    )


class ProsodyReadingRow(Base):
    """One prosodic indicator over one window.

    ``value`` is nullable and ``reason`` is not-null-when-value-is-null. That
    pairing is FR-025 expressed in DDL: a row can hold a number or a reason,
    never neither and never both, so no query can produce a zero where a reason
    belongs.
    """

    __tablename__ = "prosody_reading"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    indicator: Mapped[str] = mapped_column(String(40), nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)
    calibration: Mapped[str | None] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(40))
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Provenance on every reading. Readings used to inherit the run's audio
    #: provenance from whichever speech event happened to exist, and when none
    #: did the repository fabricated `ModelVersionId("unknown")`. A prosody
    #: estimator is its own model; its version is recorded where its output is.
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration_id: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    seed_reason: Mapped[str | None] = mapped_column(String(40))

    __table_args__ = (
        CheckConstraint(
            "(value is not null and unit is not null and reason is null) "
            "or (value is null and reason is not null)",
            name="ck_prosody_measured_xor_unavailable",
        ),
    )


class QualityAssessmentRow(Base):
    """§8 ``QualityAssessment``. Same measured-xor-unavailable rule."""

    __tablename__ = "quality_assessment"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    modality: Mapped[str] = mapped_column(String(16), nullable=False)
    metric: Mapped[str] = mapped_column(String(40), nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(String(40))
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (
        CheckConstraint(
            "(value is not null and reason is null) or (value is null and reason is not null)",
            name="ck_quality_measured_xor_unavailable",
        ),
    )


class ModalityAvailabilityRow(Base):
    """Whether a modality could serve a window, and why not (QA-02)."""

    __tablename__ = "modality_availability"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    modality: Mapped[str] = mapped_column(String(16), nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    is_usable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(40))
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (
        CheckConstraint(
            "(is_usable and reason is null) or (not is_usable and reason is not null)",
            name="ck_availability_reason_required",
        ),
    )


class CooccurrenceRow(Base):
    """§8 ``MultimodalCooccurrence``.

    There is no ``causal_inference`` column. The value is a domain constant and
    a column would be a place to store a different one - which is the failure
    FR-028 is written to prevent.
    """

    __tablename__ = "multimodal_cooccurrence"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    speech_event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    visual_event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    temporal_distance_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    fusion_window_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration_id: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "run_id", "speech_event_id", "visual_event_id", name="uq_cooccurrence_pair"
        ),
        CheckConstraint("temporal_distance_ms >= 0", name="ck_cooccurrence_distance"),
        CheckConstraint(
            "temporal_distance_ms <= fusion_window_ms", name="ck_cooccurrence_within_window"
        ),
    )


class EvidenceDocumentRow(Base):
    """The published result, stored whole.

    One per (tenant, session): the latest completed run's document. Earlier
    runs keep their normalized rows, so a reprocessed session can still be
    compared against its predecessor even though only one document is served.
    """

    __tablename__ = "evidence_document"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    document: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ConfigurationSnapshotRow(Base):
    """§8 ``ConfigurationSnapshot``. Immutable once published (US-008).

    The seed is a column rather than another key inside ``payload``, and the
    reason is the immutability check in ``PostgresConfigurationStore.freeze``:
    it compares payloads whole. Adding a key to the payload would make every
    snapshot published before this change compare unequal to itself the next
    time a session froze it, and ``freeze`` would refuse - so the first session
    after deployment would fail on an existing tenant with a correct
    configuration.
    """

    __tablename__ = "configuration_snapshot"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    taxonomy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    seed_reason: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (_seed_xor_reason("ck_configuration_seed_xor_reason"),)


class ModelVersionRow(Base):
    """§8 ``ModelVersion``."""

    __tablename__ = "model_version"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: The component this artifact fills. Was `modality`, which allowed one
    #: active audio model and could not express the canary QA-03 requires.
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    artifact_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    approval: Mapped[str] = mapped_column(String(24), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    rollback_to: Mapped[str | None] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        # One active version per role. Two would make provenance ambiguous
        # for every event produced while both were live.
        Index(
            "uq_model_active_per_role",
            "role",
            unique=True,
            postgresql_where=is_active.is_(True),
        ),
    )


class AuditRecordRow(Base):
    """§8 ``AuditRecord``. Append-only (NFR-022).

    No updated_at, no soft-delete flag, no revision column. An entry that can be
    edited is not evidence of anything, and the absence of the machinery is the
    clearest way to say so.
    """

    __tablename__ = "audit_record"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
