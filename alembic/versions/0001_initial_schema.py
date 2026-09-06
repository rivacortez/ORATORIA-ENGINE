"""Initial schema.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-04

GENERATED FILE - do not edit by hand.

Source: src/evidence_engine/adapters/outbound/persistence/postgres/models.py
Regenerate: uv run python scripts/render_initial_migration.py
Guarded by: tests/unit/test_initial_migration_is_current.py

Only the initial migration is generated. Every later one is a handwritten delta
with a tested downgrade, because a delta encodes intent - which column moved
where, and what happens to the rows already in it - and that cannot be inferred
from a schema diff.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Rendered from the declarative models for the PostgreSQL dialect, in
#: dependency order so foreign keys resolve.
CREATE_STATEMENTS: tuple[str, ...] = (
    """
        CREATE TABLE analysis_session (
        	id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	application_id VARCHAR(64) NOT NULL,
        	mode VARCHAR(16) NOT NULL,
        	locale VARCHAR(16) NOT NULL,
        	state VARCHAR(32) NOT NULL,
        	configuration_id VARCHAR(64) NOT NULL,
        	capabilities JSONB NOT NULL,
        	created_at_ms BIGINT NOT NULL,
        	completed_at_ms BIGINT,
        	clock_accumulated_ms BIGINT NOT NULL,
        	clock_segment_started_at BIGINT,
        	PRIMARY KEY (id),
        	CONSTRAINT ck_session_mode CHECK (mode in ('realtime','batch'))
        )
    """,
    """
        CREATE INDEX ix_analysis_session_application_id ON analysis_session (application_id)
    """,
    """
        CREATE INDEX ix_session_tenant ON analysis_session (tenant_id, id)
    """,
    """
        CREATE TABLE audit_record (
        	id BIGSERIAL NOT NULL,
        	actor VARCHAR(64) NOT NULL,
        	action VARCHAR(64) NOT NULL,
        	resource VARCHAR(128) NOT NULL,
        	tenant_id VARCHAR(64),
        	timestamp_ms BIGINT NOT NULL,
        	trace_id VARCHAR(64) NOT NULL,
        	outcome VARCHAR(32) NOT NULL,
        	detail TEXT NOT NULL,
        	PRIMARY KEY (id)
        )
    """,
    """
        CREATE INDEX ix_audit_record_action ON audit_record (action)
    """,
    """
        CREATE INDEX ix_audit_record_resource ON audit_record (resource)
    """,
    """
        CREATE INDEX ix_audit_record_tenant_id ON audit_record (tenant_id)
    """,
    """
        CREATE INDEX ix_audit_record_trace_id ON audit_record (trace_id)
    """,
    """
        CREATE TABLE client_application (
        	id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	name VARCHAR(255) NOT NULL,
        	status VARCHAR(32) NOT NULL,
        	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        	PRIMARY KEY (id),
        	CONSTRAINT ck_application_status CHECK (status in ('active','disabled'))
        )
    """,
    """
        CREATE INDEX ix_client_application_tenant_id ON client_application (tenant_id)
    """,
    """
        CREATE TABLE configuration_snapshot (
        	id VARCHAR(64) NOT NULL,
        	taxonomy_version VARCHAR(32) NOT NULL,
        	pipeline_version VARCHAR(32) NOT NULL,
        	schema_version VARCHAR(32) NOT NULL,
        	payload JSONB NOT NULL,
        	seed BIGINT,
        	seed_reason VARCHAR(40),
        	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        	PRIMARY KEY (id),
        	CONSTRAINT ck_configuration_seed_xor_reason CHECK ((seed is not null and seed between 0 and 9007199254740991 and seed_reason is null) or (seed is null and seed_reason is not null))
        )
    """,
    """
        CREATE TABLE evidence_document (
        	tenant_id VARCHAR(64) NOT NULL,
        	session_id VARCHAR(64) NOT NULL,
        	run_id VARCHAR(64) NOT NULL,
        	schema_version VARCHAR(32) NOT NULL,
        	document JSONB NOT NULL,
        	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        	PRIMARY KEY (tenant_id, session_id)
        )
    """,
    """
        CREATE TABLE idempotency_key (
        	application_id VARCHAR(64) NOT NULL,
        	key VARCHAR(255) NOT NULL,
        	session_id VARCHAR(64) NOT NULL,
        	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        	PRIMARY KEY (application_id, key)
        )
    """,
    """
        CREATE TABLE model_version (
        	id VARCHAR(64) NOT NULL,
        	role VARCHAR(32) NOT NULL,
        	artifact_digest VARCHAR(128) NOT NULL,
        	dataset_version VARCHAR(64) NOT NULL,
        	approval VARCHAR(24) NOT NULL,
        	metrics JSONB NOT NULL,
        	rollback_to VARCHAR(64),
        	is_active BOOLEAN NOT NULL,
        	PRIMARY KEY (id)
        )
    """,
    """
        CREATE INDEX ix_model_version_role ON model_version (role)
    """,
    """
        CREATE UNIQUE INDEX uq_model_active_per_role ON model_version (role) WHERE is_active IS true
    """,
    """
        CREATE TABLE api_key (
        	id VARCHAR(64) NOT NULL,
        	application_id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	hashed_secret VARCHAR(128) NOT NULL,
        	prefix VARCHAR(32) NOT NULL,
        	scopes JSONB NOT NULL,
        	expires_at_ms BIGINT,
        	revoked_at_ms BIGINT,
        	PRIMARY KEY (id),
        	FOREIGN KEY(application_id) REFERENCES client_application (id) ON DELETE CASCADE,
        	UNIQUE (hashed_secret)
        )
    """,
    """
        CREATE INDEX ix_api_key_application_id ON api_key (application_id)
    """,
    """
        CREATE INDEX ix_api_key_lookup ON api_key (hashed_secret)
    """,
    """
        CREATE INDEX ix_api_key_tenant_id ON api_key (tenant_id)
    """,
    """
        CREATE TABLE consent_receipt (
        	session_id VARCHAR(64) NOT NULL,
        	policy_version VARCHAR(32) NOT NULL,
        	retain_raw_media BOOLEAN NOT NULL,
        	raw_media_ttl_seconds INTEGER NOT NULL,
        	retain_derived_aggregates BOOLEAN NOT NULL,
        	granted_at_ms BIGINT NOT NULL,
        	withdrawn_at_ms BIGINT,
        	PRIMARY KEY (session_id),
        	FOREIGN KEY(session_id) REFERENCES analysis_session (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE TABLE processing_run (
        	id VARCHAR(64) NOT NULL,
        	session_id VARCHAR(64) NOT NULL,
        	pipeline_version VARCHAR(32) NOT NULL,
        	state VARCHAR(16) NOT NULL,
        	started_at_ms BIGINT NOT NULL,
        	completed_at_ms BIGINT,
        	completed_stages JSONB NOT NULL,
        	models JSONB NOT NULL,
        	PRIMARY KEY (id),
        	FOREIGN KEY(session_id) REFERENCES analysis_session (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE INDEX ix_processing_run_session_id ON processing_run (session_id)
    """,
    """
        CREATE TABLE modality_availability (
        	id BIGSERIAL NOT NULL,
        	run_id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	modality VARCHAR(16) NOT NULL,
        	start_ms BIGINT NOT NULL,
        	end_ms BIGINT NOT NULL,
        	is_usable BOOLEAN NOT NULL,
        	reason VARCHAR(40),
        	detail TEXT NOT NULL,
        	PRIMARY KEY (id),
        	CONSTRAINT ck_availability_reason_required CHECK ((is_usable and reason is null) or (not is_usable and reason is not null)),
        	FOREIGN KEY(run_id) REFERENCES processing_run (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE INDEX ix_modality_availability_run_id ON modality_availability (run_id)
    """,
    """
        CREATE INDEX ix_modality_availability_tenant_id ON modality_availability (tenant_id)
    """,
    """
        CREATE TABLE multimodal_cooccurrence (
        	id BIGSERIAL NOT NULL,
        	run_id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	speech_event_id VARCHAR(64) NOT NULL,
        	visual_event_id VARCHAR(64) NOT NULL,
        	temporal_distance_ms INTEGER NOT NULL,
        	fusion_window_ms INTEGER NOT NULL,
        	configuration_id VARCHAR(64) NOT NULL,
        	PRIMARY KEY (id),
        	CONSTRAINT uq_cooccurrence_pair UNIQUE (run_id, speech_event_id, visual_event_id),
        	CONSTRAINT ck_cooccurrence_distance CHECK (temporal_distance_ms >= 0),
        	CONSTRAINT ck_cooccurrence_within_window CHECK (temporal_distance_ms <= fusion_window_ms),
        	FOREIGN KEY(run_id) REFERENCES processing_run (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE INDEX ix_multimodal_cooccurrence_run_id ON multimodal_cooccurrence (run_id)
    """,
    """
        CREATE INDEX ix_multimodal_cooccurrence_tenant_id ON multimodal_cooccurrence (tenant_id)
    """,
    """
        CREATE TABLE prosody_reading (
        	id BIGSERIAL NOT NULL,
        	run_id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	indicator VARCHAR(40) NOT NULL,
        	start_ms BIGINT NOT NULL,
        	end_ms BIGINT NOT NULL,
        	value FLOAT,
        	unit VARCHAR(32),
        	confidence FLOAT,
        	calibration VARCHAR(16),
        	reason VARCHAR(40),
        	detail TEXT NOT NULL,
        	role VARCHAR(32) NOT NULL,
        	model_version VARCHAR(64) NOT NULL,
        	taxonomy_version VARCHAR(32) NOT NULL,
        	configuration_id VARCHAR(64) NOT NULL,
        	evidence_ref VARCHAR(255) NOT NULL,
        	seed BIGINT,
        	seed_reason VARCHAR(40),
        	PRIMARY KEY (id),
        	CONSTRAINT ck_prosody_measured_xor_unavailable CHECK ((value is not null and unit is not null and reason is null) or (value is null and reason is not null)),
        	FOREIGN KEY(run_id) REFERENCES processing_run (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE INDEX ix_prosody_reading_run_id ON prosody_reading (run_id)
    """,
    """
        CREATE INDEX ix_prosody_reading_tenant_id ON prosody_reading (tenant_id)
    """,
    """
        CREATE TABLE quality_assessment (
        	id BIGSERIAL NOT NULL,
        	run_id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	modality VARCHAR(16) NOT NULL,
        	metric VARCHAR(40) NOT NULL,
        	start_ms BIGINT NOT NULL,
        	end_ms BIGINT NOT NULL,
        	value FLOAT,
        	unit VARCHAR(32),
        	reason VARCHAR(40),
        	detail TEXT NOT NULL,
        	PRIMARY KEY (id),
        	CONSTRAINT ck_quality_measured_xor_unavailable CHECK ((value is not null and reason is null) or (value is null and reason is not null)),
        	FOREIGN KEY(run_id) REFERENCES processing_run (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE INDEX ix_quality_assessment_run_id ON quality_assessment (run_id)
    """,
    """
        CREATE INDEX ix_quality_assessment_tenant_id ON quality_assessment (tenant_id)
    """,
    """
        CREATE TABLE speech_event (
        	id VARCHAR(64) NOT NULL,
        	run_id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	type VARCHAR(32) NOT NULL,
        	raw_text TEXT NOT NULL,
        	context_role VARCHAR(24),
        	start_ms BIGINT NOT NULL,
        	end_ms BIGINT NOT NULL,
        	tolerance_ms INTEGER NOT NULL,
        	confidence FLOAT NOT NULL,
        	calibration VARCHAR(16) NOT NULL,
        	is_final BOOLEAN NOT NULL,
        	role VARCHAR(32) NOT NULL,
        	model_version VARCHAR(64) NOT NULL,
        	taxonomy_version VARCHAR(32) NOT NULL,
        	configuration_id VARCHAR(64) NOT NULL,
        	evidence_ref VARCHAR(255) NOT NULL,
        	seed BIGINT,
        	seed_reason VARCHAR(40),
        	PRIMARY KEY (id),
        	CONSTRAINT ck_speech_event_seed_xor_reason CHECK ((seed is not null and seed between 0 and 9007199254740991 and seed_reason is null) or (seed is null and seed_reason is not null)),
        	CONSTRAINT ck_speech_event_role CHECK (context_role is null or context_role in ('filler','semantic','discourse_marker','uncertain')),
        	CONSTRAINT ck_speech_event_confidence CHECK (confidence between 0 and 1),
        	FOREIGN KEY(run_id) REFERENCES processing_run (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE INDEX ix_speech_event_run_id ON speech_event (run_id)
    """,
    """
        CREATE INDEX ix_speech_event_run_position ON speech_event (run_id, start_ms)
    """,
    """
        CREATE INDEX ix_speech_event_tenant_id ON speech_event (tenant_id)
    """,
    """
        CREATE TABLE visual_event (
        	id VARCHAR(64) NOT NULL,
        	run_id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	type VARCHAR(40) NOT NULL,
        	direction VARCHAR(16),
        	magnitude FLOAT,
        	start_ms BIGINT NOT NULL,
        	end_ms BIGINT NOT NULL,
        	tolerance_ms INTEGER NOT NULL,
        	confidence FLOAT NOT NULL,
        	calibration VARCHAR(16) NOT NULL,
        	is_final BOOLEAN NOT NULL,
        	role VARCHAR(32) NOT NULL,
        	model_version VARCHAR(64) NOT NULL,
        	taxonomy_version VARCHAR(32) NOT NULL,
        	configuration_id VARCHAR(64) NOT NULL,
        	evidence_ref VARCHAR(255) NOT NULL,
        	seed BIGINT,
        	seed_reason VARCHAR(40),
        	PRIMARY KEY (id),
        	CONSTRAINT ck_visual_event_confidence CHECK (confidence between 0 and 1),
        	CONSTRAINT ck_visual_event_seed_xor_reason CHECK ((seed is not null and seed between 0 and 9007199254740991 and seed_reason is null) or (seed is null and seed_reason is not null)),
        	FOREIGN KEY(run_id) REFERENCES processing_run (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE INDEX ix_visual_event_run_id ON visual_event (run_id)
    """,
    """
        CREATE INDEX ix_visual_event_run_position ON visual_event (run_id, start_ms)
    """,
    """
        CREATE INDEX ix_visual_event_tenant_id ON visual_event (tenant_id)
    """,
    """
        CREATE TABLE word_token (
        	id VARCHAR(64) NOT NULL,
        	run_id VARCHAR(64) NOT NULL,
        	tenant_id VARCHAR(64) NOT NULL,
        	raw_text TEXT NOT NULL,
        	sequence_window_ms BIGINT NOT NULL,
        	sequence_index INTEGER NOT NULL,
        	start_ms BIGINT,
        	end_ms BIGINT,
        	tolerance_ms INTEGER,
        	placement_unavailable_reason VARCHAR(64),
        	placement_unavailable_detail TEXT NOT NULL,
        	confidence FLOAT,
        	calibration VARCHAR(16),
        	confidence_unavailable_reason VARCHAR(64),
        	confidence_unavailable_detail TEXT NOT NULL,
        	model_version VARCHAR(64) NOT NULL,
        	taxonomy_version VARCHAR(32) NOT NULL,
        	configuration_id VARCHAR(64) NOT NULL,
        	evidence_ref VARCHAR(255) NOT NULL,
        	seed BIGINT,
        	seed_reason VARCHAR(40),
        	status VARCHAR(16) NOT NULL,
        	PRIMARY KEY (id),
        	CONSTRAINT ck_token_interval CHECK (start_ms is null or end_ms >= start_ms),
        	CONSTRAINT ck_token_placement_exactly_one_state CHECK ((start_ms is not null and end_ms is not null and tolerance_ms is not null and placement_unavailable_reason is null) or (start_ms is null and end_ms is null and tolerance_ms is null and placement_unavailable_reason is not null)),
        	CONSTRAINT ck_token_confidence CHECK (confidence is null or confidence between 0 and 1),
        	CONSTRAINT ck_token_confidence_exactly_one_state CHECK ((confidence is not null and calibration is not null and confidence_unavailable_reason is null) or (confidence is null and calibration is null and confidence_unavailable_reason is not null)),
        	FOREIGN KEY(run_id) REFERENCES processing_run (id) ON DELETE CASCADE
        )
    """,
    """
        CREATE INDEX ix_token_run_position ON word_token (run_id, start_ms)
    """,
    """
        CREATE INDEX ix_token_run_sequence ON word_token (run_id, sequence_window_ms, sequence_index)
    """,
    """
        CREATE INDEX ix_word_token_run_id ON word_token (run_id)
    """,
    """
        CREATE INDEX ix_word_token_tenant_id ON word_token (tenant_id)
    """,
)

#: Reverse order, so a table is never dropped while another still references it.
DROP_STATEMENTS: tuple[str, ...] = (
    "DROP TABLE IF EXISTS word_token CASCADE",
    "DROP TABLE IF EXISTS visual_event CASCADE",
    "DROP TABLE IF EXISTS speech_event CASCADE",
    "DROP TABLE IF EXISTS quality_assessment CASCADE",
    "DROP TABLE IF EXISTS prosody_reading CASCADE",
    "DROP TABLE IF EXISTS multimodal_cooccurrence CASCADE",
    "DROP TABLE IF EXISTS modality_availability CASCADE",
    "DROP TABLE IF EXISTS processing_run CASCADE",
    "DROP TABLE IF EXISTS consent_receipt CASCADE",
    "DROP TABLE IF EXISTS api_key CASCADE",
    "DROP TABLE IF EXISTS model_version CASCADE",
    "DROP TABLE IF EXISTS idempotency_key CASCADE",
    "DROP TABLE IF EXISTS evidence_document CASCADE",
    "DROP TABLE IF EXISTS configuration_snapshot CASCADE",
    "DROP TABLE IF EXISTS client_application CASCADE",
    "DROP TABLE IF EXISTS audit_record CASCADE",
    "DROP TABLE IF EXISTS analysis_session CASCADE",
)


def upgrade() -> None:
    for statement in CREATE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    """Symmetric and tested.

    A migration that cannot be reversed is a deployment that cannot be rolled
    back, and the first time that matters is the worst time to find out.
    """
    for statement in DROP_STATEMENTS:
        op.execute(statement)
