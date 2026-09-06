"""Consent and the capture chain: the refusals, and what they are guarding.

These three records exist because their fields are captured at recruitment or
they are unreconstructable. That deadline is the whole argument for adding
them, and it is also why the refusals matter more than usual: a field that
accepts anything is a field that will be filled with anything on the day
somebody is in a hurry with a participant waiting.

Every test here covers a path that was written and never exercised. The
constructor arguments are deliberately the wrong *type* rather than the wrong
*value*, because that is the failure the code argues about: a string that looks
like a version compares equal to another copy of itself, so a check written
against the value passes while nobody can say what the value means.
"""

from __future__ import annotations

from datetime import date

import pytest

from corpus.schema.records import (
    AnnotatedRecording,
    AnnotationPass,
    ConsentBasis,
    ConsentRecord,
    RecordingConditions,
    SchemaViolation,
    Speaker,
)
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import SpeechEventType
from tests.corpus.conftest import CONDITIONS, CONSENT, annotation, recording

FILLED = SpeechEventType.FILLED_PAUSE


def _consent(**overrides: object) -> ConsentRecord:
    arguments: dict[str, object] = {
        "basis": ConsentBasis.WRITTEN_INFORMED,
        "policy_version": SemanticVersion(1, 0, 0),
        "granted_on": date(2026, 9, 1),
    }
    arguments.update(overrides)
    return ConsentRecord(**arguments)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


def test_a_basis_outside_the_published_set_is_refused() -> None:
    """§4 asks for a count of consent bases, and a basis nobody published
    cannot be counted."""
    with pytest.raises(SchemaViolation, match="ConsentBasis"):
        _consent(basis="written_informed")


def test_an_unparsed_policy_version_is_refused() -> None:
    """The same trap the schema and taxonomy versions fell into: a string that
    looks exactly like a version compares equal to another copy of itself, and
    the check passes while nobody can say which wording was read."""
    with pytest.raises(SchemaViolation, match="SemanticVersion"):
        _consent(policy_version="1.0.0")


def test_a_granted_date_that_is_not_a_date_is_refused() -> None:
    with pytest.raises(SchemaViolation, match="granted on a date"):
        _consent(granted_on="2026-09-01")


def test_a_withdrawal_date_that_is_not_a_date_is_refused() -> None:
    with pytest.raises(SchemaViolation, match="withdrawn on a date"):
        _consent(withdrawn_on="2026-09-02")


def test_withdrawal_before_the_grant_is_refused() -> None:
    """Not a date-arithmetic nicety: a withdrawal that predates its grant means
    one of the two was mistyped, and which one changes whether the recording may
    be used."""
    with pytest.raises(SchemaViolation, match="withdrawn"):
        _consent(granted_on=date(2026, 9, 5), withdrawn_on=date(2026, 9, 1))


def test_consent_starts_active_and_stops_when_withdrawn() -> None:
    granted = _consent()
    assert granted.is_active

    withdrawn = granted.withdraw(date(2026, 9, 10))
    assert not withdrawn.is_active
    assert withdrawn.withdrawn_on == date(2026, 9, 10)


def test_withdrawing_twice_keeps_the_first_date() -> None:
    """The date that matters is when the participant said stop, not when
    somebody re-ran the command."""
    once = _consent().withdraw(date(2026, 9, 10))

    assert once.withdraw(date(2026, 9, 20)).withdrawn_on == date(2026, 9, 10)


# ---------------------------------------------------------------------------
# Recording conditions
# ---------------------------------------------------------------------------


def _conditions(**overrides: object) -> RecordingConditions:
    arguments: dict[str, object] = {
        "microphone": "Realtek(R) Audio - onboard array",
        "sample_rate_hz": 16_000,
        "bit_depth": 16,
        "channels": 1,
        "virtual_audio_bypassed": True,
    }
    arguments.update(overrides)
    return RecordingConditions(**arguments)  # type: ignore[arg-type]


def test_an_unnamed_device_is_refused() -> None:
    """Naming the device *is* the confirmation. An empty string confirms
    nothing while occupying the field that says somebody checked."""
    with pytest.raises(SchemaViolation, match="unnamed device"):
        _conditions(microphone="   ")


@pytest.mark.parametrize("field", ["sample_rate_hz", "bit_depth", "channels"])
@pytest.mark.parametrize("bad", [0, -1, "16000", 16_000.0])
def test_a_non_positive_or_non_integer_capture_number_is_refused(field: str, bad: object) -> None:
    with pytest.raises(SchemaViolation, match="positive integer"):
        _conditions(**{field: bad})


@pytest.mark.parametrize("field", ["sample_rate_hz", "bit_depth", "channels"])
def test_a_bool_is_not_a_capture_number(field: str) -> None:
    """`True` is an int in Python and would pass a naive positive-integer check
    as a sample rate of 1 Hz."""
    with pytest.raises(SchemaViolation, match="positive integer"):
        _conditions(**{field: True})


def test_a_truthy_string_cannot_answer_the_bypass_question() -> None:
    """The one field that records an operator's confirmation. A truthy string
    makes the answer "yes" whatever it says - including "no"."""
    with pytest.raises(SchemaViolation, match="whatever it says"):
        _conditions(virtual_audio_bypassed="no")


# ---------------------------------------------------------------------------
# The record that holds them
# ---------------------------------------------------------------------------


def test_a_consent_that_is_not_a_consent_record_is_refused() -> None:
    """It passes an `is not None` check and then fails much later with an
    AttributeError about `str` having no `basis`."""
    with pytest.raises(SchemaViolation, match="means nothing"):
        recording(
            "ana",
            [annotation(FILLED, 1_000, 2_000, "ana")],
            consent="written_informed",  # type: ignore[arg-type]
        )


def test_conditions_that_are_not_recording_conditions_are_refused() -> None:
    with pytest.raises(SchemaViolation, match="RecordingConditions"):
        recording(
            "ana",
            [annotation(FILLED, 1_000, 2_000, "ana")],
            conditions={"microphone": "Realtek"},  # type: ignore[arg-type]
        )


def test_an_unstated_consent_is_not_an_active_one() -> None:
    """The property that makes the absence of consent look like the absence of
    consent. Answering True here would be the failure the whole field exists to
    prevent."""
    without = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])

    assert without.consent is None
    assert not without.consent_is_active


def test_a_withdrawn_consent_is_not_an_active_one() -> None:
    withdrawn = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana")],
        consent=CONSENT.withdraw(date(2026, 9, 10)),
        conditions=CONDITIONS,
    )

    assert withdrawn.consent is not None
    assert not withdrawn.consent_is_active


def test_a_granted_consent_is_active() -> None:
    granted = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana")],
        consent=CONSENT,
        conditions=CONDITIONS,
    )

    assert granted.consent_is_active


# ---------------------------------------------------------------------------
# The refusals that predate this change and had no test
# ---------------------------------------------------------------------------


def test_a_recording_needs_an_id() -> None:
    with pytest.raises(SchemaViolation, match="needs an id"):
        AnnotatedRecording(
            recording_id="  ",
            speaker=Speaker(pseudonym="P-001"),
            annotator_id="ana",
            annotation_pass=AnnotationPass.FIRST,
            duration_ms=10_000,
        )


def test_a_recording_needs_positive_duration() -> None:
    with pytest.raises(SchemaViolation, match="positive duration"):
        AnnotatedRecording(
            recording_id="pilot-001",
            speaker=Speaker(pseudonym="P-001"),
            annotator_id="ana",
            annotation_pass=AnnotationPass.FIRST,
            duration_ms=0,
        )


def test_a_speaker_needs_a_pseudonym() -> None:
    with pytest.raises(SchemaViolation, match="needs a pseudonym"):
        Speaker(pseudonym="   ")
