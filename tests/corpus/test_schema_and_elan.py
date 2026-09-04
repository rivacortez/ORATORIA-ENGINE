"""The schema's rules, and the ELAN round trip.

The schema tests matter because an annotation that is accepted in a wrong shape
becomes corpus data, and corpus data is expensive to re-collect. The round-trip
test matters because the format is the interface with the annotators: if a file
they spent three hours on cannot be read back, the three hours are gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.io.elan import (
    DISFLUENCY_TIER,
    ROLE_TIER,
    TEXT_TIER,
    WORDS_TIER,
    ElanError,
    read,
    write_template,
)
from corpus.schema.records import (
    AnnotationPass,
    DisfluencyAnnotation,
    Interval,
    SchemaViolation,
    Speaker,
    Word,
)
from corpus.schema.validation import Severity, validate
from evidence_engine.domain.shared.taxonomy import ContextualRole, SpeechEventType
from tests.corpus.conftest import annotation, recording, word

PAUSE = SpeechEventType.FILLED_PAUSE
FILLER = SpeechEventType.LEXICAL_FILLER


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def test_an_interval_needs_positive_duration() -> None:
    with pytest.raises(SchemaViolation, match="no duration"):
        Interval(1_000, 1_000)


def test_an_interval_cannot_start_before_the_recording() -> None:
    with pytest.raises(SchemaViolation, match="before the recording"):
        Interval(-10, 100)


def test_iou_is_zero_for_disjoint_intervals() -> None:
    assert Interval(0, 100).iou(Interval(500, 600)) == pytest.approx(0.0)


def test_iou_is_one_for_identical_intervals() -> None:
    assert Interval(0, 100).iou(Interval(0, 100)) == pytest.approx(1.0)


def test_iou_is_symmetric() -> None:
    a, b = Interval(0, 100), Interval(50, 200)

    assert a.iou(b) == pytest.approx(b.iou(a))


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def test_a_lexical_class_requires_its_raw_expression() -> None:
    """FR-017: precision is uncomputable without the true negatives."""
    with pytest.raises(SchemaViolation, match="FR-017"):
        DisfluencyAnnotation(
            event_type=FILLER,
            interval=Interval(0, 400),
            annotator_id="ana",
            raw_text="",
            context_role=ContextualRole.FILLER,
        )


def test_a_lexical_class_requires_a_contextual_role() -> None:
    with pytest.raises(SchemaViolation, match="uncertain"):
        DisfluencyAnnotation(
            event_type=FILLER,
            interval=Interval(0, 400),
            annotator_id="ana",
            raw_text="este",
            context_role=None,
        )


def test_an_acoustic_class_cannot_carry_a_role() -> None:
    """A filled pause has no word to attach a role to."""
    with pytest.raises(SchemaViolation, match="no word"):
        DisfluencyAnnotation(
            event_type=PAUSE,
            interval=Interval(0, 400),
            annotator_id="ana",
            context_role=ContextualRole.FILLER,
        )


def test_an_annotation_names_its_annotator() -> None:
    with pytest.raises(SchemaViolation, match="agreement needs it"):
        DisfluencyAnnotation(event_type=PAUSE, interval=Interval(0, 400), annotator_id="  ")


def test_annotator_confidence_is_distinct_from_the_uncertain_role() -> None:
    """An annotator can be certain that the context does not decide."""
    certain_about_uncertainty = annotation(
        FILLER, 0, 400, role=ContextualRole.UNCERTAIN, confidence=1.0
    )

    assert certain_about_uncertainty.is_uncertain
    assert certain_about_uncertainty.annotator_confidence == pytest.approx(1.0)


def test_confidence_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(SchemaViolation, match=r"\[0, 1\]"):
        annotation(PAUSE, 0, 400, confidence=1.4)


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------


def test_a_speaker_has_no_field_for_a_name() -> None:
    """§14.4: easier to honour when the identifying column does not exist."""
    fields = Speaker.__dataclass_fields__

    assert "name" not in fields
    assert "email" not in fields
    assert "pseudonym" in fields


def test_something_that_looks_like_an_email_is_refused() -> None:
    with pytest.raises(SchemaViolation, match="pseudonymized"):
        Speaker(pseudonym="ana.perez@universidad.pe")


def test_an_ordinary_pseudonym_is_accepted() -> None:
    assert Speaker(pseudonym="P-001").pseudonym == "P-001"


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------


def test_a_recording_refuses_annotations_by_someone_else() -> None:
    """Agreement would otherwise compare the wrong people."""
    with pytest.raises(SchemaViolation, match="wrong people"):
        recording("ana", [annotation(PAUSE, 0, 400, "beto")])


def test_a_recording_refuses_an_annotation_past_its_end() -> None:
    with pytest.raises(SchemaViolation, match="past the"):
        recording("ana", [annotation(PAUSE, 0, 90_000)], duration_ms=60_000)


# ---------------------------------------------------------------------------
# The ELAN round trip
# ---------------------------------------------------------------------------


def test_a_template_carries_the_whole_taxonomy_as_a_vocabulary(tmp_path: Path) -> None:
    """The single most valuable property: an unpublished class cannot be typed."""
    path = tmp_path / "pilot.eaf"
    write_template(
        path,
        recording_id="pilot-001",
        speaker_pseudonym="P-001",
        annotator_id="ana",
        media_url="pilot-001.wav",
    )
    content = path.read_text(encoding="utf-8")

    for event in SpeechEventType:
        assert f">{event.value}<" in content
    for role in ContextualRole:
        assert f">{role.value}<" in content


def test_a_template_declares_the_tiers_the_reader_expects(tmp_path: Path) -> None:
    path = tmp_path / "pilot.eaf"
    write_template(
        path,
        recording_id="pilot-001",
        speaker_pseudonym="P-001",
        annotator_id="ana",
        media_url="pilot-001.wav",
    )
    content = path.read_text(encoding="utf-8")

    for tier in (WORDS_TIER, DISFLUENCY_TIER, ROLE_TIER, TEXT_TIER):
        assert f'TIER_ID="{tier}"' in content
    # The role tier is a child of the disfluency tier, so a role cannot come
    # apart from the event it describes.
    assert 'PARENT_REF="disfluency"' in content


def test_a_filled_file_reads_back_as_it_was_written(tmp_path: Path) -> None:
    path = tmp_path / "filled.eaf"
    _write_filled(path)

    result = read(path)

    assert result.recording_id == "pilot-001"
    assert result.annotator_id == "ana"
    assert result.speaker.pseudonym == "P-001"
    assert result.annotation_pass is AnnotationPass.FIRST
    assert [w.text for w in result.words] == ["buenos", "dias"]
    assert len(result.disfluencies) == 1

    found = result.disfluencies[0]
    assert found.event_type is FILLER
    assert found.interval.start_ms == 900
    assert found.interval.end_ms == 1_680
    assert found.raw_text == "este"
    assert found.context_role is ContextualRole.UNCERTAIN


def test_a_file_without_an_annotator_is_refused(tmp_path: Path) -> None:
    """Agreement is between named people; a file that names nobody cannot help."""
    path = tmp_path / "anonymous.eaf"
    _write_filled(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace('NAME="annotator_id">ana', 'NAME="unused">ana'),
        encoding="utf-8",
    )

    with pytest.raises(ElanError, match="annotator_id"):
        read(path)


def test_a_class_outside_the_taxonomy_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "stale.eaf"
    _write_filled(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(">lexical_filler<", ">nervousness<", 1),
        encoding="utf-8",
    )

    with pytest.raises(ElanError, match="not a published class"):
        read(path)


def test_malformed_xml_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.eaf"
    path.write_text("<ANNOTATION_DOCUMENT><unclosed>", encoding="utf-8")

    # Escaped: the dot is a filename separator here, not "any character".
    with pytest.raises(ElanError, match=r"broken\.eaf"):
        read(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_two_overlapping_annotations_of_one_class_are_an_error() -> None:
    """A speaker cannot produce two filled pauses at once; one is a mis-drag."""
    report = validate(
        recording(
            "ana",
            [annotation(PAUSE, 1_000, 2_000), annotation(PAUSE, 1_500, 2_500)],
            words=[word("hola", 0, 500)],
        )
    )

    codes = {finding.code for finding in report.errors}
    assert "overlapping_same_class" in codes
    assert report.is_usable is False


def test_overlap_across_different_classes_is_legitimate() -> None:
    """A filled pause can sit inside a false start."""
    report = validate(
        recording(
            "ana",
            [
                annotation(PAUSE, 1_000, 2_000),
                annotation(SpeechEventType.FALSE_START, 900, 2_400, raw_text="los"),
            ],
            words=[word("los", 900, 2_400)],
        )
    )

    assert "overlapping_same_class" not in {f.code for f in report.errors}


def test_punctuation_in_the_words_tier_is_flagged() -> None:
    """FR-011: a comma is not a sound, so somebody transcribed meaning."""
    report = validate(recording("ana", [], words=[word("buenos,", 0, 400), word("dias", 400, 800)]))

    assert "orthographic_marks" in {f.code for f in report.warnings}


def test_an_expression_the_transcript_does_not_contain_is_flagged() -> None:
    """Usually the tell that an annotator typed a tidied form."""
    report = validate(
        recording(
            "ana",
            [annotation(FILLER, 1_000, 1_400, raw_text="entonces")],
            words=[word("este", 1_000, 1_400)],
        )
    )

    assert "unattested_expression" in {f.code for f in report.warnings}


def test_an_empty_words_tier_is_an_error() -> None:
    report = validate(recording("ana", [annotation(PAUSE, 0, 400)], words=[]))

    assert "no_transcription" in {f.code for f in report.errors}


def test_the_abstention_rate_is_reported_and_never_a_fault() -> None:
    """§17 prescribes `uncertain`; using it is following the manual."""
    report = validate(
        recording(
            "ana",
            [annotation(FILLER, 1_000, 1_400, role=ContextualRole.UNCERTAIN)],
            words=[word("este", 1_000, 1_400)],
        )
    )

    abstention = [f for f in report.findings if f.code == "abstention_rate"]
    assert abstention
    assert abstention[0].severity is Severity.INFO
    assert report.is_usable is True


def test_warnings_do_not_block_a_file_from_the_agreement_report() -> None:
    """Dropping warned files would discard exactly the hard cases."""
    report = validate(recording("ana", [], words=[word("buenos,", 0, 400), word("dias", 400, 800)]))

    assert report.warnings
    assert report.is_usable is True


def _write_filled(path: Path) -> None:
    """An EAF with one word tier and one annotated lexical filler."""
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="ana" DATE="2026-09-04T00:00:00+00:00" FORMAT="3.0" VERSION="3.0">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <PROPERTY NAME="recording_id">pilot-001</PROPERTY>
        <PROPERTY NAME="speaker_pseudonym">P-001</PROPERTY>
        <PROPERTY NAME="annotator_id">ana</PROPERTY>
        <PROPERTY NAME="annotation_pass">first</PROPERTY>
        <PROPERTY NAME="taxonomy_version">1.0.0</PROPERTY>
    </HEADER>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="400"/>
        <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="400"/>
        <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="800"/>
        <TIME_SLOT TIME_SLOT_ID="ts5" TIME_VALUE="900"/>
        <TIME_SLOT TIME_SLOT_ID="ts6" TIME_VALUE="1680"/>
    </TIME_ORDER>
    <TIER LINGUISTIC_TYPE_REF="verbatim" TIER_ID="words">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="w1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>buenos</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="w2" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
                <ANNOTATION_VALUE>dias</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="disfluency_class" TIER_ID="disfluency">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="d1" TIME_SLOT_REF1="ts5" TIME_SLOT_REF2="ts6">
                <ANNOTATION_VALUE>lexical_filler</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="contextual_role" PARENT_REF="disfluency" TIER_ID="role">
        <ANNOTATION>
            <REF_ANNOTATION ANNOTATION_ID="r1" ANNOTATION_REF="d1">
                <ANNOTATION_VALUE>uncertain</ANNOTATION_VALUE>
            </REF_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="attached_text" PARENT_REF="disfluency" TIER_ID="text">
        <ANNOTATION>
            <REF_ANNOTATION ANNOTATION_ID="t1" ANNOTATION_REF="d1">
                <ANNOTATION_VALUE>este</ANNOTATION_VALUE>
            </REF_ANNOTATION>
        </ANNOTATION>
    </TIER>
</ANNOTATION_DOCUMENT>
""",
        encoding="utf-8",
    )


def test_word_intervals_survive_the_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "filled.eaf"
    _write_filled(path)

    result = read(path)

    assert result.words[0].interval == Interval(0, 400)
    assert result.words[1].interval == Interval(400, 800)


def test_a_word_cannot_be_empty() -> None:
    with pytest.raises(SchemaViolation, match="unintelligible"):
        Word(text="   ", interval=Interval(0, 100))
