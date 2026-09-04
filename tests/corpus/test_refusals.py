"""What the tooling refuses, and why each refusal exists.

Every case here produces a report that renders perfectly if it is allowed
through: the coefficients compute, the confusion matrix fills in, and the
number is about something other than what a reader will take it to be about.
That is what makes them worth a test each - a wrong-but-plausible agreement
figure is not caught by anything downstream, and it is the figure the thesis
defends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.agreement.report import (
    IncompatibleVersions,
    NotIndependent,
    RefusedComparison,
    UnusableAnnotation,
    compare,
)
from corpus.cli.main import main
from corpus.io.elan import ElanError, WouldOverwrite, read, write_template
from corpus.schema.records import SCHEMA_VERSION, AnnotatedRecording, AnnotationPass
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import SpeechEventType
from tests.corpus.conftest import annotation, recording, word

FILLED = SpeechEventType.FILLED_PAUSE


def _pair() -> tuple[AnnotatedRecording, AnnotatedRecording]:
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording("beto", [annotation(FILLED, 1_050, 2_050, "beto")])
    return left, right


# ---------------------------------------------------------------------------
# Independence
# ---------------------------------------------------------------------------


def test_an_adjudicated_file_is_refused() -> None:
    """The adjudicated pass is what they agreed on *after* seeing each other.

    Comparing it to either original measures the adjudication and scores near 1
    by construction - and it is the easiest file in the folder to grab by
    mistake, because it is the one that looks finished.
    """
    left = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana")],
        annotation_pass=AnnotationPass.ADJUDICATED,
    )
    right = recording("beto", [annotation(FILLED, 1_050, 2_050, "beto")])

    with pytest.raises(NotIndependent, match="adjudicated"):
        compare(left, right)


def test_an_adjudicated_file_on_the_right_is_refused_too() -> None:
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording(
        "beto",
        [annotation(FILLED, 1_050, 2_050, "beto")],
        annotation_pass=AnnotationPass.ADJUDICATED,
    )

    with pytest.raises(NotIndependent, match="adjudicated"):
        compare(left, right)


def test_two_first_passes_by_different_annotators_are_the_normal_case() -> None:
    """The guard must not catch what it is meant to permit."""
    left, right = _pair()

    assert compare(left, right).matched == 1


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def test_files_under_different_major_schema_versions_are_refused() -> None:
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording(
        "beto",
        [annotation(FILLED, 1_050, 2_050, "beto")],
        schema_version=SemanticVersion(2, 0, 0),
    )

    with pytest.raises(IncompatibleVersions, match="schema"):
        compare(left, right)


def test_a_minor_schema_difference_is_allowed() -> None:
    """Additive within a major version, which is what the version means."""
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording(
        "beto",
        [annotation(FILLED, 1_050, 2_050, "beto")],
        schema_version=SemanticVersion(SCHEMA_VERSION.major, SCHEMA_VERSION.minor + 1, 0),
    )

    assert compare(left, right).matched == 1


def test_files_under_different_major_taxonomy_versions_are_refused() -> None:
    """A class has been redefined or removed between them.

    The 'disagreement' would be two people correctly following two different
    manuals. §17 of the protocol handles that by re-running the pilot, not by
    reporting a number for a manual that no longer exists.
    """
    left = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana")],
        taxonomy_version=SemanticVersion(1, 0, 0),
    )
    right = recording(
        "beto",
        [annotation(FILLED, 1_050, 2_050, "beto")],
        taxonomy_version=SemanticVersion(2, 0, 0),
    )

    with pytest.raises(IncompatibleVersions, match="taxonomy"):
        compare(left, right)


def test_a_minor_taxonomy_difference_is_a_note_rather_than_a_refusal() -> None:
    """Additive: the class list grew, so the comparison still stands - but a
    class added between them was available to only one of them."""
    left = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana")],
        taxonomy_version=SemanticVersion(1, 0, 0),
    )
    right = recording(
        "beto",
        [annotation(FILLED, 1_050, 2_050, "beto")],
        taxonomy_version=SemanticVersion(1, 1, 0),
    )

    report = compare(left, right)

    assert report.matched == 1
    assert any("taxonomy versions" in note for note in report.notes)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def test_a_file_with_a_validation_error_is_refused() -> None:
    """Two same-class annotations at the same instant.

    One of them is a mis-drag, and it offers the matching two candidates where
    the other annotator has one - so whatever it produces is an artefact of the
    mis-drag rather than a disagreement.
    """
    left = recording(
        "ana",
        [
            annotation(FILLED, 1_000, 2_000, "ana"),
            annotation(FILLED, 1_500, 2_500, "ana"),
        ],
    )
    right = recording("beto", [annotation(FILLED, 1_050, 2_050, "beto")])

    with pytest.raises(UnusableAnnotation, match="overlapping_same_class"):
        compare(left, right)


def test_a_file_with_no_transcript_is_refused() -> None:
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")], words=[])
    right = recording("beto", [annotation(FILLED, 1_050, 2_050, "beto")])

    with pytest.raises(UnusableAnnotation, match="no_transcription"):
        compare(left, right)


def test_warnings_still_do_not_block() -> None:
    """An agreement report over only the tidy files would measure tidiness and
    drop exactly the hard cases the pilot exists to find."""
    left = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana")],
        words=[word("buenos,", 0, 500)],  # orthographic mark: a warning
    )
    right = recording("beto", [annotation(FILLED, 1_050, 2_050, "beto")])

    assert compare(left, right).matched == 1


def test_every_refusal_shares_one_base_class() -> None:
    """So a caller can catch the category without enumerating it."""
    for exception in (NotIndependent, IncompatibleVersions, UnusableAnnotation):
        assert issubclass(exception, RefusedComparison)


# ---------------------------------------------------------------------------
# Schema version in the file
# ---------------------------------------------------------------------------


def test_the_template_records_the_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "t.eaf"
    write_template(
        path,
        recording_id="pilot-001",
        speaker_pseudonym="P-001",
        annotator_id="ana",
        media_url="pilot-001.wav",
    )

    assert f'"schema_version">{SCHEMA_VERSION}<' in path.read_text(encoding="utf-8")


def test_a_file_without_a_schema_version_is_refused(tmp_path: Path) -> None:
    """Defaulting to the current version would be a guess, and the guess would
    surface as an agreement figure rather than as an error."""
    path = tmp_path / "t.eaf"
    write_template(
        path,
        recording_id="pilot-001",
        speaker_pseudonym="P-001",
        annotator_id="ana",
        media_url="pilot-001.wav",
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace('NAME="schema_version"', 'NAME="unused"'),
        encoding="utf-8",
    )

    with pytest.raises(ElanError, match="schema_version"):
        read(path)


def test_a_file_from_a_future_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.eaf"
    write_template(
        path,
        recording_id="pilot-001",
        speaker_pseudonym="P-001",
        annotator_id="ana",
        media_url="pilot-001.wav",
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'"schema_version">{SCHEMA_VERSION}<', '"schema_version">9.0.0<'
        ),
        encoding="utf-8",
    )

    # Escaped: the dots are version separators, not "any character".
    with pytest.raises(ElanError, match=r"schema 9\.0\.0"):
        read(path)


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------


def _template_args(path: Path) -> list[str]:
    return [
        "template",
        str(path),
        "--recording-id",
        "pilot-001",
        "--speaker",
        "P-001",
        "--annotator",
        "ana",
        "--media",
        "pilot-001.wav",
    ]


def test_the_template_refuses_to_overwrite(tmp_path: Path) -> None:
    """An empty template and a finished annotation are the same kind of file
    with the same natural name. There is nothing to recover a day's work from."""
    path = tmp_path / "pilot-ana.eaf"
    path.write_text("a day of somebody's annotation", encoding="utf-8")

    with pytest.raises(WouldOverwrite, match="already exists"):
        write_template(
            path,
            recording_id="pilot-001",
            speaker_pseudonym="P-001",
            annotator_id="ana",
            media_url="pilot-001.wav",
        )

    assert path.read_text(encoding="utf-8") == "a day of somebody's annotation"


def test_the_cli_refuses_to_overwrite_and_says_how_to_mean_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "pilot-ana.eaf"
    path.write_text("existing work", encoding="utf-8")

    assert main(_template_args(path)) == 1
    assert "--force" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == "existing work"


def test_force_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "pilot-ana.eaf"
    path.write_text("existing work", encoding="utf-8")

    assert main([*_template_args(path), "--force"]) == 0
    assert "CONTROLLED_VOCABULARY" in path.read_text(encoding="utf-8")


def test_writing_a_new_file_is_unaffected(tmp_path: Path) -> None:
    path = tmp_path / "fresh.eaf"

    assert main(_template_args(path)) == 0
    assert path.exists()
