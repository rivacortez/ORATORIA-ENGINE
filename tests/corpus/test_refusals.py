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

from corpus.agreement import report
from corpus.agreement.report import (
    IncompatibleVersions,
    InvalidReportParameters,
    NotIndependent,
    RefusedComparison,
    UnusableAnnotation,
    compare,
)
from corpus.cli.main import main
from corpus.io.elan import ElanError, WouldOverwrite, read
from corpus.schema.records import (
    SCHEMA_VERSION,
    AnnotatedRecording,
    AnnotationPass,
    SchemaViolation,
    Speaker,
)
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import TAXONOMY_VERSION, SpeechEventType
from tests.corpus.conftest import TEMPLATE_FLAGS, annotation, recording, template, word

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
        # Not 2.0.0: that is the current version, and a test whose "other"
        # value drifts into being the real one stops testing anything.
        schema_version=SemanticVersion(SCHEMA_VERSION.major + 1, 0, 0),
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


@pytest.mark.parametrize(
    "other",
    [
        SemanticVersion(2, 0, 0),  # a class redefined or removed
        SemanticVersion(1, 1, 0),  # a class added, or a definition amended
        SemanticVersion(1, 0, 1),  # a wording fix that changed a judgement call
    ],
    ids=["major", "minor", "patch"],
)
def test_any_taxonomy_difference_is_refused(other: SemanticVersion) -> None:
    """Not just a major difference.

    The additive-class argument for tolerating a minor bump does not cover a
    class whose *meaning* moved, and the protocol is unambiguous: a taxonomy
    change requires re-running Pilot B. Two annotators reading two versions of
    the manual are not two annotators reading the manual.
    """
    left = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana")],
        taxonomy_version=SemanticVersion(1, 0, 0),
    )
    right = recording(
        "beto",
        [annotation(FILLED, 1_050, 2_050, "beto")],
        taxonomy_version=other,
    )

    with pytest.raises(IncompatibleVersions, match="different manuals"):
        compare(left, right)


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_missing_taxonomy_version_is_refused(side: str) -> None:
    """The hole the reviewer found.

    `_parse_version` turned an absent or unparseable version into `None`, and
    the guard returned early when either side was `None` - so a file recording
    no manual at all sailed past the check written to compare manuals. `None`
    compares unequal to nothing; a check written against a value stops being a
    check the moment the value goes missing.
    """
    versions = {"left": SemanticVersion(1, 0, 0), "right": SemanticVersion(1, 0, 0)}
    versions[side] = None  # type: ignore[assignment]

    left = recording(
        "ana", [annotation(FILLED, 1_000, 2_000, "ana")], taxonomy_version=versions["left"]
    )
    right = recording(
        "beto", [annotation(FILLED, 1_050, 2_050, "beto")], taxonomy_version=versions["right"]
    )

    with pytest.raises(IncompatibleVersions, match="no taxonomy version"):
        compare(left, right)


def test_identical_taxonomy_versions_are_the_normal_case() -> None:
    left, right = _pair()

    assert compare(left, right).matched == 1


def test_an_eaf_without_a_taxonomy_version_is_refused_at_read_time(tmp_path: Path) -> None:
    """Caught at the file boundary too, not only at comparison.

    An annotator who deletes the property gets told which file and which
    property, rather than a refusal three commands later about two files.
    """
    path = tmp_path / "t.eaf"
    template(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace('NAME="taxonomy_version"', 'NAME="unused"'),
        encoding="utf-8",
    )

    with pytest.raises(ElanError, match="taxonomy_version"):
        read(path)


def test_an_unparseable_taxonomy_version_is_refused_rather_than_dropped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.eaf"
    template(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'"taxonomy_version">{TAXONOMY_VERSION}<', '"taxonomy_version">v1<'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ElanError, match="not a taxonomy version"):
        read(path)


# ---------------------------------------------------------------------------
# Reporting parameters
# ---------------------------------------------------------------------------


def test_a_negative_boundary_review_threshold_is_refused() -> None:
    """It turns perfect agreement into a worklist.

    `abs(error) > threshold` holds for every matched pair once the threshold
    goes below zero, so two annotators who drew identical boundaries come back
    with "start off by 0 ms, end by 0 ms (over the -1 ms NFR-004 target)" for
    every event they agreed on.
    """
    left, right = _pair()

    with pytest.raises(InvalidReportParameters, match="non-negative"):
        compare(left, right, boundary_review_ms=-1)


def test_a_zero_boundary_review_threshold_is_allowed() -> None:
    """Strict but meaningful: list every pair whose boundaries differ at all."""
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording("beto", [annotation(FILLED, 1_010, 2_000, "beto")])

    report = compare(left, right, boundary_review_ms=0)

    assert [d.kind.value for d in report.disagreements] == ["boundary"]


def test_exact_agreement_at_a_zero_threshold_lists_nothing() -> None:
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording("beto", [annotation(FILLED, 1_000, 2_000, "beto")])

    assert compare(left, right, boundary_review_ms=0).disagreements == ()


def test_the_cli_reports_a_bad_parameter_instead_of_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An annotator seeing a Python traceback reads "the tool is broken", not
    "that threshold means nothing"."""
    from tests.corpus.test_cli import _annotated

    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=80, event="filled_pause")

    # Exit 2 is argparse's usage-error code, and that is what this is: the
    # files are fine, the numbers asked for are not.
    assert main(["agreement", str(left), str(right), "--iou", "0"]) == 2
    assert "deliberately" in capsys.readouterr().err

    assert main(["agreement", str(left), str(right), "--boundary-review-ms", "-1"]) == 2
    assert "non-negative" in capsys.readouterr().err


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
    template(path)

    assert f'"schema_version">{SCHEMA_VERSION}<' in path.read_text(encoding="utf-8")


def test_a_file_without_a_schema_version_is_refused(tmp_path: Path) -> None:
    """Defaulting to the current version would be a guess, and the guess would
    surface as an agreement figure rather than as an error."""
    path = tmp_path / "t.eaf"
    template(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace('NAME="schema_version"', 'NAME="unused"'),
        encoding="utf-8",
    )

    with pytest.raises(ElanError, match="schema_version"):
        read(path)


def test_a_file_from_a_future_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.eaf"
    template(path)
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
    """The `corpus template` command line, with the recruitment fields.

    `TEMPLATE_FLAGS` carries the variety, consent record and capture chain the
    command now requires. They are not optional and should not be: all three are
    captured at recruitment or they are unreconstructable.
    """
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
        *TEMPLATE_FLAGS,
    ]


def test_the_template_refuses_to_overwrite(tmp_path: Path) -> None:
    """An empty template and a finished annotation are the same kind of file
    with the same natural name. There is nothing to recover a day's work from."""
    path = tmp_path / "pilot-ana.eaf"
    path.write_text("a day of somebody's annotation", encoding="utf-8")

    with pytest.raises(WouldOverwrite, match="already exists"):
        template(path)

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


# ---------------------------------------------------------------------------
# The domain model, not only the file reader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["v1", "1.0", "", "1.0.0", 100])
def test_a_version_that_is_not_a_semantic_version_is_refused(bad: object) -> None:
    """The hole behind the hole.

    The ELAN reader parses both versions and refuses what it cannot read, so a
    record built from a file was safe. A record built in code was not: Python
    does not enforce the annotation, `taxonomy_version="v1"` was accepted, and
    two records carrying the *same* unparseable string compare equal - so the
    agreement guard passed them and reported a comparison between two manuals
    nobody can identify.

    Note `"1.0.0"` in the list. The string that looks exactly right is the one
    that would have survived review.
    """
    with pytest.raises(SchemaViolation, match="SemanticVersion"):
        AnnotatedRecording(
            recording_id="pilot-001",
            speaker=Speaker(pseudonym="P-001"),
            annotator_id="ana",
            annotation_pass=AnnotationPass.FIRST,
            duration_ms=10_000,
            taxonomy_version=bad,  # type: ignore[arg-type]
        )


def test_an_invalid_schema_version_is_refused_too() -> None:
    with pytest.raises(SchemaViolation, match="schema_version"):
        AnnotatedRecording(
            recording_id="pilot-001",
            speaker=Speaker(pseudonym="P-001"),
            annotator_id="ana",
            annotation_pass=AnnotationPass.FIRST,
            duration_ms=10_000,
            schema_version="1.0.0",  # type: ignore[arg-type]
        )


def test_two_records_with_the_same_invalid_version_can_no_longer_be_compared() -> None:
    """The end-to-end shape of the defect, stated as the harm.

    Equality is not enough: two identical unreadable strings are equal, and
    equality was the whole guard.
    """
    with pytest.raises(SchemaViolation):
        recording(
            "ana",
            [annotation(FILLED, 1_000, 2_000, "ana")],
            taxonomy_version="v1",  # type: ignore[arg-type]
        )


def test_a_missing_taxonomy_version_is_still_constructible() -> None:
    """Permitted at construction, refused at comparison.

    A record can legitimately not know its manual - one written before the
    property existed - and the refusal belongs where the number would be
    produced, not where the record is built.
    """
    built = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")], taxonomy_version=None)

    assert built.taxonomy_version is None


def test_the_taxonomy_guard_describes_what_it_actually_does() -> None:
    """A docstring test, which normally is not worth writing.

    Worth it here because this exact docstring has now been a review finding
    twice: the body was rewritten when the rule tightened and the summary line
    was left saying "a minor one becomes a note", so the file documented the
    behaviour it had just stopped having. A reader trusting it would conclude
    the tool tolerates something it refuses.
    """
    doc = report._require_comparable_taxonomies.__doc__ or ""

    assert "becomes a note" not in doc
    assert "Exactly equal" in doc
