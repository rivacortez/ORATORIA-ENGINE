"""The adjudication worklist.

The protocol's adjudication step asks the two annotators to walk every
disagreement together and record each one with its audio position, both
readings, and what was decided. A confusion matrix cannot support that: it says
a `false_start` was read as a `self_repair` three times and not *where*, so the
positions get reconstructed by hand and the ones lost in the reconstruction are
the ones nobody writes down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.agreement.report import AgreementReport, DisagreementKind, compare
from corpus.cli.main import main
from evidence_engine.domain.shared.taxonomy import ContextualRole, SpeechEventType
from tests.corpus.conftest import annotation, recording
from tests.corpus.test_cli import _annotated

FILLED = SpeechEventType.FILLED_PAUSE
PROLONG = SpeechEventType.PROLONGATION
FILLER = SpeechEventType.LEXICAL_FILLER


def _kinds(report: AgreementReport) -> list[str]:
    return [item.kind.value for item in report.disagreements]


# ---------------------------------------------------------------------------
# The four kinds, each with its own fix
# ---------------------------------------------------------------------------


def test_an_event_only_one_annotator_marked_is_listed_against_them() -> None:
    """A detection problem: one of them did not hear it."""
    left = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana"), annotation(FILLED, 8_000, 8_500, "ana")],
    )
    right = recording("beto", [annotation(FILLED, 1_050, 2_050, "beto")])

    report = compare(left, right)

    assert _kinds(report) == ["missed_by_right"]
    missed = report.disagreements[0]
    assert missed.at_ms == 8_000
    assert missed.left is not None
    assert missed.left.annotator == "ana"
    assert missed.right is None


def test_an_event_the_left_annotator_missed_is_distinguished() -> None:
    """Which of the two missed it is the whole content of the finding."""
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording(
        "beto",
        [annotation(FILLED, 1_050, 2_050, "beto"), annotation(FILLED, 8_000, 8_500, "beto")],
    )

    report = compare(left, right)

    assert _kinds(report) == ["missed_by_left"]
    assert report.disagreements[0].left is None
    assert report.disagreements[0].right is not None


def test_a_class_conflict_carries_both_readings() -> None:
    """A taxonomy problem: they both heard it and the manual did not say what
    to call it. The fix is in the manual, not in more training."""
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording("beto", [annotation(PROLONG, 1_000, 2_000, "beto")])

    report = compare(left, right)

    assert _kinds(report) == ["class"]
    conflict = report.disagreements[0]
    assert conflict.left is not None
    assert conflict.right is not None
    assert conflict.left.event_type == "filled_pause"
    assert conflict.right.event_type == "prolongation"
    assert "filled_pause vs prolongation" in conflict.detail


def test_a_role_conflict_is_its_own_kind() -> None:
    """Narrower than a class conflict, and a different section of the manual."""
    left = recording(
        "ana",
        [annotation(FILLER, 1_000, 2_000, "ana", raw_text="este", role=ContextualRole.FILLER)],
    )
    right = recording(
        "beto",
        [annotation(FILLER, 1_000, 2_000, "beto", raw_text="este", role=ContextualRole.SEMANTIC)],
    )

    report = compare(left, right)

    assert _kinds(report) == ["role"]
    assert "filler vs semantic" in report.disagreements[0].detail


def test_a_boundary_conflict_appears_only_past_the_nfr_004_target() -> None:
    """Two people agreeing about an event and drawing it 60 ms apart is what
    NFR-004's 250 ms target is measured against, not a thing to discuss."""
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording("beto", [annotation(FILLED, 1_060, 2_040, "beto")])

    assert compare(left, right).disagreements == ()


def test_a_boundary_past_the_target_is_listed() -> None:
    left = recording("ana", [annotation(FILLED, 1_000, 3_000, "ana")])
    right = recording("beto", [annotation(FILLED, 1_400, 3_050, "beto")])

    report = compare(left, right)

    assert _kinds(report) == ["boundary"]
    assert "400 ms" in report.disagreements[0].detail


def test_the_boundary_threshold_is_adjustable() -> None:
    left = recording("ana", [annotation(FILLED, 1_000, 3_000, "ana")])
    right = recording("beto", [annotation(FILLED, 1_400, 3_050, "beto")])

    assert compare(left, right, boundary_review_ms=500).disagreements == ()
    assert compare(left, right, boundary_review_ms=100).disagreements != ()


# ---------------------------------------------------------------------------
# Shape of the worklist
# ---------------------------------------------------------------------------


def test_one_entry_per_position_not_per_category() -> None:
    """A pair that differs in class *and* in boundary is one thing to discuss.

    Listing it twice makes the adjudication log double-count, and the count is
    what gets reported as "how many disagreements were resolved".
    """
    left = recording("ana", [annotation(FILLED, 1_000, 3_000, "ana")])
    right = recording("beto", [annotation(PROLONG, 1_500, 3_400, "beto")])

    report = compare(left, right)

    assert len(report.disagreements) == 1
    assert report.disagreements[0].kind is DisagreementKind.CLASS


def test_the_worklist_is_in_recording_order() -> None:
    """An adjudicator plays the recording once, front to back."""
    left = recording(
        "ana",
        [
            annotation(FILLED, 20_000, 21_000, "ana"),
            annotation(FILLED, 5_000, 6_000, "ana"),
            annotation(PROLONG, 12_000, 13_000, "ana"),
        ],
    )
    right = recording("beto", [annotation(FILLED, 12_000, 13_000, "beto")])

    report = compare(left, right)

    positions = [item.at_ms for item in report.disagreements]
    assert positions == sorted(positions)
    assert positions == [5_000, 12_000, 20_000]


def test_the_timestamp_is_seekable_in_the_annotation_tool() -> None:
    left = recording("ana", [annotation(FILLED, 83_400, 84_000, "ana")])
    right = recording("beto", [annotation(FILLED, 40_000, 40_500, "beto")])

    stamps = {item.timestamp for item in compare(left, right).disagreements}

    assert stamps == {"01:23.400", "00:40.000"}


def test_agreement_produces_an_empty_worklist() -> None:
    left = recording("ana", [annotation(FILLED, 1_000, 2_000, "ana")])
    right = recording("beto", [annotation(FILLED, 1_020, 2_020, "beto")])

    assert compare(left, right).disagreements == ()


def test_the_worklist_does_not_depend_on_which_file_came_first() -> None:
    """Same reason the matching is symmetric: the log is committed and diffed,
    and a log that reorders itself between runs cannot be diffed."""
    left = recording(
        "ana",
        [annotation(FILLED, 1_000, 2_000, "ana"), annotation(PROLONG, 5_000, 6_000, "ana")],
    )
    right = recording("beto", [annotation(FILLED, 1_050, 2_050, "beto")])

    forward = {(d.kind.value, d.at_ms) for d in compare(left, right).disagreements}
    backward = {(d.kind.value, d.at_ms) for d in compare(right, left).disagreements}

    # `missed_by_right` becomes `missed_by_left` under the swap; the position
    # and the fact that exactly one thing needs adjudicating do not change.
    assert {d[1] for d in forward} == {d[1] for d in backward}
    assert len(forward) == len(backward) == 1


# ---------------------------------------------------------------------------
# Through the command line
# ---------------------------------------------------------------------------


def test_the_json_report_carries_the_worklist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=0, event="prolongation")

    main(["agreement", str(left), str(right), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["disagreements"]["count"] == 1
    item = payload["disagreements"]["items"][0]
    assert item["kind"] == "class"
    assert item["left"]["event_type"] == "filled_pause"
    assert item["right"]["event_type"] == "prolongation"
    assert item["timestamp"] == "00:01.000"


def test_a_missing_reading_is_null_rather_than_an_empty_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty object reads as "they marked something blank here", and the
    whole point of a missed event is that they marked nothing."""
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=30_000, event="filled_pause")

    main(["agreement", str(left), str(right), "--json"])

    payload = json.loads(capsys.readouterr().out)
    kinds = {item["kind"] for item in payload["disagreements"]["items"]}
    assert kinds == {"missed_by_left", "missed_by_right"}
    for item in payload["disagreements"]["items"]:
        assert item["left"] is None or item["right"] is None


def test_the_text_report_lists_the_worklist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=0, event="prolongation")

    main(["agreement", str(left), str(right)])

    output = capsys.readouterr().out
    assert "Disagreements to adjudicate (1)" in output
    assert "00:01.000" in output
    assert "filled_pause vs prolongation" in output


def test_the_text_report_says_none_rather_than_printing_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A silent section is indistinguishable from a section that failed."""
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=20, event="filled_pause")

    main(["agreement", str(left), str(right)])

    output = capsys.readouterr().out
    assert "Disagreements to adjudicate (0)" in output
    assert "none" in output
