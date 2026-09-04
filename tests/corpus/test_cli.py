"""The command line, end to end.

This is the annotators' interface. A broken `corpus template` blocks the pilot
before it starts, and a `corpus agreement` that crashes on the first real pair
of files wastes the session it was meant to measure — so the three verbs are
exercised over real files on disk rather than mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.cli.main import main
from evidence_engine.domain.shared.taxonomy import SpeechEventType


def _template(tmp_path: Path, annotator: str) -> Path:
    path = tmp_path / f"pilot-{annotator}.eaf"
    assert (
        main(
            [
                "template",
                str(path),
                "--recording-id",
                "pilot-001",
                "--speaker",
                "P-001",
                "--annotator",
                annotator,
                "--media",
                "pilot-001.wav",
            ]
        )
        == 0
    )
    return path


def _annotated(tmp_path: Path, annotator: str, *, offset_ms: int, event: str) -> Path:
    """A minimal filled EAF: one word and one annotated event."""
    path = tmp_path / f"filled-{annotator}.eaf"
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="{annotator}" DATE="2026-09-04T00:00:00+00:00"
    FORMAT="3.0" VERSION="3.0">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <PROPERTY NAME="recording_id">pilot-001</PROPERTY>
        <PROPERTY NAME="speaker_pseudonym">P-001</PROPERTY>
        <PROPERTY NAME="annotator_id">{annotator}</PROPERTY>
        <PROPERTY NAME="annotation_pass">first</PROPERTY>
        <PROPERTY NAME="schema_version">1.0.0</PROPERTY>
        <PROPERTY NAME="taxonomy_version">1.0.0</PROPERTY>
    </HEADER>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="900"/>
        <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="{1000 + offset_ms}"/>
        <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="{1800 + offset_ms}"/>
    </TIME_ORDER>
    <TIER LINGUISTIC_TYPE_REF="verbatim" TIER_ID="words">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="w1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>buenos</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="disfluency_class" TIER_ID="disfluency">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="d1" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
                <ANNOTATION_VALUE>{event}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
</ANNOTATION_DOCUMENT>
""",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------


def test_template_writes_a_file_an_annotator_can_open(tmp_path: Path) -> None:
    path = _template(tmp_path, "ana")

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert 'TIER_ID="disfluency"' in content
    assert "CONTROLLED_VOCABULARY" in content


def test_the_template_offers_every_published_class_and_nothing_else(
    tmp_path: Path,
) -> None:
    content = _template(tmp_path, "ana").read_text(encoding="utf-8")

    for event in SpeechEventType:
        assert f">{event.value}<" in content
    # The thing that must never be selectable.
    assert "nervousness" not in content
    assert "anxiety" not in content


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_accepts_a_well_formed_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")

    assert main(["validate", str(path)]) == 0
    assert "pilot-001" in capsys.readouterr().out


def test_validate_reports_json_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The disagreement log needs numbers that can be committed and diffed."""
    path = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")

    main(["validate", str(path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["recording_id"] == "pilot-001"
    assert payload[0]["usable"] is True


def test_validate_exits_nonzero_on_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty words tier: lexical classes cannot be checked against anything."""
    path = tmp_path / "no-words.eaf"
    path.write_text(
        _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
        .read_text(encoding="utf-8")
        .replace('TIER_ID="words"', 'TIER_ID="unused"'),
        encoding="utf-8",
    )

    assert main(["validate", str(path)]) == 1


def test_a_parse_failure_names_the_file_rather_than_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An annotator needs to know which file, not that something went wrong."""
    path = tmp_path / "broken.eaf"
    path.write_text("<ANNOTATION_DOCUMENT><unclosed>", encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    assert "broken.eaf" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------


def test_agreement_reports_all_three_stages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=80, event="filled_pause")

    assert main(["agreement", str(left), str(right)]) == 0

    output = capsys.readouterr().out
    assert "Did they find the same events?" in output
    assert "Did they draw the same boundaries?" in output
    assert "Did they give them the same label?" in output
    assert "NFR-004" in output


def test_agreement_prints_undefined_rather_than_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both annotators used one class, so kappa has no chance agreement to
    correct for. Printing 0.0 is how "the sample cannot answer this" becomes
    "they agreed no better than chance" in somebody's summary table."""
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=50, event="filled_pause")

    main(["agreement", str(left), str(right)])

    assert "undefined" in capsys.readouterr().out


def test_agreement_json_carries_the_matching_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A coefficient without its threshold cannot be reproduced."""
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=80, event="filled_pause")

    main(["agreement", str(left), str(right), "--json", "--iou", "0.4"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["matching"]["criterion"] == "iou"
    assert payload["matching"]["threshold"] == pytest.approx(0.4)
    assert payload["matching"]["matched"] == 1


def test_a_class_disagreement_shows_as_a_matched_event(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """They heard the same thing and named it differently — a taxonomy problem,
    not a detection one, and the report has to keep them apart."""
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=0, event="prolongation")

    main(["agreement", str(left), str(right), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["matching"]["matched"] == 1
    assert payload["matching"]["positive_specific_agreement"] == pytest.approx(1.0)
    assert payload["class_agreement"]["confusion"] == {"filled_pause|prolongation": 1}


def test_the_tolerance_criterion_is_selectable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    right = _annotated(tmp_path, "beto", offset_ms=200, event="filled_pause")

    main(
        [
            "agreement",
            str(left),
            str(right),
            "--criterion",
            "midpoint_tolerance",
            "--tolerance-ms",
            "250",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["matching"]["criterion"] == "midpoint_tolerance"
    assert payload["matching"]["matched"] == 1


# ---------------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------------


def test_no_subcommand_prints_help_and_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 2
    assert "template" in capsys.readouterr().out
