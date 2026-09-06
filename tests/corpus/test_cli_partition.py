"""The corpus-construction verbs, over real files on disk.

`inventory`, `split` and `verify` are what a recording schedule is driven by,
so their exit codes are part of the interface and not an afterthought: a
methodologist runs `corpus inventory` weekly and stops recording when it
returns zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.cli.main import main
from corpus.partition.freeze import MANIFEST_VERSION
from evidence_engine.domain.shared.taxonomy import SpeechEventType, p0_speech_events

CLASSES = tuple(sorted(p0_speech_events(), key=lambda e: e.value))


def _eaf(path: Path, *, recording_id: str, speaker: str, events: list[str]) -> Path:
    """A filled EAF, written literally.

    The same shape the Pilot A rehearsal uses, and for the same reason:
    generating the input with the code that reads it would check the reader
    against itself.
    """
    slots: list[str] = [
        '<TIME_SLOT TIME_SLOT_ID="w1s" TIME_VALUE="0"/>',
        '<TIME_SLOT TIME_SLOT_ID="w1e" TIME_VALUE="500"/>',
    ]
    words = (
        '<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="w1" '
        'TIME_SLOT_REF1="w1s" TIME_SLOT_REF2="w1e">'
        "<ANNOTATION_VALUE>este</ANNOTATION_VALUE>"
        "</ALIGNABLE_ANNOTATION></ANNOTATION>"
    )

    tiers: list[str] = []
    roles: list[str] = []
    texts: list[str] = []
    lexical = {"lexical_filler", "repetition", "false_start", "self_repair"}
    cursor = 1_000
    for index, event in enumerate(events, start=1):
        slots.append(f'<TIME_SLOT TIME_SLOT_ID="d{index}s" TIME_VALUE="{cursor}"/>')
        slots.append(f'<TIME_SLOT TIME_SLOT_ID="d{index}e" TIME_VALUE="{cursor + 300}"/>')
        cursor += 500
        tiers.append(
            f'<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="d{index}" '
            f'TIME_SLOT_REF1="d{index}s" TIME_SLOT_REF2="d{index}e">'
            f"<ANNOTATION_VALUE>{event}</ANNOTATION_VALUE>"
            f"</ALIGNABLE_ANNOTATION></ANNOTATION>"
        )
        if event in lexical:
            roles.append(
                f'<ANNOTATION><REF_ANNOTATION ANNOTATION_ID="r{index}" '
                f'ANNOTATION_REF="d{index}">'
                f"<ANNOTATION_VALUE>filler</ANNOTATION_VALUE>"
                f"</REF_ANNOTATION></ANNOTATION>"
            )
            texts.append(
                f'<ANNOTATION><REF_ANNOTATION ANNOTATION_ID="t{index}" '
                f'ANNOTATION_REF="d{index}">'
                f"<ANNOTATION_VALUE>este</ANNOTATION_VALUE>"
                f"</REF_ANNOTATION></ANNOTATION>"
            )

    role_tier = (
        '<TIER LINGUISTIC_TYPE_REF="contextual_role" PARENT_REF="disfluency" '
        f'TIER_ID="role">{"".join(roles)}</TIER>'
    )
    text_tier = (
        '<TIER LINGUISTIC_TYPE_REF="attached_text" PARENT_REF="disfluency" '
        f'TIER_ID="text">{"".join(texts)}</TIER>'
    )

    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="ana" DATE="2026-09-04T00:00:00+00:00"
    FORMAT="3.0" VERSION="3.0">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <PROPERTY NAME="recording_id">{recording_id}</PROPERTY>
        <PROPERTY NAME="speaker_pseudonym">{speaker}</PROPERTY>
        <PROPERTY NAME="annotator_id">ana</PROPERTY>
        <PROPERTY NAME="annotation_pass">first</PROPERTY>
        <PROPERTY NAME="schema_version">2.0.0</PROPERTY>
        <PROPERTY NAME="taxonomy_version">1.0.0</PROPERTY>
        <PROPERTY NAME="speaker_variety">es-PE</PROPERTY>
        <PROPERTY NAME="consent_basis">written_informed</PROPERTY>
        <PROPERTY NAME="consent_policy_version">1.0.0</PROPERTY>
        <PROPERTY NAME="consent_granted_on">2026-09-01</PROPERTY>
        <PROPERTY NAME="consent_covers_video">false</PROPERTY>
        <PROPERTY NAME="microphone">Realtek(R) Audio - onboard array</PROPERTY>
        <PROPERTY NAME="sample_rate_hz">16000</PROPERTY>
        <PROPERTY NAME="bit_depth">16</PROPERTY>
        <PROPERTY NAME="channels">1</PROPERTY>
        <PROPERTY NAME="virtual_audio_bypassed">true</PROPERTY>
    </HEADER>
    <TIME_ORDER>{"".join(slots)}</TIME_ORDER>
    <TIER LINGUISTIC_TYPE_REF="verbatim" TIER_ID="words">{words}</TIER>
    <TIER LINGUISTIC_TYPE_REF="disfluency_class" TIER_ID="disfluency">{"".join(tiers)}</TIER>
    {role_tier}
    {text_tier}
</ANNOTATION_DOCUMENT>
""",
        encoding="utf-8",
    )
    return path


def _corpus_files(tmp_path: Path, speakers: int, *, per_class: int = 4) -> list[str]:
    """One recording per speaker, every class present."""
    events = [event.value for event in CLASSES for _ in range(per_class)]
    return [
        str(
            _eaf(
                tmp_path / f"rec-{i:03d}.eaf",
                recording_id=f"rec-{i:03d}",
                speaker=f"P-{i:03d}",
                events=events,
            )
        )
        for i in range(1, speakers + 1)
    ]


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_inventory_reports_every_class(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = _corpus_files(tmp_path, 8, per_class=5)

    main(["inventory", *files, "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["recordings"] == 8
    assert payload["speakers"] == 8
    assert {c["event_type"] for c in payload["classes"]} == {e.value for e in p0_speech_events()}


def test_inventory_exits_non_zero_while_the_corpus_is_short(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command a recording schedule is driven by: 'keep going' has to be
    machine-readable."""
    files = _corpus_files(tmp_path, 3, per_class=1)

    assert main(["inventory", *files]) == 1
    assert "SHORT" in capsys.readouterr().out


def test_inventory_exits_zero_once_the_thresholds_are_met(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _corpus_files(tmp_path, 6, per_class=6)

    assert main(["inventory", *files, "--min-instances", "30", "--min-speakers", "5"]) == 0


def test_inventory_prints_the_thresholds_it_judged_against(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A verdict without its floor cannot be reproduced or argued with."""
    files = _corpus_files(tmp_path, 6, per_class=6)

    main(["inventory", *files, "--min-instances", "12", "--min-speakers", "4"])

    output = capsys.readouterr().out
    assert "12 instance(s)" in output
    assert "4 speaker(s)" in output


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_assigns_every_speaker_exactly_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _corpus_files(tmp_path, 13)

    assert main(["split", *files, "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    placements = [s for p in payload["partitions"] for s in p["speakers"]]
    assert sorted(placements) == sorted(set(placements))
    assert len(placements) == 13


def test_split_is_reproducible_from_the_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _corpus_files(tmp_path, 13)

    main(["split", *files, "--seed", "3", "--json"])
    first = json.loads(capsys.readouterr().out)
    main(["split", *files, "--seed", "3", "--json"])
    second = json.loads(capsys.readouterr().out)

    assert first["partitions"] == second["partitions"]


def test_split_refuses_to_freeze_an_unusable_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two speakers produce `cut_off` and there are three partitions.

    Freezing that would give the missing measurement the appearance of a
    decision: the manifest gets committed and cited, and nobody reading it
    later knows the class was absent rather than deliberately excluded.
    """
    files = [
        str(
            _eaf(
                tmp_path / f"rare-{i}.eaf",
                recording_id=f"rare-{i}",
                speaker=f"R-{i:03d}",
                events=[SpeechEventType.CUT_OFF.value] * 4 + ["filled_pause"] * 4,
            )
        )
        for i in range(1, 3)
    ]
    files += [
        str(
            _eaf(
                tmp_path / f"common-{i}.eaf",
                recording_id=f"common-{i}",
                speaker=f"C-{i:03d}",
                events=["filled_pause"] * 8,
            )
        )
        for i in range(1, 12)
    ]
    manifest = tmp_path / "held-out.json"

    assert main(["split", *files, "--freeze", str(manifest)]) == 1
    assert not manifest.exists()
    assert "undefined per-class figure" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# freeze and verify, end to end
# ---------------------------------------------------------------------------


def test_freeze_then_verify_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = _corpus_files(tmp_path, 13)
    manifest = tmp_path / "held-out.json"

    assert main(["split", *files, "--freeze", str(manifest)]) == 0
    assert manifest.exists()
    capsys.readouterr()

    assert main(["verify", str(manifest), *files]) == 0
    assert "unchanged since the freeze" in capsys.readouterr().out


def test_verify_catches_a_file_changed_after_the_freeze(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure a freeze exists to make visible: regenerating the held-out
    set after seeing a disappointing number."""
    files = _corpus_files(tmp_path, 13)
    manifest = tmp_path / "held-out.json"
    main(["split", *files, "--freeze", str(manifest)])
    capsys.readouterr()

    tampered = Path(files[0])
    tampered.write_text(
        tampered.read_text(encoding="utf-8").replace(
            "<ANNOTATION_VALUE>filled_pause</ANNOTATION_VALUE>",
            "<ANNOTATION_VALUE>prolongation</ANNOTATION_VALUE>",
            1,
        ),
        encoding="utf-8",
    )

    assert main(["verify", str(manifest), *files]) == 1
    assert "has changed since the freeze" in capsys.readouterr().out


def test_verify_catches_a_recording_added_after_the_freeze(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _corpus_files(tmp_path, 13)
    manifest = tmp_path / "held-out.json"
    main(["split", *files, "--freeze", str(manifest)])
    capsys.readouterr()

    latecomer = str(
        _eaf(
            tmp_path / "rec-999.eaf",
            recording_id="rec-999",
            speaker="P-999",
            events=["filled_pause"] * 4,
        )
    )

    assert main(["verify", str(manifest), *files, latecomer]) == 1
    assert "not in the manifest" in capsys.readouterr().out


def test_the_freeze_refuses_to_overwrite_a_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-freezing after seeing a result is the failure the manifest exists to
    make visible, so it cannot be the default."""
    files = _corpus_files(tmp_path, 13)
    manifest = tmp_path / "held-out.json"
    main(["split", *files, "--freeze", str(manifest)])
    original = manifest.read_text(encoding="utf-8")
    capsys.readouterr()

    assert main(["split", *files, "--seed", "5", "--freeze", str(manifest)]) == 1
    assert manifest.read_text(encoding="utf-8") == original
    assert "--force" in capsys.readouterr().err

    assert main(["split", *files, "--seed", "5", "--freeze", str(manifest), "--force"]) == 0


def test_the_manifest_is_committable_json(tmp_path: Path) -> None:
    """It goes in the repository beside the dataset card, so it has to diff."""
    files = _corpus_files(tmp_path, 13)
    manifest = tmp_path / "held-out.json"
    main(["split", *files, "--freeze", str(manifest)])

    payload = json.loads(manifest.read_text(encoding="utf-8"))

    # Read from the module rather than pinned to a literal: the manifest
    # gained fields when the schema did, and a test asserting the old
    # number fails for the version bump rather than for the manifest.
    assert payload["manifest_version"] == str(MANIFEST_VERSION)
    assert payload["taxonomy_version"] == "1.0.0"
    assert len(payload["digest"]) == 64
    assert {r["partition"] for r in payload["recordings"]} == {"train", "dev", "held_out"}
    assert manifest.read_text(encoding="utf-8").endswith("\n")
