"""Fixtures and inspections for the Pilot A rehearsal.

Split out of ``rehearse_pilot_a.sh`` because building an EAF and comparing two
JSON reports in shell is how a rehearsal script acquires bugs of its own. The
shell script sequences the commands an annotator runs; this does the parts that
need to read a file.

Everything here writes into a temporary directory the shell script owns. None
of it is importable by the service - it lives in ``scripts``, which C7 and C8
keep out of both trees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corpus.io.elan import PROP_SCHEMA, PROP_TAXONOMY, read
from corpus.schema.records import LEXICAL_CLASSES, AnnotationPass
from evidence_engine.domain.shared.taxonomy import (
    PROHIBITED_CONCEPTS,
    ContextualRole,
    SpeechEventType,
    VisualEventType,
)

#: The class values that need a contextual role and a raw expression.
_LEXICAL = frozenset(event.value for event in LEXICAL_CLASSES)


def _eaf(
    *,
    annotator: str,
    events: list[tuple[str, int, int]],
    recording_id: str = "pilot-a-001",
    annotation_pass: str = "first",
    schema_version: str = "2.0.0",
    taxonomy_version: str = "1.0.0",
) -> str:
    """A filled EAF, written literally rather than through the writer.

    Deliberate: the writer only produces empty templates, and a rehearsal that
    generated its input with the same code that reads it would be checking the
    code against itself.
    """
    slots: list[str] = []
    words: list[str] = []
    tiers: list[str] = []
    roles: list[str] = []

    # A words tier the lexical annotations can be checked against. Without it
    # the validator reports `no_transcription` and `compare` refuses the file -
    # which is correct, and would make this rehearsal test the refusal.
    transcript = [("buenos", 0, 500), ("dias", 500, 900), ("este", 1_000, 1_400)]
    for index, (text, start, end) in enumerate(transcript, start=1):
        slots.append(f'<TIME_SLOT TIME_SLOT_ID="w{index}s" TIME_VALUE="{start}"/>')
        slots.append(f'<TIME_SLOT TIME_SLOT_ID="w{index}e" TIME_VALUE="{end}"/>')
        words.append(
            f'<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="w{index}" '
            f'TIME_SLOT_REF1="w{index}s" TIME_SLOT_REF2="w{index}e">'
            f"<ANNOTATION_VALUE>{text}</ANNOTATION_VALUE>"
            f"</ALIGNABLE_ANNOTATION></ANNOTATION>"
        )

    for index, (event, start, end) in enumerate(events, start=1):
        slots.append(f'<TIME_SLOT TIME_SLOT_ID="d{index}s" TIME_VALUE="{start}"/>')
        slots.append(f'<TIME_SLOT TIME_SLOT_ID="d{index}e" TIME_VALUE="{end}"/>')
        tiers.append(
            f'<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="d{index}" '
            f'TIME_SLOT_REF1="d{index}s" TIME_SLOT_REF2="d{index}e">'
            f"<ANNOTATION_VALUE>{event}</ANNOTATION_VALUE>"
            f"</ALIGNABLE_ANNOTATION></ANNOTATION>"
        )
        # Every lexical class, not just `lexical_filler`: FR-013 and FR-017
        # require a role and a raw expression for all four, and a fixture that
        # supplies them for one is a fixture the reader rejects.
        if event in _LEXICAL:
            roles.append(
                f'<ANNOTATION><REF_ANNOTATION ANNOTATION_ID="r{index}" '
                f'ANNOTATION_REF="d{index}">'
                f"<ANNOTATION_VALUE>{ContextualRole.FILLER.value}</ANNOTATION_VALUE>"
                f"</REF_ANNOTATION></ANNOTATION>"
            )

    texts = "".join(
        f'<ANNOTATION><REF_ANNOTATION ANNOTATION_ID="t{index}" ANNOTATION_REF="d{index}">'
        f"<ANNOTATION_VALUE>este</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION>"
        for index, (event, _, _) in enumerate(events, start=1)
        if event in _LEXICAL
    )

    role_tier = (
        '<TIER LINGUISTIC_TYPE_REF="contextual_role" PARENT_REF="disfluency" '
        f'TIER_ID="role">{"".join(roles)}</TIER>'
    )
    text_tier = (
        '<TIER LINGUISTIC_TYPE_REF="attached_text" PARENT_REF="disfluency" '
        f'TIER_ID="text">{texts}</TIER>'
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="{annotator}" DATE="2026-09-04T00:00:00+00:00"
    FORMAT="3.0" VERSION="3.0">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <PROPERTY NAME="recording_id">{recording_id}</PROPERTY>
        <PROPERTY NAME="speaker_pseudonym">P-001</PROPERTY>
        <PROPERTY NAME="annotator_id">{annotator}</PROPERTY>
        <PROPERTY NAME="annotation_pass">{annotation_pass}</PROPERTY>
        <PROPERTY NAME="{PROP_SCHEMA}">{schema_version}</PROPERTY>
        <PROPERTY NAME="{PROP_TAXONOMY}">{taxonomy_version}</PROPERTY>
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
    <TIER LINGUISTIC_TYPE_REF="verbatim" TIER_ID="words">{"".join(words)}</TIER>
    <TIER LINGUISTIC_TYPE_REF="disfluency_class" TIER_ID="disfluency">{"".join(tiers)}</TIER>
    {role_tier}
    {text_tier}
</ANNOTATION_DOCUMENT>
"""


def inspect_template(path: Path) -> None:
    """What an annotator sees when they open it."""
    content = path.read_text(encoding="utf-8")

    published = [e.value for e in SpeechEventType if f">{e.value}<" in content]
    print(f"  published speech classes offered : {len(published)}/{len(list(SpeechEventType))}")

    missing = [e.value for e in SpeechEventType if f">{e.value}<" not in content]
    print(f"  missing                          : {missing or 'none'}")

    visual = [e.value for e in VisualEventType if f">{e.value}<" in content]
    print(f"  visual classes offered           : {visual or 'none (Phase 0.5 is speech-only)'}")

    leaked = sorted(c for c in PROHIBITED_CONCEPTS if c in content.lower())
    print(f"  prohibited concepts present      : {leaked or 'none'}")

    roles = [r.value for r in ContextualRole if f">{r.value}<" in content]
    print(f"  contextual roles offered         : {len(roles)}/{len(list(ContextualRole))}")

    print(f"  schema_version recorded          : {PROP_SCHEMA in content}")
    print(f"  taxonomy_version recorded        : {PROP_TAXONOMY in content}")
    print(f"  role tier is a child of the event: {'PARENT_REF="disfluency"' in content}")

    if missing or visual or leaked:
        raise SystemExit("the template does not offer exactly the published speech taxonomy")


def write_annotated(work: Path) -> None:
    """Two annotators' files over one recording, with real disagreement in them.

    Chosen to exercise each of the four disagreement kinds, because a rehearsal
    over two identical files would prove only that the report renders when
    there is nothing to report.
    """
    ana = [
        (SpeechEventType.FILLED_PAUSE.value, 1_000, 1_800),  # both, boundaries differ
        (SpeechEventType.LEXICAL_FILLER.value, 3_000, 3_400),  # both, same
        (SpeechEventType.FALSE_START.value, 5_000, 5_600),  # class conflict
        (SpeechEventType.CUT_OFF.value, 8_000, 8_300),  # only ana
    ]
    beto = [
        (SpeechEventType.FILLED_PAUSE.value, 1_050, 2_400),  # 600 ms off at the end
        (SpeechEventType.LEXICAL_FILLER.value, 3_000, 3_400),
        (SpeechEventType.SELF_REPAIR.value, 5_050, 5_650),  # read as self_repair
        (SpeechEventType.PROLONGATION.value, 11_000, 11_500),  # only beto
    ]
    (work / "annotated-ana.eaf").write_text(_eaf(annotator="ana", events=ana), encoding="utf-8")
    (work / "annotated-beto.eaf").write_text(_eaf(annotator="beto", events=beto), encoding="utf-8")

    for name in ("annotated-ana.eaf", "annotated-beto.eaf"):
        record = read(work / name)
        print(
            f"  {name:<22} {len(record.disfluencies)} events, "
            f"{len(record.words)} words, schema {record.schema_version}, "
            f"taxonomy {record.taxonomy_version}, pass {record.annotation_pass.value}"
        )


def write_refusable(work: Path) -> None:
    """One file per refusal, so each guard is exercised against a real file."""
    events = [(SpeechEventType.FILLED_PAUSE.value, 1_000, 1_800)]

    (work / "adjudicated.eaf").write_text(
        _eaf(
            annotator="beto",
            events=events,
            annotation_pass=AnnotationPass.ADJUDICATED.value,
        ),
        encoding="utf-8",
    )
    (work / "future-schema.eaf").write_text(
        _eaf(annotator="beto", events=events, schema_version="2.0.0"), encoding="utf-8"
    )
    (work / "other-taxonomy.eaf").write_text(
        _eaf(annotator="beto", events=events, taxonomy_version="2.0.0"), encoding="utf-8"
    )
    # A *minor* difference, which used to be tolerated with a note. The
    # protocol says any taxonomy change requires re-running the pilot, so the
    # rehearsal exercises the difference the code used to let through rather
    # than only the one it always caught.
    (work / "minor-taxonomy.eaf").write_text(
        _eaf(annotator="beto", events=events, taxonomy_version="1.1.0"), encoding="utf-8"
    )
    # No taxonomy version at all. This one is refused at read time: it used to
    # become `None`, and `None` compares unequal to nothing, so the file walked
    # past the guard written to compare manuals.
    (work / "no-taxonomy.eaf").write_text(
        _eaf(annotator="beto", events=events).replace(f'NAME="{PROP_TAXONOMY}"', 'NAME="unused"'),
        encoding="utf-8",
    )
    (work / "overlapping.eaf").write_text(
        _eaf(
            annotator="beto",
            events=[
                (SpeechEventType.FILLED_PAUSE.value, 1_000, 1_800),
                (SpeechEventType.FILLED_PAUSE.value, 1_400, 2_200),
            ],
        ),
        encoding="utf-8",
    )
    print("  wrote adjudicated, future-schema, other-taxonomy, minor-taxonomy,")
    print("  no-taxonomy and overlapping files")


def compare_directions(work: Path) -> None:
    """The report read one way and the other way must agree.

    The defect this checks for did not crash and did not warn: it produced a
    different confusion matrix depending on which filename came first.
    """
    forward = json.loads((work / "forward.json").read_text(encoding="utf-8"))
    backward = json.loads((work / "backward.json").read_text(encoding="utf-8"))

    def transposed(confusion: dict[str, int]) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, count in confusion.items():
            left, right = key.split("|")
            out[f"{right}|{left}"] = count
        return out

    checks = [
        (
            "matched count",
            forward["matching"]["matched"],
            backward["matching"]["matched"],
        ),
        (
            "positive specific agreement",
            round(forward["matching"]["positive_specific_agreement"], 12),
            round(backward["matching"]["positive_specific_agreement"], 12),
        ),
        (
            "start boundary median",
            forward["boundary_error_ms"]["start_median"],
            backward["boundary_error_ms"]["start_median"],
        ),
        (
            "end boundary p95",
            forward["boundary_error_ms"]["end_p95"],
            backward["boundary_error_ms"]["end_p95"],
        ),
        (
            "class confusion matrix",
            forward["class_agreement"]["confusion"],
            transposed(backward["class_agreement"]["confusion"]),
        ),
        (
            "disagreement count",
            forward["disagreements"]["count"],
            backward["disagreements"]["count"],
        ),
        (
            "disagreement positions",
            sorted(i["at_ms"] for i in forward["disagreements"]["items"]),
            sorted(i["at_ms"] for i in backward["disagreements"]["items"]),
        ),
    ]

    failed = 0
    for label, a, b in checks:
        if a == b:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}: {a!r} vs {b!r}")
            failed += 1

    if failed:
        raise SystemExit(f"{failed} figure(s) depend on argument order")


def show_worklist(path: Path) -> None:
    """The adjudication list as the methodologist will read it."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    worklist = payload["disagreements"]

    print(f"  boundary review threshold : {worklist['boundary_review_ms']} ms (NFR-004)")
    print(f"  positions to adjudicate   : {worklist['count']}")
    for item in worklist["items"]:
        left = item["left"]
        right = item["right"]
        print(f"    {item['timestamp']}  {item['kind']:<16} {item['detail']}")
        print(f"      left  : {_reading(left)}")
        print(f"      right : {_reading(right)}")

    kinds = {item["kind"] for item in worklist["items"]}
    print(f"  kinds exercised           : {sorted(kinds)}")
    if not worklist["count"]:
        raise SystemExit("the fixture was built to disagree and the report found nothing")


def _reading(reading: dict[str, object] | None) -> str:
    if reading is None:
        return "(nothing here)"
    role = f"/{reading['context_role']}" if reading["context_role"] else ""
    return (
        f"{reading['annotator']} {reading['event_type']}{role} "
        f"[{reading['start_ms']}-{reading['end_ms']}]"
    )


def main(argv: list[str]) -> int:
    command, argument = argv[1], Path(argv[2])
    {
        "inspect-template": inspect_template,
        "write-annotated": write_annotated,
        "write-refusable": write_refusable,
        "compare-directions": compare_directions,
        "show-worklist": show_worklist,
    }[command](argument)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
