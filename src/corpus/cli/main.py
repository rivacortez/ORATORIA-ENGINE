"""``corpus`` — the annotators' and the methodologist's command line.

Six verbs, in the order the work happens.

*During the pilots (Phase 0.5).* ``template`` hands an annotator a file,
``validate`` checks what came back, ``agreement`` measures two of them against
each other and lists what has to be adjudicated.

``template`` takes more flags than looks comfortable, and the discomfort is the
point. The speaker's variety, the consent basis and policy version, and the
capture chain are all captured at recruitment or never: forty speakers recorded
without them cannot be sliced by dialect, cannot support a consent audit, and
cannot be stratified by microphone. None of them defaults, because a default
would be this tool asserting on the operator's behalf something only the
operator can know - and the assertion would be indistinguishable, afterwards,
from one somebody actually made.

*While building the corpus (Phase 1).* ``inventory`` says whether there is
enough of each class from enough different speakers to carry a per-class
figure, and exits non-zero while there is not - it is the command a recording
schedule is driven by. ``split`` assigns speakers to train/dev/held-out
speaker-independently and, with ``--freeze``, writes the manifest that makes
§14.4's gate checkable. ``verify`` answers, months later, whether the held-out
set is still the one a number was computed over.

Output is plain text on stdout and machine-readable JSON behind ``--json``.
Both, because the same numbers are read two ways: a methodologist reads them
during the pilot, and the disagreement log and the dataset card need them in a
form that can be committed and diffed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from corpus.agreement.matching import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TOLERANCE_MS,
    InvalidMatchParameters,
    MatchCriterion,
)
from corpus.agreement.report import (
    AgreementReport,
    InvalidReportParameters,
    Reading,
    RefusedComparison,
    compare,
)
from corpus.io.elan import ElanError, read, write_template
from corpus.partition.freeze import FreezeError, freeze, from_json, to_json, verify
from corpus.partition.inventory import (
    DEFAULT_MINIMUM_INSTANCES,
    DEFAULT_MINIMUM_SPEAKERS,
    InventoryError,
    inventory,
)
from corpus.partition.split import split
from corpus.schema.records import (
    AnnotatedRecording,
    AnnotationPass,
    ConsentBasis,
    ConsentRecord,
    RecordingConditions,
    SchemaViolation,
)
from corpus.schema.validation import validate
from evidence_engine.domain.shared.provenance import SemanticVersion


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return int(handler(args))
    except ElanError as error:
        # A parse failure names the file and the position. An annotator seeing
        # "raw_text is required" needs to know which of four hundred
        # annotations it is about, not that something somewhere is wrong.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except RefusedComparison as error:
        # Distinguished from a parse failure by exit code 3, so that a script
        # driving the pilot can tell "these files are unreadable" from "these
        # files are readable and must not be compared".
        print(f"refused: {error}", file=sys.stderr)
        return 3
    except (InvalidMatchParameters, InvalidReportParameters, SchemaViolation) as error:
        # Exit 2, argparse's code for a usage error, because that is what this
        # is: the files are fine and the numbers asked for are not. Caught here
        # rather than left to propagate - `corpus agreement --iou 0` used to
        # print a Python traceback at an annotator, which reads as "the tool is
        # broken" rather than "that threshold means nothing".
        #
        # `SchemaViolation` joins them because the values that build a consent
        # record and a set of recording conditions arrive as flags: `corpus
        # template --channels 0` is the same class of mistake as `--iou 0`.
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (FreezeError, InventoryError) as error:
        # Exit 1, matching `_split`'s own refusal to freeze an unusable plan.
        # These used to escape as tracebacks because they were unreachable in
        # practice - a version disagreement across a whole corpus is rare. The
        # consent and capture-chain refusals are not rare: they fire the first
        # time somebody forgets to confirm that Voicemeeter was out of the path,
        # and a traceback at that moment reads as a broken tool.
        print(f"refused: {error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus",
        description="Annotation templates, validation and inter-annotator agreement.",
    )
    subcommands = parser.add_subparsers(dest="command")

    template = subcommands.add_parser(
        "template",
        help="write an empty ELAN file wired with the taxonomy as a controlled vocabulary",
    )
    template.add_argument("output", type=Path)
    template.add_argument("--recording-id", required=True)
    template.add_argument("--speaker", required=True, help="pseudonym, never a real name")
    template.add_argument(
        "--variety",
        required=True,
        help="language tag such as es-PE; §14.2 asks for error analysis by dialect",
    )
    template.add_argument("--annotator", required=True)
    template.add_argument("--media", required=True, help="path or URL of the audio or video")
    # Consent and capture conditions. Every one of them is `required=True`, and
    # the omission is deliberate in each case: a default would be this tool
    # asserting, on nobody's behalf, something only the operator can know. The
    # consent policy version is not read from the repository for the same
    # reason - the participant read a printed wording on a day, and a constant
    # in code would attest to whatever was checked in at template time.
    template.add_argument(
        "--consent-basis", required=True, choices=[item.value for item in ConsentBasis]
    )
    template.add_argument(
        "--consent-policy",
        required=True,
        type=_semantic_version,
        metavar="X.Y.Z",
        help="the published version of CONSENT_AND_RETENTION.md the participant read",
    )
    template.add_argument("--consent-granted", required=True, type=_iso_date, metavar="YYYY-MM-DD")
    template.add_argument(
        "--consent-covers-video",
        action="store_true",
        help="video is opt-in (policy §6); without this the consent covers audio only",
    )
    template.add_argument(
        "--microphone", required=True, help="the physical device as it enumerates"
    )
    template.add_argument("--sample-rate-hz", required=True, type=int)
    template.add_argument("--bit-depth", required=True, type=int)
    template.add_argument("--channels", required=True, type=int)
    template.add_argument(
        "--virtual-audio-bypassed",
        required=True,
        choices=["yes", "no"],
        help=(
            "NVIDIA Broadcast and Voicemeeter out of the path, confirmed from the "
            "recorded file rather than a settings dialog"
        ),
    )
    template.add_argument("--room-notes", default="")
    template.add_argument(
        "--pass",
        dest="annotation_pass",
        choices=[item.value for item in AnnotationPass],
        default=AnnotationPass.FIRST.value,
    )
    template.add_argument(
        "--force",
        action="store_true",
        help="replace the file if it exists; without this, an existing file is refused",
    )
    template.set_defaults(handler=_template)

    check = subcommands.add_parser("validate", help="check one or more annotation files")
    check.add_argument("files", nargs="+", type=Path)
    check.add_argument("--json", action="store_true", dest="as_json")
    check.set_defaults(handler=_validate)

    agreement = subcommands.add_parser(
        "agreement", help="measure agreement between two annotations of one recording"
    )
    agreement.add_argument("left", type=Path)
    agreement.add_argument("right", type=Path)
    agreement.add_argument(
        "--criterion",
        choices=[item.value for item in MatchCriterion],
        default=MatchCriterion.IOU.value,
    )
    agreement.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    agreement.add_argument("--tolerance-ms", type=int, default=DEFAULT_TOLERANCE_MS)
    agreement.add_argument(
        "--boundary-review-ms",
        type=int,
        default=DEFAULT_TOLERANCE_MS,
        help=(
            "list a matched pair for adjudication when its boundaries differ by more "
            "than this (default: NFR-004's 250 ms)"
        ),
    )
    agreement.add_argument("--json", action="store_true", dest="as_json")
    agreement.set_defaults(handler=_agreement)

    stock = subcommands.add_parser(
        "inventory",
        help="count the corpus by class and by speaker, and say whether it is enough",
    )
    stock.add_argument("files", nargs="+", type=Path)
    stock.add_argument("--min-instances", type=int, default=DEFAULT_MINIMUM_INSTANCES)
    stock.add_argument(
        "--min-speakers",
        type=int,
        default=DEFAULT_MINIMUM_SPEAKERS,
        help="a class from too few speakers is one person's habit, not evidence",
    )
    stock.add_argument("--json", action="store_true", dest="as_json")
    stock.set_defaults(handler=_inventory)

    partition = subcommands.add_parser(
        "split", help="assign speakers to train/dev/held-out, speaker-independently"
    )
    partition.add_argument("files", nargs="+", type=Path)
    partition.add_argument("--seed", type=int, default=0)
    partition.add_argument("--json", action="store_true", dest="as_json")
    partition.add_argument(
        "--freeze",
        type=Path,
        metavar="MANIFEST",
        help="write the manifest that makes the held-out set checkable (§14.4)",
    )
    partition.add_argument("--force", action="store_true", help="overwrite the manifest")
    partition.set_defaults(handler=_split)

    check_freeze = subcommands.add_parser(
        "verify", help="check a frozen corpus against the files on disk"
    )
    check_freeze.add_argument("manifest", type=Path)
    check_freeze.add_argument("files", nargs="+", type=Path)
    check_freeze.add_argument("--json", action="store_true", dest="as_json")
    check_freeze.set_defaults(handler=_verify)

    return parser


def _iso_date(text: str) -> date:
    """A date argument, refused by argparse rather than by a traceback.

    Parsed with ``type=`` so a mistyped date exits 2 with a usage message. The
    alternative - parsing inside the handler - turns ``--consent-granted
    2026-13-01`` into a ``ValueError`` traceback, which reads as a broken tool
    at exactly the moment an operator is entering forty participants.
    """
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an ISO-8601 date such as 2026-09-04"
        ) from error


def _semantic_version(text: str) -> SemanticVersion:
    try:
        return SemanticVersion.parse(text)
    except Exception as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a version such as 1.0.0") from error


def _template(args: argparse.Namespace) -> int:
    write_template(
        args.output,
        recording_id=args.recording_id,
        speaker_pseudonym=args.speaker,
        speaker_variety=args.variety,
        annotator_id=args.annotator,
        media_url=args.media,
        consent=ConsentRecord(
            basis=ConsentBasis(args.consent_basis),
            policy_version=args.consent_policy,
            granted_on=args.consent_granted,
            covers_video=args.consent_covers_video,
        ),
        conditions=RecordingConditions(
            microphone=args.microphone,
            sample_rate_hz=args.sample_rate_hz,
            bit_depth=args.bit_depth,
            channels=args.channels,
            virtual_audio_bypassed=args.virtual_audio_bypassed == "yes",
            room_notes=args.room_notes,
        ),
        annotation_pass=AnnotationPass(args.annotation_pass),
        overwrite=args.force,
    )
    print(f"wrote {args.output}")
    print(
        "The disfluency and role tiers are controlled vocabularies generated from the "
        "published taxonomy: a class outside it cannot be typed."
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    reports = [validate(read(path)) for path in args.files]

    if args.as_json:
        print(
            json.dumps(
                [
                    {
                        "recording_id": report.recording_id,
                        "annotator_id": report.annotator_id,
                        "usable": report.is_usable,
                        "findings": [
                            {
                                "severity": finding.severity.value,
                                "code": finding.code,
                                "message": finding.message,
                                "at_ms": finding.at_ms,
                            }
                            for finding in report.findings
                        ],
                    }
                    for report in reports
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for report in reports:
            print(f"\n{report.recording_id}  ({report.annotator_id})")
            if not report.findings:
                print("  nothing to report")
            for finding in report.findings:
                print(f"  {finding}")

    # Errors fail the command; warnings do not. A pilot that refused every file
    # with a warning would drop exactly the hard cases it exists to find.
    return 1 if any(report.errors for report in reports) else 0


def _agreement(args: argparse.Namespace) -> int:
    report = compare(
        read(args.left),
        read(args.right),
        criterion=MatchCriterion(args.criterion),
        iou_threshold=args.iou,
        tolerance_ms=args.tolerance_ms,
        boundary_review_ms=args.boundary_review_ms,
    )

    if args.as_json:
        print(json.dumps(_as_dict(report), indent=2, ensure_ascii=False))
    else:
        _print_report(report)
    return 0


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def _read_all(paths: Sequence[Path]) -> tuple[list[AnnotatedRecording], dict[str, Path]]:
    """Every file, with the path each recording came from.

    The paths travel alongside the records because the freeze digests the bytes
    on disk, not the parsed record: a re-export that changes formatting without
    changing meaning should still be visible, and only the file can show that.
    """
    records: list[AnnotatedRecording] = []
    sources: dict[str, Path] = {}
    for path in paths:
        record = read(path)
        records.append(record)
        sources[record.recording_id] = path
    return records, sources


def _inventory(args: argparse.Namespace) -> int:
    records, _ = _read_all(args.files)
    corpus = inventory(records)
    verdicts = corpus.adequacy(
        minimum_instances=args.min_instances, minimum_speakers=args.min_speakers
    )

    if args.as_json:
        print(
            json.dumps(
                {
                    "recordings": corpus.recording_count,
                    "speakers": corpus.speaker_count,
                    "total_duration_ms": corpus.total_duration_ms,
                    "events": corpus.event_count,
                    "minimum_instances": args.min_instances,
                    "minimum_speakers": args.min_speakers,
                    "varieties": dict(corpus.variety_speaker_counts),
                    "classes": [
                        {
                            "event_type": verdict.event_type,
                            "instances": verdict.instances,
                            "speakers": verdict.speakers,
                            "adequate": verdict.is_adequate,
                            "shortfall": verdict.shortfall,
                        }
                        for verdict in verdicts
                    ],
                    "adequate": all(v.is_adequate for v in verdicts),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(
            f"\n{corpus.recording_count} recording(s), {corpus.speaker_count} speaker(s), "
            f"{corpus.event_count} annotated event(s)"
        )
        print(
            f"  a class needs {args.min_instances} instance(s) from "
            f"{args.min_speakers} speaker(s) to carry a per-class figure\n"
        )
        # Printed unconditionally, including when there is one variety. "40
        # speakers, all es-PE" is the finding §14.2's error analysis by dialect
        # needs while there is still time to recruit; printing it only when the
        # corpus is already mixed would report the answer and suppress the
        # problem.
        print("  by variety (speakers):")
        for variety, count in corpus.variety_speaker_counts.items():
            print(f"    {variety:<16} {count:>5}")
        print()
        for verdict in verdicts:
            mark = "ok" if verdict.is_adequate else "SHORT"
            print(
                f"  {mark:<5} {verdict.event_type:<16} "
                f"{verdict.instances:>5} instance(s)  {verdict.speakers:>3} speaker(s)"
                + (f"   needs {verdict.shortfall}" if verdict.shortfall else "")
            )

    # Non-zero while the corpus is short. This is the command a recording
    # schedule is driven by, so "keep going" has to be machine-readable.
    return 0 if all(v.is_adequate for v in verdicts) else 1


def _split(args: argparse.Namespace) -> int:
    records, sources = _read_all(args.files)
    plan = split(inventory(records), seed=args.seed)

    if args.as_json:
        print(
            json.dumps(
                {
                    "seed": plan.seed,
                    "shares": dict(plan.shares),
                    "partitions": [
                        {
                            "partition": contents.partition.value,
                            "speakers": list(contents.speakers),
                            "recordings": list(contents.recording_ids),
                            "duration_ms": contents.duration_ms,
                            "class_counts": dict(contents.class_counts),
                        }
                        for contents in plan.partitions
                    ],
                    "usable": plan.is_usable,
                    "findings": [
                        {
                            "severity": finding.severity.value,
                            "code": finding.code,
                            "message": finding.message,
                        }
                        for finding in plan.findings
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"\nspeaker-independent split, seed {plan.seed}")
        for contents in plan.partitions:
            print(
                f"\n  {contents.partition.value:<9} "
                f"{contents.speaker_count:>3} speaker(s)  "
                f"{len(contents.recording_ids):>3} recording(s)"
            )
            print(f"    {', '.join(contents.speakers) or '(none)'}")
            for event_type, count in contents.class_counts.items():
                print(f"      {event_type:<16} {count:>5}")
        for finding in plan.findings:
            print(f"\n  {finding}")

    if not plan.is_usable:
        print(
            "\nrefusing to freeze an unusable split: a partition missing a P0 class has "
            "an undefined per-class figure there, and an undefined figure reads as a "
            "low score while being a missing measurement.",
            file=sys.stderr,
        )
        return 1

    if args.freeze:
        if args.freeze.exists() and not args.force:
            print(
                f"refused: {args.freeze} already exists. Re-freezing a corpus after "
                "seeing a result is the failure the manifest exists to make visible; "
                "pass --force if the first freeze was a mistake.",
                file=sys.stderr,
            )
            return 1
        manifest = freeze(plan, records, sources)
        args.freeze.write_text(to_json(manifest), encoding="utf-8")
        print(f"\nfrozen: {args.freeze}")
        print(f"  digest {manifest.digest}")
        print(f"  {len(manifest.recordings)} recording(s) over {len(manifest.speakers)} speaker(s)")
    return 0


def _verify(args: argparse.Namespace) -> int:
    manifest = from_json(args.manifest.read_text(encoding="utf-8"))
    _, sources = _read_all(args.files)
    findings = verify(manifest, sources)

    if args.as_json:
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "digest": manifest.digest,
                    "verified": not findings,
                    "findings": [
                        {
                            "severity": finding.severity.value,
                            "code": finding.code,
                            "message": finding.message,
                        }
                        for finding in findings
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif not findings:
        print(f"\nverified: {len(manifest.recordings)} recording(s) unchanged since the freeze")
        print(f"  digest {manifest.digest}")
    else:
        print(f"\n{len(findings)} problem(s) with the frozen corpus:")
        for finding in findings:
            print(f"  {finding}")

    return 0 if not findings else 1


def _as_dict(report: AgreementReport) -> dict[str, Any]:
    return {
        "recording_id": report.recording_id,
        "annotators": [report.left_annotator, report.right_annotator],
        "matching": {
            "criterion": report.criterion.value,
            "threshold": report.threshold,
            "matched": report.matched,
            "left_only": report.left_only,
            "right_only": report.right_only,
            "positive_specific_agreement": report.positive_specific_agreement,
        },
        "boundary_error_ms": {
            "matched_pairs": report.boundary.count,
            "start_median": report.boundary.median_start_ms(),
            "start_p95": report.boundary.p95_start_ms(),
            "end_median": report.boundary.median_end_ms(),
            "end_p95": report.boundary.p95_end_ms(),
        },
        "class_agreement": {
            "cohens_kappa": report.class_kappa,
            "krippendorffs_alpha": report.class_alpha,
            "confusion": {f"{a}|{b}": c for (a, b), c in report.class_confusion.counts.items()},
            "per_class": [
                {
                    "event_type": item.event_type,
                    "both": item.both,
                    "left_only": item.left_only,
                    "right_only": item.right_only,
                    "specific_agreement": item.specific_agreement,
                }
                for item in report.per_class
            ],
        },
        "role_agreement": {
            "cohens_kappa": report.role_kappa,
            "per_role": [
                {
                    "role": item.event_type,
                    "both": item.both,
                    "specific_agreement": item.specific_agreement,
                }
                for item in report.role_agreement
            ],
        },
        "abstention": {
            "left_uncertain": report.left_uncertain,
            "right_uncertain": report.right_uncertain,
        },
        # Event by event, in recording order. The coefficients above say how
        # much they disagreed; this is the only part an adjudication session
        # can actually work through, and the only part that can be pasted into
        # DISAGREEMENT_LOG.md without being reconstructed by hand.
        "disagreements": {
            "boundary_review_ms": report.boundary_review_ms,
            "count": len(report.disagreements),
            "items": [
                {
                    "kind": item.kind.value,
                    "at_ms": item.at_ms,
                    "timestamp": item.timestamp,
                    "detail": item.detail,
                    "left": _reading_dict(item.left),
                    "right": _reading_dict(item.right),
                }
                for item in report.disagreements
            ],
        },
        "notes": list(report.notes),
    }


def _reading_dict(reading: Reading | None) -> dict[str, Any] | None:
    """``null`` where an annotator recorded nothing, not an empty reading.

    The difference matters downstream: an empty object reads as "they marked
    something blank here", and the whole point of a missed event is that they
    marked nothing.

    Derived from the dataclass rather than listing the fields. A hand-written
    field list is a second definition of what a reading is, and the two drift
    the first time somebody adds a field to `Reading` and does not think to
    look here - which is exactly the moment the adjudication log quietly stops
    carrying it.
    """
    return None if reading is None else asdict(reading)


def _print_report(report: AgreementReport) -> None:
    print(f"\n{report.recording_id}")
    print(f"  {report.left_annotator} vs {report.right_annotator}")
    print(f"  matching: {report.criterion.value} at {report.threshold}")

    print("\n  1. Did they find the same events?")
    print(f"     matched            {report.matched}")
    print(f"     only {report.left_annotator:<14} {report.left_only}")
    print(f"     only {report.right_annotator:<14} {report.right_only}")
    print(f"     agreement          {report.positive_specific_agreement:.3f}")

    print("\n  2. Did they draw the same boundaries?")
    if report.boundary.count:
        print(
            f"     start  median {_ms(report.boundary.median_start_ms())}"
            f"  p95 {_ms(report.boundary.p95_start_ms())}"
        )
        print(
            f"     end    median {_ms(report.boundary.median_end_ms())}"
            f"  p95 {_ms(report.boundary.p95_end_ms())}"
        )
        print("     (this is the human ceiling for NFR-004's 250 ms target)")
    else:
        print("     no matched pairs")

    print("\n  3. Did they give them the same label?")
    print(f"     Cohen's kappa      {_coefficient(report.class_kappa)}")
    print(f"     Krippendorff alpha {_coefficient(report.class_alpha)}")
    for item in report.per_class:
        print(
            f"     {item.event_type:<22} both={item.both:<4} "
            f"only-left={item.left_only:<4} only-right={item.right_only:<4} "
            f"agreement={_coefficient(item.specific_agreement)}"
        )

    print("\n  Contextual roles")
    print(f"     Cohen's kappa      {_coefficient(report.role_kappa)}")
    for item in report.role_agreement:
        print(
            f"     {item.event_type:<22} both={item.both:<4} "
            f"agreement={_coefficient(item.specific_agreement)}"
        )
    print(
        f"     abstained: {report.left_annotator}={report.left_uncertain} "
        f"{report.right_annotator}={report.right_uncertain}"
    )

    _print_disagreements(report)

    for note in report.notes:
        print(f"\n  note: {note}")


def _print_disagreements(report: AgreementReport) -> None:
    """The adjudication worklist, in the order the recording plays.

    Formatted to be walked through with the audio open: timestamp first, then
    what each annotator saw. An adjudication session driven from a confusion
    matrix has to reconstruct these positions by hand, and the ones that get
    lost in the reconstruction are the ones nobody writes down.
    """
    print(f"\n  Disagreements to adjudicate ({len(report.disagreements)})")
    if not report.disagreements:
        print("     none")
        return

    print(
        f"     boundary differences below {report.boundary_review_ms} ms are not listed (NFR-004)"
    )
    for item in report.disagreements:
        print(f"\n     {item.timestamp}  {item.kind.value}")
        print(f"       {_reading_line(report.left_annotator, item.left)}")
        print(f"       {_reading_line(report.right_annotator, item.right)}")
        print(f"       -> {item.detail}")


def _reading_line(annotator: str, reading: Reading | None) -> str:
    if reading is None:
        return f"{annotator:<12} (nothing here)"
    line = f"{reading.annotator:<12} {reading}"
    return f"{line}  # {reading.note}" if reading.note else line


def _coefficient(value: float | None) -> str:
    """`undefined` rather than a number, when it is undefined.

    Printing 0.0 for an undefined coefficient is how "the sample cannot answer
    this" becomes "the annotators agreed no better than chance" in somebody's
    summary table.
    """
    return "undefined" if value is None else f"{value:.3f}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:6.0f} ms"


if __name__ == "__main__":
    sys.exit(main())
