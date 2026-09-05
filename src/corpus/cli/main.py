"""``corpus`` — the annotators' and the methodologist's command line.

Three verbs, matching the three things that actually happen during a pilot:
hand an annotator a file, check what came back, and measure two of them against
each other.

Output is plain text on stdout and machine-readable JSON behind ``--json``.
Both, because the same numbers are read two ways: a methodologist reads them
during the pilot, and the disagreement log needs them in a form that can be
committed and diffed between manual versions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
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
from corpus.schema.records import AnnotationPass
from corpus.schema.validation import validate


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
    except (InvalidMatchParameters, InvalidReportParameters) as error:
        # Exit 2, argparse's code for a usage error, because that is what this
        # is: the files are fine and the numbers asked for are not. Caught here
        # rather than left to propagate - `corpus agreement --iou 0` used to
        # print a Python traceback at an annotator, which reads as "the tool is
        # broken" rather than "that threshold means nothing".
        print(f"error: {error}", file=sys.stderr)
        return 2


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
    template.add_argument("--annotator", required=True)
    template.add_argument("--media", required=True, help="path or URL of the audio or video")
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

    return parser


def _template(args: argparse.Namespace) -> int:
    write_template(
        args.output,
        recording_id=args.recording_id,
        speaker_pseudonym=args.speaker,
        annotator_id=args.annotator,
        media_url=args.media,
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
