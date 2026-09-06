"""The report end to end, on a pilot shaped like the real one.

The scenario below is not a happy path. Two annotators agree on most events,
disagree about a boundary, disagree about a class, disagree about a role, and
each find one event the other missed. That is what a first taxonomic pilot
looks like, and a report that only works on clean data would be useless
exactly when it is needed.
"""

from __future__ import annotations

import pytest

from corpus.agreement.report import (
    IncompatibleVersions,
    MismatchedRecordings,
    NotIndependent,
    compare,
)
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import ContextualRole, SpeechEventType
from tests.corpus.conftest import annotation, recording

PAUSE = SpeechEventType.FILLED_PAUSE
FILLER = SpeechEventType.LEXICAL_FILLER
FALSE_START = SpeechEventType.FALSE_START
SELF_REPAIR = SpeechEventType.SELF_REPAIR


@pytest.fixture
def ana():  # type: ignore[no-untyped-def]
    return recording(
        "ana",
        [
            annotation(PAUSE, 1_000, 1_800, "ana"),
            annotation(FILLER, 3_000, 3_400, "ana", raw_text="este", role=ContextualRole.FILLER),
            # Ana reads this as an abandoned start with no relation to what follows.
            annotation(FALSE_START, 6_000, 6_900, "ana", raw_text="los resulta"),
            # Only Ana heard this one.
            annotation(PAUSE, 12_000, 12_500, "ana"),
        ],
    )


@pytest.fixture
def beto():  # type: ignore[no-untyped-def]
    return recording(
        "beto",
        [
            # Same event, boundaries 120 ms and 60 ms off.
            annotation(PAUSE, 1_120, 1_860, "beto"),
            # Same event, but Beto reads "este" as a demonstrative.
            annotation(FILLER, 3_020, 3_420, "beto", raw_text="este", role=ContextualRole.SEMANTIC),
            # Same event, different class: Beto hears a relation to what follows.
            annotation(SELF_REPAIR, 6_050, 6_950, "beto", raw_text="los resulta"),
            # Only Beto heard this one.
            annotation(PAUSE, 20_000, 20_400, "beto"),
        ],
    )


def test_the_report_separates_finding_from_labelling(ana, beto) -> None:  # type: ignore[no-untyped-def]
    """The distinction the whole design exists for.

    Three events were found by both. One of them was labelled differently. A
    report that collapsed the two questions would say "three of four agreed"
    and hide that the disagreement is about the taxonomy, not about hearing.
    """
    report = compare(ana, beto)

    assert report.matched == 3
    assert report.left_only == 1
    assert report.right_only == 1
    # 2*3 / (2*3 + 1 + 1) = 0.75
    assert report.positive_specific_agreement == pytest.approx(0.75)
    # ... and of those three, one carries a different class.
    assert report.class_confusion.agreements == 2


def test_the_boundary_error_is_the_human_ceiling_for_nfr_004(ana, beto) -> None:  # type: ignore[no-untyped-def]
    """Start errors 120, 20, 50 -> median 50. End errors 60, 20, 50 -> median 50."""
    report = compare(ana, beto)

    assert report.boundary.count == 3
    assert report.boundary.median_start_ms() == pytest.approx(50.0)
    assert report.boundary.median_end_ms() == pytest.approx(50.0)
    # Well inside the 250 ms NFR-004 claims, which is the point of measuring it.
    assert report.boundary.p95_start_ms() == pytest.approx(120.0)


def test_the_false_start_self_repair_confusion_is_visible(ana, beto) -> None:  # type: ignore[no-untyped-def]
    """§17 predicts this pair will be the hard one, so it is reported by name."""
    report = compare(ana, beto)

    watched = {item.event_type: item for item in report.watched()}
    assert watched[FALSE_START.value].both == 0
    assert watched[SELF_REPAIR.value].both == 0
    assert watched[FALSE_START.value].specific_agreement == pytest.approx(0.0)


def test_role_disagreement_is_measured_apart_from_class_disagreement(ana, beto) -> None:  # type: ignore[no-untyped-def]
    """Whether "este" is a filler or a demonstrative is its own question.

    Both annotators found "este" and called it a lexical filler; they read its
    function differently. Folding that into the class figures would hide a
    problem with the role definitions behind a class that looks agreed.

    Two matched pairs carry roles. On one they agree (`filler`/`filler`), on
    the other they do not (`filler`/`semantic`):

        filler:    both=1, left_only=1, right_only=0 -> 2/(2+1) = 0.667
        semantic:  both=0, left_only=0, right_only=1 -> 0/(0+1) = 0.0

    `semantic` scoring zero is the finding: Beto used it once and Ana never
    agreed. On a real pilot that is the signal to look at the definition.
    """
    report = compare(ana, beto)

    roles = {item.event_type: item for item in report.role_agreement}
    assert roles["filler"].specific_agreement == pytest.approx(2 / 3)
    assert roles["semantic"].specific_agreement == pytest.approx(0.0)
    # ... while the class figures show `lexical_filler` as fully agreed, which
    # is exactly why the two are measured apart.
    per_class = {item.event_type: item for item in report.per_class}
    assert per_class["lexical_filler"].specific_agreement == pytest.approx(1.0)


def test_a_small_pilot_is_flagged_rather_than_reported_confidently(ana, beto) -> None:  # type: ignore[no-untyped-def]
    """Three matched events cannot support a stable kappa, and the report says so."""
    report = compare(ana, beto)

    assert any("too few" in note for note in report.notes)


def test_perfect_agreement_reports_perfectly() -> None:
    events = [annotation(PAUSE, 1_000, 1_800, "ana"), annotation(PAUSE, 5_000, 5_400, "ana")]
    mirrored = [annotation(PAUSE, 1_000, 1_800, "beto"), annotation(PAUSE, 5_000, 5_400, "beto")]

    report = compare(recording("ana", events), recording("beto", mirrored))

    assert report.positive_specific_agreement == pytest.approx(1.0)
    assert report.boundary.median_start_ms() == pytest.approx(0.0)
    # Kappa is undefined here, and that is correct: both annotators used one
    # category, so there is no chance agreement to correct for.
    assert report.class_kappa is None


def test_comparing_two_different_recordings_is_refused() -> None:
    left = recording("ana", [annotation(PAUSE, 0, 400, "ana")], recording_id="pilot-001")
    right = recording("beto", [annotation(PAUSE, 0, 400, "beto")], recording_id="pilot-002")

    with pytest.raises(MismatchedRecordings, match="different recordings"):
        compare(left, right)


def test_comparing_a_file_with_itself_is_refused() -> None:
    """Agreement is between two people; a self-comparison always scores 1.

    `NotIndependent` rather than `MismatchedRecordings`: the two files describe
    the same recording perfectly well, and what they fail to describe is two
    people - the same failure as comparing against an adjudicated pass.
    """
    same = recording("ana", [annotation(PAUSE, 0, 400, "ana")])

    with pytest.raises(NotIndependent, match="between two people"):
        compare(same, same)


def test_annotations_under_different_taxonomy_versions_are_refused() -> None:
    """A pilot that changes the manual produces exactly this.

    Refused rather than noted, and refused at *any* difference. The protocol
    says a taxonomy change requires re-running Pilot B, and a note at the
    bottom of a report does not survive being copied into a results table.
    """
    left = recording(
        "ana", [annotation(PAUSE, 0, 400, "ana")], taxonomy_version=SemanticVersion(1, 0, 0)
    )
    right = recording(
        "beto", [annotation(PAUSE, 0, 400, "beto")], taxonomy_version=SemanticVersion(1, 1, 0)
    )

    with pytest.raises(IncompatibleVersions, match="different manuals"):
        compare(left, right)


def test_nothing_matching_at_all_is_called_out(ana) -> None:  # type: ignore[no-untyped-def]
    """Usually a threshold set too strictly, or two files of different sessions."""
    elsewhere = recording("beto", [annotation(PAUSE, 50_000, 50_400, "beto")])

    report = compare(ana, elsewhere)

    assert report.matched == 0
    assert any("nothing matched" in note for note in report.notes)


def test_the_report_records_which_matching_rule_produced_it(ana, beto) -> None:  # type: ignore[no-untyped-def]
    """A coefficient without its threshold cannot be reproduced or compared."""
    report = compare(ana, beto, iou_threshold=0.6)

    assert report.criterion.value == "iou"
    assert report.threshold == pytest.approx(0.6)


def test_abstention_travels_with_the_agreement_figures(ana, beto) -> None:  # type: ignore[no-untyped-def]
    """An engine answering `uncertain` everywhere would score well on precision
    if abstentions were dropped, so the counts are part of the report."""
    report = compare(ana, beto)

    assert report.left_uncertain == 0
    assert report.right_uncertain == 0
