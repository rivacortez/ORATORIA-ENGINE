"""Corpus construction: inventory, speaker-independent split, and the freeze.

The tests that matter here are about the two things `BASELINES.md` §4 promises
and that nothing checked until now: that no speaker crosses a partition
boundary, and that "frozen" means a digest rather than an intention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.partition.freeze import (
    MANIFEST_VERSION,
    FreezeError,
    freeze,
    from_json,
    to_json,
    verify,
)
from corpus.partition.inventory import InventoryError, inventory
from corpus.partition.split import (
    DEFAULT_SHARES,
    InvalidSplit,
    Partition,
    split,
)
from corpus.schema.records import (
    SCHEMA_VERSION,
    AnnotatedRecording,
    ConsentRecord,
    RecordingConditions,
    Speaker,
)
from corpus.schema.validation import Severity
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import SpeechEventType, p0_speech_events
from tests.corpus.conftest import CONDITIONS, CONSENT, annotation, recording

FILLED = SpeechEventType.FILLED_PAUSE
PROLONG = SpeechEventType.PROLONGATION
CUT_OFF = SpeechEventType.CUT_OFF
CLASSES = tuple(sorted(p0_speech_events(), key=lambda e: e.value))


def _recording(
    speaker: str,
    recording_id: str,
    events: list[tuple[SpeechEventType, int]],
    *,
    variety: str | None = "es-PE",
    consent: ConsentRecord | None = CONSENT,
    conditions: RecordingConditions | None = CONDITIONS,
) -> AnnotatedRecording:
    """One speaker's recording, with a given number of each class.

    Built directly rather than through the `recording` helper so the speaker
    pseudonym can vary - the helper fixes it, because agreement is always about
    one speaker and partitioning never is.

    Consent and conditions are filled in here and not in `recording`, because
    this fixture stands for something a *corpus* contains: every recording that
    reaches an inventory or a freeze came from a file, and the ELAN reader will
    not produce one without them. A test that wants the absence passes `None`.
    """
    annotations = []
    cursor = 1_000
    for event_type, count in events:
        for _ in range(count):
            annotations.append(annotation(event_type, cursor, cursor + 300, "ana"))
            cursor += 500

    base = recording(
        "ana", annotations, recording_id=recording_id, consent=consent, conditions=conditions
    )
    from dataclasses import replace

    return replace(base, speaker=Speaker(pseudonym=speaker, variety=variety))


def _corpus(speakers: int, *, per_speaker: int = 4) -> list[AnnotatedRecording]:
    """A corpus where every speaker produces every class.

    The easy case on purpose: tests about coverage build the hard cases
    explicitly, and a shared fixture that is already unbalanced makes every
    other test read as though balance were the subject.
    """
    return [
        _recording(f"P-{i:03d}", f"rec-{i:03d}", [(event, per_speaker) for event in CLASSES])
        for i in range(1, speakers + 1)
    ]


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def test_the_inventory_counts_classes_and_speakers() -> None:
    result = inventory(_corpus(6, per_speaker=3))

    assert result.recording_count == 6
    assert result.speaker_count == 6
    assert result.class_counts[FILLED.value] == 18
    assert result.class_speaker_counts[FILLED.value] == 6


def test_a_class_from_one_speaker_is_inadequate_however_many_instances() -> None:
    """The whole point of the module.

    Forty `cut_off` from one person is that person's habit, not evidence about
    `cut_off`, and a per-class figure computed from it generalises to nobody.
    """
    corpus = [
        _recording("P-001", "rec-001", [(CUT_OFF, 40)]),
        *[_recording(f"P-{i:03d}", f"rec-{i:03d}", [(FILLED, 40)]) for i in range(2, 8)],
    ]

    verdicts = {v.event_type: v for v in inventory(corpus).adequacy()}

    cut_off = verdicts[CUT_OFF.value]
    assert cut_off.has_enough_instances
    assert not cut_off.has_enough_speakers
    assert not cut_off.is_adequate
    assert "4 more speakers" in cut_off.shortfall


def test_a_class_nobody_produced_is_still_reported() -> None:
    """Iterating what was found would omit exactly the classes the answer is
    about."""
    corpus = _corpus(6)
    verdicts = {v.event_type: v for v in inventory(corpus).adequacy()}

    assert set(verdicts) == {e.value for e in p0_speech_events()}


def test_the_shortfall_names_both_halves_when_both_fail() -> None:
    """'Needs 12 more instances' sends somebody to record more of the same
    speaker; 'and 3 more speakers' is the half that changes what they record."""
    corpus = [_recording("P-001", "rec-001", [(FILLED, 2)])]

    verdicts = {v.event_type: v for v in inventory(corpus).adequacy()}

    assert "more instances" in verdicts[FILLED.value].shortfall
    assert "more speakers" in verdicts[FILLED.value].shortfall


def test_two_annotations_of_one_recording_are_refused() -> None:
    """Counting both doubles every figure in the inventory."""
    corpus = [
        _recording("P-001", "rec-001", [(FILLED, 3)]),
        _recording("P-001", "rec-001", [(FILLED, 3)]),
    ]

    with pytest.raises(InventoryError, match="appears twice"):
        inventory(corpus)


def test_one_pseudonym_with_two_varieties_is_refused() -> None:
    """Either the pseudonym is reused for two people - which breaks the
    speaker-independence guarantee the partitions rest on - or the metadata is
    wrong."""
    corpus = [
        _recording("P-001", "rec-001", [(FILLED, 3)], variety="es-PE"),
        _recording("P-001", "rec-002", [(FILLED, 3)], variety="es-MX"),
    ]

    with pytest.raises(InventoryError, match="more than one variety"):
        inventory(corpus)


# ---------------------------------------------------------------------------
# Speaker independence - the guarantee the whole module exists for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("speakers", [3, 5, 8, 13, 21, 40])
def test_no_speaker_appears_in_two_partitions(speakers: int) -> None:
    """`BASELINES.md` §4: the check is mechanical rather than a matter of care."""
    plan = split(inventory(_corpus(speakers)))

    placements: dict[str, list[str]] = {}
    for contents in plan.partitions:
        for pseudonym in contents.speakers:
            placements.setdefault(pseudonym, []).append(contents.partition.value)

    crossing = {p: where for p, where in placements.items() if len(where) > 1}
    assert not crossing, f"speakers in two partitions: {crossing}"


@pytest.mark.parametrize("speakers", [3, 5, 8, 13, 21, 40])
def test_every_speaker_lands_somewhere(speakers: int) -> None:
    """A speaker in no partition is a corpus smaller than the dataset card says."""
    corpus = inventory(_corpus(speakers))
    plan = split(corpus)

    placed = {p for contents in plan.partitions for p in contents.speakers}
    assert placed == {s.pseudonym for s in corpus.speakers}


@pytest.mark.parametrize("speakers", [5, 13, 40])
def test_a_recording_follows_its_speaker(speakers: int) -> None:
    """Assigning recordings and hoping no speaker straddles a boundary is the
    failure mode itself."""
    corpus = inventory(_corpus(speakers))
    plan = split(corpus)

    by_speaker = {s.pseudonym: s.recording_ids for s in corpus.speakers}
    for contents in plan.partitions:
        expected = sorted(rid for p in contents.speakers for rid in by_speaker[p])
        assert sorted(contents.recording_ids) == expected


def test_the_plan_reports_a_speaker_in_two_partitions() -> None:
    """The construction makes this impossible today. The check is what survives
    a refactor that changes the construction - and the failure it would
    introduce is invisible downstream, because a model scored on speakers it
    memorised does not look broken, it looks good.
    """
    from corpus.partition.split import _independence

    corpus = inventory(_corpus(6))
    good = split(corpus)
    tampered = list(good.partitions)
    # Put the first train speaker into held-out as well.
    from dataclasses import replace

    train = tampered[0]
    held_out = tampered[2]
    tampered[2] = replace(held_out, speakers=(*held_out.speakers, train.speakers[0]))

    findings = _independence(tampered, corpus)

    assert [f.code for f in findings] == ["speaker_in_two_partitions"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 99])
def test_the_same_corpus_and_seed_produce_the_same_plan(seed: int) -> None:
    corpus = inventory(_corpus(13))

    first = split(corpus, seed=seed)
    second = split(corpus, seed=seed)

    assert [c.speakers for c in first.partitions] == [c.speakers for c in second.partitions]


def test_the_plan_does_not_depend_on_the_order_the_recordings_arrived() -> None:
    """A corpus is a directory listing, and directory listings are not ordered."""
    recordings = _corpus(13)
    forward = split(inventory(recordings))
    backward = split(inventory(list(reversed(recordings))))

    assert [c.speakers for c in forward.partitions] == [c.speakers for c in backward.partitions]


def test_a_different_seed_can_produce_a_different_plan() -> None:
    """Re-splitting is how somebody checks that a result is not an artifact of
    one particular assignment."""
    corpus = inventory(_corpus(13))

    plans = {tuple(c.speakers for c in split(corpus, seed=seed).partitions) for seed in range(6)}

    assert len(plans) > 1


# ---------------------------------------------------------------------------
# Stratification: the reason the assignment is not random
# ---------------------------------------------------------------------------


def test_a_rare_class_reaches_every_partition_when_it_can() -> None:
    """The failure this exists to prevent.

    Nine speakers produce `cut_off`; the rest produce only `filled_pause`. A
    random assignment routinely puts all nine in train, leaving held-out with
    an *undefined* per-class F1 for `cut_off` - which is a missing measurement,
    not a low score.
    """
    corpus = inventory(
        [
            *[
                _recording(f"C-{i:03d}", f"rec-c{i:03d}", [(CUT_OFF, 4), (FILLED, 4)])
                for i in range(1, 10)
            ],
            *[_recording(f"F-{i:03d}", f"rec-f{i:03d}", [(FILLED, 8)]) for i in range(1, 22)],
        ]
    )

    plan = split(corpus)

    for contents in plan.partitions:
        assert contents.covers(CUT_OFF.value), (
            f"{contents.partition.value} has no cut_off; counts={dict(contents.class_counts)}"
        )


def test_a_class_that_cannot_reach_every_partition_is_an_error_not_a_silence() -> None:
    """Two speakers produce `prolongation` and there are three partitions.

    Nothing can fix that arrangement, so the plan says so rather than quietly
    producing a held-out set whose prolongation figure will be undefined.
    """
    corpus = inventory(
        [
            *[
                _recording(f"P-{i:03d}", f"rec-{i:03d}", [(PROLONG, 5), (FILLED, 5)])
                for i in range(1, 3)
            ],
            *[_recording(f"F-{i:03d}", f"rec-f{i:03d}", [(FILLED, 8)]) for i in range(1, 20)],
        ]
    )

    plan = split(corpus)

    assert not plan.is_usable
    codes = {f.code for f in plan.errors}
    assert "class_missing_from_partition" in codes
    assert any("undefined" in f.message for f in plan.errors)


def test_a_class_absent_from_the_whole_corpus_is_not_a_split_error() -> None:
    """It is an inventory problem, and reporting it here would send somebody to
    re-split when they need to go and record."""
    corpus = inventory(
        [_recording(f"P-{i:03d}", f"rec-{i:03d}", [(FILLED, 8)]) for i in range(1, 20)]
    )

    plan = split(corpus)

    missing = [f for f in plan.errors if f.code == "class_missing_from_partition"]
    assert not missing


# ---------------------------------------------------------------------------
# Shares
# ---------------------------------------------------------------------------


def test_every_partition_gets_at_least_one_speaker() -> None:
    """A held-out set with no speakers in it is not a small held-out set."""
    plan = split(inventory(_corpus(3)))

    for contents in plan.partitions:
        assert contents.speaker_count >= 1


def test_too_few_speakers_is_a_recruitment_answer_not_a_parameter() -> None:
    corpus = inventory(_corpus(2))

    with pytest.raises(InvalidSplit, match="recruitment"):
        split(corpus)


@pytest.mark.parametrize(
    "shares",
    [
        {Partition.TRAIN: 0.7, Partition.DEV: 0.1, Partition.HELD_OUT: 0.3},
        {Partition.TRAIN: 0.5, Partition.DEV: 0.1, Partition.HELD_OUT: 0.1},
    ],
    ids=["over", "under"],
)
def test_shares_that_do_not_sum_to_one_are_refused(shares: dict) -> None:  # type: ignore[type-arg]
    """The difference is the fraction of the corpus in no partition at all."""
    with pytest.raises(InvalidSplit, match="sum to"):
        split(inventory(_corpus(9)), shares=shares)


def test_a_share_of_zero_is_refused() -> None:
    with pytest.raises(InvalidSplit, match=r"\(0, 1\)"):
        split(
            inventory(_corpus(9)),
            shares={Partition.TRAIN: 0.9, Partition.DEV: 0.1, Partition.HELD_OUT: 0.0},
        )


def test_a_missing_share_is_refused() -> None:
    with pytest.raises(InvalidSplit, match="no share given"):
        split(
            inventory(_corpus(9)),
            shares={Partition.TRAIN: 0.8, Partition.DEV: 0.2},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("speakers", [8, 13, 21, 40])
def test_the_speaker_counts_add_up(speakers: int) -> None:
    """Largest-remainder rather than independent rounding: rounding each share
    on its own loses or invents a speaker."""
    plan = split(inventory(_corpus(speakers)))

    assert sum(c.speaker_count for c in plan.partitions) == speakers


def test_held_out_is_larger_than_dev_by_default() -> None:
    """Held-out carries every reported number; dev only has to be big enough to
    choose a threshold on."""
    plan = split(inventory(_corpus(40)))

    assert plan.of(Partition.HELD_OUT).speaker_count > plan.of(Partition.DEV).speaker_count
    assert DEFAULT_SHARES[Partition.HELD_OUT] > DEFAULT_SHARES[Partition.DEV]


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def _sources(tmp_path: Path, recordings: list[AnnotatedRecording]) -> dict[str, Path]:
    """One file per recording, with distinct bytes."""
    sources = {}
    for record in recordings:
        path = tmp_path / f"{record.recording_id}.eaf"
        path.write_text(f"<annotation id='{record.recording_id}'/>", encoding="utf-8")
        sources[record.recording_id] = path
    return sources


def test_a_frozen_corpus_names_every_recording_with_a_digest(tmp_path: Path) -> None:
    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    plan = split(inventory(recordings))

    manifest = freeze(plan, recordings, sources)

    assert len(manifest.recordings) == 9
    assert all(len(entry.sha256) == 64 for entry in manifest.recordings)
    assert manifest.digest


def test_an_unusable_split_cannot_be_frozen(tmp_path: Path) -> None:
    """Freezing a split with a class missing from held-out would give the
    missing measurement the appearance of a decision."""
    recordings = [
        *[
            _recording(f"P-{i:03d}", f"rec-{i:03d}", [(PROLONG, 5), (FILLED, 5)])
            for i in range(1, 3)
        ],
        *[_recording(f"F-{i:03d}", f"rec-f{i:03d}", [(FILLED, 8)]) for i in range(1, 20)],
    ]
    plan = split(inventory(recordings))

    with pytest.raises(FreezeError, match="must not be frozen"):
        freeze(plan, recordings, _sources(tmp_path, recordings))


def test_a_recording_with_no_source_file_is_refused(tmp_path: Path) -> None:
    """A manifest entry with no digest cannot answer whether the file changed,
    which is the only question a freeze exists to answer."""
    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    sources.pop(recordings[0].recording_id)
    plan = split(inventory(recordings))

    with pytest.raises(FreezeError, match="no source file"):
        freeze(plan, recordings, sources)


def test_a_corpus_spanning_two_taxonomy_versions_cannot_be_frozen(
    tmp_path: Path,
) -> None:
    """A corpus assembled across a taxonomy change is a corpus whose classes
    mean two things."""
    from dataclasses import replace

    recordings = _corpus(9)
    recordings[0] = replace(recordings[0], taxonomy_version=SemanticVersion(2, 0, 0))
    plan = split(inventory(recordings))

    with pytest.raises(FreezeError, match="taxonomy versions"):
        freeze(plan, recordings, _sources(tmp_path, recordings))


def test_a_frozen_corpus_verifies_against_untouched_files(tmp_path: Path) -> None:
    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    manifest = freeze(split(inventory(recordings)), recordings, sources)

    assert verify(manifest, sources) == ()


def test_a_changed_file_is_caught(tmp_path: Path) -> None:
    """The failure the freeze exists to prevent: somebody regenerating the
    held-out set after seeing a disappointing number. It looks like ordinary
    work while it happens and is undetectable afterwards - unless the bytes
    were recorded."""
    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    manifest = freeze(split(inventory(recordings)), recordings, sources)

    changed = sources[recordings[0].recording_id]
    changed.write_text("<annotation id='tampered'/>", encoding="utf-8")

    findings = verify(manifest, sources)

    assert [f.code for f in findings] == ["frozen_recording_changed"]
    assert findings[0].severity.value == "error"


def test_a_missing_file_is_caught(tmp_path: Path) -> None:
    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    manifest = freeze(split(inventory(recordings)), recordings, sources)

    sources[recordings[0].recording_id].unlink()

    assert [f.code for f in verify(manifest, sources)] == ["frozen_recording_missing"]


def test_a_recording_added_after_the_freeze_is_caught(tmp_path: Path) -> None:
    """Including it in an evaluation would score a model on data the freeze
    does not cover."""
    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    manifest = freeze(split(inventory(recordings)), recordings, sources)

    latecomer = tmp_path / "rec-999.eaf"
    latecomer.write_text("<annotation id='rec-999'/>", encoding="utf-8")
    sources["rec-999"] = latecomer

    assert [f.code for f in verify(manifest, sources)] == ["recording_not_in_manifest"]


def test_an_edited_manifest_is_caught(tmp_path: Path) -> None:
    """The digest covers the content, so moving a recording between partitions
    in the file does not go unnoticed."""
    from dataclasses import replace

    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    manifest = freeze(split(inventory(recordings)), recordings, sources)

    moved = replace(manifest.recordings[0], partition="train")
    tampered = replace(manifest, recordings=(moved, *manifest.recordings[1:]))

    findings = verify(tampered, sources)

    assert "manifest_digest_mismatch" in {f.code for f in findings}


# ---------------------------------------------------------------------------
# The manifest as a committable file
# ---------------------------------------------------------------------------


def test_a_manifest_survives_a_round_trip(tmp_path: Path) -> None:
    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    manifest = freeze(split(inventory(recordings)), recordings, sources)

    restored = from_json(to_json(manifest))

    assert restored == manifest
    assert verify(restored, sources) == ()


def test_the_digest_does_not_depend_on_formatting(tmp_path: Path) -> None:
    """A manifest re-serialised with different indentation is the same manifest."""
    recordings = _corpus(9)
    manifest = freeze(split(inventory(recordings)), recordings, _sources(tmp_path, recordings))

    reflowed = json.dumps(json.loads(to_json(manifest)), indent=8)

    assert from_json(reflowed).digest == manifest.digest


def test_a_manifest_from_an_incompatible_version_is_refused() -> None:
    """Reading it anyway would compare two manifest shapes and report the
    difference as a changed corpus."""
    payload = json.dumps({"manifest_version": "9.0.0"})

    with pytest.raises(FreezeError, match="written under version 9"):
        from_json(payload)


def test_a_manifest_that_is_not_json_is_refused() -> None:
    with pytest.raises(FreezeError, match="not valid JSON"):
        from_json("{not json")


def test_a_manifest_missing_a_field_is_refused() -> None:
    """Read against `MANIFEST_VERSION` rather than a literal.

    Hard-coding "1.0.0" here is what made this test stop testing what it says
    the day the manifest version moved: the version check fired first and the
    missing-field path was never reached, while the test still passed for the
    wrong reason.
    """
    with pytest.raises(FreezeError, match="missing or malformed"):
        from_json(json.dumps({"manifest_version": str(MANIFEST_VERSION)}))


# ---------------------------------------------------------------------------
# The error paths, which are where the next defect will be
# ---------------------------------------------------------------------------


def test_a_corpus_with_no_speakers_cannot_be_split() -> None:
    with pytest.raises(InvalidSplit, match="no speakers"):
        split(inventory([]))


def test_speaker_of_resolves_every_speaker_to_exactly_one_partition() -> None:
    """The independence guarantee, asked the other way round: given a speaker,
    where did they go? Phase 3 asks exactly this before scoring anything."""
    corpus = inventory(_corpus(13))
    plan = split(corpus)

    for speaker in corpus.speakers:
        assert plan.speaker_of(speaker.pseudonym) is not None
    assert plan.speaker_of("nobody") is None


def test_asking_for_a_partition_that_is_not_one_raises() -> None:
    plan = split(inventory(_corpus(9)))

    with pytest.raises(KeyError):
        plan.of("train")  # type: ignore[arg-type]


def test_a_partition_off_its_target_is_a_warning_not_an_error() -> None:
    """Stratification trades balance for coverage on purpose. A held-out set one
    speaker short of its share is a far smaller problem than one missing a class
    entirely, so it must not block the freeze."""
    corpus = inventory(
        [
            *[
                _recording(f"R-{i:03d}", f"rec-r{i:03d}", [(CUT_OFF, 4), (FILLED, 4)])
                for i in range(1, 8)
            ],
            *[_recording(f"C-{i:03d}", f"rec-c{i:03d}", [(FILLED, 20)]) for i in range(1, 4)],
        ]
    )

    plan = split(corpus)

    assert plan.is_usable
    assert all(f.severity is Severity.WARNING for f in plan.warnings)


def test_the_plan_reports_a_speaker_nobody_placed() -> None:
    """Unreachable by construction, like the two-partitions check, and asserted
    for the same reason: their recordings would be silently absent from every
    partition and the corpus would be smaller than the dataset card says."""
    from dataclasses import replace

    from corpus.partition.split import _independence

    corpus = inventory(_corpus(6))
    partitions = list(split(corpus).partitions)
    dropped = partitions[0].speakers[0]
    partitions[0] = replace(partitions[0], speakers=partitions[0].speakers[1:])

    findings = _independence(partitions, corpus)

    assert [f.code for f in findings] == ["speaker_in_no_partition"]
    assert dropped in findings[0].message


def test_freezing_a_plan_whose_recording_was_not_supplied_is_refused(
    tmp_path: Path,
) -> None:
    """The manifest would name a recording nobody can check."""
    recordings = _corpus(9)
    plan = split(inventory(recordings))

    with pytest.raises(FreezeError, match="no annotation was given"):
        freeze(plan, recordings[1:], _sources(tmp_path, recordings))


def test_a_corpus_spanning_two_schema_versions_cannot_be_frozen(tmp_path: Path) -> None:
    """One manifest cannot describe two record shapes.

    The odd version is derived from `SCHEMA_VERSION` rather than written out.
    It used to be the literal 2.0.0, which became the *current* version when the
    schema gained consent and recording conditions - so the two records agreed,
    nothing was refused, and the test failed loudly. It would have been just as
    easy for it to keep passing while checking nothing.
    """
    from dataclasses import replace

    recordings = _corpus(9)
    recordings[0] = replace(
        recordings[0], schema_version=SemanticVersion(SCHEMA_VERSION.major + 1, 0, 0)
    )
    plan = split(inventory(recordings))

    with pytest.raises(FreezeError, match="schema versions"):
        freeze(plan, recordings, _sources(tmp_path, recordings))


def test_verify_reports_a_recording_whose_file_vanished(tmp_path: Path) -> None:
    """Distinct from one that was never supplied: the manifest points at a path
    and the path is gone."""
    recordings = _corpus(9)
    sources = _sources(tmp_path, recordings)
    manifest = freeze(split(inventory(recordings)), recordings, sources)

    dropped = manifest.recordings[0].recording_id
    partial = {k: v for k, v in sources.items() if k != dropped}

    codes = [f.code for f in verify(manifest, partial)]

    assert codes == ["frozen_recording_missing"]


def test_a_manifest_with_an_unreadable_version_is_refused() -> None:
    with pytest.raises(FreezeError, match="no readable version"):
        from_json(json.dumps({"manifest_version": "one"}))
