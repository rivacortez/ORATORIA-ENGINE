"""What Phase 0.5 measures, enforced rather than asserted in prose.

The protocol declares that this phase validates the speech taxonomy and says
nothing about the visual one. A declaration in a document is worth exactly as
much as the next person's memory of it, so the boundary is checked here: the
day the annotation schema learns to carry a visual class, these fail and
whoever did it has to go and rewrite the scope section.

The alternative - extending the schema to visual events now - was considered
and rejected. A visual event has none of the relationships a speech event has
(no words tier to check against, no contextual role, no lexical expression),
`insufficient_lighting` is a continuous condition rather than an event, and the
matching here is one-to-one and non-crossing, which is the wrong instrument for
unitizing a condition. See `docs/corpus/PILOT_PROTOCOL.md`.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from corpus.partition.freeze import FrozenCorpus, FrozenRecording
from corpus.schema.records import DisfluencyAnnotation, Interval, SchemaViolation
from evidence_engine.domain.shared.taxonomy import SpeechEventType, VisualEventType
from tests.corpus.conftest import annotation, template

PROTOCOL = Path(__file__).resolve().parents[2] / "docs" / "corpus" / "PILOT_PROTOCOL.md"


@pytest.mark.parametrize("event", list(SpeechEventType))
def test_the_annotation_record_accepts_every_speech_class(event: SpeechEventType) -> None:
    """The scope claim has two halves, and this is the one that must hold."""
    assert annotation(event, 1_000, 2_000).event_type is event


def test_the_annotation_record_cannot_carry_a_visual_class() -> None:
    """The other half. `DisfluencyAnnotation.event_type` is typed to the speech
    enum, and the taxonomy allowlist check refuses the value at runtime too -
    so a visual class cannot reach the agreement calculator by accident."""
    for event in VisualEventType:
        with pytest.raises((SchemaViolation, ValueError, AttributeError)):
            DisfluencyAnnotation(
                event_type=SpeechEventType(event.value),  # raises: not a speech class
                interval=Interval(1_000, 2_000),
                annotator_id="ana",
            )


def test_the_template_offers_no_visual_class(tmp_path: Path) -> None:
    """An annotator cannot select one, so the pilot cannot silently acquire
    visual annotations that nobody planned to measure."""
    path = tmp_path / "t.eaf"
    template(path)
    content = path.read_text(encoding="utf-8")

    for event in VisualEventType:
        assert f">{event.value}<" not in content
    for event in SpeechEventType:
        assert f">{event.value}<" in content


def test_the_protocol_declares_the_narrowing() -> None:
    """The code enforces the boundary; this checks somebody wrote down why.

    A guard with no stated reason gets removed by the next person who finds it
    inconvenient.
    """
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "## Scope: the speech taxonomy only" in text
    # Every excluded class named, so a reader does not have to go and diff two
    # enums to find out what was left out.
    for event in VisualEventType:
        assert event.value in text


def test_the_protocol_does_not_claim_the_visual_taxonomy_was_validated() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "enters Phase 5 unvalidated" in text


def test_the_exit_criterion_closes_only_the_speech_half() -> None:
    """The contradiction a reviewer found: the protocol excluded the visual
    classes and then said Pilot B closes Phase 0.

    A speech-only pilot cannot close a multimodal phase, and recording that it
    did would put "validated" in the project's own tracker against something
    nobody looked at. The two halves are now tracked separately so the sentence
    cannot be written without naming one.
    """
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "### Phase 0 — speech taxonomy" in text
    assert "### Phase 0 — visual taxonomy" in text
    assert "must name which half" in text

    # The unqualified claim, in the exact form it took before.
    assert "Phase 0 closes — properly this time —" not in text


def test_the_visual_half_is_recorded_as_not_started() -> None:
    """An open checklist, not a promise. If somebody ticks these without doing
    the work, at least the work is written down to be ticked against."""
    text = PROTOCOL.read_text(encoding="utf-8")

    visual = text.split("### Phase 0 — visual taxonomy", 1)[1]
    assert "Nothing below has been attempted" in visual
    # Every box in the visual section is still open.
    assert "- [x]" not in visual.split("##", 1)[0]


def test_the_protocol_states_the_enforced_taxonomy_rule() -> None:
    """The document and `compare` used to disagree: the protocol said any
    taxonomy change requires re-running the pilot, and the code tolerated a
    minor difference with a note."""
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "A taxonomy change requires re-running Pilot B" in text
    assert "at all" in text.split("A taxonomy change requires", 1)[1][:900]


# ---------------------------------------------------------------------------
# The claim has to be qualified everywhere it is made, not only in one file
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
TAXONOMY_MODULE = ROOT / "src" / "evidence_engine" / "domain" / "shared" / "taxonomy.py"
PINS = ROOT / "docs" / "governance" / "BASELINE_PINS.md"
BASELINES = ROOT / "docs" / "governance" / "BASELINES.md"
MANUAL = ROOT / "docs" / "taxonomy" / "ANNOTATION_MANUAL.md"
DATASET_CARD = ROOT / "docs" / "corpus" / "DATASET_CARD_TEMPLATE.md"


def test_the_readme_names_which_half_phase_0_5_closes() -> None:
    """The protocol was corrected and the README was not, which is how a
    contradiction survives a review: the reader checks the document that was
    just changed."""
    text = README.read_text(encoding="utf-8")

    assert "Phase 0.5 closes the speech half only" in text
    assert "has to say which one" in text


def test_the_taxonomy_module_does_not_claim_a_single_closure() -> None:
    """It defines both enums, so a docstring about "closing the taxonomy" is
    a claim about the visual classes too."""
    text = TAXONOMY_MODULE.read_text(encoding="utf-8")

    assert "published and unvalidated" in text
    assert "before Phase 0\ncan close" not in text


def test_the_pins_do_not_make_the_published_average_the_target() -> None:
    """87.8 is a ten-language average with eight synthetic languages in it,
    and includes English and German, which this thesis does not measure.

    A target taken from it would be a target against a number produced from
    data that is neither Spanish nor human nor spontaneous. The comparator is
    the es-PE held-out measurement, which does not exist yet.
    """
    text = PINS.read_text(encoding="utf-8")

    assert "87.8 is not the NFR-001 threshold" in text
    assert "the NFR-001 comparator" in text
    # The claim in the form it took before.
    assert "NFR-001 has to be read\nagainst 87.8" not in text


def test_the_pins_separate_the_frozen_artifact_from_the_frozen_run() -> None:
    """A pinned digest guarantees the same weights, not the same numbers."""
    text = PINS.read_text(encoding="utf-8")

    assert "weights and runtime-independent decoding configuration frozen" in text
    assert "Everything that decides *how the model is run* is still open" in text
    assert "Four things get frozen" in text


GITATTRIBUTES = ROOT / ".gitattributes"


def test_gitattributes_keeps_the_rules_it_had_before_the_evidence_exception() -> None:
    """A guard against the way this file was broken once already.

    Adding the `docs/evidence/** -whitespace` exception replaced the file
    instead of appending to it, silently dropping LF normalization, the
    byte-for-byte guarantee on the golden fixtures and the binary declarations.
    None of it would have failed a test: it would have surfaced as a contract
    test that passes on one platform and fails on another, months later, in
    something unrelated.

    Checked here rather than trusted to review, because the loss is invisible
    in a diff that shows a file being rewritten.
    """
    text = GITATTRIBUTES.read_text(encoding="utf-8")

    required = [
        "* text=auto eol=lf",  # repository-wide normalization
        "tests/contract/golden/** -text",  # byte-for-byte fixtures
        "*.wav binary",
        "*.mp4 binary",
        "*.webm binary",
        "docs/evidence/** -whitespace",  # the exception that caused the loss
    ]
    missing = [rule for rule in required if rule not in text]
    assert not missing, f"rules dropped from .gitattributes: {missing}"


# ---------------------------------------------------------------------------
# Absence, not only presence
# ---------------------------------------------------------------------------
#
# The tests above assert that the correct phrasing exists. A reviewer pointed
# out that this is half a check: a document can carry the qualified statement
# in one section and the unqualified one in another, and a positive assertion
# passes while the reader is still misled. Three survived that way -
# PILOT_PROTOCOL.md's own title among them.
#
# So the same documents are swept for the phrasings that must not appear
# anywhere in them.

#: Files that make claims about what has been validated. Every claim in every
#: one of them has to name a modality.
CLAIM_BEARING = (
    README,
    PROTOCOL,
    TAXONOMY_MODULE,
    BASELINES,
    PINS,
    # The parent build record. Added here the day it was written rather than
    # the day it first contradicted something: it summarises every other
    # document, so it is the one most likely to carry a claim that was true
    # when it was copied and is not any more.
    ROOT / "docs" / "BUILD_RECORD.md",
    # The dataset card template, for the same reason. It carries no closure
    # claim today because every one of its cells is pending; §5 of it is where
    # the taxonomy version and the agreement figures get written down, which is
    # where an unqualified claim will first be tempting.
    DATASET_CARD,
)

#: Phrasings that assert a closure without naming which half of the taxonomy.
#: Written as substrings rather than a regex over "taxonomy": the word appears
#: constantly and legitimately, and it is the *closure* claim that has to be
#: qualified.
UNQUALIFIED_CLOSURES = (
    "closure of the taxonomy",
    "the taxonomy gets closed",
    "closing the taxonomy",
    "close phase 0",
    "closes phase 0",
    "the taxonomy is validated",
    "the taxonomy has been validated",
)

#: Spans where a forbidden phrase is being *mentioned* rather than asserted:
#: double quotes and backticks. The documents that forbid these phrasings quote
#: them in order to forbid them, and a sweep that cannot tell a claim from its
#: prohibition flags the fix as the defect.
_MENTION = re.compile(r'"[^"\n]*"|`[^`\n]*`')


def _unwrapped(text: str) -> str:
    """The document with its line wrapping collapsed to single spaces."""
    return re.sub(r"\s+", " ", text)


def _assertions_only(text: str) -> str:
    """The document's assertions: mentions removed, then wrapping collapsed.

    The order is the whole design, and it is not symmetric.

    *Mentions first, on the wrapped text.* `_MENTION` refuses to cross a
    newline, so a quotation that opens on one line and closes on the next is
    read as an assertion rather than excused as a quotation. That is the strict
    reading and the one to keep: a claim does not stop being a claim because a
    quotation mark opened three words earlier on the previous line.

    *Wrapping second.* A phrase is a phrase wherever the paragraph happened to
    break. The annotation manual's original claim wrapped as "aplicar esta
    taxonomia de forma / consistente", and a sweep over raw text walked straight
    past it - the defect this section exists to catch, surviving inside the
    check meant to catch it, for as long as the line width was unlucky. Every
    phrase in the lists above is written unwrapped, so this is what they are
    matched against.
    """
    return _unwrapped(_MENTION.sub(" ", text))


@pytest.mark.parametrize("path", CLAIM_BEARING, ids=lambda p: p.name)
def test_no_document_claims_an_unqualified_closure(path: Path) -> None:
    """Phase 0 has two halves and only one of them is being closed.

    Scanned over assertions rather than raw text. Both README.md and
    taxonomy.py legitimately contain `"the taxonomy is validated"` inside the
    sentence that says it is never a true sentence on its own, and a check that
    cannot tell a claim from a quotation of it would demand deleting the
    warning.
    """
    text = _assertions_only(path.read_text(encoding="utf-8").lower())

    found = [phrase for phrase in UNQUALIFIED_CLOSURES if phrase in text]
    assert not found, f"{path.name} claims a closure without naming the half: {found}"


#: Phrasings that overstate what is pinned. The weights are frozen; the thing
#: that runs them is not, and will not be until Phase 3.
#:
#: Deliberately narrow. "Why baselines are frozen before anything is trained" is
#: a heading about the principle, and "Phase 3, after the environment is frozen"
#: is a correct statement about a future stage - neither is an overstatement,
#: and a list broad enough to catch them would be a list nobody can satisfy.
OVERSTATED_FREEZES = (
    "the baseline is frozen",
    "the baselines are frozen",
    "inference library is frozen",
    "the executable baseline is frozen",
    "the executable environment is frozen",
)


@pytest.mark.parametrize("path", (PINS, BASELINES))
def test_the_governance_documents_do_not_overstate_the_freeze(path: Path) -> None:
    """A pinned weights digest guarantees the same weights, not the same
    numbers: ct2 and transformers do not produce bit-identical output, and
    neither does one backend across two CUDA builds."""
    text = _assertions_only(path.read_text(encoding="utf-8").lower())

    found = [phrase for phrase in OVERSTATED_FREEZES if phrase in text]
    assert not found, f"{path.name} overstates what is pinned: {found}"


def test_both_governance_documents_carry_the_same_freeze_cycle() -> None:
    """The blocking finding: the two files disagreed about the cycle.

    `BASELINES.md` had two stages with the outputs at the end of Phase 1;
    `BASELINE_PINS.md` had three and put the executable environment at Phase 3.
    Under the first, the outputs would be generated by an unpinned runtime -
    which is a number nobody can regenerate, and the opposite of what a frozen
    baseline is for.

    Checked as an ordered sequence rather than a set: the order *is* the
    content, because it encodes which stage depends on which.
    """
    stages = (
        "Model artifacts and runtime-independent decoding configuration",
        "The held-out set",
        "The executable environment",
        "The baseline outputs over the held-out set",
    )

    for path in (PINS, BASELINES):
        text = path.read_text(encoding="utf-8")
        positions = [text.find(stage) for stage in stages]
        assert all(p >= 0 for p in positions), (
            f"{path.name} is missing freeze stages: "
            f"{[s for s, p in zip(stages, positions, strict=True) if p < 0]}"
        )
        assert positions == sorted(positions), (
            f"{path.name} lists the freeze stages out of order; the order encodes "
            "which stage depends on which"
        )


def test_the_outputs_are_scheduled_after_the_environment() -> None:
    """Stated in prose too, because the table alone is easy to skim past."""
    for path in (PINS, BASELINES):
        text = path.read_text(encoding="utf-8")
        assert "after** the environment is frozen" in text, path.name


# ---------------------------------------------------------------------------
# The annotation manual - the document the annotators actually read
# ---------------------------------------------------------------------------
#
# It escaped the sweep above for two rounds. It is generated by
# `scripts/render_taxonomy.py`, it publishes all nine visual classes in full,
# and its opening paragraph stated the Phase 0 exit criterion as two annotators
# applying *this taxonomy* consistently - the exact sentence PILOT_PROTOCOL.md
# says has to name a modality. Of every document in the repository it is the
# one whose reader has no other source: an annotator works from the manual and
# from nothing else.
#
# Fixed in the renderer's header rather than in the Markdown. A correction
# typed into a generated file has the worst possible lifetime for a warning -
# long enough to be reviewed and approved, gone at the next regeneration.

#: The Spanish half of the sweep. The manual is the one claim-bearing document
#: not written in English, so `UNQUALIFIED_CLOSURES` would scan it for phrasings
#: it cannot contain and pass for the wrong reason. Both lists are applied to
#: it: the English one because the manual could acquire an English sentence,
#: this one because Spanish is the language its claims are made in.
#:
#: Deliberately narrow, on the same principle as the English list. "cierra la
#: mitad del habla" is a correct statement that has to keep passing, and a list
#: wide enough to catch it would be a list nobody can satisfy.
UNQUALIFIED_CLOSURES_ES = (
    "esta taxonomia de forma consistente",
    "la taxonomia de forma consistente",
    "la taxonomia de manera consistente",
    "la taxonomia esta validada",
    "la taxonomia fue validada",
    "cierra la fase 0",
    "cierre de la fase 0",
    "cierra la taxonomia",
)

#: The mention the manual carries inside the sentence that forbids it, in the
#: exact form the sweep has to see through.
_MANUAL_MENTION = '"la taxonomia esta validada"'


def _unwrapped(text: str) -> str:
    """The document with its line wrapping collapsed.

    The manual is generated, so where its sentences break is a property of the
    renderer's column width rather than of what it says. Asserting against the
    wrapped form would make a reflow look like a retraction.

    Not used by the sweep below, which reads the raw text on purpose: `_MENTION`
    refuses to cross a newline, so a quotation split over two lines counts as an
    assertion. That is the stricter reading and the one to keep - a claim does
    not stop being a claim because a quotation mark opened three words earlier
    on the previous line.
    """
    return re.sub(r"\s+", " ", text)


def test_the_manual_names_which_half_its_exit_criterion_closes() -> None:
    """The defect, in the form it took: the manual opened by stating Phase 0's
    exit criterion over the whole taxonomy, on the page an annotator reads
    before touching either half."""
    text = _unwrapped(MANUAL.read_text(encoding="utf-8"))

    assert "## Alcance: cual mitad se valida y cual no" in text
    assert "Solo una de sus dos mitades se somete a validacion en la Fase 0.5." in text
    assert "clases del habla** sobre una muestra piloto" in text

    # Neither half has actually been measured yet, and a manual that implied the
    # speech half had been would be the same defect facing the other way.
    assert "La mitad del habla tampoco esta medida aun." in text


def test_the_manual_marks_the_visual_classes_as_unvalidated_where_it_publishes_them() -> None:
    """The scope section at the top is not enough on its own.

    This manual is read in sections - an annotator looking up
    `posture_deviation` arrives two hundred lines below the header and sees nine
    fully specified classes with positive examples, which is what a validated
    taxonomy looks like. The warning is repeated where the classes are.
    """
    text = MANUAL.read_text(encoding="utf-8")

    after_heading = text.split("## Clases visuales", 1)[1]
    before_first_class = _unwrapped(after_heading.split("### ", 1)[0]).lower()

    assert "publicadas y sin validar" in before_first_class

    # Still published in full. The fix is a qualification, not a deletion: the
    # engine emits these classes and Phase 5 has to annotate them.
    for event in VisualEventType:
        assert f"### `{event.value}`" in after_heading


def test_the_manual_claims_no_unqualified_closure() -> None:
    """The sweep, in the language the manual is written in."""
    text = _assertions_only(MANUAL.read_text(encoding="utf-8").lower())

    found = [phrase for phrase in UNQUALIFIED_CLOSURES_ES + UNQUALIFIED_CLOSURES if phrase in text]
    assert not found, f"ANNOTATION_MANUAL.md claims a closure without naming the half: {found}"


def test_the_manual_keeps_the_warning_the_sweep_would_be_satisfied_by_deleting() -> None:
    """The sweep above passes just as well if somebody removes the sentence.

    That is the failure mode `BUILD_RECORD.md` §4.3 records: a check that cannot
    tell a claim from its prohibition ends up demanding that the prohibition go.
    So the mention is required to be there, on one line, quoted - which is also
    what makes the sweep's mention-stripping load-bearing rather than decorative.
    """
    lines = MANUAL.read_text(encoding="utf-8").splitlines()

    carrying = [line for line in lines if _MANUAL_MENTION in line]
    assert carrying, (
        f"the manual no longer quotes {_MANUAL_MENTION} inside the sentence that "
        "forbids it; the sweep passes because the warning is gone"
    )


# ---------------------------------------------------------------------------
# The dataset card
# ---------------------------------------------------------------------------
#
# `BASELINES.md` §4 has required dataset cards since Phase 0.
# `corpus/partition/__init__.py` quotes that requirement as its reason for
# existing and `freeze.py` calls the manifest "the machine-readable half of the
# dataset card §4 asks for" - which left the other half named in three
# docstrings and written nowhere.
#
# The fields below are not expensive to collect and are impossible to recover.
# Everything the corpus keeps is either the sound itself or a pseudonymous
# annotation of it; neither remembers which room, under what consent, or why
# those speakers.

#: What §4 requires of a card, and the section of the template that carries it.
#: Checked as a mapping rather than a word search so that a template mentioning
#: "consent basis" in passing cannot satisfy the requirement to have a section
#: about it. Partition checksums map to §1 because the card's job there is to
#: cite the manifest and refuse to duplicate it.
CARD_SECTIONS = {
    "provenance": "## 2. Provenance",
    "consent basis": "## 3. Consent basis",
    "recording conditions": "## 4. Recording conditions",
    "partition checksums": "## 1. Supplied by the freeze manifest",
}

#: The requirement itself, quoted from `BASELINES.md` §4. Asserted separately so
#: that a change to §4 fails here rather than silently outrunning the template.
CARD_REQUIREMENT = (
    "dataset cards record provenance, consent basis, recording conditions and partition checksums"
)

#: Every field of the frozen manifest, from the manifest's own definitions. The
#: template lists what it does not ask a human to write down; a template that
#: promised a field the manifest does not carry would send somebody looking for
#: it in a file that has never held it.
MANIFEST_FIELDS = frozenset(
    field.name for record in (FrozenCorpus, FrozenRecording) for field in fields(record)
)


def _table_rows(markdown: str) -> list[list[str]]:
    """Every pipe-table row, cell by cell, with the underlines dropped."""
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def test_baselines_still_asks_for_the_four_things_the_template_is_built_around() -> None:
    """The template exists to satisfy one sentence in another document.

    Checked in that direction too: if §4 grows a fifth requirement, this fails
    and somebody has to decide whether the template covers it, rather than the
    template quietly describing an older rule.
    """
    text = _unwrapped(BASELINES.read_text(encoding="utf-8")).lower()

    assert CARD_REQUIREMENT in text


@pytest.mark.parametrize("requirement", sorted(CARD_SECTIONS))
def test_the_template_has_a_section_for_each_thing_a_card_must_record(requirement: str) -> None:
    text = DATASET_CARD.read_text(encoding="utf-8")

    assert CARD_SECTIONS[requirement] in text, (
        f"the dataset card template has no section for {requirement!r}"
    )


def test_the_template_only_claims_manifest_fields_the_manifest_has() -> None:
    """§1 tells a human which fields not to write down.

    It is the half of the card that is easy to get wrong in the expensive
    direction: a field listed there is a field nobody records by hand, so
    naming one the manifest does not carry produces a card with a hole in it
    and a reader who believes the hole is filled somewhere else.
    """
    text = DATASET_CARD.read_text(encoding="utf-8")
    section = text.split(CARD_SECTIONS["partition checksums"], 1)[1].split("\n---", 1)[0]

    # Digits included on purpose: `sha256` is the field most worth checking and
    # the one an identifier pattern without them silently skips.
    claimed = {
        name for row in _table_rows(section) for name in re.findall(r"`([a-z0-9_]+)`", row[0])
    }
    assert claimed, "§1 lists no manifest fields at all"

    invented = sorted(claimed - MANIFEST_FIELDS)
    assert not invented, (
        f"the template says the freeze manifest supplies {invented}, and it does not"
    )


def test_every_field_in_the_template_says_why_it_cannot_be_recovered_later() -> None:
    """The template's whole claim is that each field is cheap now and gone
    later, and a field with no reason beside it is a field somebody will skip
    when the session is running late."""
    rows = _table_rows(DATASET_CARD.read_text(encoding="utf-8"))
    assert rows, "the dataset card template has no field tables"

    thin = [row for row in rows if len(row) != 3 or not all(row)]
    assert not thin, (
        f"{len(thin)} row(s) in the dataset card template are missing a cell; every "
        f"field carries a value and the reason it is unrecoverable: {thin[:3]}"
    )


def test_no_pending_field_in_the_template_is_left_without_a_blocker() -> None:
    """`REFERENCE_ENVIRONMENT.md` and `BASELINE_PINS.md` set the register: a
    value that cannot be filled yet says so and names what is stopping it.

    A bare blank reads as "nothing to say about this", which is the one thing
    none of these fields means.
    """
    rows = _table_rows(DATASET_CARD.read_text(encoding="utf-8"))

    unexplained = [
        cell for row in rows for cell in row if cell.startswith("_pending") and " — " not in cell
    ]
    assert not unexplained, (
        f"pending cells in the dataset card template name no blocker: {unexplained}"
    )


def test_the_template_is_not_mistaken_for_a_filled_card() -> None:
    """It carries no data and has to say so where a reader starts.

    The digest check is the specific accident worth refusing: an example
    sha256 in a template gets copied into the card that replaces it, and a
    plausible sixty-four-character string is the last thing anybody re-derives.
    """
    text = DATASET_CARD.read_text(encoding="utf-8")

    assert "**Status:** template only." in text
    assert "no recording session has happened" in text

    invented_digest = re.search(r"\b[0-9a-f]{64}\b", text)
    assert invented_digest is None, (
        f"the template carries what looks like a real digest: {invented_digest}"
    )
