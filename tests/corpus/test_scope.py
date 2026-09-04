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

from pathlib import Path

import pytest

from corpus.io.elan import write_template
from corpus.schema.records import DisfluencyAnnotation, Interval, SchemaViolation
from evidence_engine.domain.shared.taxonomy import SpeechEventType, VisualEventType
from tests.corpus.conftest import annotation

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
    write_template(
        path,
        recording_id="pilot-001",
        speaker_pseudonym="P-001",
        annotator_id="ana",
        media_url="pilot-001.wav",
    )
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
