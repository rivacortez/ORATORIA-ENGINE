"""No artefact a person reads may print `0` for something never measured.

FR-025 was enforced everywhere the domain could reach: ``Unavailable`` has no
``value``, the serializer emits disjoint key sets, the DDL carries XOR check
constraints. And then ``scripts/present.py`` - the one command a person
actually runs - printed this, with the explanation beside it:

    disfluency events    0  - the detector is Phase 4 and does not exist.
                            Zero is the honest answer, not a finding.

It is not. Zero means a detector ran and found none. A reader scanning a column
of numbers reads the number; the sentence next to it is not part of the number.
`README.md` defended the same wording in prose, so the claim had propagated
from the code into the documentation that described the code.

This sweep runs over the artefacts whose output is *read as figures* rather
than parsed. It is deliberately crude - a substring scan over source text -
because the failure it catches is crude, and it survived four adversarial
review rounds by being written in a place no test looked at.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.invariant

ROOT = Path(__file__).resolve().parents[2]

#: What a person runs and reads as numbers. The engine's own JSON is covered by
#: the serializer tests; this is about the human-facing surfaces, which is where
#: the defect lived.
REPORTING_ARTEFACTS = (
    ROOT / "scripts" / "present.py",
    ROOT / "README.md",
)

#: A label followed by a bare zero, which is what the defect looked like. The
#: label list is the set of things nothing in this build can measure; each is
#: Phase 4 or Phase 5 and must be reported as unavailable until it is not.
_UNMEASURED = ("disfluency", "prosody", "visual event", "filled pause", "false start")
_LABEL_THEN_ZERO = re.compile(
    r"(" + "|".join(_UNMEASURED) + r")[a-z ]*\s+0\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("path", REPORTING_ARTEFACTS, ids=lambda p: p.name)
def test_no_unmeasured_quantity_is_printed_as_zero(path: Path) -> None:
    """The defect, as a regression.

    If this fails, some artefact is about to tell a reader that a detector
    which does not exist looked and found nothing.
    """
    text = path.read_text(encoding="utf-8")

    # Quoted spans are stripped in Markdown and **not** in Python, and getting
    # this wrong is how the first version of this test passed against the
    # defect it was written for. Prose quotes the old wording on purpose so a
    # reader learns why it was wrong; Python source keeps everything it prints
    # inside string literals, so stripping quotes there deletes exactly the
    # text under examination. The rule follows the file type, not the syntax.
    scanned = re.sub(r"`[^`\n]*`|\"[^\"\n]*\"", "", text) if path.suffix == ".md" else text
    offenders = _LABEL_THEN_ZERO.findall(scanned)

    assert not offenders, (
        f"{path.name} prints a bare 0 for {offenders}, which reads as "
        "'the detector ran and found none'. Report it as unavailable with a reason."
    )


@pytest.mark.parametrize("path", REPORTING_ARTEFACTS, ids=lambda p: p.name)
def test_the_artefact_says_unavailable_instead(path: Path) -> None:
    """Removing the zero is not enough; the absence has to be stated.

    An artefact that simply omitted the line would leave a reader to assume the
    engine measured everything it did not mention, which is the same failure
    with fewer words.
    """
    text = path.read_text(encoding="utf-8").lower()
    assert "unavailable" in text
    assert "detector_not_deployed" in text


def test_the_report_declares_how_many_words_it_could_not_place() -> None:
    """The count the temporal figures were computed without.

    Words-per-minute and the pause list run on placed tokens only. Publishing
    them beside a transcript that contains more words than they saw, without
    saying so, would understate the rate by however many the aligner missed.
    """
    text = (ROOT / "scripts" / "present.py").read_text(encoding="utf-8")

    assert "unplaced" in text
    assert "is_timed" in text
