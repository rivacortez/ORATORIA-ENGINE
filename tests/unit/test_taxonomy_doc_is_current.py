"""The annotation manual cannot drift from the taxonomy in code.

Phase 0's exit criterion is inter-annotator agreement, and agreement is
measured against whatever definition the annotators actually read. If the
Markdown and the code disagree, the score that comes back weeks later looks
like a hard boundary case and is really a documentation bug - the most
expensive kind to find, because nothing about it points at the cause.

So the manual is generated and this test fails when the checked-in file is
stale. It is a unit test rather than a CI-only script because a developer who
edits the taxonomy should learn about it in the same second, not in review.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.render_taxonomy import OUTPUT, render


def test_the_annotation_manual_matches_the_taxonomy_in_code() -> None:
    if not OUTPUT.exists():
        pytest.fail(
            "docs/taxonomy/ANNOTATION_MANUAL.md is missing; "
            "run: uv run python scripts/render_taxonomy.py"
        )

    checked_in = OUTPUT.read_text(encoding="utf-8")

    assert checked_in == render(), (
        "the annotation manual is stale. The taxonomy in "
        "src/evidence_engine/domain/shared/taxonomy.py changed without the manual "
        "being regenerated. Run: uv run python scripts/render_taxonomy.py"
    )


def test_the_manual_says_it_is_generated() -> None:
    """A hand edit should be visibly wrong to whoever opens the file."""
    content = OUTPUT.read_text(encoding="utf-8")

    assert "GENERATED FILE" in content
    assert "scripts/render_taxonomy.py" in content


def test_the_manual_lives_where_the_readme_says_it_does() -> None:
    assert (
        Path(__file__).resolve().parents[2] / "docs" / "taxonomy" / ("ANNOTATION_MANUAL.md")
        == OUTPUT
    )
