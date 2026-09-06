"""Every way an EAF can be malformed, and the message the annotator gets.

This is the file every annotator's three hours passes through. A refusal here
that says "invalid value" instead of naming the property and the file is a
refusal that costs a support conversation; a *missing* refusal is worse, because
the value gets a default and the default becomes corpus data.

The paths below were all written and none was exercised. Each test names the
property it breaks and asserts the message identifies it, because "which of the
fourteen properties" is the only question the annotator has.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.io.elan import ElanError, read
from tests.corpus.conftest import template
from tests.corpus.test_cli import _annotated

#: The properties a reader refuses a file for omitting, and a fragment of the
#: message each refusal must contain. Driven from one table so that a property
#: added without a refusal shows up as a missing row rather than as nothing.
REQUIRED = [
    ("annotator_id", "annotator_id"),
    ("speaker_pseudonym", "speaker_pseudonym"),
    ("speaker_variety", "speaker_variety"),
    ("schema_version", "schema_version"),
    ("taxonomy_version", "taxonomy_version"),
    ("consent_basis", "consent_basis"),
    ("consent_policy_version", "consent_policy_version"),
    ("consent_granted_on", "consent_granted_on"),
    ("consent_covers_video", "consent_covers_video"),
    ("microphone", "microphone"),
    ("sample_rate_hz", "sample_rate_hz"),
    ("bit_depth", "bit_depth"),
    ("channels", "channels"),
    ("virtual_audio_bypassed", "virtual_audio_bypassed"),
]


def _written(tmp_path: Path) -> Path:
    """A *filled* file, which is what an annotator hands back.

    Not an empty template: a template has no time slots, so reading one back
    fails on its duration before the header is fully checked, and every test
    below would assert against the wrong message. That refusal is correct and
    has its own test at the bottom.
    """
    return _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")


def _with(
    tmp_path: Path, *, drop: str | None = None, set_to: tuple[str, str] | None = None
) -> Path:
    """A template with one property removed or given a bad value."""
    path = _written(tmp_path)
    text = path.read_text(encoding="utf-8")
    if drop is not None:
        text = text.replace(f'NAME="{drop}"', 'NAME="unused"', 1)
    if set_to is not None:
        name, value = set_to
        start = text.index(f'NAME="{name}">') + len(f'NAME="{name}">')
        end = text.index("<", start)
        text = text[:start] + value + text[end:]
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(("prop", "fragment"), REQUIRED, ids=[p for p, _ in REQUIRED])
def test_a_missing_property_is_refused_by_name(tmp_path: Path, prop: str, fragment: str) -> None:
    with pytest.raises(ElanError, match=fragment):
        read(_with(tmp_path, drop=prop))


@pytest.mark.parametrize(
    ("prop", "value", "fragment"),
    [
        ("speaker_variety", "Peruvian Spanish", "language tag"),
        ("consent_basis", "verbal_nod", "not a consent basis"),
        ("consent_policy_version", "v1", "consent_policy_version"),
        ("consent_granted_on", "01/09/2026", "consent_granted_on"),
        ("consent_covers_video", "yes", "true"),
        ("sample_rate_hz", "sixteen thousand", "sample_rate_hz"),
        ("bit_depth", "16.5", "bit_depth"),
        ("channels", "-1", "channels"),
        ("virtual_audio_bypassed", "probably", "true"),
        ("annotation_pass", "third", "not an annotation_pass"),
    ],
    ids=lambda item: item if isinstance(item, str) and " " not in item else "",
)
def test_an_unreadable_value_is_refused_by_name(
    tmp_path: Path, prop: str, value: str, fragment: str
) -> None:
    """A value the reader cannot interpret is refused rather than defaulted.

    Every one of these has a plausible-looking default sitting next to it -
    16 000 Hz, `first`, `false` - and taking it would put a number nobody
    confirmed into the corpus under the name of a confirmation.
    """
    with pytest.raises(ElanError, match=fragment):
        read(_with(tmp_path, set_to=(prop, value)))


def test_a_file_with_no_annotations_and_no_time_slots_is_refused(tmp_path: Path) -> None:
    """An empty template read back. Its duration would be zero, and a recording
    of zero length makes every downstream proportion a division by nothing."""
    path = tmp_path / "empty.eaf"
    template(path)

    with pytest.raises(ElanError, match="no annotations and no time slots"):
        read(path)


def test_a_class_outside_the_published_taxonomy_names_the_stale_template(
    tmp_path: Path,
) -> None:
    """The controlled vocabulary should have prevented it, so the message
    points at the template rather than at the annotator."""
    path = _annotated(tmp_path, "ana", offset_ms=0, event="filled_pause")
    path.write_text(
        path.read_text(encoding="utf-8").replace(">filled_pause<", ">nervousness<", 1),
        encoding="utf-8",
    )

    with pytest.raises(ElanError, match="not a published class"):
        read(path)
