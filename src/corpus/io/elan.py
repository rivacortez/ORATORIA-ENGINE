"""ELAN (.eaf) as the annotation surface.

An existing tool, configured — not an editor written here. ELAN was chosen over
Praat TextGrid and over a bespoke web editor for three reasons, in order of
weight:

*Controlled vocabularies enforce the taxonomy at annotation time.* An annotator
picks from a list generated from ``domain.shared.taxonomy``; they cannot type a
class that is not published. That removes an entire category of disagreement —
the kind where two annotators used different words for the same thing — before
it is ever measured.

*Symbolic association ties a contextual role to its disfluency.* The role is a
child annotation of the event, so the pair cannot come apart. Encoding them as
two independent time-aligned tiers would let an annotator produce a role with
no event, or shift one and not the other.

*It is multimodal.* Phase 5 annotates visual events on the same timeline, and
a tool that handles only audio would have to be replaced exactly when the
corpus is most expensive to re-annotate.

One EAF per annotator, not one file with a tier per annotator. Double
annotation is only meaningful blind, and a shared file makes the second
annotator's work a review of the first.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from corpus.schema.records import (
    AnnotatedRecording,
    AnnotationPass,
    DisfluencyAnnotation,
    Interval,
    SchemaViolation,
    Speaker,
    Word,
)
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import (
    TAXONOMY_VERSION,
    ContextualRole,
    SpeechEventType,
    definition_of,
)

#: Tier names. Fixed rather than configurable: a corpus where each annotator
#: named their tiers differently is a corpus that needs a mapping file per
#: recording, and the mapping file is where the errors live.
WORDS_TIER = "words"
DISFLUENCY_TIER = "disfluency"
ROLE_TIER = "role"
TEXT_TIER = "text"
NOTE_TIER = "note"

#: Document properties carrying what ELAN has no native slot for.
PROP_ANNOTATOR = "annotator_id"
PROP_SPEAKER = "speaker_pseudonym"
PROP_PASS = "annotation_pass"
PROP_TAXONOMY = "taxonomy_version"
PROP_RECORDING = "recording_id"

_CV_DISFLUENCY = "disfluency_classes"
_CV_ROLE = "contextual_roles"


class ElanError(Exception):
    """An EAF file could not be read as an annotation of this corpus."""


@dataclass(frozen=True, slots=True)
class _Annotation:
    """One time-aligned annotation, before it becomes a domain record."""

    annotation_id: str
    start_ms: int
    end_ms: int
    value: str


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read(path: Path) -> AnnotatedRecording:
    """Parse one annotator's EAF into the canonical record.

    Raises rather than skipping on anything it cannot interpret. A silently
    dropped tier produces an agreement figure computed over less data than the
    annotators produced, which reads as disagreement and is a parsing bug.
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as error:
        raise ElanError(f"{path.name} is not well-formed XML: {error}") from error

    root = tree.getroot()
    properties = _properties(root)
    slots = _time_slots(root)

    annotator_id = _required_property(properties, PROP_ANNOTATOR, path)
    recording_id = properties.get(PROP_RECORDING) or path.stem

    words = tuple(
        Word(text=item.value, interval=Interval(item.start_ms, item.end_ms))
        for item in _aligned(root, WORDS_TIER, slots)
    )

    roles = _referenced(root, ROLE_TIER)
    texts = _referenced(root, TEXT_TIER)
    notes = _referenced(root, NOTE_TIER)

    disfluencies = tuple(
        _to_disfluency(item, annotator_id, roles, texts, notes, path)
        for item in _aligned(root, DISFLUENCY_TIER, slots)
    )

    duration_ms = _duration(slots, words, disfluencies)

    return AnnotatedRecording(
        recording_id=recording_id,
        speaker=Speaker(pseudonym=_required_property(properties, PROP_SPEAKER, path)),
        annotator_id=annotator_id,
        annotation_pass=AnnotationPass(properties.get(PROP_PASS, AnnotationPass.FIRST.value)),
        duration_ms=duration_ms,
        words=words,
        disfluencies=disfluencies,
        taxonomy_version=_parse_version(properties.get(PROP_TAXONOMY)),
    )


def _to_disfluency(
    item: _Annotation,
    annotator_id: str,
    roles: dict[str, str],
    texts: dict[str, str],
    notes: dict[str, str],
    path: Path,
) -> DisfluencyAnnotation:
    try:
        event_type = SpeechEventType(item.value)
    except ValueError as error:
        raise ElanError(
            f"{path.name}: '{item.value}' is not a published class. The controlled "
            "vocabulary should have prevented this - the template is probably stale; "
            "regenerate it with `corpus template`."
        ) from error

    role_value = roles.get(item.annotation_id)
    try:
        role = ContextualRole(role_value) if role_value else None
    except ValueError as error:
        raise ElanError(f"{path.name}: '{role_value}' is not a contextual role") from error

    try:
        return DisfluencyAnnotation(
            event_type=event_type,
            interval=Interval(item.start_ms, item.end_ms),
            annotator_id=annotator_id,
            raw_text=texts.get(item.annotation_id, ""),
            context_role=role,
            note=notes.get(item.annotation_id, ""),
        )
    except SchemaViolation as error:
        # Re-raised with the file and position, because an annotator reading
        # "raw_text is required" needs to know which of four hundred
        # annotations it is about.
        raise ElanError(
            f"{path.name} at {item.start_ms}-{item.end_ms} ms ({item.value}): {error}"
        ) from error


def _properties(root: ET.Element) -> dict[str, str]:
    return {
        name: (element.text or "").strip()
        for element in root.iterfind("./HEADER/PROPERTY")
        if (name := element.get("NAME")) is not None
    }


def _required_property(properties: dict[str, str], name: str, path: Path) -> str:
    value = properties.get(name, "")
    if not value:
        raise ElanError(
            f"{path.name} has no '{name}' property. Agreement is computed between "
            "named annotators over a named speaker; a file that says neither cannot "
            "take part. Use a template from `corpus template`."
        )
    return value


def _time_slots(root: ET.Element) -> dict[str, int]:
    slots: dict[str, int] = {}
    for element in root.iterfind("./TIME_ORDER/TIME_SLOT"):
        slot_id = element.get("TIME_SLOT_ID")
        value = element.get("TIME_VALUE")
        if slot_id is not None and value is not None:
            slots[slot_id] = int(value)
    return slots


def _aligned(root: ET.Element, tier_id: str, slots: dict[str, int]) -> list[_Annotation]:
    """Time-aligned annotations on a tier, in temporal order."""
    found: list[_Annotation] = []
    for tier in root.iterfind("./TIER"):
        if tier.get("TIER_ID") != tier_id:
            continue
        for element in tier.iterfind("./ANNOTATION/ALIGNABLE_ANNOTATION"):
            annotation_id = element.get("ANNOTATION_ID")
            start = slots.get(element.get("TIME_SLOT_REF1", ""))
            end = slots.get(element.get("TIME_SLOT_REF2", ""))
            if annotation_id is None or start is None or end is None:
                raise ElanError(
                    f"annotation {annotation_id} on tier '{tier_id}' references a time "
                    "slot that does not exist"
                )
            value = element.findtext("./ANNOTATION_VALUE", default="").strip()
            found.append(_Annotation(annotation_id, start, end, value))
    found.sort(key=lambda item: (item.start_ms, item.end_ms, item.annotation_id))
    return found


def _referenced(root: ET.Element, tier_id: str) -> dict[str, str]:
    """Symbolic-association annotations, keyed by the id they refer to."""
    values: dict[str, str] = {}
    for tier in root.iterfind("./TIER"):
        if tier.get("TIER_ID") != tier_id:
            continue
        for element in tier.iterfind("./ANNOTATION/REF_ANNOTATION"):
            reference = element.get("ANNOTATION_REF")
            if reference is None:
                continue
            values[reference] = element.findtext("./ANNOTATION_VALUE", default="").strip()
    return values


def _duration(
    slots: dict[str, int],
    words: tuple[Word, ...],
    disfluencies: tuple[DisfluencyAnnotation, ...],
) -> int:
    """The recording's length, from the furthest point anyone annotated.

    ELAN does not record media duration reliably - the header's media
    descriptor is often absent or wrong after a file move - so the last time
    slot is used instead. It is a lower bound, and the only consequence of
    underestimating is that the agreement report's coverage figure is
    conservative.
    """
    candidates = [max(slots.values(), default=0)]
    candidates.extend(word.interval.end_ms for word in words)
    candidates.extend(item.interval.end_ms for item in disfluencies)
    duration = max(candidates)
    if duration <= 0:
        raise ElanError("the file contains no annotations and no time slots")
    return duration


def _parse_version(raw: str | None) -> SemanticVersion | None:
    if not raw:
        return None
    try:
        return SemanticVersion.parse(raw)
    except Exception:
        # Recorded as absent rather than rejected: an annotation whose taxonomy
        # version is unreadable is still usable evidence, and the agreement
        # report flags the gap rather than refusing the file.
        return None


# ---------------------------------------------------------------------------
# Writing the template
# ---------------------------------------------------------------------------


def write_template(
    path: Path,
    *,
    recording_id: str,
    speaker_pseudonym: str,
    annotator_id: str,
    media_url: str,
    annotation_pass: AnnotationPass = AnnotationPass.FIRST,
) -> None:
    """Write an empty EAF wired with the taxonomy as controlled vocabularies.

    This is the single most valuable thing in this module. An annotator opening
    the template picks a class from a list generated from the published
    allowlist and sees its Spanish definition as the entry description — so the
    manual and the tool cannot disagree, and a class that is not published
    cannot be typed.
    """
    document = ET.Element(
        "ANNOTATION_DOCUMENT",
        {
            "AUTHOR": annotator_id,
            "DATE": "2026-09-04T00:00:00+00:00",
            "FORMAT": "3.0",
            "VERSION": "3.0",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        },
    )

    header = ET.SubElement(document, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})
    ET.SubElement(
        header,
        "MEDIA_DESCRIPTOR",
        {"MEDIA_URL": media_url, "MIME_TYPE": _mime_for(media_url), "RELATIVE_MEDIA_URL": ""},
    )
    for name, value in (
        (PROP_RECORDING, recording_id),
        (PROP_SPEAKER, speaker_pseudonym),
        (PROP_ANNOTATOR, annotator_id),
        (PROP_PASS, annotation_pass.value),
        (PROP_TAXONOMY, str(TAXONOMY_VERSION)),
    ):
        ET.SubElement(header, "PROPERTY", {"NAME": name}).text = value

    ET.SubElement(document, "TIME_ORDER")

    ET.SubElement(document, "TIER", {"LINGUISTIC_TYPE_REF": "verbatim", "TIER_ID": WORDS_TIER})
    ET.SubElement(
        document, "TIER", {"LINGUISTIC_TYPE_REF": "disfluency_class", "TIER_ID": DISFLUENCY_TIER}
    )
    for tier_id, type_ref in (
        (ROLE_TIER, "contextual_role"),
        (TEXT_TIER, "attached_text"),
        (NOTE_TIER, "attached_text"),
    ):
        ET.SubElement(
            document,
            "TIER",
            {
                "LINGUISTIC_TYPE_REF": type_ref,
                "PARENT_REF": DISFLUENCY_TIER,
                "TIER_ID": tier_id,
            },
        )

    ET.SubElement(
        document,
        "LINGUISTIC_TYPE",
        {"GRAPHIC_REFERENCES": "false", "LINGUISTIC_TYPE_ID": "verbatim", "TIME_ALIGNABLE": "true"},
    )
    ET.SubElement(
        document,
        "LINGUISTIC_TYPE",
        {
            "CONTROLLED_VOCABULARY_REF": _CV_DISFLUENCY,
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "disfluency_class",
            "TIME_ALIGNABLE": "true",
        },
    )
    ET.SubElement(
        document,
        "LINGUISTIC_TYPE",
        {
            "CONSTRAINTS": "Symbolic_Association",
            "CONTROLLED_VOCABULARY_REF": _CV_ROLE,
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "contextual_role",
            "TIME_ALIGNABLE": "false",
        },
    )
    ET.SubElement(
        document,
        "LINGUISTIC_TYPE",
        {
            "CONSTRAINTS": "Symbolic_Association",
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "attached_text",
            "TIME_ALIGNABLE": "false",
        },
    )
    ET.SubElement(
        document,
        "CONSTRAINT",
        {
            "DESCRIPTION": "1-1 association with a parent annotation",
            "STEREOTYPE": "Symbolic_Association",
        },
    )

    _controlled_vocabulary(
        document,
        _CV_DISFLUENCY,
        "Clases publicadas de la taxonomia. Generado; no editar a mano.",
        {event.value: definition_of(event).definition_es for event in SpeechEventType},
    )
    _controlled_vocabulary(
        document,
        _CV_ROLE,
        "Roles contextuales (FR-013). Solo para clases lexicas.",
        {
            ContextualRole.FILLER.value: "Hesitacion. No aporta contenido proposicional.",
            ContextualRole.SEMANTIC.value: "Tiene significado lexico en esa posicion.",
            ContextualRole.DISCOURSE_MARKER.value: "Estructura el discurso.",
            ContextualRole.UNCERTAIN.value: (
                "El contexto no lo decide. Se reporta asi; no se adivina."
            ),
        },
    )

    ET.indent(document, space="    ")
    path.write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(document, encoding="utf-8")
    )


def _controlled_vocabulary(
    document: ET.Element, cv_id: str, description: str, entries: dict[str, str]
) -> None:
    vocabulary = ET.SubElement(document, "CONTROLLED_VOCABULARY", {"CV_ID": cv_id})
    ET.SubElement(vocabulary, "DESCRIPTION", {"LANG_REF": "und"}).text = description
    for index, (value, meaning) in enumerate(entries.items(), start=1):
        entry = ET.SubElement(vocabulary, "CV_ENTRY_ML", {"CVE_ID": f"cve{index}"})
        ET.SubElement(entry, "CVE_VALUE", {"DESCRIPTION": meaning, "LANG_REF": "und"}).text = value


def _mime_for(media_url: str) -> str:
    suffix = Path(media_url).suffix.lower()
    return {
        ".wav": "audio/x-wav",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }.get(suffix, "audio/x-wav")
