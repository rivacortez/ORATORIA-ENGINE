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

*What is not on a tier.* ELAN's header properties carry everything that
describes the recording rather than a moment in it: both versions, the speaker
and their variety, the consent the recording was collected under, and the
capture chain. Nearly all of them are required, and the reader turns away a file
missing any of them for the reason ``_required_schema_version`` states —
validating before interpreting, so that a file this tool cannot read fails on
the property it lacks rather than three commands later on a number computed
from a default. The few that may be absent are declared with the reason each
one can be.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from corpus.schema.records import (
    SCHEMA_VERSION,
    AnnotatedRecording,
    AnnotationPass,
    ConsentBasis,
    ConsentRecord,
    DisfluencyAnnotation,
    Interval,
    RecordingConditions,
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
#: The record shape the file was written under. Both versions are stored, and
#: both are needed: the taxonomy version says what the *classes* meant, the
#: schema version says what the *file* means. A corpus assembled across a
#: schema change without this is a set of files that look alike and are not.
PROP_SCHEMA = "schema_version"

#: The dialect axis §14.2 asks for error analysis over.
PROP_VARIETY = "speaker_variety"

#: Consent, as `BASELINES.md` §4 requires the dataset card to carry it.
PROP_CONSENT_BASIS = "consent_basis"
PROP_CONSENT_POLICY = "consent_policy_version"
PROP_CONSENT_GRANTED = "consent_granted_on"
PROP_CONSENT_VIDEO = "consent_covers_video"
#: The one optional consent property, and the only one a template never writes.
#: A template is created at recruitment, when nobody has withdrawn; the operator
#: adds this by hand when somebody does, and from that moment every corpus-level
#: command refuses the file. It covers the window between the participant saying
#: so and the file being deleted, which the policy gives 24 hours for.
PROP_CONSENT_WITHDRAWN = "consent_withdrawn_on"

#: The four audio confirmations `REFERENCE_ENVIRONMENT.md` gates Pilot A on,
#: recorded per recording instead of once in a document.
PROP_MICROPHONE = "microphone"
PROP_SAMPLE_RATE = "sample_rate_hz"
PROP_BIT_DEPTH = "bit_depth"
PROP_CHANNELS = "channels"
PROP_VIRTUAL_AUDIO = "virtual_audio_bypassed"
PROP_ROOM_NOTES = "room_notes"

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

    # Both versions, before anything is interpreted. Every step below reads the
    # tiers under the schema's rules and resolves classes under the taxonomy's,
    # so a file this tool cannot read has to be turned away here - otherwise it
    # fails somewhere deeper with a message about time slots and neither
    # version is ever mentioned.
    schema_version = _required_schema_version(properties, path)
    taxonomy_version = _required_taxonomy_version(properties, path)

    slots = _time_slots(root)
    annotator_id = _required_property(
        properties,
        PROP_ANNOTATOR,
        path,
        reason=(
            "Agreement is computed between named annotators; a file that names nobody "
            "cannot take part."
        ),
    )
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

    speaker = _speaker(properties, path)
    return AnnotatedRecording(
        recording_id=recording_id,
        speaker=speaker,
        annotator_id=annotator_id,
        annotation_pass=_required_pass(properties, path),
        duration_ms=duration_ms,
        words=words,
        disfluencies=disfluencies,
        schema_version=schema_version,
        taxonomy_version=taxonomy_version,
        consent=_required_consent(properties, path),
        conditions=_required_conditions(properties, path),
    )


def _speaker(properties: dict[str, str], path: Path) -> Speaker:
    """The participant, with the variety they were recruited as speaking.

    The variety is required for the same reason the versions are: defaulting it
    would be a guess. ``Speaker`` used to default it to ``es-PE``, and a guess
    that names a real dialect is worse than one that names nothing - the dialect
    analysis §14.2 asks for would run over a constant and return one dialect,
    which is exactly what it would return if the corpus really were uniform.
    """
    pseudonym = _required_property(
        properties,
        PROP_SPEAKER,
        path,
        reason=(
            "Agreement is computed between named annotators over a named speaker; a "
            "file that says neither cannot take part."
        ),
    )
    variety = _required_property(
        properties,
        PROP_VARIETY,
        path,
        reason=(
            "§14.2 asks for error analysis by dialect, and a variety nobody stated "
            "cannot be told apart from one everybody shares."
        ),
    )
    try:
        return Speaker(pseudonym=pseudonym, variety=variety)
    except SchemaViolation as error:
        raise ElanError(f"{path.name}: {error}") from error


def _required_consent(properties: dict[str, str], path: Path) -> ConsentRecord:
    """The consent this recording was collected under. Required, and parsed.

    Required at the file boundary rather than at the record, because this is
    where a real participant's data enters the corpus: every recording that will
    ever be annotated arrives as one of these files. A record built in code has
    no participant behind it and carries ``None``; a file always has both.

    Refused rather than defaulted for the reason `BASELINES.md` §4 asks for the
    basis at all. A default basis is an assertion that somebody signed
    something, made by the tool on nobody's behalf, and it is unfalsifiable
    afterwards: an audit reading the manifest cannot distinguish a basis the
    operator recorded from one this function supplied.
    """
    raw_basis = _required_property(
        properties,
        PROP_CONSENT_BASIS,
        path,
        reason=(
            "`BASELINES.md` §4 requires the dataset card to record the consent basis, "
            "and a basis recovered from memory after recruitment is not a basis."
        ),
    )
    try:
        basis = ConsentBasis(raw_basis)
    except ValueError as error:
        published = ", ".join(item.value for item in ConsentBasis)
        raise ElanError(
            f"{path.name}: {raw_basis!r} is not a consent basis. Published values: {published}"
        ) from error

    policy_version = _semantic_version(
        properties,
        PROP_CONSENT_POLICY,
        path,
        reason=(
            "It names the published wording of `CONSENT_AND_RETENTION.md` the "
            "participant actually read; a later edit to that document must not "
            "retroactively become what they agreed to (US-008)."
        ),
    )
    granted_on = _required_date(properties, PROP_CONSENT_GRANTED, path)
    withdrawn_on = _optional_date(properties, PROP_CONSENT_WITHDRAWN, path)
    covers_video = _boolean(properties, PROP_CONSENT_VIDEO, path)

    try:
        return ConsentRecord(
            basis=basis,
            policy_version=policy_version,
            granted_on=granted_on,
            covers_video=covers_video,
            withdrawn_on=withdrawn_on,
        )
    except SchemaViolation as error:
        raise ElanError(f"{path.name}: {error}") from error


def _required_conditions(properties: dict[str, str], path: Path) -> RecordingConditions:
    """The capture chain. Required, because a missing chain reads as a good one.

    `REFERENCE_ENVIRONMENT.md` gates Pilot A on four confirmations and says each
    is made against the recorded file rather than a settings dialog. Nothing
    here defaults: a default ``sample_rate_hz`` of 16000 would state the
    *target* from that document as though it were the measurement, and the
    document's whole argument is that the two are different.
    """
    reason = (
        "`REFERENCE_ENVIRONMENT.md` gates Pilot A on the capture chain, and a chain "
        "recorded once in a document cannot say which session drifted."
    )
    microphone = _required_property(properties, PROP_MICROPHONE, path, reason=reason)
    try:
        return RecordingConditions(
            microphone=microphone,
            sample_rate_hz=_positive_int(properties, PROP_SAMPLE_RATE, path, reason),
            bit_depth=_positive_int(properties, PROP_BIT_DEPTH, path, reason),
            channels=_positive_int(properties, PROP_CHANNELS, path, reason),
            virtual_audio_bypassed=_boolean(properties, PROP_VIRTUAL_AUDIO, path),
            room_notes=properties.get(PROP_ROOM_NOTES, ""),
        )
    except SchemaViolation as error:
        raise ElanError(f"{path.name}: {error}") from error


def _semantic_version(
    properties: dict[str, str], name: str, path: Path, *, reason: str
) -> SemanticVersion:
    raw = _required_property(properties, name, path, reason=reason)
    try:
        return SemanticVersion.parse(raw)
    except Exception as error:
        raise ElanError(f"{path.name}: '{raw}' is not a version in '{name}'. {reason}") from error


def _optional_date(properties: dict[str, str], name: str, path: Path) -> date | None:
    """An ISO-8601 calendar date, or ``None`` where the property is absent.

    ``date.fromisoformat`` and nothing more permissive. A parser that accepted
    "04/09/2026" would have to choose between two readings of it that are five
    months apart, and it would choose silently.

    Absent is ``None`` here and only here, because this reads the one date that
    is legitimately absent - a participant who has not withdrawn. Every other
    property goes through ``_required_date``, which refuses.
    """
    raw = properties.get(name, "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as error:
        raise ElanError(
            f"{path.name}: '{raw}' in '{name}' is not an ISO-8601 date such as 2026-09-04"
        ) from error


def _required_date(properties: dict[str, str], name: str, path: Path) -> date:
    parsed = _optional_date(properties, name, path)
    if parsed is None:
        raise ElanError(
            f"{path.name} has no '{name}' property. Consent is captured at recruitment "
            "or it is unreconstructable. Use a template from `corpus template`."
        )
    return parsed


def _boolean(properties: dict[str, str], name: str, path: Path) -> bool:
    """Exactly ``true`` or ``false``, case-insensitively. Nothing else.

    Not ``value.lower() == "true"``. Under that rule every typo, every "yes",
    every "1" and every empty property becomes ``False`` - so an operator who
    confirmed the virtual audio chains were bypassed and wrote "yes" is recorded
    as having confirmed the opposite, and the refusal they eventually get points
    at their recording rather than at their spelling.
    """
    raw = properties.get(name, "").strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if not raw:
        raise ElanError(
            f"{path.name} has no '{name}' property. Use a template from `corpus template`."
        )
    raise ElanError(
        f"{path.name}: '{raw}' in '{name}' is not 'true' or 'false'. Anything else has to "
        "be guessed at, and the guess would be recorded as the operator's confirmation."
    )


def _positive_int(properties: dict[str, str], name: str, path: Path, reason: str) -> int:
    raw = _required_property(properties, name, path, reason=reason)
    try:
        return int(raw)
    except ValueError as error:
        raise ElanError(f"{path.name}: '{raw}' in '{name}' is not a whole number") from error


def _required_schema_version(properties: dict[str, str], path: Path) -> SemanticVersion:
    """The record shape this file was written under. Required, and checked.

    Required rather than defaulted, because a default is a guess: assuming the
    current version for a file that does not say would let a file written under
    a future shape be read as if it were this one, and the failure would appear
    as an agreement figure rather than as an error.

    Checked to one major version, the compatibility rule the rest of the system
    uses (NFR-017). A minor-version difference is additive by definition and
    reads fine; a major one is a different record.
    """
    raw = _required_property(
        properties,
        PROP_SCHEMA,
        path,
        reason="It says what the *file* means; assuming the current shape would be a guess.",
    )
    try:
        version = SemanticVersion.parse(raw)
    except Exception as error:
        raise ElanError(f"{path.name}: '{raw}' is not a schema version") from error

    if not version.is_compatible_with(SCHEMA_VERSION):
        raise ElanError(
            f"{path.name} was written under schema {version}; this tool reads "
            f"{SCHEMA_VERSION}. Reading it anyway would compare two different record "
            "shapes and report the difference as annotator disagreement."
        )
    return version


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


_DEFAULT_MISSING_REASON = (
    "It records something about the file that cannot be recovered from the annotations."
)


def _required_property(
    properties: dict[str, str], name: str, path: Path, *, reason: str = _DEFAULT_MISSING_REASON
) -> str:
    """One header property, or a refusal that says which one and why it matters.

    The reason travels with the property rather than being one sentence for all
    of them. There used to be four of these and one shared sentence about named
    annotators and named speakers; there are now enough that the shared sentence
    is wrong for most of them, and an annotator who reads "agreement is computed
    between named annotators" under a complaint about ``sample_rate_hz`` learns
    that the tool is confused rather than what to fix.
    """
    value = properties.get(name, "")
    if not value:
        raise ElanError(
            f"{path.name} has no '{name}' property. {reason} Use a template from `corpus template`."
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


def _required_pass(properties: dict[str, str], path: Path) -> AnnotationPass:
    """Which pass this file is, refused by name rather than by traceback.

    ``AnnotationPass(raw)`` raises a bare ``ValueError`` naming the enum, which
    reaches the annotator as a Python traceback about a class they have never
    heard of. The pass decides whether the file may enter an agreement
    computation at all - an adjudicated one may not - so a mistyped value is
    not a cosmetic problem.

    Absent still means ``first``. That default is safe in the direction that
    matters: it is the *stricter* reading, because `first` participates in
    agreement and `adjudicated` is refused, so a file that forgot to say cannot
    smuggle itself past the independence guard by omission.
    """
    raw = properties.get(PROP_PASS, AnnotationPass.FIRST.value)
    try:
        return AnnotationPass(raw)
    except ValueError as error:
        published = ", ".join(item.value for item in AnnotationPass)
        raise ElanError(
            f"{path.name}: '{raw}' is not an annotation_pass. Published values: "
            f"{published}. The pass decides whether this file may take part in an "
            "agreement computation, so it is not guessed at."
        ) from error


def _required_taxonomy_version(properties: dict[str, str], path: Path) -> SemanticVersion:
    """The taxonomy the annotator worked under. Required, and strictly parsed.

    This used to be lenient - absent or unparseable became ``None``, on the
    reasoning that an annotation whose taxonomy version is unreadable is still
    usable evidence and the report could flag the gap. That was wrong, and the
    way it was wrong is instructive: ``None`` does not compare unequal to
    anything, so a file with no version passed straight through the comparison
    guard and got measured against a file from a different manual. A field that
    silently becomes "unknown" is worse than a field that is missing, because
    the checks downstream are written against the value and not against its
    absence.

    Strict rather than one-major-version tolerant, matching
    ``EvidenceDocument``, which refuses a manifest whose taxonomy is not
    exactly this build's. The protocol is stricter still: *any* taxonomy change
    requires re-running the pilot, because a definition amended in a minor
    release is a definition two annotators did not share.
    """
    raw = _required_property(
        properties,
        PROP_TAXONOMY,
        path,
        reason=(
            "It says which manual the annotator worked under, and agreement between two "
            "manuals is not agreement between two annotators."
        ),
    )
    try:
        return SemanticVersion.parse(raw)
    except Exception as error:
        raise ElanError(
            f"{path.name}: '{raw}' is not a taxonomy version. It records which "
            "manual the annotator worked under, and agreement between two manuals "
            "is not agreement between two annotators."
        ) from error


# ---------------------------------------------------------------------------
# Writing the template
# ---------------------------------------------------------------------------


class WouldOverwrite(ElanError):
    """Writing here would destroy an existing file."""


def write_template(
    path: Path,
    *,
    recording_id: str,
    speaker_pseudonym: str,
    speaker_variety: str,
    annotator_id: str,
    media_url: str,
    consent: ConsentRecord,
    conditions: RecordingConditions,
    annotation_pass: AnnotationPass = AnnotationPass.FIRST,
    overwrite: bool = False,
) -> None:
    """Write an empty EAF wired with the taxonomy as controlled vocabularies.

    This is the single most valuable thing in this module. An annotator opening
    the template picks a class from a list generated from the published
    allowlist and sees its Spanish definition as the entry description — so the
    manual and the tool cannot disagree, and a class that is not published
    cannot be typed.

    ``speaker_variety``, ``consent`` and ``conditions`` have no defaults. They
    could have had them, and the file would still open; it is the *reader* that
    would then refuse it, which is the worst place to find out. A tool whose
    ``template`` verb can produce a file its own ``validate`` verb rejects has
    taught the annotator that the errors are noise.

    Refuses to overwrite unless asked. An empty template and a finished
    annotation are the same kind of file with the same natural name, so
    regenerating a template over a day of somebody's work is one mistyped path
    away and there is nothing to recover it from.
    """
    if not consent.is_active:
        raise ElanError(
            f"consent for {speaker_pseudonym} was withdrawn on {consent.withdrawn_on}. "
            "A template starts three hours of annotation on a recording that must be "
            "deleted, and every corpus command downstream would refuse the result."
        )
    if path.exists() and not overwrite:
        raise WouldOverwrite(
            f"{path} already exists. A finished annotation and an empty template are "
            "the same kind of file, and this would replace one with the other. Pass "
            "--force if that is what you meant."
        )

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
    # PROP_CONSENT_WITHDRAWN is deliberately absent: a template is written at
    # recruitment, and writing "consent_withdrawn_on: " empty would invite an
    # annotator to read the blank as a considered "not withdrawn" rather than as
    # a field nobody has touched.
    for name, value in (
        (PROP_RECORDING, recording_id),
        (PROP_SPEAKER, speaker_pseudonym),
        (PROP_VARIETY, speaker_variety),
        (PROP_ANNOTATOR, annotator_id),
        (PROP_PASS, annotation_pass.value),
        (PROP_SCHEMA, str(SCHEMA_VERSION)),
        (PROP_TAXONOMY, str(TAXONOMY_VERSION)),
        (PROP_CONSENT_BASIS, consent.basis.value),
        (PROP_CONSENT_POLICY, str(consent.policy_version)),
        (PROP_CONSENT_GRANTED, consent.granted_on.isoformat()),
        (PROP_CONSENT_VIDEO, _as_property(consent.covers_video)),
        (PROP_MICROPHONE, conditions.microphone),
        (PROP_SAMPLE_RATE, str(conditions.sample_rate_hz)),
        (PROP_BIT_DEPTH, str(conditions.bit_depth)),
        (PROP_CHANNELS, str(conditions.channels)),
        (PROP_VIRTUAL_AUDIO, _as_property(conditions.virtual_audio_bypassed)),
        (PROP_ROOM_NOTES, conditions.room_notes),
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


def _as_property(value: bool) -> str:
    """``true``/``false``. Not ``str(value)``.

    ``str(True)`` is ``"True"``, which is Python's spelling leaking into a file
    format that ELAN, this reader and any future importer all have to agree on.
    ``_boolean`` happens to accept it today; a tightening that stopped would
    make this module's own template unreadable by this module's own reader.
    """
    return "true" if value else "false"


def _mime_for(media_url: str) -> str:
    suffix = Path(media_url).suffix.lower()
    return {
        ".wav": "audio/x-wav",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }.get(suffix, "audio/x-wav")
