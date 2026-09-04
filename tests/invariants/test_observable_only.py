"""NFR-024 and FR-023: no endpoint or schema exposes an inferred inner state.

The allowlist in ``domain.shared.taxonomy`` is the real defence - an event type
that is not published cannot be constructed. This module is the tripwire behind
it, and it exists because the allowlist only covers *event types*. A prohibited
concept can still arrive as a field name on a dataclass, a member of some other
enum, or a new indicator someone adds to the prosody list.

So this walks the entire domain package by reflection and fails on any
identifier containing a prohibited concept. It is deliberately blunt: a false
positive costs a rename and thirty seconds of argument, while a false negative
costs the scientific validity §17 says is at stake.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import re
from enum import Enum
from types import ModuleType

import pytest

import evidence_engine.domain as domain_root
from evidence_engine.domain.shared.errors import ProhibitedLabel
from evidence_engine.domain.shared.taxonomy import (
    PROHIBITED_CONCEPTS,
    SPEECH_EVENT_DEFINITIONS,
    VISUAL_EVENT_DEFINITIONS,
    SpeechEventType,
    VisualEventType,
    require_known_speech_event,
    require_known_visual_event,
)

pytestmark = pytest.mark.invariant

#: Words that contain a prohibited substring but are legitimate here.
#: Each one is listed with why, because an unexplained exemption is how this
#: guard gets hollowed out one commit at a time.
_ALLOWED_EXACT: frozenset[str] = frozenset(
    {
        # `confidence` on its own is a calibrated probability, which is exactly
        # what NFR-014 requires every event to carry. Only the compound
        # `confidence_level` - the usual name for a self-assurance score - is
        # prohibited, and it is listed as such in PROHIBITED_CONCEPTS.
        "confidence",
        "high_confidence",
        "is_calibrated",
        "confidence_below_threshold",
        "landmark_confidence",
        "optional_class_min_confidence",
        # `PROHIBITED_CONCEPTS` itself, and the machinery that checks it, must
        # obviously be allowed to name the things it forbids.
        "prohibited_concepts",
        "prohibitedlabel",
        "prohibited_label",
    }
)


def _domain_modules() -> list[ModuleType]:
    modules: list[ModuleType] = [domain_root]
    for info in pkgutil.walk_packages(domain_root.__path__, f"{domain_root.__name__}."):
        modules.append(importlib.import_module(info.name))
    return modules


def _tokens(identifier: str) -> set[str]:
    """Split an identifier into lowercase word tokens.

    `snake_case`, `CamelCase` and `SCREAMING_CASE` all reduce to the same set,
    so `AnxietyScore`, `anxiety_score` and `ANXIETY_SCORE` are equally caught.
    """
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", identifier).replace("_", " ")
    return {token.lower() for token in spaced.split() if token}


def _offends(identifier: str) -> bool:
    if identifier.lower() in _ALLOWED_EXACT:
        return False
    return bool(_tokens(identifier) & PROHIBITED_CONCEPTS)


def test_no_enum_member_names_a_prohibited_concept() -> None:
    offenders: list[str] = []
    for module in _domain_modules():
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not (isinstance(attr, type) and issubclass(attr, Enum)):
                continue
            if attr.__module__ != module.__name__:
                continue
            for member in attr:
                if _offends(member.name) or _offends(str(member.value)):
                    offenders.append(f"{module.__name__}.{attr.__name__}.{member.name}")

    assert not offenders, (
        "NFR-024 forbids exposing emotion, anxiety, personality, deception or "
        f"clinical concepts. Offending enum members: {sorted(offenders)}"
    )


def test_no_dataclass_field_names_a_prohibited_concept() -> None:
    offenders: list[str] = []
    for module in _domain_modules():
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not (isinstance(attr, type) and dataclasses.is_dataclass(attr)):
                continue
            if attr.__module__ != module.__name__:
                continue
            for field in dataclasses.fields(attr):
                if _offends(field.name):
                    offenders.append(f"{module.__name__}.{attr.__name__}.{field.name}")

    assert not offenders, f"NFR-024 forbids these field names: {sorted(offenders)}"


def test_no_public_class_names_a_prohibited_concept() -> None:
    offenders: list[str] = []
    for module in _domain_modules():
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if not isinstance(attr, type) or attr.__module__ != module.__name__:
                continue
            if _offends(attr.__name__):
                offenders.append(f"{module.__name__}.{attr.__name__}")

    assert not offenders, f"NFR-024 forbids these class names: {sorted(offenders)}"


@pytest.mark.parametrize(
    "label",
    [
        "anxiety",
        "nervousness",
        "stress_level",
        "emotional_state",
        "deception",
        "stuttering_diagnosis",
        "personality_trait",
    ],
)
def test_prohibited_speech_label_cannot_be_resolved(label: str) -> None:
    with pytest.raises(ProhibitedLabel, match="taxonomy"):
        require_known_speech_event(label)


@pytest.mark.parametrize("label", ["nervous_gesture", "anxious_posture", "fear_expression"])
def test_prohibited_visual_label_cannot_be_resolved(label: str) -> None:
    with pytest.raises(ProhibitedLabel, match="taxonomy"):
        require_known_visual_event(label)


def test_every_published_class_resolves() -> None:
    """The allowlist and the enums cannot drift apart."""
    for event_type in SpeechEventType:
        assert require_known_speech_event(event_type.value) is event_type
    for visual_type in VisualEventType:
        assert require_known_visual_event(visual_type.value) is visual_type


def test_every_published_class_has_a_definition_and_a_negative_example() -> None:
    """Phase 0's exit criterion needs boundary cases, not just definitions."""
    for event_type in SpeechEventType:
        spec = SPEECH_EVENT_DEFINITIONS[event_type]
        assert spec.definition_es.strip()
        assert spec.positive_example_es.strip()
        assert spec.negative_examples_es, f"{event_type.value} has no negative example"

    for visual_type in VisualEventType:
        spec = VISUAL_EVENT_DEFINITIONS[visual_type]
        assert spec.definition_es.strip()
        assert spec.positive_example_es.strip()
        assert spec.negative_examples_es, f"{visual_type.value} has no negative example"
