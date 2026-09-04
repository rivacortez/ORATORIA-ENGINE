# ADR-004 — An explicit verbatim taxonomy, as a code allowlist

**Status:** Accepted · **Date:** 2026-09-04

## Context

NFR-024 forbids the service from exposing emotion, anxiety, personality,
deception or clinical diagnoses. §17 names the failure — "multimodal becomes an
emotion detector" — as scientific invalidity rather than an ethical preference,
and it is right to: the geometry is real, the inference from it to an internal
state is not supported by the instrument.

A denylist of forbidden labels only stops the ones somebody remembered to
forbid, and the space of ways to name an inner state is unbounded.

## Decision

The taxonomy is an **allowlist in code** (`domain/shared/taxonomy.py`). An event
type not published there cannot be constructed, serialized or emitted. Each
entry carries its Spanish definition, a positive example and the nearby classes
it must not be confused with — the same text the annotation manual is generated
from, so the annotators and the allowlist cannot disagree.

A denylist (`PROHIBITED_CONCEPTS`) exists as a **tripwire behind** the
allowlist, because the allowlist covers event _types_ and a prohibited concept
could still arrive as a dataclass field name or a member of some other enum. A
reflection test walks every enum member, dataclass field and public class in the
domain.

## Consequences

Adding a class is a code change with a version bump and a regenerated manual —
deliberately more friction than editing a config file. `TAXONOMY_VERSION`
travels on every event, so historical results keep resolving against the version
that produced them.

The reflection test is blunt and will occasionally reject a harmless name. A
false positive costs a rename and thirty seconds of argument; a false negative
costs the scientific validity §17 says is at stake.

## What would make us revisit this

Nothing. The friction is the feature.
