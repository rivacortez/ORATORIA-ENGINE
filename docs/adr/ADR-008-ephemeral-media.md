# ADR-008 — Ephemeral raw-media retention by default

**Status:** Accepted · **Date:** 2026-09-04

## Context

Keeping recordings is convenient for everyone building the system: replay a
session to debug a detector, re-run a promoted model over old audio, show a
student what they actually said. NFR-011 nonetheless makes ephemeral processing
the default, and §17 names the failure mode — "consent withdrawal leaves copies"
— as a legal and ethical breach.

The tension is real, and resolving it _by default_ matters: a retention setting
that defaults to "keep" is a setting nobody changes.

## Decision

Raw media is processed and discarded. Longer retention requires an explicit,
versioned policy with a positive TTL. The constructor rejects a policy claiming
retention without one, and rejects a TTL on a policy that does not retain — both
are symptoms of a caller that has not decided, and an undecided retention policy
becomes an indefinite one.

**Derived non-identifying aggregates survive** deletion of the raw media when
the consent policy allows (§8). Not a loophole: it is what makes NFR-015
reproducibility possible after the recording is gone.

## Consequences

Debugging a detector against real audio requires a session recorded under an
explicit retention policy, arranged in advance. That is friction, and it is the
correct friction.

Evidence references survive in the ledger after the media they pointed at is
deleted, so an audit can distinguish "this was measured and then erased" from
"this was never measured".

## What would make us revisit this

Nothing. If a research use needs retained media, the mechanism already exists —
it just has to be asked for.
