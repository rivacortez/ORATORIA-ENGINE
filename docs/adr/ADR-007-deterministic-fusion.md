# ADR-007 — Deterministic temporal fusion, and no ranking in the engine

**Status:** Accepted · **Date:** 2026-09-04

## Context

Two decisions that look separate and are the same decision.

**How to fuse.** A learned fusion model could weight co-occurrences by how
predictive they are. It would also be unauditable: NFR-015 requires the same
input and configuration to reproduce equivalent output, and §11.3 iteration 5
wants every co-occurrence recalculable.

**Whether to rank.** Consumers will ask for a severity field so they do not each
reimplement one. It is a reasonable request, and granting it moves the ranking
authority into a service that, by §3 driver 8, only observes.

## Decision

Fusion is deterministic interval correlation over a versioned window. Every
admissible pair is emitted; none is selected as "best", because selecting is
ranking.

`causal_inference` is a `ClassVar` equal to `False`, not a field. A boolean
field defaulting to `False` satisfies the letter of FR-028 and fails its purpose
— any call site could pass `True`. As a `ClassVar` there is no constructor
argument, no deserializer path and no test that can flip it to "check the other
branch". Asserting causality requires editing the domain, which is a reviewable
act.

`ranking_authority` is a `ClassVar` equal to `"none"` for the same reason, and
the assembled payload is additionally walked against a ranking denylist before
publication — for the realistic failure, which is a model runtime forwarding its
own confidence ordering inside an event payload.

## Consequences

Fusion cannot learn that certain pairs matter more. That is the intended
limitation: the engine reports that two things happened near each other, and
what that means is the consumer's question.

Consumers each implement their own ranking. Accepted.

## What would make us revisit this

Nothing about ranking. For fusion, a demonstrated need for graded co-occurrence
strength — which would be a new _observation_ (temporal distance is already
published) rather than a judgement.
