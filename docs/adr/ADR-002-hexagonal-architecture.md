# ADR-002 — Hexagonal architecture with a machine-checked dependency rule

**Status:** Accepted · **Date:** 2026-09-04

## Context

§4.3 states the dependency direction — `Adapters -> Application -> Domain` — and
lists what the domain must not import: FastAPI, SQLAlchemy, Redis, PyTorch,
MLflow, vendor SDKs.

Stated in a document, that rule decays. Not through carelessness, through
plausible local decisions. Somebody needs a timestamp and reaches for
`datetime.now()`. Somebody needs to serialize an entity and adds a Pydantic base
class "just for this one". Each is defensible alone; together they make the
domain untestable without infrastructure and unmovable between runtimes.

## Decision

Hexagonal architecture, with the rule enforced by `import-linter` in CI rather
than by review. Six contracts, each guarding a claim the specification makes:

| Contract                                | Guards                                          |
| --------------------------------------- | ----------------------------------------------- |
| C1 dependency rule                      | §4.3's direction                                |
| C2 domain is framework-free             | §4.3's prohibition, plus Pydantic               |
| C3 application is infrastructure-free   | NFR-016's replaceable runtimes                  |
| C4 shared kernel knows nobody           | keeps C5 from becoming decorative               |
| C5 observation packages are independent | QA-02's modality independence                   |
| C6 inbound does not reach outbound      | keeps quotas and the state machine non-optional |

Pydantic is on C2's list even though §4.3 does not name it. The domain expresses
its invariants in its own constructors, so that changing the wire format cannot
silently relax a rule.

## Consequences

The domain is plain dataclasses with hand-written validation, which is more code
than a Pydantic model. In exchange, `FabricatedValue` and `ProhibitedLabel` are
raised by the same objects no matter which transport is calling.

C6 has already earned its place: it broke when the inbound adapters imported the
composition root for the `Container` type, making every transport transitively
depend on every outbound adapter. The fix — a driving port
(`application.api.EngineApi`) that `Container` satisfies structurally — is a
better design, and nothing but the contract would have found it.

## What would make us revisit this

Nothing short of abandoning the independent-service decision in ADR-001.
