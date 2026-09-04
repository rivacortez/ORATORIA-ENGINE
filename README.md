# OratorIA Multimodal Evidence Engine

An independent, API-first service that captures audio and video from a
single-speaker oral presentation and produces a **verbatim transcript** plus
**auditable, timestamped evidence** about speech disfluencies, vocal delivery
and observable visual behaviour.

The engine observes. It does not rank, diagnose or recommend.

> Implements `ORATORIA_MULTIMODAL_ENGINE_IMPLEMENTATION_SPEC.md` v1.0.
> Section references throughout the code (`§4.3`, `FR-025`, `NFR-014`, `QA-02`)
> point at that document. Where a docstring and the spec disagree, the spec
> wins and the docstring is a bug.

---

## What it is not

Stated up front because most of the design exists to keep these out:

- It does not diagnose anxiety, stress, insecurity, deception, personality or
  any mental-health condition.
- It does not clinically diagnose stuttering or any speech disorder.
- It does not rank findings or select a top-k. `ranking_authority` is the
  constant `"none"` and the result schema is checked against a ranking
  denylist before publication (FR-029).
- It does not claim exact timestamps. Every interval carries a tolerance
  (NFR-004).
- It does not turn missing evidence into a number. An unavailable indicator has
  a reason and no numeric attribute at all (FR-025).

## Architecture

Clean/Hexagonal. The dependency rule is machine-checked, not documented:

```
bootstrap  ->  adapters  ->  application  ->  domain
```

```
src/evidence_engine/
├── domain/            pure model, zero framework imports
│   ├── shared/          identifiers, timeline, confidence, measurement,
│   │                    provenance, taxonomy (the published allowlist)
│   ├── sessions/        state machine, session clock, consent, capabilities
│   ├── transcript/      word tokens, partial/final reconciliation
│   ├── speech_events/   typed disfluency events, prosodic readings
│   ├── visual_events/   observable visual events, per-speaker calibration
│   ├── quality/         quality gates, per-indicator availability
│   └── evidence/        ledger, temporal co-occurrence, result document
├── application/       ports, commands, queries, workflows
├── adapters/
│   ├── inbound/         rest, websocket, workers
│   └── outbound/        persistence, object_storage, cache,
│                        model_runtime, telemetry
└── bootstrap/         composition root
```

`.importlinter` holds six contracts. Each guards a claim the spec makes:

| Contract                              | Guards                                                             |
| ------------------------------------- | ------------------------------------------------------------------ |
| C1 dependency rule                    | §4.3 `Adapters -> Application -> Domain`                           |
| C2 domain is pure                     | §4.3 - no FastAPI, SQLAlchemy, Redis, PyTorch, MLflow, vendor SDKs |
| C3 application is infrastructure-free | NFR-016 - runtimes replaceable behind ports                        |
| C4 shared kernel knows nobody         | keeps C5 from becoming decorative                                  |
| C5 modalities are independent         | QA-02 - video can die without taking speech with it                |
| C6 inbound does not touch outbound    | keeps quotas, idempotency and the state machine non-optional       |

## Quick start

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# the battery
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run lint-imports
uv run pytest
```

Local infrastructure (PostgreSQL, Redis, MinIO):

```bash
docker compose up -d
```

## Delivery phases

Per §13. Current position is marked.

| Phase | Deliverable                                              | State       |
| ----- | -------------------------------------------------------- | ----------- |
| 0     | Scope, taxonomy, annotation manual, consent policy, ADRs | in progress |
| 1     | Peruvian Spanish corpus and annotation platform          | not started |
| 2     | Platform skeleton and contracts                          | in progress |
| 3     | Verbatim speech baseline                                 | not started |
| 4     | Disfluency and prosody intelligence                      | not started |
| 5     | Visual evidence                                          | not started |
| 6     | Multimodal fusion                                        | not started |
| 7     | Real-time hardening                                      | not started |
| 8     | OratorIA integration                                     | not started |
| 9     | Scientific validation                                    | not started |
| 10    | Production readiness                                     | not started |

Phase 2 closes when a synthetic session can be streamed, completed, queried and
deleted **without model inference**. That is what the contract test suite
proves; it is not a claim made in prose.

## Numbers

No accuracy, latency or agreement figure appears in this repository unless it
was produced by this engine's own benchmark runner over the project corpus. The
targets in the spec's §10 are _acceptance targets to calibrate against_, not
results. Until Phase 1 freezes a held-out set, there is nothing to report and
the honest value is absent, not estimated.

## Documents

- `docs/taxonomy/` - the frozen event taxonomy and the annotation manual
- `docs/adr/` - architectural decision records (ADR-001 .. ADR-010)
- `docs/governance/` - consent, retention and deletion policy
