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
  any mental-health condition. Not "not yet" — no instrument here measures any
  of them, so a label would be an invention.
- It does not clinically diagnose stuttering or any speech disorder.
- It does not rank findings or select a top-k. `ranking_authority` is the
  `ClassVar` constant `"none"`, and the assembled payload is walked against a
  ranking denylist before publication (FR-029).
- It does not claim exact timestamps. Every interval carries a declared
  tolerance (NFR-004).
- It does not turn missing evidence into a number. An unavailable indicator is
  a separate type with no numeric attribute at all — reading one raises rather
  than defaulting to zero (FR-025).

## Where it stands

| Phase (§13) | Deliverable                                              | State                                                          |
| ----------- | -------------------------------------------------------- | -------------------------------------------------------------- |
| 0           | Scope, taxonomy, annotation manual, consent policy, ADRs | **done**                                                       |
| 1           | Peruvian Spanish corpus and annotation platform          | not started — field work                                       |
| 2           | Platform skeleton and contracts                          | **done, exit criterion proven**                                |
| 3           | Verbatim speech baseline                                 | not started                                                    |
| 4           | Disfluency and prosody intelligence                      | not started                                                    |
| 5           | Visual evidence                                          | not started                                                    |
| 6           | Multimodal fusion                                        | deterministic correlation done; needs a corpus to evaluate     |
| 7           | Real-time hardening                                      | bounded queues and backpressure done; load testing not started |
| 8           | OratorIA integration                                     | not started                                                    |
| 9           | Scientific validation                                    | not started                                                    |
| 10          | Production readiness                                     | not started                                                    |

Phase 2's exit criterion — _"a synthetic session can be streamed, completed,
queried and deleted without model inference"_ — is executed rather than
asserted, on **both** backends:

- `tests/contract/test_phase2_exit_criterion.py` — memory backend, runs on
  every commit with no infrastructure.
- `tests/integration/test_phase2_on_postgres.py` — PostgreSQL, Redis and
  MinIO.

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
├── application/       api (driving port), ports, commands, queries,
│                      workflows, services
├── adapters/
│   ├── inbound/         rest, websocket, workers
│   └── outbound/        persistence (memory + postgres), object_storage
│                        (memory + s3), cache (memory + redis),
│                        model_runtime, telemetry
└── bootstrap/         composition root, settings, ASGI factory
```

`.importlinter` holds six contracts. Each guards a claim the spec makes:

| Contract                              | Guards                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------ |
| C1 dependency rule                    | §4.3 `Adapters -> Application -> Domain`                                             |
| C2 domain is pure                     | §4.3 — no FastAPI, SQLAlchemy, Redis, PyTorch, MLflow, vendor SDKs (and no Pydantic) |
| C3 application is infrastructure-free | NFR-016 — runtimes replaceable behind ports                                          |
| C4 shared kernel knows nobody         | keeps C5 from becoming decorative                                                    |
| C5 modalities are independent         | QA-02 — video can die without taking speech with it                                  |
| C6 inbound does not touch outbound    | keeps quotas, idempotency and the state machine non-optional                         |

C6 has already earned its place: it broke when the inbound adapters imported
the composition root, making every transport transitively depend on every
outbound adapter. The fix — a driving port (`application.api.EngineApi`) that
`Container` satisfies structurally — is a better design, and nothing but the
contract would have found it.

## Invariants enforced in code, not prose

| Rule                                                   | Mechanism                                                                                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| FR-025 — unavailable is never zero                     | `Unavailable` has no `value` attribute; reading one raises `FabricatedValue`. In DDL: `value XOR reason` check constraints.                      |
| FR-028 — no causal inference                           | `causal_inference` is a `ClassVar`, so no constructor, deserializer or test can set it. No column exists.                                        |
| FR-029 — no ranking                                    | `ranking_authority` is a `ClassVar` equal to `"none"`; the payload is walked against a ranking denylist before publication.                      |
| NFR-024 — no emotion or clinical labels                | The taxonomy is an allowlist; a reflection test scans every enum member, dataclass field and public class in the domain for prohibited concepts. |
| FR-017 — raw text kept whatever the role               | Required on lexical classes regardless of `context_role`, so NFR-003's precision denominator survives.                                           |
| FR-013 — undecidable is `uncertain`                    | The assembler resolves a missing role to `UNCERTAIN`; it never guesses.                                                                          |
| FR-009 — a pause neither resets nor inflates the clock | `SessionClock` accumulates captured time only.                                                                                                   |
| §6.1 step 9 — finalized history is never rewritten     | The transcript and the evidence ledger both refuse it.                                                                                           |
| NFR-013 — tenants are invisible to each other          | Every repository read takes a tenant in its signature; another tenant's session is _not found_, never _forbidden_.                               |
| §14.2 — uncalibrated scores open no gates              | `Confidence.meets` refuses an uncalibrated value, so FR-022's precision gate cannot be opened by an unevaluated detector.                        |

## Quick start

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env    # fill in the two secrets

# the battery
uv run ruff check src tests scripts
uv run ruff format --check src tests
uv run mypy
uv run lint-imports
uv run pytest
```

With infrastructure, for the persistent backend:

```bash
docker compose up -d
ENGINE_DATABASE_URL=postgresql+asyncpg://engine:engine@localhost:5432/engine \
  uv run alembic upgrade head
uv run pytest -m integration
```

Generated files — regenerate after changing their source:

```bash
uv run python scripts/render_taxonomy.py            # docs/taxonomy/ANNOTATION_MANUAL.md
uv run python scripts/render_initial_migration.py   # alembic/versions/0001_initial_schema.py
```

Both are guarded by tests that fail when the checked-in file is stale.

## Numbers

**No accuracy, latency or agreement figure appears in this repository.** The
targets in §10 are _acceptance targets to calibrate against_, not results.
Until Phase 1 freezes a held-out corpus there is nothing to report, and the
honest value is absent rather than estimated —
see `docs/governance/BASELINES.md`, whose Results section is deliberately empty.

The default configuration ships with empty threshold maps for the same reason:
with no fitted calibration curve, `Confidence.meets` refuses every gate, so
nothing is published as _confirmed_ on the strength of a number nobody
measured.

## CI and branch protection

Two jobs, both required conceptually and neither enforced server-side yet:

- `battery` — ruff, mypy `--strict`, six import contracts, the full suite with
  warnings as errors, 88% branch-coverage floor on domain and application
- `integration` — migration round-trip plus the 16 tests against PostgreSQL,
  Redis and MinIO

Branch protection on a **private** repository requires GitHub Pro; both the
rulesets and the classic API return 403 on this account. `./scripts/install-hooks.sh`
installs a local pre-push stand-in. The gap and the three ways to close it are
written down in `docs/governance/BRANCH_PROTECTION.md` rather than left implicit.

## Documents

- `docs/taxonomy/ANNOTATION_MANUAL.md` — generated from the frozen taxonomy
- `docs/governance/SCOPE.md` — scope, non-goals and the separation of authority
- `docs/governance/CONSENT_AND_RETENTION.md` — the policy the code enforces,
  with the participant-facing text
- `docs/governance/BASELINES.md` — frozen baselines and the evaluation protocol
- `docs/adr/` — ADR-001 .. ADR-010
- `docs/evidence/` — raw battery output from runs against real infrastructure
- `docs/governance/BRANCH_PROTECTION.md` — what is and is not enforced
