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

## Hearing what the engine hears

One command, once the extras are installed. It records a presentation, runs the
pinned checkpoint over it, and prints what the engine derived.

```bash
uv sync --extra managed --extra record

# torch is installed separately, from the wheel index your card needs. This
# workstation is a Blackwell (sm_120) and needs cu130; check yours before
# copying the line.
uv pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
uv run python -c "import torch; print(torch.cuda.get_arch_list())"   # your arch must be listed

uv run python scripts/present.py devices                        # pick a physical mic
uv run python scripts/present.py record --seconds 90 --device 1 -o exposicion.wav
uv run python scripts/present.py run exposicion.wav
```

The first run downloads about 3 GB of weights, at the revision
`BASELINE_PINS.md` pins. Later runs load in eight seconds.

**`devices` flags the virtual inputs.** NVIDIA Broadcast and Voicemeeter
enumerate as microphones on this machine, and Broadcast's noise removal
suppresses exactly the breath and creak that mark `cut_off` and `prolongation`.
Recording through one would encode the enhancer's decisions as data.

### What it will and will not tell you

It gives a real transcript with real word boundaries, and the silent pauses it
derives from the gaps between them using the versioned 700 ms threshold. Silent
pause is the one taxonomy class that is *derived* rather than detected, which
is why it is the one that works today.

It reports **zero** filled pauses, false starts, repetitions, prolongations and
self-repairs — because the detector is Phase 4 and does not exist. Zero is the
honest answer, not a failure, and the report says so rather than leaving a
reader to assume a clean delivery.

And nothing it prints is a figure. The managed runtime declares
`environment_is_pinned = False`: the backend, container digest and Torch/CUDA
build freeze in Phase 3, so a number from a run today is reproducible only by
whoever ran it. Every report ends by saying that.

## Where it stands

| Phase (§13) | Deliverable                                              | State                                                          |
| ----------- | -------------------------------------------------------- | -------------------------------------------------------------- |
| 0           | Scope, taxonomy, annotation manual, consent policy, ADRs | **deliverables done; exit criterion not met** — see below                                                       |
| 0.5         | Experimental closure of the **speech** taxonomy          | **tooling done; pilots not run** — visual half deferred to Phase 5      |
| 1           | Corpus construction                                      | **tooling done; nothing recorded** — inventory, speaker-independent split, held-out freeze |
| 2           | Platform skeleton and contracts                          | **done, exit criterion proven**                                |
| 3           | Verbatim speech baseline                                 | not started                                                    |
| 4           | Disfluency and prosody intelligence                      | not started                                                    |
| 5           | Visual evidence — **and closing the visual taxonomy**    | not started                                                    |
| 6           | Multimodal fusion                                        | deterministic correlation done; needs a corpus to evaluate     |
| 7           | Real-time hardening                                      | bounded queues and backpressure done; load testing not started |
| 8           | OratorIA integration                                     | not started                                                    |
| 9           | Scientific validation                                    | not started                                                    |
| 10          | Production readiness                                     | not started                                                    |

### Phase 0 is not closed, and it has two halves

§13 separates Phase 0's *deliverables* from its *exit criterion*:

> Two annotators can apply the taxonomy consistently to a pilot sample and
> unresolved categories are documented.

The deliverables are done. The criterion needs two humans annotating a pilot
sample, and that has not happened — so **Phase 0.5** exists to close it before
any recruitment starts. Recording forty speakers and then discovering that
annotators split `false_start` from `self_repair` differently would mean a
corpus whose per-class F1 measures annotator noise, and re-annotation costs the
same as the original.

**Phase 0.5 closes the speech half only.** The pilots never show an annotator a
video frame, so the nine visual classes stay published and unvalidated. Closing
them needs its own pilot, and first a decision about whether they are events at
all — `insufficient_lighting` is a continuous condition, and the one-to-one
matching used for agreement here is the wrong instrument for it. That work sits
in Phase 5, where the corpus already exists and a taxonomy error costs
re-annotating one modality rather than re-recording everything.
`docs/corpus/PILOT_PROTOCOL.md` tracks the two halves separately, and any claim
that "the taxonomy is validated" has to say which one.

`docs/corpus/PILOT_PROTOCOL.md` has the two pilots (technical, then taxonomic),
what gets measured and why in that order. The tooling for it is built and
tested; the pilots are field work.

Two Phase 0 deliverables were also stated as policy without being instantiated,
and now have records waiting to be filled:
`docs/governance/REFERENCE_ENVIRONMENT.md` and
`docs/governance/BASELINE_PINS.md`.

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

src/corpus/           research tooling: annotation schema, ELAN import/export,
                      validation, inter-annotator agreement. A sibling of the
                      service, not a part of it - contracts C7 and C8 let it
                      share the taxonomy and nothing else.
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
| C7 corpus sees only the domain        | research tooling shares the taxonomy, never the service's transactions                |
| C8 service does not import corpus     | a deployed engine needs no annotation parser                                          |

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
- `docs/corpus/PILOT_PROTOCOL.md` — the two pilots that close the speech half
  of Phase 0, and what closing the visual half would take
- `docs/corpus/DISAGREEMENT_LOG.md` — where annotator disagreements are recorded
- `docs/adr/` — ADR-001 .. ADR-010
- `docs/evidence/` — raw battery output from runs against real infrastructure
- `docs/governance/BRANCH_PROTECTION.md` — what is and is not enforced
