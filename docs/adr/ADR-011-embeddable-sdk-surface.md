# ADR-011 — The engine ships as an embeddable SDK with an optional hosted service

**Status:** accepted, 2026-09-05
**Supersedes nothing. Constrains:** ADR-001 (independent engine), ADR-003 (ASR strategy).

## Context

ADR-001 separated this engine from the `ORATORIA-PROYECTO` monolith so it could
be consumed by more than one application. At the level a consumer actually
meets it, that separation was not real: the only way to run the engine was to
deploy a service, and `pip install oratoria-evidence-engine` installed FastAPI,
uvicorn, SQLAlchemy, asyncpg, Alembic, Redis, aioboto3 and OpenTelemetry.

Three things follow from that, and each is a cost this project was paying.

**A thesis pays for a GPU it does not need to rent.** The only component that
needs a GPU is one adapter. Requiring a deployment to run it meant either a
rented GPU on an hourly bill or nothing.

**Another researcher cannot reproduce an experiment** without standing up
PostgreSQL, Redis and an object store first.

**The batch path does not exist.** `mode: "batch"` is accepted by the API and
`adapters/inbound/workers/` is an empty package. The thing a researcher wants -
"analyse this file" - had no implementation anywhere.

## Decision

Ship the engine as **one repository and one Python wheel** whose core installs
with no dependencies, plus a public SDK facade, plus the hosted service as an
extra.

    pip install oratoria-evidence-engine              core and contracts
    pip install "oratoria-evidence-engine[local]"     local inference
    pip install "oratoria-evidence-engine[server]"    the hosted API
    pip install "oratoria-evidence-engine[record]"    microphone capture
    pip install "oratoria-evidence-engine[research]"  corpus and evaluation

The published surface, exported from `evidence_engine` and nowhere else:

| Exported                                            | Why it is public                         |
| --------------------------------------------------- | ---------------------------------------- |
| `OratoriaEngine`                                    | the facade                                |
| `EngineConfiguration`, `SessionConfiguration`       | what a consumer supplies                  |
| `AnalysisResult`, `HardwareReport`, `StreamSession` | what a consumer reads                     |
| `OratoriaError` and its three subclasses            | what a consumer catches                   |
| `__version__`                                       | what a consumer names in a bug report     |

Nothing else. Not `Container`, not a repository, not an adapter, not a concrete
runtime, not a domain entity. Those are how the engine is assembled this month,
and a consumer coupled to one would break on the next release — the coupling
ADR-001 separated this engine to avoid.

```python
engine = OratoriaEngine.local(EngineConfiguration(runtime="baseline_whisper"))

print(engine.hardware_preflight().render())   # what the machine has
await engine.warmup()                          # whether the model runs

result = await engine.analyze_file("presentation.wav")

async with engine.create_stream() as stream:
    await stream.send_audio(chunk)
    event = await stream.receive()
    result = await stream.finish()

await engine.aclose()
```

## Why these shapes and not the obvious ones

**One repository, not four.** Splitting core, SDK and server across
repositories is the wrong mechanism for a correct separation. The boundary
between layers is checked by a machine — eight import-linter contracts that
fail CI — and a second repository replaces that with a version constraint
nobody breaks the build over. It also buys four pipelines and turns a refactor
across the boundary into two pull requests and an intermediate release.

**The facade composes use cases; it does not call a runtime.** The short
version of `analyze_file` would hand the audio to `WhisperSpeechRuntime` and
return words. That is a second functional path, and the state machine, the
consent check, the quota, the evidence ledger, the co-occurrence window and the
calibration gate would each have to be re-implemented on it or silently
skipped. A rule enforced on one path and absent from the other is worse than a
rule enforced nowhere, because the two paths produce results that look alike.
So the SDK opens a session, opens a run, drives `StreamingCoordinator` and
completes the session — the same sequence the WebSocket handler drives.

**`hardware_preflight()` and `warmup()` are two methods, not one.** The
preflight inspects the machine in milliseconds: torch, CUDA, memory, compute
capability, libraries. None of that is evidence that inference succeeds.
`warmup()` loads the weights and decodes a window, and it is the only thing
that can say the model runs here. One `check()` returning `True` would be a
green result standing in for a measurement nobody took — on a machine with
corrupt weights or a driver too old for the compiled kernel it would still be
green, and a consumer would stop looking. The report ends by saying what it did
not check.

**`create_stream()` returns a session, not a generator.** A generator can be
advanced and closed. A presentation has to be paused when the speaker stops to
answer a question, resumed, finished, and abandoned when something fails.
`pause` is load-bearing rather than decorative: §6.1 excludes paused stretches
from the session clock, so a rate computed across one would be diluted by
however long the speaker was not speaking.

**No API key on the embedded path.** Authentication answers "may this request
reach the engine", and an embedded engine is already inside the consumer's
process — a credential would be the process authenticating to itself. The
process boundary is the authorization boundary. Requests run as tenant
`embedded`, application `oratoria-engine-sdk`, so a locally produced document
is identifiable in a trail that also holds hosted runs.

**Async-first, with no synchronous wrapper.** The ports are async because the
streaming path is. A sync facade would either block an event loop or own one,
and choosing which is a decision for whoever has a real use for it.

## What this is checked by

- **C1** places `evidence_engine.sdk` below `evidence_engine.bootstrap`, so the
  facade importing the composition root fails CI rather than review. That
  single import would drag the `server` extra into every embedded install.
- **C9** builds the wheel, reads `METADATA`, installs the base wheel in an empty
  virtual environment and asserts no server or ML module is loaded.
- **`tests/contract/test_sdk_surface.py`** asserts `__all__` by name, asserts
  thirteen internals are *not* reachable from the package root, and drives a
  conformance comparison: two engines on the same audio under the same
  configuration must produce the same evidence, excluding the identifiers and
  stamps that differ by construction.

## Consequences

**Accepted.** An embedded consumer takes on responsibilities a service consumer
does not: a compatible GPU, a correct CUDA install, several gigabytes of
weights, model updates and memory pressure. Python also limits direct
integration from other languages. This is why the hosted service is not
removed — it is an alternative way to run the same engine, for consumers who
cannot or should not take those on.

**Deliberately deferred.** A TypeScript SDK for browser capture and remote
consumption will be a separate npm distribution, living in this repository
under `sdk/typescript/`. ~~A remote client (`OratoriaClient`) that speaks to
the hosted service and returns the same contracts is not built; when it is,
the conformance comparison above is the test it has to pass.~~ **Built - see
the amendment below.**

**Unchanged.** The scientific contribution is still the contextual and
multimodal model, and this SDK does not create a detector or improve accuracy.
It is the mechanism by which that model, once it exists, is integrated and
reused.

## Amendment (2026-09-06) — `OratoriaClient` exists

The remote client this ADR deferred is built: `evidence_engine.sdk.client
.OratoriaClient`, exported from the package root, satisfying the same two
structural protocols as the embedded facade - `warmup`, `create_stream`,
`aclose`, `contributions` on the engine side; `send_audio`, `receive`,
`pending`, `finish`, `abort` on the session side. The pilot condition that
motivated it: the engine runs as a private service on a GPU workstation,
OratorIA's backend runs on CPU infrastructure and reaches it over HTTP and
WebSocket, and the adapter OratorIA already wrote against the embedded engine
drives this one unchanged.

**The conformance test named above is now a real test.**
`tests/contract/test_remote_client.py::test_evidence_from_json_matches_the_
embedded_translation` streams one scripted session through the hosted app
and through the embedded engine and asserts `sdk.results.evidence_from_json`
and `sdk.results.evidence_from` agree field for field, excluding the
identifiers and stamps that differ by construction. It passes.

**What closing this gap needed on the server side**, recorded in full in
`docs/BUILD_RECORD.md` §3.21 and its own sub-decisions:

- Backpressure that does not lose a chunk: the admission check in
  `StreamingCoordinator._admit` now runs *before* the chunk is marked seen,
  and `backpressure.requested` names the refused `chunk_seq` so a client
  knows exactly what to resend.
- `session.completed` states `finalized_through_ms` and `captured_ms`, so a
  client can tell how far a run got without holding the socket open for the
  whole session.
- A client-initiated `session.abort`, answered with `session.aborted`,
  distinct from a dropped connection.
- A startup warm-up decode and a `checks["speech:warm"]` readiness gate, plus
  an `instance_id` on `session.accepted` and an `instance`/`models` block on
  `/v1/capabilities` - a pilot can run more than one GPU workstation behind
  the same consuming application and needs to tell them apart.

**What is still not built.** No positive per-chunk acknowledgement exists on
the wire, so a refusal for the very last chunk of a session can still arrive
after the client has moved on to `finish()` and be missed - a residual risk
named in `sdk/client.py`'s module docstring and in BUILD_RECORD §3.21, not
one this amendment claims to have closed. `Evidence` carries no
`cooccurrences` and no `quality`; the wire renders both and neither is
translated, because inventing a public shape for two fields nothing here
reads was not what this change was asked to do.
