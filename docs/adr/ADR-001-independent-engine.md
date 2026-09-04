# ADR-001 — An independent engine rather than models embedded in the consumer

**Status:** Accepted · **Date:** 2026-09-04

## Context

The obvious alternative is to put the speech and vision models directly inside
the consuming application. Fewer moving parts, one deployment, no network hop,
no API contract to version.

Three forces push the other way.

_Different scaling shapes._ Inference wants GPUs and scales with concurrent
minutes of audio; a web application scales with page views. Coupling them means
provisioning one for the other's peak.

_Different release cadences._ §11.1 lists modifiability as a driver because
models and taxonomies will evolve during research. A model swap should not be a
release of the student-facing product, and a UI fix should not redeploy a GPU
image.

_A second consumer is plausible._ §11.1 lists interoperability: "the engine must
serve products other than OratorIA." Extracting a service later, from code that
assumed one caller, is dramatically more expensive than starting with a
boundary.

## Decision

Build the engine as an independent, API-first service with a versioned
REST/WebSocket contract, consumed through API keys. It knows nothing about the
consumer's entities.

## Consequences

**Costs, accepted.** A contract to version (NFR-017). A network hop inside the
latency budget of NFR-005. An authentication and quota surface that contributes
nothing to the scientific question. Two deployments to operate.

**Gains.** The consumer can be replaced without touching the engine's domain,
and the engine without touching the consumer's. The separation of authority in
§3 driver 8 becomes structural rather than a matter of discipline: a service
that has never heard of the consumer's ranking rules cannot accidentally apply
them.

## What would make us revisit this

A year with exactly one consumer, no independent scaling pressure, and a
measured latency budget that the network hop is actually breaking. None of
those alone; all three together.
