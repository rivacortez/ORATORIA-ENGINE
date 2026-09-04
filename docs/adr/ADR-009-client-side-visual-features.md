# ADR-009 — Client-side visual feature extraction as a first-class path

**Status:** Accepted · **Date:** 2026-09-04

## Context

The visual pipeline needs landmark geometry. It can get it two ways: receive
frames and extract server-side, or receive geometry a client already extracted.

Server-side is simpler to reason about — one implementation, one version to
promote, no trust placed in the client's extractor. It also means every frame of
a student's face crosses the network and lands, however briefly, in the
service's memory and storage.

Client-side extraction sends a few dozen floats per frame instead. No image ever
reaches the service, so there is nothing to leak, nothing to retain by accident
and nothing to delete.

## Decision

Both are supported, and `LANDMARKS` is a **first-class negotiated format**
(FR-006) rather than a degraded fallback. The pipeline downstream consumes
geometry either way, so nothing after the ingestion boundary knows which
arrived.

## Consequences

Two extraction paths whose outputs must agree, which is a real evaluation
burden: a landmark series from a browser and one from the server model are not
automatically comparable, and the subgroup audit in §13 Phase 5 has to cover
both.

The client's extractor version becomes part of provenance when that path is
used. A client that ships a new extractor changes the evidence, and NFR-014
requires that to be visible.

Against that: the strongest available privacy posture is reachable without
giving up the visual modality, and a deployment that cannot accept raw frames at
all becomes a supported configuration rather than a special case.

## What would make us revisit this

Measured disagreement between the two paths large enough that results from one
cannot be compared with the other. That would force a choice rather than a
change of mind.
