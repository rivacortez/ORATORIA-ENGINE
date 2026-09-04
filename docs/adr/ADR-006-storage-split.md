# ADR-006 — PostgreSQL for metadata, Redis for ephemeral state, object storage for media

**Status:** Accepted · **Date:** 2026-09-04

## Context

Three kinds of data with genuinely different lifetimes and access patterns:

- **Metadata and evidence.** Relational, queried by tenant, must survive
  restarts, must be transactionally consistent with the session state machine.
- **Live stream state.** The chunk ledger, the finalized frontier, the
  processing lease. Written many times per second, meaningless once the session
  ends, and required to expire on its own so a dead handler does not lock a
  session out forever.
- **Protected media.** Large, write-once, read-rarely, encrypted, deleted on a
  deadline.

Putting the second in PostgreSQL would place a transactional database on the hot
path for data that expires with the session. Putting the third there would be
worse.

## Decision

PostgreSQL for metadata and evidence. Redis for ephemeral state, leases and
quota windows. S3-compatible object storage (MinIO locally) for protected media
and model artifacts.

Every one sits behind a port, and an in-memory implementation of each is a
first-class citizen — Phase 2's exit criterion is defined without infrastructure
precisely so the contract suite runs on every commit rather than only where the
whole stack happens to be up.

## Consequences

Three systems to operate rather than one. The in-memory adapters must model the
behaviours that matter — expiring leases, rolling quota windows, expiry enforced
on read — or a bug passes CI and fails in production. They do, and their
docstrings say why.

Redis is a cache, so anything in it must be reconstructible. Nothing a result
depends on lives there.

## What would make us revisit this

Measured contention that PostgreSQL handles fine at pilot scale, which would
make Redis one system too many. Or the opposite: stream state outgrowing a
single Redis, which would mean sharding by session.
