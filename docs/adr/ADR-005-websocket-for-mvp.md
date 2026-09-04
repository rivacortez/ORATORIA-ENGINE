# ADR-005 — WebSocket for the MVP; WebRTC only on measured need

**Status:** Accepted · **Date:** 2026-09-04

## Context

WebRTC is the obvious transport for real-time media: built for it, adaptive
bitrate, NAT traversal, jitter buffering. It also brings signalling servers,
TURN infrastructure, SDP negotiation and a debugging surface substantially
harder than "read a JSON frame".

The engine's actual requirement (NFR-005) is a p95 partial-transcript latency of
1.5 seconds. WebRTC's advantages are measured in tens of milliseconds of jitter
— real, and irrelevant at this budget.

## Decision

WebSocket for the MVP. Audio arrives base64-encoded inside the JSON envelope
§7.4 requires on every message.

Base64 costs about a third in bandwidth and some parse time. A binary frame with
a packed header would be smaller and cheaper, and it is the right optimization
for Phase 7 — but it is an optimization, and doing it now would mean two parsing
paths to keep in sync while the contract is still settling. A binary frame also
has nowhere to put `schema_version`, `message_id` and `chunk_seq` without
inventing a second, parallel encoding of the envelope.

## Consequences

One transport, one parser, one place where §7.4's contract rules are enforced.
Reconnection is the client's problem, which is acceptable because the chunk
ledger makes redelivery idempotent.

Bandwidth is roughly a third higher than it needs to be. On a 16 kHz PCM16
stream that is a known quantity, not a surprise.

## What would make us revisit this

A measured p95 that misses 1.5 s _because of transport_, on the pilot hardware
under the documented reference load. Not a suspicion, and not a bandwidth bill.
