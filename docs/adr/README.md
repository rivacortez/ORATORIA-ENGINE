# Architectural decision records

One file per decision, numbered, immutable once accepted. A decision that turns
out to be wrong is **superseded by a new record**, never edited — the point of
these files is to say what was known and believed at the time, and rewriting
that destroys the only evidence of why the system looks the way it does.

Each record answers four questions: what forced a decision, what was decided,
what it costs, and what would make us revisit it. The fourth is the one most
often left out and the one that matters most later.

| ADR                                           | Decision                                                                     | Status                             |
| --------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------- |
| [001](ADR-001-independent-engine.md)          | Independent engine rather than models embedded in the consumer               | Accepted                           |
| [002](ADR-002-hexagonal-architecture.md)      | Hexagonal architecture with a machine-checked dependency rule                | Accepted                           |
| [003](ADR-003-streaming-asr-strategy.md)      | Streaming ASR with an auxiliary alignment objective vs. a Whisper decoder    | Accepted (deferred implementation) |
| [004](ADR-004-explicit-verbatim-taxonomy.md)  | An explicit verbatim taxonomy as a code allowlist                            | Accepted                           |
| [005](ADR-005-websocket-for-mvp.md)           | WebSocket for the MVP; WebRTC only on measured need                          | Accepted                           |
| [006](ADR-006-storage-split.md)               | PostgreSQL for metadata, Redis for ephemeral state, object storage for media | Accepted                           |
| [007](ADR-007-deterministic-fusion.md)        | Deterministic temporal fusion, and no ranking in the engine                  | Accepted                           |
| [008](ADR-008-ephemeral-media.md)             | Ephemeral raw-media retention by default                                     | Accepted                           |
| [009](ADR-009-client-side-visual-features.md) | Client-side visual feature extraction as a first-class path                  | Accepted                           |
| [010](ADR-010-model-promotion.md)             | Model promotion, rollback and reproducibility policy                         | Accepted                           |
