# ADR-003 — Streaming ASR with an auxiliary alignment objective

**Status:** Accepted, implementation deferred to Phase 3 · **Date:** 2026-09-04

## Context

Driver 1 is verbatim fidelity: "losing a disfluency prevents subsequent
measurement and feedback." Every commercial ASR is tuned in the opposite
direction, because its customers want readable output. A filled pause the
decoder deletes cannot be recovered by anything downstream — not by a better
classifier, not by a larger model. It is gone.

Two families were considered.

**A Whisper-style encoder-decoder.** Excellent recognition, widely available,
trained on transcripts that were themselves cleaned. Its decoder carries a
language-model prior that actively smooths disfluencies, and its timestamps are
segment-level by default, which puts NFR-004's ≤250 ms boundary target out of
reach without a separate forced-alignment pass.

**A streaming transducer (FastConformer/RNNT-shaped) with an auxiliary CTC
alignment head.** Streams natively, which §11.3 iteration 3 needs. The CTC
objective yields frame-level alignment as a by-product rather than as a second
pass, and a transducer's weaker language-model prior deletes fewer
disfluencies.

## Decision

Target the streaming transducer with an auxiliary CTC alignment objective for
the project model. Keep Whisper and CrisperWhisper as **frozen research
baselines** (`docs/governance/BASELINES.md`), never as an undocumented
production dependency — §11.1 names that constraint explicitly, and a baseline
adapter that stays wired "temporarily" is exactly how it gets violated.

Implementation is deferred to Phase 3. Nothing is trained before Phase 1 freezes
a corpus: a model trained on data whose held-out set is not yet separated cannot
produce a reportable number.

## Consequences

Higher training cost and a harder engineering path than fine-tuning a decoder.
Requires the annotated corpus to exist first, which is the real critical path
(§18).

The port boundary makes this reversible. `SpeechRuntime` returns hypotheses, not
events, so falling back to a managed baseline changes an adapter and nothing
else. That is the point of NFR-016, and it is worth more here than anywhere
else in the system — this is the decision most likely to be wrong.

## What would make us revisit this

Measured evidence that a managed verbatim-capable ASR retains P0 classes at a
rate the corpus cannot improve on, or a corpus too small to train a transducer
without overfitting. Either would make the honest move a documented baseline
plus a disfluency layer on top — which is where the contribution lives anyway.
