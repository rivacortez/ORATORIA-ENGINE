# Scope and non-goals

**Status:** approved for Phase 0 · **Version:** 1.0.0

This document exists so that scope arguments have somewhere to be settled. When
a request arrives that is not covered here, the answer is not "no" — it is "add
it here first, and say what it displaces."

---

## What the engine is

An independent, API-first service that captures audio and video from a
**single-speaker** oral presentation and produces:

- a **verbatim transcript** that preserves disfluencies rather than tidying
  them away;
- **typed, timestamped evidence** about speech disfluencies, vocal delivery and
  observable visual behaviour, each with a confidence, an interval, a tolerance
  and full provenance;
- **deterministic temporal co-occurrences** between modalities;
- **quality and availability** for every modality and indicator, including an
  explicit reason wherever a value could not be produced.

## What the engine is not

Each of these is out of scope for a reason, and the reason is not "later".

| Not in scope                                                     | Why not                                                                                                                           |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Diagnosing anxiety, stress, insecurity, deception or personality | No instrument here measures any of them. A label would be an invention, not a limitation to lift later.                           |
| Clinical diagnosis of stuttering or any speech disorder          | Same. Additionally, a disfluency rate is not a diagnosis and presenting it as one would be harmful.                               |
| "100% exact" transcription or millisecond-perfect alignment      | Not achievable, and claiming it would make every finding unauditable. Intervals carry a declared tolerance instead.               |
| Ranking findings or selecting a top-k                            | The consuming application's decision engine ranks. `ranking_authority` is the constant `"none"`.                                  |
| Generating exercises or recommendations                          | Same separation of authority. The engine observes.                                                                                |
| Replacing human assessment in high-stakes decisions              | The evidence supports a judgement; it does not make one.                                                                          |
| Multiple simultaneous speakers                                   | Version 1 is single-speaker. Diarization changes every interval assumption in the pipeline.                                       |
| Speaker diarization, emotion recognition, identity recognition   | The first is deferred; the second and third are prohibited.                                                                       |
| Permanent storage of raw audio or video by default               | Retention is ephemeral unless an explicit versioned policy says otherwise.                                                        |
| Training a foundation speech model from zero                     | Out of reach and unnecessary. The contribution is the disfluency layer and the evidence contract on top of a documented baseline. |

## Actors

| Actor                      | Type                                 | Responsibility                                                                                          |
| -------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Student                    | Human, via the consuming application | Gives a presentation and reads the feedback the consumer produces. Never talks to this engine directly. |
| System administrator       | Human                                | Manages applications, keys, models, thresholds, retention and operational health.                       |
| Consuming application      | System                               | Integrates over REST and WebSocket. **Not an additional human role.**                                   |
| Consumer's decision engine | External                             | Ranks the complete findings. The only ranking authority.                                                |

## The separation of authority

Stated once, because most of the architecture follows from it:

> **Detectors observe. The consumer's engine ranks. A language model explains.**

The engine occupies the first position only. It never occupies the second, and
it has no language model at all.

## Version 1 constraints

- **Locale:** `es-PE` only. Other locales are refused rather than served by a
  model whose error rates on them are unknown.
- **Speakers:** one.
- **Deployment:** a modular monolith. Services split into separate processes
  only when independent scaling, security isolation or failure containment is
  demonstrated _by measurement_ — not by architecture diagram.

## Changing this document

Scope changes are versioned. Adding an item to "what the engine is" requires
naming what it displaces from the current phase, because the delivery order in
the specification is a critical path and not a menu.
