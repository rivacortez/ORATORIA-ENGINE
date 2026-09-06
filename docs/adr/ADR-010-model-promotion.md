# ADR-010 — Model promotion, rollback and reproducibility policy

**Status:** Accepted · **Date:** 2026-09-04

## Context

Models will change during the project. §17 names the risk plainly: "model
updates break longitudinal comparison" — a student's metric moves and nobody can
tell whether they improved or a threshold was retuned.

QA-03 wants a new classifier canaried without touching a schema, and rolled back
in under ten minutes. §14.2 gates promotion behind evaluation.

## Decision

**Provenance lives on the evidence, not on the deployment.** Every event carries
its model version, taxonomy version, configuration snapshot and evidence
reference. During a canary two versions answer at once, so reading the version
from deployment config would be wrong for exactly the traffic that matters most.

**Amended 2026-09-05 - the unit of attribution is the component.** Provenance
names a `ModelRole` (recogniser, disfluency detector, context classifier,
prosody estimator, visual estimator); the manifest maps role to version; the
registry activates, promotes and rolls back **per role**. Keyed by modality, as
it was, "canary a classifier" and "replace the recogniser" were the same
operation and a manifest could hold one audio model - the scenario above could
not be recorded, let alone rolled back. Two versions of one role inside a
single run is refused, not recorded. See `docs/BUILD_RECORD.md` §3.20.

**Only evaluated artifacts are promotable.** The registry refuses anything not
in `EVALUATED` or `APPROVED`.

**The rollback target is recorded at promotion**, not chosen during an incident.
That is what makes QA-03's ten-minute rollback achievable: the decision is
already made when the pager goes off.

**Configuration is frozen per session.** A snapshot is bound at creation and
does not move underneath a presentation. Updating a default never alters a
historical result (US-008).

**Calibration gates thresholds.** An uncalibrated score never clears a
publication threshold — `Confidence.meets` refuses it. A raw softmax maximum is
not a probability of correctness, and thresholding it as one turns a
precision-focused gate (FR-022) into theatre.

## Consequences

More metadata on every event, and a registry that says no more often than an
engineer mid-experiment would like.

A newly added detector class publishes nothing as _confirmed_ until a
calibration curve exists for it. It can still report findings, as uncertain.
That is the honest position for a class nobody has evaluated, and it is what the
empty threshold maps in the default configuration mean today.

## What would make us revisit this

Nothing structural. The thresholds and curves themselves are expected to change
constantly — that is why they are versioned data rather than decisions.
