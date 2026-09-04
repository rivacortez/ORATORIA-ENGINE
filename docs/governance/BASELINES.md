# Frozen baselines and the evaluation protocol

**Status:** protocol approved for Phase 0 · **No results yet — see §5**

---

## 1. Why baselines are frozen before anything is trained

NFR-001 asks the project model to outperform a frozen baseline on the held-out
corpus. That comparison is only meaningful if the baseline is chosen and pinned
_before_ anyone has seen the test results. A baseline selected afterwards is
selected — consciously or not — to be beatable.

So two things are frozen, **at two different times**, and conflating them is
a methodological error:

| Frozen | When | Why then |
|---|---|---|
| The models and their configuration | Phase 0.5 | Before anyone has seen a result the choice could be tuned against. |
| Their outputs over the held-out set | end of Phase 1 | The held-out set does not exist until then. Generating an output over data that is still moving produces a reference to nothing. |

An earlier version of this file read as though outputs could be generated now.
They cannot. The pins live in `BASELINE_PINS.md`; the outputs are generated
once the held-out set is frozen, stored as a versioned artifact alongside the
digest of the input set, and the baseline is never re-run against a newer
checkpoint of itself for the remainder of the project.

## 2. The two baselines

| Baseline                               | Role                           | Why this one                                                                                                                                                                                                         |
| -------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Whisper** (pinned in `BASELINE_PINS.md`)        | The realistic incumbent        | It is what a team would reach for, and its decoder is trained to produce _readable_ output. Measuring how much disfluency it deletes is the clearest possible statement of the problem this project exists to solve. |
| **CrisperWhisper** (pinned in `BASELINE_PINS.md`) | The strong verbatim comparator | Explicitly targets verbatim transcription and filled-pause retention. Beating a weak baseline proves nothing; this is the one that makes the result defensible.                                                      |

Both are **research comparators**. Neither becomes an undocumented production
dependency — §11.1 lists that as a constraint, and it is easy to violate by
accident: a baseline adapter that stays wired "temporarily" is how a frozen
comparator becomes the thing serving traffic.

## 3. Metrics

Reported per class, never only in aggregate. NFR-003 requires ambiguous lexical
fillers to be evaluated separately from acoustic filled pauses, because the two
fail in different ways — one confuses a sound, the other confuses a meaning —
and a single averaged F1 hides both.

| Metric                    | Applies to               | Notes                                                                                                                                          |
| ------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| WER                       | Whole transcript         | The conventional number, reported for comparability.                                                                                           |
| vWER (verbatim WER)       | Whole transcript         | Computed **without** normalizing away disfluencies. The gap between WER and vWER is the size of the problem.                                   |
| Disfluency retention rate | Per P0 class             | What fraction of annotated disfluencies survive into the transcript at all. A class the ASR deletes cannot be detected downstream by anything. |
| Precision / recall / F1   | Per P0 class             | Macro-F1 is reported but never alone.                                                                                                          |
| Contextual role accuracy  | Lexical classes only     | Confusion matrix over `filler` / `semantic` / `discourse_marker` / `uncertain`.                                                                |
| Boundary error            | Finalized events         | Median **and** p95. NFR-004 sets a ≤250 ms median target; p95 must be disclosed because a good median with a long tail is not auditable.       |
| Calibration error (ECE)   | Every class with a curve | §14.2 gates promotion on calibration being reported.                                                                                           |

### The `uncertain` class is scored, not excluded

An engine that answers `uncertain` on everything would score perfectly on
precision if uncertain predictions were dropped from the denominator. They are
not. Abstention rate is reported alongside precision, so the trade is visible.

## 4. Partitions

- **Speaker-independent.** No speaker appears in more than one partition. This
  is the single most common way a speech result becomes meaningless, and the
  check is mechanical rather than a matter of care.
- The **held-out set is frozen** at the end of Phase 1 and is not touched
  during tuning. §14.4 makes this a scientific gate.
- Dataset cards record provenance, consent basis, recording conditions and
  partition checksums.

## 5. Results

**None yet.** Phase 1 has not produced a corpus, so there is no held-out set,
no baseline output and no number to report.

This section stays empty until then, and it stays empty _visibly_. Every
numerical target in §10 of the specification is an engineering acceptance
target to calibrate against, not a result — including the ones that look like
findings:

| Target                                        | Status                                                                                                                                                                                            |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-002: macro-F1 ≥ 0.80 on P0 speech classes | **Not measured.** Aggressive for unseen speakers on a thesis-scale corpus; the spec's own fallback — a statistically justified improvement over the frozen baseline — is the honest primary goal. |
| NFR-004: median boundary error ≤ 250 ms       | **Not measured.** Achievable only if the ASR adapter emits word-level timestamps; verify that property before promising the number.                                                               |
| NFR-005: p95 partial latency ≤ 1.5 s          | **Not measured.** Requires the documented reference load on the pilot hardware.                                                                                                                   |
| NFR-006: real-time factor ≤ 1.0 per replica   | **Not measured.**                                                                                                                                                                                 |

A figure that has not been produced by this engine's own benchmark runner over
the project corpus does not go into a document. That rule has no exceptions,
and this table is what it looks like when honoured.

## 6. Reference environment

Recorded in `REFERENCE_ENVIRONMENT.md` — the record exists with its fields
empty, and every one of them has to be filled before NFR-005, NFR-006 or
NFR-007 can carry a number. Performance figures are meaningless without the
pilot hardware, concurrency level and audio characteristics they were measured
under. Numbers gathered on a free-tier host are
not comparable with numbers gathered on the pilot machine and must never be
reported as if they were.
