# Frozen baselines and the evaluation protocol

**Status:** protocol approved for Phase 0 · **No results yet — see §5**

---

## 1. Why baselines are frozen before anything is trained

NFR-001 asks the project model to outperform a frozen baseline on the held-out
corpus. That comparison is only meaningful if the baseline is chosen and pinned
_before_ anyone has seen the test results. A baseline selected afterwards is
selected — consciously or not — to be beatable.

So **four** things are frozen, at four different times, and conflating any two
of them is a methodological error:

| Frozen | When | Why then |
| --- | --- | --- |
| **Model artifacts and runtime-independent decoding configuration** — identifier, revision, weights digest, mode, language, timestamp granularity | **done, Phase 0.5** | Before anyone has seen a result the choice could be tuned against. |
| **The held-out set** | end of Phase 1 | It does not exist until then (§14.4). An output over data that is still moving is a reference to nothing. |
| **The executable environment** — backend, container digest, Torch/CUDA build, backend dependencies | Phase 3 | None of it exists yet. There is no inference container and torch is not a dependency of this repository; a version string written now would be invented. |
| **The baseline outputs over the held-out set** | Phase 3, **after** the environment is frozen | An output produced by an unpinned runtime cannot be regenerated. Freezing it earlier freezes a number nobody can reproduce, which is the opposite of what a frozen baseline is for. |

**The order of the last two rows is the part that is easy to get wrong.** It is
tempting to generate the baseline outputs as soon as the held-out set exists,
at the end of Phase 1. That would produce them on whatever backend and CUDA
build happened to be installed, and a pinned weights digest does not rescue
that: ct2 and transformers do not produce bit-identical output from the same
weights, and neither does one backend across two CUDA builds. The outputs wait
for the environment.

An earlier version of this file read as though outputs could be generated now,
and then as though they were due at the end of Phase 1. Neither is right: they
are due in Phase 3, once the environment that produces them is pinned. The pins
themselves live in `BASELINE_PINS.md`, which carries this same table; the
outputs are stored as a versioned artifact alongside the digest of the input
set, and the baseline is never re-run against a newer checkpoint of itself for
the remainder of the project.

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
  check is mechanical rather than a matter of care: `corpus split` assigns
  *speakers*, never recordings, so a speaker cannot straddle a boundary by
  construction - and re-derives the guarantee anyway, because a construction
  survives exactly until the refactor that changes it and a model scored on
  speakers it memorised does not look broken, it looks good.
- The **held-out set is frozen** at the end of Phase 1 and is not touched
  during tuning. §14.4 makes this a scientific gate, and `corpus split
  --freeze` makes it checkable: the manifest carries a SHA-256 per annotation
  file and a digest over the whole assignment, so `corpus verify` answers
  months later whether this is the held-out set a number was computed over,
  whether any file changed, and whether anything was added or moved. A gate
  enforced by everyone remembering is not a gate - the failure it exists to
  prevent looks exactly like ordinary work while it happens.
- Dataset cards record provenance, consent basis, recording conditions and
  partition checksums. The manifest is the machine-readable half; the half a
  human writes - why these speakers, under what consent, in what room - is
  prose and belongs beside it.
- **Partitions are stratified, not random.** On a corpus of tens of speakers
  with nine classes, random assignment routinely lands zero instances of a rare
  class in held-out, and its per-class F1 is then *undefined* - a missing
  measurement that reads as a low score. `corpus split` places the speakers
  carrying the rarest classes first, and refuses to freeze a plan where a P0
  class is missing from any partition.

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
