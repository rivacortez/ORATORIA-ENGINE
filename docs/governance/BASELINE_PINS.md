# Frozen baseline pins

**Status:** record created, pins not yet filled.
**Phase 0 deliverable (§13).** Fill after Pilot B, before Phase 3.

---

## What "frozen" means here, and what it does not

`BASELINES.md` says the baselines are pinned and then pins nothing: it names
Whisper and CrisperWhisper as "pinned checkpoint" without an identifier, a
revision or a digest. That is a protocol, not a baseline. This file is the
pins.

**Two things get frozen, at two different times, and conflating them is a
methodological error:**

| Frozen                                  | When            | Why then                                                                                                                                                                  |
| --------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The **models and their configuration**  | now (Phase 0.5) | So the comparison is fixed before anyone has seen a result it could be tuned against.                                                                                     |
| Their **outputs over the held-out set** | end of Phase 1  | The held-out set does not exist yet. §14.4 freezes it at the end of corpus construction, and an output generated before then is an output over data that is still moving. |

`BASELINES.md` §1 currently reads as though outputs can be generated now. They
cannot, and the corrected sequencing is recorded here.

## Why pins are needed at all

NFR-001 asks the project model to outperform a frozen baseline. That comparison
is only meaningful if the baseline was chosen and fixed _before_ anyone saw the
test results — a baseline selected afterwards is selected, consciously or not,
to be beatable.

A model name is not a pin. `whisper-large-v3` has been re-uploaded, re-quantised
and re-packaged; two people running "the same model" a year apart are not
running the same weights. The digest is what makes the claim checkable.

---

## Whisper

| Field                                       | Value                              |
| ------------------------------------------- | ---------------------------------- |
| Model identifier                            | _(e.g. `openai/whisper-large-v3`)_ |
| Hub revision (commit sha)                   | _                                  |
| Weights digest (sha256 of the artifact)     | _                                  |
| Parameter count                             | _                                  |
| Quantisation                                | _(none / int8 / fp16)_             |
| Inference library and version               | _                                  |
| Decoding: beam size                         | _                                  |
| Decoding: temperature and fallback schedule | _                                  |
| Decoding: language forced or detected       | _                                  |
| Decoding: `condition_on_previous_text`      | _                                  |
| Decoding: suppressed tokens                 | _                                  |
| VAD or chunking applied before decoding     | _                                  |
| Timestamp granularity (segment / word)      | _                                  |

> **Timestamp granularity is not a detail.** NFR-004's ≤250 ms boundary target
> is unreachable from segment-level timestamps without a separate
> forced-alignment pass. If this row says `segment`, the baseline's boundary
> error is not comparable with the project model's and the comparison has to
> say so.

## CrisperWhisper

| Field                                   | Value |
| --------------------------------------- | ----- |
| Model identifier                        | _     |
| Hub revision (commit sha)               | _     |
| Weights digest (sha256 of the artifact) | _     |
| Parameter count                         | _     |
| Quantisation                            | _     |
| Inference library and version           | _     |
| Decoding configuration                  | _     |
| Timestamp granularity                   | _     |

## Reproducible environment

| Field                         | Value                                    |
| ----------------------------- | ---------------------------------------- |
| Container image and digest    | _                                        |
| Python version                | _                                        |
| Torch version and CUDA build  | _                                        |
| Random seed                   | _                                        |
| Deterministic kernels enabled | _                                        |
| Hardware                      | _(reference `REFERENCE_ENVIRONMENT.md`)_ |

Seeds and deterministic kernels are listed because NFR-015 requires equivalent
output from equivalent input. A baseline that cannot reproduce its own numbers
cannot be a reference for anything.

---

## Neither baseline becomes a production dependency

§11.1 lists this as a constraint and it is easy to violate by accident: a
baseline adapter wired in "temporarily" for a demo is how a frozen research
comparator becomes the thing serving traffic. The engine's runtime selection is
`ENGINE_RUNTIME_MODE`, which today accepts `deterministic` and refuses
`managed` with a NotImplementedError; when the managed path lands, a baseline
must not be reachable through it.

## Output generation, when the held-out set exists

Recorded now so the order is not re-litigated later:

1. Phase 1 closes. The held-out set is frozen and speaker-independent.
2. Both baselines run once over it, under the configuration pinned above.
3. Outputs are stored as a versioned artifact with the digest of the input set.
4. Neither baseline is re-run against a newer checkpoint of itself for the
   remainder of the project.

Step 4 is the one that gets broken. Re-running a baseline after a library
upgrade produces different numbers for the same "frozen" comparator, and the
project model's improvement then includes whatever the upgrade did.

---

## Filled by

|                      |                                      |
| -------------------- | ------------------------------------ |
| Filled on            | _                                    |
| Filled by            | _                                    |
| Outputs generated on | _(after the held-out set is frozen)_ |
