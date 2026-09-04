# Frozen baseline pins

**Status:** models and configuration frozen, 2026-09-04.
**Outputs:** not frozen. See "Two things get frozen" below — the held-out set
does not exist yet.
**Phase 0 deliverable (§13).**

---

## What "frozen" means here, and what it does not

`BASELINES.md` says the baselines are pinned and then pins nothing: it names
Whisper and CrisperWhisper as "pinned checkpoint" without an identifier, a
revision or a digest. That is a protocol, not a baseline. This file is the
pins.

**Two things get frozen, at two different times, and conflating them is a
methodological error:**

| Frozen                                  | When                 | Why then                                                                                                                                                                  |
| --------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The **models and their configuration**  | **done, 2026-09-04** | So the comparison is fixed before anyone has seen a result it could be tuned against.                                                                                     |
| Their **outputs over the held-out set** | end of Phase 1       | The held-out set does not exist yet. §14.4 freezes it at the end of corpus construction, and an output generated before then is an output over data that is still moving. |

`BASELINES.md` §1 currently reads as though outputs can be generated now. They
cannot, and the corrected sequencing is recorded here.

## Why pins are needed at all

NFR-001 asks the project model to outperform a frozen baseline. That comparison
is only meaningful if the baseline was chosen and fixed _before_ anyone saw the
test results — a baseline selected afterwards is selected, consciously or not,
to be beatable.

A model name is not a pin. **This project has already been bitten by exactly
that:** `BASELINES.md` names CrisperWhisper, and between the specification
being written and these pins being taken, the publisher moved the repository
from `nyrahealth/CrisperWhisper` to `nyralabs/CrisperWhisper` (the old path now
answers with a 307) and published a `CrisperWhisper2.0_large` alongside it,
with different weights, a different dtype and a different loading library.
"CrisperWhisper" now names three artifacts. The revision below names one.

---

## Whisper

| Field                                       | Value                                                                                                  |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Model identifier                            | `openai/whisper-large-v3`                                                                              |
| Hub revision (commit sha)                   | `06f233fe06e710322aca913c1bc4249a0d71fce1`                                                             |
| Artifact pinned                             | `model.safetensors`                                                                                    |
| Weights digest (sha256)                     | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`                                     |
| Artifact size (bytes)                       | `3087130976`                                                                                           |
| Parameter count                             | 1 543 490 560                                                                                          |
| Quantisation                                | fp16 as published (`F16` for all parameters)                                                           |
| Repository last modified                    | 2024-08-12T10:20:10Z                                                                                   |
| Inference library and version               | `transformers` — version pinned at Phase 3, recorded here when the runtime lands                       |
| Decoding: beam size                         | 5                                                                                                      |
| Decoding: temperature and fallback schedule | `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`, the reference fallback ladder                                        |
| Decoding: language forced or detected       | **forced** to `es`                                                                                     |
| Decoding: `condition_on_previous_text`      | `False`                                                                                                |
| Decoding: suppressed tokens                 | the checkpoint's own 88 entries; `begin_suppress_tokens` `[220, 50257]`; **no additional suppression** |
| VAD or chunking applied before decoding     | none — 30 s windows as published, no external VAD                                                      |
| Timestamp granularity                       | **word**, via `return_timestamps="word"` and the checkpoint's 10 published alignment heads             |

### Why these decoding values and not others

They are _decisions_, not facts read off the Hub, and each is the one that
keeps the comparison honest rather than the one that flatters either side.

**Language forced.** Detection introduces a failure mode that has nothing to do
with the research question: a Spanish speaker's first three seconds detected as
Portuguese produces a garbage transcript and a baseline that looks worse than
it is. The corpus is `es-PE` throughout (`Speaker.variety`), so the language is
known.

**`condition_on_previous_text=False`.** This is the setting that matters most
for this project and the reason is FR-011. With conditioning on, the decoder is
biased toward fluent continuations of what it already produced — which is
precisely the "grammatical cleanup" the corpus exists to measure the absence
of. Leaving it on would let the baseline silently repair the disfluencies it is
being scored on finding.

**No additional token suppression.** Suppressing filler tokens would be
suppressing the labels.

**Word timestamps.** NFR-004's ≤250 ms boundary target is unreachable from
segment-level timestamps without a separate forced-alignment pass, and a
comparison where the baseline is scored on segment boundaries and the project
model on word boundaries is not a comparison. `whisper-large-v3` ships
alignment heads, so word timestamps are available from the checkpoint itself
with no extra component to pin.

## CrisperWhisper

| Field                          | Value                                                                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Model identifier               | `nyralabs/CrisperWhisper`                                                                                                                       |
| Previous identifier            | `nyrahealth/CrisperWhisper` — moved, now redirects (307)                                                                                        |
| Hub revision (commit sha)      | `5a513ffabb3429bb61da3521c4e24fbd33ec3e7c`                                                                                                      |
| Artifact pinned                | `model.safetensors`                                                                                                                             |
| Weights digest (sha256)        | `ab370a321fa94ce08d419959813540a4f9b0239b88b2f4207ae28ef6607083c0`                                                                              |
| Artifact size (bytes)          | `3219908024`                                                                                                                                    |
| Parameter count                | 1 609 879 040                                                                                                                                   |
| Quantisation                   | fp16 as published (`F16` for all parameters; `torch_dtype: float16`)                                                                            |
| Architecture                   | `whisper`, 32 encoder / 32 decoder layers, `d_model` 1280, 128 mel bins                                                                         |
| Feature extraction             | 16 kHz, 30 s chunks, `n_fft` 400, `hop_length` 160, 128 mel bins                                                                                |
| Published under `transformers` | 4.37.2                                                                                                                                          |
| Repository last modified       | 2026-07-22T12:07:48Z                                                                                                                            |
| Inference library and version  | `transformers` — version pinned at Phase 3                                                                                                      |
| Decoding configuration         | as Whisper above, with the checkpoint's own `suppress_tokens` (6 745 entries) and `forced_decoder_ids` `[[1, None], [2, 50360]]` left untouched |
| Timestamp granularity          | **word**, via the checkpoint's 10 published alignment heads                                                                                     |

> **`CrisperWhisper2.0_large` is deliberately not the pin.** It exists
> (`f4334f6e8193f2691212d49b20fa12d370e13896`, BF16, loaded through a
> `crisperwhisper` library rather than `transformers`) and is newer. It is not
> the model the published work describes, it needs a different loading path,
> and swapping the baseline for a newer one mid-project is how a comparison
> stops being reproducible. If it is adopted later, it is adopted as a
> _second_ pinned baseline with its own row, not as an update to this one.

### Why CrisperWhisper is the baseline that matters

Whisper is the general reference; CrisperWhisper is the hard one. It was
trained specifically to transcribe verbatim — to keep the filled pauses,
repetitions and false starts that Whisper's training data taught it to remove.
NFR-001's claim is only interesting against a baseline that is already trying
to do the thing: beating vanilla Whisper at disfluency detection would mostly
be reporting that Whisper deletes disfluencies.

## Reproducible environment

| Field                         | Value                                                                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Container image and digest    | **not yet pinned** — no inference container exists; `ENGINE_RUNTIME_MODE` accepts `deterministic` and refuses `managed`. Pinned when the managed runtime lands (Phase 3). |
| Python version                | 3.12.10 (`CPython`, MSC v.1943, 64-bit)                                                                                                                                   |
| uv version                    | 0.9.29                                                                                                                                                                    |
| Torch version and CUDA build  | **not yet pinned** — torch is not a dependency of this repository (C2/C3 forbid it in the domain and application layers, and no runtime imports it yet).                  |
| Random seed                   | `20260904`                                                                                                                                                                |
| Deterministic kernels enabled | required: `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, verified at Phase 3                                                           |
| Hardware                      | see `REFERENCE_ENVIRONMENT.md`                                                                                                                                            |

Seeds and deterministic kernels are listed because NFR-015 requires equivalent
output from equivalent input. A baseline that cannot reproduce its own numbers
cannot be a reference for anything.

**The three unpinned rows are unpinned honestly, not by oversight.** There is
no inference container and no torch dependency in this repository yet, and
writing a plausible version string for either would be inventing a pin — the
exact failure this document exists to prevent. They are pinned in the same
commit that introduces the runtime, and Phase 3 cannot close without them.

## How to verify a pin

Nothing here has to be taken on trust:

```sh
# Revision and metadata
curl -sL "https://huggingface.co/api/models/openai/whisper-large-v3" | jq .sha

# Weights digest, from the git-lfs pointer at that exact revision
curl -sL "https://huggingface.co/openai/whisper-large-v3/raw/06f233fe06e710322aca913c1bc4249a0d71fce1/model.safetensors"
# -> oid sha256:a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95

# And after downloading, that the bytes on disk are those bytes
sha256sum model.safetensors
```

A revision that no longer resolves, or a digest that no longer matches, is not
a broken link — it is the pin doing its job.

---

## Neither baseline becomes a production dependency

§11.1 lists this as a constraint and it is easy to violate by accident: a
baseline adapter wired in "temporarily" for a demo is how a frozen research
comparator becomes the thing serving traffic. The engine's runtime selection is
`ENGINE_RUNTIME_MODE`, which today accepts `deterministic` and refuses
`managed` with a NotImplementedError; when the managed path lands, a baseline
must not be reachable through it.
