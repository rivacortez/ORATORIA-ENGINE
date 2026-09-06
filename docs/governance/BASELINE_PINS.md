# Frozen baseline pins

**Status:** **weights and runtime-independent decoding configuration frozen**,
2026-09-04 — identifier, revision, weights digest, mode, language and timestamp
granularity. Everything that decides *how the model is run* is still open: the
CrisperWhisper backend is a choice between ct2 and transformers, Whisper's
`transformers` version is unpinned, and there is no container, Torch or CUDA
build to pin. That is the Phase 3 row of the cycle below.
**Revised 2026-09-04:** the verbatim baseline was re-pinned from CrisperWhisper
v1 to 2.0. v1 is English/German-only by its own card, so it could not be the
comparator for an `es-PE` corpus. See "Why this replaces the v1 pin".
**Outputs:** not frozen, and not due until Phase 3. See the cycle below.
**Phase 0 deliverable (§13).**

---

## What "frozen" means here, and what it does not

`BASELINES.md` says the baselines are pinned and then pins nothing: it names
Whisper and CrisperWhisper as "pinned checkpoint" without an identifier, a
revision or a digest. That is a protocol, not a baseline. This file is the
pins.

**Four things get frozen, at four different times, and conflating any two of
them is a methodological error.** This table is the authority; `BASELINES.md`
carries the same four rows, and they are checked against each other by
`tests/corpus/test_scope.py`.

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

An earlier version of `BASELINES.md` read as though outputs could be generated
now, and a later one scheduled them for the end of Phase 1 - before the
environment that produces them is pinned. Both are corrected against the table
above.

## Why pins are needed at all

NFR-001 asks the project model to outperform a frozen baseline. That comparison
is only meaningful if the baseline was chosen and fixed _before_ anyone saw the
test results — a baseline selected afterwards is selected, consciously or not,
to be beatable.

A model name is not a pin. **This project has been bitten by exactly that,
twice, on the same model.**

First: `BASELINES.md` names "CrisperWhisper", and between the specification
being written and these pins being taken, the publisher moved the repository
from `nyrahealth/CrisperWhisper` to `nyralabs/CrisperWhisper` (the old path now
answers with a 307) and published `CrisperWhisper2.0_large` alongside it, with
different weights, a different dtype, a different tokenizer and a different
loading library. The name now points at three artifacts.

Second, and worse: the first version of this file resolved that ambiguity in
the wrong direction. It pinned v1 — a model whose own card says it is
English/German-only — as the verbatim comparator for a Peruvian Spanish corpus,
and argued for the choice on the grounds of not swapping baselines mid-project.
The argument was sound and the premise was not checked. A pin is only as good
as the reading of the model card that produced it.

Both failures are recorded rather than quietly corrected, because "we pinned
the wrong model and caught it in review" is the evidence that the pinning
discipline works.

---

## Whisper

| Field                                       | Value                                                                                                  |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Model identifier                            | `openai/whisper-large-v3`                                                                              |
| Hub revision (commit sha)                   | `06f233fe06e710322aca913c1bc4249a0d71fce1`                                                             |
| Artifact pinned                             | `model.safetensors`                                                                                    |
| Weights digest (sha256)                     | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`                                     |
| Artifact size (bytes)                       | `3087130976` — **the pinned artifact only.** The unfiltered repository is roughly 31.6 GB: it also carries `flax_model.msgpack` (6.17 GB), `pytorch_model.bin` (3.09 GB) and fp32 shards in both formats. Download with an explicit include list, or pay ten times the bandwidth for files no pin references. |
| Parameter count                             | 1 543 490 560                                                                                          |
| Quantisation                                | fp16 as published (`F16` for all parameters)                                                           |
| Repository last modified                    | 2024-08-12T10:20:10Z                                                                                   |
| Inference library and version               | `transformers`, **version not yet pinned** — Phase 3, with the rest of the executable environment       |
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

## CrisperWhisper 2.0 — the verbatim baseline

| Field | Value |
| --- | --- |
| Model identifier | `nyralabs/CrisperWhisper2.0_large` |
| Hub revision (commit sha) | `f4334f6e8193f2691212d49b20fa12d370e13896` |
| Artifact pinned | `model.safetensors` |
| Weights digest (sha256) | `f097f853b6992e66fcae802a15770a8242670351198d68dad577db3209fe3fcc` |
| Artifact size (bytes) | `3086843114` |
| Parameter count | 1 543 345 921 |
| Quantisation | bf16 as published (`BF16` for all parameters) |
| Architecture | `whisper`, 32 encoder / 32 decoder layers, `d_model` 1280, 80 mel bins, vocab 51 896 |
| Published under `transformers` | 4.57.6 |
| Repository last modified | 2026-08-13T15:51:53Z |
| Inference library | `crisperwhisper` **2.0.2** (PyPI) — not `transformers`; the repo declares `library_name: crisperwhisper` |
| Backend | **`transformers` on Windows; `ct2` is not available there at all.** `ctranslate2-crisperwhisper` publishes 14 wheels across 4.7.1.post2/post3, every one `manylinux_2_27_x86_64.manylinux_2_28_x86_64`, and **no sdist** - so pip cannot install it on Windows and cannot even attempt a source build. Which backend produced a figure is recorded with the figure. |
| Entry point | `from crisperwhisper import CrisperWhisperModel` |
| Decoding: mode | **`mode="verbatim"`** — the default, and the entire reason this model is here. `mode="intended"` is the clean-transcript mode and must never produce a baseline figure |
| Decoding: language | **forced** to `es` |
| Decoding: word timestamps | `word_timestamps=True` |
| Hallucination mitigation | on by default (detects and suppresses Whisper's looping-repetition failure). Left on, and recorded, because it is a decoding intervention |
| Deliberately not used | `transcribe_dual`, `verbatimize`, and the `Pro` tier — each changes what is being measured |
| Licence — software | MIT (inference code, pre- and post-processing) |
| Licence — weights **and outputs** | nyra health Non-Commercial Research Licence |

> **A licence note that reaches the thesis, not just the repo.** The split
> licence puts the model *Outputs* under the non-commercial term, and the
> frozen baseline outputs over the held-out set are Outputs. Thesis and
> research use sit inside the term; any later commercial use of this engine
> needs either a licence from nyra health or a comparison regenerated without
> this baseline. Recorded now because it is cheap now and expensive later.

### Why this replaces the v1 pin

The first version of this file pinned `nyralabs/CrisperWhisper` (v1). That was
wrong, and v1's own card says why: **"CrisperWhisper 1.0 is
English/German-only"**, `language: ['de', 'en']` in the metadata, and every
cited benchmark English (AMI IHM, TED-LIUM, TIMIT). Pinning it as the verbatim
comparator for an `es-PE` corpus meant pinning a model that does not claim to
do the task.

2.0 is a different proposition: multilingual, with verbatim/intended style
control, word timestamps, and a published disfluency-F1 leaderboard across ten
languages.

| System | Disfluency F1 (10-language average) |
| --- | ---: |
| CrisperWhisper 2.0 Pro | 93.5 |
| **CrisperWhisper 2.0** | **87.8** |
| ElevenLabs Scribe v2 | 79.2 |
| Microsoft MAI-Transcribe-1.5 | 77.5 |
| CrisperWhisper 1.0\* | 64.8 |
| Deepgram Nova-3 | 37.8 |

<sub>\* en/de only. Source: the Nyra Verbatim Speech Benchmark table in the
model card, read at revision `f4334f6e`.</sub>

**This raises the bar the project's own model has to clear**, which is the
honest consequence of pinning the right baseline. The wrong pin would have set
expectations against a model scoring 64.8 on its own two languages.

**87.8 is not the NFR-001 threshold, and must not be quoted as one.** It is a
ten-language average in which eight of the ten languages — Spanish among them —
are evaluated on synthetic verbatim sets, and the average includes English and
German, which this thesis does not measure. A target taken from it would be a
target against a number produced from data that is neither Spanish nor human
nor spontaneous.

| Figure                              | What it is                                             | Role here                                |
| ----------------------------------- | ------------------------------------------------------ | ---------------------------------------- |
| 87.8 disfluency F1                  | published 10-language average, 8 languages synthetic   | context, and the reason to pin 2.0 at all |
| **this baseline on the es-PE held-out set** | **not yet measured** — Phase 1 produces the held-out set | **the NFR-001 comparator**        |

The comparator is the second row. It does not exist yet, it is produced by
running this exact pin over the human-annotated Peruvian held-out slice, and
until it does exist NFR-001 has no number to be read against. That is the
correct state for a target whose corpus has not been recorded.

### Run on this machine, 2026-09-05

The pins above were taken from metadata. This is what happened when the
artifact was downloaded and run over real Peruvian audio — OpenSLR SLR73, all
38 speakers, 20 clips each, 77.6 minutes.

**Nothing here is a figure.** `REFERENCE_ENVIRONMENT.md` says a development
laptop never sources a reported number, and this is the annotation workstation.
What the run establishes is that the pins are runnable and that the decoding
configuration is applicable — not how well the baseline performs.

| Checked | Result |
| --- | --- |
| Weights digest against the pin | **matches** — 3 087 130 976 bytes, sha256 `a8e94b85…`, byte for byte |
| Loads at fp16 on 8 GB | yes — 4.19 GiB resident, 7.9 s |
| CUDA build for this card | `torch 2.14.0+cu130` lists `sm_120`; the RTX 5060 is compute capability (12, 0) |
| Pinned decoding applies | yes — forced `es`, beam 5, the temperature ladder, `condition_on_prev_tokens=False` |
| **Word-level timestamps** | **yes** — e.g. `Hay` 0.00–0.92, `Gaceta` 2.30–2.78 |
| Throughput | 0.267× real time, beam 5, fp16, batch 4 — indicative only |

**The word-timestamp row is the one that mattered.** NFR-004's 250 ms boundary
target is unreachable from segment-level timestamps without a separate
forced-alignment component, and this document asserted none was needed because
the checkpoint ships alignment heads. That was an argument from a model card.
It is now an observation.

**How to read them, which is not obvious.** `model.generate(...,
return_timestamps="word", return_dict_in_generate=True)` returns a dict whose
`segments` carry `start`, `end`, `tokens` and `idxs` — segment boundaries, and
**no per-word offsets**, with no error. The word offsets come from the
`automatic-speech-recognition` pipeline with `return_timestamps="word"`, as
`chunks`. Recorded because the first attempt took the obvious route and got
zero offsets back silently, which is exactly how a project ends up believing it
has word timestamps and shipping segment ones.

#### A word-error rate, and why the headline number is not the answer

Over 7 223 reference words:

| | |
| --- | --- |
| WER as computed | **0.0299** (216 errors) |
| **WER counting recognition errors only** | **0.0089** |

Classifying every error by aligning the two token streams and looking at each
substitution:

| Kind | Count | Share | Contributes |
| --- | ---: | ---: | ---: |
| Accent only | 82 | 38.0% | 0.0114 |
| Number formatting | 70 | 32.4% | 0.0097 |
| **Recognition** | **64** | **29.6%** | **0.0089** |

**Seven of every ten "errors" are not the model's.** The accent bucket runs
almost entirely in the direction of the model writing correct Spanish where the
reference does not — `este` → `esté`, `llegue` → `llegué`, `que` → `qué`,
`sandwiches` → `sándwiches`. SLR73's transcripts carry orthographic errors, and
an accent-sensitive WER counts the model's corrections as mistakes.

The normalisation keeps accents deliberately: in Spanish they are phonemic and
lexical — `esta`/`está`, `el`/`él` — and folding them would hide real errors.
That decision is right and it is also why the classification is not optional.
The number bucket is separate for the reason SLR73's own documentation gives:
its transcripts "have not been text normalized and may contain non-standard
word (NSW) tokens... such as abbreviations and cardinal numbers".

The genuine recognition errors are the kind one would expect: `resguardar` →
`reguardar`, `jeroglífico` → `jerolífico`, `cambiarles` → `cambiarle`.

**The rule this leaves.** Any WER this project reports arrives with its error
classification beside it. A bare 3% here would have been half the reference's
orthography, and no reader could have told.

#### The negative control fired correctly

Zero filler tokens written that the reference lacks, across 760 clips and 77
minutes. That is the expected answer for read speech whose collection protocol
re-recorded takes containing stuttering, and it is the cheapest available check
that a disfluency detector is not hallucinating: run it over SLR73, and if it
fires often, the detector is miscalibrated.

#### Per-speaker spread, which is the partitioning argument with data

Recognition-only WER ranges from **0.0000 to 0.0462** across the 38 speakers.
Nine speakers have no recognition error at all in roughly 200 words each. A
held-out set that happened to land on those nine would report zero and
generalise to nobody — which is why `corpus split` stratifies rather than
sampling.

### Two corrections a review round later

Recorded rather than silently applied, because the pins document's own argument
is that a pin is only as good as the checking behind it.

**The ct2 backend was described as a fallback situation and is not one.** The
row above previously said `transformers` was the fallback "if ct2 conversion is
unavailable on the pilot host", which reads as a performance trade-off. It is
not: `ctranslate2-crisperwhisper` ships Linux-only wheels and no source
distribution, so on the Windows annotation workstation the ct2 path does not
exist. Verify:

```sh
curl -s "https://pypi.org/pypi/ctranslate2-crisperwhisper/json"   | jq -r '.releases[][] | "\(.packagetype) \(.filename)"' | sort -u
# every line: bdist_wheel ...manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

**2.0's language metadata says the same thing v1's did.** The section below
argues that v1 could not be the comparator because its card declares
`language: ['de', 'en']`. Run the same query against 2.0 at the pinned
revision and it returns **`['en', 'de']`**:

```sh
curl -sL "https://huggingface.co/api/models/nyralabs/CrisperWhisper2.0_large" | jq .cardData.language
# ["en","de"]
```

So the metadata argument does not distinguish the two models, and using it
against v1 while relying on 2.0's README prose is an inconsistency an examiner
finds in one query. The honest position is narrower than the original one:

- **What the metadata says**: en, de. For both.
- **What 2.0's card body adds**: verbatim/intended style control described as
  working "across most languages", and a ten-language disfluency-F1 leaderboard
  in which **eight of the ten, Spanish among them, use synthetic verbatim
  evaluation sets**.
- **What actually separates them**: 2.0's card claims multilingual coverage and
  publishes a Spanish figure; v1's card says "CrisperWhisper 1.0 is
  English/German-only" in its own prose. That sentence, not the metadata field,
  is the discriminator.

The pin stands. The reasoning behind it is now the reasoning that survives the
query, and the field itself is recorded as unresolved: a maintainer who leaves
`language` at `['en','de']` on a model marketed as multilingual has left the
question open, and this document does not close it on their behalf.

### The caveat that does not go away

The card states that **English and German use human-labelled evaluation sets;
the other eight languages use synthetic verbatim sets.** The published Spanish
figure is therefore against synthetic data.

That does not make it a bad baseline. It makes the Peruvian evaluation
mandatory rather than courteous, and it is worth being precise about why:
support for Spanish is not evidence of performance on Peruvian Spanish
disfluencies, and a synthetic verbatim set is not evidence about spontaneous
speech at all. As far as this pin's own benchmark goes, the OratorIA corpus
will be the first human-labelled verbatim `es-PE` measurement of this model.

**Required before any NFR-001 comparison is reported:** run this baseline over
a held-out slice of the human-annotated Peruvian corpus and report its
per-class figures on the same matching rule the inter-annotator agreement uses
(`corpus.agreement.matching`), so the human ceiling, the baseline and the
project model land on one scale. If it underperforms there relative to its
published average, that is a finding about transfer and is reported as one —
not quietly banked as margin for the project model.

### v1, retained as a historical pin only

| Field | Value |
| --- | --- |
| Model identifier | `nyralabs/CrisperWhisper` |
| Previous identifier | `nyrahealth/CrisperWhisper` — moved, now redirects (307) |
| Hub revision | `5a513ffabb3429bb61da3521c4e24fbd33ec3e7c` |
| Weights digest (sha256) | `ab370a321fa94ce08d419959813540a4f9b0239b88b2f4207ae28ef6607083c0` |
| Parameter count | 1 609 879 040, fp16 |
| Languages | **English and German only**, per its own card |
| Licence | CC-BY-NC-4.0 |
| Status | **Not a comparator for this thesis.** Kept pinned so the INTERSPEECH 2024 result can be cited against the exact artifact it describes, and so the record of what was pinned first survives |

### Why a verbatim baseline is the one that matters

Whisper is the general reference; the verbatim model is the hard one. It is
built to keep the filled pauses, repetitions and false starts that Whisper's
training data taught it to remove. NFR-001's claim is only interesting against
a baseline already trying to do the task: beating vanilla Whisper at disfluency
detection would mostly be reporting that Whisper deletes disfluencies.

### On the absence of a Spanish-native verbatim baseline

Searched the Hub for one before settling on transfer from a multilingual model.
What exists: Common Voice fine-tunes of Whisper for Spanish, optimised for
*clean* transcription and therefore the opposite of what is needed; and a
handful of `disfluency-spanish` checkpoints with single-digit download counts,
no paper and no documented provenance. Pinning one of those as a scientific
comparator would be worse than having none.

Recorded as a null result rather than left implicit, because "there is no
published verbatim-Spanish ASR baseline" is part of the argument for this
thesis existing, and a reader is entitled to know it was checked rather than
assumed.

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
curl -sL "https://huggingface.co/api/models/nyralabs/CrisperWhisper2.0_large" | jq .sha

# Weights digest, from the git-lfs pointer at that exact revision
curl -sL "https://huggingface.co/openai/whisper-large-v3/raw/06f233fe06e710322aca913c1bc4249a0d71fce1/model.safetensors"
# -> oid sha256:a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95

curl -sL "https://huggingface.co/nyralabs/CrisperWhisper2.0_large/raw/f4334f6e8193f2691212d49b20fa12d370e13896/model.safetensors"
# -> oid sha256:f097f853b6992e66fcae802a15770a8242670351198d68dad577db3209fe3fcc

# The language claim, which is what the v1 pin got wrong
curl -sL "https://huggingface.co/api/models/nyralabs/CrisperWhisper" | jq .cardData.language
# -> ["de","en"]

# The inference library, which is not transformers for 2.0
curl -s "https://pypi.org/pypi/crisperwhisper/json" | jq -r .info.version
# -> 2.0.2

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

There is now a second reason, and it is not a matter of taste. Both verbatim
checkpoints are licensed for **non-commercial research use only** — v1 under
CC-BY-NC-4.0, 2.0 under the nyra health Non-Commercial Research Licence, which
extends to the model's Outputs. A "temporary" wiring that reaches production
is a licence breach, not just an architectural smell.
