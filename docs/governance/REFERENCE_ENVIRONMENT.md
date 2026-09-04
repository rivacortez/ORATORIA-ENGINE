# Reference hardware and performance test environment

**Status:** annotation workstation recorded, 2026-09-04. Pilot inference host
not yet provisioned — see below.

**Phase 0 deliverable (§13).**

---

## Why the record exists before the values

`BASELINES.md` §6 states the _rule_ — pilot hardware, concurrency and audio
characteristics are recorded alongside every latency and throughput measurement
— and names no machine. A rule with no instance is a rule nobody has had to
follow yet, and the first person who produces a latency number will produce it
on whatever laptop is in front of them.

## What a missing value costs

NFR-005 targets a p95 partial-transcript latency of 1.5 s. That figure is
meaningless without the machine, the load and the audio it was measured on:

- A free-tier host and the pilot VPS differ by more than the whole budget.
- One concurrent stream and eight differ by more than the whole budget.
- Clean 16 kHz PCM and 24 kHz Opus over a lossy connection differ in decode
  cost before inference starts.

A p95 reported without them is not a weak measurement. It is not a measurement.

---

## Annotation workstation

The machine Pilot A and Pilot B run on. Recorded because the annotators' audio
path is part of the corpus, not context around it — see "The capture chain is
part of the data" below.

| Field                         | Value                                                                |
| ----------------------------- | -------------------------------------------------------------------- |
| Role                          | development + annotation                                             |
| CPU model and core count      | 13th Gen Intel Core i7-13650HX — 14 cores / 20 threads, 2.6 GHz base |
| RAM                           | 24 GB (23.7 GB visible)                                              |
| GPU model                     | NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0)         |
| GPU memory                    | 8151 MiB                                                             |
| Secondary GPU                 | Intel UHD Graphics (integrated, display only)                        |
| CUDA version                  | UMD 13.3 (driver-reported); no CUDA toolkit installed                |
| cuDNN version                 | not installed                                                        |
| Driver version                | 610.88 (`nvidia-smi`); Windows driver package 32.0.16.1088           |
| Operating system and kernel   | Windows 11 Home Single Language, build 26200                         |
| Python version                | 3.12.10 (CPython, MSC v.1943, 64-bit)                                |
| Package manager               | uv 0.9.29                                                            |
| Container runtime and version | Docker 29.1.3 (build f52814d)                                        |
| Storage type                  | NVMe SSD — Samsung MZAL81T0HFLB-00BL2 (1 TB)                         |
| Network to the client         | n/a — annotation is local, no client/server hop                      |

> **8 GB of VRAM is a constraint worth noticing now.** `whisper-large-v3` in
> fp16 is 3.1 GB of weights and CrisperWhisper 3.2 GB, so either fits alone
> with room for activations — but not both resident at once alongside a visual
> model. Whatever Phase 3 does about that is a design decision, and it is
> better made before a benchmark is written than after one fails.

## Pilot inference host

**Not yet provisioned.** No row here is filled, and the blank is deliberate:
the engine's `ENGINE_RUNTIME_MODE` accepts `deterministic` and refuses
`managed` with a `NotImplementedError`, so there is no inference path to host.

Filling these rows with the workstation above would break the very rule this
document ends with — a development laptop is never the source of a figure that
enters a document.

| Field                                           | Value                               |
| ----------------------------------------------- | ----------------------------------- |
| Role                                            | pilot                               |
| CPU model and core count                        | _pending provisioning (Phase 3)_    |
| RAM                                             | _pending_                           |
| GPU model                                       | _pending_                           |
| GPU memory                                      | _pending_                           |
| CUDA / cuDNN / driver version                   | _pending_                           |
| Operating system and kernel                     | _pending_                           |
| Container image and digest                      | _pending — see `BASELINE_PINS.md`_  |
| Storage type                                    | _pending_                           |
| Network to the client (LAN / WAN, measured RTT) | _pending — measured, not estimated_ |

**Blocker:** the managed runtime (Phase 3). NFR-005, NFR-006 and NFR-007 cannot
carry a number until this table is filled from the machine that produced it.

## Capture devices

Capture hardware is part of the measurement, not context. FR-020's quality
gates and the visual indicators are relative to what the camera can see, and a
webcam's low-light behaviour changes the availability rate of every visual
indicator.

| Field                                      | Value                                                                                                                         |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Microphone model                           | Realtek(R) Audio — onboard array (laptop built-in)                                                                            |
| Microphone placement                       | laptop built-in                                                                                                               |
| Sample rate and bit depth as captured      | **to be pinned by the operator before Pilot A** — target 16 kHz / 16-bit PCM mono, matching both baselines' feature extractor |
| Camera model                               | "Integrated Camera", USB `VID_5986&PID_2175` (Bison/Acer module)                                                              |
| Camera resolution and frame rate           | **not measured** — recorded at Phase 5, when visual events are annotated                                                      |
| Camera position relative to eye level      | below eye level (laptop lid)                                                                                                  |
| Room lighting (approximate lux, direction) | **not measured** — needs a light meter; the operator records it at Pilot A                                                    |

> Camera position matters more than it looks. A laptop camera below eye level
> makes every speaker's gaze read as downward against an absolute reference,
> which is why `VisualCalibration` measures against the speaker's own baseline
> (ADR-009, `domain/visual_events/calibration.py`). This workstation has
> exactly that geometry, which makes it a good place to check the calibration
> does its job and a bad place to trust an absolute gaze figure from.

### The capture chain is part of the data

This workstation has **VB-Audio Voicemeeter** and **NVIDIA Broadcast** installed
and enumerating as audio endpoints. Neither may be in the recording path for
Pilot A or Pilot B, and this is not a tidiness preference:

- **NVIDIA Broadcast's noise removal is a speech enhancer.** It is built to
  suppress everything that is not clean speech — and breath, creak, glottal
  stops and the trailing energy of a cut-off word are exactly what it is
  built to remove. It would attenuate the acoustic evidence for
  `cut_off`, `prolongation` and `silent_pause` boundaries before an annotator
  ever hears them, and the corpus would encode the enhancer's decisions as
  ground truth.
- **Voicemeeter is a virtual mixer.** Anything routed through it has been
  resampled and mixed, and which of eleven virtual endpoints was actually
  recorded from is not recoverable from the resulting file.

**Requirement:** the pilot records from the named physical device, at the
declared sample rate, with both virtual chains bypassed — and the operator
confirms it by checking the recorded file's sample rate and channel count
rather than by checking a settings dialog. FR-011 asks for verbatim
transcription; a verbatim transcript of enhanced audio is a verbatim transcript
of something the speaker did not say.

## Load conditions

Every latency or throughput figure states which of these it was measured under.
**All pending** — they describe a benchmark run, and none has happened.

| Field                                            | Value                                         |
| ------------------------------------------------ | --------------------------------------------- |
| Concurrent streams                               | _pending_                                     |
| Audio window size and cadence                    | _pending_                                     |
| Video frame rate submitted                       | _pending_                                     |
| Visual path (raw frames / client-side landmarks) | _pending_                                     |
| Session duration                                 | _pending_                                     |
| Warm or cold model                               | _pending — reported separately, never merged_ |
| Measurement duration and number of runs          | _pending_                                     |

## Environments that are not comparable

Named so that nobody accidentally reports across them.

| Environment                    | Use                               | Never used for                    |
| ------------------------------ | --------------------------------- | --------------------------------- |
| CI runner (GitHub-hosted)      | correctness                       | any latency or throughput figure  |
| Annotation workstation (above) | correctness, tooling, annotation  | any figure that enters a document |
| Pilot host                     | every reported performance figure | —                                 |

§13 Phase 9 and the rule that no figure enters a document unless it came from
this project's own benchmark runner both depend on this table being honoured.

---

## How these values were obtained

Not typed from memory. Reproducible on the same machine:

```sh
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors"
powershell -NoProfile -Command "Get-CimInstance Win32_OperatingSystem | Format-List Caption,BuildNumber,TotalVisibleMemorySize"
powershell -NoProfile -Command "Get-CimInstance Win32_PnPEntity | Where-Object { \$_.PNPClass -in @('AudioEndpoint','Camera') } | Select-Object PNPClass,Name"
python -VV; uv --version; docker --version
```

## Filled by

|                  |                                                                   |
| ---------------- | ----------------------------------------------------------------- |
| Filled on        | 2026-09-04                                                        |
| Filled by        | annotation workstation section, from the machine itself           |
| Verified against | not yet — no benchmark run exists; the pilot host section is open |
