# Reference hardware and performance test environment

**Status:** record created, values not yet filled.
**Phase 0 deliverable (§13).** Fill after Pilot B, before any latency figure is
produced.

---

## Why the record exists before the values

`BASELINES.md` §6 states the _rule_ — pilot hardware, concurrency and audio
characteristics are recorded alongside every latency and throughput measurement
— and names no machine. A rule with no instance is a rule nobody has had to
follow yet, and the first person who produces a latency number will produce it
on whatever laptop is in front of them.

So the fields exist now, empty, and every one of them has to be filled before
NFR-005, NFR-006 or NFR-007 can carry a number.

## What a missing value costs

NFR-005 targets a p95 partial-transcript latency of 1.5 s. That figure is
meaningless without the machine, the load and the audio it was measured on:

- A free-tier host and the pilot VPS differ by more than the whole budget.
- One concurrent stream and eight differ by more than the whole budget.
- Clean 16 kHz PCM and 24 kHz Opus over a lossy connection differ in decode
  cost before inference starts.

A p95 reported without them is not a weak measurement. It is not a measurement.

---

## Inference host

| Field                                           | Value                        |
| ----------------------------------------------- | ---------------------------- |
| Role                                            | _(pilot / development / CI)_ |
| CPU model and core count                        | _                            |
| RAM                                             | _                            |
| GPU model                                       | _                            |
| GPU memory                                      | _                            |
| CUDA version                                    | _                            |
| cuDNN version                                   | _                            |
| Driver version                                  | _                            |
| Operating system and kernel                     | _                            |
| Python version                                  | _                            |
| Container runtime and version                   | _                            |
| Storage type (NVMe / SSD / network)             | _                            |
| Network to the client (LAN / WAN, measured RTT) | _                            |

## Capture devices

Capture hardware is part of the measurement, not context. FR-020's quality
gates and the visual indicators are relative to what the camera can see, and a
webcam's low-light behaviour changes the availability rate of every visual
indicator.

| Field                                                   | Value |
| ------------------------------------------------------- | ----- |
| Microphone model                                        | _     |
| Microphone placement (headset / desk / laptop built-in) | _     |
| Sample rate and bit depth as captured                   | _     |
| Camera model                                            | _     |
| Camera resolution and frame rate                        | _     |
| Camera position relative to eye level                   | _     |
| Room lighting (approximate lux, direction)              | _     |

> Camera position matters more than it looks. A laptop camera below eye level
> makes every speaker's gaze read as downward against an absolute reference,
> which is why `VisualCalibration` measures against the speaker's own baseline
> (ADR-009, `domain/visual_events/calibration.py`).

## Load conditions

Every latency or throughput figure states which of these it was measured under.

| Field                                            | Value |
| ------------------------------------------------ | ----- |
| Concurrent streams                               | _     |
| Audio window size and cadence                    | _     |
| Video frame rate submitted                       | _     |
| Visual path (raw frames / client-side landmarks) | _     |
| Session duration                                 | _     |
| Warm or cold model                               | _     |
| Measurement duration and number of runs          | _     |

## Environments that are not comparable

Named so that nobody accidentally reports across them.

| Environment               | Use                               | Never used for                    |
| ------------------------- | --------------------------------- | --------------------------------- |
| CI runner (GitHub-hosted) | correctness                       | any latency or throughput figure  |
| Development laptop        | correctness, tooling              | any figure that enters a document |
| Pilot host                | every reported performance figure | —                                 |

§13 Phase 9 and the rule that no figure enters a document unless it came from
this project's own benchmark runner both depend on this table being honoured.

---

## Filled by

|                  |                                                       |
| ---------------- | ----------------------------------------------------- |
| Filled on        | _                                                     |
| Filled by        | _                                                     |
| Verified against | _(the benchmark run that produced the first figures)_ |
