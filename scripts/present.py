"""Record a presentation, run it through the engine, and read the result.

The one command a person needs to hear what this engine hears.

    uv run python scripts/present.py record --seconds 60 -o mi-exposicion.wav
    uv run python scripts/present.py devices
    uv run python scripts/present.py run mi-exposicion.wav

**What the result is and is not.** It is a real transcript with real word
boundaries and real derived silent pauses, from the pinned checkpoint. It is
not a measurement: `REFERENCE_ENVIRONMENT.md` says a development laptop never
sources a reported figure, and the managed runtime's executable environment is
a Phase 3 freeze that has not happened. Every report this prints says so at the
bottom rather than leaving it to be remembered.

**What it will not find.** Filled pauses, false starts, repetitions,
prolongations. The disfluency detector does not exist - Phase 4 - so the engine
reports them as unavailable with a reason rather than as zero, which would mean
a detector ran and found none. What it *does* find, today, is every word with
its boundary and every silence over the versioned threshold, which is the part
of the taxonomy that is derived rather than detected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    MisdeclaredAudioWindow,
)
from evidence_engine.application.services.calibration import Calibrator
from evidence_engine.application.services.speech_assembly import (
    SpeechAssembler,
    silent_pauses,
)
from evidence_engine.domain.shared.identifiers import RunId
from evidence_engine.domain.shared.provenance import EvidenceRef, Modality, Provenance
from evidence_engine.domain.transcript.tokens import WordToken

#: The engine's decode target. `REFERENCE_ENVIRONMENT.md` gates Pilot A on four
#: audio confirmations and these are two of them; the recorder writes them and
#: the runner checks them rather than trusting the filename.
TARGET_RATE_HZ = 16_000
TARGET_CHANNELS = 1
TARGET_WIDTH_BYTES = 2

#: How much audio goes to the model at once. Whisper decodes a 30 s context
#: regardless, so a longer window is not slower per second of audio - but a
#: shorter one bounds how much is lost if a window fails, and keeps the word
#: offsets close to their anchor.
WINDOW_SECONDS = 25


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def list_devices() -> int:
    """What the machine can record from, and which one it would pick.

    Printed because `REFERENCE_ENVIRONMENT.md` records a real hazard on this
    workstation: NVIDIA Broadcast and Voicemeeter enumerate as inputs, and
    Broadcast's noise removal suppresses exactly the breath and creak that mark
    `cut_off` and `prolongation`. Recording through one of them would encode
    the enhancer's decisions as corpus data.
    """
    try:
        import sounddevice
    except ImportError:
        print(
            "recording needs the `record` extra: uv sync --extra record",
            file=sys.stderr,
        )
        return 2

    default_input = sounddevice.default.device[0]
    suspicious = ("broadcast", "voicemeeter", "virtual", "vb-audio", "nahimic")

    print("input devices:\n")
    for index, device in enumerate(sounddevice.query_devices()):
        if device["max_input_channels"] < 1:
            continue
        marker = "->" if index == default_input else "  "
        name = device["name"]
        warning = ""
        if any(term in name.lower() for term in suspicious):
            warning = "   <- virtual or enhanced; not for a pilot recording"
        print(f" {marker} [{index:>2}] {name}{warning}")

    print(
        "\nPick a physical microphone with --device. A virtual chain resamples and\n"
        "mixes, and an enhancer removes the breath and creak the taxonomy is about."
    )
    return 0


def record(path: Path, seconds: int, device: int | None) -> int:
    """Record to a WAV the engine will accept without conversion."""
    try:
        import sounddevice
    except ImportError:
        print(
            "recording needs the `record` extra: uv sync --extra record",
            file=sys.stderr,
        )
        return 2

    if path.exists():
        print(
            f"refused: {path} already exists. A recording is not reproducible - "
            "overwriting one is losing it. Choose another name.",
            file=sys.stderr,
        )
        return 1

    name = "default"
    if device is not None:
        name = str(sounddevice.query_devices(device)["name"])

    print(f"device   : {name}")
    print(f"format   : {TARGET_RATE_HZ} Hz, {TARGET_CHANNELS} channel, 16-bit PCM")
    print(f"duration : {seconds} s")
    print("\nspeak when the countdown ends. Ctrl-C stops early and keeps what it has.\n")
    for remaining in (3, 2, 1):
        print(f"  {remaining}...", flush=True)
        import time

        time.sleep(1)
    print("  RECORDING\n", flush=True)

    frames = seconds * TARGET_RATE_HZ
    try:
        audio = sounddevice.rec(
            frames,
            samplerate=TARGET_RATE_HZ,
            channels=TARGET_CHANNELS,
            dtype="int16",
            device=device,
        )
        sounddevice.wait()
    except KeyboardInterrupt:
        sounddevice.stop()
        print("\nstopped early; keeping what was captured")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(TARGET_CHANNELS)
        handle.setsampwidth(TARGET_WIDTH_BYTES)
        handle.setframerate(TARGET_RATE_HZ)
        handle.writeframes(audio.tobytes())

    print(f"\nwritten: {path}")
    print(
        "\nThe four audio confirmations REFERENCE_ENVIRONMENT.md gates Pilot A on:\n"
        f"  named physical device   {name}\n"
        f"  mono PCM                yes\n"
        f"  {TARGET_RATE_HZ} Hz / 16-bit        yes\n"
        "  virtual chains bypassed  <- you confirm this, not the tool"
    )
    return 0


# ---------------------------------------------------------------------------
# Running the engine over it
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Audio:
    """A recording, read once, with its declared shape checked against itself."""

    samples: bytes
    rate_hz: int
    channels: int
    width_bytes: int

    @property
    def duration_ms(self) -> int:
        frame = self.channels * self.width_bytes
        return round(len(self.samples) / frame / self.rate_hz * 1000)


def read_wav(path: Path) -> Audio:
    """Read a WAV, and refuse the shapes the engine cannot time correctly.

    Refused here rather than converted, because a silent resample is how a
    recording ends up on a clock nobody chose. `AudioWindow` would refuse it
    anyway; catching it at the file boundary names the file.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        samples = handle.readframes(handle.getnframes())

    if channels != TARGET_CHANNELS:
        raise SystemExit(
            f"{path.name} has {channels} channels; the engine decodes mono. A stereo "
            "payload timed as mono counts twice the frames it holds and agrees with a "
            "declaration of twice its real length."
        )
    if width != TARGET_WIDTH_BYTES:
        raise SystemExit(f"{path.name} is {width * 8}-bit; the engine decodes 16-bit PCM.")
    if rate != TARGET_RATE_HZ:
        raise SystemExit(
            f"{path.name} is {rate} Hz; the engine negotiates {TARGET_RATE_HZ} Hz. "
            "Re-record at 16 kHz rather than resampling here: the rate is what turns "
            "a byte count into a duration, and a window timed at a rate it was not "
            "decoded at scales every boundary by the ratio between the two."
        )
    return Audio(samples, rate, channels, width)


def windows(audio: Audio) -> list[AudioWindow]:
    """Cut the recording into windows the runtime will accept."""
    frame = audio.channels * audio.width_bytes
    per_window = WINDOW_SECONDS * audio.rate_hz * frame
    cut: list[AudioWindow] = []
    position_ms = 0
    for start in range(0, len(audio.samples), per_window):
        chunk = audio.samples[start : start + per_window]
        if len(chunk) < frame:
            break
        duration_ms = round(len(chunk) / frame / audio.rate_hz * 1000)
        cut.append(
            AudioWindow(
                session_position_ms=position_ms,
                duration_ms=duration_ms,
                sample_rate_hz=audio.rate_hz,
                samples=chunk,
                is_final_window=start + per_window >= len(audio.samples),
                channel_count=audio.channels,
                sample_width_bytes=audio.width_bytes,
            )
        )
        position_ms += duration_ms
    return cut


async def run(path: Path, as_json: bool) -> int:
    from evidence_engine.adapters.outbound.model_runtime.whisper import (
        WhisperRuntimeUnavailable,
        WhisperSpeechRuntime,
    )
    from evidence_engine.bootstrap.container import default_configuration

    if not path.exists():
        print(f"no such recording: {path}", file=sys.stderr)
        return 1

    audio = read_wav(path)
    configuration = default_configuration()
    run_id = RunId(f"run-{path.stem[:24]}")

    if not as_json:
        print(f"recording : {path.name}  {audio.duration_ms / 1000:.1f} s")
        print(f"format    : {audio.rate_hz} Hz, {audio.channels} ch, {audio.width_bytes * 8}-bit")
        print("loading the pinned checkpoint (first run downloads ~3 GB)...", flush=True)

    try:
        runtime = WhisperSpeechRuntime.load()
    except WhisperRuntimeUnavailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    assembler = SpeechAssembler(run_id, configuration, Calibrator())
    tokens: list[WordToken] = []

    cut = windows(audio)
    for index, window in enumerate(cut, start=1):
        if not as_json:
            print(f"  window {index}/{len(cut)}  {window.duration_ms / 1000:.1f} s", flush=True)
        try:
            result = await runtime.transcribe(window)
        except MisdeclaredAudioWindow as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        tokens.extend(assembler.assemble(result).tokens)

    provenance = Provenance(
        modality=Modality.AUDIO,
        model_version=runtime.model_version,
        taxonomy_version=configuration.taxonomy_version,
        configuration=configuration.id,
        evidence_ref=EvidenceRef(f"audio:{run_id.value}"),
    )
    pauses = silent_pauses(
        tuple(tokens),
        threshold_ms=configuration.silence_threshold_ms,
        provenance=provenance,
        run_id=run_id,
    )

    transcript = " ".join(token.raw_text for token in tokens)
    speaking_ms = sum(token.interval.end.ms - token.interval.start.ms for token in tokens)
    silent_ms = sum(pause.interval.end.ms - pause.interval.start.ms for pause in pauses)

    if as_json:
        print(
            json.dumps(
                {
                    "recording": path.name,
                    "duration_ms": audio.duration_ms,
                    "model_version": str(runtime.model_version),
                    "environment_is_pinned": runtime.environment_is_pinned,
                    "taxonomy_version": str(configuration.taxonomy_version),
                    "silence_threshold_ms": configuration.silence_threshold_ms,
                    "transcript": transcript,
                    "words": [
                        {
                            "text": token.raw_text,
                            "start_ms": token.interval.start.ms,
                            "end_ms": token.interval.end.ms,
                        }
                        for token in tokens
                    ],
                    "silent_pauses": [
                        {
                            "start_ms": pause.interval.start.ms,
                            "end_ms": pause.interval.end.ms,
                            "duration_ms": pause.interval.end.ms - pause.interval.start.ms,
                        }
                        for pause in pauses
                    ],
                    "disfluency_events": 0,
                    "not_measured": [
                        "filled_pause",
                        "lexical_filler",
                        "repetition",
                        "false_start",
                        "self_repair",
                        "cut_off",
                        "prolongation",
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"\n{'=' * 72}")
    print("TRANSCRIPT")
    print("=" * 72)
    print(transcript or "(nothing recognised)")

    print(f"\n{'=' * 72}")
    print("WHAT THE ENGINE DERIVED")
    print("=" * 72)
    print(f"  words                {len(tokens)}")
    unaligned = sum(1 for token in tokens if not token.is_timed)
    if unaligned:
        print(
            f"  of those, unplaced   {unaligned}  - heard, and the alignment heads\n"
            "                       returned no interval. They are in the transcript\n"
            "                       above; the pause and rate figures below were\n"
            "                       computed without them."
        )
    print(f"  recording length     {audio.duration_ms / 1000:.1f} s")
    print(f"  time inside a word   {speaking_ms / 1000:.1f} s")
    print(f"  time in a long pause {silent_ms / 1000:.1f} s")
    if audio.duration_ms:
        print(f"  words per minute     {len(tokens) / (audio.duration_ms / 60000):.0f}")

    print(f"\n  silent pauses over {configuration.silence_threshold_ms} ms: {len(pauses)}")
    for pause in pauses:
        start = pause.interval.start.ms
        length = pause.interval.end.ms - start
        print(f"    {start / 1000:>7.2f} s   {length:>5} ms")

    print(f"\n{'=' * 72}")
    print("WHAT IT DID NOT MEASURE, AND WHY")
    print("=" * 72)
    print(
        "  disfluency events    unavailable  (detector_not_deployed)\n"
        "                       Phase 4. Nothing here looks for a filled pause,\n"
        "                       a repetition, a false start or a prolongation.\n"
        "  prosody              unavailable  (detector_not_deployed)\n"
        "                       Phase 4. No pitch, intensity or rate estimator.\n"
        "  visual events        unavailable  (detector_not_deployed)\n"
        "                       Phase 5. Nothing here looks at video at all.\n"
        "\n  These used to print `0`, with the explanation beside them. That was\n"
        "  wrong in the specific way FR-025 exists to prevent: `0` means the\n"
        "  detector ran and found none, and a reader scanning a column of\n"
        "  numbers reads the number, not the sentence next to it. A clean\n"
        "  delivery and an unmeasured one are not the same result.\n"
        "\n  Silent pauses are the exception, and the reason is worth stating:\n"
        "  they are *derived* from the gaps between word boundaries using the\n"
        "  versioned threshold, not detected by a model. That is why they are\n"
        "  the one taxonomy class that produces a number today - and why their\n"
        "  count above is a measurement rather than an absence."
    )

    print(f"\n{'=' * 72}")
    print("PROVENANCE")
    print("=" * 72)
    print(f"  model            {runtime.model_version.value}")
    print(f"  taxonomy         {configuration.taxonomy_version}")
    print(f"  configuration    {configuration.id.value}")
    print(f"  environment      {'pinned' if runtime.environment_is_pinned else 'NOT PINNED'}")
    print(
        "\n  The executable environment - backend, container digest, Torch and CUDA\n"
        "  build - freezes in Phase 3. Until then a number from this run is\n"
        "  reproducible only by whoever ran it, and does not belong in a document."
    )
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="present",
        description="Record a presentation and run the evidence engine over it.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("devices", help="list microphones and flag the virtual ones")

    recorder = sub.add_parser("record", help="record a presentation to a WAV")
    recorder.add_argument("-o", "--output", type=Path, required=True)
    recorder.add_argument("--seconds", type=int, default=60)
    recorder.add_argument(
        "--device", type=int, default=None, help="index from `present.py devices`"
    )

    runner = sub.add_parser("run", help="run the engine over a recording")
    runner.add_argument("recording", type=Path)
    runner.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.command == "devices":
        return list_devices()
    if args.command == "record":
        return record(args.output, args.seconds, args.device)
    if args.command == "run":
        return asyncio.run(run(args.recording, args.as_json))

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
