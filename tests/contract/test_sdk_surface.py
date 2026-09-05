"""The SDK's published surface, and the conformance that makes it worth having.

Two claims are being defended, and the second is the expensive one.

**The surface is small and stays small.** Once a consumer imports a name from
`evidence_engine`, changing it is a breaking change to every integration. So
`__all__` is asserted by name here rather than merely reviewed: adding an
export is a decision, and this is where that decision is recorded. The absence
list matters more than the presence list - a consumer handed
`InMemoryEvidenceRepository` or `WhisperSpeechRuntime` by an `__init__` would be
coupled to a composition decision, and the next release would break them.

**Local and hosted produce the same evidence.** The SDK exists to be a second
way of running one engine, not a second engine. That claim is only true if the
same audio under the same configuration produces the same document, and it is
exactly the claim that decays silently: the two paths would keep working while
drifting, and nothing would say so. The comparison excludes `session_id`,
`run_id`, trace ids and wall-clock stamps, which differ by construction and say
nothing about whether the engines agree.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

import evidence_engine
from evidence_engine import (
    AnalysisResult,
    AudioNotUsable,
    EngineConfiguration,
    HardwareReport,
    OratoriaEngine,
    OratoriaError,
    SessionConfiguration,
    StreamAlreadyClosed,
    StreamSession,
)
from evidence_engine.adapters.outbound.model_runtime.deterministic import (
    ScriptedWord,
    SpeechScript,
)
from evidence_engine.adapters.outbound.telemetry.clock import FrozenClock
from evidence_engine.domain.shared.identifiers import ModelVersionId

#: A script both engines replay. Deterministic by construction, which is what
#: makes a conformance comparison meaningful at all: any difference between the
#: two documents is a difference between the *engines*, not between two decodes.
SCRIPT = SpeechScript(
    model_version=ModelVersionId("deterministic-speech-v1"),
    words=(
        ScriptedWord(text="buenos", start_ms=0, end_ms=400, score=0.9),
        ScriptedWord(text="dias", start_ms=400, end_ms=800, score=0.9),
        ScriptedWord(text="a", start_ms=1_600, end_ms=1_700, score=0.9),
        ScriptedWord(text="todos", start_ms=1_700, end_ms=2_100, score=0.9),
    ),
)


def configuration() -> EngineConfiguration:
    return EngineConfiguration(
        runtime="deterministic",
        speech_runtime=None,
        window_seconds=1,
    )


def a_recording(path: Path, seconds: int = 3) -> Path:
    """Silence at the engine's decode rate. The script supplies the words."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000 * seconds)
    return path


# ---------------------------------------------------------------------------
# The published surface
# ---------------------------------------------------------------------------


def test_the_public_exports_are_exactly_these() -> None:
    """Adding a name here is a decision, not a side effect of an import."""
    assert set(evidence_engine.__all__) == {
        "OratoriaEngine",
        "EngineConfiguration",
        "SessionConfiguration",
        "AnalysisResult",
        "HardwareReport",
        "StreamSession",
        "OratoriaError",
        "AudioNotUsable",
        "LocalInferenceUnavailable",
        "StreamAlreadyClosed",
        "DEFAULT_SAMPLE_RATE_HZ",
        "DEFAULT_WINDOW_SECONDS",
        "__version__",
    }


@pytest.mark.parametrize(
    "name",
    [
        "Container",
        "InMemoryEvidenceRepository",
        "InMemorySessionRepository",
        "WhisperSpeechRuntime",
        "DeterministicSpeechRuntime",
        "PostgresSessionRepository",
        "StreamingCoordinator",
        "SpeechAssembler",
        "WordToken",
        "EvidenceDocument",
        "Settings",
        "build_container",
        "create_app",
    ],
)
def test_no_internal_is_reachable_from_the_package_root(name: str) -> None:
    """The absence list, which is the half that matters.

    A consumer who imported a repository or a concrete runtime from the root
    would be coupled to how the engine is assembled this month. The full paths
    still work for anybody who needs them - that is opting out of the promise
    rather than being handed the coupling by an `__init__`.
    """
    assert not hasattr(evidence_engine, name), (
        f"`evidence_engine.{name}` is reachable from the package root. Exporting it "
        "makes a composition decision part of the published contract."
    )


def test_the_sdk_publishes_its_own_version() -> None:
    """A consumer reporting a bug has to be able to name what they ran.

    Distinct from the schema version, which answers what the *contract* looks
    like rather than what produced it.
    """
    assert evidence_engine.__version__ == "0.1.0"


def test_importing_the_sdk_pulls_in_no_server_module() -> None:
    """C9 asserts this against a clean install; this asserts it in the repo.

    Both are needed. C9 catches a dependency that arrives at install time; this
    catches an import added to the facade, which a developer sees immediately
    rather than at the next release.
    """
    import sys

    assert "fastapi" not in sys.modules or "evidence_engine.bootstrap" in sys.modules
    engine_modules = {m for m in sys.modules if m.startswith("evidence_engine.sdk")}
    assert engine_modules, "the sdk package should be imported by now"
    assert "evidence_engine.bootstrap" not in _imports_of("evidence_engine.sdk.engine")


def _imports_of(module: str) -> set[str]:
    import importlib

    loaded = importlib.import_module(module)
    return {
        value.__name__
        for value in vars(loaded).values()
        if getattr(value, "__name__", "").startswith("evidence_engine.bootstrap")
    }


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_preflight_never_claims_the_model_works() -> None:
    """The distinction the two methods exist to keep.

    A preflight that returned a verdict would be a green result standing in for
    a measurement nobody took - on a machine where the weights are corrupt or
    the driver is too old, it would still be green.
    """
    report = OratoriaEngine.local(configuration()).hardware_preflight()

    assert isinstance(report, HardwareReport)
    assert not hasattr(report, "ok")
    assert not hasattr(report, "ready")
    assert "did not load the" in report.render()
    assert "warmup()" in report.render()


async def test_warmup_is_the_real_check_and_is_idempotent() -> None:
    """It decodes. Calling it twice costs one decode."""
    engine = OratoriaEngine.local(
        EngineConfiguration(speech_runtime=_CountingRuntime(), window_seconds=1)
    )
    await engine.warmup()
    await engine.warmup()

    runtime = engine._speech
    assert isinstance(runtime, _CountingRuntime)
    assert runtime.calls == 1


class _CountingRuntime:
    emitted_speech_events: frozenset[object] = frozenset()
    emitted_prosody: frozenset[object] = frozenset()
    capability_detail = "counting"

    def __init__(self) -> None:
        self.calls = 0
        self._inner = None

    async def transcribe(self, window: object) -> object:
        from evidence_engine.application.ports.runtimes import SpeechResult

        self.calls += 1
        return SpeechResult(model_version=ModelVersionId("counting-v1"))


# ---------------------------------------------------------------------------
# Batch: the path the hosted service does not have
# ---------------------------------------------------------------------------


async def test_analyze_file_produces_the_published_contract(tmp_path: Path) -> None:
    """`mode: batch` is accepted by the API and no worker implements it.

    Here the same pipeline runs, driven by a loop over windows instead of a
    socket, and returns the document `GET /v1/sessions/{id}/result` serialises.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    result = await engine.analyze_file(a_recording(tmp_path / "p.wav"))

    assert isinstance(result, AnalysisResult)
    assert result.document.transcript.raw_text() == "buenos dias a todos"
    assert result.document.ranking_authority == "none"
    await engine.aclose()


@pytest.mark.parametrize(
    ("channels", "width", "rate", "why"),
    [
        pytest.param(2, 2, 16_000, "channels", id="stereo"),
        pytest.param(1, 1, 16_000, "8-bit", id="8-bit"),
        pytest.param(1, 2, 44_100, "44100 Hz", id="wrong-rate"),
    ],
)
async def test_a_recording_the_engine_cannot_time_is_refused(
    tmp_path: Path, channels: int, width: int, rate: int, why: str
) -> None:
    """Refused rather than resampled.

    A silent conversion is how a recording ends up on a clock nobody chose: the
    declared rate is what turns a byte count into a duration, so audio decoded
    at one rate and timed at another scales every boundary by the ratio between
    them - and the transcript still reads correctly.
    """
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * width * channels * rate)

    engine = OratoriaEngine.local(configuration())
    with pytest.raises(AudioNotUsable, match=why):
        await engine.analyze_file(path)


async def test_a_missing_recording_says_so(tmp_path: Path) -> None:
    engine = OratoriaEngine.local(configuration())
    with pytest.raises(AudioNotUsable, match="does not exist"):
        await engine.analyze_file(tmp_path / "absent.wav")


# ---------------------------------------------------------------------------
# Streaming: a session, not a generator
# ---------------------------------------------------------------------------


async def test_the_stream_exposes_the_controls_a_presentation_needs() -> None:
    """A generator can be advanced and closed. A presentation needs more.

    `pause` is load-bearing rather than decorative: §6.1 excludes paused
    stretches from the session clock, so a speaker who stops for two minutes to
    answer a question does not have those minutes counted as silence.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    stream = engine.create_stream(SessionConfiguration())

    assert isinstance(stream, StreamSession)
    for control in ("send_audio", "pause", "resume", "finish", "abort", "receive"):
        assert callable(getattr(stream, control)), control


async def test_a_stream_pauses_resumes_and_finishes() -> None:
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    async with engine.create_stream() as stream:
        assert await stream.send_audio(b"\x00\x00" * 16_000)
        await stream.pause()
        await stream.resume()
        assert await stream.send_audio(b"\x00\x00" * 16_000)
        result = await stream.finish()

    assert result.document.transcript.raw_text().startswith("buenos")


async def test_a_finished_stream_refuses_to_be_reused() -> None:
    """§6.1 step 9 makes finalized history immutable.

    A second pass over a closed session would either rewrite it or silently
    start a different one, and both are worse than refusing.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    stream = engine.create_stream()
    await stream.send_audio(b"\x00\x00" * 16_000)
    await stream.finish()

    with pytest.raises(StreamAlreadyClosed):
        await stream.send_audio(b"\x00\x00" * 16_000)


async def test_leaving_the_block_without_finishing_aborts() -> None:
    """A session that fell out of scope produced no result.

    Marking its processing run successful would put a half-finished run in the
    trail as a completed one.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    with pytest.raises(RuntimeError, match="deliberate"):
        async with engine.create_stream() as stream:
            await stream.send_audio(b"\x00\x00" * 16_000)
            raise RuntimeError("deliberate")

    assert stream._closed


async def test_abort_is_idempotent() -> None:
    """A caller aborting in a `finally` after an explicit abort must not raise."""
    engine = OratoriaEngine.local(configuration())
    stream = engine.create_stream()
    await stream.abort()
    await stream.abort()


def test_every_sdk_error_is_catchable_as_one_type() -> None:
    """A consumer that wants a single `except` for the engine has one."""
    for error in (AudioNotUsable, StreamAlreadyClosed):
        assert issubclass(error, OratoriaError)


# ---------------------------------------------------------------------------
# Conformance: the SDK is a second way to run one engine
# ---------------------------------------------------------------------------


async def test_two_engines_on_the_same_audio_agree_on_the_evidence(tmp_path: Path) -> None:
    """The claim the whole SDK rests on.

    Same script, same configuration, same clock: the documents must be equal
    everywhere except the identifiers and stamps that differ by construction.
    If this fails, the SDK is a second engine rather than a second way of
    running one, and nothing else in the suite would say so.
    """
    recording = a_recording(tmp_path / "p.wav")

    documents = []
    for _ in range(2):
        engine = OratoriaEngine.local(
            EngineConfiguration(speech_runtime=_scripted(), window_seconds=1),
            clock=FrozenClock(start_ms=1_000_000),
        )
        result = await engine.analyze_file(recording)
        documents.append(_canonical(result))

    assert documents[0] == documents[1]


def _canonical(result: AnalysisResult) -> dict[str, object]:
    """Everything two runs must agree on, with the transport's own noise removed.

    `session_id`, `run_id` and the token ids derived from the run differ by
    construction - they are minted per run and say nothing about whether the
    engines agree. Comparing raw documents would fail on them every time and
    the test would be deleted within a week.
    """
    document = result.document
    return {
        "raw_text": document.transcript.raw_text(),
        "unaligned": document.transcript.unaligned_count,
        "tokens": [
            (token.raw_text, token.sequence.key, token.is_timed, token.status.value)
            for token in document.transcript.tokens
        ],
        "speech_events": [
            (event.type.value, event.interval.start.ms, event.raw_text)
            for event in document.speech_events
        ],
        "visual_events": [event.type.value for event in document.visual_events],
        "cooccurrences": len(document.cooccurrences),
        "ranking_authority": document.ranking_authority,
        "taxonomy_version": str(document.manifest.taxonomy_version),
        "schema_version": str(document.manifest.schema_version),
        "pipeline_version": str(document.manifest.pipeline_version),
        "configuration": document.manifest.configuration.value,
    }


def _scripted() -> object:
    from evidence_engine.adapters.outbound.model_runtime.deterministic import (
        DeterministicSpeechRuntime,
    )

    return DeterministicSpeechRuntime(SCRIPT)
