"""The SDK's published surface, and the conformance that makes it worth having.

Two claims are being defended, and the second is the expensive one.

**The surface is small and stays small.** Once a consumer imports a name from
`evidence_engine`, changing it is a breaking change to every integration. So
`__all__` is asserted by name here rather than merely reviewed: adding an
export is a decision, and this is where that decision is recorded. The absence
list matters more than the presence list - a consumer handed
`InMemoryEvidenceRepository` or `WhisperSpeechRuntime` by an `__init__` would be
coupled to a composition decision, and the next release would break them.

**The local engine is deterministic.** Same audio, same configuration, same
clock, same evidence - NFR-015's requirement, and the half of conformance that
can be checked today. It is deliberately *not* called a conformance test: it
compares one implementation with itself. Local/remote equivalence needs
`OratoriaClient`, which does not exist; when it does, `_canonical` is the
comparison it has to pass, and the exclusions are already right - `session_id`,
`run_id` and wall stamps differ by construction and say nothing about whether
two engines agree.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

import evidence_engine
from evidence_engine import (
    AlignmentUnavailable,
    AnalysisResult,
    AudioNotUsable,
    ConfidenceUnavailable,
    EngineConfiguration,
    EngineNotWarmed,
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
from evidence_engine.domain.shared.provenance import ModelRole

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


def test_the_engine_and_the_client_share_the_surface_oratoria_drives() -> None:
    """OratorIA's adapter drives both through one structural protocol: an
    engine with `warmup`/`create_stream`/`aclose`/`contributions`, a session
    with `send_audio`/`receive`/`pending`/`finish`/`abort`. A member present
    on one and missing on the other is a session that dies on the path
    nobody tested against the real class - `contributions` was exactly that
    on the embedded engine until this test existed."""
    from evidence_engine.sdk.client import OratoriaClient, RemoteStreamSession

    for member in ("warmup", "create_stream", "aclose", "contributions"):
        assert hasattr(OratoriaEngine, member), f"OratoriaEngine lacks {member}"
        assert hasattr(OratoriaClient, member), f"OratoriaClient lacks {member}"
    for member in ("send_audio", "receive", "pending", "finish", "abort"):
        assert hasattr(StreamSession, member), f"StreamSession lacks {member}"
        assert hasattr(RemoteStreamSession, member), f"RemoteStreamSession lacks {member}"


def test_the_public_exports_are_exactly_these() -> None:
    """Adding a name here is a decision, not a side effect of an import."""
    assert set(evidence_engine.__all__) == {
        # The facade, its remote twin (ADR-011's deferred client), and what
        # configures them
        "OratoriaEngine",
        "OratoriaClient",
        "EngineConfiguration",
        "SessionConfiguration",
        "HardwareReport",
        "StreamSession",
        # The result types. These are the SDK's own, not the domain's: an
        # `AnalysisResult` carrying an `EvidenceDocument` passed the absence
        # test below on the letter while coupling every consumer to the
        # domain's shape through a field.
        "AnalysisResult",
        "Evidence",
        "Manifest",
        "Transcript",
        "Word",
        "SpeechEvent",
        "VisualEvent",
        "ProsodyReading",
        # Absence, in the shapes a consumer has to narrow on
        "TimedPlacement",
        "AlignmentUnavailable",
        "Confidence",
        "ConfidenceUnavailable",
        "Value",
        "ValueUnavailable",
        # Errors
        "OratoriaError",
        "AudioNotUsable",
        "EngineNotWarmed",
        "LocalInferenceUnavailable",
        "RemoteChunkLost",
        "RemoteEngineUnavailable",
        "StreamAlreadyClosed",
        # Constants and version
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
        # OratoriaClient's own internals: the session it returns and the
        # transport seam it is built on. Both are reachable at their full
        # path (`evidence_engine.sdk.client.RemoteStreamSession`) for anyone
        # implementing a custom `Transport`; neither is part of the promise.
        "RemoteStreamSession",
        "Transport",
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

    @property
    def contributions(self) -> dict[ModelRole, ModelVersionId]:
        return {ModelRole.RECOGNISER: ModelVersionId("counting-v1")}

    def __init__(self) -> None:
        self.calls = 0
        self._inner = None

    async def transcribe(self, window: object) -> object:
        from evidence_engine.application.ports.runtimes import SpeechResult

        self.calls += 1
        return SpeechResult(contributions=self.contributions)


# ---------------------------------------------------------------------------
# Batch: the path the hosted service does not have
# ---------------------------------------------------------------------------


async def test_analyze_file_produces_the_published_contract(tmp_path: Path) -> None:
    """`mode: batch` is accepted by the API and no worker implements it.

    Here the same pipeline runs, driven by a loop over windows instead of a
    socket, and returns the document `GET /v1/sessions/{id}/result` serialises.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    await engine.warmup()
    result = await engine.analyze_file(a_recording(tmp_path / "p.wav"))

    assert isinstance(result, AnalysisResult)
    assert result.evidence.transcript.raw_text == "buenos dias a todos"
    assert result.evidence.ranking_authority == "none"
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
    await engine.warmup()
    with pytest.raises(AudioNotUsable, match=why):
        await engine.analyze_file(path)


async def test_a_missing_recording_says_so(tmp_path: Path) -> None:
    engine = OratoriaEngine.local(configuration())
    await engine.warmup()
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
    await engine.warmup()
    stream = engine.create_stream(SessionConfiguration())

    assert isinstance(stream, StreamSession)
    for control in ("send_audio", "pause", "resume", "finish", "abort", "receive"):
        assert callable(getattr(stream, control)), control


async def test_a_stream_pauses_resumes_and_finishes() -> None:
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    await engine.warmup()
    async with engine.create_stream() as stream:
        assert await stream.send_audio(b"\x00\x00" * 16_000)
        await stream.pause()
        await stream.resume()
        assert await stream.send_audio(b"\x00\x00" * 16_000)
        result = await stream.finish()

    assert result.evidence.transcript.raw_text.startswith("buenos")
    # S2: the embedded engine reports the same two progress fields the hosted
    # service's `session.completed` carries - two 1 s windows were sent.
    assert result.captured_ms == 2_000
    assert result.finalized_through_ms is not None


async def test_a_finished_stream_refuses_to_be_reused() -> None:
    """§6.1 step 9 makes finalized history immutable.

    A second pass over a closed session would either rewrite it or silently
    start a different one, and both are worse than refusing.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    await engine.warmup()
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
    await engine.warmup()
    with pytest.raises(RuntimeError, match="deliberate"):
        async with engine.create_stream() as stream:
            await stream.send_audio(b"\x00\x00" * 16_000)
            raise RuntimeError("deliberate")

    assert stream._closed


async def test_abort_is_idempotent() -> None:
    """A caller aborting in a `finally` after an explicit abort must not raise."""
    engine = OratoriaEngine.local(configuration())
    await engine.warmup()
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


async def test_the_local_engine_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Two **local** engines, same input, same evidence.

    Named for what it proves. It was called a conformance test and it is not:
    it compares one implementation with itself, which demonstrates determinism
    - NFR-015's "equivalent output from equivalent input" - and says nothing
    about whether the hosted service agrees.

    Real local/remote conformance needs `OratoriaClient`, which does not exist.
    When it does, `_canonical` below is the comparison it has to pass, and this
    test is the half of it that already works.
    """
    recording = a_recording(tmp_path / "p.wav")

    documents = []
    for _ in range(2):
        engine = OratoriaEngine.local(
            EngineConfiguration(speech_runtime=_scripted(), window_seconds=1),
            clock=FrozenClock(start_ms=1_000_000),
        )
        await engine.warmup()
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
    evidence = result.evidence
    return {
        "raw_text": evidence.transcript.raw_text,
        "unaligned": evidence.transcript.unaligned_count,
        "words": [
            (word.text, word.sequence, word.is_timed, word.status)
            for word in evidence.transcript.words
        ],
        "speech_events": [(e.type, e.start_ms, e.raw_text) for e in evidence.speech_events],
        "visual_events": [e.type for e in evidence.visual_events],
        "ranking_authority": evidence.ranking_authority,
        "manifest": (
            evidence.manifest.taxonomy_version,
            evidence.manifest.schema_version,
            evidence.manifest.pipeline_version,
            evidence.manifest.configuration_id,
        ),
    }


def _scripted() -> object:
    from evidence_engine.adapters.outbound.model_runtime.deterministic import (
        DeterministicSpeechRuntime,
    )

    return DeterministicSpeechRuntime(SCRIPT)


# ---------------------------------------------------------------------------
# The three blockers a review found in the first cut of this facade
# ---------------------------------------------------------------------------


def test_constructing_an_engine_loads_no_model() -> None:
    """`local()` used to call `build_speech_runtime()`, which loads 3 GB.

    That made `hardware_preflight()` - the method whose entire purpose is to be
    asked *before* committing to that download - reachable only afterwards. The
    docstring said construction loaded no model while the line above it did.
    """
    engine = OratoriaEngine.local(EngineConfiguration(runtime="baseline_whisper"))

    assert engine._speech is None
    # And the preflight is answerable on a machine that could never load it.
    assert isinstance(engine.hardware_preflight(), HardwareReport)


def test_a_supplied_runtime_is_wired_immediately() -> None:
    """There is nothing to defer when the caller already built it."""
    runtime = _scripted()
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=runtime))
    assert engine._speech is runtime


@pytest.mark.parametrize("call", ["analyze_file", "create_stream"])
async def test_analysing_before_warmup_is_refused(tmp_path: Path, call: str) -> None:
    """Loading implicitly on first use would hide a 3 GB download in a decode.

    It would also make `hardware_preflight()` advice nobody has to take: the
    model would arrive whether or not the machine could hold it.
    """
    engine = OratoriaEngine.local(configuration())

    with pytest.raises(EngineNotWarmed, match="warmup"):
        if call == "analyze_file":
            await engine.analyze_file(a_recording(tmp_path / "p.wav"))
        else:
            engine.create_stream()


async def test_closing_releases_the_runtime() -> None:
    """`aclose()` was a no-op with a docstring saying there was nothing to release.

    True of the in-memory adapters, false of the model: a warmed
    `baseline_whisper` engine holds a CUDA context and about 4.2 GiB, so a
    consumer who closed one and built another on an 8 GB card ran out of memory
    on a machine that should have fitted both.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted()))
    await engine.warmup()
    assert engine._speech is not None

    await engine.aclose()

    assert engine._speech is None
    # And it is no longer usable without warming again, rather than silently
    # working on a runtime that was meant to be released.
    with pytest.raises(EngineNotWarmed):
        engine.create_stream()


async def test_closing_twice_is_safe() -> None:
    """A consumer closing in a `finally` after an explicit close must not raise."""
    engine = OratoriaEngine.local(configuration())
    await engine.aclose()
    await engine.aclose()


# ---------------------------------------------------------------------------
# The public result types
# ---------------------------------------------------------------------------


async def test_the_result_carries_no_domain_entity(tmp_path: Path) -> None:
    """The leak, as a regression.

    `AnalysisResult.document` was an `EvidenceDocument`. The absence test above
    passed - the class is not at the package root - while every consumer
    reaching through the field was coupled to the domain anyway.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_scripted(), window_seconds=1))
    await engine.warmup()
    result = await engine.analyze_file(a_recording(tmp_path / "p.wav"))

    assert not hasattr(result, "document")
    assert type(result.evidence).__module__.startswith("evidence_engine.sdk")
    assert type(result.transcript).__module__.startswith("evidence_engine.sdk")
    for word in result.transcript.words:
        assert type(word).__module__.startswith("evidence_engine.sdk")
        assert type(word.placement).__module__.startswith("evidence_engine.sdk")
        assert type(word.confidence).__module__.startswith("evidence_engine.sdk")


async def test_absence_survives_the_translation_to_public_types(tmp_path: Path) -> None:
    """A DTO layer is where `Measured | Unavailable` usually dies.

    The obvious translation writes `confidence: float | None`, and by the time
    the number reaches a reader it says the model scored the word zero. Both
    unions are preserved as separate types with no numeric attribute on the
    absent side.
    """
    engine = OratoriaEngine.local(EngineConfiguration(speech_runtime=_unscored(), window_seconds=1))
    await engine.warmup()
    result = await engine.analyze_file(a_recording(tmp_path / "p.wav"))

    words = result.transcript.words
    assert words, "the script should have produced words"

    unscored = [w for w in words if isinstance(w.confidence, ConfidenceUnavailable)]
    assert unscored, "the runtime reported no posterior; that must survive"
    assert unscored[0].confidence.reason == "posterior_not_reported"
    assert not hasattr(unscored[0].confidence, "value")

    unplaced = [w for w in words if isinstance(w.placement, AlignmentUnavailable)]
    assert unplaced, "the runtime emitted an unaligned word; that must survive"
    assert not hasattr(unplaced[0].placement, "start_ms")
    assert unplaced[0].text in result.transcript.raw_text
    assert result.transcript.unaligned_count == len(unplaced)


def _unscored() -> object:
    """A runtime that reports no posterior and one unplaceable word.

    Both are what the Whisper baseline actually does, reproduced without a GPU
    so the translation is tested rather than the model.
    """
    from evidence_engine.application.ports.runtimes import (
        AudioWindow,
        SpeechResult,
        TimedWordHypothesis,
        UntimedWordHypothesis,
    )
    from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable

    absent = Unavailable(reason=UnavailabilityReason.POSTERIOR_NOT_REPORTED)

    class _Runtime:
        emitted_speech_events: frozenset[object] = frozenset()
        emitted_prosody: frozenset[object] = frozenset()
        capability_detail = "no posterior, one unaligned word"

        @property
        def contributions(self) -> dict[ModelRole, ModelVersionId]:
            return {ModelRole.RECOGNISER: ModelVersionId("unscored-v1")}

        async def transcribe(self, window: AudioWindow) -> object:
            # Re-emits the same two words on every call, because a streaming
            # runtime emits its whole *active region* rather than only what is
            # new: `Transcript.with_provisional` replaces the revisable tail
            # wholesale, so a runtime reporting only new words silently drops
            # everything not yet finalized. The first version of this fake did
            # exactly that and the words vanished - the same failure
            # `deterministic.py` documents from the first end-to-end run.
            #
            # Filed under the window it was handed, as the port requires and
            # as `deterministic.py` does when it re-emits: the coordinator
            # refuses a result reported under a window it never ingested. The
            # second version of this fake reported 0 for every window.
            return SpeechResult(
                contributions=self.contributions,
                window_position_ms=window.session_position_ms,
                words=(
                    TimedWordHypothesis(
                        raw_text="hola", start_ms=0, end_ms=300, score=absent, index=0
                    ),
                    UntimedWordHypothesis(raw_text="mundo", score=absent, index=1),
                ),
                # Nothing declared stable. A runtime that finalizes a stretch
                # and then re-emits it is contradicting itself, and the
                # transcript refuses the second pass rather than rewriting
                # finalized history (§6.1 step 9). The engine caught this
                # fake doing it.
                stable_through_ms=0,
            )

    return _Runtime()
