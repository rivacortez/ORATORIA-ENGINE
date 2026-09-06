"""Provenance is attributed by component, not by modality (NFR-014, QA-03).

Seven findings from one review, all confirmed against source, all traceable
to one decision: the unit of attribution was the modality. Six of them are
regressions here, each named for the defect it would catch if reintroduced.

- **F1/F3** The recogniser was unattributed. Tokens carried no provenance and
  the manifest read events only, so a run that produced five recognised words
  and no disfluency recorded ``models: {}``.
- **F2** Silent pauses borrowed provenance from an event. With a runtime that
  emits none - the Whisper baseline - that was circular, and the one taxonomy
  class the README said "works today" was never derived on the streaming path.
- **F4** ``Mapping[Modality, ...]`` with ``setdefault``: one audio model, and the
  second silently dropped. QA-03 requires a classifier canaried beside a fixed
  recogniser, which that manifest could not even write down.
- **F5** One ``model_version`` per result, stamped on every hypothesis. A
  detector's events and a prosody estimator's readings were attributed to
  whichever model the result named.
- **F6** ``ProcessingRun`` recorded no model, so a run that failed before its
  first window left no trace of what it had attempted.
- **QA-03** The registry itself was keyed by modality: ``active_for(AUDIO)``.

The last test drives all of it through the real transport, because §4.14 of
the build record is about exactly this: a union added at one layer has to be
exercised through every layer that consumes it, in one test, or the layers
that were never exercised keep the old assumption.
"""

from __future__ import annotations

import wave
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evidence_engine import EngineConfiguration, OratoriaEngine
from evidence_engine.adapters.outbound.persistence.configuration import InMemoryModelRegistry
from evidence_engine.application.commands.complete_session import _models_used
from evidence_engine.application.ports.platform import ApprovalState, ModelVersion
from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    ProsodyHypothesis,
    SpeechEventHypothesis,
    SpeechResult,
    TimedWordHypothesis,
)
from evidence_engine.application.services.calibration import Calibrator
from evidence_engine.application.services.speech_assembly import SpeechAssembler
from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.identifiers import (
    ConfigurationSnapshotId,
    EventId,
    EvidenceRef,
    ModelVersionId,
    RunId,
)
from evidence_engine.domain.shared.provenance import (
    Modality,
    ModelRole,
    Provenance,
    ProvenanceViolation,
)
from evidence_engine.domain.shared.taxonomy import (
    TAXONOMY_VERSION,
    ProsodicIndicator,
    SpeechEventType,
)
from evidence_engine.domain.shared.timeline import Interval
from evidence_engine.domain.speech_events.events import SpeechEvent
from evidence_engine.domain.transcript import transcript as transcript_module
from tests.contract.test_phase2_exit_criterion import CREATE_BODY, _stream_session

RECOGNISER = ModelVersionId("words-only-v1")
DETECTOR = ModelVersionId("detector-v1")
PROSODY = ModelVersionId("prosody-v1")


class _Runtime:
    """A speech runtime that says exactly what it contributes and nothing more.

    ``result`` is a function of the window so one class covers every shape
    the tests need: words only, words with a gap, words plus a detector's
    events plus a prosody estimator's readings.
    """

    emitted_speech_events: frozenset[SpeechEventType] = frozenset()
    emitted_prosody: frozenset[ProsodicIndicator] = frozenset()
    capability_detail = "test runtime"

    def __init__(self, contributions: Mapping[ModelRole, ModelVersionId], words, **extra) -> None:
        self._contributions = dict(contributions)
        self._words = words
        self._extra = extra

    @property
    def contributions(self) -> Mapping[ModelRole, ModelVersionId]:
        return self._contributions

    async def transcribe(self, window: AudioWindow) -> SpeechResult:
        return SpeechResult(
            contributions=self._contributions,
            window_position_ms=window.session_position_ms,
            words=self._words,
            stable_through_ms=window.session_position_ms + window.duration_ms,
            **self._extra,
        )


def _word(text: str, start_ms: int, end_ms: int, index: int) -> TimedWordHypothesis:
    return TimedWordHypothesis(
        raw_text=text, start_ms=start_ms, end_ms=end_ms, score=0.9, index=index
    )


def _recording(path: Path, seconds: int) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000 * seconds)
    return path


async def _engine(runtime: _Runtime, seconds: int) -> OratoriaEngine:
    # One window covering the whole file, so the runtime emits once and the
    # transcript never sees a stretch it already finalized re-emitted - which
    # the lexical frontier refuses, correctly, as rewriting history.
    engine = OratoriaEngine.local(
        EngineConfiguration(speech_runtime=runtime, window_seconds=seconds)
    )
    await engine.warmup()
    return engine


# ---------------------------------------------------------------------------
# F1 / F3 - the recogniser is recorded even when nothing else happened
# ---------------------------------------------------------------------------


async def test_the_recogniser_is_recorded_with_zero_events(tmp_path: Path) -> None:
    """Five recognised words used to produce ``models: {}``.

    The manifest read provenance off *events*, tokens carried none, and a
    runtime that recognises words but detects nothing - which is what the
    Whisper baseline is - therefore recorded no model at all.
    """
    runtime = _Runtime(
        {ModelRole.RECOGNISER: RECOGNISER},
        words=(_word("buenos", 0, 400, 0), _word("dias", 400, 800, 1)),
    )
    engine = await _engine(runtime, seconds=2)
    result = await engine.analyze_file(_recording(tmp_path / "p.wav", 2))

    assert result.evidence.manifest.models == {"recogniser": RECOGNISER.value}
    assert result.evidence.speech_events == ()
    for word in result.transcript.words:
        assert word.model_version == RECOGNISER.value


# ---------------------------------------------------------------------------
# F2 - silent pauses derive from the recogniser, not from a borrowed event
# ---------------------------------------------------------------------------


async def test_silent_pauses_derive_with_a_runtime_that_emits_no_events(tmp_path: Path) -> None:
    """The circular case: no events, so no provenance, so no pauses, so no events.

    A pause is derived from two word boundaries under the versioned threshold,
    so its provenance is the recogniser's - the model whose boundaries it was
    computed from. That is available whenever there are words, which is the
    only time there is anything to derive.
    """
    runtime = _Runtime(
        {ModelRole.RECOGNISER: RECOGNISER},
        # 1 200 ms between the words, over the 700 ms threshold.
        words=(_word("hola", 0, 300, 0), _word("mundo", 1_500, 1_800, 1)),
    )
    engine = await _engine(runtime, seconds=3)
    result = await engine.analyze_file(_recording(tmp_path / "p.wav", 3))

    pauses = [e for e in result.evidence.speech_events if e.type == "silent_pause"]
    assert len(pauses) == 1, [e.type for e in result.evidence.speech_events]
    assert pauses[0].start_ms == 300
    assert pauses[0].end_ms == 1_500
    # Attributed to the recogniser, and the manifest agrees - a derived pause
    # does not conjure a detector into the manifest.
    assert result.evidence.manifest.models == {"recogniser": RECOGNISER.value}


# ---------------------------------------------------------------------------
# F4 - one manifest, several audio components; one role, one version per run
# ---------------------------------------------------------------------------


def _assembler() -> SpeechAssembler:
    return SpeechAssembler(RunId("run-test"), EngineConfiguration().snapshot(), Calibrator())


def test_two_audio_components_are_both_recorded() -> None:
    """A recogniser and a detector are both audio, and both must appear.

    Under ``Mapping[Modality, ...]`` with ``setdefault`` the second was dropped
    without a trace, so the manifest read as complete while missing a model.
    """
    result = SpeechResult(
        contributions={ModelRole.RECOGNISER: RECOGNISER, ModelRole.DISFLUENCY_DETECTOR: DETECTOR},
        words=(_word("eh", 0, 300, 0),),
        events=(
            SpeechEventHypothesis(
                type=SpeechEventType.FILLED_PAUSE,
                start_ms=0,
                end_ms=300,
                score=0.9,
                role=ModelRole.DISFLUENCY_DETECTOR,
            ),
        ),
    )
    assembled = _assembler().assemble(result)
    transcript = transcript_module.build(assembled.tokens)

    models = _models_used(transcript, assembled.events, (), ())

    assert models == {
        ModelRole.RECOGNISER: RECOGNISER,
        ModelRole.DISFLUENCY_DETECTOR: DETECTOR,
    }


def _event(identifier: str, version: ModelVersionId) -> SpeechEvent:
    return SpeechEvent(
        id=EventId(identifier),
        type=SpeechEventType.FILLED_PAUSE,
        interval=Interval.of(0, 300),
        confidence=Confidence.calibrated(0.9),
        provenance=Provenance(
            modality=Modality.AUDIO,
            role=ModelRole.DISFLUENCY_DETECTOR,
            model_version=version,
            taxonomy_version=TAXONOMY_VERSION,
            configuration=ConfigurationSnapshotId("config-test"),
            evidence_ref=EvidenceRef("audio:run-test"),
        ),
    )


def test_two_versions_of_one_role_in_one_run_are_refused() -> None:
    """A canary is between runs. Inside one, two versions of a role is a bug.

    ``setdefault`` recorded whichever came first and said nothing. Refusing
    is what stops a document from claiming a provenance it cannot substantiate.
    """
    with pytest.raises(ProvenanceViolation, match="two versions of disfluency_detector"):
        _models_used(
            transcript_module.build([]),
            (_event("ev-1", DETECTOR), _event("ev-2", ModelVersionId("detector-v2"))),
            (),
            (),
        )


# ---------------------------------------------------------------------------
# F5 - every hypothesis is attributed to its own component
# ---------------------------------------------------------------------------


def test_each_hypothesis_carries_its_own_components_version() -> None:
    """One result, three models, three attributions.

    A single ``model_version`` per result stamped every event and reading with
    the recogniser's version - two components attributed to a model that never
    saw their input.
    """
    result = SpeechResult(
        contributions={
            ModelRole.RECOGNISER: RECOGNISER,
            ModelRole.DISFLUENCY_DETECTOR: DETECTOR,
            ModelRole.PROSODY_ESTIMATOR: PROSODY,
        },
        words=(_word("eh", 0, 300, 0),),
        events=(
            SpeechEventHypothesis(
                type=SpeechEventType.FILLED_PAUSE,
                start_ms=0,
                end_ms=300,
                score=0.9,
                role=ModelRole.DISFLUENCY_DETECTOR,
            ),
        ),
        prosody=(
            ProsodyHypothesis(
                indicator=ProsodicIndicator.PITCH_MEAN_HZ,
                start_ms=0,
                end_ms=1_000,
                value=120.0,
                score=0.8,
                role=ModelRole.PROSODY_ESTIMATOR,
            ),
        ),
    )
    assembled = _assembler().assemble(result)

    assert assembled.tokens[0].provenance.model_version == RECOGNISER
    assert assembled.tokens[0].provenance.role is ModelRole.RECOGNISER
    assert assembled.events[0].provenance.model_version == DETECTOR
    assert assembled.events[0].provenance.role is ModelRole.DISFLUENCY_DETECTOR
    assert assembled.prosody[0].provenance.model_version == PROSODY
    assert assembled.prosody[0].provenance.role is ModelRole.PROSODY_ESTIMATOR


def test_a_hypothesis_from_an_undeclared_component_is_refused() -> None:
    """Falling back to the recogniser's version would be the silent substitution.

    A detector that forgot to declare itself would otherwise have its events
    attributed to a model that never saw them, and nothing would look wrong.
    """
    result = SpeechResult(
        contributions={ModelRole.RECOGNISER: RECOGNISER},
        events=(
            SpeechEventHypothesis(
                type=SpeechEventType.FILLED_PAUSE,
                start_ms=0,
                end_ms=300,
                score=0.9,
                role=ModelRole.DISFLUENCY_DETECTOR,
            ),
        ),
    )
    with pytest.raises(ProvenanceViolation, match="disfluency_detector"):
        _assembler().assemble(result)


# ---------------------------------------------------------------------------
# F6 - the run records what was wired, before any evidence exists
# ---------------------------------------------------------------------------


async def test_a_run_records_its_models_before_any_evidence() -> None:
    """A run that fails on its first window still says what it was running.

    The manifest records what *contributed*; the run records what was *wired*.
    Without the second, an aborted run was indistinguishable from one that
    never had a model.
    """
    runtime = _Runtime({ModelRole.RECOGNISER: RECOGNISER}, words=())
    engine = await _engine(runtime, seconds=1)
    stream = engine.create_stream()
    await stream.send_audio(b"\x00\x00" * 16_000)

    run_id = stream._coordinator.state.run_id
    run = await engine._runs.get(run_id)
    assert run is not None
    assert run.models[ModelRole.RECOGNISER] == RECOGNISER
    assert run.models[ModelRole.VISUAL_ESTIMATOR] == ModelVersionId("deterministic-vision-v1")

    await stream.abort()
    aborted = await engine._runs.get(run_id)
    assert aborted is not None
    assert aborted.models == run.models


# ---------------------------------------------------------------------------
# QA-03 - a classifier is canaried while the recogniser stays fixed
# ---------------------------------------------------------------------------


def _version(identifier: str, role: ModelRole, approval: ApprovalState) -> ModelVersion:
    return ModelVersion(
        id=ModelVersionId(identifier),
        role=role,
        artifact_digest=f"sha256:{identifier}",
        dataset_version="v1",
        approval=approval,
        metrics={},
    )


async def test_the_classifier_canaries_while_the_recogniser_stays_fixed() -> None:
    """ADR-010:11, made possible.

    ``active_for(AUDIO)`` could return one model. The contextual classifier and
    the recogniser are both audio, so promoting one meant replacing the other -
    or, with the manifest's ``setdefault``, silently hiding one.
    """
    registry = InMemoryModelRegistry()
    registry.register(
        _version("asr-1", ModelRole.RECOGNISER, ApprovalState.PRODUCTION), make_active=True
    )
    registry.register(
        _version("clf-1", ModelRole.CONTEXT_CLASSIFIER, ApprovalState.PRODUCTION),
        make_active=True,
    )
    registry.register(_version("clf-2", ModelRole.CONTEXT_CLASSIFIER, ApprovalState.EVALUATED))

    promoted = await registry.promote(ModelVersionId("clf-2"), canary_percent=10)

    assert promoted.approval is ApprovalState.CANARY
    assert promoted.rollback_to == ModelVersionId("clf-1")
    assert (await registry.active_for(ModelRole.CONTEXT_CLASSIFIER)).id == ModelVersionId("clf-2")
    # The recogniser did not move.
    assert (await registry.active_for(ModelRole.RECOGNISER)).id == ModelVersionId("asr-1")

    restored = await registry.rollback(ModelRole.CONTEXT_CLASSIFIER)
    assert restored.id == ModelVersionId("clf-1")
    assert (await registry.active_for(ModelRole.RECOGNISER)).id == ModelVersionId("asr-1")
    assert len(await registry.list_versions(ModelRole.CONTEXT_CLASSIFIER)) == 2
    assert len(await registry.list_versions(ModelRole.RECOGNISER)) == 1


# ---------------------------------------------------------------------------
# Every layer, one test
# ---------------------------------------------------------------------------


def test_provenance_by_role_survives_every_layer_to_the_wire(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Port, assembler, document, serializer, wire.

    The scripted runtime declares three audio components under three ids and
    the vision runtime a fourth. Each must arrive on the published payload
    under its role, every token must name its recogniser, and nothing may
    still be keyed by modality. §4.14's rule: exercise the union through every
    consuming layer at once, or the layer nobody drove keeps the old shape.
    """
    created = client.post("/v1/sessions", json=CREATE_BODY, headers=auth)
    assert created.status_code == 201, created.text
    session_id = created.json()["session_id"]

    _stream_session(client, session_id, created.json()["stream_token"])

    document = client.get(f"/v1/sessions/{session_id}/result", headers=auth).json()
    models = document["manifest"]["models"]

    assert "audio" not in models
    assert "video" not in models
    assert models["recogniser"] == "deterministic-speech-v1"
    assert models["disfluency_detector"] == "deterministic-speech-v1-detector"
    # Visual and prosody entries exist exactly when the evidence they attribute
    # exists: the manifest records what contributed, not what was wired.
    assert ("visual_estimator" in models) == bool(document["visual_events"])
    assert ("prosody_estimator" in models) == bool(document["prosody"])

    tokens = document["transcript"]["tokens"]
    assert tokens, "the scripted session produces words"
    for token in tokens:
        assert token["model_version"] == "deterministic-speech-v1"
