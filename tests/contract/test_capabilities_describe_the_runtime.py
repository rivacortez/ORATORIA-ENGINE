"""`/v1/capabilities` describes this deployment, not the taxonomy.

The query used to take a schema version and nothing else, and returned every
member of ``SpeechEventType`` and ``VisualEventType`` straight from the enums.
It had no way to know which runtimes were wired, so a deployment running a
recogniser and no disfluency detector advertised nine disfluency classes it
could not detect. A consumer integrating against that - OratorIA, in this
project's case - would build a view for findings that were never coming, and
would discover the truth only by a session producing none of them.

Three lists now, and the distinction between them is the whole point:

- the **catalogue** says what the contract can carry
- **emitted** says what will actually arrive
- **unavailable** says what the contract defines and this build cannot produce,
  each with a reason

An empty ``emitted`` list plus a populated ``unavailable`` list is the
difference between "the speaker had no disfluencies" and "nothing here looks
for disfluencies". Those read identically when the only published list is the
catalogue.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from evidence_engine.adapters.outbound.model_runtime.whisper import WhisperSpeechRuntime
from evidence_engine.application.queries.read_session import ReadCapabilities
from evidence_engine.bootstrap.container import Container
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.measurement import UnavailabilityReason
from evidence_engine.domain.shared.provenance import SemanticVersion
from evidence_engine.domain.shared.taxonomy import (
    ProsodicIndicator,
    SpeechEventType,
    VisualEventType,
)

_A_VERSION = ModelVersionId("whisper-large-v3@test")


def test_the_catalogue_still_publishes_the_whole_taxonomy(
    client: TestClient, auth: dict[str, str]
) -> None:
    """NFR-017's compatibility promise is about the schema, not the build.

    A consumer needs to know what shape a result can take across the major
    version, which is a different question from what today's deployment emits.
    Removing the catalogue would break that; the fix was to stop letting it
    answer both questions.
    """
    body = client.get("/v1/capabilities", headers=auth).json()

    assert set(body["speech_event_types"]) == {t.value for t in SpeechEventType}
    assert set(body["visual_event_types"]) == {t.value for t in VisualEventType}


def test_the_emitted_lists_come_from_the_wired_runtimes(
    client: TestClient, auth: dict[str, str]
) -> None:
    """The contract suite runs the scripted runtime, which can emit anything.

    That is a real capability - a script declares its own events - and it is
    the reason this has to be per-runtime rather than a constant: this test and
    the Whisper test below both pass, and they disagree, because the two
    deployments genuinely differ.
    """
    body = client.get("/v1/capabilities", headers=auth).json()

    assert set(body["emitted_speech_event_types"]) == {t.value for t in SpeechEventType}
    assert set(body["emitted_visual_event_types"]) == {t.value for t in VisualEventType}
    assert body["unavailable_capabilities"] == []


def test_the_whisper_baseline_advertises_no_detector_at_all() -> None:
    """The finding, as an assertion.

    whisper-large-v3 is a recogniser. It emits words; it detects no disfluency
    and measures no prosody, because neither detector has been built. Every one
    of the nine speech classes and five prosodic indicators must therefore
    appear as unavailable with a reason - not as a capability.
    """
    capabilities = ReadCapabilities(
        SemanticVersion(1, 0, 0), speech=WhisperSpeechRuntime(object(), _A_VERSION)
    ).execute()

    assert capabilities.emitted_speech_event_types == ()
    assert capabilities.emitted_prosodic_indicators == ()

    absent = {(c.kind, c.name) for c in capabilities.unavailable_capabilities}
    for member in SpeechEventType:
        assert ("speech_event", member.value) in absent
    for indicator in ProsodicIndicator:
        assert ("prosodic_indicator", indicator.value) in absent


def test_an_unavailable_capability_says_why_and_says_it_distinctly() -> None:
    """`detector_not_deployed`, not `processing_failed`.

    The other reasons in the enum describe a window that could not be measured.
    This one describes a capability absent for the whole session, and a
    consumer branches differently on the two: one is "we lost twenty seconds",
    the other is "this build does not do that".
    """
    capabilities = ReadCapabilities(
        SemanticVersion(1, 0, 0), speech=WhisperSpeechRuntime(object(), _A_VERSION)
    ).execute()

    entry = next(iter(capabilities.unavailable_capabilities))
    assert entry.reason is UnavailabilityReason.DETECTOR_NOT_DEPLOYED
    assert "Phase 4" in entry.detail


def test_emitted_and_unavailable_partition_the_catalogue_exactly() -> None:
    """No class is in both lists, and none is in neither.

    Computed by subtraction rather than listed, so a class added to the
    taxonomy appears as unavailable until something can emit it. A hand-written
    list would be a second declaration of the same fact and would be the one
    that goes stale.
    """
    capabilities = ReadCapabilities(
        SemanticVersion(1, 0, 0), speech=WhisperSpeechRuntime(object(), _A_VERSION)
    ).execute()

    emitted = set(capabilities.emitted_speech_event_types)
    absent = {c.name for c in capabilities.unavailable_capabilities if c.kind == "speech_event"}

    assert emitted & absent == set()
    assert emitted | absent == {t.value for t in SpeechEventType}


def test_the_published_body_carries_all_three_lists(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A consumer must not have to compute the difference itself.

    Publishing only the catalogue and the emitted list would leave the reason
    unstated, and the reason is the part that distinguishes an absent detector
    from a quiet speaker.
    """
    body = client.get("/v1/capabilities", headers=auth).json()

    for key in (
        "speech_event_types",
        "emitted_speech_event_types",
        "emitted_visual_event_types",
        "emitted_prosodic_indicators",
        "unavailable_capabilities",
    ):
        assert key in body, key


def test_capabilities_publishes_wired_model_versions_and_instance_identity(
    client: TestClient, auth: dict[str, str], container: Container
) -> None:
    """S4: two questions a pilot with more than one workstation has to ask.

    ``models`` is keyed by *role* and not by modality, for the reason the
    document's own manifest is (§3.20): one audio entry could not show a
    disfluency detector running beside a fixed recogniser. ``instance`` is
    the same identity `session.accepted` now carries, published where a
    caller checking compatibility before integrating would look first.
    """
    body = client.get("/v1/capabilities", headers=auth).json()

    assert body["models"]["recogniser"] == "deterministic-speech-v1"
    assert body["models"]["visual_estimator"] == "deterministic-vision-v1"
    assert body["instance"] == {
        "id": container.profile.instance_id,
        "hostname": container.profile.hostname,
    }
