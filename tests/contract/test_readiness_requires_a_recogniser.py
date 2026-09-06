"""`/health/ready` refuses a deployment with no recogniser wired.

Readiness answers for the roles a deployment wired, and reports the others as
absent rather than failing - a recogniser-only build is ready for what it
does. That rule had no floor: "absent" was acceptable for every role, the
recogniser included, so a runtime whose `contributions` named no recogniser
would have been reported ready and sent traffic it could only fail. Nothing
in the runtime protocol forces a recogniser into `contributions`; only the
per-call `SpeechResult` does, and a probe never sees one of those.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from evidence_engine.adapters.inbound.rest.health import ready
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.provenance import ModelRole


@dataclass
class _Runtime:
    contributions: Mapping[ModelRole, ModelVersionId]


@dataclass
class _Version:
    id: ModelVersionId


class _Registry:
    async def active_for(self, role: ModelRole) -> _Version:
        return _Version(id=ModelVersionId(f"{role.value}-v1"))


@dataclass
class _Profile:
    backend = "in-memory"
    runtime_mode = "deterministic"


@dataclass
class _Engine:
    speech: _Runtime
    vision: _Runtime
    registry: _Registry
    profile: _Profile
    #: Defaults to "already warm" so every test in this file that is not
    #: about warm-up keeps testing the one thing it names: the recogniser
    #: gate, not the warm-up gate.
    speech_warm_seconds: float | None = 2.5


def _engine(
    speech: Mapping[ModelRole, ModelVersionId], *, speech_warm_seconds: float | None = 2.5
) -> _Engine:
    return _Engine(
        speech=_Runtime(contributions=speech),
        vision=_Runtime(contributions={ModelRole.VISUAL_ESTIMATOR: ModelVersionId("vision-v1")}),
        registry=_Registry(),
        profile=_Profile(),
        speech_warm_seconds=speech_warm_seconds,
    )


async def test_a_deployment_with_no_recogniser_is_not_ready() -> None:
    response = await ready(_engine(speech={}))  # type: ignore[arg-type]

    assert response.status_code == 503
    body = json.loads(bytes(response.body))
    assert body["status"] == "not_ready"
    assert body["checks"]["model:recogniser"] == "unavailable (no recogniser wired)"


async def test_a_recogniser_only_deployment_is_ready() -> None:
    """The rule this test guards is not "every role", it is "the recogniser"."""
    response = await ready(  # type: ignore[arg-type]
        _engine(speech={ModelRole.RECOGNISER: ModelVersionId("asr-v1")})
    )

    assert response.status_code == 200
    body = json.loads(bytes(response.body))
    assert body["status"] == "ready"
    assert body["checks"]["model:recogniser"] == "ready (recogniser-v1)"
    assert body["checks"]["model:context_classifier"] == "absent (not wired in this deployment)"


async def test_a_wired_recogniser_that_has_not_decoded_yet_is_not_ready() -> None:
    """S4: a registered version is not evidence that a decode has run.

    `active_for()` above answers from the registry, which a deployment
    populates at wiring time - before anything has been asked to transcribe a
    single sample. `/health/ready` has to say the two are different claims,
    or a probe reaching this instance during its startup warm-up gets a 200
    for a model that has never actually run.
    """
    response = await ready(  # type: ignore[arg-type]
        _engine(
            speech={ModelRole.RECOGNISER: ModelVersionId("asr-v1")},
            speech_warm_seconds=None,
        )
    )

    assert response.status_code == 503
    body = json.loads(bytes(response.body))
    assert body["status"] == "not_ready"
    assert body["checks"]["speech:warm"] == "unavailable (warming up)"


async def test_a_completed_warm_up_reports_how_long_it_took() -> None:
    response = await ready(  # type: ignore[arg-type]
        _engine(
            speech={ModelRole.RECOGNISER: ModelVersionId("asr-v1")},
            speech_warm_seconds=3.25,
        )
    )

    assert response.status_code == 200
    body = json.loads(bytes(response.body))
    assert body["checks"]["speech:warm"] == "warm (3.2 s)"
