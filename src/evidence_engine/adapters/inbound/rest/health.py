"""Liveness and readiness (§7.1).

The distinction is operational, not cosmetic. ``/health/live`` says the process
is running and an orchestrator should not restart it. ``/health/ready`` says
dependencies and models are reachable and traffic may be routed here.

Collapsing them is a common and expensive mistake: a pod whose database
connection has dropped is *alive* - restarting it will not help and may make
the outage worse by dropping in-flight sessions - but it is not *ready*, and
the load balancer needs to know that difference.

Neither endpoint is authenticated, and neither reveals anything beyond
component names and states. NFR-018 keeps evidence out of operational
surfaces, and an unauthenticated endpoint is the last place to make an
exception.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from evidence_engine.adapters.inbound.rest.dependencies import EngineDep
from evidence_engine.adapters.inbound.rest.schemas import HealthBody
from evidence_engine.domain.shared.provenance import ModelRole

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthBody, summary="Process liveness")
async def live() -> HealthBody:
    return HealthBody(status="live")


@router.get("/health/ready", summary="Dependency and model readiness")
async def ready(engine: EngineDep) -> JSONResponse:
    checks: dict[str, str] = {}

    # Readiness answers "do the components this deployment WIRED resolve?",
    # not "is every role in the enum filled?". A recogniser-only deployment
    # is ready for what it does; the roles it does not have are *reported*
    # as absent rather than failing the probe - `/v1/capabilities` is where
    # a consumer learns what will not arrive. The first version iterated the
    # whole enum and returned 503 for every build that had no contextual
    # classifier, which is every build there is.
    wired = {**engine.speech.contributions, **engine.vision.contributions}
    for role in ModelRole:
        if role not in wired:
            checks[f"model:{role.value}"] = "absent (not wired in this deployment)"
            continue
        try:
            version = await engine.registry.active_for(role)
        except Exception as failure:
            # A readiness probe that raises returns a 500 with a stack trace
            # instead of a diagnosis, and the orchestrator learns nothing about
            # which component is missing.
            checks[f"model:{role.value}"] = f"unavailable ({type(failure).__name__})"
        else:
            checks[f"model:{role.value}"] = f"ready ({version.id.value})"

    # "Absent" is acceptable for every role but one. A deployment with no
    # recogniser has nothing to produce a transcript from, and reporting it
    # ready would send it traffic it can only fail. Nothing in the runtime
    # protocol forces a recogniser into `contributions` - only the per-call
    # `SpeechResult` does - so the probe has to say it here.
    if ModelRole.RECOGNISER not in wired:
        checks[f"model:{ModelRole.RECOGNISER.value}"] = "unavailable (no recogniser wired)"

    checks["backend"] = engine.profile.backend
    checks["runtime_mode"] = engine.profile.runtime_mode

    # A wired recogniser answering `active_for()` above says a version is
    # *registered* - it says nothing about whether a decode actually runs.
    # The startup warm-up in `bootstrap.app` is the one thing that checks
    # that, once, and this reports what it found: "unavailable" until it has,
    # so a probe reaching this deployment before its first decode gets a 503
    # rather than a 200 that sends it traffic the model has never proven it
    # can serve.
    checks["speech:warm"] = (
        f"warm ({engine.speech_warm_seconds:.1f} s)"
        if engine.speech_warm_seconds is not None
        else "unavailable (warming up)"
    )

    is_ready = all(not value.startswith("unavailable") for value in checks.values())
    body = HealthBody(status="ready" if is_ready else "not_ready", checks=checks)
    return JSONResponse(
        status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body.model_dump(),
    )
