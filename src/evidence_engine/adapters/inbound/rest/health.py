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
from evidence_engine.domain.shared.provenance import Modality

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthBody, summary="Process liveness")
async def live() -> HealthBody:
    return HealthBody(status="live")


@router.get("/health/ready", summary="Dependency and model readiness")
async def ready(engine: EngineDep) -> JSONResponse:
    checks: dict[str, str] = {}

    for modality in (Modality.AUDIO, Modality.VIDEO):
        try:
            version = await engine.registry.active_for(modality)
        except Exception as failure:
            # A readiness probe that raises returns a 500 with a stack trace
            # instead of a diagnosis, and the orchestrator learns nothing about
            # which component is missing.
            checks[f"model:{modality.value}"] = f"unavailable ({type(failure).__name__})"
        else:
            checks[f"model:{modality.value}"] = f"ready ({version.id.value})"

    checks["backend"] = engine.profile.backend
    checks["runtime_mode"] = engine.profile.runtime_mode

    is_ready = all(not value.startswith("unavailable") for value in checks.values())
    body = HealthBody(status="ready" if is_ready else "not_ready", checks=checks)
    return JSONResponse(
        status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body.model_dump(),
    )
