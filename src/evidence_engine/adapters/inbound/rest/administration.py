"""The provisioning surface: ``/v1/admin`` (FR-002, FR-003, US-006).

What a developer portal calls when somebody presses "Generate API key". Thin,
like the session routes: every rule - the operator-scope check, the refusal to
mint another operator credential, the audit write - lives in
``AdministerApiKeys`` so that a second transport cannot bypass it.

**Read this before exposing these routes.** They are reached only with a key
carrying ``Scope.ADMIN``, which ``allows()`` treats as a superset of every
other scope. That credential belongs to whoever operates the deployment. It is
never issued to a customer, and this API will not mint one - see
``ISSUABLE_SCOPES``.

The secret appears in exactly one response body in this module, from
``POST .../keys``, and nowhere else: listing returns descriptors that have no
field for it. That asymmetry is the contract a portal is built against - show
it once, store the prefix.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Path, Query, status

from evidence_engine.adapters.inbound.rest.dependencies import CallerDep, EngineDep
from evidence_engine.adapters.inbound.rest.schemas import (
    ApiKeyBody,
    ApiKeyListBody,
    ApplicationBody,
    ApplicationListBody,
    CreateApplicationBody,
    IssuedKeyBody,
    IssueKeyBody,
)
from evidence_engine.application.api import EngineApi
from evidence_engine.application.ports.platform import (
    ApiKeyDescriptor,
    ClientApplicationRecord,
    Scope,
)
from evidence_engine.domain.shared.identifiers import ApiKeyId, ApplicationId, TenantId

router = APIRouter(prefix="/v1/admin", tags=["administration"])

ApplicationIdPath = Annotated[str, Path(min_length=1, max_length=64)]
KeyIdPath = Annotated[str, Path(min_length=1, max_length=64)]
TenantQuery = Annotated[str, Query(min_length=1, max_length=64)]


@router.post(
    "/applications",
    status_code=status.HTTP_201_CREATED,
    response_model=ApplicationBody,
    summary="Register a client application under a tenant",
    description=(
        "An application is what keys belong to; a tenant is what evidence is "
        "isolated by (NFR-013). One customer typically gets one tenant and one "
        "application per environment, so a staging credential can be revoked "
        "without touching production.\n\n"
        "The tenant is named here and is not validated against a registry: "
        "there is no tenant table. Naming a new one creates it implicitly, "
        "which is why this route needs the operator credential."
    ),
)
async def create_application(
    engine: EngineDep,
    caller: CallerDep,
    body: Annotated[CreateApplicationBody, Body()],
) -> ApplicationBody:
    record = await engine.administer_keys.create_application(
        caller, TenantId(body.tenant), body.name
    )
    return _application(record, _schema_version(engine))


@router.get(
    "/applications",
    response_model=ApplicationListBody,
    summary="List a tenant's applications",
)
async def list_applications(
    engine: EngineDep, caller: CallerDep, tenant: TenantQuery
) -> ApplicationListBody:
    records = await engine.administer_keys.list_applications(caller, TenantId(tenant))
    version = _schema_version(engine)
    return ApplicationListBody(
        schema_version=version,
        applications=[_application(record, version) for record in records],
    )


@router.post(
    "/applications/{application_id}/keys",
    status_code=status.HTTP_201_CREATED,
    response_model=IssuedKeyBody,
    summary="Issue an API key (the secret is returned once)",
    description=(
        "**The `secret` field of this response is the only time the key "
        "exists outside the caller.** Only a peppered hash is stored (FR-002), "
        "so it cannot be recovered - show it once and persist the `prefix`.\n\n"
        "The key's tenant comes from the application, not from this request. "
        "`admin` cannot be issued here: it is a platform-operator credential "
        "and minting one through an API would mean a leaked operator key could "
        "create its own successor."
    ),
)
async def issue_key(
    engine: EngineDep,
    caller: CallerDep,
    application_id: ApplicationIdPath,
    body: Annotated[IssueKeyBody, Body()],
) -> IssuedKeyBody:
    secret, descriptor = await engine.administer_keys.issue_key(
        caller,
        ApplicationId(application_id),
        frozenset(Scope(value) for value in body.scopes),
        body.expires_at_ms,
    )
    version = _schema_version(engine)
    return IssuedKeyBody(
        schema_version=version,
        secret=secret,
        key=_key(descriptor, version),
    )


@router.get(
    "/applications/{application_id}/keys",
    response_model=ApiKeyListBody,
    summary="List an application's keys, revoked ones included",
    description=(
        "No secrets. Revoked keys stay in the list: a revocation stops the "
        "credential working, it does not erase that the credential existed, "
        "and an operator investigating an incident needs to see the key used."
    ),
)
async def list_keys(
    engine: EngineDep, caller: CallerDep, application_id: ApplicationIdPath
) -> ApiKeyListBody:
    descriptors = await engine.administer_keys.list_keys(caller, ApplicationId(application_id))
    version = _schema_version(engine)
    return ApiKeyListBody(
        schema_version=version,
        keys=[_key(descriptor, version) for descriptor in descriptors],
    )


@router.delete(
    "/keys/{key_id}",
    response_model=ApiKeyBody,
    summary="Revoke a key immediately",
    description=(
        "Idempotent (US-006). Revoking an already-revoked key returns the same "
        "record with its original `revoked_at_ms`, rather than an error that "
        "would read as 'the key is still live'."
    ),
)
async def revoke_key(engine: EngineDep, caller: CallerDep, key_id: KeyIdPath) -> ApiKeyBody:
    descriptor = await engine.administer_keys.revoke_key(caller, ApiKeyId(key_id))
    return _key(descriptor, _schema_version(engine))


def _application(record: ClientApplicationRecord, version: str) -> ApplicationBody:
    return ApplicationBody(
        schema_version=version,
        application_id=record.id.value,
        tenant=record.tenant.value,
        name=record.name,
        status=record.status,
    )


def _key(descriptor: ApiKeyDescriptor, version: str) -> ApiKeyBody:
    return ApiKeyBody(
        schema_version=version,
        key_id=descriptor.key_id.value,
        application_id=descriptor.application.value,
        tenant=descriptor.tenant.value,
        prefix=descriptor.prefix,
        scopes=sorted(scope.value for scope in descriptor.scopes),
        expires_at_ms=descriptor.expires_at_ms,
        revoked_at_ms=descriptor.revoked_at_ms,
    )


def _schema_version(engine: EngineApi) -> str:
    return str(engine.read_capabilities.execute().schema_version)
