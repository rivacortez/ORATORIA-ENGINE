"""Provision applications and API keys (FR-002, FR-003, US-006, NFR-022).

This is the use case a developer portal calls. Somebody signs up on a
consuming product, presses "Generate API key", and that product's backend
reaches this with its own operator credential.

Three rules, and each is enforced here rather than in a route handler because
§7.1 exposes the engine over more than one transport and a rule living in a
handler would be absent from every other entry point.

**`admin` is an operator credential, never a customer one.** `allows()` treats
`Scope.ADMIN` as a superset of every other scope, so a key carrying it can do
anything to anything. That is correct for the operator of the deployment and
catastrophic in a customer's hands.

**So the administration API refuses to mint another `admin` key.** A leaked
operator credential can create customer keys - bad, and revocable - but cannot
create a second operator credential that outlives the revocation of the first.
Operator keys come from `evidence-engine bootstrap-admin`, run by a person
against the database, and nowhere else. This bounds the blast radius of the
one credential whose loss would otherwise be unbounded, and it is the reason
`Scope.ADMIN` is filtered out of `issue_key` rather than merely discouraged.

**Every mutation is audited before it is returned.** NFR-022 names key
management as one of the trails that must land in the append-only log, and a
record written after the caller already holds the secret would be attesting to
something it could not undo.

What this deliberately does *not* do is manage people. There are no users,
passwords, sessions or emails here, and there will not be: `SCOPE.md` puts the
student and the instructor behind the consuming application and says the engine
never occupies that position. Accounts are the portal's problem; tenants,
applications and keys are this engine's.
"""

from __future__ import annotations

from evidence_engine.application.errors import ApplicationError, NotAuthorized
from evidence_engine.application.ports.clock import Clock
from evidence_engine.application.ports.platform import (
    ApiKeyAdministration,
    ApiKeyDescriptor,
    AuthenticatedCaller,
    ClientApplicationRecord,
    Scope,
)
from evidence_engine.application.ports.repositories import AuditLog, AuditRecord
from evidence_engine.domain.shared.identifiers import ApiKeyId, ApplicationId, TenantId

#: Scopes this API will mint. Every published scope except `admin`, computed by
#: subtraction rather than listed, so a scope added to the enum is issuable by
#: default and only the escalation path stays closed by construction.
ISSUABLE_SCOPES: frozenset[Scope] = frozenset(Scope) - {Scope.ADMIN}

#: The longest life a key may be issued with, in milliseconds. A year.
#: Unbounded expiry is allowed (`None`) because a machine credential rotated by
#: nothing is the common real case and pretending otherwise would push
#: operators into setting a far-future date that reads as a policy. What is not
#: allowed is an expiry so distant it is unbounded while looking bounded.
MAX_KEY_LIFETIME_MS = 366 * 24 * 60 * 60 * 1_000


class ApplicationNotFound(ApplicationError):
    """No such client application."""


class KeyNotFound(ApplicationError):
    """No such API key."""


class UnissuableScope(ApplicationError):
    """A scope was requested that this API will not mint."""


class AdministerApiKeys:
    """Use case: create applications, issue keys, list them, revoke them."""

    def __init__(
        self,
        directory: ApiKeyAdministration,
        audit: AuditLog,
        clock: Clock,
    ) -> None:
        self._directory = directory
        self._audit = audit
        self._clock = clock

    async def create_application(
        self, caller: AuthenticatedCaller, tenant: TenantId, name: str
    ) -> ClientApplicationRecord:
        """Register an integration that keys can then be issued against.

        The tenant is named by the caller rather than taken from the operator
        credential, because creating a tenant is by definition something that
        cannot happen from inside one. That is the whole reason `admin` is
        never issued to a customer: this parameter is exactly the cross-tenant
        reach that would otherwise let one customer provision against another.
        """
        self._require_operator(caller, "creating an application")

        readable = name.strip()
        if not readable:
            raise ApplicationError("an application needs a name; it is what a person revokes by")

        record = await self._directory.create_application(tenant, readable)
        await self._record(
            caller,
            action="application.create",
            resource=record.id.value,
            tenant=tenant,
            outcome="created",
            detail=f"name={readable!r}",
        )
        return record

    async def list_applications(
        self, caller: AuthenticatedCaller, tenant: TenantId
    ) -> tuple[ClientApplicationRecord, ...]:
        self._require_operator(caller, "listing applications")
        return await self._directory.list_applications(tenant)

    async def issue_key(
        self,
        caller: AuthenticatedCaller,
        application: ApplicationId,
        scopes: frozenset[Scope],
        expires_at_ms: int | None = None,
    ) -> tuple[str, ApiKeyDescriptor]:
        """Mint a key. The plaintext is returned once and never again.

        The tenant comes from the application row, not from the request. A
        request that could name both would let a caller attach a key to one
        tenant's application while stamping another tenant on the key - and the
        key's tenant is what every isolation check in the system reads.
        """
        self._require_operator(caller, "issuing a key")

        refused = scopes - ISSUABLE_SCOPES
        if refused:
            # Named explicitly rather than silently dropped. A portal that asked
            # for `admin` and got a key without it would be handing its customer
            # a credential that fails later, on an unrelated call, for a reason
            # nothing in this response mentioned.
            raise UnissuableScope(
                f"refusing to issue {sorted(scope.value for scope in refused)}: "
                "`admin` is a platform-operator credential and this API will not "
                "mint one. Use `evidence-engine bootstrap-admin` against the "
                "database, where a person has to be present."
            )
        if not scopes:
            raise UnissuableScope(
                "a key with no scopes can authenticate and do nothing; name what "
                f"it is for. Issuable: {sorted(scope.value for scope in ISSUABLE_SCOPES)}"
            )

        record = await self._directory.get_application(application)
        if record is None:
            raise ApplicationNotFound(f"application {application.value} not found")
        if record.status != "active":
            raise ApplicationError(
                f"application {application.value} is {record.status}; a disabled "
                "application does not get new credentials"
            )

        now_ms = self._clock.epoch_ms()
        if expires_at_ms is not None:
            if expires_at_ms <= now_ms:
                raise ApplicationError(
                    "expires_at_ms is in the past; the key would be dead on arrival"
                )
            if expires_at_ms - now_ms > MAX_KEY_LIFETIME_MS:
                raise ApplicationError(
                    f"expires_at_ms is more than {MAX_KEY_LIFETIME_MS // 86_400_000} days "
                    "out. Pass null for a key with no expiry rather than one whose "
                    "expiry is a decoration."
                )

        secret, descriptor = await self._directory.issue(
            application=application,
            tenant=record.tenant,
            scopes=scopes,
            expires_at_ms=expires_at_ms,
        )
        # Written before the secret is handed back. The plaintext is not in the
        # record - `detail` carries the prefix, which is in the clear by design
        # (NFR-010 forbids the raw value appearing in a log or a dump).
        await self._record(
            caller,
            action="api_key.issue",
            resource=descriptor.key_id.value,
            tenant=descriptor.tenant,
            outcome="issued",
            detail=(
                f"application={descriptor.application.value} prefix={descriptor.prefix} "
                f"scopes={sorted(scope.value for scope in descriptor.scopes)}"
            ),
        )
        return secret, descriptor

    async def list_keys(
        self, caller: AuthenticatedCaller, application: ApplicationId
    ) -> tuple[ApiKeyDescriptor, ...]:
        """What a dashboard renders. Descriptors carry no secret and cannot."""
        self._require_operator(caller, "listing keys")

        if await self._directory.get_application(application) is None:
            raise ApplicationNotFound(f"application {application.value} not found")
        return await self._directory.list_keys(application)

    async def revoke_key(self, caller: AuthenticatedCaller, key_id: ApiKeyId) -> ApiKeyDescriptor:
        """US-006: revocation blocks new calls immediately.

        Idempotent. Revoking an already-revoked key returns the same descriptor
        rather than an error, because the operator pressing the button twice is
        asking for a state, not for an event, and telling them it failed is how
        somebody concludes the key is still live.
        """
        self._require_operator(caller, "revoking a key")

        descriptor = await self._directory.revoke(key_id, self._clock.epoch_ms())
        if descriptor is None:
            raise KeyNotFound(f"api key {key_id.value} not found")

        await self._record(
            caller,
            action="api_key.revoke",
            resource=key_id.value,
            tenant=descriptor.tenant,
            outcome="revoked",
            detail=f"application={descriptor.application.value} prefix={descriptor.prefix}",
        )
        return descriptor

    @staticmethod
    def _require_operator(caller: AuthenticatedCaller, doing: str) -> None:
        """`Scope.ADMIN`, checked directly rather than through ``allows()``.

        ``allows()`` answers "may this caller do X", treating `admin` as a
        superset - which is what makes it the right check everywhere else and
        the wrong one here. Asking it whether a caller `allows(Scope.ADMIN)`
        happens to give the same answer today, and would keep giving one if
        somebody later made another scope imply administration. The membership
        test says what is meant: this needs the operator credential itself.
        """
        if Scope.ADMIN not in caller.scopes:
            raise NotAuthorized(f"{doing} requires the {Scope.ADMIN.value} scope")

    async def _record(
        self,
        caller: AuthenticatedCaller,
        *,
        action: str,
        resource: str,
        tenant: TenantId,
        outcome: str,
        detail: str,
    ) -> None:
        await self._audit.record(
            AuditRecord(
                actor=caller.application.value,
                action=action,
                resource=resource,
                timestamp_ms=self._clock.epoch_ms(),
                trace_id=caller.trace_id,
                outcome=outcome,
                tenant=tenant,
                detail=detail,
            )
        )
