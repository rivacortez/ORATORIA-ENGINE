"""API keys: issued once, stored as a hash, never logged.

FR-002 requires keys to be issued only once with only a cryptographic hash
persisted, and NFR-010 requires the raw value never to appear in logs or
database dumps. Both are properties of this module and of nothing else, so it
is deliberately small and does exactly one thing.

The hash is SHA-256 over a peppered secret, not bcrypt or argon2. That is the
right choice here and the reasoning matters, because it is the opposite of the
right choice for passwords. A password is low-entropy and human-chosen, so a
stolen hash must be made expensive to attack - that is what a slow KDF buys. An
API key is 256 bits of ``secrets.token_urlsafe``; there is no dictionary to run
against it and no cost factor that improves on the entropy. What a slow KDF
*would* buy is a hash computation on every single authenticated request, which
is a latency budget spent for nothing.

Comparison is constant-time regardless, because the lookup key is derived from
attacker-supplied input.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from evidence_engine.adapters.outbound.cache.in_memory import ClockReader
from evidence_engine.application.ports.platform import AuthenticatedCaller, Scope
from evidence_engine.domain.shared.identifiers import ApiKeyId, ApplicationId, TenantId

#: Prefix carried in the clear so a leaked key is identifiable in a scan of
#: someone's repository without anybody having to test it against the API.
KEY_PREFIX = "oek_"


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    """§8 ``ApiKey``. Note what is absent: the secret itself."""

    id: ApiKeyId
    application: ApplicationId
    tenant: TenantId
    hashed_secret: str
    prefix: str
    scopes: frozenset[Scope]
    expires_at_ms: int | None = None
    revoked_at_ms: int | None = None

    def is_usable(self, now_ms: int) -> bool:
        if self.revoked_at_ms is not None:
            return False
        return self.expires_at_ms is None or now_ms < self.expires_at_ms


def hash_secret(secret: str, pepper: str) -> str:
    """Derive the stored form of a key.

    The pepper lives in the secret manager rather than the database (§15.2), so
    a leaked database dump alone does not let an attacker verify guesses
    offline against the hashes it contains.
    """
    return hashlib.sha256(f"{pepper}:{secret}".encode()).hexdigest()


def generate_secret() -> str:
    """Mint a new key. Returned once, to the caller, and never stored (US-006)."""
    return f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"


class InMemoryApiKeyDirectory:
    """Resolves presented secrets to callers."""

    def __init__(self, pepper: str, clock: ClockReader) -> None:
        self._pepper = pepper
        self._clock = clock
        self._by_hash: dict[str, ApiKeyRecord] = {}

    def register(self, record: ApiKeyRecord) -> None:
        self._by_hash[record.hashed_secret] = record

    def issue(
        self,
        application: ApplicationId,
        tenant: TenantId,
        scopes: frozenset[Scope],
        expires_at_ms: int | None = None,
    ) -> tuple[str, ApiKeyRecord]:
        """Create a key, returning the plaintext exactly once (US-006)."""
        secret = generate_secret()
        record = ApiKeyRecord(
            id=ApiKeyId.generate(),
            application=application,
            tenant=tenant,
            hashed_secret=hash_secret(secret, self._pepper),
            prefix=secret[: len(KEY_PREFIX) + 6],
            scopes=scopes,
            expires_at_ms=expires_at_ms,
        )
        self.register(record)
        return secret, record

    def revoke(self, key_id: ApiKeyId, at_ms: int) -> bool:
        """US-006: revocation blocks new calls immediately."""
        for hashed, record in self._by_hash.items():
            if record.id != key_id:
                continue
            self._by_hash[hashed] = ApiKeyRecord(
                id=record.id,
                application=record.application,
                tenant=record.tenant,
                hashed_secret=record.hashed_secret,
                prefix=record.prefix,
                scopes=record.scopes,
                expires_at_ms=record.expires_at_ms,
                revoked_at_ms=at_ms,
            )
            return True
        return False

    async def authenticate(
        self, presented_secret: str, trace_id: str
    ) -> AuthenticatedCaller | None:
        candidate = hash_secret(presented_secret, self._pepper)

        # Walk every record with a constant-time compare rather than a dict
        # lookup on the hash. A dict lookup is O(1) and leaks nothing here in
        # practice, but the comparison itself is the part an attacker can time,
        # and doing it this way keeps the property obvious to a reader.
        matched: ApiKeyRecord | None = None
        for stored_hash, record in self._by_hash.items():
            if hmac.compare_digest(stored_hash, candidate):
                matched = record

        if matched is None or not matched.is_usable(self._clock.epoch_ms()):
            # One answer for unknown, revoked and expired. Distinguishing them
            # tells a prober which of their guesses used to work.
            return None

        return AuthenticatedCaller(
            application=matched.application,
            tenant=matched.tenant,
            key_id=matched.id,
            scopes=matched.scopes,
            trace_id=trace_id,
        )
