"""In-memory media store with the retention behaviour of the real one.

Two properties are modelled rather than stubbed, because QA-04 is measured
against them:

*Signatures are revocable.* Minted URLs are tracked and ``revoke_signatures``
invalidates them. A store that returned a URL and forgot about it could not
demonstrate the QA-04 response - "invalidate signed URLs" - and the deletion
drill in §13 Phase 10 would be checking nothing.

*Expiry is enforced on read.* An object past its deadline is gone as far as any
reader is concerned, even before the retention job sweeps it. NFR-011 makes
ephemeral processing the default, and a store where expiry only happened when a
cron ran would let an object outlive its policy for as long as the cron was
broken.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass

from evidence_engine.adapters.outbound.cache.in_memory import ClockReader
from evidence_engine.application.ports.storage import MediaKind, SignedUrl, StoredObject
from evidence_engine.domain.shared.identifiers import EvidenceRef, SessionId, TenantId


@dataclass(slots=True)
class _Entry:
    stored: StoredObject
    payload: bytes


class InMemoryMediaStore:
    """Encrypted-at-rest is out of scope here; everything else is faithful."""

    def __init__(self, clock: ClockReader) -> None:
        self._clock = clock
        self._objects: dict[tuple[str, str], _Entry] = {}
        self._signatures: dict[str, set[str]] = {}

    async def put(
        self,
        tenant: TenantId,
        session_id: SessionId,
        kind: MediaKind,
        content_type: str,
        payload: bytes,
        expires_at_ms: int,
    ) -> StoredObject:
        ref = EvidenceRef(
            f"{kind.value}:{session_id.value}:{hashlib.sha256(payload).hexdigest()[:16]}"
        )
        stored = StoredObject(
            ref=ref,
            tenant=tenant,
            session_id=session_id,
            kind=kind,
            size_bytes=len(payload),
            content_type=content_type,
            expires_at_ms=expires_at_ms,
        )
        self._objects[(tenant.value, ref.value)] = _Entry(stored=stored, payload=payload)
        return stored

    async def open(self, tenant: TenantId, ref: EvidenceRef) -> AsyncIterator[bytes]:
        entry = self._live(tenant, ref)
        if entry is None:
            raise FileNotFoundError(f"no live object at {ref}")

        async def _stream() -> AsyncIterator[bytes]:
            yield entry.payload

        return _stream()

    async def exists(self, tenant: TenantId, ref: EvidenceRef) -> bool:
        return self._live(tenant, ref) is not None

    async def delete(self, tenant: TenantId, ref: EvidenceRef) -> bool:
        return self._objects.pop((tenant.value, ref.value), None) is not None

    async def delete_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        keys = [
            key
            for key, entry in self._objects.items()
            if key[0] == tenant.value and entry.stored.session_id == session_id
        ]
        for key in keys:
            del self._objects[key]
        return len(keys)

    async def count_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        return sum(
            1
            for key, entry in self._objects.items()
            if key[0] == tenant.value
            and entry.stored.session_id == session_id
            and not self._expired(entry)
        )

    async def sign_upload(
        self,
        tenant: TenantId,
        session_id: SessionId,
        kind: MediaKind,
        content_type: str,
        ttl_seconds: int,
    ) -> SignedUrl:
        signature = secrets.token_urlsafe(24)
        self._signatures.setdefault(session_id.value, set()).add(signature)
        return SignedUrl(
            url=f"memory://{tenant.value}/{session_id.value}/{kind.value}?sig={signature}",
            expires_at_ms=self._clock.epoch_ms() + ttl_seconds * 1_000,
        )

    async def revoke_signatures(self, tenant: TenantId, session_id: SessionId) -> None:
        self._signatures.pop(session_id.value, None)

    def signature_count(self, session_id: SessionId) -> int:
        """How many signatures are still valid. Used by the deletion drill."""
        return len(self._signatures.get(session_id.value, ()))

    # -- internals --------------------------------------------------------

    def _live(self, tenant: TenantId, ref: EvidenceRef) -> _Entry | None:
        entry = self._objects.get((tenant.value, ref.value))
        if entry is None:
            return None
        if self._expired(entry):
            # Removed on read, not left for a sweeper. An object past its
            # retention deadline must be unreachable immediately.
            del self._objects[(tenant.value, ref.value)]
            return None
        return entry

    def _expired(self, entry: _Entry) -> bool:
        return self._clock.epoch_ms() >= entry.stored.expires_at_ms
