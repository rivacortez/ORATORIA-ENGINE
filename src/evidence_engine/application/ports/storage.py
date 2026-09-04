"""Object storage for protected media.

NFR-009 requires protected data to be encrypted at rest and NFR-011 makes
ephemeral processing the default. Both shape this port.

``delete`` returns whether anything was removed, and ``exists`` is offered
separately, because QA-04's measure is "deletion verification has no
recoverable object reference" - a verification job needs to *ask*, and a delete
that silently succeeds on a missing key gives it nothing to assert on.

Signed URLs are minted with an explicit TTL rather than a default, because the
risk table's "consent withdrawal leaves copies" starts with a link that outlived
the evidence it pointed at.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from evidence_engine.domain.shared.identifiers import EvidenceRef, SessionId, TenantId


class MediaKind(StrEnum):
    """What a stored object holds. Drives the retention rule applied to it."""

    #: Raw audio or video. Deleted by default once processing completes.
    RAW_MEDIA = "raw_media"
    #: Derived features - spectrograms, landmark series. Protected, but not a
    #: recording: they cannot be played back as the person's likeness or voice.
    DERIVED_FEATURES = "derived_features"


@dataclass(frozen=True, slots=True)
class StoredObject:
    """A reference to something in the object store."""

    ref: EvidenceRef
    tenant: TenantId
    session_id: SessionId
    kind: MediaKind
    size_bytes: int
    content_type: str
    #: Epoch milliseconds after which the retention job must remove this. A
    #: required field: NFR-011 has no indefinite default to fall back on, so an
    #: object with no deadline is a bug rather than a permanent object.
    expires_at_ms: int


@dataclass(frozen=True, slots=True)
class SignedUrl:
    """A time-limited URL for upload or download (§6.2)."""

    url: str
    expires_at_ms: int


class MediaStore(Protocol):
    """Encrypted storage for protected media and derived features."""

    async def put(
        self,
        tenant: TenantId,
        session_id: SessionId,
        kind: MediaKind,
        content_type: str,
        payload: bytes,
        expires_at_ms: int,
    ) -> StoredObject: ...

    async def open(self, tenant: TenantId, ref: EvidenceRef) -> AsyncIterator[bytes]:
        """Stream an object back. Used by batch workers, never by the API."""
        ...

    async def exists(self, tenant: TenantId, ref: EvidenceRef) -> bool:
        """QA-04's verification step: is there still a recoverable reference?"""
        ...

    async def delete(self, tenant: TenantId, ref: EvidenceRef) -> bool:
        """Remove one object. Returns whether it was there to remove."""
        ...

    async def delete_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        """Remove every object for a session, returning the count removed."""
        ...

    async def count_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        """How many objects a session still has stored.

        Read-only, and separate from ``delete_for_session`` for that reason.
        QA-04's measure is that deletion verification finds no recoverable
        object reference, and a verification that deleted as a side effect of
        checking would always pass - it would be reporting on its own actions
        rather than on the state deletion left behind.
        """
        ...

    async def sign_upload(
        self,
        tenant: TenantId,
        session_id: SessionId,
        kind: MediaKind,
        content_type: str,
        ttl_seconds: int,
    ) -> SignedUrl:
        """Mint an upload URL for the batch flow (§6.2 step 1)."""
        ...

    async def revoke_signatures(self, tenant: TenantId, session_id: SessionId) -> None:
        """Invalidate outstanding signed URLs for a session.

        QA-04 requires deletion to invalidate signed URLs, not merely to delete
        the bytes. A URL minted a minute before the deletion request would
        otherwise keep working against a cache or a replica.
        """
        ...
