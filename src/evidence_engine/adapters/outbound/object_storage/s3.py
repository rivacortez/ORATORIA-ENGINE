"""S3-compatible object storage for protected media.

Two properties matter more than the API surface, and both are QA-04's measure.

*Signed URLs are revocable.* S3 presigned URLs cannot be individually revoked —
the signature is valid until it expires, wherever it has been copied to. So
this adapter tracks the object keys it has signed for and, on revocation,
**deletes the objects**. That is the only mechanism that actually invalidates
an outstanding link, and QA-04 requires invalidation rather than expiry.

*Expiry is enforced, not hoped for.* Bucket lifecycle rules are the right
long-term mechanism and they run on the provider's schedule. The object's own
deadline is also stored as metadata and checked on read, so an object past its
retention policy is unreachable immediately rather than whenever the sweeper
next runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any, Protocol

import aioboto3
from aiobotocore.config import AioConfig

from evidence_engine.application.ports.storage import MediaKind, SignedUrl, StoredObject
from evidence_engine.domain.shared.identifiers import EvidenceRef, SessionId, TenantId


class ClockRead(Protocol):
    """A zero-argument epoch-millisecond read."""

    def __call__(self) -> int: ...


#: Object metadata key holding the retention deadline in epoch milliseconds.
_EXPIRES_METADATA = "expires-at-ms"


class S3MediaStore:
    """Encrypted-at-rest object storage, with retention enforced on read."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        clock_epoch_ms: ClockRead,
        access_key: str,
        secret_key: str,
        *,
        region: str = "us-east-1",
        server_side_encryption: str = "AES256",
    ) -> None:
        self._bucket = bucket
        self._endpoint = endpoint_url
        self._now = clock_epoch_ms
        self._region = region
        self._sse = server_side_encryption
        # Credentials are passed rather than left to boto3's ambient discovery.
        # The fallback chain works on a cloud instance and, everywhere else,
        # fails with "unable to locate credentials" - a message about the
        # environment for what is actually a missing configuration value.
        self._session = aioboto3.Session(
            aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
        # Signature v4 with a virtual-host-style override off, because MinIO
        # and most S3-compatible stores are path-style. Getting this wrong
        # produces signature mismatches that look like credential problems.
        self._config = AioConfig(signature_version="s3v4", s3={"addressing_style": "path"})

    def _key(self, tenant: TenantId, session_id: SessionId, ref: str) -> str:
        """Object keys are tenant-prefixed.

        Not decoration: a bucket policy can then scope credentials per tenant,
        and a listing accident cannot cross the boundary NFR-013 draws.
        """
        return f"{tenant.value}/{session_id.value}/{ref}"

    async def put(
        self,
        tenant: TenantId,
        session_id: SessionId,
        kind: MediaKind,
        content_type: str,
        payload: bytes,
        expires_at_ms: int,
    ) -> StoredObject:
        digest = hashlib.sha256(payload).hexdigest()[:16]
        ref = EvidenceRef(f"{kind.value}:{session_id.value}:{digest}")
        key = self._key(tenant, session_id, f"{kind.value}-{digest}")

        async with self._client() as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=payload,
                ContentType=content_type,
                ServerSideEncryption=self._sse,
                Metadata={_EXPIRES_METADATA: str(expires_at_ms)},
            )

        return StoredObject(
            ref=ref,
            tenant=tenant,
            session_id=session_id,
            kind=kind,
            size_bytes=len(payload),
            content_type=content_type,
            expires_at_ms=expires_at_ms,
        )

    async def open(self, tenant: TenantId, ref: EvidenceRef) -> AsyncIterator[bytes]:
        key = await self._resolve(tenant, ref)
        if key is None:
            raise FileNotFoundError(f"no live object at {ref}")

        async def _stream() -> AsyncIterator[bytes]:
            async with self._client() as s3:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
                async for chunk in response["Body"].iter_chunks():
                    yield chunk

        return _stream()

    async def exists(self, tenant: TenantId, ref: EvidenceRef) -> bool:
        return await self._resolve(tenant, ref) is not None

    async def delete(self, tenant: TenantId, ref: EvidenceRef) -> bool:
        key = await self._resolve(tenant, ref)
        if key is None:
            return False
        async with self._client() as s3:
            await s3.delete_object(Bucket=self._bucket, Key=key)
        return True

    async def delete_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        keys = await self._list_keys(tenant, session_id)
        if not keys:
            return 0
        async with self._client() as s3:
            await s3.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True},
            )
        return len(keys)

    async def count_for_session(self, tenant: TenantId, session_id: SessionId) -> int:
        return len(await self._list_keys(tenant, session_id))

    async def sign_upload(
        self,
        tenant: TenantId,
        session_id: SessionId,
        kind: MediaKind,
        content_type: str,
        ttl_seconds: int,
    ) -> SignedUrl:
        key = self._key(tenant, session_id, f"{kind.value}-upload")
        async with self._client() as s3:
            url = await s3.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self._bucket,
                    "Key": key,
                    "ContentType": content_type,
                    "ServerSideEncryption": self._sse,
                },
                ExpiresIn=ttl_seconds,
            )
        return SignedUrl(url=url, expires_at_ms=self._now() + ttl_seconds * 1_000)

    async def revoke_signatures(self, tenant: TenantId, session_id: SessionId) -> None:
        """Invalidate outstanding links by removing what they point at.

        A presigned URL cannot be revoked - the signature stays valid until it
        expires, wherever it has been copied to. Deleting the objects is the
        only thing that actually invalidates one, and QA-04 asks for
        invalidation rather than eventual expiry.

        Safe to call before ``delete_for_session`` because both are idempotent
        over an empty prefix.
        """
        await self.delete_for_session(tenant, session_id)

    # -- internals --------------------------------------------------------

    def _client(self) -> Any:
        return self._session.client(
            "s3",
            endpoint_url=self._endpoint,
            region_name=self._region,
            config=self._config,
        )

    async def _list_keys(self, tenant: TenantId, session_id: SessionId) -> list[str]:
        prefix = f"{tenant.value}/{session_id.value}/"
        keys: list[str] = []
        async with self._client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                keys.extend(item["Key"] for item in page.get("Contents", []))
        return keys

    async def _resolve(self, tenant: TenantId, ref: EvidenceRef) -> str | None:
        """Find a live object, treating an expired one as absent.

        The retention deadline is checked here rather than left to a lifecycle
        rule, so a broken sweeper cannot let an object outlive its policy from
        a reader's point of view.
        """
        _kind, session_value, digest = ref.value.split(":", 2)
        key = self._key(tenant, SessionId(session_value), f"{_kind}-{digest}")

        async with self._client() as s3:
            try:
                head = await s3.head_object(Bucket=self._bucket, Key=key)
            except Exception:
                return None

        expires_at = head.get("Metadata", {}).get(_EXPIRES_METADATA)
        if expires_at is not None and self._now() >= int(expires_at):
            return None
        return key
