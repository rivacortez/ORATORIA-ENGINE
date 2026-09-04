"""Redis: ephemeral stream state, session leases and quota windows.

Everything here expires. That is not a tidiness preference — each of the three
uses depends on it.

*Leases* must expire so a handler that dies does not lock its session out
forever. The TTL is the recovery mechanism, not a cleanup detail.

*Quota windows* must roll, so an application is throttled for a minute rather
than starved permanently by a counter that only grows.

*Stream snapshots* must expire so a session abandoned mid-presentation does not
occupy memory indefinitely.

Nothing a published result depends on lives here. Redis is a cache; anything in
it must be reconstructible, and the evidence itself is in PostgreSQL.
"""

from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from evidence_engine.application.ports.platform import QuotaDecision
from evidence_engine.application.ports.streaming import StreamSnapshot
from evidence_engine.domain.sessions.sequencing import SequenceGap
from evidence_engine.domain.shared.identifiers import ApplicationId, SessionId

_STATE_PREFIX = "stream:state:"
_LEASE_PREFIX = "stream:lease:"
_QUOTA_PREFIX = "quota:"


class RedisStreamState:
    """Per-session ephemeral state with expiring leases."""

    def __init__(self, client: Redis) -> None:
        self._redis = client

    async def load(self, session_id: SessionId) -> StreamSnapshot | None:
        raw = await self._redis.get(f"{_STATE_PREFIX}{session_id.value}")
        if raw is None:
            return None
        payload: dict[str, Any] = json.loads(raw)
        return StreamSnapshot(
            session_id=SessionId(payload["session_id"]),
            highest_audio_seq=payload["highest_audio_seq"],
            highest_video_seq=payload["highest_video_seq"],
            finalized_through_ms=payload["finalized_through_ms"],
            queue_depth=payload["queue_depth"],
            gaps=tuple(SequenceGap(first, last) for first, last in payload.get("gaps", [])),
        )

    async def save(self, snapshot: StreamSnapshot, ttl_seconds: int) -> None:
        payload = {
            "session_id": snapshot.session_id.value,
            "highest_audio_seq": snapshot.highest_audio_seq,
            "highest_video_seq": snapshot.highest_video_seq,
            "finalized_through_ms": snapshot.finalized_through_ms,
            "queue_depth": snapshot.queue_depth,
            "gaps": [[gap.first_missing, gap.last_missing] for gap in snapshot.gaps],
        }
        await self._redis.set(
            f"{_STATE_PREFIX}{snapshot.session_id.value}",
            json.dumps(payload),
            ex=ttl_seconds,
        )

    async def clear(self, session_id: SessionId) -> None:
        await self._redis.delete(
            f"{_STATE_PREFIX}{session_id.value}", f"{_LEASE_PREFIX}{session_id.value}"
        )

    async def acquire_lease(self, session_id: SessionId, ttl_seconds: int) -> bool:
        """Claim exclusive processing, atomically.

        ``SET key value NX EX ttl`` in one round trip. A GET followed by a SET
        would have a window between them, and two handlers landing in it would
        both believe they own the session — which is exactly the interleaving
        that manufactures phantom chunk gaps.
        """
        acquired = await self._redis.set(
            f"{_LEASE_PREFIX}{session_id.value}", "held", nx=True, ex=ttl_seconds
        )
        return bool(acquired)

    async def release_lease(self, session_id: SessionId) -> None:
        await self._redis.delete(f"{_LEASE_PREFIX}{session_id.value}")


class RedisQuotaGuard:
    """Fixed-window quotas per application and unit (NFR-023)."""

    def __init__(
        self,
        client: Redis,
        limits: dict[str, int] | None = None,
        window_seconds: int = 60,
    ) -> None:
        self._redis = client
        self._limits = dict(limits or {})
        self._window_seconds = window_seconds

    async def check(self, application: ApplicationId, unit: str, amount: int = 1) -> QuotaDecision:
        limit = self._limits.get(unit)
        if limit is None:
            # No configured limit means unlimited. NFR-023 asks for quotas to
            # be configurable, not mandatory, and a missing configuration
            # should not silently block traffic.
            return QuotaDecision(allowed=True, remaining=-1)

        key = await self._key(application, unit)
        used = int(await self._redis.get(key) or 0)
        remaining = limit - used
        if used + amount > limit:
            ttl = await self._redis.ttl(key)
            return QuotaDecision(
                allowed=False,
                remaining=max(remaining, 0),
                retry_after_seconds=max(ttl, 1),
                reason=f"{unit} quota of {limit} per window exhausted",
            )
        return QuotaDecision(allowed=True, remaining=remaining - amount)

    async def consume(self, application: ApplicationId, unit: str, amount: int = 1) -> None:
        """Increment and set the window TTL in one pipeline.

        The EXPIRE is unconditional rather than "only on first increment": a
        conditional version needs a read to decide, and a crash between the
        INCR and a later EXPIRE would leave a counter with no TTL — an
        application permanently at its limit with nothing to explain why.
        """
        key = await self._key(application, unit)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incrby(key, amount)
            pipe.expire(key, self._window_seconds)
            await pipe.execute()

    async def _key(self, application: ApplicationId, unit: str) -> str:
        """Derive the window key from *Redis server* time, not local time.

        Several replicas share one quota. If each computed the window from its
        own clock, a few seconds of drift would put them in different windows
        and an application would get its allowance once per replica.
        """
        seconds, _microseconds = await self._redis.time()
        window = int(seconds) // self._window_seconds
        return f"{_QUOTA_PREFIX}{application.value}:{unit}:{window}"
