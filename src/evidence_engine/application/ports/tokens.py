"""Short-lived stream grants.

§6.1 step 2: creating a session returns "a short-lived stream token". It exists
because the API key must not follow the media into a browser. The key is a
long-lived server-side credential with full scopes (FR-003); shipping it to a
page so the page can open a WebSocket would turn one leaked bundle into
standing access to create sessions and delete evidence.

The token carries the whole grant - session, tenant, application, reduced
scopes, expiry - rather than just a session id, and it is signed. Two
consequences, and both matter:

*NFR-013 holds with no unscoped lookup.* If the token named only a session, the
socket handler would have to find that session without knowing its tenant, and
the one query in the system that ignores the isolation boundary would sit on
the path that handles unauthenticated input. Carrying the tenant inside a
signed grant removes the need for that query to exist.

*Least privilege.* The scopes in the grant are the streaming subset, not the
issuing key's. A leaked stream token can push audio into one session for a few
minutes; it cannot create another session or delete anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evidence_engine.application.ports.platform import Scope
from evidence_engine.domain.shared.identifiers import (
    ApiKeyId,
    ApplicationId,
    SessionId,
    TenantId,
)

#: What a stream token is allowed to do. Capture control and nothing else:
#: completing a session goes through this too, because §7.2's
#: ``session.complete`` arrives on the socket.
STREAM_SCOPES: frozenset[Scope] = frozenset({Scope.SESSIONS_WRITE, Scope.SESSIONS_READ})


@dataclass(frozen=True, slots=True)
class StreamGrant:
    """Everything the socket handler needs, signed and self-contained."""

    session_id: SessionId
    tenant: TenantId
    application: ApplicationId
    key_id: ApiKeyId
    expires_at_ms: int
    scopes: frozenset[Scope] = STREAM_SCOPES


class StreamTokenMinter(Protocol):
    """Issues and verifies the grant that authorizes one session's stream."""

    def mint(self, grant: StreamGrant) -> str:
        """Sign a grant into a token the client can hold."""
        ...

    def verify(self, token: str, now_ms: int) -> StreamGrant | None:
        """Resolve a presented token, or ``None``.

        One return value for expired, malformed and forged tokens.
        Distinguishing them tells a prober which of their guesses was
        structurally valid.
        """
        ...
