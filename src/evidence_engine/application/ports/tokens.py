"""Short-lived stream tokens.

§6.1 step 2: creating a session returns "a short-lived stream token". It exists
because the API key must not follow the media into a browser. The key is a
long-lived server-side credential with scopes attached (FR-003); shipping it to
a page so the page can open a WebSocket would turn one leaked bundle into
standing access to every session the application can create.

The token is bound to one session and carries its own expiry, so a leak is
bounded in both blast radius and time.
"""

from __future__ import annotations

from typing import Protocol

from evidence_engine.domain.shared.identifiers import SessionId


class StreamTokenMinter(Protocol):
    """Issues and verifies the token that authorizes one session's stream."""

    def mint(self, session_id: SessionId, expires_at_ms: int) -> tuple[str, int]:
        """Return the token and the epoch-millisecond instant it expires."""
        ...

    def verify(self, token: str, now_ms: int) -> SessionId | None:
        """Resolve a presented token to its session, or ``None``.

        One return value for expired, malformed and forged tokens. Telling them
        apart tells a prober which of their guesses was structurally valid.
        """
        ...
