"""Stream tokens: HMAC-signed, self-contained, self-expiring.

The token has to survive being handed to a browser (§6.1 step 2) and be
verifiable by whichever process holds the socket. That rules out an opaque
random string in a shared table - it puts a lookup on the connect path and
becomes meaningless after a cache flush - and obviously rules out anything the
client could mint.

So the grant is signed rather than stored: a compact JSON payload, base64url
encoded, with an HMAC-SHA256 tag over it. Any process holding the signing key
verifies it with no I/O, and a client cannot alter the session, the tenant, the
scopes or the expiry without invalidating the tag.

This is deliberately not a JWT. A JWT would bring an algorithm field the client
can influence - the ``alg: none`` family of mistakes - plus a dependency and a
spec surface far wider than "sign these six fields". One algorithm, chosen
here, is the whole point.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from hashlib import sha256
from typing import Any

from evidence_engine.application.ports.platform import Scope
from evidence_engine.application.ports.tokens import StreamGrant
from evidence_engine.domain.shared.errors import InvalidIdentifier
from evidence_engine.domain.shared.identifiers import (
    ApiKeyId,
    ApplicationId,
    SessionId,
    TenantId,
)

MIN_SIGNING_KEY_LENGTH = 32


class HmacStreamTokenMinter:
    """Signs and verifies short-lived stream grants."""

    def __init__(self, signing_key: str) -> None:
        if len(signing_key) < MIN_SIGNING_KEY_LENGTH:
            raise ValueError(
                f"the stream-token signing key must be at least {MIN_SIGNING_KEY_LENGTH} "
                "characters; a short key makes the grant forgeable, which defeats the "
                "only thing this token does"
            )
        self._key = signing_key.encode("utf-8")

    def mint(self, grant: StreamGrant) -> str:
        payload = _encode(
            {
                "sid": grant.session_id.value,
                "tid": grant.tenant.value,
                "aid": grant.application.value,
                "kid": grant.key_id.value,
                "exp": grant.expires_at_ms,
                "scp": sorted(scope.value for scope in grant.scopes),
            }
        )
        return f"{payload}.{self._sign(payload)}"

    def verify(self, token: str, now_ms: int) -> StreamGrant | None:
        try:
            payload, signature = token.rsplit(".", 1)
        except ValueError:
            return None

        # Signature first, always. Parsing attacker-controlled JSON before
        # authenticating it means the parser is part of the attack surface.
        if not hmac.compare_digest(signature, self._sign(payload)):
            return None

        try:
            claims = _decode(payload)
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return None

        expires_at_ms = claims.get("exp")
        if not isinstance(expires_at_ms, int) or now_ms >= expires_at_ms:
            return None

        try:
            return StreamGrant(
                session_id=SessionId(str(claims["sid"])),
                tenant=TenantId(str(claims["tid"])),
                application=ApplicationId(str(claims["aid"])),
                key_id=ApiKeyId(str(claims["kid"])),
                expires_at_ms=expires_at_ms,
                scopes=frozenset(Scope(value) for value in claims.get("scp", [])),
            )
        except (KeyError, ValueError, InvalidIdentifier):
            # A validly signed but malformed grant means this service minted
            # something wrong, not that the client forged anything. It is still
            # not usable, and saying so is better than half-building it.
            return None

    def _sign(self, payload: str) -> str:
        digest = hmac.new(self._key, payload.encode("utf-8"), sha256).digest()
        return _b64(digest)


def _encode(claims: dict[str, Any]) -> str:
    # Sorted keys and no whitespace: the payload has to encode identically on
    # every process, or a token minted by one replica fails verification on
    # another.
    raw = json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _b64(raw)


def _decode(payload: str) -> dict[str, Any]:
    padding = "=" * (-len(payload) % 4)
    raw = base64.urlsafe_b64decode(payload + padding)
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("token payload is not an object")
    return decoded


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
