"""Stable event identifiers across provisional-to-final reconciliation.

§7.4 requires event identifiers to stay stable through reconciliation, and
US-012 requires reconnection not to duplicate finalized events. Both need the
same thing: an id that a detector arrives at independently every time it sees
the same evidence, rather than one minted fresh per emission.

A counter cannot do this. A streaming recognizer re-decodes its active window
several times before finalizing, so the same filled pause is reported three or
four times; with a counter the client sees three separate events, renders three
markers and then has no way to collapse them. A random id has the same problem.
After a reconnect it is worse: the whole window is reprocessed and every prior
event is duplicated.

So the id is derived from the evidence itself - which run, which class, where
on the timeline - and re-derives identically on every pass. Two genuinely
different events cannot collide because their positions differ; the same event
seen twice cannot diverge because nothing else feeds the hash.
"""

from __future__ import annotations

import hashlib

from evidence_engine.domain.shared.identifiers import EventId, RunId

#: Positions are bucketed before hashing. Boundary estimates move by a few
#: milliseconds between passes as the model gains right context, and an id
#: derived from the exact start would change with them - defeating the whole
#: purpose. The bucket is coarse enough to absorb that jitter and fine enough
#: that two distinct events of the same class do not land in one: NFR-004 sets
#: the expected boundary error at 250 ms, so events closer together than this
#: are not reliably separable anyway.
POSITION_BUCKET_MS = 100


def derive_event_id(
    run_id: RunId,
    event_type: str,
    start_ms: int,
    *,
    discriminator: str = "",
) -> EventId:
    """Derive the stable id for one event.

    ``discriminator`` separates two events of the same class in the same bucket
    - two different words repeated in the same 100 ms, say. Callers that have
    such a distinction pass it; those that do not leave it empty rather than
    inventing one, because a discriminator that varies between passes would
    reintroduce exactly the instability this function exists to remove.

    Scoped by run, not by session. §8 keeps runs separable so that reprocessing
    a session under a promoted model produces a comparable second result rather
    than overwriting the first.
    """
    bucket = start_ms // POSITION_BUCKET_MS
    material = f"{run_id.value}|{event_type}|{bucket}|{discriminator}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return EventId(f"ev_{digest}")


def derive_token_id(run_id: RunId, start_ms: int, raw_text: str) -> str:
    """Derive the stable id for one word token.

    Includes the text, unlike events. A word hypothesis genuinely changes
    identity when the recognizer changes its mind - "treinta" becoming
    "trescientos" is a different word, not a revision of the same one - and the
    transcript's own ``revise`` path handles the case where a caller does want
    to keep the identity.
    """
    bucket = start_ms // POSITION_BUCKET_MS
    material = f"{run_id.value}|{bucket}|{raw_text}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"tok_{digest}"
