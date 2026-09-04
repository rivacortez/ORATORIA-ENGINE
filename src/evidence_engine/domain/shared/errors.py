"""Domain errors.

A domain error means a rule of the model was violated, not that a request was
malformed. The distinction matters at the boundary: a malformed request is a
400 the caller can fix, while a domain error that reaches an adapter means the
engine was about to publish evidence that contradicts its own specification,
and the correct response is to fail rather than to emit it.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every violation of a domain rule."""


class InvalidIdentifier(DomainError):
    """An identifier was empty, blank or otherwise unusable as a reference."""


class InvalidInterval(DomainError):
    """A time interval is negative, inverted or carries a negative tolerance."""


class InvalidConfidence(DomainError):
    """A confidence value fell outside the closed unit interval."""


class FabricatedValue(DomainError):
    """Something tried to read a number out of an unavailable measurement.

    FR-025 and architectural driver 4: unavailable or uncertain evidence must
    never be represented as zero or as a confirmed finding. Raising here is
    deliberate. A silent ``0.0`` would travel all the way into a student's
    report as "you scored zero on this", which is a different and much worse
    statement than "this could not be measured".
    """


class ProhibitedLabel(DomainError):
    """An event type outside the published observable taxonomy was used.

    NFR-024 and FR-023: no endpoint or schema may expose emotion, anxiety,
    personality, deception or clinical diagnoses. The taxonomy is an allowlist
    rather than a denylist, because a denylist only stops the labels somebody
    remembered to forbid.
    """


class IllegalSessionTransition(DomainError):
    """A session was asked to move to a state its lifecycle does not allow."""


class EvidenceLedgerViolation(DomainError):
    """An attempt to rewrite finalized evidence.

    §6.1 step 9: stable windows are emitted as final "without rewriting already
    finalized history". A consumer that has already displayed a finalized event
    cannot be told later that it never happened.
    """
