"""Application-level failures.

Distinct from domain errors. A domain error means the model's own rules were
about to be broken and the engine must refuse. These mean a caller asked for
something that is not allowed or not there - conditions the API surface turns
into a status code rather than a fault.

Keeping them apart matters at the boundary: §7.4 requires a stable error
contract, and a layer that mapped every exception onto one status would make
"you lack the scope" indistinguishable from "the fusion window is negative".
"""

from __future__ import annotations


class ApplicationError(Exception):
    """Base class for a request the application declines to serve."""


class NotAuthenticated(ApplicationError):
    """No valid credential was presented."""


class NotAuthorized(ApplicationError):
    """The caller is known but lacks the scope this operation needs (FR-003)."""


class SessionNotFound(ApplicationError):
    """No such session *within the caller's tenant*.

    Raised identically for a session that never existed and one belonging to
    another tenant. NFR-013 makes tenants invisible to each other, and a
    distinguishable error would let a caller enumerate the difference.
    """


class ResultNotReady(ApplicationError):
    """The session has not produced its evidence document yet."""


class QuotaExceeded(ApplicationError):
    """The application is over its configured limit (NFR-023)."""

    def __init__(self, message: str, retry_after_seconds: int = 0) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class SessionBusy(ApplicationError):
    """Another handler already holds the processing lease for this session."""


class BackpressureRequired(ApplicationError):
    """The bounded queue is full; the caller must slow down (FR-010).

    An error rather than a silent drop. US-012 makes backpressure explicit in
    the contract, and dropping chunks quietly would resurface later as an
    unexplained gap in the evidence that FR-008 would report as data loss.
    """

    def __init__(self, message: str, queue_depth: int) -> None:
        super().__init__(message)
        self.queue_depth = queue_depth
