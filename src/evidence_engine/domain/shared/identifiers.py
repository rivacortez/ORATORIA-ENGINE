"""Typed identifiers.

Every reference in this system is a distinct type rather than a bare ``str``.
The reason is concrete: §7.4 requires that event identifiers stay stable across
the provisional-to-final reconciliation, and §8 separates raw evidence from
derived results. Both properties are enforced by code that passes identifiers
around, and a single ``str`` alias lets a run id land in a session slot without
anything complaining until the wrong row is deleted.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Self

from evidence_engine.domain.shared.errors import InvalidIdentifier


@dataclass(frozen=True, slots=True)
class Identifier:
    """A non-empty, non-blank opaque reference."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise InvalidIdentifier(f"{type(self).__name__} cannot be empty or blank")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def generate(cls) -> Self:
        """Mint a fresh identifier.

        Returns ``Self``, not ``Identifier``: ``SessionId.generate()`` has to
        type as a ``SessionId``, otherwise every call site would need a cast
        and the nominal typing this module exists for would evaporate at the
        one place identifiers are created.

        Randomness is taken here rather than injected because these values are
        opaque references with no behaviour attached. Reproducibility (QA-05)
        is defined over event labels and measurements, not over the surrogate
        keys pointing at them; two reruns of the same input are equivalent when
        their evidence matches, regardless of the ids assigned to it.
        """
        return cls(str(uuid.uuid4()))


@dataclass(frozen=True, slots=True)
class ApplicationId(Identifier):
    """A registered consuming application (§8 ``ClientApplication``)."""


@dataclass(frozen=True, slots=True)
class TenantId(Identifier):
    """The isolation boundary every data access is scoped by (NFR-013)."""


@dataclass(frozen=True, slots=True)
class ApiKeyId(Identifier):
    """A key record. Never the secret itself - only its hash is persisted."""


@dataclass(frozen=True, slots=True)
class SessionId(Identifier):
    """One analysis session: a single capture from creation to deletion."""


@dataclass(frozen=True, slots=True)
class RunId(Identifier):
    """One processing run over a session's media (§8 ``ProcessingRun``).

    A session can be processed more than once - a model was promoted, a batch
    job resumed, an evaluation item was replayed - and every derived record
    hangs off the run rather than the session so that the two results stay
    separable and comparable instead of overwriting each other.
    """


@dataclass(frozen=True, slots=True)
class EventId(Identifier):
    """A speech or visual event.

    §7.4: "Event identifiers are stable across provisional-to-final
    reconciliation." A consumer that rendered a partial event must be able to
    replace it in place when the final arrives, which is only possible if the
    id it saw first is the id it sees last.
    """


@dataclass(frozen=True, slots=True)
class TokenId(Identifier):
    """A single word hypothesis in the verbatim transcript."""


@dataclass(frozen=True, slots=True)
class EvidenceRef(Identifier):
    """A pointer to the material an event was derived from.

    NFR-014 requires every derived event to expose an evidence reference. This
    is that reference: it addresses a stored interval of media or features, and
    survives in the ledger even after the raw media it pointed at is deleted,
    so that an audit can tell "this was measured and then erased" apart from
    "this was never measured".
    """


@dataclass(frozen=True, slots=True)
class ModelVersionId(Identifier):
    """A specific model artifact in the registry (§8 ``ModelVersion``)."""


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshotId(Identifier):
    """The frozen thresholds and windows a run was executed under."""


@dataclass(frozen=True, slots=True)
class MessageId(Identifier):
    """A wire message. Used to make duplicate client messages idempotent."""
