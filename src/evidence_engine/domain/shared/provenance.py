"""Provenance: what produced this, under which rules.

NFR-014 requires every derived event to expose its source modality, interval,
confidence, model version, taxonomy version and evidence reference. NFR-015
requires that the same input, artifact, configuration and seed reproduce an
equivalent result. Neither is achievable if the versions live in a config file
somewhere and the result only carries numbers, because six months later nobody
can tell whether a metric moved because the student improved or because a
threshold was retuned - the exact failure the risk table names as "model
updates break longitudinal comparison".

So provenance is attached to the evidence itself, not to the deployment. That
includes the seed, which until now was named in this docstring and nowhere in
the code. It is uninteresting while every runtime in the tree is deterministic,
and that is exactly why recording it is cheap today: the day a stochastic
runtime lands, every result already in the ledger was produced under a seed
nobody wrote down, and no migration can recover it afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evidence_engine.domain.shared.errors import (
    DomainError,
    FabricatedValue,
    InvalidIdentifier,
)
from evidence_engine.domain.shared.identifiers import (
    ConfigurationSnapshotId,
    EvidenceRef,
    ModelVersionId,
)

#: The largest seed the engine will accept.
#:
#: ``BIGINT`` holds far more and Python holds arbitrarily more, but the seed is
#: published in the evidence document as a JSON number and the consuming
#: application parses it with ``JSON.parse``, which produces an IEEE-754
#: double. Above 2**53 two distinct seeds become the same number in transit, so
#: a reproduction would run under a seed the original never used and still
#: report a match. This constructor is the last place the two values are
#: distinguishable, which is why the refusal is here rather than in a note.
MAX_SEED = 2**53 - 1


class ProvenanceViolation(DomainError):
    """A provenance record would have claimed more than it can support."""


class Modality(StrEnum):
    """Which channel of observation produced a piece of evidence.

    Recorded on every event because QA-02 requires one modality to fail
    without invalidating the other, and that is only reportable if each finding
    says where it came from. A missing camera must be legible as a missing
    camera, never as a change in performance.
    """

    AUDIO = "audio"
    VIDEO = "video"
    #: Derived from two or more modalities - only co-occurrences (FR-027).
    MULTIMODAL = "multimodal"


@dataclass(frozen=True, slots=True, order=True)
class SemanticVersion:
    """A published, immutable version of a taxonomy, schema or threshold set.

    US-008 requires published versions to be immutable and historical results
    to be unaffected by later default changes. Comparison is defined so that
    the compatibility question NFR-017 asks - "is this consumer still within
    one major version?" - has an answer in code rather than in a changelog.
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise InvalidIdentifier(f"version components must be non-negative: {self}")

    @classmethod
    def parse(cls, text: str) -> SemanticVersion:
        parts = text.strip().split(".")
        if len(parts) != 3:
            raise InvalidIdentifier(f"expected 'major.minor.patch', got {text!r}")
        try:
            major, minor, patch = (int(p) for p in parts)
        except ValueError as exc:
            raise InvalidIdentifier(f"non-numeric version component in {text!r}") from exc
        return cls(major, minor, patch)

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """Backward compatibility within one major version (NFR-017)."""
        return self.major == other.major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class UnseededReason(StrEnum):
    """Why no seed is recorded.

    Two members, and collapsing them into one is the failure this enum exists
    to prevent. ``DETERMINISTIC_RUNTIME`` says the result is reproducible and
    there was nothing to seed; ``NOT_RECORDED`` says the run may not be
    reproducible at all and nobody can tell. A single "no seed" case would make
    the second read like the first, which is a claim of reproducibility the
    engine cannot support.

    §7.4 requires unknown enum values not to crash consumers, so adding a
    member later - a stochastic runtime that will not disclose its seed - is a
    compatible change.
    """

    #: The runtime makes the same choices on the same input, so there was
    #: nothing to seed. Every runtime in this tree today is of that kind.
    DETERMINISTIC_RUNTIME = "deterministic_runtime"
    #: Produced before the seed was recorded, or by something that reported
    #: none. This run cannot be reproduced from this record.
    NOT_RECORDED = "not_recorded"


@dataclass(frozen=True, slots=True)
class Seeded:
    """The seed the run's stochastic choices were actually made under.

    ``int``, not ``str`` or ``bytes``, because every seeding API a runtime
    could plausibly arrive with - ``random.Random``, ``numpy.random.
    default_rng``, ``torch.manual_seed`` - takes an integer. A string seed
    would need a hashing convention to become one, that convention would live
    somewhere other than the recorded value, and two deployments that disagreed
    about it would reproduce different results from the same record. The corpus
    partitioner already settled on ``int`` for the same reason.

    Non-negative because ``numpy.random.default_rng`` refuses a negative seed
    while ``torch.manual_seed`` accepts one: the intersection of what a future
    runtime will take is the non-negative range, and a seed the ledger records
    but a runtime rejects is a reproduction that cannot be run at all.
    """

    value: int

    def __post_init__(self) -> None:
        # Two refusals rather than one range check, because they fail for
        # unrelated reasons and a single message would give the wrong one.
        if self.value < 0:
            raise ProvenanceViolation(
                f"seed {self.value} is negative; a runtime seeded through numpy "
                "would refuse it, so this record would describe a reproduction "
                "nobody can run (NFR-015)"
            )
        if self.value > MAX_SEED:
            raise ProvenanceViolation(
                f"seed {self.value} exceeds {MAX_SEED}; above 2**53 a JSON consumer "
                "reads two distinct seeds as one number, so a reproduction would run "
                "under a seed the original never used and still report a match "
                "(NFR-015)"
            )

    @property
    def is_recorded(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class Unseeded:
    """No seed is recorded here, and why.

    There is no ``value`` attribute, for the reason ``Unavailable`` has none.
    ``seed or 0`` is the one-line change somebody makes to get a reproduction
    script to run, and 0 is a real seed: the rerun would complete, report a
    match against a result produced under some other seed, and the
    disagreement would never surface. Reaching for the number raises instead.
    """

    reason: UnseededReason

    @property
    def is_recorded(self) -> bool:
        return False

    def __getattr__(self, name: str) -> object:
        # Only consulted for attributes that do not exist. ``value`` is the one
        # that matters, and it gets a domain error naming the rule rather than
        # an AttributeError a caller might reasonably decide to swallow.
        if name == "value":
            raise FabricatedValue(
                f"cannot read a seed from an unseeded provenance "
                f"(reason={self.reason.value}); NFR-015 forbids substituting one"
            )
        raise AttributeError(name)


#: A provenance record either names the seed it ran under or names the reason
#: it does not. ``int | None`` would collapse "seed 0" and "no seed" into two
#: values a single ``or`` can confuse, which is the distinction NFR-015 turns
#: on.
type Seed = Seeded | Unseeded


@dataclass(frozen=True, slots=True)
class Provenance:
    """The full answer to "where did this finding come from?".

    Every field is required. An optional model version would be filled in as
    ``None`` on exactly the code paths where it matters most - a fallback
    adapter, a degraded run - and the result would claim provenance it does not
    have. Component §5 puts it plainly for the Evidence Ledger: preserve
    immutable provenance.

    ``seed`` is the one field with a default, and the default is the reason the
    rule above still holds. A ``Provenance`` built without naming a seed is a
    record that did not record one, which is precisely what
    ``NOT_RECORDED`` says - so the default is a true statement rather than a
    filler. The rejected alternative was defaulting to
    ``DETERMINISTIC_RUNTIME``: it is true of every runtime shipping today and
    becomes false, silently and for every event, on the day a stochastic one
    forgets to pass its seed. An understated record is recoverable by reading
    the runtime; an overstated one is a false claim of reproducibility that
    reads exactly like a true one.
    """

    modality: Modality
    model_version: ModelVersionId
    taxonomy_version: SemanticVersion
    configuration: ConfigurationSnapshotId
    evidence_ref: EvidenceRef
    seed: Seed = Unseeded(UnseededReason.NOT_RECORDED)
