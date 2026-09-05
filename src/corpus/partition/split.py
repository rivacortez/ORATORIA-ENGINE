"""Splitting the corpus by speaker, and checking that it worked.

`BASELINES.md` §4 says speaker independence is *"the single most common way a
speech result becomes meaningless, and the check is mechanical rather than a
matter of care"*. This is the mechanism. Two decisions carry the module.

*The unit is the speaker, not the recording.* Assigning recordings and hoping
no speaker straddles a boundary is the failure mode itself: a model that has
heard a speaker in training and is scored on that speaker in held-out is being
scored on memorisation. Speakers are assigned; their recordings follow. Under
that construction the independence check cannot fail - which is the point, and
it is asserted anyway, because a guarantee nobody re-derives is a guarantee
that survives exactly until the next refactor.

*The assignment is stratified, not random.* On a thesis-scale corpus - tens of
speakers, nine classes, some of them rare - random speaker assignment routinely
lands zero instances of a rare class in the held-out set. Its per-class F1 is
then undefined, and an undefined F1 is not a low score, it is a missing
measurement. Three of nine per-class figures missing is a results table that
cannot answer the question it was built for.

So speakers are placed by a greedy iterative stratification: the speaker
carrying the rarest class goes first, into the partition that most needs it.
This is the shape of Sechidis, Tsoumakas and Vlahavas (2011) for multi-label
data, reduced to what this corpus needs and short enough to read. It does not
guarantee coverage - with four speakers producing `cut_off` and three
partitions, something has to be empty - so the plan reports coverage per class
per partition and refuses to call itself usable when a P0 class is missing
somewhere.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from corpus.partition.inventory import CorpusInventory, SpeakerProfile
from corpus.schema.validation import Finding, Severity
from evidence_engine.domain.shared.taxonomy import p0_speech_events


class InvalidSplit(Exception):
    """Split parameters that cannot produce partitions."""


class Partition(StrEnum):
    """Where a speaker's recordings go.

    Three, not two. A development partition exists so that thresholds and
    decoding choices have somewhere to be tuned that is not the held-out set;
    without it "we only looked at held-out once" is a claim resting on
    everyone's memory.
    """

    TRAIN = "train"
    DEV = "dev"
    HELD_OUT = "held_out"


#: Default shares. Held-out is the largest of the two evaluation partitions
#: because it carries every reported number; dev only has to be big enough to
#: make a threshold choice on.
DEFAULT_SHARES: Mapping[Partition, float] = MappingProxyType(
    {Partition.TRAIN: 0.70, Partition.DEV: 0.10, Partition.HELD_OUT: 0.20}
)


@dataclass(frozen=True, slots=True)
class PartitionContents:
    """One partition, after assignment."""

    partition: Partition
    speakers: tuple[str, ...]
    recording_ids: tuple[str, ...]
    duration_ms: int
    class_counts: Mapping[str, int]

    @property
    def speaker_count(self) -> int:
        return len(self.speakers)

    def covers(self, event_type: str) -> bool:
        return self.class_counts.get(event_type, 0) > 0


@dataclass(frozen=True, slots=True)
class PartitionPlan:
    """The assignment, its coverage, and what is wrong with it.

    Graded findings rather than a boolean, matching `ValidationReport`: a
    partition missing a P0 class is an error, an unbalanced one is a warning,
    and the difference decides whether recording continues or the split is
    simply noted.
    """

    seed: int
    shares: Mapping[str, float]
    partitions: tuple[PartitionContents, ...]
    findings: tuple[Finding, ...]

    def of(self, partition: Partition) -> PartitionContents:
        for contents in self.partitions:
            if contents.partition is partition:
                return contents
        raise KeyError(partition)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def is_usable(self) -> bool:
        """Whether this split can carry the numbers Phase 3 will report."""
        return not self.errors

    def speaker_of(self, pseudonym: str) -> Partition | None:
        for contents in self.partitions:
            if pseudonym in contents.speakers:
                return contents.partition
        return None


def split(
    corpus: CorpusInventory,
    *,
    seed: int = 0,
    shares: Mapping[Partition, float] = DEFAULT_SHARES,
) -> PartitionPlan:
    """Assign every speaker to exactly one partition.

    Deterministic: the same corpus and seed produce the same plan, on any
    machine and in any Python version. The seed does not feed a random number
    generator - it rotates the deterministic speaker ordering, so that a corpus
    can be re-split without the result depending on hash ordering or on
    whatever `random` did that release.
    """
    _require_valid_shares(shares)
    if not corpus.speakers:
        raise InvalidSplit("a corpus with no speakers cannot be split")

    targets = _speaker_targets(len(corpus.speakers), shares)
    assignment = _assign(corpus, seed, targets)

    partitions = tuple(
        _contents(partition, assignment.get(partition, ()), corpus) for partition in Partition
    )
    return PartitionPlan(
        seed=seed,
        shares=MappingProxyType({p.value: s for p, s in shares.items()}),
        partitions=partitions,
        findings=tuple(_findings(partitions, corpus, targets)),
    )


def _require_valid_shares(shares: Mapping[Partition, float]) -> None:
    """Shares that do not describe a partition of the corpus.

    Checked because every one of these produces a plan that renders: partitions
    appear, counts fill in, and some fraction of the corpus is silently in none
    of them or in a partition that was never asked for.
    """
    missing = [p for p in Partition if p not in shares]
    if missing:
        raise InvalidSplit(f"no share given for {[p.value for p in missing]}")
    for partition, share in shares.items():
        if not 0.0 < share < 1.0:
            raise InvalidSplit(
                f"share for {partition.value!r} must lie in (0, 1), got {share}. A "
                "partition with none of the corpus, or all of it, is not a partition."
            )
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-9:
        raise InvalidSplit(
            f"shares sum to {total}, not 1.0. The difference is the fraction of the "
            "corpus that would end up in no partition at all."
        )


def _speaker_targets(speaker_count: int, shares: Mapping[Partition, float]) -> dict[Partition, int]:
    """How many speakers each partition should get.

    Largest-remainder rather than rounding each share independently: rounding
    loses or invents a speaker, and on a corpus of forty that is 2.5% of the
    held-out set appearing or vanishing depending on the arithmetic.
    """
    exact = {p: speaker_count * shares[p] for p in Partition}
    targets = {p: int(exact[p]) for p in Partition}

    remaining = speaker_count - sum(targets.values())
    # Ties broken by partition name so the result does not depend on dict order.
    by_remainder = sorted(Partition, key=lambda p: (-(exact[p] - targets[p]), p.value))
    for partition in by_remainder[:remaining]:
        targets[partition] += 1

    # Every partition gets at least one speaker, taken from the largest. A
    # held-out set with no speakers in it is not a small held-out set.
    for partition in Partition:
        if targets[partition] == 0:
            donor = max(Partition, key=lambda p: (targets[p], p.value))
            if targets[donor] <= 1:
                raise InvalidSplit(
                    f"{speaker_count} speaker(s) cannot fill {len(list(Partition))} "
                    "partitions with at least one each. Speaker independence means a "
                    "speaker cannot be shared, so this is a recruitment answer, not a "
                    "split parameter."
                )
            targets[donor] -= 1
            targets[partition] += 1
    return targets


def _assign(
    corpus: CorpusInventory, seed: int, targets: Mapping[Partition, int]
) -> dict[Partition, list[SpeakerProfile]]:
    """Greedy iterative stratification over speakers.

    Speakers carrying the rarest class are placed first, while there is still
    freedom to place them well. A speaker placed last goes wherever there is
    room, and if that speaker was the only source of `cut_off` the damage is
    already done.
    """
    remaining_class_need = _class_targets(corpus, targets)
    remaining_speaker_slots = dict(targets)
    rarity = _rarity(corpus)
    assignment: dict[Partition, list[SpeakerProfile]] = {p: [] for p in Partition}

    for speaker in _ordering(corpus, seed):
        partition = _best_partition(speaker, rarity, remaining_class_need, remaining_speaker_slots)
        assignment[partition].append(speaker)
        remaining_speaker_slots[partition] -= 1
        for event_type, count in speaker.class_counts.items():
            need = remaining_class_need[partition]
            need[event_type] = need.get(event_type, 0) - count
    return assignment


def _class_targets(
    corpus: CorpusInventory, targets: Mapping[Partition, int]
) -> dict[Partition, dict[str, float]]:
    """How many instances of each class each partition should ideally hold.

    Proportional to its speaker share, not to a share of the instances: the
    thing being divided is speakers, and a partition holding 20% of the
    speakers should hold roughly 20% of each class if the split is doing its
    job.
    """
    total_speakers = sum(targets.values()) or 1
    return {
        partition: {
            event_type: count * targets[partition] / total_speakers
            for event_type, count in corpus.class_counts.items()
        }
        for partition in Partition
    }


def _rarity(corpus: CorpusInventory) -> dict[str, int]:
    """How many instances of each class the corpus holds, absent ones dropped.

    The scarcity signal both the ordering and the placement are driven by.
    Absent classes are excluded rather than mapped to zero: a class nobody
    produced would otherwise be the rarest thing every speaker carries.
    """
    return {event: count for event, count in corpus.class_counts.items() if count}


def _ordering(corpus: CorpusInventory, seed: int) -> list[SpeakerProfile]:
    """Speakers, rarest-class-carriers first, with a deterministic tie-break.

    The seed rotates the order within a tie group rather than shuffling
    everything. That keeps the stratification intact - the whole point is that
    the order is *not* arbitrary - while still letting a corpus be re-split to
    check that a result is not an artifact of one particular assignment.
    """
    rarity = {event_type: count for event_type, count in corpus.class_counts.items() if count}

    def key(speaker: SpeakerProfile) -> tuple[int, int, str]:
        carried = [rarity[e] for e in speaker.class_counts if e in rarity]
        # A speaker carrying nothing rare sorts last: they are the ones there
        # is no cost to placing wherever there is room.
        rarest = min(carried, default=1 << 30)
        return (rarest, -speaker.event_count, speaker.pseudonym)

    ordered = sorted(corpus.speakers, key=key)
    offset = seed % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _best_partition(
    speaker: SpeakerProfile,
    rarity: Mapping[str, int],
    remaining_class_need: Mapping[Partition, Mapping[str, float]],
    remaining_speaker_slots: Mapping[Partition, int],
) -> Partition:
    """The partition that most needs the scarcest thing this speaker carries.

    Scored on the *rarest* class the speaker has, not on the largest
    outstanding need. Scoring by the largest need was a bug and an instructive
    one: `filled_pause` outnumbers `cut_off` by an order of magnitude, so the
    common class dominated every comparison and the rare one - the only class
    whose placement is actually at risk - had no influence at all. The
    partitions came out proportional in bulk and short of `cut_off` in dev,
    which is exactly the outcome stratification exists to prevent.

    Ties fall through to total outstanding need, then to remaining capacity,
    then to the partition name. Never to iteration order.
    """
    open_partitions = [p for p in Partition if remaining_speaker_slots[p] > 0] or list(Partition)

    carried = [event for event in speaker.class_counts if event in rarity]
    rarest = min(carried, key=lambda event: (rarity[event], event), default=None)

    def score(partition: Partition) -> tuple[float, float, int, str]:
        needs = remaining_class_need[partition]
        return (
            -(needs.get(rarest, 0.0) if rarest is not None else 0.0),
            -sum(needs.get(event, 0.0) for event in speaker.class_counts),
            -remaining_speaker_slots[partition],
            partition.value,
        )

    return min(open_partitions, key=score)


def _contents(
    partition: Partition,
    speakers: Sequence[SpeakerProfile],
    corpus: CorpusInventory,
) -> PartitionContents:
    ordered = sorted(speakers, key=lambda s: s.pseudonym)
    counts: Counter[str] = Counter()
    for speaker in ordered:
        counts.update(speaker.class_counts)
    return PartitionContents(
        partition=partition,
        speakers=tuple(s.pseudonym for s in ordered),
        recording_ids=tuple(sorted(rid for s in ordered for rid in s.recording_ids)),
        duration_ms=sum(s.duration_ms for s in ordered),
        class_counts=MappingProxyType(dict(sorted(counts.items()))),
    )


def _findings(
    partitions: Sequence[PartitionContents],
    corpus: CorpusInventory,
    targets: Mapping[Partition, int],
) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(_independence(partitions, corpus))
    findings.extend(_coverage(partitions, corpus))
    findings.extend(_balance(partitions, targets))
    return findings


def _independence(
    partitions: Sequence[PartitionContents], corpus: CorpusInventory
) -> list[Finding]:
    """The guarantee the whole module exists for, re-derived rather than assumed.

    By construction a speaker is placed once, so this cannot fail today. It is
    checked anyway: the construction is the kind of thing a later refactor
    changes without noticing, and the failure it would introduce is invisible
    in every downstream number - a model scored on speakers it memorised does
    not look broken, it looks good.
    """
    findings: list[Finding] = []

    placements: dict[str, list[str]] = {}
    for contents in partitions:
        for pseudonym in contents.speakers:
            placements.setdefault(pseudonym, []).append(contents.partition.value)

    for pseudonym, where in sorted(placements.items()):
        if len(where) > 1:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "speaker_in_two_partitions",
                    f"speaker {pseudonym!r} is in {where}. A model that has heard a "
                    "speaker in training and is scored on that speaker in held-out is "
                    "being scored on memorisation",
                )
            )

    placed = set(placements)
    everyone = {s.pseudonym for s in corpus.speakers}
    for pseudonym in sorted(everyone - placed):
        findings.append(
            Finding(
                Severity.ERROR,
                "speaker_in_no_partition",
                f"speaker {pseudonym!r} was not assigned. Their recordings would be "
                "silently absent from every partition, and the corpus would be "
                "smaller than the dataset card says",
            )
        )
    return findings


def _coverage(partitions: Sequence[PartitionContents], corpus: CorpusInventory) -> list[Finding]:
    """Every P0 class present in every partition.

    An error rather than a warning, and specifically in held-out: a class with
    no instances there has an *undefined* per-class F1, and an undefined F1 is
    not a low score - it is a missing measurement that will be read as one.
    """
    findings: list[Finding] = []
    present = {e for e, count in corpus.class_counts.items() if count}

    for event in sorted(p0_speech_events(), key=lambda e: e.value):
        if event.value not in present:
            # Absent from the whole corpus - an inventory problem, not a split
            # problem, and `CorpusInventory.adequacy` is where it is reported.
            continue
        for contents in partitions:
            if contents.covers(event.value):
                continue
            findings.append(
                Finding(
                    Severity.ERROR,
                    "class_missing_from_partition",
                    f"'{event.value}' has {corpus.class_counts[event.value]} instances "
                    f"in the corpus and none in '{contents.partition.value}'. Its "
                    "per-class figure there is undefined, which is a missing "
                    "measurement rather than a low score - recruit speakers who "
                    "produce it, or accept that this class is not reported",
                )
            )
    return findings


def _balance(
    partitions: Sequence[PartitionContents], targets: Mapping[Partition, int]
) -> list[Finding]:
    """How far the speaker counts landed from their targets.

    A warning, never an error. Stratification trades balance for coverage on
    purpose, and a held-out set one speaker short of its share is a far smaller
    problem than one missing a class entirely.
    """
    findings: list[Finding] = []
    for contents in partitions:
        target = targets[contents.partition]
        if contents.speaker_count != target:
            findings.append(
                Finding(
                    Severity.WARNING,
                    "partition_off_target",
                    f"'{contents.partition.value}' holds {contents.speaker_count} "
                    f"speaker(s), target was {target}. Stratification trades balance "
                    "for class coverage; this is the trade, not a fault",
                )
            )
    return findings
