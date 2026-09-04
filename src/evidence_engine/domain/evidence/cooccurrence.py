"""Temporal co-occurrence: two things happened near each other. That is all.

FR-027 asks for deterministic co-occurrences over a versioned window, and
FR-028 adds the sentence that governs the whole module: co-occurrences shall
always declare ``causal_inference=false``.

The obvious implementation - a boolean field defaulting to ``False`` - fails the
requirement in the only situation that matters. A field can be set. Some future
call site, in good faith, will pass ``True`` for a case that "obviously" is
causal, a reviewer will not notice among fifty other changes, and the engine
will have started making causal claims. So ``causal_inference`` is a class
constant with no constructor parameter behind it. Asserting causality would
require editing this file, which is a reviewable act.

The distinction is not pedantry. "A filled pause occurred while the gaze was
off camera" is an observation two independent detectors made. "Looking away
caused the hesitation" is a claim about cognition that no instrument here
measures, and §5 forbids the fusion engine from asserting it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from evidence_engine.domain.shared.errors import DomainError
from evidence_engine.domain.shared.identifiers import ConfigurationSnapshotId, EventId
from evidence_engine.domain.shared.timeline import Interval

#: Default half-width of the fusion window, in milliseconds. Two events count
#: as co-occurring when the gap between their intervals does not exceed it.
#: Versioned and configurable per FR-027 and FR-015 - this is the value a fresh
#: configuration snapshot starts from, not a law.
DEFAULT_FUSION_WINDOW_MS = 500


class FusionViolation(DomainError):
    """A co-occurrence was built outside the rules the fusion engine declares."""


@dataclass(frozen=True, slots=True)
class FusionWindow:
    """The versioned window a set of co-occurrences was computed under.

    Carried on every co-occurrence rather than held globally, because QA-05
    requires a frozen evaluation item to reproduce, and NFR-015 pins that to the
    recorded configuration. A window read from live config at query time would
    make yesterday's results recompute differently today.
    """

    width_ms: int
    configuration: ConfigurationSnapshotId

    def __post_init__(self) -> None:
        if self.width_ms < 0:
            raise FusionViolation(f"fusion window cannot be negative: {self.width_ms}")

    def admits(self, gap_ms: int) -> bool:
        return gap_ms <= self.width_ms


@dataclass(frozen=True, slots=True)
class MultimodalCooccurrence:
    """§8 ``MultimodalCooccurrence``: two events, and how far apart they were.

    Symmetric by construction. There is no "cause" and "effect" slot, and no
    ordering that implies one; the pair is identified by which modality each
    side came from, which is a fact about provenance rather than about
    influence.
    """

    #: FR-028, structurally. A `ClassVar` rather than a field: the dataclass
    #: machinery keeps it out of `__init__`, so no constructor argument reaches
    #: it, no deserializer can be handed a different value, and no test can flip
    #: it to check "the other branch" - there is no other branch.
    causal_inference: ClassVar[bool] = False

    speech_event_id: EventId
    visual_event_id: EventId
    temporal_distance_ms: int
    window: FusionWindow

    def __post_init__(self) -> None:
        if self.temporal_distance_ms < 0:
            raise FusionViolation(
                f"temporal distance is an absolute gap and cannot be negative: "
                f"{self.temporal_distance_ms}"
            )
        if not self.window.admits(self.temporal_distance_ms):
            raise FusionViolation(
                f"gap of {self.temporal_distance_ms} ms exceeds the fusion window of "
                f"{self.window.width_ms} ms; this pair does not co-occur under that "
                "configuration"
            )


@dataclass(frozen=True, slots=True)
class TimedEvent:
    """The minimum the fusion engine needs: an identity and an interval.

    Deliberately not ``SpeechEvent | VisualEvent``. Contract C5 keeps the two
    observation packages independent, and a fusion function that imported both
    would be the seam through which they learn about each other. It also keeps
    fusion honest: correlating on time alone means the algorithm cannot be
    tempted to weight pairs by what the events "mean".
    """

    id: EventId
    interval: Interval


def correlate(
    speech: Sequence[TimedEvent],
    visual: Sequence[TimedEvent],
    window: FusionWindow,
) -> tuple[MultimodalCooccurrence, ...]:
    """Pair speech and visual events that fall within the window.

    Deterministic, as NFR-015 and §11.3 iteration 5 require: the same inputs and
    the same window always produce the same pairs in the same order. Ordering is
    by the speech interval, then the visual interval, then the identifiers, so
    that ties break the same way on every machine and in every process.

    Every admissible pair is emitted, including one event pairing with several
    on the other side. Selecting a "best" match would be a ranking decision, and
    §5 reserves ranking for the consumer's own engine - the fusion engine
    identifies co-occurrences and stops there.
    """
    pairs: list[MultimodalCooccurrence] = []
    for speech_event in _ordered(speech):
        for visual_event in _ordered(visual):
            gap = speech_event.interval.gap_to(visual_event.interval)
            if not window.admits(gap):
                continue
            pairs.append(
                MultimodalCooccurrence(
                    speech_event_id=speech_event.id,
                    visual_event_id=visual_event.id,
                    temporal_distance_ms=gap,
                    window=window,
                )
            )
    return tuple(pairs)


def _ordered(events: Iterable[TimedEvent]) -> list[TimedEvent]:
    return sorted(events, key=lambda e: (e.interval.start.ms, e.interval.end.ms, e.id.value))
