"""Prosodic indicators: measurable quantities, and nothing more.

FR-018 lists what the prosody analyzer produces - intensity, pitch stability,
speaking rate, pause duration - and §5 lists what it must not do: infer
emotions. The gap between those two lines is where most "multimodal emotion
recognition" systems live, and §17 names crossing it as an outright scientific
invalidity, not a feature to add later.

The defence here is structural. A prosody reading is a member of the published
``ProsodicIndicator`` enum carrying a number and a unit. There is no field an
interpretation could be written into, so producing one would require adding a
type - a visible change, reviewable as such.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from evidence_engine.domain.shared.confidence import Confidence
from evidence_engine.domain.shared.errors import DomainError
from evidence_engine.domain.shared.measurement import Indicator, Measured, Unavailable
from evidence_engine.domain.shared.provenance import Provenance
from evidence_engine.domain.shared.taxonomy import ProsodicIndicator
from evidence_engine.domain.shared.timeline import Interval


class ProsodyViolation(DomainError):
    """A prosodic reading was built with the wrong unit or an unknown name."""


#: The unit each indicator is reported in. Fixed here rather than supplied by
#: the producer, so that a runtime swap (NFR-016) cannot quietly change dB to a
#: normalized 0-1 scale while every stored history keeps comparing the two.
INDICATOR_UNITS: Mapping[ProsodicIndicator, str] = MappingProxyType(
    {
        ProsodicIndicator.MEAN_INTENSITY_DB: "dBFS",
        ProsodicIndicator.PITCH_MEAN_HZ: "Hz",
        # Coefficient of variation of f0. Unitless by construction, which is
        # what makes it comparable between a bass and a soprano voice; raw Hz
        # deviation is not, and comparing it across speakers is a common error.
        ProsodicIndicator.PITCH_STABILITY: "ratio",
        ProsodicIndicator.SPEAKING_RATE_WPM: "wpm",
        ProsodicIndicator.ARTICULATION_RATE_SPS: "syllables_per_second",
        ProsodicIndicator.MEAN_PAUSE_DURATION_MS: "ms",
        ProsodicIndicator.VOICED_RATIO: "ratio",
    }
)


@dataclass(frozen=True, slots=True)
class ProsodyReading:
    """One indicator over one window, measured or explicitly unavailable."""

    indicator: ProsodicIndicator
    window: Interval
    value: Indicator
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.value, Measured):
            expected = INDICATOR_UNITS[self.indicator]
            if self.value.unit != expected:
                raise ProsodyViolation(
                    f"'{self.indicator.value}' is reported in {expected}, not {self.value.unit!r}"
                )

    @property
    def is_available(self) -> bool:
        return self.value.is_available

    @classmethod
    def measured(
        cls,
        indicator: ProsodicIndicator,
        window: Interval,
        value: float,
        confidence: Confidence,
        provenance: Provenance,
    ) -> ProsodyReading:
        """Build a measured reading with the indicator's canonical unit."""
        return cls(
            indicator=indicator,
            window=window,
            value=Measured(value=value, confidence=confidence, unit=INDICATOR_UNITS[indicator]),
            provenance=provenance,
        )

    @classmethod
    def unavailable(
        cls,
        indicator: ProsodicIndicator,
        window: Interval,
        unavailable: Unavailable,
        provenance: Provenance,
    ) -> ProsodyReading:
        """Build a reading that says why it has no number (FR-025)."""
        return cls(indicator=indicator, window=window, value=unavailable, provenance=provenance)


__all__ = ["INDICATOR_UNITS", "ProsodyReading", "ProsodyViolation"]
