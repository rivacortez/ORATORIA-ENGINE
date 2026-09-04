"""Telemetry adapters: counters, histograms and structured events.

NFR-018 requires distributed traces, structured logs and metrics correlated by
``trace_id``, "without storing transcript content in operational logs". The port
already makes that hard by accepting only scalars; this adapter adds the second
line of defence, redacting any field whose name suggests it carries evidence.

The redaction is intentionally crude. It will occasionally blank something
harmless, which costs a debugging session; the alternative failure mode is a
student's transcript sitting in a log aggregator, which costs considerably more.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import structlog

#: Field names that must never be written to an operational log. Substring
#: matching, so `raw_text`, `transcript_excerpt` and `token_text` are all
#: caught without anybody having to enumerate them.
_REDACTED_SUBSTRINGS: frozenset[str] = frozenset(
    {"text", "transcript", "token", "utterance", "word", "secret", "key", "audio", "frame"}
)

_REDACTED = "[redacted]"


def _redact(field: str, value: object) -> object:
    lowered = field.lower()
    if any(marker in lowered for marker in _REDACTED_SUBSTRINGS):
        return _REDACTED
    return value


class StructlogTelemetry:
    """Emits to structlog and keeps in-process counters for scraping."""

    def __init__(self, service: str = "evidence-engine") -> None:
        self._log = structlog.get_logger(service=service)
        self._counters: Counter[str] = Counter()
        self._histograms: dict[str, list[float]] = {}

    def counter(self, name: str, value: int = 1, **labels: str) -> None:
        self._counters[_key(name, labels)] += value

    def histogram(self, name: str, value: float, **labels: str) -> None:
        self._histograms.setdefault(_key(name, labels), []).append(value)

    def event(self, name: str, trace_id: str, **fields: str | int | float | bool) -> None:
        safe = {field: _redact(field, value) for field, value in fields.items()}
        self._log.info(name, trace_id=trace_id, **safe)

    # -- scraping ---------------------------------------------------------

    def counters(self) -> Mapping[str, int]:
        return dict(self._counters)

    def histograms(self) -> Mapping[str, tuple[float, ...]]:
        return {key: tuple(values) for key, values in self._histograms.items()}


class NullTelemetry:
    """Records nothing. For tests that are not asserting on telemetry."""

    def counter(self, name: str, value: int = 1, **labels: str) -> None:
        return None

    def histogram(self, name: str, value: float, **labels: str) -> None:
        return None

    def event(self, name: str, trace_id: str, **fields: str | int | float | bool) -> None:
        return None


def _key(name: str, labels: Mapping[str, str]) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{rendered}}}"
