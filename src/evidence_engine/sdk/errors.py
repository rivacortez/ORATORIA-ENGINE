"""The errors an SDK consumer is expected to catch.

A short, closed list, and closed on purpose. Everything a consumer branches on
is part of the published surface, so an exception type is as expensive to
change as a method signature - which is the argument for having few of them
and for keeping the engine's internal failures out of this list. A consumer
catching `TranscriptViolation` would be catching a statement about the domain's
own invariants, and coupling their retry logic to it.
"""

from __future__ import annotations


class OratoriaError(Exception):
    """Base for everything this SDK raises deliberately.

    A consumer that wants one `except` for the engine catches this. One that
    sees something else has found a bug here rather than a condition to handle.
    """


class LocalInferenceUnavailable(OratoriaError):
    """The model could not be loaded or could not complete a decode.

    Raised by ``warmup()``, which is the only thing that can say this. A
    passing ``hardware_preflight()`` does not rule it out: the weights can be
    corrupt, the driver too old for the compiled kernel, or the card already
    holding something else.
    """


class AudioNotUsable(OratoriaError):
    """The recording is not in a shape the engine can time correctly.

    Raised rather than resampled. The declared rate is what converts a byte
    count into a duration, so audio decoded at one rate and timed at another
    scales every boundary by the ratio between them - and the transcript still
    reads correctly, which is what makes it dangerous.
    """


class StreamAlreadyClosed(OratoriaError):
    """A finished or aborted session was used again."""
