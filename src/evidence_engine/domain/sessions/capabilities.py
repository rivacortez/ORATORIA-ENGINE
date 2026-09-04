"""Capability negotiation: reject what cannot be processed, before capture.

FR-006 requires the service to negotiate codecs, sample rate, frame rate and
locale, and US-011 states the acceptance criterion that makes it worth doing:
"Unsupported formats are rejected before capture." A student who records a
five-minute presentation and only then learns the sample rate was wrong has
lost the presentation, and no amount of graceful degradation gets it back.

Negotiation is therefore a hard gate, not a preference. When the request cannot
be satisfied the session is not created at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evidence_engine.domain.shared.errors import DomainError


class UnsupportedCapability(DomainError):
    """The client asked for a configuration the engine cannot process."""


class AudioCodec(StrEnum):
    """Codecs the audio preprocessor can decode.

    PCM16 is listed first and is the only one guaranteed lossless. The lossy
    entries are accepted because browsers produce them, but see
    ``requires_quality_warning``: driver 1 is verbatim fidelity, and a codec
    that discards the high-frequency detail of a cut-off makes that detail
    unrecoverable no matter how good the model is.
    """

    PCM16 = "pcm16"
    FLAC = "flac"
    OPUS = "opus"
    AAC = "aac"


class VideoFormat(StrEnum):
    """Accepted shapes for the visual channel.

    ``LANDMARKS`` is the privacy-preserving option of §7.2 and ADR-009: the
    client extracts features locally and sends only geometry, so no image ever
    reaches the service. It is a first-class format rather than a fallback
    because it is the only one that keeps raw likeness out of the system
    entirely.
    """

    JPEG_FRAMES = "jpeg_frames"
    PNG_FRAMES = "png_frames"
    LANDMARKS = "landmarks"


#: Sample rates the ASR front end supports. 16 kHz is the working rate; the
#: higher ones are resampled down. Anything below 16 kHz is refused rather than
#: upsampled: upsampling cannot restore the fricative and burst detail that
#: cut-offs and prolongations are detected from, so accepting it would mean
#: publishing per-class metrics on evidence that was never there.
SUPPORTED_SAMPLE_RATES_HZ: frozenset[int] = frozenset({16_000, 22_050, 24_000, 44_100, 48_000})

#: Frame rates the visual sampler can work with. Below 5 fps the periodicity of
#: repetitive movement (FR-021) cannot be estimated, so it is refused.
MIN_FRAME_RATE_FPS = 5
MAX_FRAME_RATE_FPS = 60

#: Locales with a validated evaluation corpus. §11.1 constrains version 1 to
#: Peruvian Spanish; other locales are refused rather than silently served by a
#: model whose error rates on them are unknown (rule: no undocumented claims).
SUPPORTED_LOCALES: frozenset[str] = frozenset({"es-PE"})

#: Codecs whose compression can erase the acoustic detail some P0 classes are
#: decided from. Accepted, but the session carries a standing quality warning.
_LOSSY_CODECS: frozenset[AudioCodec] = frozenset({AudioCodec.OPUS, AudioCodec.AAC})


@dataclass(frozen=True, slots=True)
class CapabilityRequest:
    """What the consuming application says it can send."""

    audio_codec: str
    sample_rate_hz: int
    locale: str
    video_format: str | None = None
    frame_rate_fps: int | None = None


@dataclass(frozen=True, slots=True)
class NegotiatedCapabilities:
    """What the engine agreed to accept for this session.

    Stored on the session and echoed in the result so that a reader months
    later can tell whether a weak indicator reflects the speaker or the codec.
    """

    audio_codec: AudioCodec
    sample_rate_hz: int
    locale: str
    video_format: VideoFormat | None
    frame_rate_fps: int | None

    @property
    def has_video(self) -> bool:
        return self.video_format is not None

    @property
    def video_is_privacy_preserving(self) -> bool:
        """True when only derived geometry crosses the network (ADR-009)."""
        return self.video_format is VideoFormat.LANDMARKS

    @property
    def requires_quality_warning(self) -> bool:
        """Whether the agreed audio path can erase P0 acoustic detail."""
        return self.audio_codec in _LOSSY_CODECS


def negotiate(request: CapabilityRequest) -> NegotiatedCapabilities:
    """Agree a configuration, or refuse the session outright (FR-006)."""
    codec = _resolve_codec(request.audio_codec)

    if request.sample_rate_hz not in SUPPORTED_SAMPLE_RATES_HZ:
        raise UnsupportedCapability(
            f"sample rate {request.sample_rate_hz} Hz is not supported; "
            f"accepted: {sorted(SUPPORTED_SAMPLE_RATES_HZ)}"
        )

    if request.locale not in SUPPORTED_LOCALES:
        raise UnsupportedCapability(
            f"locale '{request.locale}' has no validated corpus; "
            f"accepted: {sorted(SUPPORTED_LOCALES)}"
        )

    video_format, frame_rate = _resolve_video(request)

    return NegotiatedCapabilities(
        audio_codec=codec,
        sample_rate_hz=request.sample_rate_hz,
        locale=request.locale,
        video_format=video_format,
        frame_rate_fps=frame_rate,
    )


def _resolve_codec(requested: str) -> AudioCodec:
    try:
        return AudioCodec(requested)
    except ValueError as exc:
        raise UnsupportedCapability(
            f"audio codec '{requested}' is not supported; "
            f"accepted: {sorted(c.value for c in AudioCodec)}"
        ) from exc


def _resolve_video(request: CapabilityRequest) -> tuple[VideoFormat | None, int | None]:
    """Resolve the visual channel, which is optional by design.

    An audio-only session is a valid session, not a degraded one: QA-02 already
    requires the engine to keep producing speech evidence without video, so
    starting that way needs no special handling. The visual indicators simply
    come back unavailable with ``MODALITY_NOT_CAPTURED``.
    """
    if request.video_format is None:
        if request.frame_rate_fps is not None:
            raise UnsupportedCapability("a frame rate was requested without a video format")
        return None, None

    try:
        video_format = VideoFormat(request.video_format)
    except ValueError as exc:
        raise UnsupportedCapability(
            f"video format '{request.video_format}' is not supported; "
            f"accepted: {sorted(f.value for f in VideoFormat)}"
        ) from exc

    frame_rate = request.frame_rate_fps
    if frame_rate is None:
        raise UnsupportedCapability(
            f"video format '{video_format.value}' requires a frame rate: "
            "the periodicity indicators of FR-021 cannot be estimated without one"
        )
    if not MIN_FRAME_RATE_FPS <= frame_rate <= MAX_FRAME_RATE_FPS:
        raise UnsupportedCapability(
            f"frame rate {frame_rate} fps is outside the supported range "
            f"[{MIN_FRAME_RATE_FPS}, {MAX_FRAME_RATE_FPS}]"
        )
    return video_format, frame_rate
