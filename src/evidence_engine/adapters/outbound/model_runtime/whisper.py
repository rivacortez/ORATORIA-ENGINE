"""A speech runtime backed by the pinned Whisper checkpoint.

The first runtime in this repository that hears real audio. Everything before
it replayed a script, which is what Phase 2's exit criterion is defined on and
is the right thing for a contract test - a scripted runtime is bit-exact by
construction, so a reproducibility test over it tests the harness.

Three things this deliberately does not do.

*It reports no disfluency events and no prosody.* Whisper is a recogniser. The
disfluency detector and the prosody estimator the NFRs describe do not exist,
and returning empty tuples is what keeps the engine's ``Unavailable`` handling
honest: a runtime that invented a filled pause would make FR-025's path look
exercised while it was being bypassed. When Phase 4 lands a detector, it plugs
in beside this, not inside it.

*It does not claim a pinned environment.* ``BASELINE_PINS.md`` fixes four
freeze stages and the third - backend, container digest, Torch/CUDA build - is
Phase 3 work that has not happened. So this runtime declares its environment
unpinned, and that declaration travels: ``environment_is_pinned`` is False, and
anything downstream that wants to publish a number can ask.

*It does not import torch at module scope.* The service must remain installable
and deployable without a 3 GB machine-learning stack; the imports happen inside
``load()`` so that ``container.py`` can reference this module in every mode and
fail with a sentence rather than an ImportError when the extra is absent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from evidence_engine.application.ports.runtimes import (
    AudioWindow,
    SpeechResult,
    TimedWordHypothesis,
    UntimedWordHypothesis,
    WordHypothesis,
)
from evidence_engine.domain.shared.identifiers import ModelVersionId
from evidence_engine.domain.shared.measurement import UnavailabilityReason, Unavailable

#: The artifact `BASELINE_PINS.md` pins, by identifier and revision. Hard-coded
#: rather than configurable: a runtime whose checkpoint is a deployment setting
#: is a runtime that cannot say which weights produced a result, and §14 asks
#: exactly that. Changing these is a change to the pins document first.
PINNED_MODEL = "openai/whisper-large-v3"
PINNED_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"

#: The decoding configuration the pins fix, restated here so that the code and
#: the document cannot drift apart silently. Each is argued in
#: `BASELINE_PINS.md`; the one that matters most is
#: `condition_on_prev_tokens=False`, which stops the decoder biasing toward
#: fluent continuations of what it already produced - the grammatical cleanup
#: FR-011 exists to measure the absence of.
PINNED_DECODING: dict[str, Any] = {
    "language": "es",
    "task": "transcribe",
    "num_beams": 5,
    "condition_on_prev_tokens": False,
}


class WhisperRuntimeUnavailable(RuntimeError):
    """The managed speech runtime was asked for and cannot be built."""


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    """What a deployment may change without changing which weights ran."""

    device: str = "cuda"
    #: fp16 halves the residency and is what the pins record the artifact as
    #: published in. A deployment without a GPU has to say so.
    dtype: str = "float16"
    #: Where the artifact is cached. None uses the library default.
    cache_dir: str | None = None


class WhisperSpeechRuntime:
    """``SpeechRuntime`` over the pinned checkpoint.

    Stateless between calls, because the port requires idempotency: §6.3
    retries an idempotent window after a model timeout, and a runtime that
    accumulated state across calls would double-count the retried audio.
    `condition_on_prev_tokens=False` is part of what makes that true here as
    well as part of the pinned configuration.
    """

    #: False, and it says so rather than being silent about it. The executable
    #: environment - backend, container digest, Torch and CUDA build - freezes
    #: in Phase 3, and until then any figure this runtime produces is
    #: reproducible only by whoever ran it.
    environment_is_pinned = False

    def __init__(self, pipeline: Any, model_version: ModelVersionId) -> None:
        self._pipeline = pipeline
        self.model_version = model_version

    @classmethod
    def load(cls, settings: WhisperSettings | None = None) -> WhisperSpeechRuntime:
        """Build the runtime, or explain in one sentence why it cannot be built.

        The import failure is caught and re-raised with the install command,
        because "No module named 'torch'" from three frames inside a container
        factory is a message that costs somebody an afternoon.
        """
        resolved = settings or WhisperSettings()
        try:
            import torch
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
                pipeline,
            )
        except ImportError as error:  # pragma: no cover - exercised by hand
            raise WhisperRuntimeUnavailable(
                "the managed speech runtime needs torch and transformers, which are "
                "not installed. They are an optional extra rather than a dependency: "
                "the service deploys without a 3 GB machine-learning stack. Install "
                "with `uv sync --extra managed`, and note that the CUDA wheel index "
                "is chosen for your card - see docs/governance/BASELINE_PINS.md."
            ) from error

        if resolved.device.startswith("cuda") and not torch.cuda.is_available():
            raise WhisperRuntimeUnavailable(
                f"device {resolved.device!r} was requested and torch reports no CUDA "
                "device. Set ENGINE_WHISPER_DEVICE=cpu to run anyway - it will be "
                "roughly thirty times slower and is not a configuration any figure "
                "should come from."
            )

        processor = WhisperProcessor.from_pretrained(
            PINNED_MODEL, revision=PINNED_REVISION, cache_dir=resolved.cache_dir
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            PINNED_MODEL,
            revision=PINNED_REVISION,
            dtype=getattr(torch, resolved.dtype),
            cache_dir=resolved.cache_dir,
        ).to(resolved.device)
        model.eval()

        # Three `type: ignore`s, each for a real gap in transformers' own
        # annotations rather than a doubt about the call.
        #
        # `pipeline` is annotated as a set of Literal overloads and this task
        # string is resolved at runtime, so mypy picks the wrong one.
        # `processor.tokenizer` and `.feature_extractor` are attached by
        # `ProcessorMixin.__init__` from `attributes`, which no stub declares.
        #
        # Worth recording: these appeared only once the `managed` extra was
        # installed. Before that, mypy could not find transformers at all and
        # the `ignore_missing_imports` override silenced everything - so the
        # gate was green on a machine that could not run this code and amber on
        # one that could.
        built = pipeline(  # type: ignore[call-overload]
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,  # type: ignore[attr-defined]
            feature_extractor=processor.feature_extractor,  # type: ignore[attr-defined]
            dtype=getattr(torch, resolved.dtype),
            device=resolved.device,
        )
        # The revision, not the name. Two people running "whisper-large-v3" a
        # year apart are not running the same weights, and the provenance has
        # to be able to tell them apart.
        return cls(built, ModelVersionId(f"whisper-large-v3@{PINNED_REVISION[:12]}"))

    async def transcribe(self, window: AudioWindow) -> SpeechResult:
        """Words and their boundaries. Nothing invented.

        Word-level offsets come from the checkpoint's own alignment heads, via
        the pipeline's ``chunks``. That route is not the obvious one:
        ``generate(return_timestamps="word", return_dict_in_generate=True)``
        returns segment boundaries and **no per-word offsets, with no error**,
        which is how a project ends up believing it has word timestamps and
        shipping segment ones. NFR-004's 250 ms target is unreachable from
        segment boundaries, so the route matters.
        """
        samples = self._to_float32(window.samples)

        # Off the event loop: this is 200 ms to several seconds of GPU work and
        # the coordinator has a socket to keep answering.
        output = await asyncio.to_thread(
            self._pipeline,
            {"raw": samples, "sampling_rate": window.sample_rate_hz},
            return_timestamps="word",
            generate_kwargs=dict(PINNED_DECODING),
        )

        return SpeechResult(
            model_version=self.model_version,
            window_position_ms=window.session_position_ms,
            words=self.words_from(output, window.session_position_ms),
            events=(),
            prosody=(),
            # The whole window. Whisper decodes a window in one pass and has no
            # revisable tail, unlike a streaming transducer - so everything it
            # returns is as stable as it will ever be.
            stable_through_ms=window.session_position_ms + window.duration_ms,
        )

    @staticmethod
    def _to_float32(samples: bytes) -> Any:
        import numpy as np

        return np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def words_from(output: Any, session_position_ms: int) -> tuple[WordHypothesis, ...]:
        """Every word the pipeline reported, placed on the session clock or not.

        A static method taking the raw output rather than a private helper,
        because this is where both of the adapter's judgement calls live and
        neither was testable while they were buried behind a GPU and a 3 GB
        download.

        **A word whose alignment failed is kept.** The pipeline returns ``None``
        for a bound when the alignment heads cannot place a fragment. Refusing
        to invent the boundary is right; deleting the word is not - what is
        unknown is where it was, not whether it was said. Dropping it produced
        a transcript one word shorter that nothing downstream could detect.

        **A word with one bound is untimed too.** Half an interval is not an
        interval, and keeping the half that arrived means inventing the other.

        The offsets are relative to the window; sessions run for minutes. A
        window at 60 s carrying a word at 0.4 s is a word at 60.4 s of the
        session, and getting that wrong shifts every interval and every
        co-occurrence by the window position while the text still reads
        correctly.
        """
        words: list[WordHypothesis] = []
        # `index` is the emission order, assigned over the chunks that
        # become words. Counting dropped blanks instead would make the index
        # depend on how many empty chunks the decoder happened to emit, and
        # every token id would churn between passes.
        for chunk in output.get("chunks", ()):
            text = str(chunk.get("text", "")).strip()
            if not text:
                # Not a word missing its timing - not a word. `WordToken`
                # refuses empty text on the grounds that an unintelligible
                # stretch is an UNINTELLIGIBLE event rather than a blank word,
                # so admitting one here would only move the failure later.
                continue

            start, end = chunk.get("timestamp", (None, None))
            if start is None or end is None:
                words.append(
                    UntimedWordHypothesis(
                        raw_text=text,
                        score=NO_REPORTED_POSTERIOR,
                        index=len(words),
                        detail="the checkpoint's alignment heads returned no interval",
                    )
                )
                continue

            words.append(
                TimedWordHypothesis(
                    raw_text=text,
                    start_ms=session_position_ms + int(float(start) * 1000),
                    end_ms=session_position_ms + int(float(end) * 1000),
                    score=NO_REPORTED_POSTERIOR,
                    index=len(words),
                )
            )
        return tuple(words)


#: Whisper exposes no per-word posterior, and this says so rather than standing
#: in for one.
#:
#: It used to be the float `0.5`, defended in a comment as "the neutral value"
#: - not 1.0, which would assert unearned certainty, and not 0.0, which reads
#: as a rejection. The reasoning was sound and the conclusion was still wrong:
#: a consumer reading a number cannot tell a neutral placeholder from a genuine
#: 50% posterior, and FR-025 exists precisely so the two never look alike. The
#: honest answer was not a better number. It was that there is no number.
NO_REPORTED_POSTERIOR = Unavailable(
    reason=UnavailabilityReason.POSTERIOR_NOT_REPORTED,
    detail="whisper-large-v3 emits no per-word posterior",
)
