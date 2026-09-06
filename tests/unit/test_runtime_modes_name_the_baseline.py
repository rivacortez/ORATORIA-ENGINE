"""Whisper is a research baseline and the setting has to say so.

`RuntimeMode.MANAGED` loaded the pinned Whisper checkpoint, and ADR-003 had
written down that exact failure in advance:

    Keep Whisper and CrisperWhisper as **frozen research baselines**, never as
    an undocumented production dependency - §11.1 names that constraint
    explicitly, and a baseline adapter that stays wired "temporarily" is
    exactly how it gets violated.

Nothing in the word "managed" says "baseline". A deployment reading
``ENGINE_RUNTIME_MODE=managed`` would reasonably conclude it was running the
project's model; it was running the comparator the project exists to beat, and
any figure produced from it would have been attributed to the wrong thing.

The four modes separate the three states that were being conflated: scripted,
baseline, and the model ADR-003 targets - which does not exist, and now says so
rather than silently resolving to the baseline.
"""

from __future__ import annotations

import pytest

from evidence_engine.bootstrap.container import ProjectModelNotBuilt, build_container
from evidence_engine.bootstrap.settings import Backend, RuntimeMode, Settings

TEST_PEPPER = "test-pepper-value-at-least-32-chars-long"
TEST_SIGNING_KEY = "test-signing-key-at-least-32-characters"


def settings_for(mode: RuntimeMode) -> Settings:
    return Settings(
        environment="test",
        backend=Backend.MEMORY,
        runtime_mode=mode,
        api_key_pepper=TEST_PEPPER,
        stream_token_signing_key=TEST_SIGNING_KEY,
    )


def test_the_baseline_is_named_a_baseline() -> None:
    """The rename, as an assertion.

    `managed` is gone. A value that does not say what it loads is how a
    baseline stays wired past the point anybody remembers it is one.
    """
    assert RuntimeMode.BASELINE_WHISPER.value == "baseline_whisper"
    assert "managed" not in {mode.value for mode in RuntimeMode}


def test_the_project_model_modes_exist_and_are_distinct() -> None:
    """Named before they are built, so asking for one is answerable.

    A two-mode enum forced every deployment into `deterministic` or the
    baseline, which meant "I want the project model" had no way to be said and
    therefore no way to be refused.
    """
    assert RuntimeMode.PROJECT_MODEL_LOCAL.value == "project_model_local"
    assert RuntimeMode.PROJECT_MODEL_REMOTE.value == "project_model_remote"
    assert len(set(RuntimeMode)) == 4


@pytest.mark.parametrize(
    "mode",
    [RuntimeMode.PROJECT_MODEL_LOCAL, RuntimeMode.PROJECT_MODEL_REMOTE],
    ids=lambda m: m.value,
)
def test_asking_for_the_project_model_is_refused_not_substituted(mode: RuntimeMode) -> None:
    """Silently receiving the baseline is the dangerous outcome, not the crash.

    A deployment that asked for the project model and got Whisper would publish
    figures attributed to a model that does not exist, and ADR-010's promotion
    trail would record a promotion that never happened. The refusal names both
    what is missing and what to use instead.
    """
    with pytest.raises(ProjectModelNotBuilt) as failure:
        build_container(settings_for(mode))

    message = str(failure.value)
    assert "does not exist" in message
    assert RuntimeMode.BASELINE_WHISPER.value in message
    assert RuntimeMode.DETERMINISTIC.value in message


def test_the_refusal_says_the_baseline_is_a_comparator() -> None:
    """The message has to carry the distinction, not just the alternative.

    Pointing at `baseline_whisper` without saying what it is would send an
    operator straight into the mistake this rename exists to prevent.
    """
    with pytest.raises(ProjectModelNotBuilt, match="comparator"):
        build_container(settings_for(RuntimeMode.PROJECT_MODEL_LOCAL))


def test_the_deterministic_mode_still_builds() -> None:
    """The rename must not have disturbed the mode the whole suite runs on."""
    container = build_container(settings_for(RuntimeMode.DETERMINISTIC))
    assert container.profile.runtime_mode == "deterministic"
