"""The local development key, and the command that mints it.

Two things are being defended here, and they pull in opposite directions.

The affordance has to *work*: without it the published API is readable and
uncallable, because the memory backend keeps keys in a dict owned by one
process and nothing in the product can provision one into a running server.
So the first test drives the whole loop the operator drives - mint, set the
variable, build, call - and refuses to accept a mock of any part of it.

And the affordance has to be *contained*. A credential arriving through an
environment variable is what §15.2 exists to prevent, so every guard that keeps
it out of a real deployment is asserted by an instance that tries to start and
is refused. A guard nobody tested is a guard that will be deleted by the next
person who finds it inconvenient.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from evidence_engine.adapters.outbound.persistence.identity import (
    InMemoryApiKeyDirectory,
    hash_secret,
)
from evidence_engine.adapters.outbound.telemetry.clock import FrozenClock
from evidence_engine.bootstrap import cli
from evidence_engine.bootstrap.app import create_app
from evidence_engine.bootstrap.container import (
    BOOTSTRAP_APPLICATION,
    BOOTSTRAP_TENANT,
    build_container,
)
from evidence_engine.bootstrap.settings import Backend, RuntimeMode, Settings
from evidence_engine.domain.sessions.capabilities import AudioCodec

from .conftest import TEST_PEPPER, TEST_SIGNING_KEY

A_MINTED_KEY = "oek_kWq3Zt7nR2xLpV8sYd4hJc6mB1gF0aUe9TzXo5NrQi"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "local",
        "backend": Backend.MEMORY,
        "runtime_mode": RuntimeMode.DETERMINISTIC,
        "api_key_pepper": TEST_PEPPER,
        "stream_token_signing_key": TEST_SIGNING_KEY,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_the_minted_key_opens_the_documented_api(capsys: pytest.CaptureFixture[str]) -> None:
    """The whole loop, with nothing stubbed.

    `issue-key` prints a secret; that exact string goes into the setting; the
    container registers it; a request carrying it reaches an endpoint that
    requires a scope. If any link is wrong the operator gets a 401 from a
    docs page and no way to tell which link broke, which is the failure this
    test exists to make loud.
    """
    assert cli.main(["issue-key", "--json"]) == 0
    minted = json.loads(capsys.readouterr().out)["secret"]

    container = build_container(_settings(bootstrap_api_key=minted), clock=FrozenClock(1_000_000))
    with TestClient(create_app(container)) as client:
        response = client.post(
            "/v1/sessions",
            headers={"Authorization": f"Bearer {minted}"},
            json={"language": "es-PE"},
        )

    assert response.status_code != 401, response.text
    assert response.status_code < 500, response.text


def test_no_key_is_registered_when_the_variable_is_absent() -> None:
    """The default is a directory that authenticates nothing.

    The affordance is opt-in. An instance that came up with a key because a
    default existed would be the exact accident the whole guard is written
    against.
    """
    container = build_container(_settings(), clock=FrozenClock(1_000_000))
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)

    with TestClient(create_app(container)) as client:
        response = client.post(
            "/v1/sessions",
            headers={"Authorization": f"Bearer {A_MINTED_KEY}"},
            json={"language": "es-PE"},
        )
    assert response.status_code == 401


def test_only_the_hash_is_stored(capsys: pytest.CaptureFixture[str]) -> None:
    """FR-002 holds for this key too, not just for issued ones.

    Registered rather than issued is a different code path, and it would be an
    easy place to keep the plaintext around "just for the bootstrap case".
    """
    assert cli.main(["issue-key", "--json"]) == 0
    minted = json.loads(capsys.readouterr().out)["secret"]

    container = build_container(_settings(bootstrap_api_key=minted), clock=FrozenClock(1_000_000))
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)

    stored = list(directory._by_hash)
    assert stored == [hash_secret(minted, TEST_PEPPER)]
    assert minted not in stored


def test_the_key_is_attributed_to_a_tenant_no_study_would_use() -> None:
    """Evidence from a local trial has to be identifiable as such.

    §14.4 pseudonymises research exports, and a local demo that wrote rows
    under a plausible tenant name would put unreviewed evidence one query away
    from looking like participant data.
    """
    container = build_container(
        _settings(bootstrap_api_key=A_MINTED_KEY), clock=FrozenClock(1_000_000)
    )
    directory = container.api_keys
    assert isinstance(directory, InMemoryApiKeyDirectory)

    record = next(iter(directory._by_hash.values()))
    assert record.tenant.value == BOOTSTRAP_TENANT == "local"
    assert record.application.value == BOOTSTRAP_APPLICATION == "local-development"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param(
            {"environment": "production"},
            "local development affordance",
            id="refused-in-production",
        ),
        pytest.param(
            {"environment": "staging"},
            "local development affordance",
            id="refused-in-staging",
        ),
    ],
)
def test_the_key_is_refused_outside_a_local_environment(
    overrides: dict[str, object], expected: str
) -> None:
    """A deployment that started with this set would be the incident.

    Both non-local names are asserted rather than one, because a guard written
    as `!= "production"` passes a production test and leaves staging - which
    holds real data often enough - wide open.
    """
    settings = _settings(bootstrap_api_key=A_MINTED_KEY, **overrides)
    with pytest.raises(ValueError, match=expected):
        settings.reject_a_bootstrap_key_outside_local()


def test_the_key_is_refused_with_a_persistent_backend() -> None:
    """Silently ignoring it is the dangerous behaviour, not refusing it.

    The bootstrap key registers into the in-memory directory. Against Postgres
    it would do nothing at all, and the operator would read their own start-up
    banner, paste the key, get a 401, and go looking for a bug in
    authentication.
    """
    settings = _settings(
        bootstrap_api_key=A_MINTED_KEY,
        backend=Backend.POSTGRES,
        database_url="postgresql+asyncpg://x/y",
        redis_url="redis://x",
        object_storage_endpoint="http://x",
        object_storage_access_key="x",
        object_storage_secret_key="y",
    )
    with pytest.raises(ValueError, match="persistent backend"):
        settings.reject_a_bootstrap_key_outside_local()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("dev", id="short-and-guessable"),
        pytest.param("oek_dev", id="right-prefix-no-entropy"),
        pytest.param("a-perfectly-long-string-with-no-key-prefix", id="long-but-not-a-key"),
    ],
)
def test_a_guessable_bootstrap_key_is_refused_at_parse_time(value: str) -> None:
    """Refused where it is cheapest to refuse: before the process starts."""
    with pytest.raises(ValueError, match="look like an issued key"):
        _settings(bootstrap_api_key=value)


def test_issue_key_tells_the_operator_what_to_do_with_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command's output is the interface, so its content is a contract.

    A minted secret with no instruction is a secret that gets pasted into
    Swagger against a server that never heard of it. The variable name has to
    appear, because that is the only coupling between the two processes.
    """
    assert cli.main(["issue-key"]) == 0
    printed = capsys.readouterr().out

    assert "oek_" in printed
    assert "ENGINE_BOOTSTRAP_API_KEY" in printed
    assert "not recoverable" in printed


def test_show_config_reports_that_nothing_is_calibrated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The snapshot decides what the numbers mean, so it says what it lacks.

    An operator reading a threshold table without being told no calibration
    has been fitted would reasonably conclude the thresholds are in force.
    """
    assert cli.main(["show-config"]) == 0
    printed = capsys.readouterr().out

    assert "raw confidence never clears a threshold" in printed


def test_no_subcommand_prints_help_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    """Exit 2, not 0. A bare invocation did not do what was asked."""
    assert cli.main([]) == 2
    assert "issue-key" in capsys.readouterr().out


def test_the_published_example_is_accepted_by_the_endpoint_it_documents(
    client: TestClient, auth: dict[str, str]
) -> None:
    """The docs page's own example, sent verbatim, must work.

    This is the first thing anybody does with a Swagger page: press "Try it
    out" and then "Execute". Before the example existed, that produced a 400
    `unsupported_capability`, because the placeholder Swagger generates for an
    unconstrained string is the literal `"string"`.

    An example is documentation that can rot silently - a new required field,
    a codec withdrawn from `AudioCodec` - and nothing else in the suite would
    notice. So the example is not read from a fixture: it is the same object
    the schema publishes, driven through the real endpoint.
    """
    published = client.get("/v1/openapi.json").json()
    example = published["components"]["schemas"]["CreateSessionBody"]["examples"][0]

    response = client.post("/v1/sessions", headers=auth, json=example)

    assert response.status_code == 201, response.text
    assert response.json()["state"] == "created"


def test_the_documented_codecs_are_the_ones_the_domain_accepts(client: TestClient) -> None:
    """The codec list in the docs is generated, and this proves it stayed so.

    Someone replacing the generated string with a typed-out list would produce
    identical output today and a lie the first time a codec changes.
    """
    published = client.get("/v1/openapi.json").json()
    documented = published["components"]["schemas"]["CapabilityRequestBody"]["properties"][
        "audio_codec"
    ]["description"]

    for codec in AudioCodec:
        assert f"`{codec.value}`" in documented
