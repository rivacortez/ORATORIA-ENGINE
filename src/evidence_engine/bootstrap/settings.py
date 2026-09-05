"""Runtime settings.

Every secret is a required field with no default. §15.2 puts secrets in a
secret manager and forbids them in environment files committed to Git, and a
default value is how a development placeholder reaches production: the process
starts, nothing complains, and the signing key is whatever somebody typed while
getting the tests to pass.

So the process refuses to start without them. A crash at boot is the cheapest
possible discovery of a missing secret.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Backend(StrEnum):
    """Which set of outbound adapters to wire.

    ``MEMORY`` is not a toy. Phase 2's exit criterion is defined without model
    inference or infrastructure, and the contract suite runs against this
    backend on every commit rather than only where Postgres, Redis and MinIO
    happen to be up.
    """

    MEMORY = "memory"
    POSTGRES = "postgres"


class RuntimeMode(StrEnum):
    """Which model runtimes to wire.

    ``DETERMINISTIC`` replays scripts and produces byte-identical evidence for
    identical input, which is what QA-05's reproducibility check is built on.

    ``MANAGED`` loads the pinned Whisper checkpoint and hears real audio. It
    needs the ``managed`` extra, it needs the weights downloaded, and its
    executable environment is **not** pinned - `BASELINE_PINS.md` freezes that
    in Phase 3. So a figure produced in this mode is reproducible only by
    whoever ran it, and the runtime says so through
    ``environment_is_pinned``.
    """

    DETERMINISTIC = "deterministic"
    MANAGED = "managed"


class Settings(BaseSettings):
    """Configuration read from the environment, prefixed ``ENGINE_``."""

    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    environment: str = "local"
    backend: Backend = Backend.MEMORY
    runtime_mode: RuntimeMode = RuntimeMode.DETERMINISTIC

    #: Where the managed runtime runs. `cuda` unless a deployment says
    #: otherwise; `cpu` works and is roughly thirty times slower, which is not
    #: a configuration any reported figure should come from.
    whisper_device: str = "cuda"
    #: fp16 is what the pinned artifact is published as and halves residency.
    whisper_dtype: str = "float16"
    #: Where the weights are cached. Empty uses the library default, which
    #: respects HF_HOME.
    whisper_cache_dir: str = ""

    # -- secrets: required, no defaults ------------------------------------

    api_key_pepper: str = Field(
        min_length=32,
        description=(
            "Peppers the API-key hash. Lives in the secret manager, never in the "
            "database, so a leaked dump cannot be attacked offline."
        ),
    )
    stream_token_signing_key: str = Field(
        min_length=32,
        description="HMAC key for short-lived stream tokens (§6.1 step 2).",
    )

    #: A key the process registers at startup so that a local instance can be
    #: called - including from its own Swagger page - without a provisioning
    #: endpoint that does not exist yet.
    #:
    #: This is a development affordance and is written down as one. The memory
    #: backend keeps keys in a dict that dies with the process, so `issue-key`
    #: run in a shell cannot put anything into a *server's* directory; without
    #: this, the documented API is readable and uncallable. It is refused
    #: outside a local environment and refused with the persistent backend
    #: (`reject_a_bootstrap_key_outside_local`), because a credential arriving
    #: through an environment variable is exactly the habit §15.2 forbids.
    bootstrap_api_key: str = ""

    # -- infrastructure ----------------------------------------------------

    database_url: str = ""
    redis_url: str = ""
    object_storage_endpoint: str = ""
    object_storage_bucket: str = "evidence-engine"
    #: Credentials for the object store. Empty by default and required only for
    #: the persistent backend: the ambient-credential path that boto3 falls back
    #: to works on a cloud instance and fails on a laptop with a message about
    #: credentials rather than about configuration, so they are asked for
    #: explicitly and checked at startup.
    object_storage_access_key: str = ""
    object_storage_secret_key: str = ""

    # -- behaviour ---------------------------------------------------------

    max_queue_depth: int = Field(default=8, ge=1, le=256)
    stream_lease_ttl_seconds: int = Field(default=60, ge=5)
    #: Sessions per application per minute. NFR-023 wants quotas configurable
    #: rather than mandatory, so zero means unlimited.
    sessions_per_minute: int = Field(default=0, ge=0)

    @field_validator("api_key_pepper", "stream_token_signing_key")
    @classmethod
    def _reject_obvious_placeholders(cls, value: str) -> str:
        """Refuse the values a developer types to get past a startup error.

        Length alone does not stop `"changeme_changeme_changeme_change"`, and a
        weak signing key makes the session binding on a stream token forgeable
        - which is the one thing that token exists to prevent.
        """
        lowered = value.lower()
        for placeholder in ("changeme", "placeholder", "secret", "password", "xxxx"):
            if placeholder in lowered:
                raise ValueError(
                    f"the value contains '{placeholder}'; supply a real secret from "
                    "the secret manager (§15.2)"
                )
        return value

    @field_validator("bootstrap_api_key")
    @classmethod
    def _reject_a_guessable_bootstrap_key(cls, value: str) -> str:
        """A short bootstrap key is worse than none.

        Empty means the affordance is off, which is the normal case. A value
        that is set has to look like a key the directory would have issued -
        prefix and entropy - so that `ENGINE_BOOTSTRAP_API_KEY=dev` cannot open
        an instance somebody exposed to a network for five minutes.
        """
        if not value:
            return value
        if not value.startswith("oek_") or len(value) < 32:
            raise ValueError(
                "bootstrap_api_key must look like an issued key: the 'oek_' prefix "
                "and at least 32 characters. Generate one with "
                "`uv run evidence-engine issue-key`, which prints the export line."
            )
        return value

    def reject_a_bootstrap_key_outside_local(self) -> None:
        """Refuse the development affordance anywhere it could matter.

        Called at startup rather than as a field validator because it reads
        three fields at once, and raising here means the process dies at boot
        with a sentence instead of serving with a credential nobody meant to
        deploy.
        """
        if not self.bootstrap_api_key:
            return
        if self.environment != "local":
            raise ValueError(
                f"ENGINE_BOOTSTRAP_API_KEY is set and ENGINE_ENVIRONMENT is "
                f"{self.environment!r}. It is a local development affordance: a key "
                "delivered through an environment variable is what §15.2 forbids. "
                "Issue a real key through the control plane instead."
            )
        if self.backend is not Backend.MEMORY:
            raise ValueError(
                "ENGINE_BOOTSTRAP_API_KEY is set with a persistent backend. It "
                "registers into the in-memory directory only; against Postgres it "
                "would be silently ignored and you would be locked out believing "
                "you were not."
            )

    def require_infrastructure(self) -> None:
        """Fail fast when a non-memory backend is selected without its URLs."""
        if self.backend is not Backend.POSTGRES:
            return
        missing = [
            name
            for name, value in (
                ("ENGINE_DATABASE_URL", self.database_url),
                ("ENGINE_REDIS_URL", self.redis_url),
                ("ENGINE_OBJECT_STORAGE_ENDPOINT", self.object_storage_endpoint),
                ("ENGINE_OBJECT_STORAGE_ACCESS_KEY", self.object_storage_access_key),
                ("ENGINE_OBJECT_STORAGE_SECRET_KEY", self.object_storage_secret_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"backend='postgres' requires {', '.join(missing)}; refusing to start half-wired"
            )
