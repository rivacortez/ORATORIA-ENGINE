"""``evidence-engine`` — the operator's command line.

``pyproject.toml`` has declared this entry point since the project started and
the module did not exist, so installing the package produced a command that
raised ``ModuleNotFoundError``. That is worse than no command: a published
entry point is a promise a consumer discovers by running it.

Two verbs.

``issue-key`` mints a key and prints the line that makes a server accept it.
It is deliberately *not* a call into a running directory. The memory backend
keeps keys in a dict owned by one process, so a key created in a shell exists
nowhere the server can see it - a command that printed one and said "paste this
into Swagger" would be handing out a credential that authenticates nothing. The
honest shape is: this mints, the server registers, and the coupling is one
environment variable that both sides can see.

``show-config`` prints the configuration snapshot a run would use, because the
snapshot decides what the numbers mean - the silence threshold, the taxonomy
version, whether a calibration exists - and reading it out of a container
factory by hand is not something a person should have to do.

There is no ``revoke``, and its absence is a finding rather than an omission:
the persistent directory can revoke but cannot issue, so the product has no
provisioning path at all outside this development affordance. Recorded in
docs/BUILD_RECORD.md §5.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import fields
from enum import StrEnum
from typing import Any

from evidence_engine.adapters.outbound.persistence.identity import generate_secret
from evidence_engine.bootstrap.container import (
    BOOTSTRAP_APPLICATION,
    BOOTSTRAP_TENANT,
    default_configuration,
)
from evidence_engine.domain.shared.identifiers import Identifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evidence-engine",
        description="Operator commands for the OratorIA Multimodal Evidence Engine.",
    )
    sub = parser.add_subparsers(dest="command")

    issue = sub.add_parser(
        "issue-key",
        help="mint a local development key and print how to make a server accept it",
        description=(
            "Mints a key and prints the environment variable a local server reads to "
            "register it. The secret is printed once and is not recoverable: only a "
            "peppered hash is ever stored (FR-002, US-006). This is a local "
            "development path - ENGINE_ENVIRONMENT must be 'local' and the backend "
            "must be 'memory', or the server refuses to start with the key set."
        ),
    )
    issue.add_argument("--json", action="store_true", dest="as_json")

    config = sub.add_parser("show-config", help="print the configuration snapshot a run would use")
    config.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.command == "issue-key":
        return _issue_key(args)
    if args.command == "show-config":
        return _show_config(args)

    parser.print_help()
    return 2


def _issue_key(args: argparse.Namespace) -> int:
    secret = generate_secret()

    if args.as_json:
        print(
            json.dumps(
                {
                    "secret": secret,
                    "environment_variable": "ENGINE_BOOTSTRAP_API_KEY",
                    "tenant": BOOTSTRAP_TENANT,
                    "application": BOOTSTRAP_APPLICATION,
                },
                indent=2,
            )
        )
        return 0

    print()
    print("  key minted. It is printed once and is not recoverable.")
    print()
    print(f"    {secret}")
    print()
    print("  The server has to be told about it - a key minted here exists only in")
    print("  this shell until a process registers it. Start the server with:")
    print()
    print(f"    export ENGINE_BOOTSTRAP_API_KEY={secret}")
    print("    ./scripts/serve.sh")
    print()
    print("  Then open http://127.0.0.1:8000/v1/docs, press Authorize, and paste the")
    print("  key. It is sent as `Authorization: Bearer <key>`.")
    print()
    print(f"  Requests made with it are attributed to tenant {BOOTSTRAP_TENANT!r},")
    print(f"  application {BOOTSTRAP_APPLICATION!r} - a name no real study would use,")
    print("  so evidence produced from a local trial is identifiable as such in the")
    print("  audit log.")
    return 0


def _show_config(args: argparse.Namespace) -> int:
    """What the snapshot says, without importing a container factory by hand."""
    configuration = default_configuration()

    def readable(value: Any) -> str:
        """Render one field without asking it a question it may refuse.

        The obvious version of this used `hasattr(value, "value")`, and the
        domain caught it: `Unseeded` raises `FabricatedValue` from
        `__getattr__` for exactly that name, so `hasattr` propagated rather
        than returning False and the whole command died on the provenance
        field. That is NFR-015 working - a deterministic run has no seed, and
        the type refuses to be asked for one instead of returning a plausible
        blank. Types are matched explicitly here, so nothing gets probed.
        """
        if isinstance(value, Identifier | StrEnum):
            return str(value.value)
        if isinstance(value, Mapping | frozenset | set | tuple | list):
            return f"{len(value)} entries"
        return str(value)

    if args.as_json:
        print(
            json.dumps(
                {f.name: readable(getattr(configuration, f.name)) for f in fields(configuration)},
                indent=2,
            )
        )
        return 0

    print("\nconfiguration snapshot:\n")
    for field in fields(configuration):
        print(f"  {field.name:<26} {readable(getattr(configuration, field.name))}")
    print(
        "\n  Every threshold here is a starting point, not a calibrated value. An\n"
        "  empty threshold map and an absent calibration are why no publication\n"
        "  gate opens today: an uncalibrated class reports a raw confidence, and a\n"
        "  raw confidence never clears a threshold (§14.2)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
