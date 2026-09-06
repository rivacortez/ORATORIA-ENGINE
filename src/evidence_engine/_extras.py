"""Telling a consumer which extra they are missing, and how to install it.

The core installs with no dependencies at all. Everything that turns it into a
running service - FastAPI, SQLAlchemy, Redis, an object store - or into a local
recogniser lives behind an extra, which means a consumer will eventually import
something that is not there.

What they get without this module is ``ModuleNotFoundError: No module named
'fastapi'`` raised from four frames inside a composition root. That is a true
statement and a useless one: it names a package rather than a capability, and
it does not say that the package is deliberately optional or what to type.

So the guard fires at the package boundary, names the extra, and prints the
install line. One sentence, at the point where the decision was made.
"""

from __future__ import annotations

from importlib.util import find_spec

#: The distribution name a consumer types. Kept here rather than read from
#: metadata, because this has to work when the package is on `sys.path`
#: without being installed - which is how the repository itself runs.
DISTRIBUTION = "oratoria-evidence-engine"


class MissingExtra(ImportError):
    """A capability was reached for and its extra is not installed."""


def require(extra: str, *modules: str, because: str) -> None:
    """Refuse, with a sentence, when an optional dependency is absent.

    ``because`` says what the caller was trying to do. Without it the message
    is still actionable and not yet *informative*: a consumer who did not know
    the engine could run without a database needs to be told that is a
    supported configuration rather than a broken install.

    Checks with ``find_spec`` instead of importing, so the guard costs a path
    lookup rather than pulling a web framework into memory to discover that it
    is present.
    """
    absent = [name for name in modules if find_spec(name) is None]
    if not absent:
        return

    raise MissingExtra(
        f"{because}\n\n"
        f"  missing: {', '.join(absent)}\n"
        f'  install: pip install "{DISTRIBUTION}[{extra}]"\n\n'
        f"This is not a broken install. The core of this engine has no "
        f"dependencies at all - it is embeddable, and every runtime concern is "
        f"opt-in. See the `{extra}` extra in pyproject.toml for what it adds "
        f"and why."
    )
