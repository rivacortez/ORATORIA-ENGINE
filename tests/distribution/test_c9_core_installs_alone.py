"""C9 - the distribution contract: the core installs and imports alone.

The other eight contracts govern *imports between modules* and import-linter
checks them against the source tree. None of them can see a `pyproject.toml`.
So `domain/` and `application/` were provably free of infrastructure while
`pip install oratoria-evidence-engine` still pulled FastAPI, SQLAlchemy,
asyncpg, Redis, aioboto3 and OpenTelemetry - because the dependency list said
so, and no test read the dependency list.

That gap is the whole reason this file builds an artefact instead of reading
one. A dependency can be declared correctly here and arrive anyway, dragged in
transitively by something else; the only statement worth making is about the
wheel a consumer actually installs.

**Why this is slow and why it stays.** It builds a wheel and creates a virtual
environment, which costs seconds rather than milliseconds. The claim it defends
- "this engine is embeddable" - is one of the two claims ADR-001 rests on, and
a claim that expensive to check is exactly the kind that rots when nobody
checks it. Marked `distribution` so it can be deselected locally; the evidence
battery runs it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import venv
import zipfile
from email import message_from_string
from pathlib import Path

import pytest

pytestmark = pytest.mark.distribution

ROOT = Path(__file__).resolve().parents[2]

#: Every runtime concern that must be opt-in. Written as distribution names
#: because that is what METADATA carries, and matched loosely so that
#: `sqlalchemy[asyncio]` and `uvicorn[standard]` are caught too.
FORBIDDEN_IN_BASE = (
    "fastapi",
    "uvicorn",
    "starlette",
    "sqlalchemy",
    "asyncpg",
    "alembic",
    "redis",
    "aioboto3",
    "boto3",
    "opentelemetry",
    "structlog",
    "pydantic",
    "torch",
    "transformers",
    "numpy",
    "librosa",
    "soundfile",
)

#: What a consumer of the core must be able to import. The public contracts are
#: the point: an embedding consumer reads the taxonomy, builds an evidence
#: document and narrows a `Measured | Unavailable`, and needs none of the above
#: to do it.
CORE_IMPORTS = (
    "evidence_engine",
    "evidence_engine.domain",
    "evidence_engine.domain.shared.taxonomy",
    "evidence_engine.domain.shared.measurement",
    "evidence_engine.domain.evidence.document",
    "evidence_engine.domain.transcript.transcript",
    "evidence_engine.application.ports.runtimes",
    "evidence_engine.application.services.speech_assembly",
)

#: Modules that must NOT be loaded as a side effect of the imports above. A
#: core that imported its own bootstrap would satisfy every import assertion
#: and still drag in the whole server on first use.
FORBIDDEN_LOADED = ("fastapi", "sqlalchemy", "redis", "torch", "pydantic")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the wheel once, from this working tree."""
    out = tmp_path_factory.mktemp("dist")

    # `uv build`, not `python -m pip wheel`: this repository's interpreter is a
    # uv-managed virtual environment and does not carry pip, so invoking pip
    # through it fails before the build starts. uv is how the suite is run, so
    # it is present by construction wherever this test runs at all.
    uv = shutil.which("uv")
    if uv is None:  # pragma: no cover - uv is how this suite is invoked
        pytest.fail("uv is not on PATH; C9 builds the wheel with it")

    build = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out), str(ROOT)],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:  # pragma: no cover - a packaging failure, not a test failure
        pytest.fail(f"could not build the wheel:\n{build.stdout}\n{build.stderr}")

    built = sorted(out.glob("oratoria_evidence_engine-*.whl"))
    assert built, f"no wheel produced in {out}: {list(out.iterdir())}"
    return built[0]


def test_the_wheel_declares_no_unconditional_runtime_dependency(wheel: Path) -> None:
    """Read METADATA, not pyproject.toml.

    The build backend is what decides what ends up in `Requires-Dist`, and a
    field in the source is a statement of intent rather than of fact.
    """
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = message_from_string(archive.read(name).decode("utf-8"))

    # A `Requires-Dist` with no environment marker is unconditional; one with
    # `; extra == "server"` is opt-in and expected.
    unconditional = [
        requirement
        for requirement in metadata.get_all("Requires-Dist") or []
        if "extra ==" not in requirement
    ]

    offenders = [
        requirement
        for requirement in unconditional
        for forbidden in FORBIDDEN_IN_BASE
        if requirement.lower().startswith(forbidden)
    ]
    assert not offenders, (
        f"the base wheel unconditionally requires {offenders}. Installing the core "
        "must not install a web framework, a database driver or a machine-learning "
        "stack - move it to an extra."
    )
    assert unconditional == [], (
        f"the core is meant to require nothing, and requires {unconditional}"
    )


def test_the_extras_are_the_ones_the_install_lines_promise(wheel: Path) -> None:
    """A published install line that resolves to nothing is worse than no line."""
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = message_from_string(archive.read(name).decode("utf-8"))

    provided = set(metadata.get_all("Provides-Extra") or [])
    assert {"server", "local", "record", "research"} <= provided, provided


@pytest.fixture(scope="module")
def clean_environment(wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Install the base wheel into a virtual environment with nothing else in it.

    `--no-deps` is deliberately **not** passed: the whole question is what pip
    pulls in when told to install this and left to resolve. Passing it would
    make the test assert its own premise.
    """
    root = tmp_path_factory.mktemp("venv")
    venv.create(root, with_pip=True, clear=True)
    python = root / ("Scripts" if sys.platform == "win32" else "bin") / "python"

    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", str(wheel)],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:  # pragma: no cover
        pytest.fail(f"could not install the base wheel:\n{install.stdout}\n{install.stderr}")
    return python


@pytest.mark.parametrize("module", CORE_IMPORTS)
def test_the_core_imports_with_nothing_installed(clean_environment: Path, module: str) -> None:
    """Each public surface, imported in an environment that has only the core."""
    result = subprocess.run(
        [str(clean_environment), "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"`import {module}` failed in a clean venv:\n{result.stderr}"


def test_importing_the_core_loads_no_server_or_ml_module(clean_environment: Path) -> None:
    """Nothing arrives by side effect.

    An import can succeed and still have pulled half a server into memory - a
    package `__init__` that imports its own bootstrap would pass every check
    above. This reads `sys.modules` afterwards and names what it found.
    """
    probe = (
        "import sys\n"
        + "\n".join(f"import {module}" for module in CORE_IMPORTS)
        + "\nloaded = [m for m in sys.modules if m.split('.')[0] in "
        + repr(list(FORBIDDEN_LOADED))
        + "]\nprint(sorted(set(loaded)))\n"
    )
    result = subprocess.run([str(clean_environment), "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"importing the core loaded {result.stdout.strip()}. Something under "
        "domain/ or application/ reaches into infrastructure at import time."
    )


def test_reaching_for_the_server_says_which_extra_is_missing(clean_environment: Path) -> None:
    """The refusal has to be actionable, or the core being light costs more than it saves.

    Without the guard this is `ModuleNotFoundError: No module named 'fastapi'`
    from inside a composition root: true, and it names a package rather than a
    capability, and it does not say the package is optional on purpose.
    """
    result = subprocess.run(
        [str(clean_environment), "-c", "import evidence_engine.bootstrap"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    message = result.stderr
    assert "MissingExtra" in message
    assert 'pip install "oratoria-evidence-engine[server]"' in message
    assert "not a broken install" in message
