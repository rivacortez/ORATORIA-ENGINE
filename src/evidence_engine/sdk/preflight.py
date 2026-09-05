"""What the machine has, and nothing about whether the model will work.

The distinction this module exists to keep is narrow and load-bearing.

``hardware_preflight()`` inspects the machine: is torch importable, does it see
a CUDA device, how much memory does that device have, what compute capability,
are the inference libraries installed. Every one of those is answerable in
milliseconds and none of them is evidence that inference succeeds.

``OratoriaEngine.warmup()`` loads the weights and decodes a short window. That
is the check. It costs a 3 GB download the first time and several seconds
after, and it is the only thing that can honestly say the model runs here.

Collapsing the two would be the tempting shape - one `check()` that returns
True - and it would produce exactly the failure this project keeps refusing: a
green result that stands in for a measurement nobody took. A preflight that
returned `ok` on a machine where the weights are corrupt, the driver is too
old for the kernel, or the model does not fit beside whatever else holds VRAM
would be worse than no preflight, because a consumer would stop looking.

So the report says what it saw and, at the end, says what it did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec

#: fp16 `whisper-large-v3` is 3.09 GB of weights and loads in roughly 4.2 GiB
#: with activations. Stated as the observed residency on this project's
#: workstation rather than as a vendor figure, and used to *warn* rather than
#: to refuse: a card below it may still work with a smaller window, and this
#: module is not entitled to decide that.
BASELINE_RESIDENCY_MIB = 4_300


@dataclass(frozen=True, slots=True)
class HardwareReport:
    """What the preflight observed. Not a verdict on the model."""

    torch_installed: bool
    torch_version: str | None
    inference_libraries_installed: bool
    cuda_available: bool
    device_name: str | None
    device_memory_mib: int | None
    compute_capability: tuple[int, int] | None
    #: Everything that would stop local inference, in the order it would fail.
    #: Empty does **not** mean the model runs - see `warmup()`.
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def can_attempt_local_inference(self) -> bool:
        """Whether it is worth calling ``warmup()``.

        Deliberately not named ``ok`` or ``ready``. Nothing here has run the
        model, and a property called `ready` would be read as though something
        had.
        """
        return not self.blockers

    def render(self) -> str:
        """The report a person reads, ending with what it did not check."""
        libraries = "installed" if self.inference_libraries_installed else "missing"
        lines = [
            f"  torch                {self.torch_version or 'not installed'}",
            f"  inference libraries  {libraries}",
            f"  CUDA device          {self.device_name or 'none'}",
        ]
        if self.device_memory_mib is not None:
            lines.append(f"  device memory        {self.device_memory_mib} MiB")
        if self.compute_capability is not None:
            major, minor = self.compute_capability
            lines.append(f"  compute capability   {major}.{minor}")
        for blocker in self.blockers:
            lines.append(f"  BLOCKER              {blocker}")
        for warning in self.warnings:
            lines.append(f"  warning              {warning}")

        lines.append("")
        lines.append("  This checked the machine and nothing else. It did not load the")
        lines.append("  weights, verify their digest, or decode a single sample, so it")
        lines.append("  cannot tell you the model runs here. `await engine.warmup()`")
        lines.append("  does that, and it is the only thing that does.")
        return "\n".join(lines)


def hardware_preflight(device: str = "cuda") -> HardwareReport:
    """Inspect the machine. Costs milliseconds and proves nothing about the model.

    Imports are done through ``find_spec`` and inside the function, so calling
    this on a core-only install answers "torch is not installed" instead of
    raising - which is the answer a consumer deciding whether to install the
    `local` extra actually wants.
    """
    blockers: list[str] = []
    warnings: list[str] = []

    torch_installed = find_spec("torch") is not None
    libraries = all(find_spec(name) is not None for name in ("transformers", "numpy"))

    if not torch_installed:
        blockers.append(
            "torch is not installed. It is deliberately absent from every extra: "
            "the wheel index depends on your card, and resolving it from PyPI "
            "installs the CPU build over a working CUDA one."
        )
    if not libraries:
        blockers.append(
            'the inference libraries are missing: pip install "oratoria-evidence-engine[local]"'
        )

    if not torch_installed:
        return HardwareReport(
            torch_installed=False,
            torch_version=None,
            inference_libraries_installed=libraries,
            cuda_available=False,
            device_name=None,
            device_memory_mib=None,
            compute_capability=None,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        )

    import torch

    cuda = bool(torch.cuda.is_available())
    name: str | None = None
    memory_mib: int | None = None
    capability: tuple[int, int] | None = None

    if cuda:
        properties = torch.cuda.get_device_properties(0)
        name = str(properties.name)
        memory_mib = int(properties.total_memory // (1024 * 1024))
        capability = (int(properties.major), int(properties.minor))
        if memory_mib < BASELINE_RESIDENCY_MIB:
            warnings.append(
                f"{memory_mib} MiB of device memory; the pinned baseline was observed "
                f"at about {BASELINE_RESIDENCY_MIB} MiB. It may still run with a "
                "shorter window - warmup() is what finds out."
            )
    elif device.startswith("cuda"):
        blockers.append(
            f"device {device!r} was requested and torch reports no CUDA device. "
            "CPU inference works and is far slower than real time, which is not a "
            "configuration any reported figure should come from."
        )

    return HardwareReport(
        torch_installed=True,
        torch_version=str(torch.__version__),
        inference_libraries_installed=libraries,
        cuda_available=cuda,
        device_name=name,
        device_memory_mib=memory_mib,
        compute_capability=capability,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
