"""ASE calculator factories — the seam between SLAB and the physics engines.

SLAB never implements physics. Engines are reached through the ASE
``Calculator`` contract; this module maps an engine name to a ready calculator
instance. Adding a backend means adding a factory here — nothing in the
tracing, lifecycle, or retention layers knows engines exist.

Available engines:

* ``mace`` — the MACE foundation model (``pip install slab[mace]``); options
  are forwarded to ``mace.calculators.mace_mp`` (``model=``, ``device=``, ...).
  The first use downloads the model checkpoint to ``~/.cache/mace``.
* ``emt`` — ASE's built-in effective-medium-theory calculator. No extra
  install, milliseconds per step; only fit for the elements it parametrizes
  (Al/Cu/Ag/Au/Ni/Pd/Pt). Ideal for tests and quick demos, not for science.
* ``lj`` — ASE's Lennard-Jones calculator; a toy, present for completeness.
"""

from __future__ import annotations

from typing import Any, Protocol

from slab.errors import EngineNotAvailableError


class Calculator(Protocol):
    """Structural stand-in for ``ase.calculators.calculator.Calculator``."""

    def get_potential_energy(self, atoms: Any = None) -> float: ...


def available_engines() -> tuple[str, ...]:
    """Names accepted by :func:`get_calculator`.

    Examples:
        >>> available_engines()
        ('emt', 'lj', 'mace')
    """
    return ("emt", "lj", "mace")


def get_calculator(engine: str, **options: Any) -> Any:
    """Build an ASE calculator for *engine*, forwarding *options* to it.

    Raises:
        EngineNotAvailableError: The engine name is unknown, or its backend
            package is not installed (the message says how to install it).

    Examples:
        >>> calc = get_calculator("emt")
        >>> type(calc).__name__
        'EMT'
    """
    normalized = engine.strip().lower()
    if normalized == "emt":
        from ase.calculators.emt import EMT

        return EMT(**options)
    if normalized == "lj":
        from ase.calculators.lj import LennardJones

        return LennardJones(**options)
    if normalized == "mace":
        return _mace_calculator(**options)
    raise EngineNotAvailableError(
        f"unknown engine {engine!r}; available: {', '.join(available_engines())}"
    )


def _mace_calculator(**options: Any) -> Any:
    try:
        from mace.calculators import mace_mp
    except ImportError as e:
        raise EngineNotAvailableError(
            "engine 'mace' needs the mace-torch package: pip install 'slab[mace]'"
        ) from e
    options.setdefault("model", "small")
    options.setdefault("device", "cpu")
    options.setdefault("default_dtype", "float64")
    return mace_mp(**options)
