"""ASE calculator factories — the seam between SLAB and the physics engines.

SLAB never implements physics. Engines are reached through the ASE
``Calculator`` contract; this module maps an engine name to a ready calculator
instance. Two sources feed the mapping:

* **Built-ins** — engines the ``slab`` package can construct on its own:
  ``mace`` (in-process, via the ``slab[mace]`` extra), ``rootstock`` (an MLIP
  served from a cluster's pre-built rootstock install, via the
  ``slab[rootstock]`` extra), and ASE's ``emt``/``lj`` toys for tests.
* **The cluster engine registry** (:mod:`slab.engines`) — names a cluster
  maintainer declared (``lammps``, ``qe``, ``vasp``, site-specific MLIP
  aliases, ...), resolved rootstock-style: the client only finds the registry
  file; the file says how each engine is built *here*.

Built-ins win on name collision; everything else consults the registry.
Adding a backend means adding a registry entry (or a factory here) — nothing
in the tracing, lifecycle, or retention layers knows engines exist.

Engine choices worth knowing:

* ``rootstock`` — options are forwarded to ``rootstock.RootstockCalculator``:
  ``checkpoint`` (canonical id, required), ``cluster`` or ``root``,
  ``device``, ``setup_kwargs``, ... The heavy MLIP dependencies live in the
  cluster's pre-built environments, not in your Python environment; the
  calculator spawns a worker subprocess, so it must be closed —
  :func:`close_calculator` does this and :func:`slab.tasks.relax` calls it
  automatically.
* ``mace`` — the MACE foundation model in-process; options are forwarded to
  ``mace.calculators.mace_mp`` (``model=``, ``device=``, ...). First use
  downloads the checkpoint to ``~/.cache/mace``.
* ``emt``/``lj`` — ASE built-ins. Milliseconds per step, fit only for the
  elements they parametrize; ideal for tests, not for science.
"""

from __future__ import annotations

from typing import Any, Protocol

from slab.engines import EngineRegistry, build_engine, load_registry, registry_engine_names
from slab.errors import EngineNotAvailableError


class Calculator(Protocol):
    """Structural stand-in for ``ase.calculators.calculator.Calculator``."""

    def get_potential_energy(self, atoms: Any = None) -> float: ...


def available_engines(registry: EngineRegistry | None = None) -> tuple[str, ...]:
    """Names accepted by :func:`get_calculator`: built-ins plus registry entries.

    Pass a loaded registry to include its names; ``None`` lists built-ins only
    (callers wanting the ambient registry pass ``load_registry()``).

    Examples:
        >>> available_engines()
        ('emt', 'lj', 'mace', 'rootstock')
    """
    builtin = ("emt", "lj", "mace", "rootstock")
    extra = tuple(name for name in registry_engine_names(registry) if name not in builtin)
    return builtin + extra


def get_calculator(engine: str, **options: Any) -> Any:
    """Build an ASE calculator for *engine*, forwarding *options* to it.

    Resolution: built-ins first, then the cluster engine registry (discovered
    via ``$SLAB_ENGINES`` / ``~/.config/slab/engines.json`` — see
    :mod:`slab.engines`). Registry defaults merge under caller options.

    Raises:
        EngineNotAvailableError: The engine name is unknown here, or its
            backend package is not installed (the message says how to fix it).

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
    if normalized == "rootstock":
        return _rootstock_calculator(**options)

    registry = load_registry()
    if registry is not None and normalized in registry.engines:
        return build_engine(normalized, registry.engines[normalized], **options)
    known = ", ".join(available_engines(registry))
    hint = "" if registry is not None else " (no engine registry configured — see $SLAB_ENGINES)"
    raise EngineNotAvailableError(f"unknown engine {engine!r}; available: {known}{hint}")


def describe_engine(engine: str) -> dict[str, Any]:
    """Identity of an engine name: where it resolves, what version, what spec.

    Used by tasks both as *provenance* (which install produced this result)
    and as *cache identity*: for registry engines the full spec is included,
    so any change a maintainer makes to the entry — version, options, env,
    calculator — changes the fingerprint and honestly invalidates cached
    results, not just version bumps.

    For the ``rootstock`` built-in, identity beyond the client library version
    lives in ``calculator_options`` (checkpoint id, cluster) which are traced
    inputs already; the served environment's internals (env rebuilds,
    in-place weight edits on the cluster) are outside what the client can
    see — rootstock's contract is that checkpoint ids are stable identities.

    Examples:
        >>> describe_engine("emt")["source"]
        'builtin'
    """
    normalized = engine.strip().lower()
    if normalized in ("emt", "lj", "mace", "rootstock"):
        return {"engine": normalized, "source": "builtin", "version": None}
    registry = load_registry()
    if registry is not None and normalized in registry.engines:
        spec = registry.engines[normalized]
        return {
            "engine": normalized,
            "source": f"registry:{registry.cluster}" if registry.cluster else "registry",
            "version": spec.version,
            "calculator": spec.calculator,
            "spec": spec.model_dump(mode="json"),
        }
    return {"engine": engine, "source": "unknown", "version": None}


def close_calculator(calculator: Any) -> None:
    """Release a calculator's resources, if it holds any.

    Worker-backed calculators (rootstock spawns a subprocess per instance)
    expose ``close()``; in-process ones don't. Safe on both, and safe to call
    twice.

    Examples:
        >>> close_calculator(get_calculator("emt"))  # no-op for in-process engines
    """
    close = getattr(calculator, "close", None)
    if callable(close):
        close()


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


def _rootstock_calculator(**options: Any) -> Any:
    try:
        from rootstock import RootstockCalculator
    except ImportError as e:
        raise EngineNotAvailableError(
            "engine 'rootstock' needs the rootstock package: pip install 'slab[rootstock]'"
        ) from e
    if "checkpoint" not in options:
        raise EngineNotAvailableError(
            "engine 'rootstock' requires a checkpoint id, e.g. "
            "calculator_options={'checkpoint': 'mace-mp-0-medium', 'cluster': 'delta'}"
        )
    return RootstockCalculator(**options)
