"""ASE calculator factories — the seam between SLAB and the physics engines.

SLAB never implements physics. Engines are reached through the ASE
``Calculator`` contract; this module maps an engine name to a ready calculator
instance. Three sources feed the mapping, in resolution order:

* **Built-ins** — engines the ``slab`` package can construct on its own:
  ``mace`` (in-process, via the ``slab[mace]`` extra), ``rootstock`` (an MLIP
  served from a cluster's pre-built rootstock install, via the
  ``slab[rootstock]`` extra), and ASE's ``emt``/``lj`` toys for tests.
* **The cluster engine registry** (:mod:`slab.engines`) — names a cluster
  maintainer declared (``lammps``, ``qe``, ``vasp``, site-specific MLIP
  aliases, ...), resolved rootstock-style: the client only finds the registry
  file; the file says how each engine is built *here*.
* **Rootstock checkpoint ids, served silently** — any canonical checkpoint id
  the cluster's rootstock install declares (``mace-mp-0-medium``,
  ``uma-s-1p1``, ...) works directly as an engine name; rootstock resolves
  the hosting environment and serves the model. No registry entry needed —
  the rootstock install *is* the declaration, exactly in the spirit of
  "the install describes itself".

Registry entries deliberately win over checkpoint ids: a maintainer's curated
alias (with baked-in device/setup options) beats bare resolution. Adding a
backend means adding a registry entry (or a factory here) — nothing in the
tracing, lifecycle, or retention layers knows engines exist.

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

from pathlib import Path
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

    Resolution order: built-ins, then the cluster engine registry (discovered
    via ``$SLAB_ENGINES`` / ``~/.config/slab/engines.json`` — see
    :mod:`slab.engines`; defaults merge under caller options), then rootstock
    *checkpoint ids*: any canonical id the cluster's rootstock install
    declares works directly as the engine name —
    ``get_calculator("mace-mp-0-medium", cluster="delta")`` serves the MACE
    model silently from its pre-built environment. The install is found via
    ``cluster=``/``root=`` options, else rootstock's own defaults
    (``$ROOTSTOCK_ROOT``, ``~/.config/rootstock/config.toml``).

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

    resolution, note = _resolve_rootstock_checkpoint(normalized, options)
    if resolution is not None:
        if "checkpoint" in options:
            raise EngineNotAvailableError(
                f"engine {engine!r} is itself a rootstock checkpoint id; do not also "
                f"pass checkpoint={options['checkpoint']!r} in calculator_options"
            )
        return _rootstock_calculator(checkpoint=normalized, **options)

    known = ", ".join(available_engines(registry))
    notes = []
    if registry is None:
        notes.append("no engine registry configured — see $SLAB_ENGINES")
    if note:
        notes.append(note)
    detail = f" ({'; '.join(notes)})" if notes else ""
    raise EngineNotAvailableError(f"unknown engine {engine!r}; available: {known}{detail}")


def describe_engine(
    engine: str, calculator_options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Identity of an engine name: where it resolves, what version, what spec.

    Used by tasks both as *provenance* (which install produced this result)
    and as *cache identity*, mirroring :func:`get_calculator`'s resolution
    order exactly. For registry engines the full spec is included, so any
    change a maintainer makes to the entry — version, options, env,
    calculator — changes the fingerprint and honestly invalidates cached
    results, not just version bumps.

    A name that resolves as a rootstock checkpoint id reports
    ``source="rootstock"`` with the rootstock *client* version. The identity
    is deliberately the checkpoint id plus client version, not the serving
    install's path or hosting environment: rootstock's contract is that
    canonical ids are stable identities, so the same id on another install is
    the same computation, while cluster-side internals (env rebuilds,
    in-place weight edits) are invisible to any client.

    Args:
        calculator_options: The options the calculator would be built with —
            checkpoint resolution may need ``cluster``/``root`` from them.

    Examples:
        >>> describe_engine("emt")["source"]
        'builtin'
    """
    options = calculator_options or {}
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
    resolution, _note = _resolve_rootstock_checkpoint(normalized, options)
    if resolution is not None:
        return {
            "engine": normalized,
            "source": "rootstock",
            "version": _dist_version("rootstock"),
            "checkpoint": normalized,
        }
    return {"engine": engine, "source": "unknown", "version": None}


def _resolve_rootstock_checkpoint(
    name: str, options: dict[str, Any]
) -> tuple[dict[str, str] | None, str | None]:
    """Classify *name* as a rootstock checkpoint id, if an install declares it.

    Returns ``(resolution, note)``: a resolution dict when some installed
    environment declares the id; otherwise ``None`` plus an optional note
    explaining why (for error messages). Quietly not-a-checkpoint when the
    rootstock package is absent — silent serving is opt-in via the extra.
    """
    try:
        from rootstock.clusters import get_cluster
        from rootstock.config import resolve_default_root
        from rootstock.environment import CheckpointNotFoundError, resolve_checkpoint
    except ImportError:
        return None, None
    if "root" in options:
        root = Path(options["root"])
    elif "cluster" in options:
        try:
            root = get_cluster(options["cluster"]).root
        except ValueError as e:  # unknown cluster: their message lists known ones
            raise EngineNotAvailableError(str(e)) from e
    else:
        root = resolve_default_root()
    if root is None:
        return None, (
            "rootstock is installed but no install root is configured — pass "
            "calculator_options={'cluster': ...} or set $ROOTSTOCK_ROOT to serve "
            "checkpoint ids directly"
        )
    try:
        resolved = resolve_checkpoint(root, name, options.get("cluster"))
    except CheckpointNotFoundError:
        return None, (
            f"not declared as a checkpoint by the rootstock install at {root} "
            f"('slab engines list' shows what is)"
        )
    except OSError as e:
        return None, f"could not read the rootstock install at {root}: {e}"
    return {
        "checkpoint": resolved.checkpoint,
        "env_name": resolved.env_name,
        "root": str(root),
    }, None


def _dist_version(distribution: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in all test envs
        return None


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
