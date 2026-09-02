"""The cluster engine registry: rootstock's management pattern, generalized.

`Garden-AI/rootstock <https://github.com/Garden-AI/rootstock>`_ runs MLIPs on
HPC clusters behind one ASE calculator by combining four ideas SLAB adopts
wholesale for *all* engines (LAMMPS, Quantum ESPRESSO, VASP, MLIPs):

1. **The client is only a bootstrap.** Nothing about a cluster's software is
   baked into the ``slab`` package — a baked-in table goes stale in pinned
   clients. The client merely *finds* a registry file; everything else is
   declared by the file, which lives with the install and travels with it.
2. **Canonical names, capability resolution.** Workflow code says
   ``engine="vasp"`` or ``engine="mace-mp"``; the registry maps the name to a
   concrete, maintainer-verified way of building an ASE calculator on *this*
   cluster. The same workflow script runs unchanged on any cluster whose
   registry declares the names it uses.
3. **Maintainer-verified installs.** Each entry may declare a ``probe`` —
   a cheap command proving the engine is actually usable here. ``slab
   engines verify`` runs them all; users trust names that verify.
4. **An explicit layout contract.** The registry carries a
   ``layout_version``; a client that meets a newer version than it
   understands says so and stops instead of misreading.

The registry is one JSON file of :class:`EngineRegistry` shape. Discovery
order: an explicit path, else ``$SLAB_ENGINES``, else ``[paths] engines``
from the slab config (:mod:`slab.config`), else
``~/.config/slab/engines.json``. Cluster maintainers typically ship the file
at a shared path and point at it from the site config (or export
``SLAB_ENGINES`` from a module file, mirroring rootstock's
``ROOTSTOCK_ROOT``).

Every entry builds an ASE calculator from a dotted path — the ASE
``Calculator`` contract is SLAB's engine seam, and that includes rootstock
itself (``rootstock.RootstockCalculator``)::

    {
      "layout_version": 1,
      "cluster": "delta",
      "engines": {
        "mace-mp": {
          "calculator": "rootstock.RootstockCalculator",
          "options": {"cluster": "delta", "checkpoint": "mace-mp-0-medium"},
          "version": "rootstock/almanac",
          "description": "MACE-MP via rootstock-managed environments"
        },
        "qe-delta": {
          "calculator": "slab.backends.qe_calculator",
          "options": {"command": "srun pw.x", "pseudo_dir": "/sw/pseudos/sssp"},
          "version": "7.3.1",
          "probe": ["pw.x", "-h"]
        }
      }
    }

(``qe-delta``, not ``qe``: plain ``qe`` is a built-in engine, and entries
shadowing built-in names are refused at load. And ``options``, not
``env: {"ASE_CONFIG_PATH": ...}``: ASE parses its config file once at
import time, before any registry entry runs, so that variable is refused —
see :class:`EngineSpec`.)
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slab.errors import EngineNotAvailableError

LAYOUT_VERSION = 1
REGISTRY_ENV_VAR = "SLAB_ENGINES"
_USER_REGISTRY = Path("~/.config/slab/engines.json")
_PROBE_TIMEOUT_S = 120
_BUILTIN_ENGINES = ("emt", "lammps", "lj", "qe", "rootstock")
# Variables ASE reads exactly once, at import time — and slab imports ASE
# before any registry entry runs, so declaring them in an entry's env would
# silently never apply. Refused at validation instead (see EngineSpec).
_IMPORT_TIME_ENV = frozenset({"ASE_CONFIG_PATH"})
# Variables slab's OWN resolution consults at run time — built-in engine
# commands, config/registry/pseudos discovery, scheduler detection. An entry
# setting one can redirect slab's behavior for the rest of the process, so
# the application warns even when the variable was previously unset.
# (Explicit config still outranks the environment where both exist.)
_BUILTIN_STEERING_ENV = frozenset(
    {
        "ASE_LAMMPSRUN_COMMAND",
        "LAMMPS_POTENTIALS",
        "SLAB_CONFIG",
        "SLAB_SITE_CONFIG",
        "SLAB_ENGINES",
        "SLAB_PSEUDOS",
        "SLAB_WORKSPACE",
        "SLURM_JOB_ID",
        "XDG_CONFIG_HOME",
    }
)
# What build_engine wrote into os.environ: variable -> (ORIGINAL value
# before the first application, None = originally unset; the value most
# recently APPLIED). Registry env exists for THIS process's calculators;
# slab.hpc.submit uses this map to hand sbatch the environment as it was
# before any registry entry ran — but only where the current value still IS
# the applied one: a value the user changed since is their intent, not
# residue, and silently reverting it would be its own poisoning.
_APPLIED_ENV: dict[str, tuple[str | None, str]] = {}


def applied_env() -> dict[str, tuple[str | None, str]]:
    """Registry-applied environment: variable -> (original, applied) values.

    Consumers restore, never just delete: a variable the user's own shell
    exported and a registry entry overwrote goes back to the shell's value
    at process boundaries (job submission), not to nothing — and a variable
    the user re-set AFTER application is left exactly as they set it.
    """
    return dict(_APPLIED_ENV)


def _warn_always(message: str) -> None:
    """``warnings.warn`` without the once-per-location dedup.

    Env poisoning must be visible on every occurrence: a long-lived process
    (the MCP server, a Mason session) alternating between entries would
    otherwise warn once and mutate silently ever after. ``warn_explicit``
    with a fresh registry defeats the dedup while still honoring the active
    filters — so test recorders and user-set error filters behave normally.
    """
    warnings.warn_explicit(message, UserWarning, filename=__file__, lineno=0, registry=None)


class EngineSpec(BaseModel):
    """One engine as a cluster's registry declares it.

    Fields:
        kind: How the entry is realized. Only ``"ase"`` exists today — build
            an ASE calculator from ``calculator`` — but the field (plus
            ``layout_version``) is the seam for future kinds; older clients
            refuse newer entries loudly instead of misreading them.
        calculator: Dotted path to the ASE calculator class or factory
            (``ase.calculators.espresso.Espresso``, ``ase.calculators.vasp.Vasp``,
            ``ase.calculators.lammpsrun.LAMMPS``, ``rootstock.RootstockCalculator``).
        options: Default keyword arguments for the calculator. Caller options
            override these key-by-key.
        env: Environment variables the code reads *at run time*
            (``VASP_PP_PATH``, ``ASE_VASP_COMMAND``, ...). Applied to the
            process environment at build time and left in place — those
            calculators consult ``os.environ`` when they calculate, so the
            application is deliberately process-wide (see
            :func:`build_engine`); entries needing conflicting values for the
            same variable cannot share a process. Variables ASE reads only
            once at import time (``ASE_CONFIG_PATH``) are refused here: slab
            imports ASE before any registry entry runs, so the value would
            silently never apply — declare the engine explicitly instead
            (e.g. ``calculator = "slab.backends.qe_calculator"`` with
            ``command``/``pseudo_dir`` in ``options``), or export the
            variable before Python starts (shell, module file).
        version: The maintainer's declared version of the underlying code —
            recorded into task recipes as provenance and, together with the
            rest of the spec, folded into relax's cache key.
        probe: Command argv that proves the engine works here (used by
            ``slab engines verify``); exit code 0 is a pass.

    Examples:
        >>> spec = EngineSpec.model_validate({
        ...     "calculator": "ase.calculators.emt.EMT", "version": "ase-built-in",
        ... })
        >>> spec.kind
        'ase'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["ase"] = "ase"
    calculator: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    version: str | None = None
    description: str = ""
    probe: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _env_is_runtime_read(self) -> EngineSpec:
        dead = sorted(set(self.env) & _IMPORT_TIME_ENV)
        if dead:
            raise ValueError(
                f"env declares {dead}, which ASE reads exactly once at import "
                f"time — and slab imports ASE before any registry entry runs, "
                f"so the value would silently never apply. Declare the engine "
                f"explicitly instead (calculator = 'slab.backends.qe_calculator' "
                f"with command/pseudo_dir in options), or export the variable "
                f"before Python starts (shell, module file)"
            )
        return self


class EngineRegistry(BaseModel):
    """A cluster's declared engines — policy-as-data, like retention.

    Examples:
        >>> registry = EngineRegistry.model_validate({
        ...     "cluster": "delta",
        ...     "engines": {"emt-demo": {"calculator": "ase.calculators.emt.EMT"}},
        ... })
        >>> sorted(registry.engines)
        ['emt-demo']
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout_version: int = 1
    cluster: str = ""
    engines: dict[str, EngineSpec] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _layout_supported(self) -> EngineRegistry:
        if self.layout_version > LAYOUT_VERSION:
            raise ValueError(
                f"engine registry declares layout_version {self.layout_version}, newer than "
                f"this slab supports ({LAYOUT_VERSION}); upgrade slab to read it"
            )
        # Engine names are canonical: lowercase, unpadded. Lookups normalize
        # the caller's name the same way, so a declared name and a resolvable
        # name can never diverge (a case-variant like "QE" would otherwise
        # validate and then silently resolve to the built-in).
        non_canonical = sorted(n for n in self.engines if n != n.strip().lower())
        if non_canonical:
            raise ValueError(f"engine names must be lowercase and unpadded: {non_canonical}")
        # Ambiguity is refused, never guessed (a rootstock lesson): an entry
        # shadowing a built-in engine name would silently lose or silently win
        # depending on resolution order — either way a user gets a different
        # engine than the registry maintainer or slab intended.
        shadowed = sorted(set(self.engines) & set(_BUILTIN_ENGINES))
        if shadowed:
            raise ValueError(
                f"engine registry redefines built-in engine name(s) {shadowed}; "
                f"pick distinct names (e.g. 'qe-delta' instead of 'qe')"
            )
        return self


class ProbeResult(BaseModel):
    """Outcome of verifying one engine (``slab engines verify``)."""

    model_config = ConfigDict(frozen=True)

    engine: str
    ok: bool
    detail: str


def find_registry_path(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate the registry file: explicit path > $SLAB_ENGINES > [paths] engines > ~/.config.

    Returns None when no candidate exists — running without a registry is the
    normal laptop case (built-in engines only). An explicit path or an
    ``$SLAB_ENGINES`` value that does not exist is an error, not a silent
    fallback: a maintainer pointed there on purpose.

    Examples:
        >>> import os
        >>> os.environ.pop("SLAB_ENGINES", None) and None
        >>> find_registry_path() is None or find_registry_path().name == "engines.json"
        True
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"engine registry not found: {path}")
        return path
    from_env = os.environ.get(REGISTRY_ENV_VAR)
    if from_env:
        path = Path(from_env).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"${REGISTRY_ENV_VAR} points to {path}, which does not exist")
        return path
    from slab.config import config_value

    configured = config_value("paths.engines")
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"[paths] engines in the slab config points to {path}, which does not exist"
            )
        return path
    # $XDG_CONFIG_HOME wins over ~/.config, as it does for the user config
    # (slab.config.user_config_path); the two must agree on where "user"
    # lives, and tests isolate through that variable.
    xdg = os.environ.get("XDG_CONFIG_HOME")
    user_path = (
        Path(xdg).expanduser() / "slab" / "engines.json" if xdg else _USER_REGISTRY.expanduser()
    )
    return user_path if user_path.exists() else None


def load_registry(path: str | os.PathLike[str] | None = None) -> EngineRegistry | None:
    """Load the engine registry, or None when no registry is configured.

    Invalid content is an error, never a silent empty registry — a broken
    declaration on a cluster must surface, not degrade to "engine unknown".

    Examples:
        >>> import json, tempfile
        >>> path = tempfile.mktemp(suffix=".json")
        >>> _ = open(path, "w").write(json.dumps({
        ...     "cluster": "delta",
        ...     "engines": {"emt-demo": {"calculator": "ase.calculators.emt.EMT"}},
        ... }))
        >>> load_registry(path).cluster
        'delta'
    """
    resolved = find_registry_path(path)
    if resolved is None:
        return None
    with open(resolved, encoding="utf-8") as handle:
        try:
            return EngineRegistry.model_validate(json.load(handle))
        except (json.JSONDecodeError, ValueError) as e:
            # Name the FILE: on a machine with several candidate registry
            # locations, a bare validation error leaves the maintainer
            # guessing which one to fix. (pydantic's ValidationError is a
            # ValueError; its per-entry detail rides along in str(e).)
            raise EngineNotAvailableError(f"engine registry {resolved} is invalid: {e}") from e


def registry_engine_names(registry: EngineRegistry | None) -> tuple[str, ...]:
    """Engine names a registry declares (empty when there is no registry)."""
    return () if registry is None else tuple(sorted(registry.engines))


def build_engine(name: str, spec: EngineSpec, **options: Any) -> Any:
    """Build the ASE calculator a registry entry declares.

    Applies the entry's ``env`` to the process environment, imports the
    dotted calculator, and calls it with the entry's defaults merged under
    the caller's options (caller wins key-by-key).

    The env application is deliberately *persistent*: the calculators these
    variables exist for (VASP's ``VASP_PP_PATH``, ``ASE_VASP_COMMAND``) read
    ``os.environ`` at calculate time, after this function has returned, so a
    scoped save-and-restore would silently break them. The consequence —
    declared env is process-wide, and entries needing conflicting values for
    the same variable cannot share a process — is a stated property of the
    registry, and the poisoning is visible, never silent: an overwrite of an
    existing variable with a *different* value warns, and a variable the
    built-in engines' own resolution consults warns even when previously
    unset (setting it redirects the built-ins for the rest of the process).
    Variables ASE reads only at import time are refused at validation — see
    :class:`EngineSpec`. (Probes in :func:`verify_engines` use a
    subprocess-scoped overlay instead, because a probe's env must not leak
    into the maintainer's shell session.)

    Raises:
        EngineNotAvailableError: The dotted path cannot be imported — the
            registry promises something this environment cannot deliver.

    Examples:
        >>> spec = EngineSpec.model_validate({"calculator": "ase.calculators.emt.EMT"})
        >>> type(build_engine("emt-via-registry", spec)).__name__
        'EMT'
    """
    factory = _import_dotted(name, spec.calculator)
    for key, value in spec.env.items():
        previous = os.environ.get(key)
        if key in _BUILTIN_STEERING_ENV and previous != value:
            was = f" (was {previous!r})" if previous is not None else ""
            _warn_always(
                f"engine {name!r} sets ${key}={value!r}{was}, which slab's own "
                f"resolution consults — config discovery, built-in engine "
                f"commands, or scheduler detection in this process may now "
                f"behave differently (explicit config still outranks the "
                f"environment where both exist)"
            )
        elif previous is not None and previous != value:
            _warn_always(
                f"engine {name!r} overwrites ${key} ({previous!r} -> {value!r}); "
                f"declared env is process-wide — engines needing different values "
                f"for the same variable cannot share a process"
            )
        if previous != value:
            original = _APPLIED_ENV.get(key, (previous, value))[0]
            _APPLIED_ENV[key] = (original, value)
        os.environ[key] = value
    merged = {**spec.options, **options}
    return factory(**merged)


def verify_engines(registry: EngineRegistry) -> list[ProbeResult]:
    """Run every engine's declared probe; import-check entries without one.

    This is the maintainer's smoke test (``slab engines verify``): an entry
    with a ``probe`` passes when the command exits 0; an entry without one
    gets its dotted calculator import-checked, which at least proves the
    Python side of the declaration.

    Examples:
        >>> registry = EngineRegistry.model_validate({
        ...     "engines": {"emt-demo": {"calculator": "ase.calculators.emt.EMT"}},
        ... })
        >>> [(r.engine, r.ok) for r in verify_engines(registry)]
        [('emt-demo', True)]
    """
    results: list[ProbeResult] = []
    for engine_name in sorted(registry.engines):
        spec = registry.engines[engine_name]
        if spec.probe is None:
            try:
                _import_dotted(engine_name, spec.calculator)
            except EngineNotAvailableError as e:
                results.append(ProbeResult(engine=engine_name, ok=False, detail=str(e)))
            else:
                results.append(
                    ProbeResult(
                        engine=engine_name,
                        ok=True,
                        detail=f"import ok: {spec.calculator} (no probe declared)",
                    )
                )
            continue
        if Path(spec.probe[0]).name == "srun" and not os.environ.get("SLURM_JOB_ID"):
            # The same trap the built-in engines refuse: srun outside an
            # allocation queues for a fresh one, and a probe that queues is
            # a probe that hangs.
            results.append(
                ProbeResult(
                    engine=engine_name,
                    ok=False,
                    detail=(
                        "probe starts with srun outside a SLURM allocation — it "
                        "would queue instead of answering; probe the bare binary, "
                        "or verify from inside a job"
                    ),
                )
            )
            continue
        env = {**os.environ, **spec.env}
        try:
            # A private cwd and closed stdin, for the same reasons the version
            # probes in slab.backends use them: pw.x writes input_tmp.in debris
            # into cwd, and lmp without arguments blocks reading the terminal.
            with tempfile.TemporaryDirectory(prefix="slab-probe-") as probe_dir:
                completed = subprocess.run(
                    spec.probe,
                    capture_output=True,
                    text=True,
                    timeout=_PROBE_TIMEOUT_S,
                    env=env,
                    check=False,
                    cwd=probe_dir,
                    stdin=subprocess.DEVNULL,
                )
        except (OSError, subprocess.TimeoutExpired) as e:
            results.append(ProbeResult(engine=engine_name, ok=False, detail=str(e)))
            continue
        if completed.returncode == 0:
            results.append(
                ProbeResult(engine=engine_name, ok=True, detail=f"probe ok: {' '.join(spec.probe)}")
            )
        else:
            tail = (completed.stderr or completed.stdout).strip().splitlines()
            results.append(
                ProbeResult(
                    engine=engine_name,
                    ok=False,
                    detail=f"probe exited {completed.returncode}: {tail[-1] if tail else ''}",
                )
            )
    return results


def _import_dotted(engine_name: str, dotted: str) -> Any:
    module_name, _, attribute = dotted.rpartition(".")
    if not module_name:
        raise EngineNotAvailableError(
            f"registry engine {engine_name!r} declares calculator {dotted!r}, "
            f"which is not a dotted module path"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise EngineNotAvailableError(
            f"registry engine {engine_name!r} needs {module_name!r}, which is not "
            f"installed in this environment: {e}"
        ) from e
    try:
        return getattr(module, attribute)
    except AttributeError as e:
        raise EngineNotAvailableError(
            f"registry engine {engine_name!r} declares {dotted!r}, but "
            f"{module_name!r} has no attribute {attribute!r}"
        ) from e
