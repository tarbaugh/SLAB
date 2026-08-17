"""ASE calculator factories — the seam between SLAB and the physics engines.

SLAB never implements physics. Engines are reached through the ASE
``Calculator`` contract; this module maps an engine name to a ready calculator
instance. Three sources feed the mapping, in resolution order:

* **Built-ins** — engines the ``slab`` package can construct on its own:
  ``mace`` (in-process, via the ``slab[mace]`` extra), ``qe`` (Quantum
  ESPRESSO's ``pw.x`` via ASE's file-IO calculator — no extra needed, just
  the executable and pseudopotentials), ``rootstock`` (an MLIP served from a
  cluster's pre-built rootstock install, via the ``slab[rootstock]`` extra),
  and ASE's ``emt``/``lj`` toys for tests.
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
* ``qe`` — Quantum ESPRESSO ``pw.x`` through ``ase.calculators.espresso``.
  Where the code lives comes from ``command=`` + ``pseudo_dir=`` in
  ``calculator_options`` (or a ready ``profile=``, or ASE's own config file);
  pseudopotentials can alternatively come from an installed family —
  ``pseudo_family="SSSP/1.3/PBEsol/efficiency"`` resolves the directory and
  the element->file mapping from :mod:`slab.pseudos` (and named input
  *protocols* expand to full options via
  :func:`slab.protocols.qe_protocol_options`); everything else
  (``pseudopotentials=``, ``input_data=``, ``kpts=``, ...) is forwarded to
  the ``Espresso`` calculator verbatim. Unless ``directory=`` is
  given, each calculation runs in a slab-managed scratch directory that
  :func:`close_calculator` removes — capture evidence first (relax does:
  the last ``espresso.pwo`` is kept as an intermediate artifact on success,
  and input/output/``CRASH`` are kept with the parsed ``Error in routine``
  message on failure).
* ``emt``/``lj`` — ASE built-ins. Milliseconds per step, fit only for the
  elements they parametrize; ideal for tests, not for science.
"""

from __future__ import annotations

import functools
import os
import shlex
import shutil
import tempfile
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
        ('emt', 'lj', 'mace', 'qe', 'rootstock')
    """
    builtin = ("emt", "lj", "mace", "qe", "rootstock")
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
    if normalized == "qe":
        return _qe_calculator(**options)
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
    if normalized == "qe":
        # The detected pw.x version, the resolved command, and the resolved
        # pseudo_dir are all part of the cache key: upgrading the executable,
        # pointing at a different binary, or switching pseudopotential
        # libraries honestly invalidates cached results — including when the
        # location comes from ASE's config file rather than traced options.
        # Undetectable version (missing binary, unparseable banner) degrades
        # to None. For a bare pseudo_dir the directory *path* is the
        # identity (file contents are not hashed); a pseudo_family upgrades
        # this to content-derived identity — the family name plus a digest
        # of its per-element checksums, portable across machines and roots.
        # An unknown family raises here, loudly: a name that cannot resolve
        # must never produce a cache key.
        command, pseudo_dir = _qe_locator(options)
        identity = {
            "engine": "qe",
            "source": "builtin",
            "version": _qe_version(options),
            "command": command,
            "pseudo_dir": pseudo_dir,
        }
        if "pseudo_family" in options:
            from slab.pseudos import family_digest, find_family

            family, _family_path = find_family(str(options["pseudo_family"]))
            # Family identity is name + content digest ONLY — the local
            # install path is deliberately dropped, so the same family bytes
            # hash identically on any machine and under any root.
            identity["pseudo_dir"] = None
            identity["pseudo_family"] = family.name
            identity["pseudo_family_digest"] = family_digest(family)
        return identity
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
    expose ``close()``; in-process ones don't. File-IO calculators built with
    a slab-managed scratch directory (``qe`` without an explicit
    ``directory=``) have that scratch removed here — capture any evidence you
    need first (:func:`collect_engine_outputs`,
    :func:`collect_failure_evidence`); :func:`slab.tasks.relax` does both.
    Safe on all calculators, and safe to call twice.

    Examples:
        >>> close_calculator(get_calculator("emt"))  # no-op for in-process engines
    """
    close = getattr(calculator, "close", None)
    if callable(close):
        close()
    scratch = getattr(calculator, "_slab_scratch", None)
    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)


def resolve_pseudopotentials(atoms_or_symbols: Any, pseudo_dir: str | Path) -> dict[str, str]:
    """Map each element to the single matching pseudopotential file in *pseudo_dir*.

    A file matches element ``X`` when its name is ``X`` followed by ``.``,
    ``_``, or ``-`` (case-insensitive) and ends in ``.upf`` — so
    ``Si.pz-vbc.UPF`` and ``si_pbe_v1.uspp.F.UPF`` both match ``Si``, and
    neither matches ``S``. Exactly one match per element is required: zero or
    several raises an error listing what was found, never a silent guess —
    which pseudopotential to use is a science decision.

    Returns the ``pseudopotentials=`` mapping the ``qe`` engine expects
    (filenames relative to ``pseudo_dir``).

    Args:
        atoms_or_symbols: An ``ase.Atoms`` or an iterable of element symbols.
        pseudo_dir: Directory holding ``.upf`` files.

    Raises:
        EngineNotAvailableError: *pseudo_dir* is not a directory, an element
            has no matching file, or it has several.

    Examples:
        >>> import pathlib, tempfile
        >>> d = pathlib.Path(tempfile.mkdtemp())
        >>> _ = (d / "Si.pz-vbc.UPF").write_text("<UPF/>")
        >>> resolve_pseudopotentials(["Si", "Si"], d)
        {'Si': 'Si.pz-vbc.UPF'}
    """
    get_symbols = getattr(atoms_or_symbols, "get_chemical_symbols", None)
    symbols = list(get_symbols()) if callable(get_symbols) else list(atoms_or_symbols)
    root = Path(pseudo_dir).expanduser()
    if not root.is_dir():
        raise EngineNotAvailableError(f"pseudo_dir {str(root)!r} is not a directory")
    upf_names = sorted(p.name for p in root.iterdir() if p.suffix.lower() == ".upf")
    mapping: dict[str, str] = {}
    for symbol in dict.fromkeys(symbols):
        matches = [
            name
            for name in upf_names
            if name[: len(symbol)].lower() == symbol.lower()
            and len(name) > len(symbol)
            and name[len(symbol)] in "._-"
        ]
        if len(matches) == 1:
            mapping[symbol] = matches[0]
        elif not matches:
            raise EngineNotAvailableError(
                f"no pseudopotential for {symbol!r} in {root}: expected one "
                f"'{symbol}.<...>.upf'-style file, found {len(upf_names)} .upf file(s) total"
            )
        else:
            raise EngineNotAvailableError(
                f"ambiguous pseudopotential for {symbol!r} in {root}: {matches}; "
                f"pass pseudopotentials= explicitly to choose"
            )
    return mapping


def collect_engine_outputs(calculator: Any) -> list[tuple[str, Path]]:
    """Primary output files a file-IO calculator produced, as ``(suffix, path)``.

    For ``qe`` this is the *last* SCF's ``espresso.pwo`` (ASE reruns ``pw.x``
    in the same directory for every force evaluation, overwriting the file) —
    the natural artifact to keep after a successful task. In-process
    calculators (emt, mace, ...) write no files: empty list. Never raises.

    Examples:
        >>> collect_engine_outputs(get_calculator("emt"))
        []
    """
    try:
        template = getattr(calculator, "template", None)
        directory = getattr(calculator, "directory", None)
        name = getattr(template, "outputname", None)
        if not (isinstance(name, str) and name and directory is not None):
            return []
        path = Path(directory) / name
        if not (path.is_file() and path.stat().st_size > 0):
            return []
        return [(_file_suffix(name), path)]
    except Exception:  # pragma: no cover - defensive: hostile calculator attrs
        return []


def collect_failure_evidence(calculator: Any) -> tuple[list[str], list[tuple[str, Path]]]:
    """What a failed file-IO engine left behind: notes plus files worth keeping.

    A crashed ``pw.x`` surfaces in Python as a bare
    ``CalledProcessError: ... returned non-zero exit status`` — the actual
    story is in the files it wrote. Returns ``(notes, files)``:

    * *notes* — short strings for ``Exception.add_note``: the parsed
      ``Error in routine ...`` block(s) from the output file (QE fences them
      in ``%%%%`` lines), falling back to the ``CRASH`` file, falling back to
      the output tail; plus the stderr tail when non-empty.
    * *files* — ``(suffix, path)`` pairs of the engine's input, output,
      stderr, and ``CRASH`` files that exist and are non-empty, for keeping
      as artifacts before the scratch directory vanishes.

    Both empty for in-process calculators. Best-effort: never raises (it runs
    inside exception handlers).

    Examples:
        >>> collect_failure_evidence(get_calculator("emt"))
        ([], [])
    """
    notes: list[str] = []
    files: list[tuple[str, Path]] = []
    try:
        template = getattr(calculator, "template", None)
        directory = getattr(calculator, "directory", None)
        if template is None or directory is None:
            return notes, files
        base = Path(directory)
        named: dict[str, Path] = {}
        for attr in ("inputname", "outputname", "errorname"):
            name = getattr(template, attr, None)
            if isinstance(name, str) and name:
                named[attr] = base / name
        named["crash"] = base / "CRASH"
        files = [
            (_file_suffix(path.name) if attr != "crash" else "crash", path)
            for attr, path in named.items()
            if path.is_file() and path.stat().st_size > 0
        ]

        output = named.get("outputname")
        output_text = ""
        if output is not None and output.is_file():
            output_text = output.read_text(errors="replace")
        blocks = _error_blocks(output_text)
        source = output.name if output is not None else "CRASH"
        crash = named["crash"]
        if not blocks and crash.is_file():
            crash_text = crash.read_text(errors="replace")
            blocks = _error_blocks(crash_text) or [" ".join(crash_text.split())]
            source = "CRASH"
        if blocks:
            notes.extend(f"engine error ({source}): {block}" for block in blocks[:3])
        elif output is not None and output_text.strip():
            # No fenced error block (e.g. "convergence NOT achieved" stops
            # without one) — surface the lines that explain the stop, and
            # only fall back to the raw tail when nothing is flagged.
            flagged = [
                line.strip()
                for line in output_text.splitlines()
                if any(marker in line.lower() for marker in _FAILURE_MARKERS)
            ]
            if flagged:
                notes.append(
                    f"engine output flagged ({output.name}): " + " | ".join(flagged[-3:])
                )
            else:
                tail = [line.strip() for line in output_text.splitlines() if line.strip()][-3:]
                notes.append(f"engine output tail ({output.name}): " + " | ".join(tail))

        stderr = named.get("errorname")
        if stderr is not None and stderr.is_file():
            err_text = stderr.read_text(errors="replace")
            err_lines = [line.strip() for line in err_text.splitlines() if line.strip()]
            if err_lines:
                notes.append(
                    f"engine stderr tail ({stderr.name}): " + " | ".join(err_lines[-3:])
                )
    except Exception:  # pragma: no cover - defensive: hostile calculator attrs
        return notes, files
    return notes, files


# Lines worth surfacing from an engine output that stopped without a fenced
# error block: QE prints these for SCF non-convergence, walltime stops, ...
_FAILURE_MARKERS = ("error", "not achieved", "stopping", "maximum cpu time", "timed out")


def _error_blocks(text: str) -> list[str]:
    """QE-style ``%%%%``-fenced error blocks in *text*, one string per block.

    QE's ``errore`` prints::

         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              Error in routine electrons (1):
              charge is wrong: smearing is needed
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Each block is flattened to one line; duplicates (one copy per MPI rank)
    collapse. An unclosed block (truncated output) still flushes.
    """
    blocks: list[str] = []
    current: list[str] | None = None

    def flush(lines: list[str]) -> None:
        content = " ".join(part for part in (line.strip() for line in lines) if part)
        if content and content not in blocks:
            blocks.append(content)

    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 5 and set(stripped) == {"%"}:
            if current is None:
                current = []
            else:
                flush(current)
                current = None
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        flush(current)
    return blocks


def _file_suffix(name: str) -> str:
    """Artifact-name suffix for an engine file: extension, else the lowercased name."""
    return Path(name).suffix.lstrip(".").lower() or Path(name).name.lower()


def _qe_calculator(**options: Any) -> Any:
    from ase.calculators.calculator import BadConfiguration
    from ase.calculators.espresso import Espresso, EspressoProfile

    profile = options.pop("profile", None)
    command = options.pop("command", None)
    pseudo_dir = options.pop("pseudo_dir", None)
    pseudo_family = options.pop("pseudo_family", None)
    directory = options.pop("directory", None)
    if profile is not None and (command is not None or pseudo_dir is not None):
        raise EngineNotAvailableError(
            "engine 'qe': pass either profile= or command=/pseudo_dir=, not both"
        )
    if pseudo_family is not None:
        if pseudo_dir is not None or profile is not None:
            raise EngineNotAvailableError(
                "engine 'qe': pass either pseudo_family= or pseudo_dir=/profile=, not both"
            )
        from slab.pseudos import family_pseudos, find_family

        family, family_path = find_family(str(pseudo_family))
        pseudo_dir = family_path
        # The family knows its files: default the element->filename mapping
        # (write_espresso_in only consults the species actually present).
        options.setdefault("pseudopotentials", family_pseudos(family))
    if profile is None and pseudo_dir is None:
        # Machine defaults: the slab config's [engines.qe], then ASE's own
        # [espresso] section — one slab.toml can configure a whole cluster.
        pseudo_dir = _qe_setting("pseudo_dir")
    if profile is None and (command is not None or pseudo_dir is not None):
        if pseudo_dir is None:
            raise EngineNotAvailableError(
                "engine 'qe': a command alone is not enough — also set pseudo_dir "
                "(the directory holding your .upf pseudopotential files), as an "
                "option or under [engines.qe] in the slab config"
            )
        # Same fallback chain _qe_locator uses, so the stamped identity is
        # always the binary that actually ran.
        command = command or _qe_setting("command") or "pw.x"
        profile = EspressoProfile(command=command, pseudo_dir=str(Path(pseudo_dir).expanduser()))

    # Force printing defaults on: pw.x omits forces unless tprnfor is set,
    # and slab's tasks drive optimizers with them. Output verbosity, not
    # physics — and an explicit tprnfor in input_data still wins. Deep-copied
    # first: the caller's dict is a traced task input and must not grow keys
    # behind the tracer's back (a reused options dict would spuriously miss
    # the cache on its second use).
    from copy import deepcopy

    from ase.io.espresso_namelist.namelist import Namelist

    input_data = Namelist(deepcopy(options.get("input_data")))
    input_data.to_nested("pw")
    input_data["control"].setdefault("tprnfor", True)
    options["input_data"] = input_data

    # No directory= means a slab-managed scratch: calculations must not
    # pollute the caller's cwd with espresso.* files. close_calculator
    # removes it; relax captures evidence (kept artifacts, notes) first.
    scratch: Path | None = None
    if directory is None:
        scratch = Path(tempfile.mkdtemp(prefix="slab-qe-"))
        directory = scratch
    try:
        calculator: Any = Espresso(profile=profile, directory=directory, **options)
        resolved_command = str(calculator.profile.command)
        executable = shlex.split(resolved_command)[0]
        if shutil.which(executable) is None:
            raise EngineNotAvailableError(
                f"engine 'qe' resolved command {resolved_command!r}, but "
                f"{executable!r} is not on PATH — install Quantum ESPRESSO or "
                f"pass calculator_options={{'command': '/path/to/pw.x', ...}}"
            )
    except BadConfiguration as e:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        raise EngineNotAvailableError(
            "engine 'qe' is not configured: pass calculator_options="
            "{'command': 'pw.x', 'pseudo_dir': '/path/to/pseudos', ...}, set "
            "command/pseudo_dir under [engines.qe] in the slab config "
            "('slab config init' writes a template), or add an [espresso] "
            "section to your ASE config file (~/.config/ase/config.ini or "
            "$ASE_CONFIG_PATH)"
        ) from e
    except Exception:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        raise
    if scratch is not None:
        calculator._slab_scratch = scratch
    return calculator


def _qe_locator(options: dict[str, Any]) -> tuple[str, str | None]:
    """``(command, pseudo_dir)`` that ``engine="qe"`` would resolve to.

    Mirrors the calculator's own resolution — ``profile=`` > explicit
    options > the slab config > the ASE config file > bare ``pw.x`` — so
    cache identity and version detection always describe the binary and
    pseudopotential directory that actually run. Never raises.
    """
    try:
        profile = options.get("profile")
        if profile is not None:
            pseudo = getattr(profile, "pseudo_dir", None)
            return str(profile.command), None if pseudo is None else str(pseudo)
        if "command" in options:
            command = str(options["command"])
        else:
            command = _qe_setting("command") or "pw.x"
        pseudo_dir = options.get("pseudo_dir")
        if pseudo_dir is None:
            pseudo_dir = _qe_setting("pseudo_dir")
        return command, None if pseudo_dir is None else str(Path(pseudo_dir).expanduser())
    except Exception:  # pragma: no cover - defensive: hostile profile attrs
        return "pw.x", None


def _qe_version(options: dict[str, Any]) -> str | None:
    """Version of the pw.x that ``engine="qe"`` would run, or None.

    Parses the banner ``pw.x`` prints on startup (QE has no ``--version``
    flag). Any failure degrades to None: this runs inside cache-key
    construction and must never raise.
    """
    try:
        command, _pseudo_dir = _qe_locator(options)
        return _banner_version(command)
    except Exception:  # pragma: no cover - defensive: lru_cache internals
        return None


def _banner_version(command: str) -> str | None:
    """Version banner of *command*, memoized on the executable's identity.

    The probe result is cached per ``(command, resolved executable, mtime)``:
    the binary is spawned once, not on every task call — and replacing the
    executable (an upgrade, a module swap changing a symlink target) changes
    the mtime and forces a fresh probe, so long-lived processes (the MCP
    server) still see version bumps.
    """
    argv = shlex.split(command)
    executable = shutil.which(argv[0]) if argv else None
    if executable is None:
        return None
    try:
        mtime_ns = os.stat(executable).st_mtime_ns
    except OSError:  # pragma: no cover - raced deletion between which and stat
        mtime_ns = -1
    return _probe_banner_version(command, executable, mtime_ns)


@functools.lru_cache(maxsize=64)
def _probe_banner_version(command: str, executable: str, mtime_ns: int) -> str | None:
    """Run *command* once and parse QE's ``Program PWSCF v.X`` banner.

    Runs in a private temp dir with stdin closed — ``pw.x`` without input
    prints its banner, writes ``input_tmp.in`` debris in cwd, and exits —
    and under a hard timeout, because a command like ``srun pw.x`` could
    otherwise block inside cache-key construction waiting for an allocation.
    Every failure path returns None. ``executable`` and ``mtime_ns`` exist
    to key the memo cache.
    """
    del executable, mtime_ns
    try:
        import subprocess

        from ase.calculators.espresso import EspressoProfile

        with tempfile.TemporaryDirectory(prefix="slab-qe-version-") as probe_dir:
            completed = subprocess.run(
                shlex.split(command),
                cwd=probe_dir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_VERSION_PROBE_TIMEOUT_S,
                check=False,
            )
        return str(EspressoProfile.parse_version(completed.stdout))
    except Exception:
        return None


_VERSION_PROBE_TIMEOUT_S = 20


def _qe_configured(key: str) -> str | None:
    """A value from the [espresso] section of ASE's own config file, if set."""
    try:
        from ase.config import cfg

        if "espresso" in cfg.parser:
            value = cfg.parser["espresso"].get(key)
            return None if value is None else str(value)
    except Exception:  # pragma: no cover - defensive: malformed ASE config
        return None
    return None


def _qe_setting(key: str) -> str | None:
    """``[engines.qe]`` from the slab config, else ASE's ``[espresso]`` section.

    The slab config outranks ASE's: it is the file cluster maintainers ship.
    A *broken* slab config file raises (loud beats quiet); an absent one
    falls through.
    """
    from slab.config import config_value

    value = config_value(f"engines.qe.{key}")
    if value is not None:
        return str(value)
    return _qe_configured(key)


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
