"""ASE calculator factories — the seam between SLAB and the physics engines.

SLAB never implements physics. Engines are reached through the ASE
``Calculator`` contract; this module maps an engine name to a ready calculator
instance. Three sources feed the mapping, in resolution order:

* **Built-ins** — engines the ``slab`` package can construct on its own:
  ``mace`` (in-process, via the ``slab[mace]`` extra), ``qe`` (Quantum
  ESPRESSO's ``pw.x`` via ASE's file-IO calculator — no extra needed, just
  the executable and pseudopotentials), ``lammps`` (the ``lmp`` binary via
  ASE's ``lammpsrun`` calculator — likewise just the executable plus your
  potential files), ``rootstock`` (an MLIP served from a cluster's pre-built
  rootstock install, via the ``slab[rootstock]`` extra), and ASE's
  ``emt``/``lj`` toys for tests.
* **The cluster engine registry** (:mod:`slab.engines`) — names a cluster
  maintainer declared (``vasp``, curated site aliases like ``qe-delta`` or
  ``lammps-delta``, site-specific MLIP aliases, ...), resolved
  rootstock-style: the client only finds the registry file; the file says
  how each engine is built *here*.
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
* ``lammps`` — the LAMMPS binary through ``ase.calculators.lammpsrun``. The
  command comes from ``command=`` in ``calculator_options`` (or
  ``[engines.lammps]`` in the slab config, or ``$ASE_LAMMPSRUN_COMMAND``,
  defaulting to ``lmp``). The *interatomic potential* is required:
  ``pair_style=`` and ``pair_coeff=`` (plus ``files=`` for potential files)
  must be passed explicitly — ASE's silent fallback is a dimensionless
  ``lj/cut`` toy that would "work" for any material, and which potential to
  use is a science decision. ``files=`` entries are staged into the scratch
  and bare-basename references to them in ``pair_coeff`` resolve to the
  staged copies, so ``files=["/pots/Cu_u3.eam"]`` with
  ``pair_coeff=["1 1 Cu_u3.eam"]`` works from any cwd. Everything else
  (``units=``, ``specorder=``, ``masses=``, ...) is forwarded to the
  ``LAMMPS`` calculator verbatim.
  Unless ``tmp_dir=`` is given, each calculator runs in a slab-managed
  scratch directory that :func:`close_calculator` removes — the scratch is
  what makes failure evidence exist at all: LAMMPS's real error message
  (``ERROR: ...``) surfaces in Python only as a bare
  ``RuntimeError: Failed to retrieve any thermo_style-output`` or an exit
  code, while the story lives in the log file the scratch retains.
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
import re
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
        ('emt', 'lammps', 'lj', 'mace', 'qe', 'rootstock')
    """
    builtin = ("emt", "lammps", "lj", "mace", "qe", "rootstock")
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
    if normalized == "lammps":
        return _lammps_calculator(**options)
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
    if normalized in ("emt", "lj", "rootstock"):
        return {"engine": normalized, "source": "builtin", "version": None}
    if normalized == "mace":
        # Which MLIP actually runs = the resolved checkpoint plus the
        # mace-torch code executing it, so both are cache identity: bumping
        # the package or changing the resolved "small" default must
        # invalidate cached results honestly — the default lives in
        # _mace_calculator, whose source the cache key never hashes.
        return {
            "engine": "mace",
            "source": "builtin",
            "version": _dist_version("mace-torch"),
            "model": str(options.get("model", "small")),
        }
    if normalized == "lammps":
        # The detected LAMMPS version and the resolved command are the cache
        # identity, mirroring qe: upgrading the binary or pointing at a
        # different one honestly invalidates cached results. Potential file
        # *paths* are identity too, resolved: a relative files= entry (or a
        # ~) names different bytes from a different cwd, so the resolved
        # absolute paths are stamped here — the traced options alone carry
        # only the literal relative string. File *contents* are not hashed,
        # exactly like a bare pseudo_dir.
        identity: dict[str, Any] = {
            "engine": "lammps",
            "source": "builtin",
            "version": _lammps_version(options),
            "command": _lammps_locator(options),
        }
        sources = _lammps_file_sources(options)
        if sources is not None:
            identity["files"] = [str(source) for source in sources]
        return identity
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
    expose ``close()``; in-process ones don't; calculators with no ``close``
    of their own get a ``_slab_close`` hook from their factory (``lammps``
    keeps a persistent ``lmp`` subprocess that would otherwise leak).
    File-IO calculators built with a slab-managed scratch directory (``qe``
    without an explicit ``directory=``, ``lammps`` without an explicit
    ``tmp_dir=``) have that scratch removed here — capture any evidence you
    need first (:func:`collect_engine_outputs`,
    :func:`collect_failure_evidence`); :func:`slab.tasks.relax` does both.
    Safe on all calculators, and safe to call twice.

    Examples:
        >>> close_calculator(get_calculator("emt"))  # no-op for in-process engines
    """
    closer = getattr(calculator, "_slab_close", None)
    if callable(closer):
        closer()
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
    in the same directory for every force evaluation, overwriting the file);
    for ``lammps`` the last force evaluation's log (thermo table included) —
    the natural artifact to keep after a successful task. In-process
    calculators (emt, mace, ...) write no files: empty list. Never raises.

    Both engine shapes are duck-typed (``template``/``directory`` for
    ASE's GenericFileIO calculators, ``name == "lammpsrun"`` plus a
    ``tmp_dir`` for LAMMPS), so registry-built calculators of the same
    shapes get the same treatment as the built-ins.

    Examples:
        >>> collect_engine_outputs(get_calculator("emt"))
        []
    """
    try:
        lammps_dir = _lammpsrun_dir(calculator)
        if lammps_dir is not None:
            log = _latest_lammps_file(lammps_dir, "log_")
            return [] if log is None else [("log", log)]
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

    A crashed engine surfaces in Python as a bare
    ``CalledProcessError: ... returned non-zero exit status`` (``pw.x``) or a
    ``RuntimeError: Failed to retrieve any thermo_style-output`` (LAMMPS,
    whose real ``ERROR: ...`` line dies in a reader thread) — the actual
    story is in the files the engine wrote. Returns ``(notes, files)``:

    * *notes* — short strings for ``Exception.add_note``: for ``qe`` the
      parsed ``Error in routine ...`` block(s) from the output file (QE
      fences them in ``%%%%`` lines), falling back to the ``CRASH`` file,
      falling back to the output tail, plus the stderr tail when non-empty;
      for ``lammps`` the ``ERROR`` line(s) from the log with one line of
      preceding context (the echoed command or the last thermo row before
      death), with the same flagged-lines-then-tail fallback.
    * *files* — ``(suffix, path)`` pairs of the engine's input, output,
      stderr, and ``CRASH``/data files that exist and are non-empty, for
      keeping as artifacts before the scratch directory vanishes.

    Both empty for in-process calculators. Best-effort: never raises (it runs
    inside exception handlers).

    Examples:
        >>> collect_failure_evidence(get_calculator("emt"))
        ([], [])
    """
    notes: list[str] = []
    files: list[tuple[str, Path]] = []
    try:
        lammps_dir = _lammpsrun_dir(calculator)
        if lammps_dir is not None:
            return _lammps_failure_evidence(lammps_dir)
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


def _lammps_calculator(**options: Any) -> Any:
    from ase.calculators.lammpsrun import LAMMPS

    command = options.pop("command", None)
    tmp_dir = options.pop("tmp_dir", None)
    # The potential is required, loudly. ASE's lammpsrun silently defaults to
    # pair_style "lj/cut 2.5" with pair_coeff "* * 1 1" — a dimensionless toy
    # that would "run" for any material and return numbers meaning nothing.
    # Which potential describes a system is a science decision, exactly like
    # which pseudopotential (resolve_pseudopotentials refuses ambiguity for
    # the same reason).
    if "pair_style" not in options or "pair_coeff" not in options:
        raise EngineNotAvailableError(
            "engine 'lammps' requires an interatomic potential: pass both "
            "pair_style= and pair_coeff= in calculator_options (plus files= "
            "for potential files), e.g. {'pair_style': 'eam/alloy', "
            "'pair_coeff': ['* * Cu.eam.alloy Cu'], "
            "'files': ['/path/to/Cu.eam.alloy']} — without them ASE would "
            "silently fall back to a dimensionless lj/cut toy potential"
        )
    command = str(command) if command is not None else (_lammps_setting("command") or "lmp")
    _payload_guard(command, "lammps")
    payload = _command_payload(command)
    if payload and shutil.which(payload[0]) is None:
        raise EngineNotAvailableError(
            f"engine 'lammps' resolved command {command!r}, but "
            f"{payload[0]!r} is not on "
            f"PATH — install LAMMPS (or module-load it), pass "
            f"calculator_options={{'command': '/path/to/lmp', ...}}, or set "
            f"command under [engines.lammps] in the slab config"
        )
    _launcher_guard(command, "lammps")
    # No tmp_dir= means a slab-managed scratch, and the scratch is what makes
    # evidence possible: lammpsrun only *retains* the input/log/data files
    # when a tmp_dir is supplied — with its own auto-created directory the log
    # is a consumed pipe and a crash leaves nothing to read.
    scratch: Path | None = None
    if tmp_dir is None:
        scratch = Path(tempfile.mkdtemp(prefix="slab-lammps-"))
        tmp_dir = scratch
    try:
        # Stage against the realpath — lammpsrun realpaths tmp_dir itself, so
        # this keeps one canonical directory across the staged references,
        # the calculator's parameters, and the files it writes.
        options = _stage_lammps_files(options, Path(os.path.realpath(tmp_dir)))
        calculator: Any = LAMMPS(command=command, tmp_dir=str(tmp_dir), **options)
    except Exception:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        raise
    if scratch is not None:
        calculator._slab_scratch = scratch
    # lammpsrun keeps a persistent lmp subprocess (keep_alive=True) but has no
    # close(); its _lmp_end is the idempotent terminator close_calculator
    # needs, exposed through the generic _slab_close hook.
    calculator._slab_close = calculator._lmp_end
    return calculator


def _stage_lammps_files(options: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
    """Make ``files=`` cwd-independent: absolute sources, staged references.

    ASE's documented usage — ``files=["Cu_u3.eam"]`` with
    ``pair_coeff=["1 1 Cu_u3.eam"]`` — is a trap on its own: lammpsrun
    copies the files into its working directory, but the spawned ``lmp``
    inherits the *caller's* cwd, so the bare name in ``pair_coeff`` only
    resolves when the caller happens to sit next to the potential file. A
    traced task must not depend on where it was launched from, so slab makes
    the copy meaningful: each ``files`` entry is absolutized (against the
    caller's cwd, where the user wrote it), and any ``pair_coeff`` token
    that exactly equals a declared file's basename is rewritten to the
    staged copy's path in *tmp_dir*. The declaration in ``files`` is the
    authorization — absolute references and ordinary coefficients are never
    touched. Two entries sharing a basename would silently overwrite each
    other's copy, so that is refused.

    Returns a new options dict; the caller's dict and lists are traced task
    inputs and are never mutated.
    """
    rewritten = dict(options)
    # A single string is one coefficient line, not a character sequence —
    # normalized here unconditionally, because ASE's input writer iterates
    # pair_coeff and would emit one garbage line per character.
    pair_coeff = options.get("pair_coeff")
    if isinstance(pair_coeff, str):
        rewritten["pair_coeff"] = [pair_coeff]
        pair_coeff = rewritten["pair_coeff"]
    sources = _lammps_file_sources(options)
    if sources is None:
        return rewritten
    names = [source.name for source in sources]
    # The staging directory collapses names — and does so case-insensitively
    # on macOS's default filesystem, so the guard is case-insensitive too:
    # refusing a legal pair on Linux beats a silent overwrite on a Mac.
    lowered = [name.lower() for name in names]
    duplicates = sorted({name for name in names if lowered.count(name.lower()) > 1})
    if duplicates:
        raise EngineNotAvailableError(
            f"engine 'lammps': files= entries share basename(s) {duplicates} "
            f"(compared case-insensitively); they would overwrite each other "
            f"in the staging directory — rename the files or merge the potentials"
        )
    staged = {source.name: str(tmp_dir / source.name) for source in sources}
    rewritten["files"] = [str(source) for source in sources]
    if isinstance(pair_coeff, (list, tuple)):
        tokens = {token for line in pair_coeff for token in str(line).split()}
        ambiguous = sorted(name for name in staged if name in tokens and _is_element_symbol(name))
        if ambiguous:
            # 'pair_coeff * * alloy.eam Cu' ends in element names; a staged
            # file named exactly 'Cu' makes the token undecidable — file
            # reference or element? Refused, never guessed.
            raise EngineNotAvailableError(
                f"engine 'lammps': files= entr{'ies' if len(ambiguous) > 1 else 'y'} "
                f"named {ambiguous} appear in pair_coeff, but the name is also a "
                f"chemical element symbol — slab cannot tell a staged-file "
                f"reference from an element token; rename the file (e.g. add an "
                f"extension) or reference it by absolute path"
            )
        rewritten["pair_coeff"] = [
            " ".join(staged.get(token, token) for token in str(line).split())
            for line in pair_coeff
        ]
    return rewritten


def _lammps_file_sources(options: dict[str, Any]) -> list[Path] | None:
    """The ``files=`` entries as absolute paths, or None when unset.

    A single string is one file, not a character sequence. Relative entries
    resolve against the caller's cwd — at *call* time, which is also when
    :func:`describe_engine` stamps them into the cache identity, so the same
    relative string in a different cwd is honestly a different computation.
    """
    files = options.get("files")
    if not files:
        return None
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    sources = [Path(str(entry)).expanduser() for entry in files]
    return [source if source.is_absolute() else Path.cwd() / source for source in sources]


def _is_element_symbol(name: str) -> bool:
    from ase.data import chemical_symbols

    return name in chemical_symbols


def _lammps_locator(options: dict[str, Any]) -> str:
    """The command ``engine="lammps"`` would run.

    Mirrors the calculator's own resolution — explicit option > the slab
    config > ASE's ``$ASE_LAMMPSRUN_COMMAND`` convention > bare ``lmp`` — so
    cache identity and version detection always describe the binary that
    actually runs. An explicit ``command=None`` (a JSON ``null``, an
    ``os.environ.get`` miss) means *absent* here exactly as it does in the
    factory — key presence must not fork the two resolutions. Never raises.
    """
    try:
        command = options.get("command")
        if command is not None:
            return str(command)
        return _lammps_setting("command") or "lmp"
    except Exception:  # pragma: no cover - defensive: hostile option values
        return "lmp"


def _lammps_version(options: dict[str, Any]) -> str | None:
    """Version of the LAMMPS that ``engine="lammps"`` would run, or None.

    Parses the ``Large-scale Atomic/Molecular Massively Parallel Simulator -
    <version>`` banner of ``<command> -h``. Any failure degrades to None:
    this runs inside cache-key construction and must never raise.
    """
    try:
        command = _lammps_locator(options)
        identity = _executable_identity(command)
        if identity is None:
            return None
        return _probe_lammps_version(command, *identity)
    except Exception:  # pragma: no cover - defensive: lru_cache internals
        return None


@functools.lru_cache(maxsize=64)
def _probe_lammps_version(command: str, executable: str, mtime_ns: int) -> str | None:
    """Run ``<command> -h`` once and parse the LAMMPS version banner.

    ``-h`` is required — LAMMPS without arguments blocks reading stdin.
    Runs in a private temp dir under a hard timeout (a command like
    ``srun lmp`` could otherwise block inside cache-key construction).
    Every failure path returns None. ``executable`` and ``mtime_ns`` exist
    to key the memo cache, exactly like the qe probe: one spawn per binary
    identity, and a replaced binary is re-probed.
    """
    del executable, mtime_ns
    if _srun_without_allocation(command):
        return None  # doomed to queue, not to answer — skip the timeout
    try:
        import subprocess

        with tempfile.TemporaryDirectory(prefix="slab-lammps-version-") as probe_dir:
            completed = subprocess.run(
                [*shlex.split(command), "-h"],
                cwd=probe_dir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_VERSION_PROBE_TIMEOUT_S,
                check=False,
            )
        match = _LAMMPS_BANNER.search(completed.stdout)
        return match.group(1).strip() if match else None
    except Exception:
        return None


_LAMMPS_BANNER = re.compile(r"Massively Parallel Simulator\s*-\s*(.+)")


def _lammps_setting(key: str) -> str | None:
    """``[engines.lammps]`` from the slab config, else ASE's own convention.

    The slab config outranks ASE's (it is the file cluster maintainers
    ship); ASE's convention for lammpsrun is the ``ASE_LAMMPSRUN_COMMAND``
    key, honored from the ``[environment]`` section of ASE's config file and
    from the environment — the same order ASE itself uses.
    """
    from slab.config import config_value

    value = config_value(f"engines.lammps.{key}")
    if value is not None:
        return str(value)
    if key != "command":
        return None
    try:
        from ase.config import cfg

        if cfg.parser.has_section("environment"):
            configured = cfg.parser["environment"].get("ASE_LAMMPSRUN_COMMAND")
            if configured:
                return str(configured)
    except Exception:  # pragma: no cover - defensive: malformed ASE config
        pass
    return os.environ.get("ASE_LAMMPSRUN_COMMAND") or None


def _lammpsrun_dir(calculator: Any) -> Path | None:
    """The working directory of a lammpsrun-shaped calculator, or None.

    Duck-typed like the GenericFileIO probe in the collectors: any
    ``ase.calculators.lammpsrun.LAMMPS`` qualifies — built by the ``lammps``
    engine or by a registry entry — so both get the same evidence handling.
    (A registry-built one without ``tmp_dir`` retains no files; the
    collectors then honestly find nothing.)
    """
    if getattr(calculator, "name", None) != "lammpsrun":
        return None
    parameters = getattr(calculator, "parameters", None)
    if not isinstance(parameters, dict):
        return None
    tmp_dir = parameters.get("tmp_dir")
    if not isinstance(tmp_dir, (str, os.PathLike)):
        return None
    path = Path(tmp_dir)
    return path if path.is_dir() else None


def _latest_lammps_file(directory: Path, prefix: str, *, min_size: int = 1) -> Path | None:
    """The newest ``{prefix}*`` file of at least *min_size* bytes, or None.

    lammpsrun names its per-invocation files ``in_``/``log_``/``data_`` plus
    the label, a call counter, and a random suffix; the last force
    evaluation's files are the newest. Modification time decides (the
    counter is embedded mid-name), with the name as a deterministic
    tiebreak. (The name tiebreak trusts lammpsrun's six-digit zero padding;
    a calculator past 999999 force evaluations *and* an exact
    nanosecond-mtime tie would sort wrongly — noted, not defended.) A file
    deleted while this scans is skipped, never an error: the collectors run
    inside exception handlers.
    """
    candidates: list[tuple[int, str, Path]] = []
    for path in directory.glob(prefix + "*"):
        try:
            stat = path.stat()
        except OSError:  # racing deletion between glob and stat
            continue
        if not path.is_file() or stat.st_size < min_size:
            continue
        candidates.append((stat.st_mtime_ns, path.name, path))
    if not candidates:
        return None
    return max(candidates)[2]


def _lammps_failure_evidence(directory: Path) -> tuple[list[str], list[tuple[str, Path]]]:
    """Notes and files for a failed lammpsrun calculator's scratch directory.

    The log is the story: LAMMPS prints ``ERROR: ...`` (with the source
    location) into it, and ``-echo log`` echoes the input commands, so the
    line preceding the first error is either the command that died or the
    last thermo row before the run blew up — both worth one note. The dump
    file is deliberately not kept (binary, and the partial ASE trajectory
    already is).

    The *newest* log decides, even when empty: a log that exists but holds
    nothing means the process died before writing (out of memory, a
    scheduler kill) — that fact is the note, and no older evaluation's
    healthy log is kept in its place, because evidence from the wrong
    evaluation is worse than none. A log that cannot be read costs only its
    notes, never the already-found files; nothing here raises.
    """
    notes: list[str] = []
    files = [
        (suffix, path)
        for prefix, suffix in (("in_", "in"), ("data_", "data"))
        if (path := _latest_lammps_file(directory, prefix)) is not None
    ]
    log = _latest_lammps_file(directory, "log_", min_size=0)
    if log is None:
        return notes, files
    try:
        if log.stat().st_size == 0:
            notes.append(
                f"engine log ({log.name}) is empty — the process died before "
                f"writing anything (killed? out of memory? walltime?)"
            )
            return notes, files
        files.append(("log", log))
        lines = [
            line.strip() for line in log.read_text(errors="replace").splitlines() if line.strip()
        ]
        errors = [line for line in lines if "ERROR" in line]
        if errors:
            notes.append(f"engine error ({log.name}): " + " | ".join(errors[:3]))
            first = lines.index(errors[0])
            if first > 0:
                notes.append(f"engine log context ({log.name}): {lines[first - 1]}")
        else:
            flagged = [
                line
                for line in lines
                if any(marker in line.lower() for marker in _FAILURE_MARKERS)
            ]
            if flagged:
                notes.append(f"engine output flagged ({log.name}): " + " | ".join(flagged[-3:]))
            elif lines:
                notes.append(f"engine output tail ({log.name}): " + " | ".join(lines[-3:]))
    except Exception:  # unreadable or vanished log: the found files still stand
        notes.append(f"engine log ({log.name}) could not be read")
    return notes, files


_MPI_LAUNCHERS = frozenset({"srun", "mpirun", "mpiexec"})
_SHELL_ENV_ASSIGNMENT = re.compile(r"\w+=.*")


def _command_payload(command: str) -> list[str] | None:
    """Argv of *command* with a leading ``env VAR=val ...`` wrapper stripped.

    ``command = "env OMP_NUM_THREADS=1 pw.x"`` is the sanctioned way to give
    one engine its own environment variables: ASE execs engine commands as a
    plain argv (no shell), so ``/usr/bin/env`` applies the assignments to
    that engine's subprocess alone — nothing leaks into this process or any
    other engine. The guards and PATH checks must therefore judge the
    program that actually runs, not the wrapper. Returns None when the
    payload cannot be identified statically (an ``env`` flag like ``-i``, or
    an unparseable line) — callers then skip payload checks rather than
    refuse a command that would work.

    Examples:
        >>> _command_payload("env OMP_NUM_THREADS=1 srun pw.x")
        ['srun', 'pw.x']
        >>> _command_payload("pw.x")
        ['pw.x']
        >>> _command_payload("env -i pw.x") is None
        True
    """
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv or Path(argv[0]).name != "env":
        return argv
    rest = argv[1:]
    while rest and _SHELL_ENV_ASSIGNMENT.fullmatch(rest[0]):
        rest = rest[1:]
    if rest and rest[0].startswith("-"):
        return None  # env flags: finding the payload would mean parsing env's CLI
    return rest


def _payload_guard(command: str, engine: str) -> None:
    """Refuse commands whose payload is a shell idiom or nothing at all.

    ASE spawns engine commands as a plain argv, so ``VAR=val pw.x`` — a
    shell idiom — would exec ``VAR=val`` as the program. Without this check
    the refusal is the generic "'VAR=val' is not on PATH", which reads as a
    missing binary rather than what it is. The fix is spelled out because it
    is one word: the ``env`` wrapper form, which scopes the variables to
    this engine's subprocess alone.
    """
    try:
        argv = shlex.split(command)
    except ValueError:
        return
    if argv and _SHELL_ENV_ASSIGNMENT.fullmatch(argv[0]):
        assignments = [t for t in argv if _SHELL_ENV_ASSIGNMENT.fullmatch(t)]
        payload = [t for t in argv if not _SHELL_ENV_ASSIGNMENT.fullmatch(t)]
        suggested = " ".join(["env", *assignments, *payload])
        raise EngineNotAvailableError(
            f"engine {engine!r}: command {command!r} starts with an environment "
            f"assignment, which only a shell would apply — ASE execs the command "
            f"directly, so {argv[0]!r} would be run as the program. Use the env "
            f"wrapper instead: command = {suggested!r} scopes the variables to "
            f"this engine's subprocess alone"
        )
    resolved = _command_payload(command)
    if resolved is not None and not resolved:
        raise EngineNotAvailableError(
            f"engine {engine!r}: command {command!r} names no program to run"
        )


def _srun_without_allocation(command: str) -> bool:
    """True when *command* runs srun but no SLURM allocation exists.

    ``srun`` outside a job requests a *fresh* allocation, and ASE runs engine
    commands through ``subprocess`` with no timeout — so the calculation
    queues silently instead of failing. Login nodes are exactly where the
    scenario's smoke test runs first. An ``env``-wrapped srun is still srun.
    """
    argv = _command_payload(command)
    if not argv:
        return False
    return Path(argv[0]).name == "srun" and not os.environ.get("SLURM_JOB_ID")


def _launcher_guard(command: str, engine: str) -> None:
    """Refuse a launcher-prefixed command that cannot work here, loudly.

    Two first-contact cluster traps: ``shutil.which`` on ``argv[0]``
    validates the *launcher* (``which srun`` passes on every login node even
    when the engine binary is missing or behind a module load); and ``srun``
    outside an allocation hangs silently (see
    :func:`_srun_without_allocation`). Only the unambiguous two-token form
    (``srun pw.x``) gets the payload PATH check — a flagged launcher line
    (``srun -n 4 pw.x``) would need the launcher's own CLI parsed to find
    the payload. Both checks look through an ``env VAR=val`` wrapper.
    """
    argv = _command_payload(command)
    if not argv or Path(argv[0]).name not in _MPI_LAUNCHERS:
        return
    if len(argv) == 2 and not argv[1].startswith("-") and shutil.which(argv[1]) is None:
        raise EngineNotAvailableError(
            f"engine {engine!r}: command {command!r} launches {argv[1]!r}, "
            f"which is not on PATH — the launcher exists but the engine "
            f"binary does not (is a module load missing on this node?)"
        )
    if _srun_without_allocation(command):
        raise EngineNotAvailableError(
            f"engine {engine!r}: command {command!r} starts with srun, but "
            f"this process is not inside a SLURM allocation — srun would "
            f"queue for a fresh one and the calculation would hang silently. "
            f"Use the bare executable for login-node smoke tests; keep the "
            f"srun form for batch jobs (slab hpc submit)"
        )


def _qe_calculator(**options: Any) -> Any:
    from ase.calculators.calculator import BadConfiguration
    from ase.calculators.espresso import Espresso, EspressoProfile

    # A protocol *name* must never reach ASE: Espresso(**options) accepts
    # unknown keys and the input writer drops them without a warning, so
    # calculator_options={'protocol': 'balanced'} would silently produce an
    # input with no cutoffs and a Γ-only mesh. The factory also cannot
    # expand it itself — expansion needs the structure (k-mesh, per-atom
    # thresholds), which the factory never sees.
    if "protocol" in options:
        raise EngineNotAvailableError(
            f"engine 'qe' has no protocol= option — expand it first with the "
            f"structure in hand: calculator_options=qe_protocol_options("
            f"atoms, protocol={options['protocol']!r}) (from slab.protocols)"
        )
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
        _payload_guard(resolved_command, "qe")
        payload = _command_payload(resolved_command)
        if payload and shutil.which(payload[0]) is None:
            raise EngineNotAvailableError(
                f"engine 'qe' resolved command {resolved_command!r}, but "
                f"{payload[0]!r} is not on PATH — install Quantum ESPRESSO or "
                f"pass calculator_options={{'command': '/path/to/pw.x', ...}}"
            )
        _launcher_guard(resolved_command, "qe")
        # ecutwfc is mandatory pw.x input and ASE does not validate it —
        # without this refusal the job dies only at runtime (on a cluster,
        # after the queue wait) with a namelist read error. pseudo_family
        # resolves files, never cutoffs; only a protocol expansion or an
        # explicit value supplies them. Machine problems (missing binary,
        # unusable launcher) surface first, above.
        if "ecutwfc" not in input_data["system"]:
            raise EngineNotAvailableError(
                "engine 'qe': input_data sets no ecutwfc, which pw.x "
                "requires — expand a named protocol (qe_protocol_options("
                "atoms, protocol=...) supplies family-recommended cutoffs) "
                "or set input_data={'system': {'ecutwfc': ..., 'ecutrho': ...}}"
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
        # An explicit command=None means absent, exactly as the factory
        # treats it — key presence must not fork the two resolutions (a
        # stamped command of the literal string 'None' would cache under an
        # identity no binary matches).
        if options.get("command") is not None:
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


def _executable_identity(command: str) -> tuple[str, int] | None:
    """``(resolved executable, mtime_ns)`` of a command line, or None.

    The identity that keys the memoized version probes: the binary is
    spawned once per identity, not on every task call — and replacing the
    executable (an upgrade, a module swap changing a symlink target) changes
    the mtime and forces a fresh probe, so long-lived processes (the MCP
    server) still see version bumps. The identity is the *payload* binary:
    for ``env``-wrapped commands, keying on ``/usr/bin/env``'s mtime would
    never notice the engine binary changing under it.
    """
    payload = _command_payload(command)
    argv = shlex.split(command) if payload is None else payload
    executable = shutil.which(argv[0]) if argv else None
    if executable is None:
        return None
    try:
        mtime_ns = os.stat(executable).st_mtime_ns
    except OSError:  # pragma: no cover - raced deletion between which and stat
        mtime_ns = -1
    return executable, mtime_ns


def _banner_version(command: str) -> str | None:
    """pw.x's version banner for *command*, memoized on the executable's identity."""
    identity = _executable_identity(command)
    if identity is None:
        return None
    return _probe_banner_version(command, *identity)


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
    if _srun_without_allocation(command):
        return None  # doomed to queue, not to answer — skip the 20s timeout
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
        # On clusters the MLIP normally is not installed in the client env at
        # all — rootstock serves it from a pre-built env, and its checkpoint
        # ids work directly as engine names. Point there when it exists,
        # instead of teaching an agent to pip-install torch on a login node.
        hint = ""
        try:
            import rootstock  # noqa: F401

            hint = (
                "; this machine has rootstock — prefer a served checkpoint id "
                "as the engine name (e.g. engine='mace-mp-0-medium'; "
                "'slab engines list' shows what is served)"
            )
        except ImportError:
            pass
        raise EngineNotAvailableError(
            f"engine 'mace' needs the mace-torch package: pip install 'slab[mace]'{hint}"
        ) from e
    options.setdefault("model", "small")
    options.setdefault("device", "cpu")
    options.setdefault("default_dtype", "float64")
    return _fetch_named_checkpoint(mace_mp, options, engine="mace")


_CHECKPOINT_FETCH_TIMEOUT_S = 60.0


def _fetch_named_checkpoint(factory: Any, options: dict[str, Any], *, engine: str) -> Any:
    """Call an MLIP factory whose named checkpoint may download on first use.

    mace-torch resolves names like ``"small"`` against a local cache and
    otherwise downloads — and on a firewalled compute node the download is
    not a failure the user can read, it is a raw ``URLError`` (or, on paths
    without their own timeout, a silent hang inside the batch job's time
    limit). The bounded default socket timeout is a floor for the paths that
    set none of their own; the translation into instructions — pre-warm,
    point at a file, or serve — is the actual fix. A checkpoint already in
    the cache never opens a socket, so pre-warmed nodes are unaffected; the
    default-timeout window is scoped to the construction, and every other
    slab network call sets its own explicit per-request timeout.
    """
    import socket
    import urllib.error

    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_CHECKPOINT_FETCH_TIMEOUT_S)
    try:
        return factory(**options)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        model = options.get("model", "small")
        raise EngineNotAvailableError(
            f"engine {engine!r} could not fetch checkpoint {model!r}: {e} — "
            f"compute nodes are typically firewalled. Pre-warm the cache from "
            f"a node with internet (python -c \"from mace.calculators import "
            f"mace_mp; mace_mp(model='{model}')\"), point model= at a "
            f"checkpoint file on disk, or use a rootstock-served checkpoint "
            f"id as the engine name"
        ) from e
    finally:
        socket.setdefaulttimeout(previous)


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


def qe_calculator(**options: Any) -> Any:
    """The built-in ``qe`` engine as a dotted-path factory for registry entries.

    A registry entry cannot express ``EspressoProfile`` objects in JSON, and
    pointing ``env`` at an ASE config file does not work — ASE parses that
    file once at import time, before any registry entry runs (the registry
    refuses ``ASE_CONFIG_PATH`` for exactly this reason). This factory takes
    the same JSON-able options as ``engine="qe"`` — ``command=``,
    ``pseudo_dir=``, ``pseudo_family=``, ``input_data=``, ... — with all of
    the built-in engine's guards, so a curated site setup can live under a
    stable alias::

        "qe-delta": {
          "calculator": "slab.backends.qe_calculator",
          "options": {"command": "srun pw.x", "pseudo_dir": "/sw/pseudos"}
        }
    """
    return _qe_calculator(**options)


def lammps_calculator(**options: Any) -> Any:
    """The built-in ``lammps`` engine as a dotted-path factory for registry entries.

    The registry analog of :func:`qe_calculator`: the built-in engine's
    guards (required potential, PATH and launcher checks, evidence-retaining
    scratch) under a maintainer-curated alias, with JSON-able options.
    """
    return _lammps_calculator(**options)
