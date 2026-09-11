"""LAMMPS as a script runner: one input file, run whole, log and files back.

The ``lammps`` engine (:mod:`slab.backends`) drives ``lmp`` through ASE
for forces, one ``run 0`` per call. This module runs a LAMMPS input
script the way LAMMPS is meant to run: ``lmp -in in.lammps -log
log.lammps`` in a slab-managed scratch directory, with the dynamics
integrated inside LAMMPS and the output written by the script's own
``thermo``, ``dump``, ``write_data``, and ``fix`` commands. It talks to
the binary and reads what it wrote, and it knows nothing about runs;
``foundation.tasks.run_lammps`` is the traced task on top.

The binary, the setup lines, and the version probe are the engine's own
(``[engines.lammps]`` in the config, ``$ASE_LAMMPSRUN_COMMAND``, bare
``lmp``), so a script and a force call use the same LAMMPS, and a KOKKOS
or MPI launch rides in the command exactly as it does for the engine.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slab.errors import EngineNotAvailableError, LammpsScriptError

#: The names the runner fixes inside the scratch directory.
INPUT_NAME = "in.lammps"
LOG_NAME = "log.lammps"
SCREEN_NAME = "screen.lammps"

_EVIDENCE_LIMIT = 30
_ERROR_LINES_SHOWN = 3


@dataclass(frozen=True)
class LammpsOutcome:
    """One finished LAMMPS script: what ran, the log, and the screen capture."""

    command: str
    argv: tuple[str, ...]
    log: str
    screen: str


def lammps_command(command: str | None = None) -> str:
    """The command that runs LAMMPS: per-call, else the engine's own resolution.

    The ``{ntasks}``, ``{threads}``, and ``{gpus}`` placeholders are
    filled from this launch's :func:`slab.resources.envelope`.

    Examples:
        >>> lammps_command("/opt/lammps/bin/lmp")
        '/opt/lammps/bin/lmp'
        >>> import os
        >>> os.environ.update(SLAB_CPUS="0,1,2,3", SLAB_GPUS="0,1")
        >>> os.environ.update(SLAB_NTASKS="2", SLAB_THREADS="2")
        >>> lammps_command("mpirun -np {ntasks} lmp -k on g {gpus} t {threads} -sf kk")
        'mpirun -np 2 lmp -k on g 2 t 2 -sf kk'
        >>> for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS"):
        ...     del os.environ[name]
    """
    from slab.backends import _lammps_locator

    return _lammps_locator({"command": command})


def lammps_template(command: str | None = None) -> str:
    """The LAMMPS command as written, placeholders unfilled: what a listing shows.

    Examples:
        >>> lammps_template("mpirun -np {ntasks} lmp")
        'mpirun -np {ntasks} lmp'
    """
    from slab.backends import _lammps_template

    return _lammps_template({"command": command})


def lammps_setup(setup: str | tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    """Setup lines for the LAMMPS subprocess: per-call, else ``[engines.lammps]``."""
    from slab.backends import _engine_setup

    return _engine_setup(setup, "lammps")


def describe_lammps(
    command: str | None = None,
    setup: str | tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Identity of the LAMMPS a script would run under: provenance and cache identity.

    The resolved command, the detected version, and the setup lines, from
    the engine's own probe, so a script and a force call agree on which
    LAMMPS they name.

    Examples:
        >>> describe_lammps("definitely-not-installed-lmp")["engine"]
        'lammps'
    """
    from slab.backends import describe_engine

    options: dict[str, Any] = {"command": command}
    if setup is not None:
        options["setup"] = setup
    described = describe_engine("lammps", options)
    return {key: value for key, value in described.items() if key != "source"}


LAMMPS_FACTORY = "slab.backends.lammps_calculator"


def lammps_route(engine: str | None = None) -> dict[str, Any]:
    """The LAMMPS a route name stands for: its command and setup lines.

    A machine keeps more than one LAMMPS: a plain build for smoke tests
    and small cells, a KOKKOS build for the GPU partition. Each is a
    route with a name. ``lammps`` is the built-in route, and its command
    and setup come from ``[engines.lammps]``. Every other route is a
    registry alias whose calculator is ``slab.backends.lammps_calculator``,
    and its command and setup come from the alias's ``options``. A name
    that is neither is refused with the routes that exist, so a script
    never runs under a binary nobody named. A route's ``command`` or
    ``setup`` is None where the route leaves it to the engine's own
    resolution.

    Examples:
        >>> import os
        >>> os.environ.pop("SLAB_ENGINES", None) and None
        >>> lammps_route()["engine"]
        'lammps'
        >>> lammps_route("lammps")["source"]
        'builtin'
    """
    from slab.engines import load_registry

    name = (engine or "lammps").strip()
    if name.lower() == "lammps":
        return {"engine": "lammps", "source": "builtin", "command": None, "setup": None}
    registry = load_registry()
    spec = registry.engines.get(name) if registry is not None else None
    if spec is None or spec.calculator != LAMMPS_FACTORY:
        known = ", ".join(lammps_routes())
        what = "is not a LAMMPS route" if spec is not None else "names no engine here"
        raise EngineNotAvailableError(
            f"engine {name!r} {what}; the LAMMPS routes on this machine are: {known}. "
            f"A route is the built-in 'lammps' or a registry alias with calculator "
            f"{LAMMPS_FACTORY!r}"
        )
    cluster = registry.cluster if registry is not None else None
    source = f"registry:{cluster}" if cluster else "registry"
    return {
        "engine": name,
        "source": source,
        "command": spec.options.get("command"),
        "setup": spec.options.get("setup"),
    }


def lammps_routes() -> dict[str, dict[str, Any]]:
    """Every LAMMPS route on this machine, resolved: command, setup, KOKKOS switches.

    The built-in ``lammps`` first, then each registry alias that runs the
    LAMMPS factory, in name order. ``command`` is the route's command as
    written and ``placeholders`` the ``{ntasks}``, ``{threads}``, and
    ``{gpus}`` it asks for, which a launch fills. The switches are parsed
    from the command filled for this process's envelope, or from the
    template when it cannot be filled here, because SLAB adds none. No
    binary is probed.

    Examples:
        >>> import os
        >>> os.environ.pop("SLAB_ENGINES", None) and None
        >>> list(lammps_routes())
        ['lammps']
        >>> lammps_routes()["lammps"]["placeholders"]
        []
    """
    from slab.engines import load_registry
    from slab.resources import envelope, fill, placeholders

    routes: dict[str, dict[str, Any]] = {}
    names = ["lammps"]
    registry = load_registry()
    if registry is not None:
        names += sorted(
            name for name, spec in registry.engines.items() if spec.calculator == LAMMPS_FACTORY
        )
    for name in names:
        route = lammps_route(name)
        template = lammps_template(route["command"])
        try:
            filled = fill(template, envelope(), route=name)
        except EngineNotAvailableError:
            filled = template
        routes[name] = {
            "source": route["source"],
            "command": template,
            "placeholders": placeholders(template),
            "setup": list(lammps_setup(route["setup"])),
            "kokkos": kokkos_switches(filled),
        }
    return routes


def script_scratch_dir() -> Path:
    """A fresh slab-managed scratch directory for one script (``[paths] scratch``)."""
    from slab.backends import _scratch_dir

    return _scratch_dir("slab-lammps-script-")


def run_lammps_script(
    *,
    cwd: str | os.PathLike[str],
    command: str | None = None,
    setup: str | tuple[str, ...] | list[str] | None = None,
    timeout_s: float = 86400.0,
) -> LammpsOutcome:
    """Run ``in.lammps`` in *cwd* whole and classify the outcome.

    The caller owns staging: the script and every file it names are
    already in *cwd* under their basenames. LAMMPS writes its log to
    ``log.lammps`` there, and this function writes the captured screen
    output (stdout and stderr) to ``screen.lammps``, so both survive for
    the caller to keep. The subprocess runs in its own process group; on
    timeout the whole group is killed, so MPI ranks die with the launcher.
    Success returns the log and the screen text; failure (a nonzero exit,
    or an ``ERROR`` line in the log or on the screen) raises
    :class:`~slab.errors.LammpsScriptError` carrying both.
    """
    from slab.backends import _launcher_guard, _payload_guard, _setup_guard

    resolved = lammps_command(command)
    lines = lammps_setup(setup)
    _payload_guard(resolved, "lammps")
    if lines:
        _setup_guard(resolved, lines, "lammps")
    else:
        _require_available(resolved)
        _launcher_guard(resolved, "lammps")
    run_argv = _run_argv(resolved, lines)
    directory = Path(cwd)
    env = {**os.environ, "LANG": "C", "LC_ALL": "C"}
    try:
        process = subprocess.Popen(
            run_argv,
            cwd=os.fspath(directory),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            start_new_session=True,
        )
    except OSError as e:
        raise LammpsScriptError(f"cannot run {resolved!r}: {e}") from e
    try:
        screen, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        partial = _partial_output(e.stdout) or _kill_process_group(process)
        _write_screen(directory, partial)
        raise LammpsScriptError(
            f"LAMMPS did not finish within {timeout_s:.0f}s; its process group "
            f"was killed (command: {resolved})",
            log=_read_log(directory),
            screen=partial,
        ) from e
    screen = screen or ""
    _write_screen(directory, screen)
    log = _read_log(directory)
    if process.returncode != 0 or _has_error(log) or _has_error(screen):
        evidence = "\n  ".join(error_lines(log if log.strip() else screen))
        raise LammpsScriptError(
            f"LAMMPS failed (exit {process.returncode}):\n  {evidence}", log=log, screen=screen
        )
    return LammpsOutcome(command=resolved, argv=tuple(run_argv), log=log, screen=screen)


_KOKKOS_MODE = re.compile(r"^KOKKOS mode\b.*\bis enabled", re.MULTILINE)
_KOKKOS_GPUS = re.compile(r"will use up to (\d+) GPU\(s\) per node")
_KOKKOS_THREADS = re.compile(r"using (\d+) OpenMP thread\(s\) per MPI task")
_KOKKOS_STYLE = re.compile(r"^\s*\(\d+\)\s+(?:pair|fix|compute)\s+(\S+/kk\S*)", re.MULTILINE)


def kokkos_switches(command: str) -> dict[str, Any]:
    """What a LAMMPS command line asks of the KOKKOS package.

    SLAB never adds a switch: the command runs as written, plus ``-in``
    and ``-log``. So this is the whole answer to "does this run use
    KOKKOS": ``enabled`` is ``-k on`` (or ``-kokkos on``), ``gpus`` and
    ``threads`` are its ``g N`` and ``t N`` arguments, ``suffix`` is
    ``-sf kk`` (or ``-suffix kk``), and ``package`` is the text after
    ``-pk kokkos`` (or ``-package kokkos``), or None. A command that
    cannot be parsed reports nothing enabled.

    Examples:
        >>> kokkos_switches("mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"
        ...                 )  # doctest: +NORMALIZE_WHITESPACE
        {'enabled': True, 'gpus': 1, 'threads': None, 'suffix': True,
         'package': 'newton on neigh half'}
        >>> kokkos_switches("lmp -k on t 8 -sf kk")
        {'enabled': True, 'gpus': None, 'threads': 8, 'suffix': True, 'package': None}
        >>> kokkos_switches("lmp")
        {'enabled': False, 'gpus': None, 'threads': None, 'suffix': False, 'package': None}
    """
    switches: dict[str, Any] = {
        "enabled": False,
        "gpus": None,
        "threads": None,
        "suffix": False,
        "package": None,
    }
    try:
        tokens = shlex.split(command)
    except ValueError:
        return switches
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("-k", "-kokkos") and index + 1 < len(tokens):
            switches["enabled"] = tokens[index + 1] == "on"
            index += 2
            while index + 1 < len(tokens) and tokens[index] in ("g", "t"):
                try:
                    count = int(tokens[index + 1])
                except ValueError:
                    break
                switches["gpus" if tokens[index] == "g" else "threads"] = count
                index += 2
            continue
        if token in ("-sf", "-suffix") and index + 1 < len(tokens):
            if tokens[index + 1] == "kk":
                switches["suffix"] = True
            index += 2
            continue
        is_package = token in ("-pk", "-package") and index + 1 < len(tokens)
        if is_package and tokens[index + 1] == "kokkos":
            index += 2
            options: list[str] = []
            while index < len(tokens) and not tokens[index].startswith("-"):
                options.append(tokens[index])
                index += 1
            switches["package"] = " ".join(options) or None
            continue
        index += 1
    return switches


def kokkos_report(log: str) -> dict[str, Any]:
    r"""What the log says KOKKOS did: the facts a GPU run has to show.

    A KOKKOS run prints ``KOKKOS mode ... is enabled`` at startup, then
    the GPUs per node and the OpenMP threads per MPI task it will use,
    and the neighbor list info names every style that ran its ``/kk``
    version. A run without those lines ran the plain styles on the host,
    whatever the build contains.

    Examples:
        >>> log = ("LAMMPS (22 Jul 2025 - Update 4)\n"
        ...        "KOKKOS mode with Kokkos version 4.6.1 is enabled (src/KOKKOS/kokkos.cpp:72)\n"
        ...        "  will use up to 1 GPU(s) per node\n"
        ...        "  using 1 OpenMP thread(s) per MPI task\n"
        ...        "Neighbor list info ...\n"
        ...        "  (1) pair eam/alloy/kk, perpetual\n")
        >>> kokkos_report(log)
        {'enabled': True, 'gpus': 1, 'threads': 1, 'styles': ['eam/alloy/kk']}
        >>> kokkos_report("LAMMPS (22 Jul 2025)\nTotal wall time: 0:00:01\n")
        {'enabled': False, 'gpus': None, 'threads': None, 'styles': []}
    """
    gpus = _KOKKOS_GPUS.search(log)
    threads = _KOKKOS_THREADS.search(log)
    styles = sorted({match.group(1).rstrip(",") for match in _KOKKOS_STYLE.finditer(log)})
    return {
        "enabled": _KOKKOS_MODE.search(log) is not None,
        "gpus": int(gpus.group(1)) if gpus else None,
        "threads": int(threads.group(1)) if threads else None,
        "styles": styles,
    }


def error_lines(log: str, limit: int = _EVIDENCE_LIMIT) -> list[str]:
    """The lines of a LAMMPS log worth reading after a failure.

    The ``ERROR`` lines are the evidence, with the line before the first
    one: the echoed command that died, or the last thermo row before the
    blow-up. A log without one contributes its last non-empty lines, and
    an empty log says so, because a process that wrote nothing died before
    LAMMPS started (a missing library, a wrong binary, a scheduler kill).

    Examples:
        >>> error_lines("pair_style eam/aloy\\nERROR: Unrecognized pair style 'eam/aloy'")
        ['context: pair_style eam/aloy', "ERROR: Unrecognized pair style 'eam/aloy'"]
        >>> error_lines("")
        ['LAMMPS wrote nothing: the process died before the script started']
    """
    stripped = [line.strip() for line in log.splitlines() if line.strip()]
    if not stripped:
        return ["LAMMPS wrote nothing: the process died before the script started"]
    errors = [index for index, line in enumerate(stripped) if _is_error_line(line)]
    if errors:
        first = errors[0]
        out = [f"context: {stripped[first - 1]}"] if first > 0 else []
        out.extend(stripped[index] for index in errors[:_ERROR_LINES_SHOWN])
        return out[:limit]
    return stripped[-5:][:limit]


def _is_error_line(line: str) -> bool:
    return line.startswith("ERROR") or line.startswith("Last command:")


def _has_error(text: str) -> bool:
    return any(line.strip().startswith("ERROR") for line in text.splitlines())


def _run_argv(command: str, setup: tuple[str, ...]) -> list[str]:
    """The argv to execute: the command directly, or through a setup shell.

    With setup lines the invocation becomes a fail-fast login shell, so
    the ``module`` shell function exists and a failing load kills the run
    instead of exec'ing into the wrong environment, exactly the engine
    wrapper's semantics.
    """
    try:
        payload = shlex.split(command)
    except ValueError as e:
        raise LammpsScriptError(f"cannot parse the LAMMPS command {command!r}: {e}") from e
    if not payload:
        raise LammpsScriptError("the LAMMPS command is empty")
    args = [*payload, "-in", INPUT_NAME, "-log", LOG_NAME]
    if not setup:
        return args
    quoted = " ".join(shlex.quote(token) for token in args)
    script = "\n".join(["set -e", *setup, f"exec {quoted}"])
    return ["/bin/bash", "-l", "-c", script]


def _require_available(command: str) -> None:
    from slab.backends import _command_payload, _which_payload

    payload = _command_payload(command)
    if payload and _which_payload(payload[0], command) is None:
        raise EngineNotAvailableError(
            f"the LAMMPS command {command!r} names {payload[0]!r}, which is not on "
            "PATH. Install LAMMPS (or module-load it), pass command='/path/to/lmp', "
            "or set command under [engines.lammps] in the slab config"
        )


def _write_screen(directory: Path, text: str) -> None:
    with contextlib.suppress(OSError):
        (directory / SCREEN_NAME).write_text(text, encoding="utf-8")


def _read_log(directory: Path) -> str:
    try:
        return (directory / LOG_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _kill_process_group(process: subprocess.Popen[str]) -> str:
    """Kill the subprocess's whole group, then reap. Never raises."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    with contextlib.suppress(Exception):
        remainder, _ = process.communicate(timeout=10)
        return _partial_output(remainder)
    return ""


def _partial_output(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw if isinstance(raw, str) else ""
