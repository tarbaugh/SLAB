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

    Examples:
        >>> lammps_command("/opt/lammps/bin/lmp")
        '/opt/lammps/bin/lmp'
    """
    from slab.backends import _lammps_locator

    return _lammps_locator({"command": command})


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
