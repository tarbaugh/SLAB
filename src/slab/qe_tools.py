"""Quantum ESPRESSO's post-processing tools: ``dos.x`` and ``projwfc.x``.

The ``qe`` engine (:mod:`slab.backends`) drives ``pw.x`` through ASE.
``dos.x`` and ``projwfc.x`` read what ``pw.x`` wrote, in the same save
directory, and they are not calculators, so ASE has no place for them.
This module resolves each tool's command from the engine's own pw.x
resolution and runs it in the directory the caller owns.

:func:`qe_tool_command` takes the filled ``pw.x`` command line and swaps
the ``pw.x`` token for its sibling in the same directory. The tool then
follows the same install, the same build (cpu or gpu), and the same
launcher as ``pw.x``, because it is the same install. Flags that only
``pw.x`` takes are dropped from the tail. A caller that needs another
line sets ``dos_command`` or ``projwfc_command`` in
``calculator_options``.

:func:`run_qe_tool` writes the namelist, runs the command, and captures
the standard output. A nonzero exit or a QE error block raises
:class:`slab.errors.QeToolError` with the tail of the output.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slab.errors import EngineNotAvailableError, QeToolError

__all__ = ["QeToolOutcome", "namelist_text", "qe_tool_command", "run_qe_tool"]

#: The tools this module runs. Each is a sibling of ``pw.x`` in the same
#: Quantum ESPRESSO install.
QE_TOOLS = ("dos.x", "projwfc.x")

#: The ``calculator_options`` key that overrides each tool's command.
COMMAND_KEYS = {"dos.x": "dos_command", "projwfc.x": "projwfc_command"}

#: Flags that divide ``pw.x``'s work between processes. ``dos.x`` and
#: ``projwfc.x`` do not take all of them, and a tool that is given one it
#: does not know stops on the flag. Each takes one value, which is
#: dropped with it. The launcher's own flags come before the executable
#: and are kept.
PW_ONLY_FLAGS = frozenset(
    {
        "-nk",
        "-npool",
        "-npools",
        "-nd",
        "-ndiag",
        "-northo",
        "-nb",
        "-nband",
        "-nbgrp",
        "-nt",
        "-ntg",
        "-ntask_groups",
        "-ni",
        "-nimage",
    }
)

#: How long a post-processing tool may run before its process group is
#: killed. ``dos.x`` and ``projwfc.x`` read one save directory and write
#: one table, so an hour is far above any honest run and still well below
#: a queue's own wall clock.
DEFAULT_TIMEOUT_S = 3600.0

#: How many lines of the output a failure message carries.
_TAIL_LINES = 20

#: How long a killed process group has to give up its last output.
_KILL_GRACE_S = 10.0


@dataclass(frozen=True)
class QeToolOutcome:
    """One finished post-processing run: what ran, and what it printed."""

    tool: str
    command: str
    input_path: Path
    output_path: Path
    output: str
    elapsed_s: float


def qe_tool_command(tool: str, options: Mapping[str, Any] | None = None) -> str:
    """The command line that runs *tool*, derived from the engine's ``pw.x``.

    *options* are the engine's ``calculator_options``. A
    ``dos_command`` or ``projwfc_command`` key wins, filled from this
    launch's :func:`slab.resources.envelope` like every other engine
    command. Otherwise the resolved ``pw.x`` line
    (:func:`slab.backends._qe_locator`) has its ``pw.x`` token replaced by
    *tool* in the same directory, and the flags of
    :data:`PW_ONLY_FLAGS` are dropped from the tail.

    Raises:
        ValueError: *tool* is not one of :data:`QE_TOOLS`.
        slab.errors.EngineNotAvailableError: the ``pw.x`` line cannot be
            read, or it names a directory that holds no *tool*.

    Examples:
        >>> qe_tool_command("dos.x", {"command": "mpirun -np 4 pw.x -nk 4"})
        'mpirun -np 4 dos.x'
        >>> qe_tool_command("projwfc.x", {"projwfc_command": "projwfc.x -i"})
        'projwfc.x -i'
    """
    from slab.backends import _qe_locator, _split_env_wrapper
    from slab.resources import envelope, fill

    if tool not in QE_TOOLS:
        raise ValueError(f"qe_tool_command runs {' and '.join(QE_TOOLS)}, not {tool!r}")
    settings = dict(options or {})
    override = settings.get(COMMAND_KEYS[tool])
    if override:
        return str(fill(str(override), envelope(), engine="qe"))
    pw_command, _pseudo_dir = _qe_locator(settings)
    split = _split_env_wrapper(pw_command)
    if split is None:
        raise EngineNotAvailableError(
            f"the pw.x command {pw_command!r} cannot be read as an argument "
            f"list, so {tool} cannot be derived from it; set "
            f"{COMMAND_KEYS[tool]!r} in calculator_options"
        )
    prefix, payload = split
    index = next((i for i, token in enumerate(payload) if Path(token).name == "pw.x"), None)
    if index is None:
        raise EngineNotAvailableError(
            f"the pw.x command {pw_command!r} names no 'pw.x' token, so "
            f"{tool} cannot be derived from it; set "
            f"{COMMAND_KEYS[tool]!r} in calculator_options"
        )
    executable = Path(payload[index])
    sibling = executable.parent / tool if executable.parent != Path("") else Path(tool)
    if executable.parent != Path("") and not sibling.is_file():
        raise EngineNotAvailableError(
            f"the Quantum ESPRESSO install at {str(executable.parent)!r} holds "
            f"pw.x but no {tool} ({sibling} does not exist); build the "
            f"post-processing tools, or set {COMMAND_KEYS[tool]!r} in "
            f"calculator_options"
        )
    argv = [*prefix, *payload[:index], str(sibling), *_tool_tail(payload[index + 1 :])]
    return " ".join(shlex.quote(token) for token in argv)


def namelist_text(name: str, values: Mapping[str, Any]) -> str:
    """One Fortran namelist, as the QE tools read it.

    A string value is quoted, a bool is written ``.true.`` or
    ``.false.``, and None is dropped, so a caller passes a dict with the
    keys it has and leaves the rest to the tool's own defaults.

    Examples:
        >>> print(namelist_text("DOS", {"prefix": "pwscf", "DeltaE": 0.05, "bz_sum": None}))
        &DOS
          prefix = 'pwscf'
          DeltaE = 0.05
        /
        <BLANKLINE>
    """
    lines = [f"&{name}"]
    for key, value in values.items():
        if value is None:
            continue
        lines.append(f"  {key} = {_namelist_value(value)}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def run_qe_tool(
    tool: str,
    namelist: Mapping[str, Any],
    *,
    cwd: str | Path,
    options: Mapping[str, Any] | None = None,
    setup: Sequence[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> QeToolOutcome:
    """Run *tool* on the save directory in *cwd*, and capture what it printed.

    *namelist* holds the keys of ``&DOS`` or ``&PROJWFC``. The input is
    written as ``<stem>.in`` and the standard output, with the standard
    error folded in, as ``<stem>.out``, both in *cwd*, so a caller keeps
    either as an artifact. The tool runs in its own process group, and a
    timeout kills the whole group, so MPI ranks die with the launcher.

    Raises:
        slab.errors.QeToolError: the tool exited nonzero, printed a QE
            error block, or did not finish in *timeout_s*.
        slab.errors.EngineNotAvailableError: the command names an
            executable this machine does not have.
    """
    from slab.backends import _materialize_setup_wrapper, _payload_guard, _qe_setup, _setup_guard

    directory = Path(cwd)
    stem = Path(tool).stem
    section = "DOS" if tool == "dos.x" else "PROJWFC"
    input_path = directory / f"{stem}.in"
    output_path = directory / f"{stem}.out"
    input_path.write_text(namelist_text(section, namelist), encoding="utf-8")
    command = qe_tool_command(tool, options)
    _payload_guard(command, "qe")
    lines = _qe_setup(list(setup) if setup is not None else None)
    wrapper_dir: Path | None = None
    if lines:
        # The install's own dependencies (module loads, exports) scoped to
        # this subprocess, exactly as the engine factory scopes them.
        _setup_guard(command, lines, "qe")
        wrapper_dir, wrapper = _materialize_setup_wrapper("qe", lines, command)
        argv = [str(wrapper)]
    else:
        argv = shlex.split(command)
        if shutil.which(argv[0]) is None and not Path(argv[0]).is_file():
            raise EngineNotAvailableError(
                f"the {tool} command {command!r} starts with {argv[0]!r}, "
                f"which is not on PATH; install the Quantum ESPRESSO "
                f"post-processing tools or set "
                f"{COMMAND_KEYS[tool]!r} in calculator_options"
            )
    argv = [*argv, "-i", input_path.name]
    started = time.monotonic()
    try:
        try:
            process = subprocess.Popen(
                argv,
                cwd=os.fspath(directory),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "LANG": "C", "LC_ALL": "C"},
                text=True,
                start_new_session=True,
            )
        except OSError as e:
            raise QeToolError(f"cannot run {command!r}: {e}", tool=tool) from e
        try:
            captured, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            captured = _partial(e.stdout) + _kill_group(process)
            output_path.write_text(captured, encoding="utf-8")
            raise QeToolError(
                f"{tool} did not finish within {timeout_s:.0f}s; its process "
                f"group was killed (command: {command})",
                tool=tool,
                log=captured,
            ) from e
    finally:
        if wrapper_dir is not None:
            shutil.rmtree(wrapper_dir, ignore_errors=True)
    captured = captured or ""
    output_path.write_text(captured, encoding="utf-8")
    blocks = _error_blocks(captured)
    if process.returncode != 0 or blocks:
        evidence = "; ".join(blocks) if blocks else _tail(captured)
        raise QeToolError(
            f"{tool} failed (exit {process.returncode}): {evidence}",
            tool=tool,
            log=captured,
        )
    return QeToolOutcome(
        tool=tool,
        command=command,
        input_path=input_path,
        output_path=output_path,
        output=captured,
        elapsed_s=time.monotonic() - started,
    )


def _tool_tail(tail: Sequence[str]) -> list[str]:
    """The tokens after ``pw.x``, without the flags only ``pw.x`` takes."""
    kept: list[str] = []
    skip = False
    for token in tail:
        if skip:
            skip = False
            continue
        if token in PW_ONLY_FLAGS:
            skip = True
            continue
        kept.append(token)
    return kept


def _namelist_value(value: Any) -> str:
    """One namelist value, in Fortran's spelling."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _error_blocks(text: str) -> list[str]:
    """The ``%%%%``-fenced error blocks of *text*, one string each."""
    from slab.backends import _error_blocks as blocks

    return blocks(text)


def _tail(text: str) -> str:
    """The last lines of *text*, as one line of evidence."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-_TAIL_LINES:]) if lines else "it printed nothing"


def _partial(captured: Any) -> str:
    """Whatever a timed-out process had already printed."""
    if captured is None:
        return ""
    return captured if isinstance(captured, str) else captured.decode("utf-8", "replace")


def _kill_group(process: subprocess.Popen[str]) -> str:
    """Kill the process group and return whatever else it printed."""
    with suppress(OSError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    try:
        rest, _ = process.communicate(timeout=_KILL_GRACE_S)
    except Exception:  # pragma: no cover - defensive: the group is already dead
        return ""
    return rest or ""
