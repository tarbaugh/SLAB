"""Gracemaker, the potential trainer: fit GRACE MLIPs, no calculations.

Gracemaker (pip package ``tensorpotential``; Bochkarev, Lysogorskiy and
Drautz, https://gracemaker.readthedocs.io) trains GRACE machine-learned
interatomic potentials and fine-tunes GRACE foundation models. It produces
a potential, never energies or forces for a traced task, so it is **not an
engine**: ``engine="gracemaker"`` does not exist, and nothing here builds a
calculator. SLAB names this kind of tool a *builder*, configured under
``[builders.gracemaker]`` in the config file::

    [builders.gracemaker]
    command = "gracemaker"
    setup = [
      "module load cuda",
      "source ~/grace/bin/activate",
      "export TF_FORCE_GPU_ALLOW_GROWTH=true",
    ]

Gracemaker lives in its own python environment (TensorFlow, usually python
3.11), reached through the ``setup`` lines. Never install tensorpotential
into SLAB's environment. This module is the subprocess seam only: resolve
the command, probe the version, run one invocation in a caller-chosen
directory, and classify the outcome. The traced task that stages the
input file and the dataset in and reads the trained model out is
``foundation.tasks.train_potential``.

Version detection has no ``gracemaker --version`` to call. The probe asks
the python environment that owns the console script for the installed
``tensorpotential`` distribution version instead: through the setup shell
when setup lines exist, else through the interpreter that sits beside the
resolved ``gracemaker`` script (a console script's environment is its
parent directory; a bare ``python`` from PATH would probe the wrong
environment). An unprobeable version degrades to the executable
fingerprint, the same honest fallback the engines use.

Failure classification reads the exit code and the log. Gracemaker is a
python CLI: real failures exit nonzero and print a traceback, and the
traceback tail is the evidence worth reading. A fit that converged badly
is not a failure here — it is a metrics question, answered by the metrics
the task reports and judged by a ``@check``.

Every subprocess runs with ``LANG=C``/``LC_ALL=C``, with stdin closed, in
its own process group under a hard timeout: TensorFlow training spawns
workers, and killing only the direct child on timeout would orphan GPU
processes on a shared node.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slab.errors import BuilderError, BuilderNotAvailableError

#: The line python prints first when an exception escapes; its presence in a
#: gracemaker log marks failure even under an exit code of 0.
TRACEBACK_MARKER = "Traceback (most recent call last"

_VERSION_PROBE = (
    "from importlib.metadata import version; print(version('tensorpotential'))"
)
_VERSION_PATTERN = re.compile(r"^[0-9][\w.+-]*$")
_VERSION_PROBE_TIMEOUT_S = 60
_EVIDENCE_LIMIT = 30


@dataclass(frozen=True)
class GracemakerOutcome:
    """One successful gracemaker invocation: what ran, and everything it printed."""

    command: str
    args: tuple[str, ...]
    log: str


def gracemaker_command(command: str | None = None) -> str:
    """The command that runs gracemaker: per-call, else config, else ``gracemaker``.

    Examples:
        >>> gracemaker_command("/opt/grace/bin/gracemaker")
        '/opt/grace/bin/gracemaker'
    """
    if command is not None:
        return command
    configured = _gracemaker_setting("command")
    return str(configured) if configured else "gracemaker"


def gracemaker_setup(
    setup: str | tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Scoped setup lines for gracemaker: per-call, else ``[builders.gracemaker]``.

    Setup lines are shell for a private login-shell wrapper around the
    gracemaker subprocess only (module loads, environment activation,
    exports) — the same per-tool rule as ``[engines.qe] setup``, never
    job-wide.
    """
    if setup is not None:
        lines = [setup] if isinstance(setup, str) else list(setup)
        return tuple(str(line) for line in lines)
    configured = _gracemaker_setting("setup")
    if not configured:
        return ()
    return tuple(str(line) for line in configured)


def gracemaker_version(
    command: str | None = None, setup: str | tuple[str, ...] | list[str] | None = None
) -> str | None:
    """The installed ``tensorpotential`` version, or None. Never raises.

    Gracemaker has no ``--version`` flag, so the probe runs ``python -c``
    with an ``importlib.metadata`` lookup inside the environment that owns
    the console script. With setup lines that is the setup shell, uncached
    (the binary may only exist inside it; fits are rare and the probe is
    cheap). Without them the probe resolves the ``gracemaker`` executable
    and uses the interpreter beside it — falling back to its shebang line —
    memoized on the executable's resolved path and mtime so a long-lived
    process still re-probes after an environment swap.
    """
    resolved = gracemaker_command(command)
    lines = gracemaker_setup(setup)
    if lines:
        return _probe_version_in_shell(lines)
    from slab.backends import _executable_identity

    identity = _executable_identity(resolved)
    if identity is None:
        return None
    return _probe_version(resolved, identity)


def describe_gracemaker(
    command: str | None = None, setup: str | tuple[str, ...] | list[str] | None = None
) -> dict[str, object]:
    """Identity of the gracemaker install: provenance and cache identity in one.

    The resolved command and the detected tensorpotential version enter
    every ``train_potential`` cache key, so pointing at a different
    environment or upgrading it honestly invalidates cached fits. An
    undetectable version degrades to the resolved executable's path+mtime
    fingerprint, the same honest fallback the engines use.

    Examples:
        >>> describe_gracemaker("definitely-not-installed-gracemaker")["builder"]
        'gracemaker'
    """
    resolved = gracemaker_command(command)
    lines = gracemaker_setup(setup)
    identity: dict[str, object] = {
        "builder": "gracemaker",
        "command": resolved,
        "version": gracemaker_version(command, setup),
    }
    if lines:
        identity["setup"] = list(lines)
    if identity["version"] is None:
        from slab.backends import _versionless_fingerprint

        identity.update(_versionless_fingerprint(resolved, lines))
    return identity


def train_scratch_dir() -> Path:
    """A fresh slab-managed scratch directory for one fit (``[paths] scratch``)."""
    from slab.backends import _scratch_dir

    return _scratch_dir("slab-gracemaker-")


def run_gracemaker(
    args: list[str] | tuple[str, ...],
    *,
    cwd: str | os.PathLike[str],
    command: str | None = None,
    setup: str | tuple[str, ...] | list[str] | None = None,
    timeout_s: float = 86400.0,
) -> GracemakerOutcome:
    """Run one gracemaker invocation in *cwd* and classify the outcome.

    *args* is gracemaker's own argument list (an input file name, restart
    and export flags) without the leading command. The caller — normally
    ``foundation.tasks.train_potential`` — owns staging, so this seam does
    not police path-shaped tokens. The subprocess runs in its own process
    group; on timeout the whole group is killed, so TensorFlow worker
    processes die with the trainer. Success returns the full captured log;
    failure (nonzero exit, or a python traceback in the log) raises
    :class:`~slab.errors.BuilderError` carrying the traceback tail and the
    full log.
    """
    argv = _guarded_args(args)
    resolved = gracemaker_command(command)
    lines = gracemaker_setup(setup)
    if not lines:
        _require_available(resolved)
    run_argv = _run_argv(resolved, lines, argv)
    env = {**os.environ, "LANG": "C", "LC_ALL": "C"}
    try:
        process = subprocess.Popen(
            run_argv,
            cwd=os.fspath(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            start_new_session=True,
        )
    except OSError as e:
        raise BuilderError(f"cannot run {resolved!r}: {e}") from e
    try:
        log, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        # TimeoutExpired carries the output read so far as bytes, even
        # under text=True; decoding it is what keeps the evidence.
        partial = _partial_output(e.stdout) or _kill_process_group(process)
        raise BuilderError(
            f"gracemaker did not finish within {timeout_s:.0f}s; its process "
            f"group was killed (command: {resolved})",
            log=partial,
        ) from e
    log = log or ""
    if process.returncode != 0 or TRACEBACK_MARKER in log:
        evidence = "\n  ".join(error_lines(log))
        raise BuilderError(
            f"gracemaker failed (exit {process.returncode}):\n  {evidence}",
            log=log,
        )
    return GracemakerOutcome(command=resolved, args=tuple(argv), log=log)


def error_lines(log: str, limit: int = _EVIDENCE_LIMIT) -> list[str]:
    """The lines of a gracemaker log worth reading after a failure.

    A python traceback is the evidence: everything from the last
    ``Traceback`` line to the end, bounded. A log without one contributes
    lines mentioning an error, else its last non-empty lines — evidence is
    never empty.

    Examples:
        >>> error_lines("epoch 1\\nTraceback (most recent call last):\\nKeyError: 'cutoff'")
        ['Traceback (most recent call last):', "KeyError: 'cutoff'"]
    """
    stripped = [line.rstrip() for line in log.splitlines()]
    for index in range(len(stripped) - 1, -1, -1):
        if TRACEBACK_MARKER in stripped[index]:
            tail = [line for line in stripped[index:] if line.strip()]
            return tail[-limit:] or ["(gracemaker produced no output)"]
    interesting = [
        line.strip()
        for line in stripped
        if "error" in line.lower() and line.strip()
    ]
    if not interesting:
        interesting = [line.strip() for line in stripped if line.strip()][-5:]
    return interesting[:limit] or ["(gracemaker produced no output)"]


def _guarded_args(args: list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(args, str):
        raise BuilderError(
            "pass gracemaker arguments as a list of tokens (foundation's "
            "train_potential task builds the list for you)"
        )
    argv = list(args)
    if not argv:
        raise BuilderError("no gracemaker arguments given; pass an input file name")
    for token in argv:
        if not isinstance(token, str):
            raise BuilderError(f"gracemaker arguments must be strings; got {token!r}")
    return argv


def _run_argv(command: str, setup: tuple[str, ...], args: list[str]) -> list[str]:
    """The argv to execute: the command directly, or through a setup shell.

    With setup lines the invocation becomes a fail-fast login shell (so the
    ``module`` shell function exists, and a failing load kills the run
    instead of exec'ing into the wrong environment), exactly the engines'
    wrapper semantics.
    """
    try:
        payload = shlex.split(command)
    except ValueError as e:
        raise BuilderError(f"cannot parse builder command {command!r}: {e}") from e
    if not payload:
        raise BuilderError("the gracemaker command is empty")
    if not setup:
        return [*payload, *args]
    quoted = " ".join(shlex.quote(token) for token in [*payload, *args])
    script = "\n".join(["set -e", *setup, f"exec {quoted}"])
    return ["/bin/bash", "-l", "-c", script]


def _kill_process_group(process: subprocess.Popen[str]) -> str:
    """Kill the subprocess's whole group, then reap. Never raises.

    Returns whatever output the reap still collected (often empty: the
    pipe was drained by the communicate that timed out).
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    with contextlib.suppress(Exception):
        remainder, _ = process.communicate(timeout=10)
        return _partial_output(remainder)
    return ""


def _partial_output(raw: object) -> str:
    """The text of a subprocess's partial output, whatever type it arrived as.

    Examples:
        >>> _partial_output(b"epoch 1\\n")
        'epoch 1\\n'
        >>> _partial_output("epoch 1\\n")
        'epoch 1\\n'
        >>> _partial_output(None)
        ''
    """
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw if isinstance(raw, str) else ""


def _require_available(command: str) -> None:
    from slab.backends import _command_payload

    payload = _command_payload(command)
    try:
        argv = shlex.split(command) if payload is None else payload
    except ValueError as e:
        raise BuilderError(f"cannot parse builder command {command!r}: {e}") from e
    if not argv or shutil.which(argv[0]) is None:
        raise BuilderNotAvailableError(
            f"gracemaker executable {argv[0] if argv else command!r} not found on "
            "PATH. Install tensorpotential in its own python environment "
            "(https://gracemaker.readthedocs.io), then point "
            "[builders.gracemaker] command at the gracemaker script, or add "
            "[builders.gracemaker] setup lines (module loads, an environment "
            "activation) that provide it"
        )


def _probe_version_in_shell(setup: tuple[str, ...]) -> str | None:
    quoted = " ".join(shlex.quote(token) for token in ["python", "-c", _VERSION_PROBE])
    script = "\n".join(["set -e", *setup, f"exec {quoted}"])
    return _spawn_version_probe(["/bin/bash", "-l", "-c", script])


@functools.lru_cache(maxsize=16)
def _probe_version(command: str, identity: tuple[str | int, ...]) -> str | None:
    """The environment's tensorpotential version, memoized on the executable."""
    del identity  # keys the memo, exactly like the engine probes
    interpreter = _sibling_python(command)
    if interpreter is None:
        return None
    return _spawn_version_probe([interpreter, "-c", _VERSION_PROBE])


def _sibling_python(command: str) -> str | None:
    """The interpreter of the environment that owns the gracemaker script.

    A console script lives in its environment's bin directory next to its
    interpreter, and its first line is a shebang naming that interpreter. A
    bare ``python`` from PATH would be SLAB's own environment — the wrong
    one by construction — so the probe never falls back to it.
    """
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv:
        return None
    resolved = shutil.which(argv[0])
    if resolved is None:
        return None
    script = Path(resolved)
    for name in ("python", "python3"):
        candidate = script.parent / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    try:
        with open(script, "rb") as handle:
            first = handle.readline(512).decode("utf-8", errors="replace").strip()
    except OSError:
        return None
    if first.startswith("#!"):
        shebang = first[2:].split()
        # Only an absolute interpreter names the script's own environment;
        # ``#!/usr/bin/env python3`` would resolve to whatever PATH holds.
        if shebang and shebang[-1].startswith("/") and Path(shebang[-1]).name.startswith("python"):
            return shebang[-1]
    return None


def _spawn_version_probe(argv: list[str]) -> str | None:
    try:
        with tempfile.TemporaryDirectory(prefix="slab-gracemaker-version-") as probe_dir:
            completed = subprocess.run(
                argv,
                cwd=probe_dir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env={**os.environ, "LANG": "C", "LC_ALL": "C"},
                text=True,
                timeout=_VERSION_PROBE_TIMEOUT_S,
                check=False,
            )
        if completed.returncode != 0:
            return None
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if lines and _VERSION_PATTERN.match(lines[-1]):
            return lines[-1]
        return None
    except Exception:
        return None


def _gracemaker_setting(key: str) -> Any:
    from slab.config import config_value

    return config_value(f"builders.gracemaker.{key}")
