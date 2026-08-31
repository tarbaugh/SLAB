"""Atomsk, the structure builder: build and transform structures, no physics.

Atomsk (P. Hirel, Comput. Phys. Commun. 197 (2015) 212,
https://atomsk.univ-lille.fr) is a command-line tool that creates and
transforms atomic structures: unit cells, supercells, dislocations, grain
boundaries, polycrystals, and conversions between the file formats the
engines read. It computes no energies and no forces, so it is **not an
engine**: ``engine="atomsk"`` does not exist, and nothing here builds a
calculator. SLAB names this kind of tool a *builder*, configured under
``[builders.atomsk]`` in the config file::

    [builders.atomsk]
    command = "atomsk"                       # or an absolute path
    setup = ["module load atomsk/0.13"]      # per-builder scoped shell

This module is the subprocess seam only: resolve the command, probe the
version, run one invocation in a caller-chosen directory, and classify the
outcome. The traced task that stages structures in and reads structures out
is ``foundation.tasks.build_structure``.

Failure detection reads the log, not only the exit code. Atomsk exits 0
after many of its own errors (a malformed ``--create``, for example) and
prints ``X!X ERROR`` lines instead; other failures (a missing input file
under a closed stdin) exit nonzero with the useful line marked ``/!\\``.
Both observations come from real executions, and :func:`run_atomsk` treats
either signal as failure.

Every subprocess runs with ``LANG=C``/``LC_ALL=C`` (atomsk localizes its
messages from ``$LANG``, and the markers above are the English ones), with
stdin closed (atomsk prompts interactively when an output file exists or an
input file is missing; at end-of-file it overwrites or dies — it never
hangs), and under a hard timeout.
"""

from __future__ import annotations

import functools
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slab.errors import BuilderError, BuilderNotAvailableError

#: The marker atomsk prints on its own error lines (English locale).
ERROR_MARKER = "X!X"

_VERSION_PATTERN = re.compile(r"[Vv]ersion\s+(\S+)")
_VERSION_PROBE_TIMEOUT_S = 20
_EVIDENCE_LIMIT = 10


@dataclass(frozen=True)
class AtomskOutcome:
    """One successful atomsk invocation: what ran, and everything it printed."""

    command: str
    args: tuple[str, ...]
    log: str


def atomsk_command(command: str | None = None) -> str:
    """The command that runs atomsk: per-call value, else config, else ``atomsk``.

    Examples:
        >>> atomsk_command("/opt/atomsk/atomsk")
        '/opt/atomsk/atomsk'
    """
    if command is not None:
        return command
    configured = _atomsk_setting("command")
    return str(configured) if configured else "atomsk"


def atomsk_setup(setup: str | tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    """Scoped setup lines for atomsk: per-call value, else ``[builders.atomsk]``.

    Setup lines are shell for a private login-shell wrapper around the
    atomsk subprocess only (module loads, exports) — the same per-tool rule
    as ``[engines.qe] setup``, never job-wide.
    """
    if setup is not None:
        lines = [setup] if isinstance(setup, str) else list(setup)
        return tuple(str(line) for line in lines)
    configured = _atomsk_setting("setup")
    if not configured:
        return ()
    return tuple(str(line) for line in configured)


def atomsk_version(
    command: str | None = None, setup: str | tuple[str, ...] | list[str] | None = None
) -> str | None:
    """The version ``atomsk --version`` reports, or None. Never raises.

    Without setup lines the probe is memoized on the executables' resolved
    path and mtime (the qe/lammps convention), so a long-lived process still
    re-probes after a binary swap. With setup lines the binary may only
    exist inside the setup shell; the probe then runs through that shell,
    uncached — builds are rare and the probe is cheap.
    """
    resolved = atomsk_command(command)
    lines = atomsk_setup(setup)
    if lines:
        return _probe_version_in_shell(resolved, lines)
    from slab.backends import _executable_identity

    identity = _executable_identity(resolved)
    if identity is None:
        return None
    return _probe_version(resolved, identity)


def describe_atomsk(
    command: str | None = None, setup: str | tuple[str, ...] | list[str] | None = None
) -> dict[str, object]:
    """Identity of the atomsk install: provenance and cache identity in one.

    The resolved command and the detected version enter every
    ``build_structure`` cache key, so pointing at a different binary or
    upgrading it honestly invalidates cached structures. An undetectable
    version degrades to the resolved executables' path+mtime fingerprint,
    the same honest fallback the engines use.

    Examples:
        >>> describe_atomsk("definitely-not-installed-atomsk")["builder"]
        'atomsk'
    """
    resolved = atomsk_command(command)
    lines = atomsk_setup(setup)
    identity: dict[str, object] = {
        "builder": "atomsk",
        "command": resolved,
        "version": atomsk_version(command, setup),
    }
    if lines:
        identity["setup"] = list(lines)
    if identity["version"] is None:
        from slab.backends import _versionless_fingerprint

        identity.update(_versionless_fingerprint(resolved, lines))
    return identity


def build_scratch_dir() -> Path:
    """A fresh slab-managed scratch directory for one build (``[paths] scratch``)."""
    from slab.backends import _scratch_dir

    return _scratch_dir("slab-atomsk-")


def run_atomsk(
    args: list[str] | tuple[str, ...],
    *,
    cwd: str | os.PathLike[str],
    command: str | None = None,
    setup: str | tuple[str, ...] | list[str] | None = None,
    timeout_s: float = 600.0,
) -> AtomskOutcome:
    """Run one atomsk invocation in *cwd* and classify the outcome.

    *args* is atomsk's own argument list — mode, options, and file names —
    without the leading command. Every file the invocation reads or writes
    must live in *cwd* under a bare name: an argument that looks like a
    path somewhere else is refused up front, because a traced build whose
    inputs live outside its working directory records an identity that
    lies. Success returns the full captured log; failure (nonzero exit, or
    ``X!X`` error lines in the log) raises :class:`~slab.errors.BuilderError`
    carrying the extracted error lines and the full log.
    """
    argv = _guarded_args(args)
    resolved = atomsk_command(command)
    lines = atomsk_setup(setup)
    if not lines:
        _require_available(resolved)
    run_argv = _run_argv(resolved, lines, argv)
    env = {**os.environ, "LANG": "C", "LC_ALL": "C"}
    try:
        completed = subprocess.run(
            run_argv,
            cwd=os.fspath(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        partial = e.stdout if isinstance(e.stdout, str) else ""
        raise BuilderError(
            f"atomsk did not finish within {timeout_s:.0f}s and was killed "
            f"(command: {resolved})",
            log=partial,
        ) from e
    except OSError as e:
        raise BuilderError(f"cannot run {resolved!r}: {e}") from e
    log = completed.stdout or ""
    if completed.returncode != 0 or ERROR_MARKER in log:
        evidence = "\n  ".join(error_lines(log))
        raise BuilderError(
            f"atomsk failed (exit {completed.returncode}):\n  {evidence}",
            log=log,
        )
    return AtomskOutcome(command=resolved, args=tuple(argv), log=log)


def error_lines(log: str, limit: int = _EVIDENCE_LIMIT) -> list[str]:
    """The lines of an atomsk log worth reading after a failure.

    Atomsk's own error lines carry ``X!X``, its warnings ``/!\\`` (the
    missing-input-file report is a warning), and a crash of the Fortran
    runtime announces itself by name. A log with none of these contributes
    its last non-empty lines instead — evidence is never empty.

    Examples:
        >>> error_lines("banner\\nX!X ERROR: no such lattice\\ndone")
        ['X!X ERROR: no such lattice']
    """
    interesting = [
        line.strip()
        for line in log.splitlines()
        if "X!X" in line or "/!\\" in line or "Fortran runtime error" in line
    ]
    if not interesting:
        interesting = [line.strip() for line in log.splitlines() if line.strip()][-5:]
    return interesting[:limit] or ["(atomsk produced no output)"]


def _guarded_args(args: list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(args, str):
        raise BuilderError(
            "pass atomsk arguments as a list of tokens (foundation's "
            "build_structure task splits a string for you)"
        )
    argv = list(args)
    if not argv:
        raise BuilderError("no atomsk arguments given; pass a mode, options, and file names")
    for token in argv:
        if not isinstance(token, str):
            raise BuilderError(f"atomsk arguments must be strings; got {token!r}")
        if "/" in token or "\\" in token or token.startswith("~") or token == "..":
            raise BuilderError(
                f"argument {token!r} names a path outside the working directory; "
                "atomsk runs in a private scratch directory, so use bare file "
                "names and stage input files into it first (build_structure "
                "does this through its inputs= mapping)"
            )
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
        raise BuilderError("the atomsk command is empty")
    if not setup:
        return [*payload, *args]
    quoted = " ".join(shlex.quote(token) for token in [*payload, *args])
    script = "\n".join(["set -e", *setup, f"exec {quoted}"])
    return ["/bin/bash", "-l", "-c", script]


def _require_available(command: str) -> None:
    from slab.backends import _command_payload

    payload = _command_payload(command)
    try:
        argv = shlex.split(command) if payload is None else payload
    except ValueError as e:
        raise BuilderError(f"cannot parse builder command {command!r}: {e}") from e
    if not argv or shutil.which(argv[0]) is None:
        raise BuilderNotAvailableError(
            f"atomsk executable {argv[0] if argv else command!r} not found on PATH. "
            "Install atomsk (https://atomsk.univ-lille.fr), or point "
            "[builders.atomsk] command at it, or add [builders.atomsk] setup "
            "lines (module loads) that provide it"
        )


def _probe_version_in_shell(command: str, setup: tuple[str, ...]) -> str | None:
    quoted = " ".join(shlex.quote(token) for token in [*shlex.split(command), "--version"])
    script = "\n".join(["set -e", *setup, f"exec {quoted}"])
    return _spawn_version_probe(["/bin/bash", "-l", "-c", script])


@functools.lru_cache(maxsize=16)
def _probe_version(command: str, identity: tuple[str | int, ...]) -> str | None:
    """Parse ``atomsk --version``, memoized on the executables' identity."""
    del identity  # keys the memo, exactly like the engine probes
    try:
        argv = [*shlex.split(command), "--version"]
    except ValueError:
        return None
    return _spawn_version_probe(argv)


def _spawn_version_probe(argv: list[str]) -> str | None:
    try:
        with tempfile.TemporaryDirectory(prefix="slab-atomsk-version-") as probe_dir:
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
        match = _VERSION_PATTERN.search(completed.stdout)
        return match.group(1) if match else None
    except Exception:
        return None


def _atomsk_setting(key: str) -> Any:
    from slab.config import config_value

    return config_value(f"builders.atomsk.{key}")
