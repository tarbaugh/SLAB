"""SLURM, taken at its word: render sbatch scripts from config, submit, poll.

This is deliberately a *thin* scheduler layer, not a workflow engine — runs,
caching, and verification stay in :mod:`foundation.runtime` regardless of where the
Python process executes. What this module adds is the cluster-shaped plumbing
around it, driven entirely by the ``[hpc]`` section of the slab config
(:mod:`slab.config`):

* :func:`render_sbatch` turns a partition declaration plus a command into a
  complete batch script. Only fields the config *sets* become ``#SBATCH``
  directives (Parsl's convention) — SLAB adds no silent resource defaults,
  and the rendered script is plain text anyone can read before trusting.
  The usual payload is ``foundation run workflow.py``: the scheduler moves the
  process; provenance is unchanged.
* :func:`submit` writes the script next to the job's outputs and calls
  ``sbatch --parsable``.
* :func:`job_state` polls ``squeue`` first and falls back to ``sacct``
  (finished jobs leave the queue), collapsing SLURM's ~22 states onto a
  seven-value enum (the QToolKit/PSI-J pattern) while preserving the raw
  state string as evidence.
* :func:`cancel` wraps ``scancel`` idempotently.

A machine without ``sbatch`` refuses loudly at the point of use — a laptop
is the normal case, not an error, until something actually tries to submit.
"""

from __future__ import annotations

import getpass
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from slab.config import HpcConfig, load_config
from slab.errors import SlabError

_SBATCH_TIMEOUT_S = 60
_SQUEUE_TIMEOUT_S = 60
_FALLBACK_JOB_ID = re.compile(r"Submitted batch job (\d+)")
_JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_TIME_LIMIT = re.compile(r"[0-9][0-9:\-]*")

# SLURM state -> JobState. Anything unlisted maps to UNDETERMINED with the
# raw string preserved — an unknown state must never masquerade as a known one.
_STATE_MAP = {
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "REQUEUED": "pending",
    "REQUEUE_HOLD": "pending",
    "REQUEUE_FED": "pending",
    "RESV_DEL_HOLD": "pending",
    "SUSPENDED": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "STAGE_OUT": "running",
    "SIGNALING": "running",
    "COMPLETED": "completed",
    "TIMEOUT": "timeout",
    "CANCELLED": "cancelled",
    "REVOKED": "cancelled",
    "FAILED": "failed",
    "BOOT_FAIL": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "DEADLINE": "failed",
    "PREEMPTED": "failed",
}


class SchedulerError(SlabError):
    """A scheduler interaction failed (submission, polling, cancellation)."""


class SchedulerNotAvailableError(SchedulerError):
    """The scheduler's commands are not on this machine's PATH."""


class JobState(StrEnum):
    """The seven states SLAB distinguishes (raw SLURM state kept alongside)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNDETERMINED = "undetermined"

    @property
    def is_terminal(self) -> bool:
        """True once the scheduler will never advance this job again.

        Examples:
            >>> JobState.RUNNING.is_terminal, JobState.TIMEOUT.is_terminal
            (False, True)
        """
        return self in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMEOUT,
        )


class JobStatus(BaseModel):
    """One poll's answer: the collapsed state plus the scheduler's own words."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    state: JobState
    raw: str | None = None
    detail: str | None = None


class SubmittedJob(BaseModel):
    """What :func:`submit` hands back — enough to poll, cancel, and audit."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    job_name: str
    partition: str
    script_path: str


def render_sbatch(
    command: str,
    *,
    job_name: str,
    partition: str | None = None,
    config: HpcConfig | None = None,
    time_limit: str | None = None,
    output: str | None = None,
    prologue: Iterable[str] = (),
    use_launcher: bool = True,
    include_global_setup: bool = True,
) -> str:
    """A complete sbatch script for *command* on the given partition.

    Only declared fields become ``#SBATCH`` directives; the ``[hpc]``-level
    ``account`` and ``setup`` apply unless the partition sets its own. The
    partition's ``launcher`` (``srun``, ``mpirun -np 4`` ...) prefixes the
    command — except a command invoking one of the drivers
    (``foundation run ...``), which is always emitted unlaunched with an
    explanatory comment: launching the driver replicates the whole workflow
    once per task and deadlocks the engines' own nested ``srun``.
    *time_limit* overrides the partition's; *output* defaults to
    ``<job_name>-%j.out`` (SLURM merges stderr into it when no separate
    error file is asked for). The result is pure text — render it, read it,
    then submit it.

    *prologue* lines run after the setup lines and before the command —
    where a caller puts bookkeeping the job must do for itself. Passing
    ``use_launcher=False`` suppresses the partition's launcher prefix, for
    payloads that must run in the batch script's own shell rather than as a
    parallel step (a single-node inference server, notably: it has to stay
    on the node whose hostname the prologue announced).
    ``include_global_setup=False`` drops the ``[hpc]``-level setup lines while
    keeping the partition's own — for jobs that bring a self-contained
    environment (the model-serve job's venv) that global engine module loads
    would fight.

    Examples:
        >>> from slab.config import HpcConfig
        >>> hpc = HpcConfig.model_validate({
        ...     "account": "abc-123",
        ...     "default_partition": "cpu",
        ...     "partitions": {"cpu": {"time_limit": "04:00:00", "ntasks": 8,
        ...                            "launcher": "srun"}},
        ... })
        >>> script = render_sbatch("pw.x -in si.pwi", job_name="si", config=hpc)
        >>> print(script)
        #!/bin/bash -l
        #SBATCH --job-name=si
        #SBATCH --partition=cpu
        #SBATCH --output=si-%j.out
        #SBATCH --account=abc-123
        #SBATCH --time=04:00:00
        #SBATCH --ntasks=8
        <BLANKLINE>
        set -euo pipefail
        <BLANKLINE>
        srun pw.x -in si.pwi
    """
    _validate_job_name(job_name)
    if time_limit is not None and not _TIME_LIMIT.fullmatch(time_limit):
        raise SchedulerError(
            f"time limit {time_limit!r} is not a SLURM time form (e.g. 04:00:00, 1-00:00:00)"
        )
    if output is not None and any(c.isspace() for c in output):
        raise SchedulerError(f"output path {output!r} must not contain whitespace")
    hpc = config if config is not None else load_config().hpc
    name, spec = hpc.resolve_partition(partition)
    lines = ["#!/bin/bash -l", f"#SBATCH --job-name={job_name}", f"#SBATCH --partition={name}"]
    lines.append(f"#SBATCH --output={output or f'{job_name}-%j.out'}")

    def directive(flag: str, value: object | None) -> None:
        if value is not None:
            lines.append(f"#SBATCH --{flag}={value}")

    directive("account", spec.account or hpc.account)
    directive("qos", spec.qos)
    directive("time", time_limit or spec.time_limit)
    directive("nodes", spec.nodes)
    directive("ntasks", spec.ntasks)
    directive("ntasks-per-node", spec.ntasks_per_node)
    directive("cpus-per-task", spec.cpus_per_task)
    directive("mem", spec.mem)
    directive("gres", spec.gres)
    directive("constraint", spec.constraint)
    directive("reservation", spec.reservation)
    lines.extend(f"#SBATCH {extra}" for extra in spec.sbatch_extra)

    body = ["", "set -euo pipefail", ""]
    if include_global_setup:
        body.extend(hpc.setup)
    body.extend(spec.setup)
    body.extend(prologue)
    launcher = spec.launcher if use_launcher else None
    driver = _driver_payload_name(command) if launcher else None
    if driver is not None:
        body.append(
            f"# partition launcher omitted: '{driver}' is a single-process driver;"
        )
        body.append('# engines bring their own MPI (e.g. [engines.qe] command = "srun pw.x")')
        launcher = None
    body.append(f"{launcher} {command}" if launcher else command)
    return "\n".join(lines + body)


_DRIVERS = frozenset({"slab", "foundation", "mason"})


def _driver_payload_name(command: str) -> str | None:
    """The driver *command* invokes (``foundation run ...``), or None.

    All three console scripts are single-process drivers.
    ``srun``-launching one replicates the whole workflow once per task — N
    concurrent writers on one runs database — and each replica's own engine
    ``srun`` then deadlocks on the already-consumed allocation. The launcher
    belongs on engine commands *inside* the driver ([engines.qe]
    ``command = "srun pw.x"``), never on the driver.
    """
    # Env assignments legitimately prefix shell payloads
    # (OMP_NUM_THREADS=4 foundation run ...), with or without the explicit
    # 'env' wrapper (flags included: env -u VAR foundation run ...); the check
    # must see through them all with the SAME parser the engine guards use,
    # or the launcher sneaks back onto the driver for a form one parser
    # accepts and the other does not.
    from slab.backends import _SHELL_ENV_ASSIGNMENT, _command_payload

    payload = _command_payload(command)
    if payload is None:
        return None
    for token in payload:
        if _SHELL_ENV_ASSIGNMENT.fullmatch(token):
            continue
        name = Path(token).name
        return name if name in _DRIVERS else None
    return None


def submit(
    script: str,
    *,
    job_name: str,
    partition: str,
    directory: str | os.PathLike[str] | None = None,
) -> SubmittedJob:
    """Write *script* into *directory* and ``sbatch`` it; keep it per job id.

    The script is submitted as ``<job_name>.sbatch`` and then renamed to
    ``<job_name>-<job_id>.sbatch``: a submitted job's exact script is
    provenance, and two jobs sharing a name in one directory must not
    overwrite each other's evidence while both are still queued. Raises
    :class:`SchedulerNotAvailableError` when ``sbatch`` is not on PATH and
    :class:`SchedulerError` (carrying stderr) when submission fails.
    """
    _require("sbatch")
    _validate_job_name(job_name)
    workdir = Path(directory) if directory is not None else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    script_path = workdir / f"{job_name}.sbatch"
    script_path.write_text(script, encoding="utf-8")
    result = _run_scheduler_command(
        ["sbatch", "--parsable", script_path.name],
        cwd=workdir,
        timeout=_SBATCH_TIMEOUT_S,
        env=_submission_env(),
    )
    if result.returncode != 0:
        raise SchedulerError(
            f"sbatch refused {script_path.name} (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip() or 'no output'}"
        )
    job_id = _parse_job_id(result.stdout)
    if job_id is None:
        raise SchedulerError(
            f"sbatch exited 0 but its output has no job id: {result.stdout.strip()!r}"
        )
    kept = workdir / f"{job_name}-{job_id}.sbatch"
    try:
        script_path.replace(kept)
    except OSError:  # keeping the evidence is best-effort; the job is submitted
        kept = script_path
    return SubmittedJob(
        job_id=job_id, job_name=job_name, partition=partition, script_path=str(kept)
    )


def _parse_job_id(stdout: str) -> str | None:
    """``--parsable`` output is ``jobid[;cluster]``; wrapped sbatch prints prose.

    Examples:
        >>> _parse_job_id("123456;cluster\\n")
        '123456'
        >>> _parse_job_id("Submitted batch job 7890\\n")
        '7890'
        >>> _parse_job_id("") is None
        True
    """
    first = stdout.strip().splitlines()[0].split(";")[0].strip() if stdout.strip() else ""
    if first.isdigit():
        return first
    match = _FALLBACK_JOB_ID.search(stdout)
    return match.group(1) if match else None


def job_state(job_id: str) -> JobStatus:
    """Ask the scheduler about one job: ``squeue`` first, ``sacct`` fallback.

    Finished jobs leave ``squeue``; ``sacct`` (when accounting is enabled)
    remembers them. When neither answers, the state is ``UNDETERMINED`` with
    a ``detail`` saying why — an unknown must never be reported as a known.
    """
    _require("squeue")
    queued = _run_scheduler_command(
        ["squeue", "-j", job_id, "-h", "-o", "%T"], timeout=_SQUEUE_TIMEOUT_S
    )
    rows = [line.strip() for line in queued.stdout.strip().splitlines() if line.strip()]
    if queued.returncode == 0 and rows:
        # A job array answers one row per task; the most active row speaks
        # for the job (a partly-running array is running, not 'completed').
        state = _aggregate(_collapse(row) for row in rows)
        return JobStatus(job_id=job_id, state=state, raw=", ".join(dict.fromkeys(rows)))
    return _sacct_state(job_id)


_AGGREGATION_ORDER = (
    JobState.RUNNING,
    JobState.PENDING,
    JobState.UNDETERMINED,
    JobState.FAILED,
    JobState.TIMEOUT,
    JobState.CANCELLED,
    JobState.COMPLETED,
)


def _aggregate(states: Iterable[JobState]) -> JobState:
    """The single state that best describes many rows of one job.

    Examples:
        >>> _aggregate([JobState.COMPLETED, JobState.RUNNING]).value
        'running'
        >>> _aggregate([JobState.COMPLETED, JobState.FAILED]).value
        'failed'
    """
    seen = set(states)
    for state in _AGGREGATION_ORDER:
        if state in seen:
            return state
    return JobState.UNDETERMINED


def _sacct_state(job_id: str) -> JobStatus:
    if shutil.which("sacct") is None:
        return JobStatus(
            job_id=job_id,
            state=JobState.UNDETERMINED,
            detail="job not in squeue and sacct is not on PATH (no accounting here); "
            "check the job's output file instead",
        )
    result = _run_scheduler_command(
        ["sacct", "-j", job_id, "-n", "-X", "-P", "-o", "State,ExitCode"],
        timeout=_SQUEUE_TIMEOUT_S,
    )
    line = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    if result.returncode != 0 or not line:
        return JobStatus(
            job_id=job_id,
            state=JobState.UNDETERMINED,
            detail=f"job not in squeue and sacct has no record of it "
            f"({result.stderr.strip() or 'empty answer'})",
        )
    raw, _, exit_code = line.partition("|")
    raw = raw.strip()
    status = JobStatus(
        job_id=job_id,
        state=_collapse(raw),
        raw=raw,
        detail=f"exit code {exit_code.strip()}" if exit_code.strip() else None,
    )
    return status


def _collapse(raw: str) -> JobState:
    """Map a raw SLURM state onto the seven-value enum.

    ``sacct`` decorates some states (``CANCELLED by 501``); the first word
    carries the state.

    Examples:
        >>> _collapse("CANCELLED by 501").value
        'cancelled'
        >>> _collapse("SPECIAL_SAUCE").value
        'undetermined'
    """
    keyword = raw.split()[0].split("+")[0] if raw.split() else ""
    mapped = _STATE_MAP.get(keyword)
    return JobState(mapped) if mapped else JobState.UNDETERMINED


def active_job_ids() -> frozenset[str]:
    """The user's job ids the scheduler still holds (pending or running).

    One ``squeue`` call for the whole answer, so a sweep over many job
    files pays for a single scheduler round trip. Raises
    :class:`SchedulerNotAvailableError` where there is no ``squeue`` — the
    caller decides what absence means (a purge on a laptop treats it as
    "nothing can be running") — and :class:`SchedulerError` when ``squeue``
    answers with a failure, because "the scheduler would not say" must
    never be read as "nothing is active".
    """
    _require("squeue")
    result = _run_scheduler_command(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%A"],
        timeout=_SQUEUE_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise SchedulerError(
            f"squeue could not list this user's jobs (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip() or 'no output'}"
        )
    return frozenset(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )


def cancel(job_id: str) -> None:
    """``scancel`` the job; cancelling an already-terminal job is a no-op.

    Raises only when ``scancel`` is missing — cancellation is idempotent by
    design (the job may finish in the race window), so a nonzero exit is
    not an error.
    """
    _require("scancel")
    _run_scheduler_command(["scancel", "-Q", job_id], timeout=_SQUEUE_TIMEOUT_S)


def _validate_job_name(job_name: str) -> None:
    """A job name lands in ``#SBATCH`` lines and a filename — keep it inert."""
    if not _JOB_NAME.fullmatch(job_name):
        raise SchedulerError(
            f"job name {job_name!r} is not usable: letters, digits, and ._+- only "
            f"(it becomes an #SBATCH directive and the script's filename)"
        )


def _require(executable: str) -> None:
    if shutil.which(executable) is None:
        raise SchedulerNotAvailableError(
            f"{executable!r} is not on PATH — this machine does not look like a "
            f"SLURM cluster (or the scheduler module is not loaded)"
        )


def _submission_env() -> dict[str, str] | None:
    """The environment sbatch should export into the job, or None for as-is.

    sbatch defaults to ``--export=ALL``: the job inherits the submitting
    process's environment wholesale. That is normal SLURM behavior for what
    the *user's shell* holds — but this process may also carry registry-engine
    residue (:func:`slab.engines.build_engine` writes entries' ``env``
    process-wide, for in-process calculators). A batch job is a fresh process
    that resolves its own config; leaking the residue would make the kept
    ``.sbatch`` script — the job's documented provenance — behave differently
    depending on which engines this process happened to build first. So the
    submission environment is the environment as it was *before* any registry
    entry ran: applied variables are restored to their original values, or
    dropped when they were originally unset.
    """
    from slab.engines import applied_env

    residue = applied_env()
    if not residue:
        return None
    env = dict(os.environ)
    for key, (original, applied) in residue.items():
        if os.environ.get(key) != applied:
            # The user changed (or deleted) it AFTER the registry applied it:
            # the current state is their intent, not residue — reverting it
            # would be its own silent poisoning.
            continue
        if original is None:
            env.pop(key, None)
        else:
            env[key] = original
    return env


def _run_scheduler_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = _SQUEUE_TIMEOUT_S,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise SchedulerError(
            f"{argv[0]} did not answer within {timeout:.0f}s (controller overloaded?)"
        ) from e
    except OSError as e:
        raise SchedulerError(f"could not execute {argv[0]}: {e}") from e
