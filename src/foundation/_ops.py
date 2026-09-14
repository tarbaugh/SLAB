"""Shared operations behind the Foundation CLI and the MCP server.

One code path, two skins. Everything here returns plain JSON-able dicts (or
domain objects the callers format), so the CLI renders text, the MCP server
returns structure, and the behavior cannot drift between them.

The engine-capability half of this module lives in :mod:`slab._ops`, because
it describes what SLAB can compute rather than what Foundation has run.
"""

from __future__ import annotations

import io
import json
import os
import re
import runpy
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING, Any

from foundation.artifacts import ReadThroughStore
from foundation.errors import (
    ArtifactNotFoundError,
    FoundationError,
    IllegalStatusChangeError,
    IllegalTransitionError,
    NestedRunError,
    ReplayError,
    ResourcesError,
    RunStateError,
    ScriptExitError,
    SessionNotFoundError,
    StorageError,
)
from foundation.lifecycle import ExecutionStatus, LifecycleState
from foundation.models import (
    ArtifactRole,
    CheckResult,
    GpuExclusion,
    Reservation,
    Run,
    TaskRecord,
    utcnow,
)
from foundation.references import find_artifact, hash_holders
from foundation.retention import DEFAULT_POLICY, RetentionPolicy, _reachable_hashes, sweep_scratch
from foundation.runtime import Replay, Workspace, describe_liveness, this_host, this_job
from foundation.serialize import loads
from slab.scratch import RUN_ENV, scratch_root

if TYPE_CHECKING:
    from slab.resources import JobSize

DEFAULT_ROOT = ".slab"
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])$")
_DAYS_PER_UNIT = {"s": 1 / 86_400, "m": 1 / 1_440, "h": 1 / 24, "d": 1.0}
_EFFECTIVELY_NOW = 1e-9  # ~90µs in days: "--older-than 0d" means "everything, now"


def resolve_root(explicit: str | os.PathLike[str] | None) -> Path:
    """Workspace root: explicit flag > $SLAB_WORKSPACE > config > ``./.slab``.

    The config layer is ``[workspace] root`` in :mod:`foundation.config`.

    Examples:
        >>> resolve_root("/tmp/ws")
        PosixPath('/tmp/ws')
        >>> import os, tempfile
        >>> os.environ.pop("SLAB_WORKSPACE", None) and None
        >>> os.environ.pop("SLAB_CONFIG", None) and None
        >>> os.environ.pop("SLAB_SITE_CONFIG", None) and None
        >>> os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
        >>> resolve_root(None)
        PosixPath('.slab')
    """
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get("SLAB_WORKSPACE")
    if from_env:
        return Path(from_env)
    from foundation.config import config_value

    configured = config_value("workspace.root")
    return Path(configured) if configured else Path(DEFAULT_ROOT)


def parse_duration_days(text: str) -> float:
    """Parse ``"30d"``, ``"12h"``, ``"45m"``, ``"90s"`` into days.

    Zero means "expire everything unpromoted, now" and maps to an epsilon
    (retention TTLs must be positive).

    Examples:
        >>> parse_duration_days("30d")
        30.0
        >>> parse_duration_days("12h")
        0.5
        >>> parse_duration_days("0d")
        1e-09
    """
    match = _DURATION.fullmatch(text.strip().lower())
    if match is None:
        raise ValueError(
            f"cannot parse duration {text!r}: use <number><unit> with unit s/m/h/d, e.g. 30d"
        )
    value = float(match.group(1)) * _DAYS_PER_UNIT[match.group(2)]
    return value if value > 0 else _EFFECTIVELY_NOW


def load_policy(root: Path, explicit_path: str | os.PathLike[str] | None = None) -> RetentionPolicy:
    """Load the retention policy: explicit file, else ``<root>/policy.json``, else defaults.

    Examples:
        >>> import tempfile
        >>> load_policy(Path(tempfile.mkdtemp())) is DEFAULT_POLICY
        True
    """
    path = Path(explicit_path) if explicit_path is not None else root / "policy.json"
    if explicit_path is None and not path.exists():
        return DEFAULT_POLICY
    with open(path, encoding="utf-8") as handle:
        return RetentionPolicy.model_validate(json.load(handle))


def ttl_override_policy(days: float) -> RetentionPolicy:
    """A policy whose only effect is 'expire quarantined/verified older than *days*'.

    Examples:
        >>> ttl_override_policy(7).verified.ttl_days
        7.0
    """
    return RetentionPolicy.model_validate(
        {"quarantined": {"ttl_days": days}, "verified": {"ttl_days": days}}
    )


def run_summary(run: Run) -> dict[str, Any]:
    """Compact JSON-able view of a run (used by list/promote/launch results)."""
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state.value,
        "status": run.status.value,
        "intent": run.intent,
        "session": run.session,
        "error": run.error,
        "created_at": run.created_at.isoformat(),
        "state_entered_at": run.state_entered_at.isoformat(),
        "resources": run.resources,
        "job_id": run.job_id,
    }


def describe_resources(resources: dict[str, Any] | None) -> str:
    """One phrase for the slice a run held, for a listing or a transcript.

    Examples:
        >>> describe_resources({"cpus": [0, 1, 2, 3], "gpus": ["0"], "ntasks": 2, "threads": 2})
        '4 cpu(s) 0-3, 1 gpu(s) 0; 2 rank(s) x 2 thread(s)'
        >>> describe_resources({"cpus": [1, 3], "gpus": [], "ntasks": 1, "threads": 1})
        '2 cpu(s) 1,3, no gpu; 1 rank(s) x 1 thread(s)'
        >>> describe_resources(None)
        'not reserved'
    """
    if not resources:
        return "not reserved"
    cpus = [int(cpu) for cpu in resources.get("cpus") or []]
    gpus = [str(gpu) for gpu in resources.get("gpus") or []]
    gpu_text = f"{len(gpus)} gpu(s) {','.join(gpus)}" if gpus else "no gpu"
    return (
        f"{len(cpus)} cpu(s) {_id_ranges(cpus)}, {gpu_text}; "
        f"{resources.get('ntasks', 1)} rank(s) x {resources.get('threads', 1)} thread(s)"
    )


def age_text(moment: datetime) -> str:
    """How long ago *moment* was, in one unit: ``12s``, ``3m``, ``2h``, ``1d``.

    Examples:
        >>> from datetime import timedelta
        >>> age_text(utcnow() - timedelta(minutes=3))
        '3m'
        >>> age_text(utcnow() + timedelta(hours=1))
        '0s'
    """
    seconds = max(0.0, (utcnow() - moment).total_seconds())
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86_400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86_400)}d"


def describe_reservation(held: Reservation) -> str:
    """One line for a live reservation: the slice, the run or the holder, the age.

    Examples:
        >>> held = Reservation(host="n1", cpus=(0, 1), gpus=(), ntasks=2, holder_pid=41)
        >>> describe_reservation(held).startswith(
        ...     "2 cpu(s) 0-1, no gpu; 2 rank(s) x 1 thread(s)  unclaimed, holder 41 on n1  "
        ... )
        True
        >>> claimed = held.model_copy(update={"run_id": "run-1"})
        >>> "claimed by run run-1" in describe_reservation(claimed)
        True
    """
    slice_text = describe_resources(held.slice)
    age = age_text(held.created_at)
    if held.run_id is not None:
        return f"{slice_text}  claimed by run {held.run_id}  {age}"
    return f"{slice_text}  unclaimed, holder {held.holder_pid} on {held.host}  {age}"


def free_resources(ws: Workspace) -> dict[str, Any]:
    """What is free on this host right now, and who holds the rest.

    :meth:`Workspace.free_resources` plus ``held``: one line per live
    reservation from :func:`describe_reservation`, in the order the
    reservations were made, and ``excluded_lines``: one line per gpu that
    refused a launch under this job (:func:`exclusion_line`). Dead
    reservations are released first, so the answer is what a launch made
    now would be judged against.
    """
    ws.reap_dead(caller="free_resources")
    # One read: a reservation that ends between two reads would be listed
    # by one and missing from the other, and the tool is called exactly
    # while concurrent launches finish.
    live = ws.runs.live_reservations(this_host(), this_job())
    answer = ws.free_resources(live=live)
    answer["held"] = [describe_reservation(row) for row in live]
    answer["excluded_lines"] = [
        exclusion_line(GpuExclusion.model_validate({**row, "host": answer["host"]}))
        for row in answer["excluded"]
    ]
    return answer


#: A device that refuses a launch this soon after LAMMPS started is taken
#: as broken for the rest of the job. A refusal later in a run is a
#: different fault (a device lost mid-run), and it is left to the reader.
REFUSAL_WINDOW_S = 60.0


def exclusion_line(row: GpuExclusion) -> str:
    """One line for an excluded gpu: ``gpu 0: excluded (refused at 14:02, job 812)``.

    The time is the host's local clock, the one an operator reads.

    Examples:
        >>> from datetime import datetime, timezone
        >>> row = GpuExclusion(gpu="0", host="n1", job_id="812", reason="refused",
        ...                    at=datetime(2026, 9, 14, 12, 2, tzinfo=timezone.utc))
        >>> exclusion_line(row).startswith("gpu 0: excluded (refused at ")
        True
        >>> exclusion_line(row).endswith(", job 812)")
        True
        >>> exclusion_line(row.model_copy(update={"job_id": None})).endswith(":02)")
        True
    """
    at = row.at.astimezone().strftime("%H:%M")
    job = f", job {row.job_id}" if row.job_id is not None else ""
    return f"gpu {row.gpu}: excluded (refused at {at}{job})"


def refused_gpu(
    evidence: str, held: Sequence[str], *, ranks: int | None, elapsed_s: float
) -> tuple[str | None, str]:
    """The held gpu a launch's failure evidence convicts, and why or why not.

    A device is convicted when all of these hold:

    * the evidence names ``cudaErrorDevicesUnavailable``
      (:func:`slab.lammps.cuda_device_error`);
    * LAMMPS failed within :data:`REFUSAL_WINDOW_S` of its start;
    * the launch ran no more MPI ranks than it held gpus, because a second
      rank on an exclusive-mode device gets the same error from a healthy
      device (an unknown rank count convicts nothing);
    * the device resolves: the ``cudaSetDevice(N)`` ordinal indexes the
      held ids (:func:`slab.lammps.cuda_device_ordinal`), else the launch
      held exactly one gpu.

    Returns ``(gpu, reason)`` with *gpu* None when nothing is convicted;
    *reason* then says which condition failed.

    Examples:
        >>> line = "what():  cudaSetDevice(1) error( cudaErrorDevicesUnavailable): busy"
        >>> refused_gpu(line, ("2", "3"), ranks=2, elapsed_s=1.2)
        ('3', 'cudaErrorDevicesUnavailable on gpu 3 1.2 s after LAMMPS started')
        >>> refused_gpu(line, ("2",), ranks=4, elapsed_s=1.2)[0] is None
        True
        >>> refused_gpu(line, ("2", "3"), ranks=2, elapsed_s=600)[1]
        'the device refused 600 s after LAMMPS started, past the 60 s window'
    """
    from slab.lammps import cuda_device_error, cuda_device_ordinal

    line = cuda_device_error(evidence)
    if line is None or "cudaErrorDevicesUnavailable" not in line:
        return None, "the evidence names no cudaErrorDevicesUnavailable"
    if elapsed_s > REFUSAL_WINDOW_S:
        return None, (
            f"the device refused {elapsed_s:.0f} s after LAMMPS started, past the "
            f"{REFUSAL_WINDOW_S:.0f} s window"
        )
    if not held:
        return None, "the launch held no gpu"
    if ranks is None or ranks > len(held):
        count = "an unknown number of" if ranks is None else str(ranks)
        return None, (
            f"the launch ran {count} rank(s) on {len(held)} gpu(s), so a healthy "
            f"exclusive-mode device refuses the same way"
        )
    ordinal = cuda_device_ordinal(line)
    if ordinal is not None and ordinal < len(held):
        gpu = held[ordinal]
    elif ordinal is None and len(held) == 1:
        gpu = held[0]
    else:
        return None, f"the error line names no device among the held ids {','.join(held)}"
    return gpu, (
        f"cudaErrorDevicesUnavailable on gpu {gpu} {elapsed_s:.1f} s after LAMMPS started"
    )


def exclude_refused_gpus(
    runs: Any,
    *,
    evidence: str,
    held: Sequence[str],
    ranks: int | None,
    elapsed_s: float,
    run_id: str | None = None,
    host: str | None = None,
    job_id: str | EllipsisType | None = ...,
) -> tuple[GpuExclusion | None, str]:
    """Record the gpu a launch was refused on, when :func:`refused_gpu` convicts one.

    *runs* is the run store. The row is keyed by this host and this job
    unless *host* and *job_id* are given, and it keeps the gpu out of every
    reservation of the job until the job ends. Returns the row, or None,
    and the reason from :func:`refused_gpu`.
    """
    gpu, reason = refused_gpu(evidence, held, ranks=ranks, elapsed_s=elapsed_s)
    if gpu is None:
        return None, reason
    row = runs.exclude_gpu(
        gpu,
        host=host if host is not None else this_host(),
        job_id=this_job() if job_id is ... else job_id,
        reason=reason,
        run_id=run_id,
    )
    return row, reason


def exclusion_note(row: GpuExclusion) -> str:
    """The failure note that says a gpu is now out of the budget.

    Mason reads the note back from the failure record to suggest a
    ``remember`` (:data:`EXCLUSION_NOTE` matches it).

    Examples:
        >>> note = exclusion_note(GpuExclusion(gpu="0", host="n1", job_id="7", reason="r"))
        >>> note
        "gpu 0 excluded: reservations skip it for the rest of job 7; 'slab runs gpus' lists it"
        >>> EXCLUSION_NOTE.match(note).group("gpu", "job")
        ('0', '7')
    """
    scope = f"the rest of job {row.job_id}" if row.job_id is not None else (
        f"this machine until 'slab runs gpus --clear {row.gpu}'"
    )
    return (
        f"gpu {row.gpu} excluded: reservations skip it for {scope}; "
        f"'slab runs gpus' lists it"
    )


#: Matches :func:`exclusion_note`; ``job`` is None for an exclusion outside a job.
EXCLUSION_NOTE = re.compile(
    r"^gpu (?P<gpu>\S+) excluded: reservations skip it for "
    r"(?:the rest of job (?P<job>[^;\s]+)|this machine)"
)


def budget_counts(
    cpus: Sequence[int], gpus: Sequence[str], gpu_source: str
) -> dict[str, Any]:
    """The budget as a listing shows it: two counts, the gpu ids, and their source.

    Examples:
        >>> budget_counts((0, 1, 2, 3), ("1",), "slurm_job_gpus")
        {'cpus': 4, 'gpus': 1, 'gpu_ids': ['1'], 'gpu_source': 'slurm_job_gpus'}
    """
    return {
        "cpus": len(cpus),
        "gpus": len(gpus),
        "gpu_ids": list(gpus),
        "gpu_source": gpu_source,
    }


def gpu_lines(gpu_ids: Sequence[str]) -> list[str]:
    """One line per budget gpu with the memory it has in use, per ``nvidia-smi``.

    An occupied device shows before a launch holds it. Nothing for an
    empty budget; one line saying so when ``nvidia-smi`` is not on the
    path or lists no device.

    Examples:
        >>> gpu_lines(())
        []
    """
    from slab.resources import device_status

    if not gpu_ids:
        return []
    devices = device_status()
    if devices is None:
        return ["  gpus: nvidia-smi not found"]
    by_id = {device["id"]: device for device in devices}
    lines: list[str] = []
    for gpu_id in gpu_ids:
        device = by_id.get(gpu_id)
        if device is None:
            lines.append(f"  gpu {gpu_id}: not listed by nvidia-smi")
        else:
            lines.append(
                f"  gpu {gpu_id}: {device['memory_used']} used, mode {device['mode']}"
            )
    return lines


def free_line(answer: dict[str, Any]) -> str:
    """The one-line trailer: ``free now: N cpu(s), M gpu(s)``.

    Examples:
        >>> free_line({"free": {"cpus": [2, 3], "gpus": []}})
        'free now: 2 cpu(s), 0 gpu(s)'
    """
    free = answer["free"]
    return f"free now: {len(free['cpus'])} cpu(s), {len(free['gpus'])} gpu(s)"


def _id_ranges(ids: list[int]) -> str:
    """Sorted ids as ranges: ``0-3,6``.

    Examples:
        >>> _id_ranges([6, 0, 1, 2, 3])
        '0-3,6'
        >>> _id_ranges([])
        ''
    """
    pieces: list[str] = []
    for value in sorted(set(ids)):
        if pieces and value == _range_end(pieces[-1]) + 1:
            start = pieces[-1].split("-")[0]
            pieces[-1] = f"{start}-{value}"
        else:
            pieces.append(str(value))
    return ",".join(pieces)


def _range_end(piece: str) -> int:
    return int(piece.split("-")[-1])


def resources_column(resources: dict[str, Any] | None) -> str:
    """The short form for a listing column: ``8c``, ``8c/2g``, or blank.

    Examples:
        >>> resources_column({"cpus": [0, 1], "gpus": ["0", "1"]})
        '2c/2g'
        >>> resources_column({"cpus": [0], "gpus": []}), resources_column(None)
        ('1c', '')
    """
    if not resources:
        return ""
    text = f"{len(resources.get('cpus') or [])}c"
    gpus = resources.get("gpus") or []
    return f"{text}/{len(gpus)}g" if gpus else text


def check_entry(result: CheckResult) -> dict[str, Any]:
    """One check result as the evidence surfaces print it.

    Examples:
        >>> check_entry(CheckResult(run_id="r", name="drift", passed=False,
        ...     message="returned False", evidence={"source": "def drift(): ...", "keys": {}}))
        ... # doctest: +NORMALIZE_WHITESPACE
        {'name': 'drift', 'kind': 'custom', 'passed': False, 'message': 'returned False',
         'observed': None, 'expected': None, 'pass_no': 1,
         'evidence': {'source': 'def drift(): ...', 'keys': {}}}
    """
    entry: dict[str, Any] = {
        "name": result.name,
        "kind": result.kind,
        "passed": result.passed,
        "message": result.message,
        "observed": result.observed,
        "expected": result.expected,
        "pass_no": result.pass_no,
    }
    if result.evidence is not None:
        entry["evidence"] = result.evidence
    return entry


def pass_tallies(results: Sequence[CheckResult]) -> list[dict[str, int]]:
    """``pass_no``, ``passed``, and ``total`` for each verification pass, oldest first.

    Examples:
        >>> pass_tallies([CheckResult(run_id="r", name="a", passed=False),
        ...               CheckResult(run_id="r", name="a", passed=True, pass_no=2)])
        [{'pass_no': 1, 'passed': 0, 'total': 1}, {'pass_no': 2, 'passed': 1, 'total': 1}]
    """
    tallies: dict[int, dict[str, int]] = {}
    for result in results:
        tally = tallies.setdefault(
            result.pass_no, {"pass_no": result.pass_no, "passed": 0, "total": 0}
        )
        tally["passed"] += int(result.passed)
        tally["total"] += 1
    return [tallies[number] for number in sorted(tallies)]


def run_details(ws: Workspace, run_id: str) -> dict[str, Any]:
    """Everything about one run: fields, checks, tasks, artifacts, history.

    This is the evidence surface for agents (``slab show`` and MCP
    ``show_run``): failed runs and tasks carry their structured ``failure``
    record (:func:`foundation.errors.failure_record`), and checks carry the
    ``observed``/``expected`` values their assertions compared — the numbers a
    correction gets computed from. A failed check that named no observed
    value carries ``evidence`` instead: its source, the keys of the dicts
    it read, and the line that raised. ``checks`` is the latest
    verification pass, and ``earlier_passes`` tallies the passes before
    it (see :func:`reverify_run`). Artifact entries carry ``bytes_available``
    — whether the content is still in the artifact store or has been
    hash-and-discarded. A cache-hit task carries ``artifacts_on``: the run
    where the task executed, which holds the files it keeps (None when
    that run is gone). :func:`read_artifact` follows the same edge.
    """
    run = ws.runs.get(run_id)
    checks = ws.runs.list_check_results(run.id)
    tasks = ws.runs.list_tasks(run.id)
    artifacts = ws.runs.list_artifacts(run.id)
    history = ws.runs.history(run.id)

    def task_entry(t: TaskRecord) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "seq": t.seq,
            "name": t.name,
            "status": t.status.value,
            "cache_hit": t.cache_hit,
        }
        if t.cache_hit:
            producer = ws.runs.producing_task(t)
            entry["artifacts_on"] = None if producer is None else producer.run_id
        return entry | {
            "error": t.error,
            "failure": t.failure,
            "duration_s": (
                None
                if t.finished_at is None
                else round((t.finished_at - t.started_at).total_seconds(), 3)
            ),
            "recipe": t.recipe,
            "inputs": t.inputs,
            "outputs": t.outputs,
        }
    return {
        "run": run_summary(run)
        | {
            "meta": run.meta,
            "failure": run.failure,
            "started_at": None if run.started_at is None else run.started_at.isoformat(),
            "finished_at": None if run.finished_at is None else run.finished_at.isoformat(),
            "pid": run.pid,
            "host": run.host,
        },
        "checks": [check_entry(c) for c in checks],
        "earlier_passes": pass_tallies(ws.runs.list_check_results(run.id, all_passes=True))[:-1],
        "tasks": [task_entry(t) for t in tasks],
        "artifacts": [
            {
                "name": a.name,
                "role": a.role.value,
                "hash": a.hash,
                "size_bytes": a.size_bytes,
                "bytes_available": ws.artifacts.has(a.hash),
            }
            for a in artifacts
        ],
        "history": [
            {
                "from": t.from_state.value,
                "to": t.to_state.value,
                "actor": t.actor,
                "reason": t.reason,
                "forced": t.forced,
                "at": t.at.isoformat(),
            }
            for t in history
        ],
    }


@dataclass(frozen=True)
class ArtifactRead:
    """What :func:`read_artifact` found: where it came from, and its text.

    *head* is the lines to show before the content, and the first says
    where the bytes came from. *name* is the artifact's name, which picks
    an engine-output digest ('' for a hash no run names). *text* is None
    when the bytes look binary, and *reason* then says so.
    """

    head: list[str]
    name: str
    digest: str
    size_bytes: int
    run_id: str | None
    text: str | None
    reason: str | None = None


def read_artifact(
    ws: Workspace,
    *,
    run_id: str | None = None,
    name: str | None = None,
    digest: str | None = None,
    session: str | None = None,
) -> ArtifactRead:
    """Read an artifact by run and name, or by hash: ``read_artifact``.

    By run and name: *run_id* resolves as for :func:`resolve_run`, and
    *name* is the artifact's name or a hash prefix. When the run holds no
    such artifact, each of its cache-hit tasks is followed to the run
    where the task executed (:func:`foundation.references.find_artifact`),
    and the first line of *head* says so.

    By hash: *digest* is a SHA-256 or a unique prefix of 6+ characters of
    any bytes the workspace holds, a run's file or a task's input or
    output value. *head* lists the runs and tasks that reference it. A
    task value comes back decoded: JSON as indented text, anything else
    as its ``repr``.

    Raises:
        ValueError: Neither a hash nor a run and a name, or both.
        RunNotFoundError: No run matches *run_id*.
        ArtifactNotFoundError: Nothing matches, or retention discarded
            the bytes; the message says which.
    """
    if digest:
        if run_id or name:
            raise ValueError("read by hash, or by run_id and name, not both")
        return _read_by_hash(ws, digest)
    if not (run_id and name):
        raise ValueError("read_artifact needs run_id and name, or hash")
    rid, note = resolve_run(ws, run_id, session=session)
    found = find_artifact(ws.runs, rid, name)
    ref = found.ref
    if not ws.artifacts.has(ref.hash):
        raise ArtifactNotFoundError(
            f"the bytes of {ref.name!r} are no longer stored (retention reclaimed "
            f"them); the record keeps its hash {ref.hash[:12]}",
            digest=ref.hash,
            run_id=ref.run_id,
            name=ref.name,
        )
    head = [line for line in (found.note(), note.rstrip("\n")) if line]
    head.append(f"{ref.name} ({ref.size_bytes} bytes, sha256 {ref.hash[:12]})")
    raw = ws.artifacts.get(ref.hash).read_bytes()
    text, reason = _artifact_text(raw, value=False)
    return ArtifactRead(head, ref.name, ref.hash, ref.size_bytes, ref.run_id, text, reason)


def _read_by_hash(ws: Workspace, digest: str) -> ArtifactRead:
    """The bytes a hash names, headed by what references them."""
    holders = hash_holders(ws.runs, ws.artifacts, digest)
    size = holders.path.stat().st_size
    head = [f"sha256 {holders.digest[:12]} ({size} bytes)"]
    lines = holders.lines()
    if lines:
        head.append("referenced by: " + "; ".join(lines))
    else:
        head.append("referenced by no run: the bytes are stored and no record names them")
    raw = holders.path.read_bytes()
    text, reason = _artifact_text(raw, value=bool(holders.tasks) and not holders.artifacts)
    name = holders.artifacts[0].name if holders.artifacts else ""
    run_id = holders.artifacts[0].run_id if holders.artifacts else None
    return ArtifactRead(head, name, holders.digest, size, run_id, text, reason)


def _artifact_text(raw: bytes, *, value: bool) -> tuple[str | None, str | None]:
    """The text to show for stored bytes, or None and the reason.

    A *value* is a traced task's serialized input or output: JSON comes
    back indented, and anything pickled comes back as its ``repr``.

    Examples:
        >>> from foundation.serialize import dumps
        >>> _artifact_text(dumps({"a0": 3.16}), value=True)
        ('{\\n "a0": 3.16\\n}', None)
        >>> _artifact_text(dumps((1, 2)), value=True)
        ('(1, 2)', None)
        >>> _artifact_text(b"\\x00\\x01", value=False)
        (None, 'looks binary; read_artifact only reads text')
    """
    if value:
        try:
            decoded = loads(raw)
        except Exception:  # not a tagged value after all: show the bytes
            decoded = raw
        if not isinstance(decoded, bytes):
            if raw[:2] == b"J\n":
                return json.dumps(decoded, indent=1, ensure_ascii=False), None
            return repr(decoded), None
        raw = decoded
    if b"\x00" in raw[:8192]:
        return None, "looks binary; read_artifact only reads text"
    return raw.decode("utf-8", errors="replace"), None


PERMANENT_STATES = (LifecycleState.PROMOTED, LifecycleState.ARCHIVED)


def sessions_summary(ws: Workspace, *, limit: int | None = None) -> dict[str, Any]:
    """The sessions that created runs, newest first, plus the unstamped count.

    This is ``slab sessions`` and the MCP ``list_sessions`` tool: it
    answers "which conversation produced which runs" so a user can promote a
    whole session without collecting run ids. Runs created before session
    stamping, or by a client that sets none, carry no session; they are
    counted once rather than listed.
    """
    summaries = ws.runs.list_sessions(limit=limit)
    # Unlimited on purpose: a row limit must not make the unstamped tally,
    # which is every run minus every stamped one, look larger than it is.
    stamped = sum(s.runs for s in ws.runs.list_sessions())
    return {
        "sessions": [
            {
                "session": s.session,
                "runs": s.runs,
                "states": s.states,
                "breakdown": s.breakdown(),
                "newest_at": s.newest_at.isoformat(),
            }
            for s in summaries
        ],
        "unstamped": len(ws.runs.list_runs()) - stamped,
    }


def promote_session(
    ws: Workspace,
    session: str,
    *,
    reason: str | None = None,
    force: bool = False,
    actor: str = "user",
) -> dict[str, Any]:
    """Promote every run one client session created; report each outcome.

    This is ``slab promote --session`` and the MCP ``promote_session``
    tool. *session* is a full session id or a unique prefix. Every stamped run
    is considered and reported:

    - ``verified`` runs are promoted;
    - ``promoted``/``archived`` runs are reported as already permanent;
    - unverified runs are skipped unless *force* is set;
    - failed runs are skipped even under *force*, because a bulk command must
      not sweep failures into permanence (promote such a run by its own id);
    - expired runs are skipped; the transition is illegal.

    Each run commits on its own, so a partial failure is safe to rerun: every
    outcome is idempotent. ``complete`` is True when no run was skipped.

    Raises:
        SessionNotFoundError: No run carries the session.
        AmbiguousSessionError: The prefix matches several sessions.
    """
    resolved = ws.runs.resolve_session(session)
    why = reason if reason else f"promoted with session {resolved}"
    # Oldest first, so the report reads in the order the session worked.
    outcomes = [
        _promote_one(ws, run, reason=why, force=force, actor=actor)
        for run in reversed(ws.runs.list_runs(session=resolved))
    ]
    counted = {kind: sum(1 for o in outcomes if o["outcome"] == kind) for kind in _OUTCOMES}
    return {
        "session": resolved,
        "reason": why,
        "outcomes": outcomes,
        "complete": counted["skipped"] == 0 and bool(outcomes),
        **counted,
    }


_OUTCOMES = ("promoted", "already", "skipped")


def _promote_one(
    ws: Workspace, run: Run, *, reason: str, force: bool, actor: str
) -> dict[str, Any]:
    """Decide and apply one run's fate inside a session promote."""
    outcome, detail = _verdict(run, force=force)
    if outcome == "promoted":
        try:
            ws.runs.transition(
                run.id,
                LifecycleState.PROMOTED,
                actor=actor,
                reason=reason,
                force=force,
                expected=run.state,
            )
        except IllegalTransitionError as e:
            # Someone else moved the run between the listing and the write.
            outcome, detail = "skipped", str(e)
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state.value,
        "status": run.status.value,
        "outcome": outcome,
        "detail": detail,
    }


def run_commands(ws: Workspace, run_id: str) -> list[dict[str, Any]]:
    """The external commands a run's tasks resolved, one entry per distinct command.

    A traced task that drives a binary stamps the resolved command, the
    setup lines, and the detected version into its recipe (``extra``):
    the ``lammps`` and ``qe`` engines under ``relax`` and
    ``single_point``, ``run_lammps``, ``build_structure`` for atomsk, and
    ``train_potential`` for gracemaker. This collects them, so a
    transcript can say what a run ran. Tasks that share one command
    collapse into one entry: ``tasks`` counts them, ``cache_hits`` says
    how many never executed it, and ``seq`` is the first. A LAMMPS entry
    carries the KOKKOS switches the command asks for, because SLAB adds
    none. The exact argument vector adds the runner's own flags (``-in``
    and ``-log`` for a script, ASE's ``-echo``, ``-screen``, and ``-log``
    for the engine). ``command`` is the line that ran: for a command with
    ``{ntasks}``, ``{threads}``, or ``{gpus}`` that is the filled line
    from the recipe's ``provenance``, and ``template`` is the line as
    written, which is the cache identity.
    """
    from slab.lammps import kokkos_switches

    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    for task in ws.runs.list_tasks(run_id):
        extra = task.recipe.get("extra") if isinstance(task.recipe, dict) else None
        if not isinstance(extra, dict) or not extra.get("command"):
            continue
        template = str(extra["command"])
        provenance = extra.get("provenance")
        command = template
        if isinstance(provenance, dict) and provenance.get("command"):
            command = str(provenance["command"])
        setup = [str(line) for line in extra.get("setup") or []]
        engine = extra.get("engine") or extra.get("builder")
        key = (engine, command, tuple(setup))
        entry = entries.get(key)
        if entry is None:
            entry = {
                "run_id": run_id,
                "seq": task.seq,
                "task": task.name,
                "tasks": 0,
                "cache_hits": 0,
                "engine": engine,
                "command": command,
                "setup": setup,
                "version": extra.get("version"),
            }
            if template != command:
                entry["template"] = template
            if extra.get("build"):
                entry["build"] = extra["build"]
            if engine == "lammps":
                entry["kokkos"] = kokkos_switches(command)
            entries[key] = entry
        entry["tasks"] += 1
        if task.cache_hit:
            entry["cache_hits"] += 1
    return list(entries.values())


def _verdict(run: Run, *, force: bool) -> tuple[str, str]:
    """What a session promote does with one run, and why (pure).

    Examples:
        >>> _verdict(Run(state="verified"), force=False)[0]
        'promoted'
        >>> _verdict(Run(state="quarantined", status="completed"), force=False)
        ('skipped', 'not verified: pass --force to promote it anyway')
        >>> _verdict(Run(state="quarantined", status="failed"), force=True)[0]
        'skipped'
        >>> _verdict(Run(state="quarantined", status="running"), force=True)
        ('skipped', 'running run: promote it by its own id if you mean to')
        >>> _verdict(Run(state="promoted"), force=False)
        ('already', 'already permanent')
    """
    if run.state in PERMANENT_STATES:
        return "already", "already permanent"
    if run.state is LifecycleState.VERIFIED:
        return "promoted", "checks passed"
    if run.state is LifecycleState.QUARANTINED:
        if run.status is not ExecutionStatus.COMPLETED:
            # Failed, still running, or never started: a bulk sweep must
            # not make any of these permanent.
            return "skipped", f"{run.status.value} run: promote it by its own id if you mean to"
        if not force:
            return "skipped", "not verified: pass --force to promote it anyway"
        return "promoted", "forced: never verified"
    return "skipped", f"{run.state.value}: nothing to promote"


RETIRE_MODES = ("keep", "expire", "purge")


def retire_session(
    ws: Workspace,
    session: str,
    *,
    keep: Sequence[str] = (),
    mode: str = "expire",
    reason: str | None = None,
    actor: str = "system",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promote what a session's finish cites; expire what it did not.

    This is ``slab retire``, the MCP ``retire_session`` tool, and what
    Mason does after a root session's ``finish``. *session* is a full id
    or a unique prefix. *keep* lists the cited runs (full ids or unique
    prefixes); a kept run may belong to another session, because a
    campaign cites the anchors it built on earlier. *mode* is what
    happens to this session's uncited runs: ``keep`` leaves them to the
    TTL sweep, ``expire`` moves them to ``expired``, and ``purge`` also
    deletes their rows, their unshared bytes, and the scratch directories
    they made under ``[paths] scratch``.

    Outcomes per kept run: ``promoted``, ``already`` (permanent before
    the call), or ``skipped`` with a detail (``not verified``,
    ``failed``, ``expired``, ``running``, ``unknown id``). A kept run is
    never forced. An uncited run of the session is expired when it is
    quarantined or verified and not running; a running run, a permanent
    run, and an already expired run are reported under ``skipped``.
    Every write is compare-and-swap guarded, so a run that changes state
    under the call is skipped rather than moved on stale information.

    Every outcome is idempotent: a second call reports ``already`` for
    the kept runs and expires nothing new. *dry_run* computes the same
    report and writes nothing.

    The report carries the retention numbers a benchmark records:
    ``runs_total`` (the session's runs), ``runs_promoted``,
    ``runs_expired``, and the bytes reachable from each group
    (``bytes_total``, ``bytes_promoted``, ``bytes_expired``).

    Raises:
        AmbiguousSessionError: The prefix matches several sessions.
    """
    if mode not in RETIRE_MODES:
        raise ValueError(f"retire mode must be keep, expire, or purge, not {mode!r}")
    try:
        resolved = ws.runs.resolve_session(session)
        session_runs = list(reversed(ws.runs.list_runs(session=resolved)))
    except SessionNotFoundError:
        # A session that launched nothing has nothing to expire; it may
        # still cite runs from earlier sessions.
        resolved, session_runs = session, []
    why = reason if reason else f"cited by the finish of session {resolved}"

    kept: list[dict[str, Any]] = []
    kept_ids: set[str] = set()
    for value in keep:
        try:
            run_id = ws.runs.resolve(str(value))
        except FoundationError as e:
            kept.append(
                {
                    "id": str(value),
                    "name": None,
                    "state": None,
                    "status": None,
                    "outcome": "skipped",
                    "detail": f"unknown id: {e}",
                }
            )
            continue
        if run_id in kept_ids:
            continue
        kept_ids.add(run_id)
        run = ws.runs.get(run_id)
        outcome, detail = _keep_verdict(run)
        if outcome == "promoted" and not dry_run:
            try:
                ws.runs.transition(
                    run.id,
                    LifecycleState.PROMOTED,
                    actor=actor,
                    reason=why,
                    expected=run.state,
                )
            except IllegalTransitionError as e:
                outcome, detail = "skipped", str(e)
        kept.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state.value,
                "status": run.status.value,
                "outcome": outcome,
                "detail": detail,
            }
        )

    expired: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for run in session_runs:
        if run.id in kept_ids:
            continue
        if mode == "keep":
            continue
        entry = {
            "id": run.id,
            "name": run.name,
            "state": run.state.value,
            "status": run.status.value,
        }
        if run.status is ExecutionStatus.RUNNING:
            skipped.append(entry | {"detail": "running: a live process owns it"})
            continue
        if run.state in PERMANENT_STATES:
            skipped.append(entry | {"detail": "already permanent"})
            continue
        if run.state is LifecycleState.EXPIRED:
            skipped.append(entry | {"detail": "already expired"})
            continue
        if not dry_run:
            try:
                ws.runs.transition(
                    run.id,
                    LifecycleState.EXPIRED,
                    actor=actor,
                    reason=f"uncited by the finish of session {resolved}",
                    expected=run.state,
                )
            except IllegalTransitionError as e:
                skipped.append(entry | {"detail": str(e)})
                continue
        expired.append(entry)

    def reachable(ids: Iterable[str]) -> set[str]:
        digests: set[str] = set()
        for run_id in ids:
            digests |= _reachable_hashes(ws.runs, ws.runs.get(run_id))
        return digests

    def size_of(digests: set[str]) -> int:
        return sum(ws.artifacts.size(d) for d in digests if ws.artifacts.has(d))

    permanent = [k["id"] for k in kept if k["outcome"] in ("promoted", "already")]
    bytes_promoted = size_of(reachable(permanent))
    bytes_expired = size_of(reachable(e["id"] for e in expired))
    bytes_total = size_of(reachable(run.id for run in session_runs))

    purged: dict[str, Any] = {}
    if mode == "purge" and expired:
        report = ws.purge_expired(dry_run=dry_run, only=[e["id"] for e in expired])
        # The purged runs' scratch directories go with their rows: nothing
        # will ever name them again.
        scratch = sweep_scratch(ws, dry_run=dry_run, only=report.deleted)
        purged = {
            "deleted": report.deleted,
            "freed_bytes": report.freed_bytes,
            "scratch_removed": [entry["path"] for entry in scratch.removed],
        }

    return {
        "session": resolved,
        "mode": mode,
        "dry_run": dry_run,
        "reason": why,
        "kept": kept,
        "expired": expired,
        "purged": purged,
        "skipped": skipped,
        "complete": all(k["outcome"] != "skipped" for k in kept),
        "runs_total": len(session_runs),
        "runs_promoted": sum(1 for k in kept if k["outcome"] == "promoted"),
        "runs_expired": len(expired),
        "bytes_total": bytes_total,
        "bytes_promoted": bytes_promoted,
        "bytes_expired": bytes_expired,
    }


def _keep_verdict(run: Run) -> tuple[str, str]:
    """What a retire does with one cited run, and why (pure). Never forces.

    Examples:
        >>> _keep_verdict(Run(state="verified"))
        ('promoted', 'checks passed')
        >>> _keep_verdict(Run(state="promoted"))
        ('already', 'already permanent')
        >>> _keep_verdict(Run(state="quarantined", status="completed"))
        ('skipped', 'not verified')
        >>> _keep_verdict(Run(state="quarantined", status="failed"))
        ('skipped', 'failed')
        >>> _keep_verdict(Run(state="quarantined", status="running"))
        ('skipped', 'running')
        >>> _keep_verdict(Run(state="expired"))
        ('skipped', 'expired')
    """
    if run.state in PERMANENT_STATES:
        return "already", "already permanent"
    if run.status is ExecutionStatus.RUNNING:
        return "skipped", "running"
    if run.state is LifecycleState.VERIFIED:
        return "promoted", "checks passed"
    if run.state is LifecycleState.EXPIRED:
        return "skipped", "expired"
    if run.status is ExecutionStatus.FAILED:
        return "skipped", "failed"
    return "skipped", "not verified"


def launch_script(
    root: Path,
    script: str | os.PathLike[str],
    *,
    name: str | None = None,
    intent: str | None = None,
    session: str | None = None,
    argv: tuple[str, ...] = (),
    capture_output: bool = False,
    reservation: Reservation | str | None = None,
    dry_run: bool = False,
    from_run: str | None = None,
) -> dict[str, Any]:
    """Execute a workflow script inside a fresh run context; return the outcome.

    This is ``slab run`` and the MCP ``launch_workflow`` tool: the script is
    plain Python with ``@task`` calls and ``@check`` declarations — the runner
    supplies the workspace and the run context, so scripts carry zero
    ceremony. Scripts that manage their own ``Workspace.start_run`` should be
    executed with plain ``python`` instead (nesting is refused with a hint).

    *session* stamps the run with the client session that launched it; when
    omitted, ``$SLAB_SESSION`` applies (see
    :func:`foundation.runtime.resolve_session_id`). *reservation* is the
    slice a session process checked out for this run; the run claims it
    (see :meth:`foundation.runtime.Workspace.start_run`), and a claim that
    is refused raises :class:`~foundation.errors.ResourcesError` before
    any run exists.

    The result dict carries ``run_id``, final ``state``/``status``, check
    counts, and — with ``capture_output=True`` — everything the script printed
    (used by the MCP server, whose stdout is the protocol channel). If the
    script raised, the run's structured ``failure`` record (trimmed traceback
    and diagnostic notes, :func:`foundation.errors.failure_record`) is included; a
    raw ``traceback`` is the fallback for failures the run itself never saw
    (runner machinery).

    With *dry_run* the script rehearses: see :func:`dry_run_script`, which
    this calls and whose report it returns. *reservation* is not used
    then, because a rehearsal writes no store, and *root* is only read,
    apart from the dry-run record of a failed ``run_lammps``. *from_run*
    (with *dry_run* only) rehearses against that run's cached task
    results in *root*.
    """
    script_path = Path(script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"no such workflow script: {script_path}")
    if from_run is not None and not dry_run:
        raise FoundationError(
            "from_run rehearses a script against a run's cached results; pass it with "
            "dry_run, or re-verify a run with reverify_run"
        )
    if dry_run:
        return dry_run_script(
            script_path,
            name=name,
            intent=intent,
            session=session,
            argv=argv,
            capture_output=capture_output,
            root=root,
            from_run=from_run,
        )

    try:
        workspace = Workspace(root)
    except Exception as e:
        raise StorageError(f"cannot open workspace at {root}: {e}") from e
    with workspace as ws:
        run_id, error, buffer = _run_script_in(
            ws,
            script_path,
            name=name,
            intent=intent,
            session=session,
            argv=argv,
            capture_output=capture_output,
            reservation=reservation,
        )
        run = ws.runs.get(run_id)
        checks = ws.runs.list_check_results(run_id)
        result: dict[str, Any] = run_summary(run) | {
            "run_id": run.id,
            "checks_passed": sum(1 for c in checks if c.passed),
            "checks_total": len(checks),
            "tasks_recorded": len(ws.runs.list_tasks(run_id)),
        }
    if run.failure is not None:
        result["failure"] = run.failure
    elif error is not None:
        # The failure escaped the run context (runner machinery, storage):
        # report the raw traceback so the evidence still reaches the caller.
        result["traceback"] = error
    if capture_output:
        result["output"] = buffer.getvalue()
    return result


def _run_script_in(
    ws: Workspace,
    script_path: Path,
    *,
    name: str | None,
    intent: str | None,
    session: str | None,
    argv: tuple[str, ...],
    capture_output: bool,
    reservation: Reservation | str | None = None,
    dry_run: bool = False,
    replay: Replay | None = None,
    source: Workspace | None = None,
) -> tuple[str, str | None, io.StringIO]:
    """Run *script_path* inside a fresh run of *ws*; return its id, the raw
    traceback of a failure the run never saw, and the captured output.

    With *replay* every task call is answered from another run's results,
    and a call that has none raises :class:`~foundation.errors.ReplayError`
    out of this function, even when the script caught it.
    """
    buffer = io.StringIO()
    error: str | None = None
    run_id: str | None = None
    # The interpreter state is touched only once the workspace is open, so
    # a failed open (in a long-lived MCP process) leaves argv and path as
    # they were.
    old_argv = sys.argv
    sys.argv = [str(script_path), *argv]
    sys.path.insert(0, str(script_path.parent))
    try:
        # capture wraps the whole run context: @check hooks evaluate at
        # context exit, and their prints must not reach the real stdout
        # (under MCP, stdout is the protocol channel).
        with ExitStack() as stack:
            if capture_output:
                stack.enter_context(redirect_stdout(buffer))
                stack.enter_context(redirect_stderr(buffer))
            with ws.start_run(
                name=name or script_path.stem,
                intent=intent,
                session=session,
                reservation=reservation,
                dry_run=dry_run,
                replay=replay,
                source=source,
            ) as active:
                run_id = active.id
                # The script is the run's recompute root and its own
                # best explanation. Kept by name, so show_run lists it
                # and read_artifact reads it: one real lead searched
                # three filesystems for a script the run record held.
                active.keep(script_path.name, script_path, role=ArtifactRole.INPUT)
                try:
                    _execute_script(script_path)
                    if replay is not None and replay.refusal is not None:
                        raise replay.refusal  # the script caught it; it still stands
                except (KeyError, TypeError, IndexError, AttributeError) as e:
                    # A shape mistake after a completed run_lammps: the
                    # note names the keys the result holds, so the next
                    # script reads them instead of remembering them.
                    note = _result_shape_note(ws, active.id)
                    if note:
                        e.add_note(note)
                    raise
    except NestedRunError:
        raise FoundationError(
            f"{script_path.name} manages its own runs (it calls start_run); "
            f"execute it with plain 'python {script_path.name}' instead of "
            f"'slab run'"
        ) from None
    except (ResourcesError, ReplayError):
        raise
    except Exception:
        error = traceback.format_exc(limit=8)
    finally:
        sys.argv = old_argv
        sys.path.remove(str(script_path.parent))

    if run_id is None:
        # The run never started (unwritable database, storage failure...):
        # surface the real cause instead of pretending a run exists.
        raise StorageError(f"could not start a run for {script_path.name}:\n{error}")
    return run_id, error, buffer


DRY_RUN_MARKER = "dry run:"
"""The line ``slab run --dry-run`` prints before its JSON report."""
_DRY_RUN_MARKER_LINE = re.compile(rf"^{re.escape(DRY_RUN_MARKER)}$", re.MULTILINE)

DRY_RUN_CHECK_FAILED = "expected in a dry run: LAMMPS integrated no steps"
"""The reading of a check that failed on a rehearsal's empty result."""
DRY_RUN_CHECK_PASSED = "passed on no data; not evidence"
"""The reading of a check that passed on a rehearsal's empty result."""
#: Prepended to a real launch of a script text this session never dry-ran.
NO_DRY_RUN_WARNING = (
    "warning: no dry run of this script text in this session; a dry run costs one "
    "LAMMPS start and catches script and post-processing errors before the MD leg"
)
WORKSPACE_ENV = "SLAB_WORKSPACE"


@contextmanager
def _throwaway_workspace() -> Iterator[Workspace]:
    """A fresh workspace under ``[paths] scratch`` (else the temp dir), removed after.

    While it is open, ``$SLAB_WORKSPACE`` names it, so a child a script
    starts lands there too.
    """
    scratch = scratch_root()
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    throwaway = Path(tempfile.mkdtemp(prefix="slab-dry-run-", dir=scratch))
    workspace_before = os.environ.get(WORKSPACE_ENV)
    os.environ[WORKSPACE_ENV] = str(throwaway)
    try:
        with Workspace(throwaway) as ws:
            yield ws
    finally:
        if workspace_before is None:
            os.environ.pop(WORKSPACE_ENV, None)
        else:
            os.environ[WORKSPACE_ENV] = workspace_before
        shutil.rmtree(throwaway, ignore_errors=True)


def _replaying(scratch: Workspace, source: Workspace, run_id: str) -> Replay:
    """Point *scratch* at *source*'s artifacts for reads; return the replay of *run_id*."""
    scratch.artifacts = ReadThroughStore(scratch.artifacts.root, fallback=source.artifacts)
    return Replay(source, run_id)


def dry_run_script(
    script: str | os.PathLike[str],
    *,
    name: str | None = None,
    intent: str | None = None,
    session: str | None = None,
    argv: tuple[str, ...] = (),
    capture_output: bool = False,
    root: str | os.PathLike[str] | None = None,
    from_run: str | None = None,
) -> dict[str, Any]:
    """Rehearse a workflow script without a real run: ``slab run --dry-run``.

    The script runs to its end, or to its first exception, inside a run
    opened with ``dry_run=True`` in a throwaway workspace: a fresh
    directory under ``[paths] scratch`` (else the platform temp dir) that
    is removed when the rehearsal ends, whatever happened. So nothing
    lands in the real store or its cache, no reservation is claimed, and
    every ``run_lammps`` call runs with its loops emptied: every ``run``
    line becomes ``run 0`` and every ``minimize`` gets zero iterations,
    so the data file, every pair style, fix, and compute are set up,
    every command runs in order, each loop prints one thermo row and one
    loop line, and no step is integrated. The result keeps its real
    shape, so the Python after the call is exercised. While the script
    runs, ``$SLAB_WORKSPACE`` names the throwaway
    workspace, so a child the script starts lands there too. *source* is
    the real workspace, and a ``run:<id>/<name>`` entry in ``files=``
    reads the artifact from it, because the throwaway store holds no
    earlier run.

    With *from_run* (a run id in the workspace at *root*) no engine
    starts: every task call takes that run's cached result instead (see
    :class:`foundation.runtime.Replay`), so the Python after each call
    and every check run on real data. A task call the run has no result
    for is refused with :class:`~foundation.errors.ReplayError`.

    The report says what the rehearsal found:

    * ``dry_run``: True.
    * ``from_run``: the run whose results answered, or None.
    * ``reached_end``: the script ran to its last line.
    * ``traceback``: the failure record's traceback, or None.
    * ``lammps``: one entry per ``run_lammps`` call, in order, with the
      ``label`` and an ``outcome`` of ``setup ok``, the first error line
      of the failure (an ``ERROR`` line, a Kokkos abort, and their kin,
      see :func:`slab.lammps.error_lines`), or ``replayed from run <id>``.
    * ``checks``: every ``@check`` with ``name``, ``passed``, ``message``,
      and ``reading``, which says what the outcome is worth. A check
      that raised reads ``check raised <Exception>: <text>``, because
      that is a bug in the check. On an empty result, a failed check
      reads as expected and a passed one as not evidence.
    * ``outputs``: the names of the files ``run_lammps`` kept, which are
      the names a real run would keep.
    * ``record``: the dry-run record of the failed ``run_lammps`` calls,
      or None. It holds ``id`` (``dry-<stamp>``), ``path``, and ``files``.
    * ``output``: what the script printed, with *capture_output*.

    The throwaway workspace is gone when the report returns, so the files
    a failed ``run_lammps`` kept (``{label}-failed.in``, ``.log``,
    ``.screen``, and what the script wrote) are copied first into a
    dry-run record in *root*, the real workspace: see
    :func:`keep_dry_run_record`. Without *root* no record is kept.
    """
    script_path = Path(script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"no such workflow script: {script_path}")
    with ExitStack() as stack:
        real = _existing_workspace(root)
        if real is not None:
            stack.enter_context(real)
        if from_run is not None and real is None:
            raise FoundationError("from_run needs the workspace that holds the run")
        ws = stack.enter_context(_throwaway_workspace())
        replay = None if from_run is None else _replaying(ws, real, str(from_run))
        run_id, error, buffer = _run_script_in(
            ws,
            script_path,
            name=name,
            intent=intent,
            session=session,
            argv=argv,
            capture_output=capture_output,
            dry_run=True,
            replay=replay,
            source=real,
        )
        report = _dry_run_report(
            ws, run_id, error, from_run=None if replay is None else replay.run_id
        )
        report["record"] = None
        if root is not None:
            report["record"] = keep_dry_run_record(
                Path(root), ws, run_id, script=script_path, session=session
            )
    if capture_output:
        report["output"] = buffer.getvalue()
    return report


def _existing_workspace(root: str | os.PathLike[str] | None) -> Workspace | None:
    """The workspace at *root* when it already holds a run store, else None.

    A dry run reads earlier runs from it and must not create one where
    none was.
    """
    if root is None or not (Path(root).expanduser() / "runs.db").is_file():
        return None
    try:
        return Workspace(root)
    except (FoundationError, sqlite3.Error, OSError):
        return None


def _result_shape_note(ws: Workspace, run_id: str) -> str | None:
    """One line naming the keys of the run's last completed ``run_lammps`` result."""
    for record in reversed(ws.runs.list_tasks(run_id)):
        if record.name != "run_lammps" or record.status is not ExecutionStatus.COMPLETED:
            continue
        digest = record.outputs.get("return[0]")
        if digest is None:
            return None
        try:
            result = loads(ws.artifacts.get(digest).read_bytes())
        except Exception:
            return None
        if not isinstance(result, dict):
            return None
        return (
            f"run_lammps result keys: {shape_line(result)}; the lammps-scripting "
            f"skill section 2 has the shape"
        )
    return None


def shape_line(value: dict[str, Any]) -> str:
    """The keys of a dict one level deep, as a reader scans them.

    Examples:
        >>> shape_line({"thermo": {"Step": 1, "Temp": 2.0}, "tables": [{"columns": [],
        ...             "n_rows": 1}, {}], "averages": {"fs.dat": {}}, "steps": 10,
        ...             "wall_time": "0:00:01"})
        'thermo{Step,Temp}, tables[2]{columns,n_rows}, averages{fs.dat}, steps, wall_time'
    """
    parts = []
    for key, item in value.items():
        if isinstance(item, dict):
            parts.append(f"{key}{{{','.join(str(k) for k in item)}}}")
        elif isinstance(item, list):
            inner = item[0] if item and isinstance(item[0], dict) else None
            keys = f"{{{','.join(str(k) for k in inner)}}}" if inner else ""
            parts.append(f"{key}[{len(item)}]{keys}")
        else:
            parts.append(str(key))
    return ", ".join(parts)


def _dry_run_report(
    ws: Workspace, run_id: str, error: str | None, *, from_run: str | None = None
) -> dict[str, Any]:
    """The dry-run report, read from the throwaway run's records."""
    run = ws.runs.get(run_id)
    failure = run.failure
    lammps: list[dict[str, str]] = []
    outputs: list[str] = []
    for task in ws.runs.list_tasks(run_id):
        if task.name != "run_lammps":
            continue
        label = _lammps_label(ws, task)
        if from_run is not None and task.cache_hit:
            outcome = f"replayed from run {from_run}"
        elif task.status is ExecutionStatus.COMPLETED:
            outcome = "setup ok"
        else:
            outcome = _first_error_line(task.failure, task.error)
        lammps.append({"label": label, "outcome": outcome})
        outputs.extend(
            ref.name
            for ref in ws.runs.list_artifacts(run_id, role=ArtifactRole.INTERMEDIATE)
            if ref.name.startswith(label) and ref.name not in outputs
        )
    # A rehearsal that emptied a LAMMPS loop gave its checks no data.
    no_data = from_run is None and bool(lammps)
    return {
        "dry_run": True,
        "from_run": from_run,
        "reached_end": run.status is ExecutionStatus.COMPLETED,
        "traceback": failure["traceback"] if failure is not None else error,
        "lammps": lammps,
        "checks": [
            _rehearsed_check(result, no_data=no_data, from_run=from_run)
            for result in ws.runs.list_check_results(run_id)
        ],
        "outputs": outputs,
    }


def _rehearsed_check(
    result: CheckResult, *, no_data: bool, from_run: str | None
) -> dict[str, Any]:
    """One check of a dry-run report; a check that raised names its line and the keys it read."""
    entry: dict[str, Any] = {
        "name": result.name,
        "passed": result.passed,
        "message": result.message,
        "reading": check_reading(result, no_data=no_data, from_run=from_run),
    }
    evidence = result.evidence or {}
    if entry["reading"].startswith("check raised") and evidence:
        entry["evidence"] = {k: evidence[k] for k in ("line", "keys") if k in evidence}
    return entry


def dry_run_clean(report: dict[str, Any]) -> bool:
    """Whether a dry run found nothing to fix before the real launch.

    Clean means the script reached its end, every ``run_lammps`` set up
    or was replayed, and no check raised. A check that failed is not a
    finding, because a rehearsal's empty result is expected to fail one.

    Examples:
        >>> dry_run_clean({"reached_end": True, "lammps": [{"outcome": "setup ok"}],
        ...                "checks": [{"reading": "expected in a dry run: ..."}]})
        True
        >>> dry_run_clean({"reached_end": True, "lammps": [],
        ...                "checks": [{"reading": "check raised KeyError: 'rows'"}]})
        False
    """
    return (
        bool(report.get("reached_end"))
        and all(
            entry.get("outcome") == "setup ok"
            or str(entry.get("outcome", "")).startswith("replayed from run")
            for entry in report.get("lammps") or []
        )
        and not any(
            str(check.get("reading", "")).startswith("check raised")
            for check in report.get("checks") or []
        )
    )


def check_reading(result: CheckResult, *, no_data: bool, from_run: str | None = None) -> str:
    """What one check outcome of a rehearsal is worth, in one phrase.

    A check that raised is a bug in the check whatever the data was, so
    it reads as the exception and never as an expected failure.

    Examples:
        >>> raised = CheckResult(run_id="r", name="f", kind="error", passed=False,
        ...                      message="check raised ZeroDivisionError: division by zero")
        >>> check_reading(raised, no_data=True)
        'check raised ZeroDivisionError: division by zero'
        >>> held = CheckResult(run_id="r", name="held", passed=True)
        >>> check_reading(held, no_data=True)
        'passed on no data; not evidence'
        >>> check_reading(held.model_copy(update={"passed": False}), no_data=True)
        'expected in a dry run: LAMMPS integrated no steps'
        >>> check_reading(held, no_data=False, from_run="ab12")
        'passed on the cached result of run ab12'
    """
    if result.kind == "error" and result.message.startswith("check raised"):
        return result.message
    verdict = "passed" if result.passed else "failed"
    if from_run is not None:
        return f"{verdict} on the cached result of run {from_run}"
    if not no_data:
        return verdict
    return DRY_RUN_CHECK_PASSED if result.passed else DRY_RUN_CHECK_FAILED


def reverify_run(
    ws: Workspace,
    run_id: str,
    script: str | os.PathLike[str],
    *,
    capture_output: bool = False,
) -> dict[str, Any]:
    """Run a script's checks again on a run's stored results: ``slab runs reverify``.

    The script runs in a throwaway workspace, and every task call takes
    the named run's own result (see :class:`foundation.runtime.Replay`),
    so no engine starts and no new run is recorded. The checks the
    script registers are then stored on the named run as a new
    verification pass. The run moves to verified when every check of
    the pass passed, and stays quarantined when one failed. The earlier
    passes stay in the store, and the script is kept on the run as
    ``reverify-<pass>-<name>``.

    A task call whose inputs differ from the run's (a changed LAMMPS
    script, say) is refused with :class:`~foundation.errors.ReplayError`
    naming the task, because a changed task needs a new computation.

    Returns ``run_id``, ``pass_no``, ``state``, ``checks_passed``,
    ``checks_total``, ``checks`` (see :func:`check_entry`),
    ``earlier_passes`` (see :func:`pass_tallies`), ``tasks_replayed``,
    ``script`` (the name the script was kept under), and ``output`` with
    *capture_output*.

    Raises:
        RunStateError: The run is not quarantined.
        ReplayError: A task call has no matching result in the run.
        FoundationError: The run did not complete, the script raised
            before its checks ran, or it registers no check.
    """
    script_path = Path(script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"no such workflow script: {script_path}")
    run = ws.runs.get(run_id)
    if run.state is not LifecycleState.QUARANTINED:
        raise RunStateError(run.id, run.state, "re-verify")
    if run.status is not ExecutionStatus.COMPLETED:
        raise FoundationError(
            f"run {run.id} is {run.status.value}; re-verify runs checks on the results of "
            f"a completed run, so relaunch the script (its finished tasks cache-hit)"
        )
    with _throwaway_workspace() as scratch:
        replay = _replaying(scratch, ws, run.id)
        replay_id, error, buffer = _run_script_in(
            scratch,
            script_path,
            name=run.name,
            intent=None,
            session=None,
            argv=(),
            capture_output=capture_output,
            replay=replay,
        )
        replayed = scratch.runs.get(replay_id)
        results = scratch.runs.list_check_results(replay_id)
    if replayed.status is not ExecutionStatus.COMPLETED:
        trace = replayed.failure["traceback"] if replayed.failure is not None else error
        raise FoundationError(
            f"{script_path.name} raised before its checks ran, so no pass was recorded on "
            f"run {run.id}:\n{str(trace or '').rstrip()}"
        )
    if not results:
        raise FoundationError(
            f"{script_path.name} registers no @check, so there is nothing to re-verify"
        )
    every = ws.runs.list_check_results(run.id, all_passes=True)
    earlier = pass_tallies(every)
    last_no = earlier[-1]["pass_no"] if earlier else 0
    pass_no = last_no + 1
    # A script that dropped a failing check is not a fix: the run stays
    # quarantined, and the answer names what the pass no longer covers.
    dropped = sorted({r.name for r in every if r.pass_no == last_no} - {r.name for r in results})
    stored = ws.runs.add_check_results(
        run.id, [result.model_copy(update={"pass_no": pass_no}) for result in results]
    )
    kept = f"reverify-{pass_no}-{script_path.name}"
    digest = ws.artifacts.put(script_path)
    ws.runs.add_artifact(
        run.id,
        name=kept,
        role=ArtifactRole.INPUT,
        hash=digest,
        size_bytes=ws.artifacts.size(digest),
    )
    passed = sum(1 for result in stored if result.passed)
    if passed == len(stored) and not dropped:
        # suppress: someone moved the run meanwhile; the state read below says where
        with suppress(IllegalTransitionError):
            ws.runs.transition(
                run.id,
                LifecycleState.VERIFIED,
                actor="reverify",
                reason=f"pass {pass_no}: {passed}/{len(stored)} assertions passed",
                expected=LifecycleState.QUARANTINED,
            )
    answer: dict[str, Any] = {
        "run_id": run.id,
        "pass_no": pass_no,
        "state": ws.runs.get(run.id).state.value,
        "checks_passed": passed,
        "checks_total": len(stored),
        "checks": [check_entry(result) for result in stored],
        "earlier_passes": earlier,
        "dropped": dropped,
        "tasks_replayed": len(replay.taken),
        "script": kept,
    }
    if capture_output:
        answer["output"] = buffer.getvalue()
    return answer


def reverify_lines(answer: dict[str, Any]) -> list[str]:
    r"""The reply to a re-verify, as the CLI, Mason, and MCP print it.

    Examples:
        >>> print("\n".join(reverify_lines({"run_id": "ab12", "pass_no": 2, "state": "verified",
        ...     "checks_passed": 1, "checks_total": 1, "tasks_replayed": 3,
        ...     "earlier_passes": [{"pass_no": 1, "passed": 0, "total": 1}],
        ...     "checks": [{"name": "drift", "passed": True, "message": "returned True"}]})))
        run ab12: pass 2 verified, 1/1 checks passed, 3 task result(s) replayed
        earlier: pass 1 0/1
          drift passed: returned True
        >>> print("\n".join(reverify_lines({"run_id": "ab12", "pass_no": 2,
        ...     "state": "quarantined", "checks_passed": 1, "checks_total": 1,
        ...     "tasks_replayed": 3, "earlier_passes": [{"pass_no": 1, "passed": 0, "total": 2}],
        ...     "dropped": ["melted"], "checks": [{"name": "drift", "passed": True,
        ...     "message": "returned True"}]})))
        run ab12: pass 2 quarantined, 1/1 checks passed, 3 task result(s) replayed
        earlier: pass 1 0/2
        dropped since pass 1: melted; the run stays quarantined until a pass covers it
          drift passed: returned True
    """
    lines = [
        f"run {answer['run_id']}: pass {answer['pass_no']} {answer['state']}, "
        f"{answer['checks_passed']}/{answer['checks_total']} checks passed, "
        f"{answer['tasks_replayed']} task result(s) replayed"
    ]
    if answer.get("earlier_passes"):
        lines.append(
            "earlier: "
            + ", ".join(
                f"pass {p['pass_no']} {p['passed']}/{p['total']}" for p in answer["earlier_passes"]
            )
        )
    if answer.get("dropped"):
        last = answer["earlier_passes"][-1]["pass_no"] if answer.get("earlier_passes") else 0
        lines.append(
            f"dropped since pass {last}: {', '.join(answer['dropped'])}; the run stays "
            f"quarantined until a pass covers it"
        )
    for check in answer["checks"]:
        verdict = "passed" if check["passed"] else "failed"
        lines.append(f"  {check['name']} {verdict}: {check['message']}")
        evidence = check.get("evidence")
        if evidence:
            lines.extend(f"    {line}" for line in evidence_lines(evidence))
    return lines


def evidence_lines(evidence: dict[str, Any]) -> list[str]:
    r"""A failed check's evidence as indented text lines.

    Examples:
        >>> for line in evidence_lines({"source": "def f():\n    return r['n'] > 0",
        ...     "keys": {"r": ["rows"]}, "raised": "KeyError: 'n'",
        ...     "line": "md.py:9: return r['n'] > 0"}):
        ...     print(line)
        raised KeyError: 'n' at md.py:9: return r['n'] > 0
        keys of r: rows
        source:
          def f():
              return r['n'] > 0
    """
    lines: list[str] = []
    if evidence.get("raised"):
        lines.append(f"raised {evidence['raised']} at {evidence.get('line')}")
    for var, keys in (evidence.get("keys") or {}).items():
        lines.append(f"keys of {var}: {', '.join(keys) or 'none'}")
    if evidence.get("source"):
        lines.append("source:")
        lines.extend(f"  {line}" for line in str(evidence["source"]).splitlines())
    return lines

def _lammps_label(ws: Workspace, task: Any) -> str:
    """The ``label`` a ``run_lammps`` task was called with, ``lammps`` by default."""
    label_hash = task.inputs.get("label")
    if label_hash is not None:
        with suppress(Exception):
            loaded = loads(ws.artifacts.get(label_hash).read_bytes())
            if loaded is not None:
                return str(loaded)
    return "lammps"


DRY_RUN_RECORDS = "dry-runs"
"""The directory of the real workspace that holds the dry-run records."""
DRY_RUN_PREFIX = "dry-"
"""The prefix of a dry-run record's id, which ``read_artifact`` takes as a run id."""
_DRY_RUN_RECORD_FILE = "record.json"


def keep_dry_run_record(
    root: Path,
    ws: Workspace,
    run_id: str,
    *,
    script: Path,
    session: str | None = None,
) -> dict[str, Any] | None:
    """Copy a dry run's failed ``run_lammps`` files into the real workspace.

    The record is ``<root>/dry-runs/<stamp>/``: the files under their
    artifact names, and ``record.json``, one row of kind ``dry_run``
    with the ``id`` (``dry-<stamp>``), the ``stamp``, the ``session``,
    the ``script``, and the ``files``. ``read_artifact`` opens the files
    by that id for the rest of the session, and ``slab purge`` removes
    the record. Returns ``{"id", "path", "files"}``, or None when no
    ``run_lammps`` failed. Never raises: a record that cannot be written
    is left out, and the report still says what failed.
    """
    try:
        failed = [
            _lammps_label(ws, task)
            for task in ws.runs.list_tasks(run_id)
            if task.name == "run_lammps" and task.status is not ExecutionStatus.COMPLETED
        ]
        refs = [
            ref
            for ref in ws.runs.list_artifacts(run_id, role=ArtifactRole.INTERMEDIATE)
            if any(ref.name.startswith(f"{label}-failed") for label in failed)
        ]
        if not refs:
            return None
        records = Path(root) / DRY_RUN_RECORDS
        records.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        for _ in range(100):
            stamp = f"{now:%Y%m%d-%H%M%S}-{os.urandom(2).hex()}"
            directory = records / stamp
            try:
                directory.mkdir()
                break
            except FileExistsError:
                continue
        else:
            return None
        files: list[str] = []
        for ref in refs:
            if not ws.artifacts.has(ref.hash):
                continue
            shutil.copyfile(ws.artifacts.get(ref.hash), directory / ref.name)
            files.append(ref.name)
        row = {
            "kind": "dry_run",
            "id": DRY_RUN_PREFIX + stamp,
            "stamp": stamp,
            "created_at": now.isoformat(timespec="seconds"),
            "session": session or os.environ.get("SLAB_SESSION") or None,
            "script": str(script),
            "files": files,
        }
        (directory / _DRY_RUN_RECORD_FILE).write_text(
            json.dumps(row, indent=1) + "\n", encoding="utf-8"
        )
    except Exception:
        return None
    return {"id": row["id"], "path": str(directory), "files": files}


def dry_run_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every dry-run record in the workspace at *root*, oldest first, with its row.

    A directory under ``dry-runs`` whose row is missing or unreadable is
    listed with an empty row, so purge still reaches it.
    """
    base = Path(root) / DRY_RUN_RECORDS
    if not base.is_dir():
        return []
    found: list[tuple[Path, dict[str, Any]]] = []
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        row: dict[str, Any] = {}
        with suppress(OSError, ValueError):
            loaded = json.loads((directory / _DRY_RUN_RECORD_FILE).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                row = loaded
        found.append((directory, row))
    return found


def dry_run_record(root: Path, record_id: str) -> tuple[Path, dict[str, Any]]:
    """The dry-run record *record_id* (``dry-<stamp>``): its directory and its row.

    Raises :class:`~foundation.errors.FoundationError` when no such record
    is kept, which is also the case after ``slab purge`` removed it.

    Examples:
        >>> import tempfile
        >>> dry_run_record(Path(tempfile.mkdtemp()), "dry-20260914-101500-ab12")
        Traceback (most recent call last):
        ...
        foundation.errors.FoundationError: no dry-run record dry-20260914-101500-ab12 in ...
    """
    stamp = record_id.removeprefix(DRY_RUN_PREFIX)
    for directory, row in dry_run_records(root):
        if directory.name == stamp and stamp:
            return directory, row
    raise FoundationError(
        f"no dry-run record {record_id} in {Path(root) / DRY_RUN_RECORDS}; slab purge "
        f"removes the records, and a dry run keeps one only when a run_lammps failed"
    )


def _first_error_line(failure: dict[str, Any] | None, error: str | None) -> str:
    r"""The first error line a failed ``run_lammps`` recorded, else its one-line error.

    An error line is one :func:`slab.lammps.is_error_line` takes: an
    ``ERROR`` line, a Kokkos abort, a loader failure, and their kin. A
    message that ends in a screen tail gives its first line and the
    tail's last line.

    Examples:
        >>> record = {"message": "LAMMPS failed (exit 1):\n  context: unfix nosuch\n"
        ...           "  ERROR: Could not find fix ID nosuch to delete (src/modify.cpp:1071)"}
        >>> _first_error_line(record, None)
        'ERROR: Could not find fix ID nosuch to delete (src/modify.cpp:1071)'
        >>> _first_error_line({"message": "LAMMPS failed (exit 1):\n  context: banner\n"
        ...                    "  Kokkos ERROR: Cuda execution space"}, None)
        'Kokkos ERROR: Cuda execution space'
        >>> _first_error_line({"message": "LAMMPS failed (exit 137):\n  screen tail:\n"
        ...                    "  LAMMPS (22 Jul 2025)\n  Killed"}, None)
        'LAMMPS failed (exit 137): the screen tail ends with: Killed'
        >>> _first_error_line(None, "LammpsScriptError: timed out")
        'LammpsScriptError: timed out'
    """
    from slab.lammps import is_error_line

    texts: list[str] = []
    if failure is not None:
        texts.append(str(failure.get("message", "")))
        texts.extend(str(note) for note in failure.get("notes", ()))
    for text in texts:
        for line in text.splitlines():
            if is_error_line(line.strip()):
                return line.strip()
    if failure is not None and failure.get("message"):
        lines = str(failure["message"]).splitlines()
        if "  screen tail:" in lines and lines[-1].strip() != "screen tail:":
            return f"{lines[0]} the screen tail ends with: {lines[-1].strip()}"
        return lines[0]
    return error or "failed"


def parse_dry_run_report(text: str) -> dict[str, Any] | None:
    r"""The JSON report after the last line that is only ``dry run:``, or None.

    Examples:
        >>> parse_dry_run_report('noise\ndry run:\n{"dry_run": true, "reached_end": true}\n')
        {'dry_run': True, 'reached_end': True}
        >>> parse_dry_run_report('a dry run: no marker line') is None
        True
        >>> parse_dry_run_report("no report") is None
        True
    """
    markers = list(_DRY_RUN_MARKER_LINE.finditer(text))
    if not markers:
        return None
    body = text[markers[-1].end() :]
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def runs_as_child(reservation: Reservation, *, sized: bool) -> bool:
    """Whether a launch must run as a child process to hold only its slice.

    A sized launch always does. An unsized one does when this process can
    see a gpu the reservation does not hold. In this process the run would
    read this process's envelope, see every gpu, and choose the gpu build
    it never reserved. The child runs under the reservation's envelope,
    with an empty ``CUDA_VISIBLE_DEVICES``, so it runs the plain build.

    Examples:
        >>> held = Reservation(host="n1", cpus=(0,), gpus=(), ntasks=1, threads=1, holder_pid=1)
        >>> os.environ.update(SLAB_CPUS="0,1", SLAB_GPUS="0,1")
        >>> runs_as_child(held, sized=False)
        True
        >>> os.environ["SLAB_GPUS"] = ""
        >>> runs_as_child(held, sized=False), runs_as_child(held, sized=True)
        (False, True)
        >>> for name in ("SLAB_CPUS", "SLAB_GPUS"):
        ...     del os.environ[name]
    """
    from slab.resources import envelope

    return sized or bool(set(envelope().gpus) - set(reservation.gpus))


def launch_child(
    root: Path,
    script: str | os.PathLike[str],
    *,
    reservation: Reservation,
    name: str | None = None,
    intent: str | None = None,
    session: str | None = None,
    argv: tuple[str, ...] = (),
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    wait: bool = True,
    log_path: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a workflow script as a child ``foundation run --reservation`` process.

    A sized launch never runs in the session's own process: the child
    takes the affinity mask and the GPU variables of *reservation*, and
    every rank it starts inherits them. The reservation is handed to the
    child's pid as soon as it exists, so a child that dies before claiming
    leaves a reservation whose holder is dead, which the next reap
    releases. The child's output goes to *log_path* (default
    ``<root>/launches/<reservation id>.log``).

    With ``wait=True`` the call blocks and returns the same result dict as
    :func:`launch_script` plus ``exit_code``, ``log``, and ``output`` (the
    log text). With ``wait=False`` it returns ``pid``, ``log``,
    ``reservation``, and ``command`` at once; the run appears in the store
    once the child claims the reservation. *env* is the base environment
    for the child (this process's when omitted); the envelope variables
    are added on top.

    With *dry_run* the child rehearses under the slice (``slab run
    --dry-run``): it takes the envelope, claims nothing, releases the
    reservation when it ends, and prints the report of
    :func:`dry_run_script` after a ``dry run:`` line. With ``wait=True``
    that report is returned, plus ``exit_code``, ``log``, and ``output``.
    """
    from slab.resources import env_for

    script_path = Path(script).resolve()
    if not script_path.exists():
        with Workspace(root) as ws:
            ws.runs.release_reservation(reservation.id)
        raise FileNotFoundError(f"no such workflow script: {script_path}")
    log = Path(log_path) if log_path is not None else Path(root) / "launches" / (
        f"{reservation.id}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "foundation.cli",
        "run",
        str(script_path),
        *argv,
        "--reservation",
        reservation.id,
        "-w",
        str(root),
    ]
    if name:
        command += ["--name", name]
    if intent:
        command += ["--intent", intent]
    if session:
        command += ["--session", session]
    if dry_run:
        command.append("--dry-run")
    child_env = {
        **(env if env is not None else os.environ),
        **env_for(reservation.envelope()),
        "PYTHONUNBUFFERED": "1",
    }
    # The child opens its own run and exports its own id; the parent's
    # must not stamp what the child makes.
    child_env.pop(RUN_ENV, None)
    try:
        with open(log, "ab") as handle:
            process = subprocess.Popen(
                command,
                cwd=None if cwd is None else os.fspath(cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=child_env,
            )
    except OSError as e:
        with Workspace(root) as ws:
            ws.runs.release_reservation(reservation.id)
        raise StorageError(f"could not start the child run: {e}") from e
    with Workspace(root) as ws, suppress(ResourcesError):
        # A refusal means the child claimed it already; nothing to hand over.
        ws.runs.transfer_reservation(reservation.id, holder_pid=process.pid)
    launched: dict[str, Any] = {
        "pid": process.pid,
        "log": str(log),
        "reservation": reservation.id,
        "command": shlex.join(command),
    }
    if not wait:
        return launched
    exit_code = process.wait()
    output = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    if dry_run:
        with Workspace(root) as ws, suppress(FoundationError):
            ws.runs.release_reservation(reservation.id)
        report = parse_dry_run_report(output)
        if report is None:
            tail = "\n".join(output.strip().splitlines()[-20:])
            raise StorageError(
                f"the child dry run (pid {process.pid}) exited {exit_code} without a "
                f"report; its log {log} ends with:\n{tail}"
            )
        report.update(launched)
        report["exit_code"] = exit_code
        report["output"] = output
        return report
    with Workspace(root) as ws:
        run = ws.runs.run_for_reservation(reservation.id)
        if run is None:
            ws.runs.release_reservation(reservation.id)
            tail = "\n".join(output.strip().splitlines()[-20:])
            raise StorageError(
                f"the child run (pid {process.pid}) exited {exit_code} before starting a "
                f"run; its log {log} ends with:\n{tail}"
            )
        checks = ws.runs.list_check_results(run.id)
        result: dict[str, Any] = run_summary(run) | {
            "run_id": run.id,
            "checks_passed": sum(1 for c in checks if c.passed),
            "checks_total": len(checks),
            "tasks_recorded": len(ws.runs.list_tasks(run.id)),
        }
    if run.failure is not None:
        result["failure"] = run.failure
    result.update(launched)
    result["exit_code"] = exit_code
    result["output"] = output
    return result


def _execute_script(script_path: Path) -> None:
    """runpy the script, taming SystemExit: the `sys.exit(main())` idiom is
    everyday Python and must not escape into typer or the MCP task group."""
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise ScriptExitError(f"script called sys.exit({e.code!r})") from None


# -- the task vocabulary ------------------------------------------------------


def _short_signature(sig: Any) -> str:
    """A signature a reader can scan: names and defaults, no annotations."""
    parts: list[str] = []
    for param in sig.parameters.values():
        kind = param.kind
        if kind is param.VAR_POSITIONAL:
            parts.append(f"*{param.name}")
            continue
        if kind is param.VAR_KEYWORD:
            parts.append(f"**{param.name}")
            continue
        if kind is param.KEYWORD_ONLY and "*" not in parts:
            parts.append("*")
        default = "" if param.default is param.empty else f"={param.default!r}"
        parts.append(f"{param.name}{default}")
    return ", ".join(parts)


def task_catalog() -> list[dict[str, str]]:
    """Name, signature, and one-line summary of every public foundation task.

    Public means not underscore-prefixed, defined in :mod:`foundation.tasks`
    itself, and documented: every symbol a workflow script may name. This
    is the ``list_tasks`` tool of the resident agent and of the MCP server.

    Examples:
        >>> names = [entry["name"] for entry in task_catalog()]
        >>> "relax" in names and "single_point" in names
        True
    """
    import inspect

    from foundation import tasks as _tasks

    entries: list[dict[str, str]] = []
    for name in sorted(vars(_tasks)):
        if name.startswith("_"):
            continue
        obj = getattr(_tasks, name)
        if not callable(obj) or not getattr(obj, "__doc__", None):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        # A traced task decorated with @task wraps the underlying function;
        # signature() returns the wrapped signature. Helpers imported into
        # the module are skipped by requiring the module of definition.
        if getattr(obj, "__module__", "") != _tasks.__name__:
            continue
        summary = (obj.__doc__ or "").strip().splitlines()[0]
        entries.append({"name": name, "signature": _short_signature(sig), "summary": summary})
    return entries


def describe_task(name: str) -> dict[str, str]:
    """The full signature and docstring of one task, or a ValueError naming the known ones."""
    name = name.strip()
    if not name:
        raise ValueError("describe_task requires 'name' — call list_tasks to see them")
    catalog = {entry["name"]: entry for entry in task_catalog()}
    if name not in catalog:
        raise ValueError(f"no task {name!r}; known: {', '.join(sorted(catalog))}")
    from foundation import tasks as _tasks

    function = getattr(_tasks, name)
    doc = (function.__doc__ or "").strip() or "(no docstring)"
    return {"name": name, "signature": catalog[name]["signature"], "doc": doc}


# -- progress, and waiting for a run ------------------------------------------


def tally_line(statuses: list[str], hits: int) -> str:
    """``3 completed, 1 running (2 cache hits)`` from a run's task statuses.

    Examples:
        >>> tally_line(["completed", "completed", "running"], 1)
        '2 completed, 1 running (1 cache hit)'
        >>> tally_line([], 0)
        'no tasks'
    """
    from collections import Counter

    tally = Counter(statuses)
    order = ("completed", "running", "failed")
    parts = [f"{tally[status]} {status}" for status in order if tally.get(status)]
    parts += [f"{n} {status}" for status, n in sorted(tally.items()) if status not in order]
    line = ", ".join(parts) or "no tasks"
    if hits:
        line += f" ({hits} cache hit{'s' if hits != 1 else ''})"
    return line


def run_progress(ws: Workspace, run_id: str) -> str:
    """One line of what a run has done so far: its task tally and its checks."""
    tasks = ws.runs.list_tasks(run_id)
    line = "tasks: " + tally_line(
        [task.status.value for task in tasks], sum(1 for task in tasks if task.cache_hit)
    )
    checks = ws.runs.list_check_results(run_id)
    if checks:
        line += f"; checks: {sum(1 for c in checks if c.passed)}/{len(checks)} passed"
    return line


def resolve_run(ws: Workspace, value: str, *, session: str | None = None) -> tuple[str, str]:
    """A run id or unique prefix, else the session's newest run of that name.

    The name is what a client remembers (a real transcript passed the
    script's name twice and was told no run matched), so it resolves too,
    with a note saying which run was taken. Without a session, only ids
    and prefixes resolve.
    """
    from foundation.errors import RunNotFoundError, SessionNotFoundError

    try:
        return ws.runs.resolve(value), ""
    except RunNotFoundError as not_found:
        if session is None:
            raise
        try:
            runs = ws.runs.list_runs(session=session, limit=200)
        except SessionNotFoundError:
            runs = []
        named = [run for run in runs if run.name == value]
        if not named:
            raise not_found from None
        run = named[0]  # newest first
        return run.id, (
            f"(resolved {value!r} by name to run {run.id[:10]}, this session's "
            f"newest run of that name)\n"
        )


#: The longest one wait_for_run call blocks. A 3-hour run needs one call;
#: a longer one needs a second, and the reply says the cap applied.
MAX_WAIT_S = 6 * 3600.0
#: The default wait when the caller names none.
DEFAULT_WAIT_S = 900.0
#: The longest gap between two reads of the run store while a wait blocks.
#: The first reads come sooner (1 s, doubling), so a short run is collected
#: at once and a long one costs one read every half minute.
WAIT_POLL_S = 30.0
#: Said with every still_running answer: waiting again is the right call.
STILL_RUNNING_LINE = "the run is alive and progressing; waiting again is the right call"
_LAMMPS_SCRATCH_PREFIX = "slab-lammps-script-"


def span_text(seconds: float) -> str:
    """A duration in the largest two units that fit.

    Examples:
        >>> span_text(42), span_text(190), span_text(4380), span_text(93_600)
        ('42s', '3m 10s', '1h 13m', '26h 0m')
    """
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60}s"
    return f"{total // 3600}h {total % 3600 // 60}m"


def lammps_live_log(run_id: str) -> Path | None:
    """The log of the LAMMPS script a running run is executing now, or None.

    ``run_lammps`` runs its script in a scratch directory whose owner
    marker names the run (:mod:`slab.scratch`). This looks under the
    ``[paths] scratch`` root and the platform temp directory, where a
    scratch made without that setting lands, and returns the newest
    ``log.lammps`` of a directory that the run owns.
    """
    from slab.scratch import read_owner

    roots = [root for root in (scratch_root(), Path(tempfile.gettempdir())) if root is not None]
    newest: tuple[float, Path] | None = None
    for root in dict.fromkeys(roots):
        with suppress(OSError):
            for directory in root.glob(f"{_LAMMPS_SCRATCH_PREFIX}*"):
                owner = read_owner(directory)
                if owner is None or owner.run_id != run_id:
                    continue
                log = directory / "log.lammps"
                with suppress(OSError):
                    mtime = log.stat().st_mtime
                    if newest is None or mtime > newest[0]:
                        newest = (mtime, log)
    return None if newest is None else newest[1]


def run_advance(run: Run, *, now: datetime | None = None) -> str:
    """How far a running run has got: the time since it started, and its MD step.

    The step comes from the live LAMMPS log of a ``run_lammps`` task
    (:func:`lammps_live_log`), read against the step the current ``run``
    command stops at. A run with no LAMMPS log gives the time alone.
    """
    from slab.outputs import lammps_run_progress

    began = run.started_at or run.created_at
    moment = now or utcnow()
    text = f"started {span_text((moment - began).total_seconds())} ago"
    log = lammps_live_log(run.id)
    if log is None:
        return text
    try:
        progress = lammps_run_progress(log.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return text
    if progress is None:
        # LAMMPS writes its log in blocks, so a young run shows no row yet.
        return f"{text}; the LAMMPS log holds no thermo row yet"
    step, target = progress["step"], progress["target"]
    if target is None:
        return f"{text}; the log shows step {step}"
    return f"{text}; the log shows step {step} of {target}"


def wait_for_run(
    root: Path,
    *,
    run_id: str | None = None,
    session: str | None = None,
    timeout_s: float = DEFAULT_WAIT_S,
    poll_s: float = WAIT_POLL_S,
    grace_s: float = 10.0,
    all_runs: bool = False,
) -> dict[str, Any]:
    """Block until a run finishes or the timeout passes; report where it stands.

    With *run_id* (an id, a unique prefix, or a run name the session
    created), waits for that run. Without it, waits on the running runs
    the *session* created (every running run when there is no session)
    and returns as soon as the first of them finishes, so a free slice
    is never left idle behind a longer run; *all_runs* waits until none
    is running instead. A background launch takes a moment to register
    its run, so an empty record gets *grace_s* before "nothing is
    running" counts as an answer.

    The wait is capped at :data:`MAX_WAIT_S`; ``timeout_s`` in the
    result is the wait applied and ``asked_s`` the wait asked. The store
    is read after 1 s, then at doubling gaps up to *poll_s*, and nothing
    else happens between reads. Every read first reaps the dead: a
    running run whose recorded process on this host is gone is marked
    failed (:meth:`Workspace.reap_dead`), so a wait never blocks on a
    hard-killed record.

    The result's ``outcome`` is one of ``finished`` (``run`` holds the
    :class:`Run` and ``progress`` its tally; an id-less wait adds
    ``also_finished``, the other runs that finished in the same interval,
    and ``running``, the ones still running), ``process_gone`` (the same
    keys; this call found the run's process dead and marked it failed),
    ``still_running`` (``running`` holds ``(Run, progress, liveness,
    advance)`` tuples: the liveness phrase from :func:`describe_liveness`
    and the advance phrase from :func:`run_advance`), ``none_running``
    (``runs`` holds the session's finished runs), or ``no_runs``.
    ``note`` says when a name was resolved to a run id. Callers format
    the text; the MCP server converts the runs with :func:`run_summary`.
    """
    import time

    from foundation.errors import SessionNotFoundError

    asked = max(0.0, float(timeout_s))
    timeout = min(asked, MAX_WAIT_S)
    deadline = time.monotonic() + timeout
    grace_until = time.monotonic() + min(grace_s, timeout)
    resolved: str | None = None
    note = ""
    watched: set[str] = set()  # the runs an id-less wait has seen running
    gap = min(1.0, poll_s)

    def still(ws: Workspace, runs: list[Run]) -> list[tuple[Run, str, str, str]]:
        return [(r, run_progress(ws, r.id), describe_liveness(r), run_advance(r)) for r in runs]

    while True:
        with Workspace(root) as ws:
            reaped = {r.id for r in ws.reap_dead(caller="wait_for_run")}
            if run_id:
                if resolved is None:
                    resolved, note = resolve_run(ws, run_id, session=session)
                run = ws.runs.get(resolved)
                if run.status.value != "running":
                    return {
                        "outcome": "process_gone" if run.id in reaped else "finished",
                        "note": note,
                        "run": run,
                        "progress": run_progress(ws, run.id),
                    }
                running = [run]
            else:
                try:
                    runs = ws.runs.list_runs(session=session, limit=50)
                except SessionNotFoundError:
                    runs = []
                running = [r for r in runs if r.status.value == "running"]
                ended = [r for r in runs if r.id in watched and r.status.value != "running"]
                if ended and not all_runs:
                    first = next((r for r in ended if r.id in reaped), ended[0])
                    return {
                        "outcome": "process_gone" if first.id in reaped else "finished",
                        "note": note,
                        "run": first,
                        "progress": run_progress(ws, first.id),
                        "also_finished": [r for r in ended if r.id != first.id],
                        "running": still(ws, running),
                    }
                # The grace covers a launch that has not registered yet; once
                # a watched run has finished, nothing more is on its way.
                if not running and (watched or time.monotonic() >= grace_until):
                    if not runs:
                        return {"outcome": "no_runs", "note": note, "runs": []}
                    return {"outcome": "none_running", "note": note, "runs": runs[:10]}
                watched.update(r.id for r in running)
            if time.monotonic() >= deadline:
                return {
                    "outcome": "still_running",
                    "note": note,
                    "timeout_s": timeout,
                    "asked_s": asked,
                    "running": still(ws, running),
                }
        time.sleep(min(gap, max(0.05, deadline - time.monotonic())))
        gap = min(gap * 2, poll_s)


# -- the scheduler ------------------------------------------------------------


def submit_job(
    root: Path,
    *,
    hpc: Any,
    command: str,
    name: str,
    partition: str | None = None,
    time_limit: str | None = None,
    session: str | None = None,
    project: Path | None = None,
    size: JobSize | None = None,
) -> dict[str, Any]:
    """Submit *command* as a SLURM batch job under the workspace's ``jobs/``.

    *hpc* is the ``[hpc]`` config (:class:`slab.config.HpcConfig`). The
    exported *session* stamps every run the job launches, so a batch
    result joins the session that asked for it; the export is explicit
    because a cluster may submit with ``--export=NONE``. The prologue
    changes into *project* so the payload runs where the client works,
    while the job files stay under the workspace for ``slab purge``.
    *size* (a :class:`slab.resources.JobSize`) sizes the job within the
    partition's declared node; a size that does not fit is refused before
    anything is written.
    """
    from slab.hpc import render_sbatch, submit

    chosen, _spec = hpc.resolve_partition(partition)
    # The job opens its own runs; the submitter's run id must not stamp
    # the scratch the job makes, or a sweep would take it once the
    # submitting run completes.
    prologue: list[str] = ["unset SLAB_RUN_ID"]
    if session:
        prologue.append(f"export SLAB_SESSION={shlex.quote(session)}")
    if project is not None:
        prologue.append(f"cd {shlex.quote(str(project))}")
    script = render_sbatch(
        command,
        job_name=name,
        partition=chosen,
        config=hpc,
        time_limit=time_limit,
        prologue=tuple(prologue),
        size=size,
    )
    job = submit(script, job_name=name, partition=chosen, directory=Path(root) / "jobs")
    return {
        "job_id": job.job_id,
        "job_name": job.job_name,
        "partition": job.partition,
        "script_path": str(job.script_path),
        "size": None if size is None else size.model_dump(),
    }


def job_status(job_id: str) -> dict[str, Any]:
    """The scheduler's state of one job: pending, running, completed, failed, ..."""
    from slab.hpc import job_state

    status = job_state(job_id)
    return {
        "job_id": status.job_id,
        "state": status.state.value,
        "raw": status.raw,
        "detail": status.detail,
    }


def cancel_job(job_id: str, *, workspace: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Cancel one job, and settle what it leaves behind in *workspace*.

    The scheduler is asked first (a no-op once the job has finished). Then,
    when a workspace root is given, every run stamped with *job_id* that is
    still ``running`` is marked failed with a reason, because its process
    died with the job and nothing else will advance the record. Failing a
    run releases the reservation it held, and the host's dead reservations
    are released with it. The scratch directories the failed runs made
    under ``[paths] scratch`` are removed, and the gpus the job's
    launches were refused on stop being excluded, because the exclusion
    lived as long as the job. Nothing is expired or purged.

    The summary also lists the machine memories written since the job's
    first run started. A dead job may have recorded a fact it never
    verified, so the operator reviews them; nothing is deleted here.

    Returns ``job_id``, ``runs_failed`` (id and name), ``reservations_released``
    (id and slice), ``scratch_removed`` (paths), ``gpus_cleared`` (gpu and
    host), and ``memories`` (name, description, when written).
    """
    from foundation import memory as memory_store
    from slab.hpc import cancel

    cancel(job_id)
    summary: dict[str, Any] = {
        "job_id": job_id,
        "cancel": "requested",
        "runs_failed": [],
        "reservations_released": [],
        "scratch_removed": [],
        "gpus_cleared": [],
        "memories": [],
    }
    if workspace is None:
        return summary
    with Workspace(workspace) as ws:
        summary["gpus_cleared"] = [
            {"gpu": row.gpu, "host": row.host} for row in ws.runs.expire_excluded_gpus([job_id])
        ]
        stamped = ws.runs.list_runs(job_id=job_id)
        doomed = [run for run in stamped if run.status == ExecutionStatus.RUNNING]
        if not doomed:
            return summary
        held_before = {r.id: r for r in ws.runs.list_reservations()}
        failed_ids: set[str] = set()
        for run in doomed:
            with suppress(IllegalStatusChangeError):  # it may end under us; fine
                failed = ws.runs.set_status(
                    run.id,
                    ExecutionStatus.FAILED,
                    error=f"job {job_id} cancelled by the operator; the process died with it",
                )
                summary["runs_failed"].append({"id": failed.id, "name": failed.name})
                failed_ids.add(failed.id)
        ws.release_dead()
        still = {r.id for r in ws.runs.list_reservations()}
        # Only the slices the job's own runs held: release_dead sweeps this
        # host's other dead reservations too, and they are not the cancel's.
        summary["reservations_released"] = [
            {"id": held.id, "slice": describe_resources(held.slice)}
            for held in held_before.values()
            if held.id not in still and held.run_id in failed_ids
        ]
        scratch = sweep_scratch(ws, only=failed_ids)
        summary["scratch_removed"] = [entry["path"] for entry in scratch.removed]
        # The cutoff is the job's first run, finished or not: a fact the job
        # recorded during a run that completed is as unverified as any other.
        earliest = min(run.created_at for run in stamped)
    summary["memories"] = [
        {
            "name": memory.name,
            "description": memory.description,
            "written_at": datetime.fromtimestamp(memory.path.stat().st_mtime, tz=UTC).isoformat(),
        }
        for memory in memory_store.written_since(earliest)
    ]
    return summary


def cancel_lines(summary: dict[str, Any]) -> list[str]:
    """The lines every surface prints for a :func:`cancel_job` summary.

    Examples:
        >>> cancel_lines({"job_id": "7", "runs_failed": [], "reservations_released": [],
        ...               "memories": []})
        ['cancel requested for job 7']
    """
    lines = [f"cancel requested for job {summary['job_id']}"]
    for run in summary.get("runs_failed", []):
        lines.append(
            f"failed  {run['id']}  {run['name']}  job {summary['job_id']} cancelled by the "
            f"operator; the process died with it"
        )
    for held in summary.get("reservations_released", []):
        lines.append(f"released {held['id']}  {held['slice']}")
    for path in summary.get("scratch_removed", []):
        lines.append(f"removed scratch {path}")
    for row in summary.get("gpus_cleared", []):
        lines.append(f"cleared  gpu {row['gpu']} on {row['host']}: no longer excluded")
    for memory in summary.get("memories", []):
        age = age_text(datetime.fromisoformat(memory["written_at"]))
        lines.append(
            f"memory  {memory['name']}  written {age} ago "
            f"('slab memory show {memory['name']}' to review)"
        )
    return lines
