"""The define-by-run runtime: workspaces, active runs, and verification hooks.

A :class:`Workspace` bundles the run database and the artifact store under one
root directory (``.slab/`` by convention). ``Workspace.start_run`` opens the
context that makes plain Python *traced* Python::

    ws = Workspace(".slab")
    with ws.start_run(name="si-relax", intent="baseline lattice constant") as run:
        relaxed, info = relax(atoms)          # @task calls are recorded

        @check
        def forces_converged():
            return converged(info["fmax"], below=0.05)

        run.keep("relaxed", relaxed)          # declare a terminal artifact

On exit the runtime completes the run, evaluates every registered ``@check``
hook, stores the results, and — only if every assertion passed and at least
one exists — moves the run ``quarantined -> verified``. A run with no checks
stays quarantined: verification is earned, never defaulted. If the block
raises, the run is marked ``failed`` (checks are skipped) and the exception
propagates; failed runs simply age out.
"""

from __future__ import annotations

import inspect
import os
import socket
import textwrap
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING, Any, overload

from foundation.artifacts import ArtifactStore
from foundation.checks import Assertion
from foundation.errors import (
    IllegalStatusChangeError,
    IllegalTransitionError,
    NestedRunError,
    NoActiveRunError,
    ReplayError,
    ResourcesError,
    RunStateError,
    failure_record,
)
from foundation.lifecycle import ExecutionStatus, LifecycleState
from foundation.models import (
    ArtifactRef,
    ArtifactRole,
    CheckResult,
    Reservation,
    Run,
    TaskRecord,
)
from foundation.retention import (
    DEFAULT_POLICY,
    GcReport,
    PurgeReport,
    RetentionPolicy,
    expire_due,
    gc,
    purge_expired,
    sweep_scratch,
)
from foundation.serialize import dumps
from foundation.store import SQLiteRunStore
from slab.hpc import JobState, SchedulerError, job_state
from slab.scratch import RUN_ENV, process_alive

if TYPE_CHECKING:
    from datetime import datetime

    from slab.resources import Budget

_CURRENT: ContextVar[ActiveRun | None] = ContextVar("slab_active_run", default=None)

_CheckFn = Callable[[], object]


SESSION_ENV = "SLAB_SESSION"


def resolve_session_id(explicit: str | None = None) -> str | None:
    """The session stamp for a new run: explicit argument > ``$SLAB_SESSION``.

    Foundation treats the value as an opaque string. Mason passes its chat's
    transcript stem; another client passes whatever identifies its own
    conversation. An empty value means "no session", so an exported but blank
    variable never stamps runs with the empty string.

    Examples:
        >>> import os
        >>> os.environ.pop("SLAB_SESSION", None) and None
        >>> resolve_session_id("chat-1")
        'chat-1'
        >>> resolve_session_id() is None
        True
        >>> os.environ["SLAB_SESSION"] = "from-env"
        >>> (resolve_session_id(), resolve_session_id("explicit"))
        ('from-env', 'explicit')
        >>> del os.environ["SLAB_SESSION"]
    """
    if explicit:
        return explicit
    return os.environ.get(SESSION_ENV) or None


def current_run() -> ActiveRun | None:
    """Return the run active in this context, or None.

    Examples:
        >>> current_run() is None
        True
    """
    return _CURRENT.get()


@overload
def check(fn: _CheckFn) -> _CheckFn: ...
@overload
def check(*, name: str | None = None) -> Callable[[_CheckFn], _CheckFn]: ...
def check(
    fn: _CheckFn | None = None, *, name: str | None = None
) -> _CheckFn | Callable[[_CheckFn], _CheckFn]:
    """Register a verification hook with the active run (decorator).

    The decorated function takes no arguments — close over whatever it needs.
    It runs when the run completes and may return an :class:`~foundation.checks.Assertion`
    (or a list of them), a bare bool, a ``(bool, observed)`` or
    ``(bool, observed, expected)`` tuple, a ``{"passed": bool, "observed":
    ..., "expected": ...}`` dict, or nothing at all — plain ``assert``
    statements work: raising ``AssertionError`` records a failure, returning
    ``None`` records a pass. Carry the observed value when you can: a
    record that says ``returned False`` made one real session dig the
    artifact store by hand to learn what the lattice constant was.

    Raises:
        NoActiveRunError: Used outside a ``with ws.start_run(...)`` block.

    Examples:
        >>> try:
        ...     @check
        ...     def orphan():
        ...         return True
        ... except NoActiveRunError:
        ...     print("checks need an active run")
        checks need an active run
    """
    active = current_run()
    if active is None:
        raise NoActiveRunError()
    if fn is None:
        return active.check(name=name)
    return active.check(fn)


class ActiveRun:
    """Handle to the run currently executing inside ``Workspace.start_run``.

    Exposes the run's identity, artifact declaration (:meth:`keep`), and check
    registration (:meth:`check`); the ``@task`` tracer records through it.
    ``dry_run`` is True inside a throwaway run opened by
    ``Workspace.start_run(dry_run=True)``: a task that can run its engine
    without integrating anything (``run_lammps`` with its loops emptied) reads it.
    ``replay`` is set inside a run opened with ``replay=``: the tracer then
    answers every ``@task`` call from another run's results (see :class:`Replay`).
    """

    def __init__(
        self,
        runs: SQLiteRunStore,
        artifacts: ArtifactStore,
        run_id: str,
        *,
        dry_run: bool = False,
        replay: Replay | None = None,
    ) -> None:
        self.runs = runs
        self.artifacts = artifacts
        self.id = run_id
        self.dry_run = dry_run
        self.replay = replay
        self._checks: list[tuple[str, _CheckFn]] = []

    def __repr__(self) -> str:
        return f"ActiveRun({self.id!r})"

    @property
    def run(self) -> Run:
        """A fresh snapshot of the run's current state."""
        return self.runs.get(self.id)

    @overload
    def check(self, fn: _CheckFn) -> _CheckFn: ...
    @overload
    def check(self, *, name: str | None = None) -> Callable[[_CheckFn], _CheckFn]: ...
    def check(
        self, fn: _CheckFn | None = None, *, name: str | None = None
    ) -> _CheckFn | Callable[[_CheckFn], _CheckFn]:
        """Register a verification hook on this run (see module-level :func:`check`)."""

        def register(f: _CheckFn) -> _CheckFn:
            self._checks.append((name or f.__name__, f))
            return f

        return register if fn is None else register(fn)

    def keep(
        self,
        name: str,
        value: object,
        *,
        role: ArtifactRole | str = ArtifactRole.TERMINAL,
        recipe: dict[str, object] | None = None,
    ) -> ArtifactRef:
        """Declare an artifact of this run — by default a *terminal* one.

        A :class:`~pathlib.Path` value stores the file's raw bytes; any other
        value (including a plain string) is serialized as a Python value.
        Declaring an already-stored value is cheap: content addressing
        deduplicates, so promoting a task output to terminal costs one
        database row, not a copy.

        Examples:
            >>> import tempfile
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> with ws.start_run(name="demo") as run:
            ...     ref = run.keep("energy", {"value": -10.84, "unit": "eV"})
            >>> ref.role.value
            'terminal'
            >>> ws.close()
        """
        if isinstance(value, (Path, os.PathLike)):
            digest = self.artifacts.put(value)
        else:
            digest = self.artifacts.put_bytes(dumps(value))
        return self.runs.add_artifact(
            self.id,
            name=name,
            role=role,
            hash=digest,
            size_bytes=self.artifacts.size(digest),
            recipe=recipe,
        )

    def _finish_completed(self) -> None:
        """Evaluate checks, store results, and gate quarantined -> verified."""
        results: list[CheckResult] = []
        for name, fn in self._checks:
            results.extend(_evaluate_check(self.id, name, fn))
        if results:
            self.runs.add_check_results(self.id, results)
        passed = sum(1 for r in results if r.passed)
        if results and passed == len(results):
            # suppress: someone moved the run mid-flight (force-promote etc.); leave it
            with suppress(IllegalTransitionError):
                self.runs.transition(
                    self.id,
                    LifecycleState.VERIFIED,
                    actor="checks",
                    reason=f"{passed}/{len(results)} assertions passed",
                    expected=LifecycleState.QUARANTINED,
                )


def _evaluate_check(run_id: str, name: str, fn: _CheckFn) -> list[CheckResult]:
    """Run one check function and coerce its outcome into results.

    A failed result that names no observed value carries evidence (see
    :func:`check_evidence`), so the record says what the check saw.
    """
    raised: list[BaseException] = []
    results = _coerce_check(run_id, name, fn, raised)
    if all(r.passed or r.observed is not None for r in results):
        return results
    evidence = check_evidence(fn, raised[0] if raised else None)
    return [
        r if r.passed or r.observed is not None else r.model_copy(update={"evidence": evidence})
        for r in results
    ]


#: The longest check source a failure record carries, in lines.
EVIDENCE_SOURCE_LINES = 40
#: The most keys a failure record lists for one dict the check read.
EVIDENCE_KEYS = 40


def check_evidence(fn: _CheckFn, raised: BaseException | None = None) -> dict[str, Any]:
    """What a failed check saw, for a record that has no observed value.

    The evidence holds:

    * ``source``: the check's text, at most :data:`EVIDENCE_SOURCE_LINES`
      lines, or None when Python cannot find it.
    * ``keys``: for each dict the check reads by name (a closure cell or a
      module global), its top-level keys. A check that indexes a missing
      key shows here which keys the result held.
    * ``raised`` and ``line``, for a check that raised: the exception, and
      the line of the check's file that raised it.

    Examples:
        >>> result = {"rows": [], "steps": 0}
        >>> def fraction_ok():
        ...     return result["n_fcc"] / result["steps"] > 0.9
        >>> try:
        ...     fraction_ok()
        ... except KeyError as e:
        ...     evidence = check_evidence(fraction_ok, e)
        >>> evidence["keys"]
        {'result': ['rows', 'steps']}
        >>> evidence["raised"]
        "KeyError: 'n_fcc'"
    """
    evidence: dict[str, Any] = {"source": _check_source(fn), "keys": _dict_keys_read(fn)}
    if raised is not None:
        evidence["raised"] = f"{type(raised).__name__}: {raised}"
        evidence["line"] = _raising_line(fn, raised)
    return evidence


def _check_source(fn: _CheckFn) -> str | None:
    try:
        lines = textwrap.dedent(inspect.getsource(fn)).splitlines()
    except (OSError, TypeError):
        return None
    if len(lines) > EVIDENCE_SOURCE_LINES:
        more = len(lines) - EVIDENCE_SOURCE_LINES
        lines = [*lines[:EVIDENCE_SOURCE_LINES], f"... ({more} more lines)"]
    return "\n".join(lines)


def _dict_keys_read(fn: _CheckFn) -> dict[str, list[str]]:
    """The top-level keys of each dict *fn* reads by name."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return {}
    named: dict[str, object] = {}
    for var, cell in zip(code.co_freevars, getattr(fn, "__closure__", None) or (), strict=False):
        with suppress(ValueError):  # an empty cell
            named[var] = cell.cell_contents
    module_globals = getattr(fn, "__globals__", {})
    for var in code.co_names:
        if var in module_globals and var not in named:
            named[var] = module_globals[var]
    keys: dict[str, list[str]] = {}
    for var, value in named.items():
        if isinstance(value, dict):
            listed = [str(k) for k in list(value)[:EVIDENCE_KEYS]]
            if len(value) > EVIDENCE_KEYS:
                listed.append(f"... ({len(value) - EVIDENCE_KEYS} more)")
            keys[var] = listed
    return keys


def _raising_line(fn: _CheckFn, raised: BaseException) -> str | None:
    """``file:line: text`` of the last frame in the check's file, else the last frame."""
    frames = traceback.extract_tb(raised.__traceback__)
    if not frames:
        return None
    code = getattr(fn, "__code__", None)
    own = [f for f in frames if code is not None and f.filename == code.co_filename]
    frame = (own or frames)[-1]
    return f"{Path(frame.filename).name}:{frame.lineno}: {(frame.line or '').strip()}"


def _coerce_check(
    run_id: str, name: str, fn: _CheckFn, raised: list[BaseException]
) -> list[CheckResult]:
    """Run one check function; *raised* receives what it raised, if anything."""
    try:
        outcome = fn()
        # A generator-based check has not executed yet — its body runs during
        # materialization, so that must sit inside the same coercion or an
        # assert inside the generator would crash the (already completed) run.
        if isinstance(outcome, Iterable) and not isinstance(
            outcome, (str, bytes, dict, bool, Assertion, list, tuple)
        ):
            outcome = list(outcome)
    except AssertionError as e:
        raised.append(e)
        message = str(e) or "assertion failed"
        return [CheckResult(run_id=run_id, name=name, kind="assert", passed=False, message=message)]
    except Exception as e:  # a crashing check is a failing check, never a crashed run
        raised.append(e)
        message = f"check raised {type(e).__name__}: {e}"
        return [CheckResult(run_id=run_id, name=name, kind="error", passed=False, message=message)]

    outcome = _coerce_foreign_bool(outcome)
    if outcome is None:
        return [
            CheckResult(
                run_id=run_id,
                name=name,
                kind="assert",
                passed=True,
                message="completed without assertion errors",
            )
        ]
    if isinstance(outcome, bool):
        return [
            CheckResult(
                run_id=run_id,
                name=name,
                kind="custom",
                passed=outcome,
                message=f"returned {outcome}",
            )
        ]
    if isinstance(outcome, tuple) and outcome and isinstance(outcome[0], bool):
        observed = outcome[1] if len(outcome) > 1 else None
        expected = outcome[2] if len(outcome) > 2 else None
        return [_custom_with_values(run_id, name, outcome[0], observed, expected)]
    if isinstance(outcome, dict) and isinstance(outcome.get("passed"), bool):
        return [
            _custom_with_values(
                run_id, name, outcome["passed"], outcome.get("observed"), outcome.get("expected")
            )
        ]
    if isinstance(outcome, Assertion):
        return [_from_assertion(run_id, name, outcome)]
    if isinstance(outcome, Iterable):
        results: list[CheckResult] = []
        items = list(outcome)
        for i, item in enumerate(items):
            item_name = name if len(items) == 1 else f"{name}[{i}]"
            if isinstance(item, Assertion):
                results.append(_from_assertion(run_id, item_name, item))
            else:
                results.append(
                    CheckResult(
                        run_id=run_id,
                        name=item_name,
                        kind="error",
                        passed=False,
                        message=f"check yielded unsupported type {type(item).__name__!r}",
                    )
                )
        if not results:
            results.append(
                CheckResult(
                    run_id=run_id,
                    name=name,
                    kind="error",
                    passed=False,
                    message="check returned an empty collection of assertions",
                )
            )
        return results
    return [
        CheckResult(
            run_id=run_id,
            name=name,
            kind="error",
            passed=False,
            message=(
                f"check returned unsupported type {type(outcome).__name__!r} "
                f"(expected Assertion, bool, iterable of Assertions, or None)"
            ),
        )
    ]


def _coerce_foreign_bool(outcome: object) -> object:
    """Turn numpy's bool scalars (np.True_/np.False_) into plain bools.

    Checks comparing numpy values naturally return them; without coercion they
    would land in the unsupported-type branch and fail a healthy run.
    """
    kind = type(outcome)
    if kind.__module__ == "numpy" and kind.__name__ in ("bool_", "bool"):
        return bool(outcome)
    return outcome


def _custom_with_values(
    run_id: str, name: str, passed: bool, observed: object, expected: object
) -> CheckResult:
    """A custom check that said what it saw, so the record can show it."""
    message = f"returned {passed}; observed {observed!r}"
    if expected is not None:
        message += f", expected {expected!r}"
    return CheckResult(
        run_id=run_id,
        name=name,
        kind="custom",
        passed=passed,
        message=message,
        observed=observed,
        expected=expected,
    )


def _from_assertion(run_id: str, name: str, assertion: Assertion) -> CheckResult:
    return CheckResult(
        run_id=run_id,
        name=name,
        kind=assertion.kind,
        passed=assertion.passed,
        message=assertion.message,
        observed=assertion.observed,
        expected=assertion.expected,
    )



# -- replay ----------------------------------------------------------------------


class Replay:
    """Answers a script's ``@task`` calls from one completed run's results.

    A re-verify and a dry run from a run execute the script again in a
    throwaway workspace, with no engine started. Each task call is
    matched to a completed task of the source run, and the call returns
    that task's stored outputs. A call matches when it has the source
    task's cache identity, or the same name, inputs, code, engine
    versions, and engine command. A call with no match raises
    :class:`~foundation.errors.ReplayError` naming the task and what
    differs, and :attr:`refusal` keeps the error, so a script that
    catches it still ends refused.

    Examples:
        >>> import tempfile
        >>> ws = Workspace(tempfile.mkdtemp())
        >>> with ws.start_run(name="empty") as run:
        ...     pass
        >>> replay = Replay(ws, run.id)
        >>> try:
        ...     replay.take("relax", "0" * 64, {}, {})
        ... except ReplayError as e:
        ...     print(str(e).split(";")[0])
        run ... has no result for task relax (call 1): it completed 0 relax call(s)
        >>> ws.close()
    """

    def __init__(self, source: Workspace, run_id: str) -> None:
        self.run_id = source.runs.get(run_id).id
        self.artifacts = source.artifacts
        self.tasks = [
            t
            for t in source.runs.list_tasks(self.run_id)
            if t.status is ExecutionStatus.COMPLETED
        ]
        self.taken: list[TaskRecord] = []
        self.refusal: ReplayError | None = None
        self._calls: dict[str, int] = {}

    def take(
        self,
        task_name: str,
        cache_key: str,
        inputs: Mapping[str, str],
        recipe: Mapping[str, Any],
    ) -> TaskRecord:
        """The source task that answers this call, or raise :class:`ReplayError`."""
        call = self._calls.get(task_name, 0)
        self._calls[task_name] = call + 1
        same_name = [t for t in self.tasks if t.name == task_name]
        match = next((t for t in same_name if t.cache_key == cache_key), None)
        if match is None:
            match = next(
                (
                    t
                    for t in same_name
                    if dict(t.inputs) == dict(inputs)
                    and _replay_identity(t.recipe) == _replay_identity(recipe)
                ),
                None,
            )
        if match is None:
            self.refusal = ReplayError(self._difference(task_name, call, same_name, inputs, recipe))
            raise self.refusal
        gone = [h for h in match.outputs.values() if not self.artifacts.has(h)]
        if gone:
            self.refusal = ReplayError(
                f"the output bytes of task {task_name} (call {call + 1}) in run "
                f"{self.run_id} are gone, so the run cannot be replayed; relaunch the script"
            )
            raise self.refusal
        self.taken.append(match)
        return match

    def _difference(
        self,
        task_name: str,
        call: int,
        same_name: list[TaskRecord],
        inputs: Mapping[str, str],
        recipe: Mapping[str, Any],
    ) -> str:
        label = _replay_label(recipe)
        where = f"task {task_name}{label} (call {call + 1})"
        remedy = (
            "a replay answers every task call from the run's own results and starts "
            "no engine; relaunch the script to compute the changed task"
        )
        if call >= len(same_name):
            return (
                f"run {self.run_id} has no result for {where}: it completed "
                f"{len(same_name)} {task_name} call(s); {remedy}"
            )
        source = same_name[call]
        changed = sorted(
            name for name in set(inputs) | set(source.inputs)
            if inputs.get(name) != source.inputs.get(name)
        )
        if changed:
            what = f"its input {', '.join(changed)} changed"
        else:
            mine, theirs = _replay_identity(recipe), _replay_identity(source.recipe)
            parts = ("code", "cache identity")
            what = ", ".join(
                f"its {part} changed" for part, a, b in zip(parts, mine, theirs, strict=True)
                if a != b
            ) or "its cache identity changed"
        return f"{where} differs from run {self.run_id}'s: {what}; {remedy}"


#: Recipe keys that name the engine a task resolved in the process that ran
#: it. A replay starts no engine, and the reverifying process may hold a
#: different slice (no gpu, so the plain build) or probe a different version,
#: so these never decide whether a stored result answers a replayed call.
_ENGINE_IDENTITY_KEYS = frozenset(
    {"build", "command", "version", "engine_version", "setup", "setup_mode", "provenance"}
)


def _replay_identity(recipe: Mapping[str, Any]) -> tuple[object, object]:
    """The parts of a recipe that decide a task's answer, apart from its inputs.

    Examples:
        >>> gpu = {"code_sha256": "c", "engines": {"lammps": "2"}, "extra": {"build": "gpu",
        ...        "command": "mpirun -n 1 lmp -k on g 1", "version": "2 Aug 2023",
        ...        "files_sha256": {"a": "1"}}}
        >>> cpu = {**gpu, "engines": {"lammps": "3"}, "extra": {**gpu["extra"], "build": "cpu",
        ...        "command": "lmp", "version": "29 Aug 2024"}}
        >>> _replay_identity(gpu) == _replay_identity(cpu)
        True
        >>> _replay_identity({**cpu, "code_sha256": "d"}) == _replay_identity(cpu)
        False
    """
    extra = {
        k: v for k, v in (recipe.get("extra") or {}).items() if k not in _ENGINE_IDENTITY_KEYS
    }
    return recipe.get("code_sha256"), extra


def _replay_label(recipe: Mapping[str, Any]) -> str:
    label = (recipe.get("params") or {}).get("label")
    return f" {label!r}" if isinstance(label, str) else ""


# -- run liveness --------------------------------------------------------------


def this_host() -> str:
    """The hostname stamped on a run this process starts.

    Examples:
        >>> isinstance(this_host(), str) and this_host() != ""
        True
    """
    return socket.gethostname()


def this_job() -> str | None:
    """The scheduler job this process runs under, or None outside a job.

    Every batch job, sandbox jobs included, carries ``$SLURM_JOB_ID``. It
    is read here and nowhere else, so a run's stamp, a reservation's
    stamp, and a liveness verdict all name the same job.

    Examples:
        >>> before = os.environ.pop("SLURM_JOB_ID", None)
        >>> this_job() is None
        True
        >>> os.environ["SLURM_JOB_ID"] = "4242"
        >>> this_job()
        '4242'
        >>> _ = os.environ.pop("SLURM_JOB_ID")
        >>> if before is not None: os.environ["SLURM_JOB_ID"] = before
    """
    return os.environ.get("SLURM_JOB_ID") or None


def run_liveness(
    run: Run,
    *,
    host: str | None = None,
    job: str | EllipsisType | None = ...,
    job_states: Mapping[str, JobState] | None = None,
) -> str:
    """Where a running run's process stands, as seen from *host* (this one
    by default) inside *job* (this process's job by default).

    One of ``job-ended`` (the run's job is terminal in *job_states*, so
    its process died with the job wherever it ran), ``other-job`` (the
    run's job is set and is not *job*, so its pid means nothing here:
    a sandbox job has its own PID namespace, and a pid from another
    job can match an unrelated process), ``alive`` (the recorded process
    exists here), ``gone`` (it does not), ``elsewhere`` (the run was
    started on another host, so nothing can be checked from here), or
    ``unrecorded`` (the run predates the pid stamp, or is not running).
    *job_states* maps job ids to the scheduler's answer, resolved once
    per sweep by the caller; without it ``job-ended`` is never returned.

    Examples:
        >>> run_liveness(Run(status="running", pid=os.getpid(), host=this_host()), job=None)
        'alive'
        >>> run_liveness(Run(status="running", pid=2**22 - 1, host=this_host()), job=None)
        'gone'
        >>> run_liveness(Run(status="running", pid=os.getpid(), host="another-node"))
        'elsewhere'
        >>> run_liveness(Run(status="running"))
        'unrecorded'
        >>> run_liveness(Run(status="completed", pid=1, host=this_host()))
        'unrecorded'
        >>> stamped = Run(status="running", pid=os.getpid(), host=this_host(), job_id="7")
        >>> run_liveness(stamped, job="7")
        'alive'
        >>> run_liveness(stamped, job="8")
        'other-job'
        >>> run_liveness(stamped, job=None)
        'other-job'
        >>> run_liveness(stamped, job="8", job_states={"7": JobState.CANCELLED})
        'job-ended'
        >>> run_liveness(stamped, job="7", job_states={"7": JobState.RUNNING})
        'alive'
    """
    if run.status is not ExecutionStatus.RUNNING:
        return "unrecorded"
    if run.job_id is not None:
        state = (job_states or {}).get(run.job_id)
        if state is not None and state.is_terminal:
            return "job-ended"
        here_job = this_job() if job is ... else job
        if run.job_id != here_job:
            return "other-job"
    if run.pid is None or run.host is None:
        return "unrecorded"
    here = host if host is not None else this_host()
    if run.host != here:
        return "elsewhere"
    return "alive" if process_alive(run.pid) else "gone"


def describe_liveness(
    run: Run,
    *,
    host: str | None = None,
    job: str | EllipsisType | None = ...,
    job_states: Mapping[str, JobState] | None = None,
) -> str:
    """One phrase for a listing: what :func:`run_liveness` found and why.

    Examples:
        >>> describe_liveness(Run(status="running", pid=7, host="n1"), host="n2")
        'process 7 on n1, not this host (n2); liveness not checked from here'
        >>> describe_liveness(Run(status="running"))
        'no process recorded; liveness unknown'
        >>> stamped = Run(status="running", pid=7, host="n1", job_id="7")
        >>> describe_liveness(stamped, host="n1", job="8")
        'process 7 on n1 belongs to job 7, not this job (8); liveness not checked from here'
        >>> describe_liveness(stamped, host="n1", job=None, job_states={"7": JobState.TIMEOUT})
        'job 7 is timeout; the process died with it'
    """
    verdict = run_liveness(run, host=host, job=job, job_states=job_states)
    where = f"process {run.pid} on {run.host}"
    if run.job_id is not None:
        where += f" (job {run.job_id})"
    if verdict == "alive":
        return f"{where} is alive"
    if verdict == "gone":
        return f"{where} is gone"
    if verdict == "elsewhere":
        here = host if host is not None else this_host()
        return f"{where}, not this host ({here}); liveness not checked from here"
    if verdict == "other-job":
        here_job = this_job() if job is ... else job
        return (
            f"process {run.pid} on {run.host} belongs to job {run.job_id}, not this job "
            f"({here_job or 'none'}); liveness not checked from here"
        )
    if verdict == "job-ended":
        state = (job_states or {})[str(run.job_id)]
        return f"job {run.job_id} is {state.value}; the process died with it"
    return "no process recorded; liveness unknown"


def job_states_for(runs: Iterable[Run]) -> dict[str, JobState]:
    """Ask the scheduler once about every distinct job the *runs* were stamped with.

    An empty mapping where the scheduler cannot be reached (inside a
    sandbox there is no ``squeue``) or the runs carry no job, so a
    caller that cannot know never fails a run for it. An ``undetermined``
    answer is kept as such and is not terminal.
    """
    jobs = sorted({run.job_id for run in runs if run.job_id is not None})
    states: dict[str, JobState] = {}
    for job_id in jobs:
        try:
            states[job_id] = job_state(job_id).state
        except SchedulerError:
            return {}
    return states


def _claimable_from(reservation: Reservation, host: str) -> Reservation:
    """Refuse, before a run exists, a reservation the run could not claim."""
    if reservation.run_id is not None:
        raise ResourcesError(
            f"reservation {reservation.id!r} is already claimed by run "
            f"{reservation.run_id}; one reservation serves one run"
        )
    if reservation.host != host:
        raise ResourcesError(
            f"reservation {reservation.id!r} was made for host {reservation.host!r}, "
            f"not {host!r}; a slice of one host cannot be claimed from another"
        )
    return reservation


class Workspace:
    """A SLAB workspace: run database + artifact store under one root.

    Args:
        root: Workspace directory (created if missing). Holds ``runs.db``
            and the ``cas/`` artifact tree.

    Examples:
        >>> import tempfile
        >>> from foundation import task, check, converged
        >>> @task
        ... def double(x):
        ...     return 2 * x
        >>> ws = Workspace(tempfile.mkdtemp())
        >>> with ws.start_run(name="demo", intent="doctest") as run:
        ...     y = double(21)
        ...     @check
        ...     def sane():
        ...         return converged(0.01, below=0.05)
        >>> y
        42
        >>> ws.runs.get(run.id).state.value  # checks passed -> verified
        'verified'
        >>> ws.close()
    """

    def __init__(self, root: str | os.PathLike[str] = ".slab") -> None:
        self.root = Path(root).expanduser()
        self.runs = SQLiteRunStore(self.root / "runs.db")
        try:
            self.artifacts = ArtifactStore(self.root / "cas")
        except BaseException:
            self.runs.close()  # the store opened; a failed second half must not leak it
            raise

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"

    def close(self) -> None:
        """Close the underlying run store."""
        self.runs.close()

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def start_run(
        self,
        *,
        name: str = "",
        intent: str | None = None,
        session: str | None = None,
        reservation: Reservation | str | None = None,
        dry_run: bool = False,
        replay: Replay | None = None,
    ) -> Iterator[ActiveRun]:
        """Open a traced run; yield its :class:`ActiveRun` handle.

        The run is created ``quarantined``/``running``. On normal exit it is
        marked ``completed``, its checks are evaluated, and it becomes
        ``verified`` only if every assertion passed (and at least one exists).
        On exception it is marked ``failed`` — recording a one-line ``error``
        and a structured failure record (trimmed traceback and diagnostic
        notes, see :func:`foundation.errors.failure_record`) — checks are skipped,
        and the exception propagates.

        *session* stamps the run with the client session that created it (see
        :func:`resolve_session_id`), so one conversation's runs can be listed
        and promoted together. While the block runs, ``$SLAB_RUN_ID`` names
        the run, so every scratch directory made inside it carries the id.
        Inside a SLURM job the run is also stamped with ``$SLURM_JOB_ID``, so
        :func:`foundation._ops.cancel_job` can fail the runs a cancelled job
        took down.

        *reservation* (a :class:`~foundation.models.Reservation` or its id)
        is the slice a session process checked out for this run with
        :meth:`reserve`. The run claims it: the slice is copied onto the run
        record as ``resources``, the reservation is marked as this run's,
        and the run is set running with this process's pid, all in one
        transaction, so no moment exists in which the slice is neither
        the holder's nor a running run's. The reservation is released when
        the run ends. A reservation that was released, is already claimed,
        or was made for another host is refused with
        :class:`ResourcesError` before the run exists.

        *dry_run* marks the run as a rehearsal: the handle's ``dry_run``
        is True, and a task that can rehearse its engine (``run_lammps``
        with its loops emptied) integrates nothing. The run record itself is
        an ordinary run; open it in a throwaway workspace.

        *replay* answers every ``@task`` call inside the run from another
        run's results (see :class:`Replay`). Open it in a throwaway
        workspace too.

        Raises:
            NestedRunError: A run is already active in this context.
            ResourcesError: The reservation cannot be claimed.

        Examples:
            >>> import tempfile
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> with ws.start_run(name="probe", intent="why not") as run:
            ...     pass
            >>> final = ws.runs.get(run.id)
            >>> (final.status.value, final.state.value)  # no checks: stays quarantined
            ('completed', 'quarantined')
            >>> final.session is None
            True
            >>> ws.close()
        """
        if _CURRENT.get() is not None:
            raise NestedRunError()
        host = this_host()
        reservation_id = reservation.id if isinstance(reservation, Reservation) else reservation
        if reservation_id is not None:
            _claimable_from(self.runs.get_reservation(reservation_id), host)
        created = self.runs.create(
            Run(
                name=name,
                intent=intent,
                session=resolve_session_id(session),
                # Every batch job, sandbox jobs included, carries the variable,
                # so a cancel of the job can find the runs it takes down.
                job_id=this_job(),
            )
        )
        if reservation_id is not None:
            try:
                self.runs.claim_reservation(
                    reservation_id, created.id, host=host, pid=os.getpid()
                )
            except ResourcesError as e:
                self.runs.set_status(
                    created.id, ExecutionStatus.FAILED, error=f"ResourcesError: {e}"
                )
                raise
        else:
            self.runs.set_status(
                created.id, ExecutionStatus.RUNNING, pid=os.getpid(), host=host
            )
        active = ActiveRun(
            self.runs, self.artifacts, created.id, dry_run=dry_run, replay=replay
        )
        token = _CURRENT.set(active)
        # Every scratch directory a calculation makes inside the run is
        # stamped with the run's id (slab.scratch reads the variable), so
        # a sweep after the process dies knows which run it belonged to.
        run_env_before = os.environ.get(RUN_ENV)
        os.environ[RUN_ENV] = created.id
        try:
            yield active
        except BaseException as exc:
            record = failure_record(exc)
            try:
                self.runs.set_status(
                    created.id,
                    ExecutionStatus.FAILED,
                    error=f"{record['type']}: {record['message']}",
                    failure=record,
                )
            except Exception as secondary:
                # The run was already marked failed by someone else (an
                # expire sweep, the script itself), or the store refused.
                # The body's exception is the real cause and must be what
                # the caller sees; the bookkeeping failure rides as a note.
                exc.add_note(f"(recording the failure also failed: {secondary})")
            raise
        else:
            self.runs.set_status(created.id, ExecutionStatus.COMPLETED)
            active._finish_completed()
        finally:
            if run_env_before is None:
                os.environ.pop(RUN_ENV, None)
            else:
                os.environ[RUN_ENV] = run_env_before
            _CURRENT.reset(token)

    def reserve(
        self,
        *,
        ntasks: int | None = None,
        threads: int | None = None,
        gpus: int = 0,
        host: str | None = None,
        holder_pid: int | None = None,
        budget: Budget | None = None,
    ) -> Reservation:
        """Check out a slice of this host for a launch that does not exist yet.

        A session process (Mason, the MCP server) calls this before it
        starts a run, and hands the reservation to the run, which claims
        it through :meth:`start_run`. The store computes what is free from
        the live reservations, so two reservers on one store never
        overlap, and a request that does not fit raises
        :class:`ResourcesError` carrying the free ids.

        *ntasks* and *threads* size the slice (``ntasks * threads`` cpus)
        and *gpus* counts the gpus. With neither count the whole free cpu
        budget is taken, so an unsized launch is accounted for like any
        other. With *gpus* and neither count the launch runs one rank per
        gpu and the free cpus as threads, because a KOKKOS build gives
        each MPI rank one device. *budget* is this process's
        :func:`slab.resources.budget` unless given, and the rank and
        thread defaults of an unsized cpu slice are this process's
        :func:`slab.resources.envelope`.

        Examples:
            >>> import tempfile
            >>> from slab.resources import Budget
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> held = ws.reserve(ntasks=2, gpus=1, budget=Budget(cpus=(0, 1, 2, 3), gpus=("0",)))
            >>> (held.cpus, held.gpus, held.run_id)
            ((0, 1), ('0',), None)
            >>> ws.free_resources(budget=Budget(cpus=(0, 1, 2, 3), gpus=("0",)))["free"]
            {'cpus': [2, 3], 'gpus': []}
            >>> ws.close()
        """
        from slab.resources import budget as discover_budget
        from slab.resources import envelope

        found = budget if budget is not None else discover_budget()
        defaults = envelope()
        return self.runs.reserve(
            host=host if host is not None else this_host(),
            holder_pid=holder_pid if holder_pid is not None else os.getpid(),
            job_id=this_job(),
            budget_cpus=found.cpus,
            budget_gpus=found.gpus,
            ntasks=ntasks,
            threads=threads,
            gpus=gpus,
            default_ntasks=defaults.ntasks,
            default_threads=defaults.threads,
        )

    def free_resources(
        self,
        *,
        host: str | None = None,
        budget: Budget | None = None,
        live: list[Reservation] | None = None,
    ) -> dict[str, Any]:
        """What this host's budget holds and what is free right now.

        The read side of :meth:`reserve`: ``budget`` and ``free`` each list
        cpu ids and gpu ids, ``budget`` says where its gpu ids came from
        (``gpu_source``, see :func:`slab.resources.budget`), and
        ``reservations`` lists the ids of the live reservations that hold
        the difference. Derived, never counted. The budget is one
        allocation, so only the reservations of this process's job
        (:func:`this_job`) count against it.
        A caller that has already read the live reservations passes them
        as *live*, so one answer rests on one read.
        """
        from slab.resources import budget as discover_budget

        found = budget if budget is not None else discover_budget()
        where = host if host is not None else this_host()
        if live is None:
            live = self.runs.live_reservations(where, this_job())
        used_cpus = {cpu for row in live for cpu in row.cpus}
        used_gpus = {gpu for row in live for gpu in row.gpus}
        return {
            "host": where,
            "budget": {
                "cpus": list(found.cpus),
                "gpus": list(found.gpus),
                "gpu_source": found.gpu_source,
            },
            "free": {
                "cpus": [cpu for cpu in found.cpus if cpu not in used_cpus],
                "gpus": [gpu for gpu in found.gpus if gpu not in used_gpus],
            },
            "reservations": [row.id for row in live],
        }

    def release_dead(self) -> list[Reservation]:
        """Release every reservation on this host that no live process holds.

        An unclaimed reservation whose holder died, and a claimed one whose
        run is no longer running and alive, are deleted and returned. Only
        the reservations of this process's job are judged, because another
        job's process cannot be seen from here.
        :meth:`reap_dead` calls this, so every reap and every wait poll
        cleans up.
        """
        return self.runs.release_dead(this_host(), this_job())

    def reap_dead(self, *, caller: str) -> list[Run]:
        """Mark failed every running run whose process is gone, or whose job ended.

        A hard-killed process (SIGKILL, OOM, a node reboot) leaves its run at
        status ``running`` forever, and a reader of the record cannot tell it
        from a live one. Each running run stamped with this host's name and
        this job is checked with its pid. A run stamped with a job is also
        judged by the scheduler: the distinct job ids of the running runs
        are resolved once (:func:`job_states_for`), and a run whose job is
        terminal (cancelled, timed out, failed, completed) is marked failed
        wherever it ran, because its process died with the job. A run of
        another job that is not terminal is left alone by pid, a run stamped
        with another host is left alone, because nothing about that host
        can be seen from here, and so is a run from before the stamp
        existed. Where the scheduler cannot be reached, or answers
        ``undetermined``, no run is failed for its job. Do not call this
        on the host of a running sandbox job: ``--containall`` gives the
        container its own PID namespace and its hostname matches the host,
        so the job's own runs would be judged by pids that mean nothing on
        the host, and marked gone. :meth:`settle_ended_jobs` asks only the
        scheduler and is safe from any host. *caller* names who
        marked the run in its error line. Returns the runs marked failed.
        The reservations those runs held, and every other dead reservation
        of this job on this host, are released on the way
        (:meth:`release_dead`), and the scratch directories those runs made
        are removed (:func:`foundation.retention.sweep_scratch`).

        Examples:
            >>> import tempfile
            >>> from foundation.models import Run
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> dead = ws.runs.create(Run(name="killed"))
            >>> _ = ws.runs.set_status(dead.id, "running", pid=2**22 - 1, host=this_host())
            >>> away = ws.runs.create(Run(name="elsewhere"))
            >>> _ = ws.runs.set_status(away.id, "running", pid=1, host="another-node")
            >>> [r.name for r in ws.reap_dead(caller="doctest")]
            ['killed']
            >>> ws.runs.get(dead.id).error
            'process 4194303 on ... is gone; marked failed by doctest'
            >>> ws.runs.get(away.id).status.value
            'running'
            >>> ws.close()
        """
        reaped: list[Run] = []
        running = self.runs.list_runs(status=ExecutionStatus.RUNNING)
        states = job_states_for(running)
        for run in running:
            verdict = run_liveness(run, job_states=states)
            if verdict == "gone":
                error = f"process {run.pid} on {run.host} is gone; marked failed by {caller}"
            elif verdict == "job-ended":
                error = (
                    f"job {run.job_id} is {states[str(run.job_id)].value}; "
                    f"marked failed by {caller}"
                )
            else:
                continue
            with suppress(IllegalStatusChangeError):  # it may finish under us; fine
                reaped.append(self.runs.set_status(run.id, ExecutionStatus.FAILED, error=error))
        self.release_dead()
        if reaped:
            sweep_scratch(self, only=[run.id for run in reaped])
        return reaped

    def settle_ended_jobs(
        self, *, caller: str, ended: Iterable[str] = (), dry_run: bool = False
    ) -> list[Run]:
        """Mark failed every running run whose scheduler job has ended.

        The scheduler is asked once about each distinct job the running
        runs carry (:func:`job_states_for`). A run whose job is terminal is
        failed with the error line ``job N is <state>; marked failed by
        <caller>``. A job in *ended* is taken as ended without asking, for
        a job the scheduler cannot place (no accounting, or a record that
        has aged out), and its runs are failed with ``job N ended; marked
        failed by <caller>``. No pid is consulted, so the answer is the
        same from any host. A run with no job, a run of a job the
        scheduler still holds or cannot place, and every run where the
        scheduler cannot be reached stay as they are. Each failed run's
        reservation is released with it, and the scratch directories the
        failed runs made are removed. With *dry_run* nothing changes, and
        the runs that would be failed are returned as they stand.

        Examples:
            >>> import tempfile
            >>> from foundation.models import Run
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> left = ws.runs.create(Run(name="left", job_id="8"))
            >>> _ = ws.runs.set_status(left.id, "running", pid=1, host="n1")
            >>> [r.name for r in ws.settle_ended_jobs(caller="t", ended=["8"], dry_run=True)]
            ['left']
            >>> ws.runs.get(left.id).status.value
            'running'
            >>> [r.error for r in ws.settle_ended_jobs(caller="t", ended=["8"])]
            ['job 8 ended; marked failed by t']
            >>> ws.close()
        """
        forced = set(ended)
        running = [
            run
            for run in self.runs.list_runs(status=ExecutionStatus.RUNNING)
            if run.job_id is not None
        ]
        states = job_states_for([run for run in running if run.job_id not in forced])
        settled: list[Run] = []
        for run in running:
            if run.job_id in forced:
                error = f"job {run.job_id} ended; marked failed by {caller}"
            else:
                state = states.get(str(run.job_id))
                if state is None or not state.is_terminal:
                    continue
                error = f"job {run.job_id} is {state.value}; marked failed by {caller}"
            if dry_run:
                settled.append(run)
                continue
            with suppress(IllegalStatusChangeError):  # it may finish under us; fine
                settled.append(self.runs.set_status(run.id, ExecutionStatus.FAILED, error=error))
        if settled and not dry_run:
            sweep_scratch(self, only=[run.id for run in settled])
        return settled

    def fail_run(self, run_id: str, *, reason: str) -> Run:
        """Mark one running run failed on the operator's word; return the snapshot.

        The verb exists for a run that ``reap_dead`` cannot judge: one
        started on another host, or one from before the pid stamp. It is
        refused while the run's process is alive on this host, because a
        live process will advance its own status, and refused for a run
        that is not running, because there is nothing to retire.

        Raises:
            RunStateError: The run is not running, or its process is alive here.

        Examples:
            >>> import tempfile
            >>> from foundation.models import Run
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> away = ws.runs.create(Run(name="elsewhere"))
            >>> _ = ws.runs.set_status(away.id, "running", pid=1, host="another-node")
            >>> ws.fail_run(away.id, reason="node drained").error
            'marked failed by the operator: node drained'
            >>> ws.close()
        """
        run = self.runs.get(run_id)
        if run.status is not ExecutionStatus.RUNNING:
            raise RunStateError(
                run.id, run.state, f"fail (its status is {run.status.value!r}, not 'running')"
            )
        if run_liveness(run) == "alive":
            raise RunStateError(
                run.id,
                run.state,
                f"fail (its process {run.pid} on {run.host} is alive; "
                f"stop it first, or wait for it)",
            )
        return self.runs.set_status(
            run.id, ExecutionStatus.FAILED, error=f"marked failed by the operator: {reason}"
        )

    def expire_due(
        self,
        policy: RetentionPolicy = DEFAULT_POLICY,
        *,
        now: datetime | None = None,
        include_running: bool = False,
    ) -> list[Run]:
        """Run the TTL sweep on this workspace (see :func:`foundation.retention.expire_due`).

        Examples:
            >>> import tempfile
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> ws.expire_due()
            []
            >>> ws.close()
        """
        return expire_due(self.runs, policy, now=now, include_running=include_running)

    def gc(self, policy: RetentionPolicy = DEFAULT_POLICY, *, dry_run: bool = False) -> GcReport:
        """Reclaim undemanded artifact bytes (see :func:`foundation.retention.gc`).

        Examples:
            >>> import tempfile
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> ws.gc().freed_bytes
            0
            >>> ws.close()
        """
        return gc(self.runs, self.artifacts, policy, dry_run=dry_run)

    def purge_expired(
        self, *, dry_run: bool = False, only: Iterable[str] | None = None
    ) -> PurgeReport:
        """Delete expired runs outright (see :func:`foundation.retention.purge_expired`).

        Examples:
            >>> import tempfile
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> ws.purge_expired().deleted
            []
            >>> ws.close()
        """
        return purge_expired(self.runs, self.artifacts, dry_run=dry_run, only=only)
