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

import os
import socket
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from foundation.artifacts import ArtifactStore
from foundation.checks import Assertion
from foundation.errors import (
    IllegalStatusChangeError,
    IllegalTransitionError,
    NestedRunError,
    NoActiveRunError,
    ResourcesError,
    RunStateError,
    failure_record,
)
from foundation.lifecycle import ExecutionStatus, LifecycleState
from foundation.models import ArtifactRef, ArtifactRole, CheckResult, Reservation, Run
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
    """

    def __init__(self, runs: SQLiteRunStore, artifacts: ArtifactStore, run_id: str) -> None:
        self.runs = runs
        self.artifacts = artifacts
        self.id = run_id
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
    """Run one check function and coerce whatever it produces into results."""
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
        message = str(e) or "assertion failed"
        return [CheckResult(run_id=run_id, name=name, kind="assert", passed=False, message=message)]
    except Exception as e:  # a crashing check is a failing check, never a crashed run
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



# -- run liveness --------------------------------------------------------------


def this_host() -> str:
    """The hostname stamped on a run this process starts.

    Examples:
        >>> isinstance(this_host(), str) and this_host() != ""
        True
    """
    return socket.gethostname()


def run_liveness(run: Run, *, host: str | None = None) -> str:
    """Where a running run's process stands, as seen from *host* (this one by default).

    One of ``alive`` (the recorded process exists here), ``gone`` (it does
    not), ``elsewhere`` (the run was started on another host, so nothing
    can be checked from here), or ``unrecorded`` (the run predates the
    pid stamp, or is not running).

    Examples:
        >>> run_liveness(Run(status="running", pid=os.getpid(), host=this_host()))
        'alive'
        >>> run_liveness(Run(status="running", pid=os.getpid(), host="another-node"))
        'elsewhere'
        >>> run_liveness(Run(status="running"))
        'unrecorded'
        >>> run_liveness(Run(status="completed", pid=1, host=this_host()))
        'unrecorded'
    """
    if run.status is not ExecutionStatus.RUNNING or run.pid is None or run.host is None:
        return "unrecorded"
    here = host if host is not None else this_host()
    if run.host != here:
        return "elsewhere"
    return "alive" if process_alive(run.pid) else "gone"


def describe_liveness(run: Run, *, host: str | None = None) -> str:
    """One phrase for a listing: what :func:`run_liveness` found and why.

    Examples:
        >>> describe_liveness(Run(status="running", pid=7, host="n1"), host="n2")
        'process 7 on n1, not this host (n2); liveness not checked from here'
        >>> describe_liveness(Run(status="running"))
        'no process recorded; liveness unknown'
    """
    verdict = run_liveness(run, host=host)
    if verdict == "alive":
        return f"process {run.pid} on {run.host} is alive"
    if verdict == "gone":
        return f"process {run.pid} on {run.host} is gone"
    if verdict == "elsewhere":
        here = host if host is not None else this_host()
        return (
            f"process {run.pid} on {run.host}, not this host ({here}); "
            f"liveness not checked from here"
        )
    return "no process recorded; liveness unknown"


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
                job_id=os.environ.get("SLURM_JOB_ID") or None,
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
        active = ActiveRun(self.runs, self.artifacts, created.id)
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
        other. *budget* is this process's :func:`slab.resources.budget`
        unless given, and the rank and thread defaults of an unsized slice
        are this process's :func:`slab.resources.envelope`.

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
        cpu ids and gpu ids, and ``reservations`` the ids of the live
        reservations that hold the difference. Derived, never counted.
        A caller that has already read the live reservations passes them
        as *live*, so one answer rests on one read.
        """
        from slab.resources import budget as discover_budget

        found = budget if budget is not None else discover_budget()
        where = host if host is not None else this_host()
        if live is None:
            live = self.runs.live_reservations(where)
        used_cpus = {cpu for row in live for cpu in row.cpus}
        used_gpus = {gpu for row in live for gpu in row.gpus}
        return {
            "host": where,
            "budget": {"cpus": list(found.cpus), "gpus": list(found.gpus)},
            "free": {
                "cpus": [cpu for cpu in found.cpus if cpu not in used_cpus],
                "gpus": [gpu for gpu in found.gpus if gpu not in used_gpus],
            },
            "reservations": [row.id for row in live],
        }

    def release_dead(self) -> list[Reservation]:
        """Release every reservation on this host that no live process holds.

        An unclaimed reservation whose holder died, and a claimed one whose
        run is no longer running and alive, are deleted and returned.
        :meth:`reap_dead` calls this, so every reap and every wait poll
        cleans up.
        """
        return self.runs.release_dead(this_host())

    def reap_dead(self, *, caller: str) -> list[Run]:
        """Mark failed every running run whose process on this host is gone.

        A hard-killed process (SIGKILL, OOM, a node reboot) leaves its run at
        status ``running`` forever, and a reader of the record cannot tell it
        from a live one. Each running run stamped with this host's name is
        checked with its pid; a run stamped with another host is left alone,
        because nothing about that host can be seen from here, and so is a
        run from before the stamp existed. *caller* names who marked the run
        in its error line. Returns the runs marked failed. The reservations
        those runs held, and every other dead reservation on this host, are
        released on the way (:meth:`release_dead`), and the scratch
        directories those runs made are removed
        (:func:`foundation.retention.sweep_scratch`).

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
        for run in self.runs.list_runs(status=ExecutionStatus.RUNNING):
            if run_liveness(run) != "gone":
                continue
            with suppress(IllegalStatusChangeError):  # it may finish under us; fine
                reaped.append(
                    self.runs.set_status(
                        run.id,
                        ExecutionStatus.FAILED,
                        error=(
                            f"process {run.pid} on {run.host} is gone; "
                            f"marked failed by {caller}"
                        ),
                    )
                )
        self.release_dead()
        if reaped:
            sweep_scratch(self, only=[run.id for run in reaped])
        return reaped

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
