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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, overload

from slab.artifacts import ArtifactStore
from slab.checks import Assertion
from slab.errors import (
    IllegalTransitionError,
    NestedRunError,
    NoActiveRunError,
    failure_record,
)
from slab.lifecycle import ExecutionStatus, LifecycleState
from slab.models import ArtifactRef, ArtifactRole, CheckResult, Run
from slab.retention import DEFAULT_POLICY, GcReport, RetentionPolicy, expire_due, gc
from slab.serialize import dumps
from slab.store import SQLiteRunStore

if TYPE_CHECKING:
    from datetime import datetime

_CURRENT: ContextVar[ActiveRun | None] = ContextVar("slab_active_run", default=None)

_CheckFn = Callable[[], object]


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
    It runs when the run completes and may return an :class:`~slab.checks.Assertion`
    (or a list of them), a bare bool, or nothing at all — plain ``assert``
    statements work: raising ``AssertionError`` records a failure, returning
    ``None`` records a pass.

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


class Workspace:
    """A SLAB workspace: run database + artifact store under one root.

    Args:
        root: Workspace directory (created if missing). Holds ``runs.db``
            and the ``cas/`` artifact tree.

    Examples:
        >>> import tempfile
        >>> from slab import task, check, converged
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
        self.artifacts = ArtifactStore(self.root / "cas")

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
    def start_run(self, *, name: str = "", intent: str | None = None) -> Iterator[ActiveRun]:
        """Open a traced run; yield its :class:`ActiveRun` handle.

        The run is created ``quarantined``/``running``. On normal exit it is
        marked ``completed``, its checks are evaluated, and it becomes
        ``verified`` only if every assertion passed (and at least one exists).
        On exception it is marked ``failed`` — recording a one-line ``error``
        and a structured failure record (trimmed traceback and diagnostic
        notes, see :func:`slab.errors.failure_record`) — checks are skipped,
        and the exception propagates.

        Raises:
            NestedRunError: A run is already active in this context.

        Examples:
            >>> import tempfile
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> with ws.start_run(name="probe", intent="why not") as run:
            ...     pass
            >>> final = ws.runs.get(run.id)
            >>> (final.status.value, final.state.value)  # no checks: stays quarantined
            ('completed', 'quarantined')
            >>> ws.close()
        """
        if _CURRENT.get() is not None:
            raise NestedRunError()
        created = self.runs.create(Run(name=name, intent=intent))
        self.runs.set_status(created.id, ExecutionStatus.RUNNING)
        active = ActiveRun(self.runs, self.artifacts, created.id)
        token = _CURRENT.set(active)
        try:
            yield active
        except BaseException as exc:
            record = failure_record(exc)
            self.runs.set_status(
                created.id,
                ExecutionStatus.FAILED,
                error=f"{record['type']}: {record['message']}",
                failure=record,
            )
            raise
        else:
            self.runs.set_status(created.id, ExecutionStatus.COMPLETED)
            active._finish_completed()
        finally:
            _CURRENT.reset(token)

    def expire_due(
        self,
        policy: RetentionPolicy = DEFAULT_POLICY,
        *,
        now: datetime | None = None,
        include_running: bool = False,
    ) -> list[Run]:
        """Run the TTL sweep on this workspace (see :func:`slab.retention.expire_due`).

        Examples:
            >>> import tempfile
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> ws.expire_due()
            []
            >>> ws.close()
        """
        return expire_due(self.runs, policy, now=now, include_running=include_running)

    def gc(self, policy: RetentionPolicy = DEFAULT_POLICY, *, dry_run: bool = False) -> GcReport:
        """Reclaim undemanded artifact bytes (see :func:`slab.retention.gc`).

        Examples:
            >>> import tempfile
            >>> ws = Workspace(tempfile.mkdtemp())
            >>> ws.gc().freed_bytes
            0
            >>> ws.close()
        """
        return gc(self.runs, self.artifacts, policy, dry_run=dry_run)
