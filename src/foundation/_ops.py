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
import sys
import traceback
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from foundation.errors import (
    FoundationError,
    IllegalTransitionError,
    NestedRunError,
    ScriptExitError,
    StorageError,
)
from foundation.lifecycle import ExecutionStatus, LifecycleState
from foundation.models import ArtifactRole, Run
from foundation.retention import DEFAULT_POLICY, RetentionPolicy
from foundation.runtime import Workspace

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
    }


def run_details(ws: Workspace, run_id: str) -> dict[str, Any]:
    """Everything about one run: fields, checks, tasks, artifacts, history.

    This is the evidence surface for agents (``slab show`` and MCP
    ``show_run``): failed runs and tasks carry their structured ``failure``
    record (:func:`foundation.errors.failure_record`), and checks carry the
    ``observed``/``expected`` values their assertions compared — the numbers a
    correction gets computed from. Artifact entries carry ``bytes_available``
    — whether the content is still in the artifact store or has been
    hash-and-discarded.
    """
    run = ws.runs.get(run_id)
    checks = ws.runs.list_check_results(run.id)
    tasks = ws.runs.list_tasks(run.id)
    artifacts = ws.runs.list_artifacts(run.id)
    history = ws.runs.history(run.id)
    return {
        "run": run_summary(run)
        | {
            "meta": run.meta,
            "failure": run.failure,
            "started_at": None if run.started_at is None else run.started_at.isoformat(),
            "finished_at": None if run.finished_at is None else run.finished_at.isoformat(),
        },
        "checks": [
            {
                "name": c.name,
                "kind": c.kind,
                "passed": c.passed,
                "message": c.message,
                "observed": c.observed,
                "expected": c.expected,
            }
            for c in checks
        ],
        "tasks": [
            {
                "seq": t.seq,
                "name": t.name,
                "status": t.status.value,
                "cache_hit": t.cache_hit,
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
            for t in tasks
        ],
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


def launch_script(
    root: Path,
    script: str | os.PathLike[str],
    *,
    name: str | None = None,
    intent: str | None = None,
    session: str | None = None,
    argv: tuple[str, ...] = (),
    capture_output: bool = False,
) -> dict[str, Any]:
    """Execute a workflow script inside a fresh run context; return the outcome.

    This is ``slab run`` and the MCP ``launch_workflow`` tool: the script is
    plain Python with ``@task`` calls and ``@check`` declarations — the runner
    supplies the workspace and the run context, so scripts carry zero
    ceremony. Scripts that manage their own ``Workspace.start_run`` should be
    executed with plain ``python`` instead (nesting is refused with a hint).

    *session* stamps the run with the client session that launched it; when
    omitted, ``$SLAB_SESSION`` applies (see
    :func:`foundation.runtime.resolve_session_id`).

    The result dict carries ``run_id``, final ``state``/``status``, check
    counts, and — with ``capture_output=True`` — everything the script printed
    (used by the MCP server, whose stdout is the protocol channel). If the
    script raised, the run's structured ``failure`` record (trimmed traceback
    and diagnostic notes, :func:`foundation.errors.failure_record`) is included; a
    raw ``traceback`` is the fallback for failures the run itself never saw
    (runner machinery).
    """
    script_path = Path(script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"no such workflow script: {script_path}")

    buffer = io.StringIO()
    error: str | None = None
    run_id: str | None = None
    try:
        workspace = Workspace(root)
    except Exception as e:
        raise StorageError(f"cannot open workspace at {root}: {e}") from e
    # The interpreter state is touched only once the workspace is open, so
    # a failed open (in a long-lived MCP process) leaves argv and path as
    # they were.
    old_argv = sys.argv
    sys.argv = [str(script_path), *argv]
    sys.path.insert(0, str(script_path.parent))
    with workspace as ws:
        try:
            # capture wraps the whole run context: @check hooks evaluate at
            # context exit, and their prints must not reach the real stdout
            # (under MCP, stdout is the protocol channel).
            with ExitStack() as stack:
                if capture_output:
                    stack.enter_context(redirect_stdout(buffer))
                    stack.enter_context(redirect_stderr(buffer))
                with ws.start_run(
                    name=name or script_path.stem, intent=intent, session=session
                ) as active:
                    run_id = active.id
                    # The script is the run's recompute root and its own
                    # best explanation. Kept by name, so show_run lists it
                    # and read_artifact reads it: one real lead searched
                    # three filesystems for a script the run record held.
                    active.keep(script_path.name, script_path, role=ArtifactRole.INPUT)
                    _execute_script(script_path)
        except NestedRunError:
            raise FoundationError(
                f"{script_path.name} manages its own runs (it calls start_run); "
                f"execute it with plain 'python {script_path.name}' instead of "
                f"'slab run'"
            ) from None
        except Exception:
            error = traceback.format_exc(limit=8)
        finally:
            sys.argv = old_argv
            sys.path.remove(str(script_path.parent))

        if run_id is None:
            # The run never started (unwritable database, storage failure...):
            # surface the real cause instead of pretending a run exists.
            raise StorageError(f"could not start a run for {script_path.name}:\n{error}")
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


def _execute_script(script_path: Path) -> None:
    """runpy the script, taming SystemExit: the `sys.exit(main())` idiom is
    everyday Python and must not escape into typer or the MCP task group."""
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise ScriptExitError(f"script called sys.exit({e.code!r})") from None
