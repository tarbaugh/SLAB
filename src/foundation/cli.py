"""The run-lifecycle verbs of the ``slab`` command.

Verbs mirror the lifecycle: ``run`` lands work in quarantine, ``list``/``show``
inspect it, ``promote`` makes it permanent, ``expire`` and ``gc`` are the
two-phase housekeeping. Every verb goes through :mod:`foundation._ops`, the
same code paths the MCP server exposes to agents.

The front door that mounts them is :mod:`slab_stack.cli`; there they sit at
the top level (``slab run``, ``slab list``, ``slab promote``). Engines,
protocols, pseudopotentials, and the scheduler are the machine groups; the
resident agent is ``slab mason``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from foundation import _ops
from foundation.errors import FoundationError
from foundation.lifecycle import LifecycleState
from foundation.models import Reservation, utcnow
from foundation.runtime import Workspace, describe_liveness, run_liveness, this_host
from foundation.store import process_alive
from slab.errors import SlabError

app = typer.Typer(
    help="Foundation — workflows, runs, and state for SLAB.",
    no_args_is_help=True,
    add_completion=False,
)


_WorkspaceOpt = Annotated[
    Path | None,
    typer.Option(
        "--workspace",
        "-w",
        envvar="SLAB_WORKSPACE",
        help="Workspace directory (default ./.slab).",
    ),
]


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _state_text(state: str, status: str) -> str:
    """The lifecycle state as a reader should take it.

    Every run is born quarantined, so a running run that reads
    ``quarantined`` is in its initial state, not in trouble.

    Examples:
        >>> _state_text("quarantined", "running")
        'quarantined (initial state)'
        >>> _state_text("quarantined", "failed")
        'quarantined'
    """
    if state == "quarantined" and status == "running":
        return "quarantined (initial state)"
    return state


def _age(moment: datetime) -> str:
    seconds = max(0.0, (utcnow() - moment).total_seconds())
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86_400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86_400)}d"


def _open(workspace: Path | None) -> Workspace:
    """Open the workspace, reporting every failure as an error line.

    ``Workspace`` creates directories and opens SQLite, so beyond the domain
    errors (a broken config file, a database from a newer foundation) the
    open can raise the OSError family: the path is a file, or its parent is
    not writable. The CLI's contract is that all of them print ``error: ...``
    and exit 1, never a traceback.
    """
    try:
        return Workspace(_ops.resolve_root(workspace))
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))


@app.command()
def run(
    script: Annotated[Path, typer.Argument(help="Workflow script (plain Python).")],
    args: Annotated[
        list[str] | None, typer.Argument(help="Arguments passed to the script.")
    ] = None,
    workspace: _WorkspaceOpt = None,
    name: Annotated[str | None, typer.Option(help="Run name (default: script stem).")] = None,
    intent: Annotated[
        str | None, typer.Option(help="Why this run exists — narrative provenance.")
    ] = None,
    session: Annotated[
        str | None,
        typer.Option(
            "--session",
            envvar="SLAB_SESSION",
            help="Stamp the run with the client session that launched it.",
        ),
    ] = None,
    reservation: Annotated[
        str | None,
        typer.Option(
            "--reservation",
            help="Claim this reservation (made by the launching session) and run inside "
            "its slice: affinity mask, OMP_NUM_THREADS, CUDA_VISIBLE_DEVICES.",
        ),
    ] = None,
) -> None:
    """Execute a workflow script; the run lands in quarantine.

    The script is plain Python: ``@task`` calls are traced and ``@check``
    declarations gate verification. Scripts that call ``start_run`` themselves
    should be executed with plain ``python`` instead.
    """
    root = _ops.resolve_root(workspace)
    if reservation is not None:
        _enter_reservation(root, reservation)
    try:
        result = _ops.launch_script(
            root,
            script,
            name=name,
            intent=intent,
            session=session,
            argv=tuple(args or ()),
            reservation=reservation,
        )
    except (FoundationError, SlabError, FileNotFoundError) as e:
        _fail(str(e))
    failure = result.get("failure")
    if failure:
        typer.echo(failure["traceback"], err=True)
    elif result.get("traceback"):
        typer.echo(result["traceback"], err=True, nl=False)
    typer.echo(
        f"run {result['run_id']}  {result['name']}  "
        f"state={result['state']} status={result['status']} "
        f"checks={result['checks_passed']}/{result['checks_total']} "
        f"tasks={result['tasks_recorded']}"
    )
    # Anything but a clean completion exits nonzero — including a run left at
    # status 'running' because recording its failure itself failed.
    if result["status"] != "completed":
        raise typer.Exit(code=1)


def _enter_reservation(root: Path, reservation_id: str) -> None:
    """Take the reservation's slice before the script runs, so every rank inherits it."""
    import os

    from slab.resources import apply, env_for

    with _open(root) as ws:
        try:
            held = ws.runs.get_reservation(reservation_id)
        except FoundationError as e:
            _fail(str(e))
    envelope = held.envelope()
    os.environ.update(env_for(envelope))
    apply(envelope)


@app.command("list")
def list_(
    workspace: _WorkspaceOpt = None,
    state: Annotated[
        str | None, typer.Option(help="Filter by lifecycle state (e.g. quarantined).")
    ] = None,
    status: Annotated[
        str | None, typer.Option(help="Filter by execution status (e.g. completed).")
    ] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", help="Only runs from this session (id or unique prefix)."),
    ] = None,
    limit: Annotated[int, typer.Option(help="Maximum rows.")] = 20,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print run ids only.")] = False,
) -> None:
    """List runs, newest first."""
    with _open(workspace) as ws:
        try:
            runs = ws.runs.list_runs(state=state, status=status, session=session, limit=limit)
        except (FoundationError, ValueError) as e:
            _fail(str(e))
        if quiet:
            for item in runs:
                typer.echo(item.id)
            return
        if not runs:
            typer.echo("no runs")
            return
        header = (
            f"{'ID':<12} {'STATE':<12} {'STATUS':<10} {'AGE':>5}  {'RES':<7} {'NAME':<20} INTENT"
        )
        typer.echo(header)
        for item in runs:
            intent_text = (item.intent or "")[:40]
            state_text = _state_text(item.state.value, item.status.value)
            typer.echo(
                f"{item.id[:10]:<12} {state_text:<12} {item.status.value:<10} "
                f"{_age(item.created_at):>5}  {_ops.resources_column(item.resources):<7} "
                f"{item.name[:20]:<20} {intent_text}"
            )


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="Run id or unique prefix.")],
    workspace: _WorkspaceOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show one run: state, intent, checks, tasks, artifacts, history."""
    with _open(workspace) as ws:
        try:
            details = _ops.run_details(ws, run_id)
        except (FoundationError, SlabError) as e:
            _fail(str(e))
        if as_json:
            typer.echo(json.dumps(details, indent=2))
            return
        _render_details(details)


def _render_details(details: dict[str, object]) -> None:
    run = details["run"]
    assert isinstance(run, dict)
    tasks = details["tasks"]
    assert isinstance(tasks, list)
    typer.echo(f"run {run['id']}  {run['name']}")
    state_text = _state_text(str(run["state"]), str(run["status"]))
    typer.echo(f"  state:   {state_text}    status: {run['status']}")
    if run["status"] == "running" and run.get("pid") is not None:
        typer.echo(f"  process: {run['pid']} on {run['host']}")
    resources = run.get("resources")
    if isinstance(resources, dict):
        typer.echo(f"  resources: {_ops.describe_resources(resources)}")
    if run.get("error"):
        typer.echo(f"  error:   {run['error']}")
    # The run's failure renders here unless a failed task carries the SAME
    # exception (it propagated; that task renders it below). A task failure
    # the script caught before failing differently must not hide the run's.
    run_failure = run.get("failure")
    if isinstance(run_failure, dict) and not _explained_by_task(run_failure, tasks):
        _echo_failure(run_failure, indent="    ")
    typer.echo(f"  created: {run['created_at']}")
    if run.get("session"):
        typer.echo(f"  session: {run['session']}")
    if run.get("intent"):
        typer.echo(f"  intent:  {run['intent']}")

    checks = details["checks"]
    assert isinstance(checks, list)
    if checks:
        passed = sum(1 for c in checks if c["passed"])
        typer.echo(f"  checks:  {passed}/{len(checks)} passed")
        for c in checks:
            mark = "+" if c["passed"] else "x"
            typer.echo(f"    [{mark}] {c['name']}: {c['message']}")

    if tasks:
        typer.echo("  tasks:")
        for position, t in enumerate(tasks, start=1):
            cached = " (cached)" if t["cache_hit"] else ""
            duration = "" if t["duration_s"] is None else f"  {t['duration_s']}s"
            error = "" if not t["error"] else f"  error: {t['error']}"
            typer.echo(f"    {position}. {t['name']}  {t['status']}{cached}{duration}{error}")
            if t.get("failure"):
                _echo_failure(t["failure"], indent="       ")

    artifacts = details["artifacts"]
    assert isinstance(artifacts, list)
    if artifacts:
        typer.echo("  artifacts:")
        for a in artifacts:
            presence = "bytes" if a["bytes_available"] else "hash-only"
            typer.echo(
                f"    {a['name']}  {a['role']}  {a['size_bytes']}B  {presence}  {a['hash'][:12]}"
            )

    history = details["history"]
    assert isinstance(history, list)
    if history:
        typer.echo("  history:")
        for h in history:
            forced = " (forced)" if h["forced"] else ""
            reason = "" if not h["reason"] else f": {h['reason']}"
            typer.echo(f"    {h['from']} -> {h['to']}{forced}  by {h['actor']}{reason}")


def _explained_by_task(run_failure: dict[str, object], tasks: list[dict[str, object]]) -> bool:
    """True if a failed task carries the same exception as the run failure."""
    return any(
        (f := t.get("failure")) is not None
        and isinstance(f, dict)
        and f.get("type") == run_failure.get("type")
        and f.get("message") == run_failure.get("message")
        for t in tasks
    )


def _echo_failure(failure: object, indent: str) -> None:
    """Print a failure record's traceback, indented (notes appear at its end)."""
    assert isinstance(failure, dict)
    for line in str(failure.get("traceback") or "").splitlines():
        typer.echo(f"{indent}{line}")


@app.command()
def promote(
    run_ids: Annotated[
        list[str] | None, typer.Argument(help="Run ids or unique prefixes.")
    ] = None,
    workspace: _WorkspaceOpt = None,
    session: Annotated[
        str | None,
        typer.Option(
            "--session",
            help="Promote every run this session created (id or unique prefix).",
        ),
    ] = None,
    reason: Annotated[str | None, typer.Option(help="Why this run is worth keeping.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Promote a run that was never verified.")
    ] = False,
) -> None:
    """Make runs permanent: verified -> promoted (--force: quarantined -> promoted).

    Name the runs, or name the session that created them with ``--session``.
    A session promote reports every run it considered: it promotes the
    verified ones, skips the unverified ones unless ``--force`` is given, and
    never promotes a failed run. List the sessions with ``slab sessions``.
    """
    if (run_ids and session) or not (run_ids or session):
        _fail("give run ids or --session, not both (and not neither)")
    with _open(workspace) as ws:
        if session is not None:
            _promote_session(ws, session, reason=reason, force=force)
            return
        failures = 0
        for run_id in run_ids or []:
            try:
                updated = ws.runs.transition(
                    run_id, LifecycleState.PROMOTED, actor="user", reason=reason, force=force
                )
            except (FoundationError, SlabError) as e:
                typer.echo(f"error: {e}", err=True)
                failures += 1
                continue
            typer.echo(f"promoted {updated.id}  {updated.name}")
        if failures:
            raise typer.Exit(code=1)


def _promote_session(ws: Workspace, session: str, *, reason: str | None, force: bool) -> None:
    """Promote one session's runs and report every outcome, one line each."""
    try:
        result = _ops.promote_session(ws, session, reason=reason, force=force)
    except (FoundationError, SlabError) as e:
        _fail(str(e))
    marks = {"promoted": "+", "already": "=", "skipped": "-"}
    for item in result["outcomes"]:
        typer.echo(
            f"  [{marks[item['outcome']]}] {item['id'][:10]}  {item['name'][:20]:<20} "
            f"{item['outcome']:<8} {item['detail']}"
        )
    typer.echo(
        f"session {result['session']}: {result['promoted']} promoted, "
        f"{result['already']} already permanent, {result['skipped']} skipped"
    )
    if not result["complete"]:
        raise typer.Exit(code=1)


@app.command()
def retire(
    session: Annotated[
        str,
        typer.Option("--session", help="The session to retire (id or unique prefix)."),
    ],
    workspace: _WorkspaceOpt = None,
    keep: Annotated[
        list[str] | None,
        typer.Option(
            "--keep",
            help="A run the session's answer rests on (id or unique prefix); repeatable. "
            "Verified ones are promoted; the rest are reported.",
        ),
    ] = None,
    uncited: Annotated[
        str | None,
        typer.Option(
            "--uncited",
            help="What happens to the session's other runs: keep, expire, or purge "
            "(default: the retention policy's finish rule, expire).",
        ),
    ] = None,
    reason: Annotated[
        str | None, typer.Option(help="Why the kept runs are worth keeping.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would happen; write nothing.")
    ] = False,
) -> None:
    """Promote the runs a session's answer cites; expire the session's other runs.

    This is what Mason does after a campaign's finish. Run it by hand for a
    session whose finish never ran, or to check one with ``--dry-run``.
    A kept run is never forced: an unverified one is reported and left as
    it is. The exit code is 1 while any kept run was skipped.
    """
    try:
        root = _ops.resolve_root(workspace)
        mode = uncited if uncited is not None else _ops.load_policy(root).finish.uncited
        if mode not in _ops.RETIRE_MODES:
            raise ValueError(f"--uncited must be keep, expire, or purge, not {mode!r}")
    except (FoundationError, SlabError, ValueError, OSError) as e:
        _fail(str(e))
    try:
        with Workspace(root) as ws:
            result = _ops.retire_session(
                ws,
                session,
                keep=keep or [],
                mode=mode,
                reason=reason,
                actor="user",
                dry_run=dry_run,
            )
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))
    if not result["runs_total"] and not keep:
        _fail(f"no run carries session {session!r}; list the sessions with 'slab sessions'")
    _echo_retire(result)
    if not result["complete"]:
        raise typer.Exit(code=1)


def _echo_retire(result: dict[str, object]) -> None:
    """One line per run, then the numbers; the marks are ``slab promote``'s."""
    marks = {"promoted": "+", "already": "=", "skipped": "-"}
    would = "would be " if result["dry_run"] else ""
    for item in result["kept"]:  # type: ignore[attr-defined]
        typer.echo(
            f"  [{marks[item['outcome']]}] {item['id'][:10]}  {str(item['name'])[:20]:<20} "
            f"{item['outcome']:<8} {item['detail']}"
        )
    for item in result["expired"]:  # type: ignore[attr-defined]
        typer.echo(
            f"  [x] {item['id'][:10]}  {item['name'][:20]:<20} expired  "
            f"{would}uncited by the finish"
        )
    for item in result["skipped"]:  # type: ignore[attr-defined]
        typer.echo(f"  [-] {item['id'][:10]}  {item['name'][:20]:<20} skipped  {item['detail']}")
    purged = result["purged"]
    if purged:
        typer.echo(
            f"  purged {len(purged['deleted'])} run(s), "  # type: ignore[index]
            f"{purged['freed_bytes']} bytes freed"  # type: ignore[index]
        )
    typer.echo(
        f"session {result['session']}: {result['runs_promoted']} promoted, "
        f"{result['runs_expired']} expired, {len(result['skipped'])} skipped "  # type: ignore[arg-type]
        f"of {result['runs_total']} run(s); "
        f"bytes {result['bytes_promoted']} promoted, {result['bytes_expired']} expired, "
        f"{result['bytes_total']} in the session"
        + (" (dry run)" if result["dry_run"] else "")
    )


@app.command()
def sessions(
    workspace: _WorkspaceOpt = None,
    limit: Annotated[int, typer.Option(help="Maximum rows.")] = 20,
) -> None:
    """List the client sessions that created runs, newest first.

    Each row is one conversation. Promote a whole row with
    ``slab promote --session <id>``.
    """
    with _open(workspace) as ws:
        try:
            summary = _ops.sessions_summary(ws, limit=limit)
        except ValueError as e:
            _fail(str(e))
        rows = summary["sessions"]
        if not rows:
            typer.echo("no sessions")
        else:
            typer.echo(f"{'SESSION':<26} {'RUNS':>4} {'AGE':>5}  STATES")
            for row in rows:
                typer.echo(
                    f"{row['session'][:26]:<26} {row['runs']:>4} "
                    f"{_age(datetime.fromisoformat(row['newest_at'])):>5}  {row['breakdown']}"
                )
        if summary["unstamped"]:
            typer.echo(f"({summary['unstamped']} run(s) carry no session)")


@app.command()
def expire(
    workspace: _WorkspaceOpt = None,
    older_than: Annotated[
        str | None,
        typer.Option(
            "--older-than",
            help="Override policy TTLs, e.g. 30d / 12h (0d = everything unpromoted).",
        ),
    ] = None,
    policy_file: Annotated[
        Path | None, typer.Option("--policy", help="Retention policy JSON file.")
    ] = None,
    include_running: Annotated[
        bool,
        typer.Option(
            "--include-running",
            help="Also expire overdue runs stuck at status 'running' (a hard-killed "
            "process never advances its own status). They are marked failed first.",
        ),
    ] = False,
) -> None:
    """Expire unpromoted runs that outlived their TTL (state change only; see gc)."""
    try:
        root = _ops.resolve_root(workspace)
        if older_than is not None:
            policy = _ops.ttl_override_policy(_ops.parse_duration_days(older_than))
        else:
            policy = _ops.load_policy(root, policy_file)
    except (FoundationError, SlabError, ValueError, OSError) as e:
        _fail(str(e))
    try:
        with Workspace(root) as ws:
            expired = ws.expire_due(policy, include_running=include_running)
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))
    for item in expired:
        typer.echo(f"expired {item.id}  {item.name}")
    typer.echo(f"{len(expired)} run(s) expired")


runs_app = typer.Typer(
    help="Run liveness: reap the runs whose process is gone, or retire one by hand.",
    no_args_is_help=True,
    add_completion=False,
)


@runs_app.command("reap")
def runs_reap(workspace: _WorkspaceOpt = None) -> None:
    """Mark failed every running run whose recorded process on this host is gone,
    and release every reservation no live process holds."""
    with _open(workspace) as ws:
        released = ws.release_dead()
        for held in released:
            typer.echo(f"released {held.id}  {_describe_reservation(held)}")
        reaped = ws.reap_dead(caller="slab runs reap")
        for item in reaped:
            typer.echo(f"failed  {item.id}  {item.name}  {item.error}")
        for item in ws.runs.list_runs(status="running"):
            typer.echo(f"running {item.id}  {item.name}  {describe_liveness(item)}")
    typer.echo(f"{len(reaped)} run(s) marked failed, {len(released)} reservation(s) released")


@runs_app.command("reservations")
def runs_reservations(workspace: _WorkspaceOpt = None) -> None:
    """List the reservations in the store: the slice, the holder, the run, the age."""
    with _open(workspace) as ws:
        rows = ws.runs.list_reservations()
        if not rows:
            typer.echo("no reservations")
            return
        free = ws.free_resources()
        for held in rows:
            typer.echo(f"{held.id}  {_describe_reservation(held)}")
        typer.echo(
            f"free on {free['host']}: {len(free['free']['cpus'])} of "
            f"{len(free['budget']['cpus'])} cpu(s), {len(free['free']['gpus'])} of "
            f"{len(free['budget']['gpus'])} gpu(s)"
        )


def _describe_reservation(held: Reservation) -> str:
    """One line for a reservation: slice, then who holds it and for how long."""
    slice_text = _ops.describe_resources(held.slice)
    age = _age(held.created_at)
    if held.run_id is not None:
        return f"{slice_text}  claimed by run {held.run_id}  {age}"
    if held.host != this_host():
        alive = "not this host"
    else:
        alive = "alive" if process_alive(held.holder_pid) else "gone"
    return f"{slice_text}  unclaimed, holder {held.holder_pid} on {held.host} ({alive})  {age}"


@runs_app.command("fail")
def runs_fail(
    run_id: Annotated[str, typer.Argument(help="Run id or unique prefix.")],
    reason: Annotated[
        str, typer.Option("--reason", help="Why the operator retires it; recorded on the run.")
    ],
    workspace: _WorkspaceOpt = None,
) -> None:
    """Mark one running run failed by hand. Refused while its process is alive on this host."""
    with _open(workspace) as ws:
        try:
            before = ws.runs.get(run_id)
            failed = ws.fail_run(before.id, reason=reason)
        except (FoundationError, ValueError) as e:
            _fail(str(e))
        typer.echo(f"failed {failed.id}  {failed.name}  {failed.error}")
        if run_liveness(before) != "gone":
            typer.echo(f"(not checked: {describe_liveness(before)})")


@app.command()
def gc(
    workspace: _WorkspaceOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be dropped; delete nothing.")
    ] = False,
    policy_file: Annotated[
        Path | None, typer.Option("--policy", help="Retention policy JSON file.")
    ] = None,
) -> None:
    """Drop artifact bytes no retention rule demands (hashes and recipes survive)."""
    try:
        root = _ops.resolve_root(workspace)
        policy = _ops.load_policy(root, policy_file)
    except (FoundationError, SlabError, ValueError, OSError) as e:
        _fail(str(e))
    try:
        with Workspace(root) as ws:
            report = ws.gc(policy, dry_run=dry_run)
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))
    verb = "would drop" if dry_run else "dropped"
    typer.echo(
        f"{verb} {len(report.dropped)} blob(s), freeing {report.freed_bytes} bytes; "
        f"{len(report.kept)} kept"
    )
    ttl = policy.orphan_ttl_days
    if report.orphans_dropped:
        typer.echo(
            f"{verb} {len(report.orphans_dropped)} orphan blob(s) unreferenced for "
            f"{ttl:g} day(s) or more"
        )
    if report.orphans:
        why = "orphan_ttl_days is null" if ttl is None else f"younger than {ttl:g} day(s)"
        typer.echo(f"orphans kept (unreferenced, {why}): {len(report.orphans)}")
    if report.missing:
        typer.echo(
            f"WARNING: {len(report.missing)} demanded blob(s) missing from the store",
            err=True,
        )


@app.command()
def mcp(
    workspace: _WorkspaceOpt = None,
) -> None:
    """Serve the workspace to agents over MCP (stdio transport)."""
    try:
        from foundation.mcp_server import serve
    except ImportError:
        _fail("the MCP server needs the mcp package: pip install 'slab-stack[mcp]'")
    try:
        root = _ops.resolve_root(workspace)
    except (FoundationError, SlabError) as e:
        _fail(str(e))
    serve(root, project=Path.cwd())  # pragma: no cover - blocks on stdio


if __name__ == "__main__":  # pragma: no cover - module execution convenience
    app()
