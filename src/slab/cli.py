"""The ``slab`` command-line interface.

Verbs mirror the lifecycle: ``run`` lands work in quarantine, ``list``/``show``
inspect it, ``promote`` makes it permanent, ``expire`` and ``gc`` are the
two-phase housekeeping. Every verb goes through :mod:`slab._ops`, the same
code paths the MCP server exposes to agents.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from slab import _ops
from slab._version import __version__
from slab.errors import SlabError
from slab.lifecycle import LifecycleState
from slab.models import utcnow
from slab.runtime import Workspace

app = typer.Typer(
    help="SLAB — agent-native workflow orchestration for atomistic materials modeling.",
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


def _open(workspace: Path | None) -> Workspace:
    return Workspace(_ops.resolve_root(workspace))


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _age(moment: datetime) -> str:
    seconds = max(0.0, (utcnow() - moment).total_seconds())
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86_400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86_400)}d"


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"slab {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the slab version and exit.",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """SLAB — runs are born ephemeral and promoted to permanent."""


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
) -> None:
    """Execute a workflow script; the run lands in quarantine.

    The script is plain Python: ``@task`` calls are traced and ``@check``
    declarations gate verification. Scripts that call ``start_run`` themselves
    should be executed with plain ``python`` instead.
    """
    try:
        result = _ops.launch_script(
            _ops.resolve_root(workspace),
            script,
            name=name,
            intent=intent,
            argv=tuple(args or ()),
        )
    except (SlabError, FileNotFoundError) as e:
        _fail(str(e))
    if result.get("traceback"):
        typer.echo(result["traceback"], err=True, nl=False)
    typer.echo(
        f"run {result['run_id']}  {result['name']}  "
        f"state={result['state']} status={result['status']} "
        f"checks={result['checks_passed']}/{result['checks_total']} "
        f"tasks={result['tasks_recorded']}"
    )
    if result["status"] == "failed":
        raise typer.Exit(code=1)


@app.command("list")
def list_(
    workspace: _WorkspaceOpt = None,
    state: Annotated[
        str | None, typer.Option(help="Filter by lifecycle state (e.g. quarantined).")
    ] = None,
    status: Annotated[
        str | None, typer.Option(help="Filter by execution status (e.g. completed).")
    ] = None,
    limit: Annotated[int, typer.Option(help="Maximum rows.")] = 20,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print run ids only.")] = False,
) -> None:
    """List runs, newest first."""
    with _open(workspace) as ws:
        try:
            runs = ws.runs.list_runs(state=state, status=status, limit=limit)
        except ValueError as e:
            _fail(str(e))
        if quiet:
            for item in runs:
                typer.echo(item.id)
            return
        if not runs:
            typer.echo("no runs")
            return
        header = f"{'ID':<12} {'STATE':<12} {'STATUS':<10} {'AGE':>5}  {'NAME':<20} INTENT"
        typer.echo(header)
        for item in runs:
            intent_text = (item.intent or "")[:40]
            typer.echo(
                f"{item.id[:10]:<12} {item.state.value:<12} {item.status.value:<10} "
                f"{_age(item.created_at):>5}  {item.name[:20]:<20} {intent_text}"
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
        except SlabError as e:
            _fail(str(e))
        if as_json:
            typer.echo(json.dumps(details, indent=2))
            return
        _render_details(details)


def _render_details(details: dict[str, object]) -> None:
    run = details["run"]
    assert isinstance(run, dict)
    typer.echo(f"run {run['id']}  {run['name']}")
    typer.echo(f"  state:   {run['state']}    status: {run['status']}")
    if run.get("error"):
        typer.echo(f"  error:   {run['error']}")
    typer.echo(f"  created: {run['created_at']}")
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

    tasks = details["tasks"]
    assert isinstance(tasks, list)
    if tasks:
        typer.echo("  tasks:")
        for position, t in enumerate(tasks, start=1):
            cached = " (cached)" if t["cache_hit"] else ""
            duration = "" if t["duration_s"] is None else f"  {t['duration_s']}s"
            error = "" if not t["error"] else f"  error: {t['error']}"
            typer.echo(f"    {position}. {t['name']}  {t['status']}{cached}{duration}{error}")

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


@app.command()
def promote(
    run_id: Annotated[str, typer.Argument(help="Run id or unique prefix.")],
    workspace: _WorkspaceOpt = None,
    reason: Annotated[str | None, typer.Option(help="Why this run is worth keeping.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Promote a run that was never verified.")
    ] = False,
) -> None:
    """Make a run permanent: verified -> promoted (--force: quarantined -> promoted)."""
    with _open(workspace) as ws:
        try:
            updated = ws.runs.transition(
                run_id, LifecycleState.PROMOTED, actor="user", reason=reason, force=force
            )
        except SlabError as e:
            _fail(str(e))
        typer.echo(f"promoted {updated.id}  {updated.name}")


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
) -> None:
    """Expire unpromoted runs that outlived their TTL (state change only; see gc)."""
    root = _ops.resolve_root(workspace)
    try:
        if older_than is not None:
            policy = _ops.ttl_override_policy(_ops.parse_duration_days(older_than))
        else:
            policy = _ops.load_policy(root, policy_file)
    except (ValueError, OSError) as e:
        _fail(str(e))
    with Workspace(root) as ws:
        expired = ws.expire_due(policy)
        for item in expired:
            typer.echo(f"expired {item.id}  {item.name}")
        typer.echo(f"{len(expired)} run(s) expired")


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
    root = _ops.resolve_root(workspace)
    try:
        policy = _ops.load_policy(root, policy_file)
    except (ValueError, OSError) as e:
        _fail(str(e))
    with Workspace(root) as ws:
        report = ws.gc(policy, dry_run=dry_run)
    verb = "would drop" if dry_run else "dropped"
    typer.echo(
        f"{verb} {len(report.dropped)} blob(s), freeing {report.freed_bytes} bytes; "
        f"{len(report.kept)} kept"
    )
    if report.orphans:
        typer.echo(f"orphans (unreferenced, not deleted): {len(report.orphans)}")
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
        from slab.mcp_server import serve
    except ImportError:
        _fail("the MCP server needs the mcp package: pip install 'slab[mcp]'")
    serve(_ops.resolve_root(workspace))  # pragma: no cover - blocks on stdio


def main() -> None:  # pragma: no cover - console-script shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
