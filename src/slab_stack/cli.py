"""The ``slab-stack`` command-line interface: the destructive pair, and memory.

``fast-forward`` and ``purge`` together are the "I am done with everything
I did not promote" gesture. They stay out of the per-package CLIs on
purpose: ``foundation``'s verbs each honor the retention policy, and
these two exist to override it — the command name should say whose rules
apply. Deletion only ever reaches the ``expired`` state, so the promoted
record survives any invocation.

``memory`` is here for the same layering reason rather than the same
destructive one. The store (:mod:`foundation.memory`) holds what agents
learned about this *machine*, so it belongs to no single project and to no
single package: mason writes it, foundation owns it, and the human reads
and prunes it from here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from foundation import _ops
from foundation import memory as memory_store
from foundation.errors import FoundationError
from foundation.runtime import Workspace
from mason.serve import mason_dir, read_record
from mason.session import transcript_groups
from slab._version import __version__
from slab.errors import SlabError
from slab.hpc import SchedulerNotAvailableError, active_job_ids

app = typer.Typer(
    help="slab-stack — housekeeping across the whole SLAB workspace.",
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

# A submitted job's files carry its id: <job_name>-<job_id>.sbatch / .out.
_JOB_FILE_ID = re.compile(r"-(\d+)\.(?:sbatch|out)$")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"slab-stack {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the version and exit.",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """slab-stack — housekeeping across the whole SLAB workspace."""


@app.command("fast-forward")
def fast_forward(
    workspace: _WorkspaceOpt = None,
    include_running: Annotated[
        bool,
        typer.Option(
            "--include-running",
            help="Also expire runs stuck at status 'running' (a hard-killed "
            "process never advances its own status). They are marked failed "
            "first. Off by default: a genuinely running job would be expired "
            "under itself.",
        ),
    ] = False,
) -> None:
    """Move every unpromoted run to expired, now.

    Promote what you intend to keep first; everything else becomes
    eligible for 'slab-stack purge'. A state change only — nothing is
    deleted until purge.
    """
    try:
        root = _ops.resolve_root(workspace)
        policy = _ops.ttl_override_policy(_ops.parse_duration_days("0s"))
        with Workspace(root) as ws:
            expired = ws.expire_due(policy, include_running=include_running)
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))
    for run in expired:
        typer.echo(f"expired {run.id}  {run.name}")
    typer.echo(f"{len(expired)} run(s) fast-forwarded to expired")


def _job_file_sweep(root: Path, active: frozenset[str]) -> list[Path]:
    """Finished jobs' scripts and SLURM output files, active jobs excluded.

    Two directories hold them: ``<workspace>/jobs`` (the agent's submitted
    jobs) and ``<workspace>/mason`` (serve jobs). Only ``*.sbatch`` and
    ``*.out`` are candidates, so the serve endpoint record is never
    touched. A file whose embedded job id is still in the queue is kept —
    SLURM is writing its ``.out``.
    """
    victims: list[Path] = []
    for directory in (root / "jobs", mason_dir(root)):
        if not directory.is_dir():
            continue
        for pattern in ("*.sbatch", "*.out"):
            for path in sorted(directory.glob(pattern)):
                match = _JOB_FILE_ID.search(path.name)
                if match is not None and match.group(1) in active:
                    continue
                victims.append(path)
    return victims


@app.command()
def purge(
    workspace: _WorkspaceOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would go; delete nothing.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Skip the confirmation prompt.")
    ] = False,
    all_sessions: Annotated[
        bool,
        typer.Option(
            "--all-sessions",
            help="Also delete the newest conversation transcript (kept by "
            "default so 'mason chat --resume' still works).",
        ),
    ] = False,
) -> None:
    """Delete all expired data for real: rows, bytes, transcripts, job files.

    Expired runs lose their database rows (run, transitions, artifact
    references, tasks, checks) and any artifact bytes no surviving run
    references. Session transcripts are deleted with their delegation
    siblings, except the newest conversation. Finished jobs' .sbatch and
    .out files are swept from the workspace; jobs still in the queue, the
    running model server included, keep theirs. Irreversible — run with
    --dry-run first.

    The machine's memory is not touched. This command clears project state
    that was never promoted; a memory is durable machine state that no
    project owns, so it is forgotten one at a time and on purpose, with
    'slab-stack memory forget'.
    """
    try:
        root = _ops.resolve_root(workspace)
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))

    groups = transcript_groups(root)
    if groups and not all_sessions:
        groups = groups[:-1]  # the newest conversation stays resumable
    try:
        active = active_job_ids()
    except SchedulerNotAvailableError:
        active = frozenset()  # no scheduler here, so nothing can be running
    except SlabError as e:
        _fail(str(e))
    record = read_record(root)
    if record is not None and record.job_id:
        active = active | {str(record.job_id)}
    job_files = _job_file_sweep(root, active)

    if not dry_run and not yes:
        typer.confirm(
            f"permanently delete every expired run, "
            f"{sum(1 + len(s) for _, s in groups)} transcript file(s), and "
            f"{len(job_files)} job file(s) from {root}?",
            abort=True,
        )

    try:
        with Workspace(root) as ws:
            report = ws.purge_expired(dry_run=dry_run)
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))
    verb = "would delete" if dry_run else "deleted"
    for run_id in report.deleted:
        typer.echo(f"{verb} run {run_id}")
    typer.echo(
        f"{verb} {len(report.deleted)} run(s); {verb} {len(report.dropped)} "
        f"blob(s), freeing {report.freed_bytes} bytes; "
        f"{len(report.kept)} blob(s) kept for surviving runs"
    )

    removed_transcripts = 0
    for conversation, siblings in groups:
        for path in (conversation, *siblings):
            typer.echo(f"{verb} transcript {path.name}")
            if not dry_run:
                path.unlink(missing_ok=True)
            removed_transcripts += 1
    if not all_sessions and transcript_groups(root):
        typer.echo("kept the newest conversation (--all-sessions removes it too)")

    for path in job_files:
        typer.echo(f"{verb} job file {path.name}")
        if not dry_run:
            path.unlink(missing_ok=True)
    typer.echo(
        f"{verb} {removed_transcripts} transcript file(s) and "
        f"{len(job_files)} job file(s)"
    )


memory_app = typer.Typer(
    help="Read and prune what agents learned about this machine.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")


@memory_app.command("list")
def memory_list(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the catalog as JSON.")
    ] = False,
) -> None:
    """List every memory on this machine: name, date, writer, description."""
    try:
        memories = memory_store.discover()
    except FoundationError as e:
        _fail(str(e))
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "name": m.name,
                        "description": m.description,
                        "path": str(m.path),
                        "created": m.created,
                        "updated": m.updated,
                        "agent": m.agent,
                        "model": m.model,
                    }
                    for m in memories.values()
                ],
                indent=2,
            )
        )
        return
    if not memories:
        typer.echo(f"no memories recorded yet ({memory_store.memory_dir()})")
        return
    width = max(len(name) for name in memories)
    for memory in memories.values():
        stamp = memory.updated or memory.created or "-"
        typer.echo(f"{memory.name:<{width}}  {stamp}  {memory.agent or '-':<16}  "
                   f"{memory.description}")
    typer.echo(f"{len(memories)} memory(s) in {memory_store.memory_dir()}")


@memory_app.command("show")
def memory_show(name: Annotated[str, typer.Argument(help="The memory's name.")]) -> None:
    """Print one memory whole, exactly as it is stored."""
    try:
        memories = memory_store.discover()
        if name not in memories:
            known = ", ".join(memories) or "none"
            _fail(f"no memory named {name!r} (memories here: {known})")
        typer.echo(memories[name].path.read_text(encoding="utf-8").rstrip())
    except (FoundationError, OSError) as e:
        _fail(str(e))


@memory_app.command("forget")
def memory_forget(
    name: Annotated[str, typer.Argument(help="The memory's name.")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete one memory.

    The only way a memory leaves the machine. Agents consolidate by
    rewriting, so nothing an agent does can erase a fact you still want.
    """
    try:
        memories = memory_store.discover()
        found = memories.get(name)
        if found is None:
            known = ", ".join(memories) or "none"
            _fail(f"no memory named {name!r} (memories here: {known})")
        if not yes:
            typer.echo(f"{found.name}: {found.description}")
            typer.confirm(f"permanently delete {found.path}?", abort=True)
        typer.echo(f"forgot {memory_store.delete(name)}")
    except FoundationError as e:
        _fail(str(e))


@memory_app.command("path")
def memory_path() -> None:
    """Print the memory directory, for reading or editing the files by hand."""
    typer.echo(memory_store.memory_dir())


if __name__ == "__main__":  # pragma: no cover - module execution convenience
    app()
