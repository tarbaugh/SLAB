"""The ``slab`` command-line interface: the front door to the whole stack.

One command, grouped by intent. The lifecycle verbs (``run``, ``list``,
``show``, ``promote``, ``sessions``, ``expire``, ``gc``, ``mcp``) come from
:mod:`foundation.cli`; the machine groups (``engines``, ``pseudos``,
``protocols``, ``hpc``, ``config``) come from :mod:`slab.cli`; the resident
agent mounts whole as ``slab mason`` from :mod:`mason.cli`. This module
composes them, because ``slab_stack`` is the one package allowed to import
all three layers.

Three families are implemented here rather than mounted. ``fast-forward``
and ``purge`` together are the "I am done with everything I did not
promote" gesture: the lifecycle verbs each honor the retention policy, and
these two exist to override it. Deletion only ever reaches the ``expired``
state, so the promoted record survives any invocation. ``memory`` is here
for the layering reason: the store (:mod:`foundation.memory`) holds what
agents learned about this *machine*, so it belongs to no single project
and to no single package — mason writes it, foundation owns it, and the
human reads and prunes it from here. ``benchmark`` runs the fixed research
campaigns through mason and scores them against foundation's run record
(:mod:`slab_stack.benchmark`), which is every layer at once.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from foundation import _ops
from foundation import cli as foundation_cli
from foundation import memory as memory_store
from foundation.errors import FoundationError
from foundation.runtime import Workspace
from mason.cli import app as mason_app
from mason.errors import MasonError
from mason.serve import mason_dir, read_record
from mason.session import transcript_groups
from slab._version import __version__
from slab.cli import config_app, engines_app, hpc_app, mp_app, protocols_app, pseudos_app
from slab.errors import SlabError
from slab.hpc import SchedulerNotAvailableError, active_job_ids
from slab_stack import benchmark

_PANEL_LIFECYCLE = "Runs and lifecycle"
_PANEL_HOUSEKEEPING = "Housekeeping"
_PANEL_MACHINE = "This machine"
_PANEL_AGENT = "The resident agent"
_PANEL_INTEGRATION = "Integration"
_PANEL_DOCTOR = "Doctor"

app = typer.Typer(
    help="SLAB — runs, engines, the resident agent, and the housekeeping.",
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
        typer.echo(f"slab {__version__}")
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
    """SLAB — runs, engines, the resident agent, and the housekeeping."""


@app.command("fast-forward", rich_help_panel=_PANEL_HOUSEKEEPING)
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
    eligible for 'slab purge'. A state change only — nothing is
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


@app.command(rich_help_panel=_PANEL_HOUSEKEEPING)
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
            "default so 'slab mason chat --resume' still works).",
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
    'slab memory forget'.
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


@app.command(rich_help_panel=_PANEL_DOCTOR)
def doctor(
    workspace: _WorkspaceOpt = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Skip the endpoint and roster probes.")
    ] = False,
    deep: Annotated[
        bool,
        typer.Option(
            "--deep",
            help="Also run one real single-point per declared rootstock "
            "checkpoint — slow, possibly GPU-bound; meant for the cluster, "
            "before a campaign.",
        ),
    ] = False,
) -> None:
    """The whole-stack preflight: is this machine ready to launch a campaign?

    Probes the real campaign path in order: configuration, workspace,
    memory store, engines, scheduler, model endpoint, sandbox, and the
    freshness of the rendered job. Exits nonzero only on an [x] row; an
    [=] row is a fact about this machine, not a failure. The focused
    endpoint check remains 'slab mason doctor'.
    """
    from slab_stack import doctor as stack_doctor

    failures = stack_doctor.run(
        workspace, offline=offline, deep=deep, emit=typer.echo
    )
    if failures:
        raise typer.Exit(code=1)


memory_app = typer.Typer(
    help="Read and prune what agents learned about this machine.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory", rich_help_panel=_PANEL_HOUSEKEEPING)


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


@memory_app.command("purge")
def memory_purge(
    patterns: Annotated[
        list[str] | None,
        typer.Argument(
            help="Shell-style glob(s) matched against memory names "
            "(e.g. 'rootstock-*'). No pattern matches every memory.",
        ),
    ] = None,
    before: Annotated[
        str | None,
        typer.Option(
            "--before",
            help="Only memories last updated before this date (YYYY-MM-DD).",
        ),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete every memory that matches, in one confirmed gesture.

    Use it after a change that makes a family of memories stale: a harness
    fix that retires the workarounds agents recorded, or a machine
    reinstall. Filters combine: a memory must match a pattern (any of
    them) and, with ``--before``, be older than the date. The command
    lists what will go and asks once.
    """
    from fnmatch import fnmatch

    if before is not None:
        try:
            date.fromisoformat(before)
        except ValueError:
            _fail(f"--before wants YYYY-MM-DD, got {before!r}")
    try:
        memories = memory_store.discover()
    except FoundationError as e:
        _fail(str(e))
    def hits(memory: memory_store.Memory) -> bool:
        if patterns and not any(fnmatch(memory.name, p) for p in patterns):
            return False
        if before is None:
            return True
        stamp = memory.updated or memory.created
        # An undated file (hand-made) never matches an age filter.
        return stamp is not None and str(stamp) < before

    matched = [m for m in memories.values() if hits(m)]
    if not matched:
        typer.echo(f"nothing matched (memories here: {len(memories)})")
        return
    for memory in matched:
        typer.echo(f"{memory.name}: {memory.description}")
    if not yes:
        typer.confirm(
            f"permanently delete these {len(matched)} of {len(memories)} memory(s)?",
            abort=True,
        )
    try:
        for memory in matched:
            memory_store.delete(memory.name)
    except FoundationError as e:
        _fail(str(e))
    typer.echo(f"purged {len(matched)} memory(s) from {memory_store.memory_dir()}")


@memory_app.command("path")
def memory_path() -> None:
    """Print the memory directory, for reading or editing the files by hand."""
    typer.echo(memory_store.memory_dir())


# -- the benchmark ------------------------------------------------------------

benchmark_app = typer.Typer(
    help="Run the fixed research campaigns per model and score the answers.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark", rich_help_panel=_PANEL_AGENT)

_BENCH_ERRORS = (benchmark.BenchmarkError, MasonError, FoundationError, SlabError, OSError)
_MachineOpt = Annotated[
    str | None,
    typer.Option(
        "--machine",
        help="A label you choose for this machine (default: the compute profile). "
        "Never a hostname.",
    ),
]
_RecordsOpt = Annotated[
    Path | None,
    typer.Option("--records", help="The records file (default benchmarks/results.jsonl)."),
]


def _record_line(record: dict[str, Any]) -> str:
    verdict = "pass" if record["passed"] else f"fail: {record['reason']}"
    return (
        f"Q{record['question']} {record['key']:<9} {record['model']:<24} "
        f"{record['machine']:<12} {verdict}"
    )


@benchmark_app.command("list")
def benchmark_list() -> None:
    """The questions, their result keys, and what passes."""
    for question in benchmark.QUESTIONS:
        typer.echo(f"{question.number}. [{question.key}] {question.instruction}")
        keys = ", ".join(f"{name} ({unit})" for name, unit in question.results.items())
        typer.echo(f"   results: {keys}")
        for cls in (benchmark.DFT, benchmark.MLIP):
            band = question.tolerance[cls]
            if question.kind == "threshold":
                rule = ", ".join(f"{k} ≤ {v:g}" for k, v in band.items())
            else:
                rule = ", ".join(
                    f"{k} = {question.reference[cls][k]:g} ± {v:g}" for k, v in band.items()
                )
            typer.echo(f"   {cls}: {rule}")
        if question.experiment:
            typer.echo(f"   experiment: {question.experiment}")


@benchmark_app.command("run")
def benchmark_run(
    question: Annotated[str, typer.Argument(help="Question number or key (see list).")],
    workspace: _WorkspaceOpt = None,
    model: Annotated[str | None, typer.Option("--model", help="Model to run under.")] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    endpoint: Annotated[str | None, typer.Option("--endpoint")] = None,
    max_turns: Annotated[int | None, typer.Option("--max-turns", min=1)] = None,
    agent: Annotated[str | None, typer.Option("--agent", help="Entry card (default pi).")] = None,
    machine: _MachineOpt = None,
    records: _RecordsOpt = None,
) -> None:
    """Run one campaign here, autonomously, then score and record it.

    For a laptop or an interactive node. On a cluster, prefer 'launch',
    which runs the campaign as a sandbox job, then 'score' after it ends.
    """
    try:
        asked = benchmark.find_question(question)
        session_id, result = benchmark.run_campaign(
            asked,
            workspace=workspace,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_turns=max_turns,
            agent=agent,
        )
        typer.echo(
            f"session {session_id}: stopped by {result.stop_reason} "
            f"after {result.steps} step(s)"
        )
        root = _ops.resolve_root(workspace)
        record = benchmark.score_session(root, session_id, question=asked, machine=machine)
        path = records or benchmark.records_path()
        benchmark.append_record(path, record)
    except _BENCH_ERRORS as e:
        _fail(str(e))
    typer.echo(_record_line(record))
    typer.echo(f"recorded in {path}")


@benchmark_app.command("launch")
def benchmark_launch(
    question: Annotated[str, typer.Argument(help="Question number or key (see list).")],
    workspace: _WorkspaceOpt = None,
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition for the engine legs.")
    ] = None,
    time_limit: Annotated[str | None, typer.Option("--time-limit")] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for the rendered files (default: ./sandbox)."),
    ] = None,
) -> None:
    """Submit one campaign as a sandbox job; score it with 'score' after it ends."""
    from mason.cli import launch_sandbox

    out_dir = (out if out is not None else Path.cwd() / "sandbox").resolve()
    try:
        asked = benchmark.find_question(question)
        job = launch_sandbox(
            asked.instruction,
            workspace=workspace,
            partition=partition,
            time_limit=time_limit,
            out_dir=out_dir,
            engine_tasks=None,
            emit=typer.echo,
        )
    except _BENCH_ERRORS as e:
        _fail(str(e))
    typer.echo(
        f"submitted job {job.job_id} ({job.job_name}) to {job.partition} for Q{asked.number}"
    )
    typer.echo(f"watch it with 'slab hpc status {job.job_id}'; when it ends: slab benchmark score")


@benchmark_app.command("score")
def benchmark_score(
    workspace: _WorkspaceOpt = None,
    session: Annotated[
        list[str] | None,
        typer.Option(
            "--session",
            help="Session id or prefix; repeatable. Default: every unscored campaign.",
        ),
    ] = None,
    question: Annotated[
        str | None,
        typer.Option("--question", help="Score the named sessions as this question."),
    ] = None,
    machine: _MachineOpt = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the model the transcript names.")
    ] = None,
    rescore: Annotated[
        bool, typer.Option("--rescore", help="Score sessions already recorded.")
    ] = False,
    records: _RecordsOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the records as JSON.")] = False,
) -> None:
    """Score campaigns from their transcripts and the run record, and append the records."""
    from mason.session import transcript_groups

    path = records or benchmark.records_path()
    try:
        root = _ops.resolve_root(workspace)
        asked = benchmark.find_question(question) if question is not None else None
        known = benchmark.recorded_sessions(benchmark.load_records(path))
        if session:
            targets = list(session)
        else:
            targets = [
                conversation.stem
                for conversation, _ in transcript_groups(root)
                if benchmark.question_for(conversation) is not None
            ]
        scored: list[dict[str, Any]] = []
        skipped = 0
        for target in targets:
            record = benchmark.score_session(
                root, target, question=asked, machine=machine, model=model
            )
            if record["session"] in known and not rescore:
                skipped += 1
                continue
            benchmark.append_record(path, record)
            scored.append(record)
    except _BENCH_ERRORS as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(scored, indent=2, ensure_ascii=False))
        return
    for record in scored:
        typer.echo(_record_line(record))
    if not scored:
        typer.echo("nothing new to score" + (f" ({skipped} already recorded)" if skipped else ""))
    elif skipped:
        typer.echo(f"{skipped} already recorded (pass --rescore to score again)")


@benchmark_app.command("render")
def benchmark_render(
    question: Annotated[str, typer.Argument(help="Question number or key (see list).")],
    workspace: _WorkspaceOpt = None,
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition for the engine legs.")
    ] = None,
    time_limit: Annotated[str | None, typer.Option("--time-limit")] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for the rendered files (default: ./sandbox)."),
    ] = None,
) -> None:
    """Write the campaign's sandbox job files without submitting them.

    For tweaks the config cannot express: read and edit the rendered
    .sbatch, then submit it with sbatch yourself. 'launch' always
    re-renders, so hand edits only survive a manual sbatch.
    """
    from mason.cli import render_sandbox_files

    try:
        asked = benchmark.find_question(question)
        script_path, _script = render_sandbox_files(
            asked.instruction,
            workspace=workspace,
            partition=partition,
            time_limit=time_limit,
            out=out,
            engine_tasks=None,
        )
    except _BENCH_ERRORS as e:
        _fail(str(e))
    typer.echo(
        f"rendered Q{asked.number} ({asked.key}); edit if needed, then: sbatch {script_path}"
    )
    typer.echo("after the job ends: slab benchmark score")


@benchmark_app.command("tables")
def benchmark_tables(
    docs: Annotated[
        Path | None, typer.Option("--docs", help="The docs page (default docs/benchmark.md).")
    ] = None,
    readme: Annotated[
        Path | None, typer.Option("--readme", help="The README (default README.md).")
    ] = None,
    records: _RecordsOpt = None,
) -> None:
    """Rewrite the benchmark tables inside their marker regions in the docs and the README."""
    path = records or benchmark.records_path()
    docs_path = docs if docs is not None else Path("docs") / "benchmark.md"
    readme_path = readme if readme is not None else Path("README.md")
    try:
        changed = benchmark.render(
            benchmark.load_records(path), docs=docs_path, readme=readme_path
        )
    except _BENCH_ERRORS as e:
        _fail(str(e))
    for touched in changed:
        typer.echo(f"rewrote {touched}")
    if not changed:
        typer.echo("tables already current")


# -- the front door -----------------------------------------------------------
#
# The machine groups mount as they are; the agent mounts whole; the lifecycle
# verbs re-register one by one. Iterating registered_commands means a verb
# added to foundation.cli later can never be forgotten here —
# tests/test_slab_front_cli.py pins the resulting tree exactly.

for _group, _name in (
    (engines_app, "engines"),
    (pseudos_app, "pseudos"),
    (protocols_app, "protocols"),
    (mp_app, "mp"),
    (hpc_app, "hpc"),
    (config_app, "config"),
):
    app.add_typer(_group, name=_name, rich_help_panel=_PANEL_MACHINE)

app.add_typer(mason_app, name="mason", rich_help_panel=_PANEL_AGENT)


def _command_name(info: typer.models.CommandInfo) -> str:
    """The name typer will give this command (explicit, or from the function)."""
    return info.name or info.callback.__name__.replace("_", "-")  # type: ignore[union-attr]


for _info in foundation_cli.app.registered_commands:
    # CommandInfo is a plain class in typer, so a shallow copy carries every
    # setting and only the help panel is ours to choose.
    _mounted = copy.copy(_info)
    _mounted.rich_help_panel = (
        _PANEL_HOUSEKEEPING
        if _command_name(_info) in ("expire", "gc")
        else _PANEL_INTEGRATION
        if _command_name(_info) == "mcp"
        else _PANEL_LIFECYCLE
    )
    app.registered_commands.append(_mounted)


if __name__ == "__main__":  # pragma: no cover - module execution convenience
    app()
