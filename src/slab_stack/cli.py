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
from mason.serve import read_record
from slab._version import __version__
from slab.cli import (
    _CpusPerTaskOpt,
    _GpusPerNodeOpt,
    _MemOpt,
    _NodesOpt,
    _NtasksPerNodeOpt,
    config_app,
    engines_app,
    hpc_app,
    mp_app,
    protocols_app,
    pseudos_app,
)
from slab.errors import SlabError
from slab.hpc import SchedulerNotAvailableError, active_job_ids
from slab_stack import _ops as stack_ops
from slab_stack import benchmark, review

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
            # A run whose process died is failed first, so the sweep expires
            # a dead run as a failed one and not as a running one.
            reaped = ws.reap_dead(caller="slab fast-forward")
            expired = ws.expire_due(policy, include_running=include_running)
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))
    for run in reaped:
        typer.echo(f"failed  {run.id}  {run.name}  its process is gone")
    for run in expired:
        typer.echo(f"expired {run.id}  {run.name}")
    typer.echo(f"{len(expired)} run(s) fast-forwarded to expired")


def _job_id_of(name: str) -> str | None:
    """The job id a job file's name carries, or None.

    Examples:
        >>> _job_id_of("cu-relax-1244113.out")
        '1244113'
        >>> _job_id_of("notes.txt") is None
        True
    """
    match = _JOB_FILE_ID.search(name)
    return None if match is None else match.group(1)


def _active_jobs(root: Path) -> frozenset[str]:
    """The job ids the scheduler still holds, the serve record's job included."""
    try:
        active = active_job_ids()
    except SchedulerNotAvailableError:
        active = frozenset()  # no scheduler here, so nothing can be running
    except SlabError as e:
        _fail(str(e))
    try:
        record = read_record(root)
    except MasonError as e:  # an unreadable endpoint record is a plain error here
        _fail(str(e))
    if record is not None and record.job_id:
        active = active | {str(record.job_id)}
    return active


@app.command(rich_help_panel=_PANEL_HOUSEKEEPING)
def purge(
    workspace: _WorkspaceOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the inventory; delete nothing.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Skip the confirmation prompt.")
    ] = False,
    all_sessions: Annotated[
        bool,
        typer.Option(
            "--all-sessions",
            help="Also delete the newest conversation's transcript and sidecars (kept "
            "by default so 'slab mason chat --resume' still works), the newest "
            "harness record, and the session files no transcript claims.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the inventory as JSON.")
    ] = False,
) -> None:
    """Delete all expired data for real: rows, bytes, session files, job files, scratch.

    The inventory comes first, and the confirmation names its totals.
    Expired runs lose their database rows (run, transitions, artifact
    references, tasks, checks) and any artifact bytes no surviving run
    references. Session transcripts are deleted with their delegation
    siblings, compaction summaries, and review records, except the newest
    conversation's. Harness session records whose session has no run in
    flight go, and so do session locks no process holds. Finished jobs'
    .sbatch and .out files are swept from the workspace; jobs still in
    the queue, the running model server included, keep theirs. Under
    [paths] scratch, every slab-* directory whose calculation is over or
    whose process is gone is removed. Irreversible: run with --dry-run
    first.

    The machine's memory and the project directory are not touched. This
    command clears workspace state that was never promoted; a memory is
    durable machine state that no project owns, so it is forgotten one at
    a time and on purpose, with 'slab memory forget'.
    """
    try:
        root = _ops.resolve_root(workspace)
    except (FoundationError, SlabError, OSError) as e:
        _fail(str(e))
    active = _active_jobs(root)
    try:
        inventory = stack_ops.purge_inventory(
            root, all_sessions=all_sessions, active=active, job_id_of=_job_id_of, dry_run=True
        )
    except (FoundationError, MasonError, SlabError, OSError) as e:
        _fail(str(e))

    if dry_run:
        if as_json:
            typer.echo(json.dumps(inventory.model_dump(), indent=2))
        else:
            for line in inventory.lines("would delete"):
                typer.echo(line)
        return

    if not yes:
        typer.confirm(
            f"permanently delete {inventory.summary()} from {root}?",
            abort=True,
        )
    try:
        deleted = stack_ops.purge_inventory(
            root, all_sessions=all_sessions, active=active, job_id_of=_job_id_of, dry_run=False
        )
    except (FoundationError, MasonError, SlabError, OSError) as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(deleted.model_dump(), indent=2))
        return
    for line in deleted.lines("deleted", detail=False):
        typer.echo(line)


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
    freshness of the rendered job. Exits nonzero only on an \\[x] row; an
    \\[=] row is a fact about this machine, not a failure. The focused
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
    """List every memory on this machine: name, date, writer, description.

    A memory stamped with software that has changed since it was written
    carries a note saying what changed. The versions are probed only when
    some memory carries a stamp.
    """
    try:
        memories = memory_store.discover()
    except FoundationError as e:
        _fail(str(e))
    live: dict[str, str] = {}
    if any(m.against for m in memories.values()):
        from slab._ops import software_versions

        live = software_versions()
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
                        "against": m.against,
                        "changed": m.drift(live),
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
        changed = memory.drift(live)
        note = f" [changed since: {'; '.join(changed)}]" if changed else ""
        typer.echo(f"{memory.name:<{width}}  {stamp}  {memory.agent or '-':<16}  "
                   f"{memory.description}{note}")
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

_BENCH_ERRORS = (
    benchmark.BenchmarkError,
    review.ReviewError,
    MasonError,
    FoundationError,
    SlabError,
    OSError,
)
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
_ConditionOpt = Annotated[
    str | None,
    typer.Option(
        "--condition",
        help="Harness condition: slab (default), protocol, or bare. The record carries it.",
    ),
]
_WithoutOpt = Annotated[
    list[str] | None,
    typer.Option("--without", help="Switch one mechanism off (repeatable); see 'matrix'."),
]
_RefereeOpt = Annotated[
    bool,
    typer.Option(
        "--referee",
        help="Also ask a model to referee each campaign (one model call per campaign).",
    ),
]
_RefereeModelOpt = Annotated[
    str | None,
    typer.Option("--referee-model", help="Model for the referee (default: the [agent] model)."),
]
_RefereeEndpointOpt = Annotated[
    str | None, typer.Option("--referee-endpoint", help="Endpoint for the referee.")
]
_RefereeProviderOpt = Annotated[
    str | None, typer.Option("--referee-provider", help="Provider for the referee.")
]


def _referee(
    root: Path, wanted: bool, model: str | None, endpoint: str | None, provider: str | None
) -> Any:
    """The referee's chat client when asked for, else None."""
    if not wanted:
        return None
    return review.referee_client(root, model=model, endpoint=endpoint, provider=provider)


def _record_line(record: dict[str, Any]) -> str:
    verdict = "pass" if record["passed"] else f"fail: {record['reason']}"
    raised = len(record.get("flags") or [])
    flagged = f"  [{raised} flag{'s' if raised != 1 else ''}]" if raised else ""
    arm = benchmark.harness_label(record.get("condition"), record.get("ablated") or ())
    return (
        f"Q{record['question']} {record['key']:<9} {record['model']:<24} "
        f"{record['machine']:<12} {arm:<9} {verdict}{flagged}"
    )


def _flag_line(flag: dict[str, Any]) -> str:
    return f"  {flag['rule']:<22} {flag['raised_by']:<8} {flag['evidence']}: {flag['note']}"


@benchmark_app.command("list")
def benchmark_list() -> None:
    """The questions, their result keys, and what passes."""
    for question in benchmark.QUESTIONS:
        typer.echo(f"{question.number}. [{question.key}] {question.instruction}")
        keys = ", ".join(f"{name} ({unit})" for name, unit in question.results.items())
        typer.echo(f"   results: {keys}")
        for cls in benchmark.CLASSES:
            band = question.tolerance[cls]
            reference = question.reference.get(cls, {})
            if question.kind == "threshold":
                rule = ", ".join(f"{k} ≤ {v:g}" for k, v in band.items())
            elif not reference:
                rule = "no checked reference yet; a campaign in this class is refused"
            else:
                rule = ", ".join(f"{k} = {reference[k]:g} ± {v:g}" for k, v in band.items())
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
    agent: Annotated[
        str | None, typer.Option("--agent", help="Entry card (default: the condition's).")
    ] = None,
    condition: _ConditionOpt = None,
    without: _WithoutOpt = None,
    machine: _MachineOpt = None,
    records: _RecordsOpt = None,
    referee: _RefereeOpt = False,
    referee_model: _RefereeModelOpt = None,
    referee_endpoint: _RefereeEndpointOpt = None,
    referee_provider: _RefereeProviderOpt = None,
) -> None:
    """Run one campaign here, autonomously, then score, review, and record it.

    For a laptop or an interactive node. On a cluster, prefer 'launch',
    which runs the campaign as a sandbox job, then 'score' after it ends.
    --condition picks the harness arm; the scorer's verification rule is
    the same for every arm.
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
            condition=condition,
            without=tuple(without or ()),
        )
        typer.echo(
            f"session {session_id}: stopped by {result.stop_reason} "
            f"after {result.steps} step(s)"
        )
        root = _ops.resolve_root(workspace)
        judge = _referee(root, referee, referee_model, referee_endpoint, referee_provider)
        record = benchmark.score_session(
            root, session_id, question=asked, machine=machine, referee=judge
        )
        path = records or benchmark.records_path()
        benchmark.append_record(path, record)
    except _BENCH_ERRORS as e:
        _fail(str(e))
    typer.echo(_record_line(record))
    for flag in record["flags"]:
        typer.echo(_flag_line(flag))
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
    agent: Annotated[
        str | None,
        typer.Option("--agent", help="Agent card the job runs as (default: the condition's)."),
    ] = None,
    condition: _ConditionOpt = None,
    without: _WithoutOpt = None,
    nodes: _NodesOpt = None,
    ntasks_per_node: _NtasksPerNodeOpt = None,
    cpus_per_task: _CpusPerTaskOpt = None,
    gpus_per_node: _GpusPerNodeOpt = None,
    mem: _MemOpt = None,
) -> None:
    """Submit one campaign as a sandbox job; score it with 'score' after it ends.

    The five size flags size the sandbox job itself within the partition's
    declared node; inside it the agent sizes each launch.
    """
    from mason.cli import launch_sandbox
    from slab.resources import job_size

    out_dir = (out if out is not None else Path.cwd() / "sandbox").resolve()
    try:
        size = job_size(
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            cpus_per_task=cpus_per_task,
            gpus_per_node=gpus_per_node,
            mem=mem,
        )
        asked = benchmark.find_question(question)
        job = launch_sandbox(
            asked.instruction,
            workspace=workspace,
            partition=partition,
            time_limit=time_limit,
            out_dir=out_dir,
            engine_tasks=None,
            emit=typer.echo,
            agent=agent,
            condition=condition,
            without=tuple(without or ()),
            size=size,
            expected_results=dict(asked.results),
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
    referee: _RefereeOpt = False,
    referee_model: _RefereeModelOpt = None,
    referee_endpoint: _RefereeEndpointOpt = None,
    referee_provider: _RefereeProviderOpt = None,
) -> None:
    """Score and review campaigns from their transcripts and the run record.

    The rules review every campaign; --referee also asks a model. Each
    record appends to the records file with its flags.
    """
    from mason.session import transcript_groups

    path = records or benchmark.records_path()
    try:
        root = _ops.resolve_root(workspace)
        judge = _referee(root, referee, referee_model, referee_endpoint, referee_provider)
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
            if target in known and not rescore:
                skipped += 1
                continue
            try:
                record = benchmark.score_session(
                    root, target, question=asked, machine=machine, model=model, referee=judge
                )
            except benchmark.BenchmarkError as e:
                # A session that cannot be judged (no checked reference for
                # its class, a non-campaign named by --session) must not
                # stop the sweep for the ones that can.
                typer.secho(f"skipped {target}: {e}", err=True, fg=typer.colors.YELLOW)
                continue
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
        for flag in record.get("flags") or []:
            typer.echo(_flag_line(flag))
        if record.get("referee_error"):
            typer.secho(f"  referee failed: {record['referee_error']}", err=True,
                        fg=typer.colors.YELLOW)
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
    agent: Annotated[
        str | None,
        typer.Option("--agent", help="Agent card the job runs as (default: the condition's)."),
    ] = None,
    condition: _ConditionOpt = None,
    without: _WithoutOpt = None,
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
            agent=agent,
            condition=condition,
            without=tuple(without or ()),
            expected_results=dict(asked.results),
        )
    except _BENCH_ERRORS as e:
        _fail(str(e))
    typer.echo(
        f"rendered Q{asked.number} ({asked.key}); edit if needed, then: sbatch {script_path}"
    )
    typer.echo("after the job ends: slab benchmark score")


@benchmark_app.command("matrix")
def benchmark_matrix(
    workspace: _WorkspaceOpt = None,
    condition: Annotated[
        list[str] | None,
        typer.Option("--condition", help="A condition to include (repeatable; default all)."),
    ] = None,
    question: Annotated[
        list[str] | None,
        typer.Option("--question", help="A question number or key (repeatable; default all)."),
    ] = None,
    without: Annotated[
        list[str] | None,
        typer.Option(
            "--without",
            help="A mechanism to switch off from the slab condition, one job per "
            "mechanism (repeatable).",
        ),
    ] = None,
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition for the engine legs.")
    ] = None,
    time_limit: Annotated[str | None, typer.Option("--time-limit")] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for the grid (default: ./sandbox/matrix)."),
    ] = None,
) -> None:
    """Render the launch grid, conditions x questions x ablations, without submitting.

    Every cell gets its own directory with the four sandbox files, and
    matrix.json lists them. Submit each script with sbatch; when the jobs
    end, 'score' records every campaign with its condition.
    """
    from mason.cli import render_sandbox_files

    out_dir = (out if out is not None else Path.cwd() / "sandbox" / "matrix").resolve()
    try:
        arms = list(condition) if condition else list(benchmark.CONDITIONS)
        asked = (
            [benchmark.find_question(q) for q in question]
            if question
            else list(benchmark.QUESTIONS)
        )
        cells = benchmark.matrix_cells(arms, asked, tuple(without or ()))
        manifest: list[dict[str, Any]] = []
        for cell in cells:
            script_path, _script = render_sandbox_files(
                cell.question.instruction,
                workspace=workspace,
                partition=partition,
                time_limit=time_limit,
                out=out_dir / cell.subdir,
                engine_tasks=None,
                condition=cell.condition,
                without=cell.without,
                expected_results=dict(cell.question.results),
            )
            manifest.append(
                {
                    "condition": cell.condition,
                    "without": list(cell.without),
                    "harness": cell.label,
                    "question": cell.question.number,
                    "key": cell.question.key,
                    "script": str(script_path),
                }
            )
            typer.echo(
                f"{cell.label:<32} Q{cell.question.number} {cell.question.key:<9} {script_path}"
            )
        (out_dir / "matrix.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except _BENCH_ERRORS as e:
        _fail(str(e))
    typer.echo(f"{len(cells)} job(s) rendered under {out_dir}; nothing submitted")
    typer.echo("submit each with sbatch <script>; after the jobs end: slab benchmark score")


@benchmark_app.command("flags")
def benchmark_flags(
    target: Annotated[
        str | None,
        typer.Option("--target", help="Only this target, e.g. skill:equation-of-state."),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="Only flags in this status: open, pending, or unknown."),
    ] = None,
    records: _RecordsOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the flags as JSON.")] = False,
) -> None:
    """The defect list: every flag on the latest record per model, machine, and question.

    A flag is 'open' while its target is unchanged since it was raised,
    'pending' when the skill has a newer revision no campaign has run
    under, and 'unknown' when the skill is not in the catalog.
    """
    from mason.skills import discover_skills

    path = records or benchmark.records_path()
    try:
        rows = review.ledger(benchmark.load_records(path), discover_skills(Path.cwd()))
    except _BENCH_ERRORS as e:
        _fail(str(e))
    if target is not None:
        rows = [row for row in rows if row["target"] == target]
    if status is not None:
        rows = [row for row in rows if row["status"] == status]
    if as_json:
        typer.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        typer.echo("no flags")
        return
    current: str | None = None
    for row in rows:
        if row["target"] != current:
            current = row["target"]
            typer.echo(current)
        cell = f"Q{row['question']} {row['model']}/{row['machine']}"
        against = f" against {row['against']}" if row["against"] else ""
        typer.echo(
            f"  {row['status']:<8} {row['rule']:<22} {cell}{against}\n"
            f"           {row['raised_by']}: {row['evidence']}: {row['note']}"
        )


@benchmark_app.command("gate")
def benchmark_gate(
    skill: Annotated[str, typer.Argument(help="The skill whose current revision to validate.")],
    records: _RecordsOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the report as JSON.")] = False,
) -> None:
    """Whether the benchmark validates the catalog's revision of a skill.

    For every model, machine, and question that exercised the skill, the
    newest record under the current revision is compared with the newest
    under any earlier one. Exit 1 unless every cell is validated: a
    campaign ran under this revision, it did not regress, and it raises
    no flag against the skill.
    """
    from mason.skills import discover_skills

    path = records or benchmark.records_path()
    try:
        report = review.gate(skill, benchmark.load_records(path), discover_skills(Path.cwd()))
    except _BENCH_ERRORS as e:
        _fail(str(e))
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "skill": report.skill,
                    "digest": report.digest,
                    "validated": report.validated,
                    "cells": [cell.__dict__ for cell in report.cells],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        typer.echo(f"{report.skill} revision {report.digest}")
        for cell in report.cells:
            typer.echo(
                f"  Q{cell.question} {cell.key:<9} {cell.model:<24} {cell.machine:<12} "
                f"{cell.verdict}: {cell.detail}"
            )
        if not report.cells:
            typer.echo("  not validated: no scored campaign lists or loaded this skill")
        typer.echo("validated" if report.validated else "not validated")
    if not report.validated:
        raise typer.Exit(code=1)


@benchmark_app.command("tables")
def benchmark_tables(
    docs: Annotated[
        Path | None, typer.Option("--docs", help="The docs page (default docs/benchmark.md).")
    ] = None,
    readme: Annotated[
        Path | None, typer.Option("--readme", help="The README (default README.md).")
    ] = None,
    records: _RecordsOpt = None,
    retention: Annotated[
        bool,
        typer.Option(
            "--retention",
            help="Print the retention table instead: what each campaign's finish "
            "promoted of what its session produced. Rewrites nothing.",
        ),
    ] = False,
    utilisation: Annotated[
        bool,
        typer.Option(
            "--utilisation",
            help="Print the utilisation table instead: the cpu-hours and gpu-hours each "
            "campaign's runs held, against the session's budget over its wall time. "
            "Rewrites nothing.",
        ),
    ] = False,
) -> None:
    """Rewrite the benchmark tables inside their marker regions in the docs and the README."""
    path = records or benchmark.records_path()
    if retention or utilisation:
        table = benchmark.retention_table if retention else benchmark.utilisation_table
        try:
            typer.echo(table(benchmark.load_records(path)))
        except _BENCH_ERRORS as e:
            _fail(str(e))
        return
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


def _command_name(info: typer.models.CommandInfo) -> str:
    """The name typer will give this command (explicit, or from the function)."""
    return info.name or info.callback.__name__.replace("_", "-")  # type: ignore[union-attr]


# ``hpc cancel`` is the one machine verb that reaches into the workspace: a
# cancelled job leaves runs at ``running`` and reservations held, and only
# foundation can settle them. ``slab.cli`` cannot import foundation, so the
# group mounts here with its ``cancel`` replaced by the workspace-aware one.
_hpc_group = typer.Typer(
    help=hpc_app.info.help, no_args_is_help=True, rich_markup_mode=hpc_app.rich_markup_mode
)
for _info in hpc_app.registered_commands:
    if _command_name(_info) != "cancel":
        _hpc_group.registered_commands.append(copy.copy(_info))


@_hpc_group.command("cancel")
def hpc_cancel(
    job_id: Annotated[str, typer.Argument(help="SLURM job id.")],
    workspace: _WorkspaceOpt = None,
) -> None:
    """Cancel a job and settle the runs, reservations, and memories it leaves behind.

    The scheduler is asked first. Then every run the job was still
    executing is marked failed, the reservations those runs held are
    released, and each machine memory written since the job's first run
    started is listed for review. Nothing is expired, purged, or deleted.
    """
    try:
        summary = _ops.cancel_job(job_id, workspace=_ops.resolve_root(workspace))
    except (SlabError, FoundationError, OSError) as e:
        _fail(str(e))
    for line in _ops.cancel_lines(summary):
        typer.echo(line)


for _group, _name in (
    (engines_app, "engines"),
    (pseudos_app, "pseudos"),
    (protocols_app, "protocols"),
    (mp_app, "mp"),
    (_hpc_group, "hpc"),
    (config_app, "config"),
):
    app.add_typer(_group, name=_name, rich_help_panel=_PANEL_MACHINE)

app.add_typer(mason_app, name="mason", rich_help_panel=_PANEL_AGENT)
app.add_typer(foundation_cli.runs_app, name="runs", rich_help_panel=_PANEL_LIFECYCLE)


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
