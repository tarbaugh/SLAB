"""The ``slab mason`` command group: the resident research agent.

``chat`` and ``run`` drive the agent, interactively and autonomously.
``doctor`` is the empirical check that the configured endpoint answers and
calls tools. ``serve`` starts, locates, and stops the model server as an
ordinary batch job, because on a cluster the GPU node is the scheduler's
choice and the endpoint is discovered rather than configured.

The front door that mounts this group is :mod:`slab_stack.cli`. Runs and
artifacts are ``slab run`` and its siblings; engines, protocols, and the
scheduler are the machine groups.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer

if TYPE_CHECKING:
    from mason.config import AgentConfig
    from mason.roster import AgentSpec
    from mason.session import MasonSession
    from slab.config import HpcConfig

from foundation import _ops
from foundation.errors import FoundationError
from mason.errors import MasonError
from slab.errors import SlabError

app = typer.Typer(
    help="Mason — the resident research agent.",
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


_STEP_PREVIEW_CHARS = 2_000


def _echo_step(kind: str, attribution: str, text: str) -> None:
    """Live step output for chat: reasoning dimmed, interim text plain.

    Long traces are clipped for the terminal only; the transcript holds
    the full text.
    """
    shown = text.strip()
    if len(shown) > _STEP_PREVIEW_CHARS:
        clipped = len(shown) - _STEP_PREVIEW_CHARS
        shown = (
            shown[:_STEP_PREVIEW_CHARS]
            + f" [... {clipped} more characters in the transcript]"
        )
    if kind == "reasoning":
        typer.secho(f"\n[reasoning] {attribution}{shown}", dim=True)
    else:
        typer.echo(f"\n{attribution}{shown}")


def _ask_approval(tool_name: str, preview: str) -> bool:
    from click.exceptions import Abort

    typer.echo(f"\n[approval] {tool_name}: {preview}")
    try:
        return typer.confirm("allow?", default=False)
    except (Abort, EOFError):
        # stdin is not interactive (sbatch, piped input, closed terminal):
        # refusing is the documented degradation; dying mid-turn is not.
        typer.echo("[approval] no interactive stdin — refused")
        return False


def _override_agent(agent: AgentConfig, updates: dict[str, object]) -> AgentConfig:
    """Apply CLI overrides through the model's own validation.

    ``mason.config.override_agent`` does the validated rebuild — a mistyped
    ``--provider anthorpic`` is refused, never silently probed — and this
    wrapper names the flag that supplied the bad value.
    """
    if not updates:
        return agent
    from pydantic import ValidationError

    from mason.config import override_agent

    try:
        return override_agent(agent, updates)
    except ValidationError as e:
        flags = ", ".join(f"--{key.replace('_', '-')}" for key in updates)
        first = e.errors()[0]
        field = ".".join(str(piece) for piece in first["loc"]) or "value"
        _fail(f"invalid {flags}: {field}: {first['msg']}")


def _mason_session(
    workspace: Path | None,
    *,
    auto: bool,
    model: str | None,
    endpoint: str | None,
    provider: str | None = None,
    max_turns: int | None = None,
    interactive: bool = True,
) -> MasonSession:
    from mason import MasonSession

    # Non-interactive entry points (slab mason run) get the refuse-everything
    # gate instead of a terminal prompt, matching mason_run's docstring:
    # without --auto, mutating tools are refused — there is no one to ask.
    session = MasonSession(
        workspace_root=_ops.resolve_root(workspace),
        approver=_ask_approval if interactive else None,
        auto_approve=auto,
    )
    updates: dict[str, object] = {}
    if model is not None:
        updates["model"] = model
    if provider is not None:
        updates["provider"] = provider
    if max_turns is not None:
        updates["max_turns"] = max_turns
    if endpoint is not None:
        # Into flag_updates like the others, so delegated children inherit
        # it: a child re-resolves its own endpoint, and without this it
        # would rediscover the serve record's URL — which, in the sandbox,
        # is exactly the address the namespace cannot reach.
        updates["endpoint"] = endpoint
    if updates:
        session.agent = _override_agent(session.agent, updates)
        # Remembered so the loop can re-assert them over any
        # [agent.roster.<name>] table: a flag outranks config.
        session.flag_updates = updates
    # After a provider change, which endpoint is right changes too; a
    # --endpoint flag outranks both the config and any discovered server.
    if updates:
        session.resolve_endpoint(endpoint)
    # Child processes stamp their runs with this chat's id: the agent's shell
    # tool runs 'slab run' directly, and only an exported variable
    # reaches it. In-process launches pass the id explicitly instead.
    os.environ["SLAB_SESSION"] = session.session_id
    return session


def _resolve_spec(agent_name: str | None) -> tuple[AgentSpec, dict[str, AgentSpec]]:
    """The entry agent's card and the roster, refusing unknown names loudly."""
    from mason.roster import discover_roster

    roster = discover_roster(Path.cwd())
    chosen = agent_name or "pi"
    spec = roster.get(chosen)
    if spec is None:
        _fail(f"no agent named {chosen!r}; the roster: {', '.join(sorted(roster))}")
    return spec, roster


_ModelOpt = Annotated[str | None, typer.Option("--model", help="Override [agent] model.")]
_EndpointOpt = Annotated[
    str | None,
    typer.Option("--endpoint", help="Override [agent] endpoint and any served endpoint."),
]
_AutoOpt = Annotated[
    bool, typer.Option("--auto", help="Approve every tool call (batch/HPC use).")
]
_ProviderOpt = Annotated[
    str | None,
    typer.Option("--provider", help="Override [agent] provider: openai or anthropic."),
]
_AgentOpt = Annotated[
    str | None,
    typer.Option(
        "--agent", help="Agent card to run as (default pi); 'slab mason roster' lists them."
    ),
]


@app.command("chat")
def mason_chat(
    workspace: _WorkspaceOpt = None,
    auto: _AutoOpt = False,
    model: _ModelOpt = None,
    endpoint: _EndpointOpt = None,
    provider: _ProviderOpt = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue the newest session in this workspace.")
    ] = False,
    agent: _AgentOpt = None,
) -> None:
    """Interactive session ( /quit to leave, /status for token counts )."""
    from mason import Mason

    try:
        session = _mason_session(
            workspace, auto=auto, model=model, endpoint=endpoint, provider=provider
        )
        # Chat is where a person watches the loop; 'slab mason run' never
        # wires an observer, so batch output stays the final report alone.
        if session.agent.show_reasoning:
            session.observer = _echo_step
        spec, roster = _resolve_spec(agent)
        resume_from = None
        if resume:
            latest = session.latest_transcript()
            if latest is None:
                _fail("nothing to resume: no session transcripts in this workspace")
            resume_from = session.load_messages(latest)
            typer.echo(f"resuming {latest.name} ({len(resume_from)} messages)")
        mason = Mason(session, resume_from=resume_from, spec=spec, roster=roster)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo(
        f"mason ready: {session.agent_name} — {session.agent.model} at "
        f"{session.endpoint} [{session.endpoint_origin}]"
    )
    typer.echo(f"workspace {session.workspace_root}; notebook {session.notebook_path}")
    while True:
        try:
            text = input("\nmason> ").strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - terminal signals
            typer.echo("")
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/status":
            typer.echo(
                f"tokens: {session.prompt_tokens} prompt, "
                f"{session.completion_tokens} completion; "
                f"session {session.session_id}; "
                f"transcript {session.transcript_path}"
            )
            continue
        if text == "/compact":
            mason._compact()
            typer.echo("history compacted; summary appended to the notebook")
            continue
        try:
            result = mason.run_turn(text)
        except KeyboardInterrupt:  # pragma: no cover - terminal signals
            typer.echo("\n[turn interrupted; state kept]")
            continue
        except (MasonError, FoundationError, SlabError) as e:
            typer.echo(f"error: {e}", err=True)
            continue
        typer.echo(f"\n{result.text}")
        if result.stop_reason not in ("answer", "finish"):
            typer.echo(f"[stopped: {result.stop_reason} after {result.steps} steps]", err=True)


@app.command("run")
def mason_run(
    goal: Annotated[str, typer.Argument(help="The research goal for this run.")],
    workspace: _WorkspaceOpt = None,
    auto: _AutoOpt = False,
    model: _ModelOpt = None,
    endpoint: _EndpointOpt = None,
    provider: _ProviderOpt = None,
    max_turns: Annotated[
        int | None, typer.Option("--max-turns", help="Model-call budget for this goal.")
    ] = None,
    agent: _AgentOpt = None,
) -> None:
    """One autonomous goal: loop until finish, an answer, or a harness stop.

    Without --auto, mutating tools are refused (there is no one to ask);
    reads still work, so inspection goals run safely by default.
    """
    from mason import Mason

    try:
        session = _mason_session(
            workspace,
            auto=auto,
            model=model,
            endpoint=endpoint,
            provider=provider,
            max_turns=max_turns,
            interactive=False,
        )
        spec, roster = _resolve_spec(agent)
        mason = Mason(session, spec=spec, roster=roster)
        result = mason.run_turn(goal)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo(result.text)
    typer.echo(
        f"[{result.stop_reason} after {result.steps} step(s); "
        f"tokens {session.prompt_tokens}+{session.completion_tokens}; "
        f"session {session.session_id}; "
        f"transcript {session.transcript_path}]",
        err=True,
    )
    if result.stop_reason not in ("answer", "finish"):
        raise typer.Exit(code=1)


@app.command("roster")
def mason_roster() -> None:
    """The agents: name, layer, effective model, and the skills each sees."""
    from mason.config import load_config, roster_agent_config
    from mason.roster import check_overrides, discover_roster, skills_for
    from mason.skills import discover_skills

    try:
        agent = load_config().agent
        roster = discover_roster(Path.cwd())
        check_overrides(agent, roster)
        skills = discover_skills(Path.cwd())
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    for name in sorted(roster, key=lambda n: (n != "pi", n)):
        spec = roster[name]
        effective = roster_agent_config(agent, name)
        model = effective.model or "(model not configured)"
        visible = len(skills_for(spec, skills))
        marker = "  [delegates]" if spec.delegates else ""
        typer.echo(
            f"{name:<18} {spec.source:<9} {model:<28} {visible} skill(s){marker}"
        )


@app.command("skills")
def mason_skills(
    agent: Annotated[
        str | None,
        typer.Option("--agent", help="Show only the skills this agent card sees."),
    ] = None,
) -> None:
    """The skills: name, layer, which agents see them, bundled scripts."""
    from mason.roster import discover_roster, skills_for
    from mason.skills import discover_skills

    try:
        skills = discover_skills(Path.cwd())
        if agent is not None:
            roster = discover_roster(Path.cwd())
            spec = roster.get(agent)
            if spec is None:
                _fail(f"no agent named {agent!r}; the roster: {', '.join(sorted(roster))}")
            skills = skills_for(spec, skills)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    if not skills:
        typer.echo("no skills visible here; add <project>/skills/<name>/SKILL.md")
        return
    for name in sorted(skills):
        skill = skills[name]
        agents = "all agents" if skill.agents is None else " ".join(sorted(skill.agents))
        scripts_dir = skill.root / "scripts"
        scripts = sum(1 for p in scripts_dir.iterdir() if p.is_file()) if (
            scripts_dir.is_dir()
        ) else 0
        note = "  [allowed-tools ignored]" if skill.ignored_allowed_tools else ""
        typer.echo(f"{name:<26} {skill.source:<9} {agents:<28} {scripts} script(s){note}")


serve_app = typer.Typer(
    help="The agent's model server as a batch job: render, start, watch, stop.",
    no_args_is_help=True,
)
app.add_typer(serve_app, name="serve")

_PartitionOpt = Annotated[
    str | None, typer.Option("--partition", "-p", help="Partition (default: [agent.serve]'s).")
]
_PortOpt = Annotated[int | None, typer.Option("--port", help="Override [agent.serve] port.")]
_TimeOpt = Annotated[str | None, typer.Option("--time", help="Override the job's time limit.")]


def _serve_inputs(workspace: Path | None) -> tuple[AgentConfig, HpcConfig, Path]:
    """The three things every serve verb needs, each from its owner."""
    from mason.config import load_config
    from slab.config import load_config as load_slab_config

    return (
        load_config().agent,
        load_slab_config().hpc,
        _ops.resolve_root(workspace),
    )


@serve_app.command("render")
def mason_serve_render(
    workspace: _WorkspaceOpt = None,
    partition: _PartitionOpt = None,
    port: _PortOpt = None,
    time_limit: _TimeOpt = None,
) -> None:
    """Print the batch script 'serve start' would submit — read it first."""
    from mason.serve import render_serve_script

    try:
        agent, hpc, root = _serve_inputs(workspace)
        script = render_serve_script(
            agent, hpc, root, partition=partition, port=port, time_limit=time_limit
        )
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo(script)


@serve_app.command("start")
def mason_serve_start(
    workspace: _WorkspaceOpt = None,
    partition: _PartitionOpt = None,
    port: _PortOpt = None,
    time_limit: _TimeOpt = None,
    wait: Annotated[
        bool, typer.Option("--wait", help="Block until the endpoint answers (or time out).")
    ] = False,
) -> None:
    """Submit the model server as a batch job; it records the URL it lands on."""
    from mason.serve import start, wait_for_record, wait_until_ready

    try:
        agent, hpc, root = _serve_inputs(workspace)
        job = start(agent, hpc, root, partition=partition, port=port, time_limit=time_limit)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo(f"submitted job {job.job_id} ({job.job_name}) to {job.partition}")
    typer.echo(f"script: {job.script_path}")
    if not wait:
        typer.echo(
            "the endpoint appears once the model finishes loading; "
            "watch it with 'slab mason serve status'"
        )
        return
    budget = agent.serve.ready_timeout_s
    typer.echo(f"waiting for the queue, then for the model to load (up to {budget:.0f}s each)...")
    try:
        record = wait_for_record(root, job.job_id, timeout_s=budget)
        typer.echo(f"node {record.node or 'unnamed'} announced {record.endpoint}; loading...")
        names = wait_until_ready(record.endpoint, timeout_s=budget)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo(f"[+] {record.endpoint} answers; serving: {', '.join(names) or 'none'}")


@serve_app.command("status")
def mason_serve_status(
    workspace: _WorkspaceOpt = None,
) -> None:
    """What the recorded server is: endpoint, job state, and a live probe."""
    from mason.serve import describe

    try:
        agent, hpc, root = _serve_inputs(workspace)
        for line in describe(agent, root, cluster=hpc.cluster or ""):
            typer.echo(line)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))


@serve_app.command("stop")
def mason_serve_stop(
    workspace: _WorkspaceOpt = None,
) -> None:
    """Cancel the recorded server job and remove its endpoint record."""
    from mason.serve import stop

    try:
        _agent, hpc, root = _serve_inputs(workspace)
        typer.echo(stop(root, cluster=hpc.cluster or ""))
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))


_READ_PREVIEW_CHARS = 1_500


def _clip(text: str, full: bool) -> str:
    text = text.rstrip()
    if full or len(text) <= _READ_PREVIEW_CHARS:
        return text
    omitted = len(text) - _READ_PREVIEW_CHARS
    return f"{text[:_READ_PREVIEW_CHARS]}\n[... {omitted} more characters; --full shows them]"


def _render_event(event: dict[str, Any], full: bool) -> None:
    """One transcript event, in the same visual language as 'slab mason chat'."""
    stamp = str(event.get("at", ""))[11:19]
    kind = event.get("type")
    if kind == "message":
        message = event.get("message", {})
        role = message.get("role")
        if role == "user":
            typer.secho(f"\n=== user @ {stamp} " + "=" * 46, bold=True)
            typer.echo(_clip(str(message.get("content") or ""), full))
        elif role == "assistant":
            content = message.get("content")
            if content:
                typer.secho(f"\n--- assistant @ {stamp}", bold=True)
                typer.echo(_clip(str(content), full))
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                arguments = str(function.get("arguments", ""))
                if not full and len(arguments) > 200:
                    arguments = arguments[:200] + " ..."
                typer.secho(
                    f"[{stamp}] -> {function.get('name', '?')} {arguments}",
                    fg=typer.colors.CYAN,
                )
        elif role == "tool":
            typer.echo(_clip(str(message.get("content") or ""), full))
    elif kind == "reasoning":
        typer.secho(f"\n[reasoning @ {stamp}]", dim=True)
        typer.secho(_clip(str(event.get("text", "")), full), dim=True)
    elif kind == "skill":
        typer.secho(f"[{stamp}] skill loaded: {event.get('name')} ({event.get('source')})")
    elif kind == "compaction":
        typer.secho(f"\n=== compaction @ {stamp} " + "=" * 40, bold=True)
        typer.echo(_clip(str(event.get("summary", "")), full))
    elif kind == "finish":
        typer.secho(f"\n=== final report @ {stamp} " + "=" * 39, bold=True)
        typer.echo(_clip(str(event.get("report", "")), full))
    elif kind == "resume":
        typer.echo(f"[{stamp}] resumed with {event.get('messages')} prior message(s)")
    # usage events are accumulated by the caller, not printed per step.


@app.command("read")
def mason_read(
    transcript: Annotated[
        Path, typer.Argument(help="A session transcript (.jsonl) to render.")
    ],
    full: Annotated[
        bool, typer.Option("--full", help="Show everything; no truncation.")
    ] = False,
) -> None:
    """Render a session transcript for human reading.

    Same visual language as 'slab mason chat': dimmed reasoning, cyan tool
    calls, plain results. A malformed line is marked and skipped — a
    viewer must show the readable majority of a damaged file, unlike
    --resume, which must refuse it.
    """
    import json as _json

    if not transcript.is_file():
        _fail(f"no transcript at {transcript}")
    prompt_tokens = completion_tokens = steps = 0
    for number, line in enumerate(transcript.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = _json.loads(line)
        except _json.JSONDecodeError:
            typer.secho(f"[line {number}: not valid JSON; skipped]", fg=typer.colors.YELLOW)
            continue
        if event.get("type") == "usage":
            prompt_tokens += int(event.get("prompt_tokens") or 0)
            completion_tokens += int(event.get("completion_tokens") or 0)
            steps += 1
            continue
        _render_event(event, full)
    typer.secho(
        f"\n[{steps} model call(s); tokens {prompt_tokens}+{completion_tokens}]", dim=True
    )


def _format_span(seconds: float | None) -> str:
    if seconds is None:
        return "unknown span"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def _offenders(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} x{n}" for name, n in list(counts.items())[:4])


@app.command("report")
def mason_report(
    transcript: Annotated[
        Path | None,
        typer.Argument(
            help="A session transcript (.jsonl); default: the newest conversation."
        ),
    ] = None,
    workspace: _WorkspaceOpt = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the digest as JSON.")
    ] = False,
) -> None:
    """Digest one session: steps, tokens, tools, runs, friction, outcome.

    The digest is arithmetic over the transcript; 'slab mason read' remains
    the event-by-event viewer. Delegation transcripts roll into the totals,
    and the runs the session created come from the workspace record.
    """
    import json

    from foundation.runtime import Workspace
    from mason.report import session_runs, summarize
    from mason.session import transcript_groups

    try:
        root = _ops.resolve_root(workspace)
    except (FoundationError, SlabError) as e:
        _fail(str(e))
    groups = transcript_groups(root)
    if transcript is None:
        if not groups:
            _fail(f"no session transcripts under {root}")
        transcript, siblings = groups[-1]
    else:
        if not transcript.is_file():
            _fail(f"no transcript at {transcript}")
        resolved = transcript.resolve()
        siblings = next(
            (group for conversation, group in groups if conversation.resolve() == resolved),
            [],
        )
    summary = summarize(transcript, siblings)
    try:
        with Workspace(root) as ws:
            summary["runs"] = session_runs(ws, str(summary["session"]))
    except (FoundationError, SlabError, OSError):
        summary["runs"] = None  # no workspace record is not a report failure

    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return

    tokens = f"{summary['total_prompt_tokens']}+{summary['total_completion_tokens']}"
    typer.echo(
        f"session {summary['session']} — {summary['total_steps']} step(s), "
        f"tokens {tokens}, {_format_span(summary['span_s'])}"
    )
    typer.echo(f"  transcript {summary['transcript']}")
    for child in summary["delegations"]:
        typer.echo(
            f"  delegation {child['agent']}: {child['steps']} step(s), "
            f"tokens {child['prompt_tokens']}+{child['completion_tokens']}"
        )
    runs = summary["runs"]
    if runs is None:
        typer.echo("runs: workspace record unavailable")
    elif runs:
        typer.echo("runs this session created:")
        for run in runs:
            typer.echo(
                f"  {run['id'][:10]:<12} {run['name'][:24]:<24} "
                f"{run['state']:<12} {run['status']}"
            )
    else:
        typer.echo("runs this session created: none")
    tools = summary["tools"]
    if tools:
        typer.echo(f"tool calls ({sum(tools.values())}):")
        for name, count in tools.items():
            typer.echo(f"  {name:<22} {count}")
    if summary["refusals"]:
        typer.echo(
            f"refusals: {summary['refusals']} ({_offenders(summary['refused_tools'])})"
        )
    if summary["errored_calls"]:
        typer.echo(
            f"errored calls: {summary['errored_calls']} "
            f"({_offenders(summary['errored_tools'])})"
        )
    typer.echo(f"memory: {summary['recall']} recall, {summary['remember']} remember")
    if summary["skills"]:
        typer.echo(f"skills loaded: {', '.join(summary['skills'])}")
    if summary["compactions"] or summary["resumes"]:
        typer.echo(
            f"compactions: {summary['compactions']}; resumes: {summary['resumes']}"
        )
    if summary["malformed_lines"]:
        typer.echo(f"malformed lines skipped: {summary['malformed_lines']}")
    if summary["first_launch_step"] is not None:
        typer.echo(f"first launch at step {summary['first_launch_step']}")
    finish = summary["finish"]
    if finish["reported"]:
        head = f": {finish['head']}" if finish["head"] else ""
        typer.echo(f"finish reported{head}")
    else:
        typer.echo("no finish report (halted, interrupted, or still running)")


sandbox_app = typer.Typer(
    help=(
        "The no-network container for autonomous runs: preflight the host, "
        "render the batch job, and the in-job plumbing it uses."
    ),
    no_args_is_help=True,
)
app.add_typer(sandbox_app, name="sandbox")


@sandbox_app.command("check")
def mason_sandbox_check(
    workspace: _WorkspaceOpt = None,
) -> None:
    """What this machine still needs before 'sandbox render' output can run."""
    from mason.sandbox import preflight

    try:
        agent, _hpc, root = _serve_inputs(workspace)
        rows = preflight(agent, root)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    for mark, message in rows:
        typer.echo(f"[{mark}] {message}")
    if any(mark == "-" for mark, _ in rows):
        raise typer.Exit(code=1)


@sandbox_app.command("render")
def mason_sandbox_render(
    goal: Annotated[str, typer.Argument(help="The goal 'slab mason run --auto' receives.")],
    workspace: _WorkspaceOpt = None,
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition for the engine legs.")
    ] = None,
    time_limit: _TimeOpt = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for the rendered files (default: ./sandbox)."),
    ] = None,
    engine_tasks: Annotated[
        int | None,
        typer.Option(
            "--engine-tasks",
            min=1,
            help="Pin the MPI rank count for in-job engines (default: the "
            "allocation's SLURM_NTASKS — often too many for small cells).",
        ),
    ] = None,
) -> None:
    """Write the batch script, slab.toml, context.md, and render.json.

    Read them, then submit with sbatch — or use 'slab mason sandbox
    launch', which renders fresh and submits in one motion.
    """
    script_path, _script = _render_sandbox_files(
        goal,
        workspace=workspace,
        partition=partition,
        time_limit=time_limit,
        out=out,
        engine_tasks=engine_tasks,
    )
    typer.echo(
        "read these files, then submit with: "
        f"sbatch {script_path} — the job aborts unless the container "
        "proves it is offline and the bridged endpoint answers"
    )


def _render_sandbox_files(
    goal: str,
    *,
    workspace: Path | None,
    partition: str | None,
    time_limit: str | None,
    out: Path | None,
    engine_tasks: int | None,
) -> tuple[Path, str]:
    """Render and write the four sandbox files; echo warnings and paths."""
    import json

    from mason.sandbox import (
        render_record,
        render_sandbox_script,
        sandbox_toml,
        snapshot_engines,
    )
    from slab.config import load_config as load_slab_config

    project = Path.cwd()
    out_dir = (out if out is not None else project / "sandbox").resolve()
    toml_path = out_dir / "slab.toml"
    try:
        agent, hpc, root = _serve_inputs(workspace)
        slab_cfg = load_slab_config(project)
        # Runs each engine's setup lines once, here on the host, so their
        # module loads can be frozen into exports and binds the container
        # can actually use.
        snapshots = snapshot_engines(slab_cfg)
        toml_text, toml_warnings = sandbox_toml(slab_cfg, agent, root.resolve(), snapshots)
        script, bind_warnings, sandbox_context = render_sandbox_script(
            agent,
            hpc,
            slab_cfg,
            root.resolve(),
            project,
            goal,
            toml_path=toml_path,
            partition=partition,
            time_limit=time_limit,
            snapshots=snapshots,
            engine_tasks=engine_tasks,
        )
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "mason-sandbox.sbatch"
    script_path.write_text(script.rstrip("\n") + "\n", encoding="utf-8")
    toml_path.write_text(toml_text, encoding="utf-8")
    context_path = out_dir / "context.md"
    context_path.write_text(sandbox_context.rstrip("\n") + "\n", encoding="utf-8")
    record = render_record(
        goal,
        partition=partition,
        time_limit=time_limit,
        engine_tasks=engine_tasks,
        out_dir=out_dir,
    )
    record_path = out_dir / "render.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for note in (*toml_warnings, *bind_warnings):
        if "snapshotted from the host" in note:
            typer.echo(f"[=] {note}")
        else:
            typer.secho(f"[!] {note}", err=True, fg=typer.colors.YELLOW)
    typer.echo(f"wrote {script_path}")
    typer.echo(f"wrote {toml_path}")
    typer.echo(f"wrote {context_path}")
    typer.echo(f"wrote {record_path}")
    return script_path, script


@sandbox_app.command("launch")
def mason_sandbox_launch(
    goal: Annotated[
        str | None,
        typer.Argument(
            help="The goal; omit to reuse the last render's recorded arguments."
        ),
    ] = None,
    workspace: _WorkspaceOpt = None,
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition for the engine legs.")
    ] = None,
    time_limit: _TimeOpt = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for the rendered files (default: ./sandbox)."),
    ] = None,
    engine_tasks: Annotated[
        int | None,
        typer.Option("--engine-tasks", min=1, help="Pin the MPI rank count for in-job engines."),
    ] = None,
) -> None:
    """Preflight, render fresh, and submit — one motion, never a stale render.

    Every launch re-renders, so the submitted job always matches the
    installed code and the current config. Without a goal, the arguments
    recorded in render.json are reused (a flag still overrides a recorded
    value); with a goal, only the flags given here apply.
    """
    from mason.sandbox import preflight, read_render_record
    from slab.hpc import submit

    project = Path.cwd()
    out_dir = (out if out is not None else project / "sandbox").resolve()
    if goal is None:
        record = read_render_record(out_dir)
        if record is None:
            _fail(
                f"no goal given and no usable render.json in {out_dir}; "
                f"pass the goal (later launches can then omit it)"
            )
        goal = str(record["goal"])
        if partition is None and record.get("partition") is not None:
            partition = str(record["partition"])
        if time_limit is None and record.get("time_limit") is not None:
            time_limit = str(record["time_limit"])
        if engine_tasks is None and record.get("engine_tasks") is not None:
            engine_tasks = int(str(record["engine_tasks"]))
    try:
        agent, hpc, root = _serve_inputs(workspace)
        rows = preflight(agent, root)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    for mark, message in rows:
        typer.echo(f"[{mark}] {message}")
    if any(mark == "-" for mark, _ in rows):
        _fail("preflight failed; fix the [-] rows, then launch again")
    _script_path, script = _render_sandbox_files(
        goal,
        workspace=workspace,
        partition=partition,
        time_limit=time_limit,
        out=out_dir,
        engine_tasks=engine_tasks,
    )
    try:
        resolved, _spec = hpc.resolve_partition(partition)
        job = submit(script, job_name="mason-sandbox", partition=resolved, directory=out_dir)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo(f"submitted job {job.job_id} ({job.job_name}) to {job.partition}")
    typer.echo(f"script: {job.script_path}")
    typer.echo(f"watch it with 'slab hpc status {job.job_id}'; the .out lands in {out_dir}")


@sandbox_app.command("forward")
def mason_sandbox_forward(
    socket_path: Annotated[str, typer.Argument(help="The bridge's unix socket.")],
    port: Annotated[int, typer.Option("--port", help="Loopback port to serve.")] = 8000,
) -> None:
    """In-job plumbing: relay 127.0.0.1:PORT to the bridge socket (runs until killed)."""
    from mason.sandbox import forward

    forward(socket_path, port)


@sandbox_app.command("bridge")
def mason_sandbox_bridge(
    socket_path: Annotated[str, typer.Argument(help="The unix socket to serve.")],
    upstream: Annotated[
        str, typer.Argument(help="Fixed destination: host:port, or an http(s) URL.")
    ],
    key_env: Annotated[
        str | None,
        typer.Option(
            "--key-env",
            help="Name of the environment variable holding the gateway's API key "
            "(URL upstreams only; the value never enters the container).",
        ),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option(
            "--header",
            help="Fixed 'Name: value' header injected into every gateway request "
            "(repeatable; URL upstreams only) — the routing header a gateway "
            "deployment requires, e.g. 'x-portkey-provider: openai'. Chosen "
            "here on the host, never by the sandbox.",
        ),
    ] = None,
) -> None:
    """In-job plumbing: relay the bridge socket to the model endpoint (host side)."""
    from mason.sandbox import bridge, parse_bridge_headers

    try:
        bridge(
            socket_path,
            upstream,
            key_env=key_env,
            headers=parse_bridge_headers(header or []),
        )
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))


@sandbox_app.command("verify")
def mason_sandbox_verify(
    port: Annotated[int, typer.Option("--port", help="The bridged loopback port.")] = 8000,
    probe_url: Annotated[
        str | None,
        typer.Option(
            "--probe-url",
            help="A URL that must NOT be reachable. Unset probes a default set "
            "including a raw IP, so a broken resolver cannot look like darkness.",
        ),
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for the endpoint.")
    ] = 30.0,
) -> None:
    """In-job plumbing: prove the sandbox is dark and the endpoint answers."""
    from mason.sandbox import verify

    try:
        names = verify(port, probe_url=probe_url, ready_timeout_s=timeout)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo("[+] no external route is reachable (the sandbox is dark)")
    typer.echo(f"[+] 127.0.0.1:{port} answers; serving: {', '.join(names) or 'none'}")


def _serve_hint(agent: AgentConfig, root: Path, origin: str, *, cluster: str = "") -> list[str]:
    """Why an unreachable endpoint might be unreachable, when we can tell."""
    from mason.serve import read_record
    from slab.hpc import job_state

    if agent.provider != "openai":
        return []
    try:
        record = read_record(root)
    except (MasonError, FoundationError, SlabError) as e:
        return [f"    (the endpoint record is unreadable: {e})"]
    if record is None:
        # A one-shot --endpoint meant that server, so 'start your own' is noise.
        # A *configured* endpoint that no longer answers is the opposite case:
        # it is usually last allocation's node, and serve start is the fix.
        if origin == "--endpoint":
            return []
        return [
            "    no server is recorded here; start one with 'slab mason serve start' "
            "(or point [agent] endpoint at a server you started yourself)"
        ]
    if not record.job_id:
        return [f"    a record exists ({record.endpoint}) but names no job to ask about"]
    if record.cluster and record.cluster != cluster:
        # A job id is only meaningful on its own cluster; asking this one
        # would describe an unrelated job that happens to share the number.
        return [
            f"    the record belongs to cluster {record.cluster!r}; job "
            f"{record.job_id} is not queried from here (job ids are per-cluster)"
        ]
    try:
        status = job_state(record.job_id)
    except (MasonError, FoundationError, SlabError) as e:
        return [f"    job {record.job_id}: state unknown — {e}"]
    if status.state.is_terminal:
        return [
            f"    job {record.job_id} ended as {status.state.value}; the record is "
            f"stale — 'slab mason serve stop' clears it"
        ]
    return [f"    job {record.job_id} is {status.state.value}; the model may still be loading"]


@app.command("doctor")
def mason_doctor(
    workspace: _WorkspaceOpt = None,
    model: _ModelOpt = None,
    endpoint: _EndpointOpt = None,
    provider: _ProviderOpt = None,
) -> None:
    """Check the model endpoint: reachable, model served, tool calls parsed."""
    from mason.client import ChatClient, LlmError
    from mason.serve import discover_endpoint

    try:
        agent, hpc, root = _serve_inputs(workspace)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    cluster = hpc.cluster or ""
    overrides: dict[str, object] = {}
    if provider is not None:
        overrides["provider"] = provider
    if endpoint is not None:
        overrides["endpoint"] = endpoint
    agent = _override_agent(agent, overrides)
    try:
        resolved_endpoint, origin = discover_endpoint(agent, root)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    if endpoint is not None:
        origin = "--endpoint"
    resolved_model = model or agent.model
    typer.echo(f"provider: {agent.provider}")
    typer.echo(f"endpoint: {resolved_endpoint}  [{origin}]")
    typer.echo(f"model:    {resolved_model or '(not configured)'}")
    client: Any
    api_key = _probe_key(agent, label="")
    if agent.provider == "anthropic":
        from mason.anthropic import AnthropicClient

        assert api_key is not None  # resolved_api_key_env always names one here
        client = AnthropicClient(
            resolved_model or "unconfigured", api_key, endpoint=resolved_endpoint, timeout_s=60.0
        )
    else:
        client = ChatClient(
            resolved_endpoint,
            resolved_model or "unconfigured",
            api_key=api_key,
            timeout_s=60.0,
        )
    failed = 0
    try:
        names = client.model_names()
        typer.echo(f"[+] endpoint answers; {len(names)} model(s) served")
    except LlmError as e:
        typer.echo(f"[x] endpoint: {e}")
        for line in _serve_hint(agent, root, origin, cluster=cluster):
            typer.echo(line)
        raise typer.Exit(code=1) from None
    if resolved_model is None:
        typer.echo(f"[x] no model configured; served here: {', '.join(names) or 'none'}")
        raise typer.Exit(code=1)
    if resolved_model in names:
        typer.echo(f"[+] model {resolved_model!r} is served")
    else:
        failed += 1
        typer.echo(f"[x] model {resolved_model!r} not served; available: {', '.join(names)}")
    ping = {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Reply with a pong.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    try:
        reply = client.chat(
            [{"role": "user", "content": "Call the ping tool now."}], tools=[ping]
        )
        if reply.tool_calls:
            typer.echo("[+] native tool calls work")
        else:
            failed += 1
            typer.echo(
                "[x] the model answered without a tool call — the server may lack a "
                "tool-call parser; try [agent] tool_protocol = \"fenced\""
            )
    except LlmError as e:
        failed += 1
        typer.echo(f"[x] tool-call probe: {e}")
    primary = (agent.provider, resolved_endpoint, resolved_model)
    failed += _doctor_roster(agent, root, seen={primary})
    if failed:
        raise typer.Exit(code=1)


def _probe_key(agent: AgentConfig, *, label: str) -> str | None:
    """The API key a doctor probe must send, or exit reporting the missing one.

    The probe has to authenticate exactly as the session will. Sending
    nothing where the config names a key turns a working connection into a
    401 whose message tells you to configure what you already configured.
    """
    key_var = agent.resolved_api_key_env
    if key_var is None:
        return None
    api_key = os.environ.get(key_var)
    if not api_key:
        typer.echo(f"[x] {label}${key_var} is not set — [agent] api_key_env names it")
        raise typer.Exit(code=1)
    return api_key


def _doctor_roster(
    agent: AgentConfig, root: Path, *, seen: set[tuple[str, str, str | None]]
) -> int:
    """Probe the roster's distinct model connections; return the failure count.

    A specialist pinned to an unserved model should fail the doctor, not
    the first delegation. Only ``[agent.roster.<name>]`` tables are probed —
    an agent without a table shares the primary connection checked above.
    """
    if not agent.roster:
        return 0
    from mason.client import ChatClient, LlmError
    from mason.config import roster_agent_config
    from mason.roster import check_overrides, discover_roster
    from mason.serve import discover_endpoint

    try:
        check_overrides(agent, discover_roster(Path.cwd()))
    except (MasonError, FoundationError, SlabError) as e:
        typer.echo(f"[x] roster: {e}")
        return 1
    failures = 0
    for name in sorted(agent.roster):
        effective = roster_agent_config(agent, name)
        try:
            endpoint, _origin = discover_endpoint(effective, root)
        except (MasonError, FoundationError, SlabError) as e:
            typer.echo(f"[x] {name}: {e}")
            failures += 1
            continue
        key = (effective.provider, endpoint, effective.model)
        if key in seen:
            continue
        seen.add(key)
        if effective.model is None:
            typer.echo(f"[x] {name}: no model configured for its connection")
            failures += 1
            continue
        client: Any
        try:
            api_key = _probe_key(effective, label=f"{name}: ")
        except typer.Exit:
            failures += 1
            continue
        if effective.provider == "anthropic":
            from mason.anthropic import AnthropicClient

            assert api_key is not None  # resolved_api_key_env always names one here
            client = AnthropicClient(
                effective.model, api_key, endpoint=endpoint, timeout_s=60.0
            )
        else:
            client = ChatClient(endpoint, effective.model, api_key=api_key, timeout_s=60.0)
        try:
            names = client.model_names()
        except LlmError as e:
            typer.echo(f"[x] {name}: endpoint {endpoint}: {e}")
            failures += 1
            continue
        if effective.model in names:
            typer.echo(f"[+] {name}: model {effective.model!r} is served at {endpoint}")
        else:
            typer.echo(
                f"[x] {name}: model {effective.model!r} not served at {endpoint}; "
                f"available: {', '.join(names) or 'none'}"
            )
            failures += 1
    return failures


if __name__ == "__main__":  # pragma: no cover - module execution convenience
    app()
