"""The ``mason`` command-line interface.

``chat`` and ``run`` drive the resident research agent, interactively and
autonomously. ``doctor`` is the empirical check that the configured endpoint
answers and calls tools. ``serve`` starts, locates, and stops the model server
as an ordinary batch job, because on a cluster the GPU node is the scheduler's
choice and the endpoint is discovered rather than configured.

Runs and artifacts are the ``foundation`` command; engines, protocols, and the
scheduler are the ``slab`` command.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer

if TYPE_CHECKING:
    from mason.config import AgentConfig
    from mason.session import MasonSession
    from slab.config import HpcConfig

from foundation import _ops
from foundation.errors import FoundationError
from mason.errors import MasonError
from slab._version import __version__
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


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"mason {__version__}")
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
    """Mason — the resident research agent."""


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

    # Non-interactive entry points (mason run) get the refuse-everything
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
    if updates:
        session.agent = session.agent.model_copy(update=updates)
    # After a provider change, which endpoint is right changes too; a
    # --endpoint flag outranks both the config and any discovered server.
    if updates or endpoint is not None:
        session.resolve_endpoint(endpoint)
    return session


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
) -> None:
    """Interactive session ( /quit to leave, /status for token counts )."""
    from mason import Mason

    try:
        session = _mason_session(
            workspace, auto=auto, model=model, endpoint=endpoint, provider=provider
        )
        resume_from = None
        if resume:
            latest = session.latest_transcript()
            if latest is None:
                _fail("nothing to resume: no session transcripts in this workspace")
            resume_from = session.load_messages(latest)
            typer.echo(f"resuming {latest.name} ({len(resume_from)} messages)")
        mason = Mason(session, resume_from=resume_from)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo(
        f"mason ready: {session.agent.model} at {session.endpoint} "
        f"[{session.endpoint_origin}]"
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
        mason = Mason(session)
        result = mason.run_turn(goal)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    typer.echo(result.text)
    typer.echo(
        f"[{result.stop_reason} after {result.steps} step(s); "
        f"tokens {session.prompt_tokens}+{session.completion_tokens}; "
        f"transcript {session.transcript_path}]",
        err=True,
    )
    if result.stop_reason not in ("answer", "finish"):
        raise typer.Exit(code=1)


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
            "watch it with 'mason serve status'"
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
            "    no server is recorded here; start one with 'mason serve start' "
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
            f"stale — 'mason serve stop' clears it"
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
    from mason.config import load_config
    from mason.serve import discover_endpoint
    from slab.config import load_config as load_slab_config

    try:
        agent = load_config().agent
        cluster = load_slab_config().hpc.cluster or ""
        root = _ops.resolve_root(workspace)
    except (MasonError, FoundationError, SlabError) as e:
        _fail(str(e))
    if provider is not None:
        agent = agent.model_copy(update={"provider": provider})
    if endpoint is not None:
        agent = agent.model_copy(update={"endpoint": endpoint})
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
    if agent.provider == "anthropic":
        from mason.anthropic import AnthropicClient

        key_var = agent.resolved_api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(key_var)
        if not api_key:
            typer.echo(f"[x] ${key_var} is not set — the Anthropic provider needs a key")
            raise typer.Exit(code=1)
        client = AnthropicClient(
            resolved_model or "unconfigured", api_key, endpoint=resolved_endpoint, timeout_s=60.0
        )
    else:
        client = ChatClient(resolved_endpoint, resolved_model or "unconfigured", timeout_s=60.0)
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
    if failed:
        raise typer.Exit(code=1)


def main() -> None:  # pragma: no cover - console-script shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
