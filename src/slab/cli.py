"""The ``slab`` command-line interface.

Verbs describe what can be computed here and how to reach it: ``engines``
inspects and verifies the cluster registry, ``pseudos`` installs and checks
pseudopotential families, ``protocols`` shows the named Quantum ESPRESSO
input protocols, ``hpc`` renders and submits SLURM jobs, and ``config``
explains where every setting came from.

Runs, artifacts, and verification are the ``foundation`` command.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer

if TYPE_CHECKING:
    from slab.config import AgentConfig, HpcConfig
    from slab.mason.session import MasonSession

from foundation import _ops
from slab._ops import engines_overview
from slab._version import __version__
from slab.errors import SlabError

app = typer.Typer(
    help="SLAB — access to atomistic engines, registries, protocols, "
    "pseudopotentials, and the scheduler.",
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
    """SLAB — the simplest layer for atomistic backends."""


engines_app = typer.Typer(
    help="Inspect and verify the cluster engine registry (rootstock-style software management).",
    no_args_is_help=True,
)
app.add_typer(engines_app, name="engines")

_RegistryOpt = Annotated[
    Path | None,
    typer.Option(
        "--registry",
        help="Engine registry JSON file (default: $SLAB_ENGINES, then ~/.config/slab/).",
    ),
]


@engines_app.command("list")
def engines_list(registry_path: _RegistryOpt = None) -> None:
    """List available engines: built-ins plus everything this cluster declares."""
    try:
        overview = engines_overview(registry_path)
    except (SlabError, OSError, ValueError) as e:
        _fail(str(e))
    typer.echo(f"built-in: {', '.join(overview['builtin'])}")
    registry = overview["registry"]
    if registry is None:
        typer.echo("registry: none configured (set $SLAB_ENGINES to a cluster's engines.json)")
    else:
        where = f" [{registry['cluster']}]" if registry["cluster"] else ""
        typer.echo(f"registry{where}: {registry['path']}")
        for name, spec in registry["engines"].items():
            version = spec["version"] or "unversioned"
            probe = "probe" if spec["verified_by_probe"] else "no probe"
            description = f"  {spec['description']}" if spec["description"] else ""
            typer.echo(f"  {name:<14} {version:<14} {spec['calculator']}  ({probe}){description}")
    rootstock = overview["rootstock"]
    if rootstock is not None:
        typer.echo(f"rootstock checkpoints (usable directly as engine=): {rootstock['root']}")
        if rootstock.get("error"):
            typer.echo(f"  error reading install: {rootstock['error']}")
        for env_name, ids in rootstock["checkpoints"].items():
            typer.echo(f"  {env_name}: {', '.join(ids)}")
    typer.echo(f"qe protocols: {', '.join(overview['qe_protocols'])} ('slab protocols show')")
    families = overview["pseudo_families"]
    if overview.get("pseudo_families_error"):
        typer.echo(f"pseudo families: error — {overview['pseudo_families_error']}")
    elif families:
        typer.echo(f"pseudo families: {', '.join(families)}")
    else:
        typer.echo("pseudo families: none installed ('slab pseudos install sssp' fetches one)")
    hpc = overview.get("hpc")
    if overview.get("hpc_error"):
        typer.echo(f"hpc partitions: error — {overview['hpc_error']}")
    elif hpc:
        cluster = f" [{hpc['cluster']}]" if hpc["cluster"] else ""
        default = f" (default {hpc['default_partition']})" if hpc["default_partition"] else ""
        typer.echo(
            f"hpc partitions{cluster}: {', '.join(hpc['partitions'])}{default} "
            f"('slab hpc partitions')"
        )


@engines_app.command("verify")
def engines_verify(registry_path: _RegistryOpt = None) -> None:
    """Run every registry engine's probe; exit nonzero if any engine fails."""
    from slab.engines import load_registry, verify_engines

    try:
        registry = load_registry(registry_path)
    except (SlabError, OSError, ValueError) as e:
        _fail(str(e))
    if registry is None:
        _fail("no engine registry configured (set $SLAB_ENGINES or pass --registry)")
    results = verify_engines(registry)
    failed = 0
    for result in results:
        mark = "+" if result.ok else "x"
        typer.echo(f"[{mark}] {result.engine}: {result.detail}")
        failed += 0 if result.ok else 1
    typer.echo(f"{len(results) - failed}/{len(results)} engines verified")
    if failed:
        raise typer.Exit(code=1)


pseudos_app = typer.Typer(
    help="Install and inspect pseudopotential families (aiida-pseudo's pattern).",
    no_args_is_help=True,
)
app.add_typer(pseudos_app, name="pseudos")


@pseudos_app.command("install")
def pseudos_install(
    kind: Annotated[str, typer.Argument(help="Family kind; only 'sssp' installs today.")] = "sssp",
    version: Annotated[
        str, typer.Option("--version", "-v", help="SSSP version, e.g. 1.3.")
    ] = "1.3",
    functional: Annotated[
        str, typer.Option("--functional", "-x", help="XC functional: PBE or PBEsol.")
    ] = "PBEsol",
    precision: Annotated[
        str, typer.Option("--precision", "-p", help="SSSP variant: efficiency or precision.")
    ] = "efficiency",
    force: Annotated[bool, typer.Option("--force", help="Replace an existing install.")] = False,
) -> None:
    """Download and verify a pseudopotential family from its official archive."""
    from slab.pseudos import family_digest, install_sssp

    if kind.strip().lower() != "sssp":
        _fail(
            f"only 'sssp' families install today, not {kind!r} (PseudoDojo is served over "
            f"unverified HTTP upstream; point pseudo_dir= at your own files instead)"
        )
    typer.echo(f"downloading SSSP {version} {functional} {precision} from Materials Cloud ...")
    try:
        family, directory = install_sssp(
            version=version, functional=functional, precision=precision, force=force
        )
    except SlabError as e:
        _fail(str(e))
    typer.echo(
        f"installed {family.name} ({len(family.elements)} elements, "
        f"digest {family_digest(family)}) at {directory}"
    )


@pseudos_app.command("list")
def pseudos_list() -> None:
    """List installed pseudopotential families."""
    from slab.pseudos import family_digest, list_families, pseudos_root

    try:
        families = list_families()
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"root: {pseudos_root()}")
    if not families:
        typer.echo("no families installed ('slab pseudos install sssp' fetches one)")
        return
    for family, directory in families:
        typer.echo(
            f"  {family.name:<34} {len(family.elements):>3} elements  "
            f"digest {family_digest(family)}  {directory}"
        )


@pseudos_app.command("verify")
def pseudos_verify(
    name: Annotated[str, typer.Argument(help="Installed family name (version prefix ok).")],
) -> None:
    """Re-hash a family's files against its manifest; exit nonzero on mismatch."""
    from slab.pseudos import find_family, verify_family

    try:
        family, directory = find_family(name)
    except SlabError as e:
        _fail(str(e))
    problems = verify_family(family, directory)
    for problem in problems:
        typer.echo(f"[x] {problem}")
    if problems:
        raise typer.Exit(code=1)
    typer.echo(f"[+] {family.name}: all {len(family.elements)} files match their checksums")


hpc_app = typer.Typer(
    help="SLURM plumbing driven by the [hpc] section of the slab config.",
    no_args_is_help=True,
)
app.add_typer(hpc_app, name="hpc")


@hpc_app.command("partitions")
def hpc_partitions() -> None:
    """List the partitions this machine's config declares."""
    from slab.config import load_config

    try:
        hpc = load_config().hpc
    except SlabError as e:
        _fail(str(e))
    if not hpc.partitions:
        typer.echo(
            "no partitions declared (add [hpc.partitions.NAME] tables to the slab "
            "config; 'slab config init' shows the shape)"
        )
        return
    if hpc.cluster:
        typer.echo(f"cluster: {hpc.cluster}")
    for name in sorted(hpc.partitions):
        spec = hpc.partitions[name]
        default = " (default)" if name == hpc.default_partition else ""
        time_limit = spec.time_limit or "no time limit set"
        extras = ", ".join(
            piece
            for piece in (spec.gres, spec.mem and f"mem {spec.mem}", spec.qos and f"qos {spec.qos}")
            if piece
        )
        detail = f"  {extras}" if extras else ""
        description = f"  {spec.description}" if spec.description else ""
        typer.echo(f"  {name:<12}{default:<10} {time_limit}{detail}{description}")


@hpc_app.command("render")
def hpc_render(
    command: Annotated[str, typer.Argument(help="Command the job runs, e.g. 'slab run relax.py'.")],
    name: Annotated[str, typer.Option("--name", "-n", help="Job name.")] = "slab-job",
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition (default: config's).")
    ] = None,
    time_limit: Annotated[
        str | None, typer.Option("--time", help="Override the partition's time limit.")
    ] = None,
) -> None:
    """Render the sbatch script that submit would use — read before trusting."""
    from slab.hpc import render_sbatch

    try:
        script = render_sbatch(command, job_name=name, partition=partition, time_limit=time_limit)
    except SlabError as e:
        _fail(str(e))
    typer.echo(script)


@hpc_app.command("submit")
def hpc_submit(
    command: Annotated[str, typer.Argument(help="Command the job runs, e.g. 'slab run relax.py'.")],
    name: Annotated[str, typer.Option("--name", "-n", help="Job name.")] = "slab-job",
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition (default: config's).")
    ] = None,
    time_limit: Annotated[
        str | None, typer.Option("--time", help="Override the partition's time limit.")
    ] = None,
    directory: Annotated[
        Path | None, typer.Option("--dir", help="Where the job runs (default: cwd).")
    ] = None,
) -> None:
    """Render and submit a job; the exact script is kept next to its outputs."""
    from slab.config import load_config
    from slab.hpc import render_sbatch, submit

    try:
        hpc = load_config().hpc
        resolved, _spec = hpc.resolve_partition(partition)
        script = render_sbatch(
            command, job_name=name, partition=resolved, config=hpc, time_limit=time_limit
        )
        job = submit(script, job_name=name, partition=resolved, directory=directory)
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"submitted job {job.job_id} ({job.job_name}) to {job.partition}")
    typer.echo(f"script: {job.script_path}")


@hpc_app.command("status")
def hpc_status(
    job_id: Annotated[str, typer.Argument(help="SLURM job id.")],
) -> None:
    """One job's state (squeue first, sacct fallback)."""
    from slab.hpc import job_state

    try:
        status = job_state(job_id)
    except SlabError as e:
        _fail(str(e))
    raw = f"  ({status.raw})" if status.raw and status.raw != status.state.value.upper() else ""
    detail = f"  {status.detail}" if status.detail else ""
    typer.echo(f"job {status.job_id}: {status.state.value}{raw}{detail}")


@hpc_app.command("cancel")
def hpc_cancel(
    job_id: Annotated[str, typer.Argument(help="SLURM job id.")],
) -> None:
    """Cancel a job (idempotent: cancelling a finished job is a no-op)."""
    from slab.hpc import cancel

    try:
        cancel(job_id)
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"cancel requested for job {job_id}")


config_app = typer.Typer(
    help="Layered TOML configuration: site, user, and project files merged key-by-key.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show the merged configuration and which file each value came from."""
    from slab.config import find_config_files, load_config_with_origins

    try:
        files = find_config_files()
        merged, origins = load_config_with_origins()
    except SlabError as e:
        _fail(str(e))
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "files": [{"layer": layer, "path": str(path)} for layer, path in files],
                    "config": merged.model_dump(mode="json"),
                    "origins": origins,
                },
                indent=2,
            )
        )
        return
    if not files:
        typer.echo(
            "no config files found (site: $SLAB_SITE_CONFIG, user: "
            "~/.config/slab/config.toml, project: ./slab.toml or $SLAB_CONFIG); "
            "'slab config init' writes a template"
        )
        return
    for layer, path in files:
        typer.echo(f"{layer}: {path}")
    for dotted in sorted(origins):
        value = _dig(merged, dotted)
        typer.echo(f"  {dotted} = {value!r}  [{origins[dotted]}]")
    typer.echo("unset keys use built-in defaults ('slab config init' shows them all)")


def _dig(model: object, dotted: str) -> object:
    """Fetch a dotted key path off the validated config model."""
    node = model
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else getattr(node, part, None)
    return node


@config_app.command("init")
def config_init(
    path: Annotated[
        Path, typer.Argument(help="Where to write the template (default ./slab.toml).")
    ] = Path("slab.toml"),
    user: Annotated[
        bool, typer.Option("--user", help="Write the user-layer file (~/.config/slab/) instead.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
) -> None:
    """Write a fully commented configuration template."""
    from slab.config import user_config_path, write_template

    target = user_config_path() if user else path
    try:
        written = write_template(target, force=force)
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"wrote {written}")


protocols_app = typer.Typer(
    help="Named QE input protocols (adapted from aiida-quantumespresso).",
    no_args_is_help=True,
)
app.add_typer(protocols_app, name="protocols")


@protocols_app.command("list")
def protocols_list() -> None:
    """List the named protocols and what they trade off."""
    from slab.protocols import available_protocols, get_protocol

    for name in available_protocols():
        protocol = get_protocol(name)
        typer.echo(f"{name:<10} {protocol.description}")


@protocols_app.command("show")
def protocols_show(
    name: Annotated[str, typer.Argument(help="fast, balanced, or stringent.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show one protocol's data (QE-native units) and derived values."""
    from slab.protocols import protocol_details

    try:
        details = protocol_details(name)
    except SlabError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(details, indent=1, sort_keys=True))
        return
    for key in sorted(details):
        typer.echo(f"{key}: {details[key]}")


mason_app = typer.Typer(
    help="Mason, the resident research agent — a coding-agent harness for open models.",
    no_args_is_help=True,
)
app.add_typer(mason_app, name="mason")


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
    from slab.mason import MasonSession

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


@mason_app.command("chat")
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
    from slab.mason import Mason

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
    except SlabError as e:
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
        except SlabError as e:
            typer.echo(f"error: {e}", err=True)
            continue
        typer.echo(f"\n{result.text}")
        if result.stop_reason not in ("answer", "finish"):
            typer.echo(f"[stopped: {result.stop_reason} after {result.steps} steps]", err=True)


@mason_app.command("run")
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
    from slab.mason import Mason

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
    except SlabError as e:
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
mason_app.add_typer(serve_app, name="serve")

_PartitionOpt = Annotated[
    str | None, typer.Option("--partition", "-p", help="Partition (default: [agent.serve]'s).")
]
_PortOpt = Annotated[int | None, typer.Option("--port", help="Override [agent.serve] port.")]
_TimeOpt = Annotated[str | None, typer.Option("--time", help="Override the job's time limit.")]


def _serve_inputs(workspace: Path | None) -> tuple[AgentConfig, HpcConfig, Path]:
    from slab.config import load_config

    config = load_config()
    return config.agent, config.hpc, _ops.resolve_root(workspace)


@serve_app.command("render")
def mason_serve_render(
    workspace: _WorkspaceOpt = None,
    partition: _PartitionOpt = None,
    port: _PortOpt = None,
    time_limit: _TimeOpt = None,
) -> None:
    """Print the batch script 'serve start' would submit — read it first."""
    from slab.mason.serve import render_serve_script

    try:
        agent, hpc, root = _serve_inputs(workspace)
        script = render_serve_script(
            agent, hpc, root, partition=partition, port=port, time_limit=time_limit
        )
    except SlabError as e:
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
    from slab.mason.serve import start, wait_for_record, wait_until_ready

    try:
        agent, hpc, root = _serve_inputs(workspace)
        job = start(agent, hpc, root, partition=partition, port=port, time_limit=time_limit)
    except SlabError as e:
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
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"[+] {record.endpoint} answers; serving: {', '.join(names) or 'none'}")


@serve_app.command("status")
def mason_serve_status(
    workspace: _WorkspaceOpt = None,
) -> None:
    """What the recorded server is: endpoint, job state, and a live probe."""
    from slab.mason.serve import describe

    try:
        agent, hpc, root = _serve_inputs(workspace)
        for line in describe(agent, root, cluster=hpc.cluster or ""):
            typer.echo(line)
    except SlabError as e:
        _fail(str(e))


@serve_app.command("stop")
def mason_serve_stop(
    workspace: _WorkspaceOpt = None,
) -> None:
    """Cancel the recorded server job and remove its endpoint record."""
    from slab.mason.serve import stop

    try:
        _agent, hpc, root = _serve_inputs(workspace)
        typer.echo(stop(root, cluster=hpc.cluster or ""))
    except SlabError as e:
        _fail(str(e))


def _serve_hint(agent: AgentConfig, root: Path, origin: str, *, cluster: str = "") -> list[str]:
    """Why an unreachable endpoint might be unreachable, when we can tell."""
    from slab.hpc import job_state
    from slab.mason.serve import read_record

    if agent.provider != "openai":
        return []
    try:
        record = read_record(root)
    except SlabError as e:
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
    except SlabError as e:
        return [f"    job {record.job_id}: state unknown — {e}"]
    if status.state.is_terminal:
        return [
            f"    job {record.job_id} ended as {status.state.value}; the record is "
            f"stale — 'slab mason serve stop' clears it"
        ]
    return [f"    job {record.job_id} is {status.state.value}; the model may still be loading"]


@mason_app.command("doctor")
def mason_doctor(
    workspace: _WorkspaceOpt = None,
    model: _ModelOpt = None,
    endpoint: _EndpointOpt = None,
    provider: _ProviderOpt = None,
) -> None:
    """Check the model endpoint: reachable, model served, tool calls parsed."""
    from slab.config import load_config
    from slab.mason.client import ChatClient, LlmError
    from slab.mason.serve import discover_endpoint

    try:
        doctor_config = load_config()
        agent = doctor_config.agent
        root = _ops.resolve_root(workspace)
    except SlabError as e:
        _fail(str(e))
    if provider is not None:
        agent = agent.model_copy(update={"provider": provider})
    if endpoint is not None:
        agent = agent.model_copy(update={"endpoint": endpoint})
    try:
        resolved_endpoint, origin = discover_endpoint(agent, root)
    except SlabError as e:
        _fail(str(e))
    if endpoint is not None:
        origin = "--endpoint"
    resolved_model = model or agent.model
    typer.echo(f"provider: {agent.provider}")
    typer.echo(f"endpoint: {resolved_endpoint}  [{origin}]")
    typer.echo(f"model:    {resolved_model or '(not configured)'}")
    client: Any
    if agent.provider == "anthropic":
        from slab.mason.anthropic import AnthropicClient

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
        for line in _serve_hint(agent, root, origin, cluster=doctor_config.hpc.cluster or ""):
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
