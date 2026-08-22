"""Standing the agent's model server up on a GPU node, and finding it again.

Mason talks to an OpenAI-compatible endpoint. On a laptop that endpoint is a
constant — Ollama at ``http://localhost:11434/v1`` — and a config file can
name it. On a cluster it cannot: the server runs on whichever GPU node the
scheduler hands out, so its URL does not exist until the job starts. Writing
``endpoint = "http://gpu-node-01:8000/v1"`` into a config file is a guess
that goes stale on the next allocation.

So the URL is **discovered, not configured**. ``[agent.serve]`` declares the
launch (partition, port, vLLM flags); the batch job's first act is to write
its own endpoint into ``<workspace>/mason/endpoint.json``, and to delete that
record when the server exits, so a dead node cannot keep answering for a live
one. :func:`discover_endpoint` reads it, and reports where the endpoint came
from — the same origin-tracking habit as ``slab config show``.

The record file is the only coupling between the login node and the compute
node, which assumes the workspace is on a shared filesystem. That is the
normal HPC arrangement (``/scratch/$USER/...``), and when it does not hold the
symptom is honest: no record, and the endpoint falls back to the provider
default.

Nothing here is required to *use* Mason on a cluster — an endpoint set by hand
still wins. This module removes the ceremony, not the option.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from mason.errors import MasonError
from slab.config import AgentConfig, HpcConfig, ServeConfig
from slab.errors import SlabError
from slab.hpc import SubmittedJob, job_state, render_sbatch, submit

RECORD_NAME = "endpoint.json"
_DEFAULT_POLL_S = 10.0
# A model name reaches a shell command, a JSON heredoc, and the served-model
# check in 'mason doctor' — so it must survive all three as *one* string,
# untransformed. Hugging Face ids and absolute paths fit; anything stranger is
# refused rather than quoted-and-hoped-for (an explicit command is the escape).
# '~' is excluded deliberately: expanding it would make the name the server
# reports differ from the name the config asked for.
_MODEL_NAME = re.compile(r"[A-Za-z0-9/][A-Za-z0-9._/+:@=-]*")


class ServeError(MasonError):
    """The model server could not be described, started, or located."""


class ServeRecord(BaseModel):
    """What a running server job says about itself.

    Unknown keys are *ignored* here, unlike everywhere else in SLAB: this file
    is written by whatever slab version is installed on the compute node, which
    may be newer than the one reading it. A record must stay readable across
    that skew — refusing it would strand a perfectly good server.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    endpoint: str
    model: str
    job_id: str = ""
    node: str = ""
    port: int = 0
    started_at: str = ""
    # Job ids are per-cluster; on centers where one filesystem is mounted
    # across several SLURM clusters, acting on a bare numeric id from the
    # wrong cluster would query — or scancel — someone else's job.
    cluster: str = ""


def mason_dir(workspace_root: str | os.PathLike[str]) -> Path:
    """Where serve scripts, logs, and the endpoint record live."""
    return Path(workspace_root) / "mason"


def record_path(workspace_root: str | os.PathLike[str]) -> Path:
    """The endpoint record's path (written by the job, read by the client)."""
    return mason_dir(workspace_root) / RECORD_NAME


def read_record(workspace_root: str | os.PathLike[str]) -> ServeRecord | None:
    """The running server's record, or None when no server has announced one.

    A record that exists but cannot be read is an error, not a None: silently
    falling back to ``localhost`` when the file says otherwise would point the
    agent at the wrong model without telling anyone.
    """
    path = record_path(workspace_root)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ServeError(
            f"{path} exists but is not readable JSON ({e}); the server job may be "
            f"mid-write — retry, or remove it with 'mason serve stop'"
        ) from e
    try:
        return ServeRecord.model_validate(raw)
    except ValidationError as e:
        raise ServeError(f"{path} is not an endpoint record: {e}") from e


def clear_record(workspace_root: str | os.PathLike[str]) -> bool:
    """Delete the endpoint record; True when there was one."""
    path = record_path(workspace_root)
    existed = path.is_file()
    path.unlink(missing_ok=True)
    return existed


def discover_endpoint(
    agent: AgentConfig, workspace_root: str | os.PathLike[str]
) -> tuple[str, str]:
    """The endpoint to use and where it came from, in precedence order.

    An explicit ``[agent] endpoint`` wins — a served record must never
    override what someone wrote down. Otherwise a running server job's record
    supplies it, and failing that the provider's own default. Only the
    OpenAI-compatible provider consults the record: an Anthropic endpoint is
    not something a compute node serves.

    Examples:
        >>> import tempfile
        >>> root = tempfile.mkdtemp()
        >>> discover_endpoint(AgentConfig(), root)
        ('http://localhost:11434/v1', 'provider default')
        >>> discover_endpoint(AgentConfig(endpoint="http://gpu-1:8000/v1"), root)
        ('http://gpu-1:8000/v1', '[agent] endpoint')
    """
    if agent.endpoint:
        return agent.endpoint, "[agent] endpoint"
    if agent.provider == "openai":
        record = read_record(workspace_root)
        if record is not None:
            where = record.node or "an unnamed node"
            job = f"job {record.job_id}" if record.job_id else "a server job"
            return record.endpoint, f"{job} on {where}"
    return agent.resolved_endpoint, "provider default"


def render_serve_script(
    agent: AgentConfig,
    hpc: HpcConfig,
    workspace_root: str | os.PathLike[str],
    *,
    partition: str | None = None,
    port: int | None = None,
    time_limit: str | None = None,
) -> str:
    """The batch script that serves ``[agent] model`` and announces its URL.

    The server runs in the batch script's own shell rather than under the
    partition's launcher: it must stay on the node whose hostname the prologue
    just wrote into the record.

    Examples:
        >>> from slab.config import AgentConfig, HpcConfig
        >>> agent = AgentConfig(model="meta-models/Muse-Glimmer-30B",
        ...                     serve={"partition": "gpu", "port": 8010,
        ...                            "tool_call_parser": "llama4_pythonic"})
        >>> hpc = HpcConfig.model_validate(
        ...     {"partitions": {"gpu": {"gres": "gpu:a100:4", "launcher": "srun"}}})
        >>> script = render_serve_script(agent, hpc, "/scratch/ws")
        >>> print(script.splitlines()[-1])
        vllm serve meta-models/Muse-Glimmer-30B --host 0.0.0.0 --port "$port" \
--enable-auto-tool-choice --tool-call-parser llama4_pythonic
    """
    serve = agent.serve
    model = _require_model(agent)
    chosen_port = port if port is not None else serve.port
    command = _server_command(agent, model)
    return render_sbatch(
        command,
        job_name=serve.job_name,
        partition=partition or serve.partition,
        config=hpc,
        time_limit=time_limit or serve.time_limit,
        output=f"{serve.job_name}-%j.out",
        prologue=[
            *_setup_omission_note(serve, hpc),
            *serve.setup,
            *_server_preflight(serve),
            # Absolute, always: the default workspace root is relative
            # (``.slab``), and the job's working directory is not the one the
            # script was rendered from — a relative record path would be
            # written somewhere the client never looks. Symlinks are left
            # alone (a site's /scratch link may mean different things on the
            # login node and the compute node).
            *_announce(
                model,
                _absolute(record_path(workspace_root)),
                chosen_port,
                cluster=hpc.cluster or "",
            ),
        ],
        use_launcher=False,
        include_global_setup=serve.include_hpc_setup,
    )


def start(
    agent: AgentConfig,
    hpc: HpcConfig,
    workspace_root: str | os.PathLike[str],
    *,
    partition: str | None = None,
    port: int | None = None,
    time_limit: str | None = None,
) -> SubmittedJob:
    """Submit the server job, refusing to run a second one behind the first.

    Two servers sharing one workspace would overwrite each other's record and
    the agent would talk to whichever wrote last. Stop the first one instead.
    """
    existing = read_record(workspace_root)
    if existing is not None:
        raise ServeError(
            f"a server is already recorded at {record_path(workspace_root)} "
            f"({existing.model} on {existing.node or 'an unnamed node'}, "
            f"job {existing.job_id or 'unknown'}); stop it with 'mason serve stop' "
            f"before starting another (two servers would overwrite each other's record)"
        )
    resolved, _spec = hpc.resolve_partition(partition or agent.serve.partition)
    script = render_serve_script(
        agent, hpc, workspace_root, partition=resolved, port=port, time_limit=time_limit
    )
    return submit(
        script,
        job_name=agent.serve.job_name,
        partition=resolved,
        directory=mason_dir(workspace_root),
    )


def stop(workspace_root: str | os.PathLike[str], *, cluster: str = "") -> str:
    """Cancel the recorded server job and drop its record; describe what happened.

    *cluster* is this config's ``[hpc] cluster``; a record announcing a
    different cluster is refused, not cancelled — job ids are per-cluster,
    and on shared filesystems mounted across clusters a bare numeric id
    from elsewhere could name someone else's job here.
    """
    from slab.hpc import cancel

    record = read_record(workspace_root)
    if record is None:
        return f"no server recorded at {record_path(workspace_root)}; nothing to stop"
    if not record.job_id:
        # Nothing would be cancelled; clearing a jobless record is safe from
        # anywhere, so this precedes the cluster check.
        clear_record(workspace_root)
        return f"record removed; it named no job id, so nothing was cancelled ({record.endpoint})"
    if record.cluster and record.cluster != cluster:
        # An unnamed local cluster cannot PROVE it is the record's, so an
        # empty *cluster* refuses too — the bypass would be exactly the
        # foreign scancel this guard exists to prevent.
        named = f"[hpc] cluster = {cluster!r}" if cluster else "no [hpc] cluster at all"
        raise ServeError(
            f"the recorded server belongs to cluster {record.cluster!r}, but this "
            f"config names {named} — job ids are per-cluster, so cancelling job "
            f"{record.job_id} from here could kill an unrelated job. Run "
            f"'mason serve stop' under a config with [hpc] cluster = "
            f"{record.cluster!r} (on that cluster) to confirm"
        )
    # Cancel first: if cancellation fails, the record must survive, or the next
    # 'serve start' would cheerfully stack a second server on a live one.
    cancel(record.job_id)
    clear_record(workspace_root)
    return f"cancel requested for job {record.job_id}; record removed ({record.endpoint})"


def probe(endpoint: str, *, timeout_s: float = 20.0) -> list[str] | None:
    """The model names the endpoint serves, or None when it does not answer."""
    from mason.client import ChatClient, LlmError

    try:
        return ChatClient(endpoint, "probe", timeout_s=timeout_s).model_names()
    except LlmError:
        return None


def wait_until_ready(
    endpoint: str,
    *,
    timeout_s: float,
    poll_s: float = _DEFAULT_POLL_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Poll until the endpoint serves models, or raise after *timeout_s*.

    Loading tens of billions of weights takes minutes, and a server that is
    still loading refuses connections exactly like one that crashed — so the
    timeout must be generous and its expiry must say what to read next.
    """
    deadline = clock() + timeout_s
    while True:
        names = probe(endpoint, timeout_s=min(poll_s, 20.0))
        if names is not None:
            return names
        remaining = deadline - clock()
        if remaining <= 0:
            raise ServeError(
                f"{endpoint} did not answer within {timeout_s:.0f}s. A large model can "
                f"take that long to load, but so can a job that never started: check "
                f"'mason serve status' and the job's output file"
            )
        sleep(min(poll_s, remaining))


def wait_for_record(
    workspace_root: str | os.PathLike[str],
    job_id: str,
    *,
    timeout_s: float,
    poll_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ServeRecord:
    """Wait for the job to announce its endpoint, failing fast if the job dies.

    A queued job announces nothing, and neither does a job that failed on the
    first module load — so the wait watches the scheduler too, and reports the
    difference instead of burning the whole timeout on a job that is already
    gone. Each pass costs one ``squeue`` call, hence the unhurried default
    interval: a queue wait can last hours, and the controller is shared.
    """
    deadline = clock() + timeout_s
    while True:
        record = read_record(workspace_root)
        if record is not None:
            return record
        status = job_state(job_id)
        if status.state.is_terminal:
            raise ServeError(
                f"job {job_id} ended as {status.state.value} without announcing an "
                f"endpoint; its output file is in {mason_dir(workspace_root)}"
            )
        remaining = deadline - clock()
        if remaining <= 0:
            raise ServeError(
                f"job {job_id} is {status.state.value} but has announced no endpoint "
                f"after {timeout_s:.0f}s; check 'mason serve status'"
            )
        sleep(min(poll_s, remaining))


def describe(
    agent: AgentConfig, workspace_root: str | os.PathLike[str], *, cluster: str = ""
) -> list[str]:
    """Human-readable lines about the recorded server: record, job, live probe."""
    record = read_record(workspace_root)
    if record is None:
        endpoint, origin = discover_endpoint(agent, workspace_root)
        return [
            f"no server recorded at {record_path(workspace_root)}",
            f"endpoint in use: {endpoint}  [{origin}]",
        ]
    lines = [
        f"endpoint: {record.endpoint}",
        f"model:    {record.model}",
        f"node:     {record.node or 'unnamed'}  port {record.port or 'unknown'}",
    ]
    if record.cluster:
        lines.append(f"cluster:  {record.cluster}")
    if record.started_at:
        lines.append(f"started:  {record.started_at}")
    foreign = bool(record.cluster and record.cluster != cluster)
    if record.job_id and foreign:
        # A job id is only meaningful on its own cluster; querying it here
        # would describe an unrelated job that happens to share the number.
        named = f"[hpc] cluster = {cluster!r}" if cluster else "no [hpc] cluster"
        lines.append(
            f"job {record.job_id}: belongs to cluster {record.cluster!r}; "
            f"not queried from here (this config names {named})"
        )
    elif record.job_id:
        try:
            status = job_state(record.job_id)
            raw = f" ({status.raw})" if status.raw else ""
            lines.append(f"job {record.job_id}: {status.state.value}{raw}")
        except SlabError as e:
            lines.append(f"job {record.job_id}: state unknown — {e}")
    names = probe(record.endpoint, timeout_s=10.0)  # a status command must stay snappy
    if names is None:
        lines.append("[x] endpoint does not answer yet (still loading, or the job is gone)")
    else:
        served = ", ".join(names) or "none"
        mark = "+" if record.model in names else "x"
        lines.append(f"[{mark}] endpoint answers; serving: {served}")
    return lines


def _absolute(path: Path) -> Path:
    """An absolute path without resolving symlinks (see the caller's note)."""
    return path if path.is_absolute() else Path.cwd() / path


def _require_model(agent: AgentConfig) -> str:
    """The model to serve, validated for a shell command it will land in."""
    if not agent.model:
        raise ServeError(
            "no model to serve: set [agent] model in the slab config "
            "(e.g. model = \"meta-models/Muse-Glimmer-30B\")"
        )
    if not _MODEL_NAME.fullmatch(agent.model):
        raise ServeError(
            f"model name {agent.model!r} is not usable in a batch script: letters, "
            f"digits, and ._/+:@=- only, and a local model needs an absolute path "
            f"(the name must reach the server unchanged, so no '~' expansion). "
            f"Give an explicit [agent.serve] command for anything stranger."
        )
    return agent.model


def _server_command(agent: AgentConfig, model: str) -> str:
    """The vLLM invocation, or the configured command verbatim."""
    serve = agent.serve
    if serve.command:
        if "$port" not in serve.command and "${port}" not in serve.command:
            raise ServeError(
                "[agent.serve] command never references \"$port\", but the job "
                "announces http://<node>:<port>/v1 built from [agent.serve] "
                "port — a server bound elsewhere would leave a record no "
                "process answers. Bind the script's $port variable in the "
                "command (it is set from [agent.serve] port)"
            )
        return serve.command
    if agent.tool_protocol == "native" and not serve.tool_call_parser:
        raise ServeError(
            "[agent.serve] needs a tool_call_parser: vLLM only emits native tool "
            "calls when started with --enable-auto-tool-choice --tool-call-parser "
            "NAME, and the right NAME depends on the model ('vllm serve --help' "
            "lists what your build registers). Set [agent.serve] tool_call_parser, "
            "or set [agent] tool_protocol = \"fenced\" to use the text protocol "
            "instead, or give an explicit [agent.serve] command."
        )
    pieces = ["vllm", "serve", shlex.quote(model), "--host", "0.0.0.0", "--port", '"$port"']
    if serve.tool_call_parser:
        pieces += ["--enable-auto-tool-choice", "--tool-call-parser"]
        pieces.append(shlex.quote(serve.tool_call_parser))
    pieces.extend(serve.args)
    return " ".join(pieces)


def _setup_omission_note(serve: ServeConfig, hpc: HpcConfig) -> list[str]:
    """A rendered comment naming what the serve script deliberately skipped.

    The script is the readable artifact ("render it, read it, then submit
    it"), so a config whose ``[hpc] setup`` exists but is excluded says so
    IN the script — the same convention as the omitted-launcher comment —
    instead of silently differing from every other job's rendering.
    """
    if serve.include_hpc_setup or not hpc.setup:
        return []
    return [
        "# [hpc]-level setup omitted: serve jobs bring their own environment",
        "# (engine module stacks fight the server's venv); set [agent.serve]",
        "# include_hpc_setup = true to include it. The partition's setup applies.",
    ]


def _server_preflight(serve: ServeConfig) -> list[str]:
    """A loud existence check for the default server binary, after setup runs.

    Without it the failure is bash's own ``vllm: command not found`` — true,
    but mute about the cause and printed only after a queue wait. The check
    runs before the endpoint record is written, so a doomed job never
    announces itself. An explicit ``[agent.serve] command`` is the
    maintainer's own recipe and gets no check — its payload is opaque here.
    """
    if serve.command:
        return []
    return [
        "command -v vllm >/dev/null 2>&1 || { "
        "echo \"mason serve: 'vllm' not found after setup — do the "
        '[agent.serve] setup lines activate the vLLM venv?" >&2; exit 1; }',
    ]


def _announce(model: str, record: Path, port: int, *, cluster: str = "") -> Iterable[str]:
    """Shell that writes this node's endpoint down, and cleans up after itself."""
    return [
        "# --- mason serve: announce this node's endpoint, then run the server ---",
        f'port={port}',
        f'model={shlex.quote(model)}',
        f'cluster={shlex.quote(cluster)}',
        'host="$(hostname -f)"',
        f"record={shlex.quote(str(record))}",
        'mkdir -p "$(dirname "$record")"',
        'cat > "$record" <<SLAB_ENDPOINT_RECORD',
        '{"endpoint": "http://$host:$port/v1", "model": "$model",'
        ' "job_id": "${SLURM_JOB_ID:-}", "node": "$host", "port": $port,'
        ' "cluster": "$cluster",'
        ' "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}',
        "SLAB_ENDPOINT_RECORD",
        # A record outliving its server would point the agent at a dead node.
        "trap 'rm -f \"$record\"' EXIT",
        'echo "mason serve: http://$host:$port/v1"',
        "",
    ]
