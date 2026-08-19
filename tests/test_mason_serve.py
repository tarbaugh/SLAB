"""Serving the agent's model on a cluster node: render, announce, discover.

The rendered script is executed for real by ``bash`` here (with a stub server
on PATH): the record it writes is the only coupling between the login node and
the compute node, so "the shell we generate actually produces a readable
record, and clears it on exit" is the load-bearing claim to test rather than
assert about.
"""

import json
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import LlmScript
from slab.cli import app
from slab.config import AgentConfig, HpcConfig, load_config
from slab.errors import SlabError
from slab.mason import MasonSession
from slab.mason.serve import (
    ServeError,
    ServeRecord,
    clear_record,
    describe,
    discover_endpoint,
    probe,
    read_record,
    record_path,
    render_serve_script,
    start,
    stop,
    wait_for_record,
    wait_until_ready,
)

runner = CliRunner()

HPC = HpcConfig.model_validate(
    {
        "cluster": "testcluster",
        "account": "abc-123",
        "default_partition": "cpu",
        "setup": ["module load slab"],
        "partitions": {
            "cpu": {"time_limit": "04:00:00", "launcher": "srun"},
            "gpu": {
                "time_limit": "08:00:00",
                "gres": "gpu:a100:4",
                "launcher": "srun",
                "setup": ["module load cuda/12.4"],
            },
        },
    }
)

GLIMMER = "meta-models/Muse-Glimmer-30B"


def _agent(**serve: Any) -> AgentConfig:
    serve.setdefault("partition", "gpu")
    serve.setdefault("tool_call_parser", "llama4_pythonic")
    return AgentConfig(model=GLIMMER, serve=serve)


def _fake(bin_dir: Path, name: str, body: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / name
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@pytest.fixture()
def scheduler_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "fake-slurm"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return bin_dir


def _write_record(root: Path, endpoint: str, **extra: Any) -> Path:
    path = record_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"endpoint": endpoint, "model": GLIMMER, "job_id": "4242", "node": "gpu-07",
               "port": 8000, "started_at": "2026-08-17T09:00:00Z", **extra}
    path.write_text(json.dumps(payload))
    return path


# -- rendering ---------------------------------------------------------------


def test_render_serves_the_configured_model_with_vllm(tmp_path: Path) -> None:
    script = render_serve_script(_agent(), HPC, tmp_path)
    assert script.splitlines()[-1] == (
        f"vllm serve {GLIMMER} --host 0.0.0.0 --port \"$port\" "
        "--enable-auto-tool-choice --tool-call-parser llama4_pythonic"
    )
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:a100:4" in script
    assert "#SBATCH --time=08:00:00" in script
    assert "#SBATCH --job-name=mason-serve" in script
    assert "#SBATCH --output=mason-serve-%j.out" in script


def test_render_does_not_put_the_server_under_the_launcher(tmp_path: Path) -> None:
    # srun would move the server off the node whose hostname the prologue
    # just recorded (and can hang waiting for a step allocation).
    script = render_serve_script(_agent(), HPC, tmp_path)
    assert "srun" not in script


def test_render_orders_setup_hpc_then_partition_then_serve(tmp_path: Path) -> None:
    """With include_hpc_setup opted in, the layering matches every other job:
    hpc-level, then the partition's, then the serve job's own lines."""
    script = render_serve_script(
        _agent(setup=["source ~/venvs/vllm/bin/activate"], include_hpc_setup=True),
        HPC,
        tmp_path,
    )
    body = script.splitlines()
    assert body.index("module load slab") < body.index("module load cuda/12.4")
    assert body.index("module load cuda/12.4") < body.index("source ~/venvs/vllm/bin/activate")
    assert body.index("source ~/venvs/vllm/bin/activate") < body.index("port=8000")


def test_render_points_the_record_at_this_workspace(tmp_path: Path) -> None:
    script = render_serve_script(_agent(), HPC, tmp_path)
    assert f"record={record_path(tmp_path)}" in script


def test_a_relative_workspace_still_records_absolutely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default workspace root is relative; the job's cwd is not ours.

    A relative ``record=.slab/mason/endpoint.json`` in the script would be
    written under the job's own working directory, where the client never
    looks — the server would come up and appear to have announced nothing.
    """
    monkeypatch.chdir(tmp_path)
    script = render_serve_script(_agent(), HPC, Path(".slab"))
    assert f"record={tmp_path / '.slab' / 'mason' / 'endpoint.json'}" in script


def test_render_honors_port_and_time_overrides(tmp_path: Path) -> None:
    script = render_serve_script(
        _agent(port=8123), HPC, tmp_path, port=9001, time_limit="02:00:00"
    )
    assert "port=9001" in script
    assert '"port": $port' in script
    assert "#SBATCH --time=02:00:00" in script


def test_render_extra_args_reach_the_command(tmp_path: Path) -> None:
    script = render_serve_script(
        _agent(args=["--tensor-parallel-size 4", "--max-model-len 131072"]), HPC, tmp_path
    )
    assert script.endswith("--tensor-parallel-size 4 --max-model-len 131072")


def test_explicit_command_replaces_the_vllm_default(tmp_path: Path) -> None:
    script = render_serve_script(
        _agent(command='sglang launch --port "$port"', tool_call_parser=None), HPC, tmp_path
    )
    assert script.splitlines()[-1] == 'sglang launch --port "$port"'
    assert "vllm" not in script


# -- refusals ----------------------------------------------------------------


def test_native_tool_calls_need_a_parser_and_the_refusal_teaches(tmp_path: Path) -> None:
    agent = AgentConfig(model=GLIMMER, serve={"partition": "gpu"})
    with pytest.raises(ServeError) as excinfo:
        render_serve_script(agent, HPC, tmp_path)
    message = str(excinfo.value)
    assert "tool_call_parser" in message
    assert "--enable-auto-tool-choice" in message
    assert 'tool_protocol = "fenced"' in message  # the two ways out are both named
    assert "[agent.serve] command" in message


def test_fenced_protocol_needs_no_parser(tmp_path: Path) -> None:
    agent = AgentConfig(model=GLIMMER, tool_protocol="fenced", serve={"partition": "gpu"})
    script = render_serve_script(agent, HPC, tmp_path)
    assert "--tool-call-parser" not in script
    assert script.splitlines()[-1].startswith(f"vllm serve {GLIMMER}")


def test_no_model_refuses_with_the_key_to_set(tmp_path: Path) -> None:
    with pytest.raises(ServeError, match=r"\[agent\] model"):
        render_serve_script(AgentConfig(serve={"partition": "gpu"}), HPC, tmp_path)


@pytest.mark.parametrize("name", ['x"; rm -rf ~; #', "$(id)", "~/models/glimmer", "a b"])
def test_a_shell_hostile_model_name_is_refused_not_quoted(name: str, tmp_path: Path) -> None:
    # The name lands in a shell command, a JSON heredoc, and doctor's
    # served-model check; refusing beats hoping the quoting held, and '~' is
    # refused because expanding it would change the name the server reports.
    agent = AgentConfig(model=name, serve={"partition": "gpu", "tool_call_parser": "hermes"})
    with pytest.raises(ServeError, match="not usable in a batch script"):
        render_serve_script(agent, HPC, tmp_path)


def test_an_absolute_local_model_path_serves(tmp_path: Path) -> None:
    agent = AgentConfig(
        model="/shared/models/Muse-Glimmer-30B",
        serve={"partition": "gpu", "tool_call_parser": "hermes"},
    )
    script = render_serve_script(agent, HPC, tmp_path)
    assert "vllm serve /shared/models/Muse-Glimmer-30B" in script


def test_a_dangling_serve_partition_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="not declared in"):
        render_serve_script(_agent(partition="quantum"), HPC, tmp_path)


# -- the rendered script, actually executed ----------------------------------


def test_the_rendered_script_announces_a_readable_record_and_clears_it(
    tmp_path: Path,
) -> None:
    """Run the real script with a stub server: record in, record out."""
    bin_dir = tmp_path / "bin"
    _fake(bin_dir, "vllm", 'echo "vllm $*" > "$0.args"')
    _fake(bin_dir, "module", "exit 0")  # the setup lines run too, on this laptop
    root = tmp_path / "ws"
    script = render_serve_script(_agent(port=8055), HPC, root)
    body = "\n".join(
        line for line in script.splitlines() if not line.startswith("#SBATCH")
    )
    runner_script = tmp_path / "serve.sh"
    # The stub exits immediately, so the record must be readable *before* the
    # server ends: read it from inside the script, in the server's place.
    body = body.replace('vllm serve', 'cp "$record" "$record.snapshot"\nvllm serve')
    runner_script.write_text(body)

    result = subprocess.run(
        ["bash", str(runner_script)],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "SLURM_JOB_ID": "998877"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    snapshot = json.loads((record_path(root).with_suffix(".json.snapshot")).read_text())
    record = ServeRecord.model_validate(snapshot)
    assert record.model == GLIMMER
    assert record.job_id == "998877"
    assert record.port == 8055
    assert record.node and record.endpoint == f"http://{record.node}:8055/v1"
    assert record.started_at.endswith("Z")
    assert (bin_dir / "vllm.args").read_text().startswith(f"vllm serve {GLIMMER}")
    # The trap fires on exit: a record must never outlive its server.
    assert not record_path(root).exists()


def test_the_rendered_script_is_valid_shell(tmp_path: Path) -> None:
    script = render_serve_script(
        _agent(args=["--tensor-parallel-size 4"], setup=["export FOO=1"]), HPC, tmp_path
    )
    check = subprocess.run(
        ["bash", "-n"], input=script, capture_output=True, text=True, check=False
    )
    assert check.returncode == 0, check.stderr


# -- the record --------------------------------------------------------------


def test_no_record_reads_as_none(tmp_path: Path) -> None:
    assert read_record(tmp_path) is None


def test_unknown_record_keys_are_ignored_for_version_skew(tmp_path: Path) -> None:
    _write_record(tmp_path, "http://gpu-07:8000/v1", future_field="from a newer slab")
    record = read_record(tmp_path)
    assert record is not None and record.node == "gpu-07"


def test_unreadable_record_refuses_instead_of_falling_back(tmp_path: Path) -> None:
    path = record_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"endpoint": "http://gpu-07:80')  # a half-written file
    with pytest.raises(ServeError, match="not readable JSON"):
        read_record(tmp_path)


def test_a_record_missing_its_endpoint_is_not_a_record(tmp_path: Path) -> None:
    path = record_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "x"}')
    with pytest.raises(ServeError, match="not an endpoint record"):
        read_record(tmp_path)


def test_clear_record_reports_whether_there_was_one(tmp_path: Path) -> None:
    assert clear_record(tmp_path) is False
    _write_record(tmp_path, "http://gpu-07:8000/v1")
    assert clear_record(tmp_path) is True
    assert read_record(tmp_path) is None


# -- discovery ---------------------------------------------------------------


def test_discovery_falls_back_to_the_provider_default(tmp_path: Path) -> None:
    endpoint, origin = discover_endpoint(AgentConfig(model="x"), tmp_path)
    assert endpoint == "http://localhost:11434/v1"
    assert origin == "provider default"


def test_a_running_server_supplies_the_endpoint(tmp_path: Path) -> None:
    _write_record(tmp_path, "http://gpu-07:8000/v1")
    endpoint, origin = discover_endpoint(AgentConfig(model=GLIMMER), tmp_path)
    assert endpoint == "http://gpu-07:8000/v1"
    assert origin == "job 4242 on gpu-07"


def test_configured_endpoint_outranks_a_running_server(tmp_path: Path) -> None:
    _write_record(tmp_path, "http://gpu-07:8000/v1")
    endpoint, origin = discover_endpoint(
        AgentConfig(model=GLIMMER, endpoint="http://written-down:8000/v1"), tmp_path
    )
    assert endpoint == "http://written-down:8000/v1"
    assert origin == "[agent] endpoint"


def test_the_anthropic_provider_ignores_a_served_record(tmp_path: Path) -> None:
    _write_record(tmp_path, "http://gpu-07:8000/v1")
    endpoint, _origin = discover_endpoint(
        AgentConfig(provider="anthropic", model="claude-opus-5"), tmp_path
    )
    assert endpoint == "https://api.anthropic.com/v1"


def test_session_talks_to_the_discovered_endpoint(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _write_record(root, "http://gpu-07:8000/v1")
    (tmp_path / "slab.toml").write_text(f'[agent]\nmodel = "{GLIMMER}"\n')
    session = MasonSession(tmp_path, workspace_root=root)
    assert session.endpoint == "http://gpu-07:8000/v1"
    assert session.endpoint_origin == "job 4242 on gpu-07"
    assert session.agent.resolved_endpoint == "http://gpu-07:8000/v1"


def test_an_endpoint_override_outranks_discovery(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _write_record(root, "http://gpu-07:8000/v1")
    session = MasonSession(tmp_path, workspace_root=root)
    session.resolve_endpoint("http://laptop:11434/v1")
    assert session.endpoint == "http://laptop:11434/v1"
    assert session.endpoint_origin == "--endpoint"
    assert session.agent.resolved_endpoint == "http://laptop:11434/v1"


# -- start / stop ------------------------------------------------------------


def test_start_submits_and_keeps_the_script_in_the_workspace(
    tmp_path: Path, scheduler_bin: Path
) -> None:
    _fake(scheduler_bin, "sbatch", 'echo "515151;testcluster"')
    root = tmp_path / "ws"
    job = start(_agent(), HPC, root)
    assert job.job_id == "515151"
    assert job.partition == "gpu"
    kept = Path(job.script_path)
    assert kept.parent == root / "mason"
    assert kept.name == "mason-serve-515151.sbatch"
    assert "vllm serve" in kept.read_text()


def test_start_refuses_to_stack_a_second_server_on_the_record(
    tmp_path: Path, scheduler_bin: Path
) -> None:
    _fake(scheduler_bin, "sbatch", 'echo "1"')
    root = tmp_path / "ws"
    _write_record(root, "http://gpu-07:8000/v1")
    with pytest.raises(ServeError, match="already recorded"):
        start(_agent(), HPC, root)


def test_stop_cancels_the_recorded_job_and_clears_the_record(
    tmp_path: Path, scheduler_bin: Path
) -> None:
    marker = tmp_path / "cancelled"
    _fake(scheduler_bin, "scancel", f'echo "$@" > "{marker}"')
    _write_record(tmp_path, "http://gpu-07:8000/v1")
    message = stop(tmp_path)
    assert "4242" in message
    assert "4242" in marker.read_text()
    assert read_record(tmp_path) is None


def test_stop_without_a_record_says_so_instead_of_failing(tmp_path: Path) -> None:
    assert "nothing to stop" in stop(tmp_path)


def test_a_failed_cancel_keeps_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # If we cannot cancel, the record must survive: 'serve start' refuses while
    # it exists, and that refusal is what stops a second server stacking on a
    # live one.
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # no scancel
    _write_record(tmp_path, "http://gpu-07:8000/v1")
    with pytest.raises(SlabError):
        stop(tmp_path)
    assert read_record(tmp_path) is not None


def test_stop_of_a_record_without_a_job_id_still_clears_it(tmp_path: Path) -> None:
    _write_record(tmp_path, "http://gpu-07:8000/v1", job_id="")
    message = stop(tmp_path)
    assert "nothing was cancelled" in message
    assert read_record(tmp_path) is None


# -- waiting -----------------------------------------------------------------


def test_wait_for_record_returns_it_once_the_job_writes_it(
    tmp_path: Path, scheduler_bin: Path
) -> None:
    _fake(scheduler_bin, "squeue", 'echo "RUNNING"')
    writes: list[float] = []

    def sleep(seconds: float) -> None:
        writes.append(seconds)
        if len(writes) == 2:
            _write_record(tmp_path, "http://gpu-07:8000/v1")

    record = wait_for_record(tmp_path, "4242", timeout_s=60.0, poll_s=5.0, sleep=sleep)
    assert record.endpoint == "http://gpu-07:8000/v1"
    assert writes == [5.0, 5.0]


def test_wait_for_record_fails_fast_when_the_job_died(
    tmp_path: Path, scheduler_bin: Path
) -> None:
    _fake(scheduler_bin, "squeue", "exit 1")
    _fake(scheduler_bin, "sacct", 'echo "FAILED|1:0"')
    slept: list[float] = []
    with pytest.raises(ServeError, match="ended as failed without announcing"):
        wait_for_record(
            tmp_path, "4242", timeout_s=3600.0, sleep=lambda s: slept.append(s)
        )
    assert slept == []  # the whole point: no waiting on a dead job


def test_wait_for_record_times_out_naming_the_job_state(
    tmp_path: Path, scheduler_bin: Path
) -> None:
    _fake(scheduler_bin, "squeue", 'echo "PENDING"')
    clock = iter([0.0, 0.0, 100.0, 100.0])
    with pytest.raises(ServeError, match="is pending but has announced no endpoint"):
        wait_for_record(
            tmp_path, "4242", timeout_s=10.0, sleep=lambda s: None, clock=lambda: next(clock)
        )


def test_wait_until_ready_returns_the_served_names(
    llm_server: tuple[str, LlmScript],
) -> None:
    endpoint, script = llm_server
    script.get_response = (200, {"data": [{"id": GLIMMER}]})
    assert wait_until_ready(endpoint, timeout_s=30.0) == [GLIMMER]


def test_wait_until_ready_times_out_with_where_to_look(tmp_path: Path) -> None:
    clock = iter([0.0, 100.0])
    with pytest.raises(ServeError, match="did not answer within"):
        wait_until_ready(
            "http://127.0.0.1:1/v1",
            timeout_s=1.0,
            poll_s=0.01,
            sleep=lambda s: None,
            clock=lambda: next(clock),
        )


def test_wait_until_ready_keeps_polling_while_the_model_loads(
    llm_server: tuple[str, LlmScript],
) -> None:
    endpoint, script = llm_server
    script.get_response = (503, {"error": "still loading"})
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        script.get_response = (200, {"data": [{"id": GLIMMER}]})

    assert wait_until_ready(endpoint, timeout_s=60.0, poll_s=3.0, sleep=sleep) == [GLIMMER]
    assert slept == [3.0]


def test_probe_reports_none_for_a_dead_endpoint() -> None:
    assert probe("http://127.0.0.1:1/v1", timeout_s=1.0) is None


# -- describe ----------------------------------------------------------------


def test_describe_without_a_record_names_the_endpoint_in_use(tmp_path: Path) -> None:
    lines = describe(AgentConfig(model=GLIMMER), tmp_path)
    assert "no server recorded" in lines[0]
    assert "provider default" in lines[1]


def test_describe_reports_job_state_and_a_live_probe(
    tmp_path: Path, scheduler_bin: Path, llm_server: tuple[str, LlmScript]
) -> None:
    endpoint, script = llm_server
    script.get_response = (200, {"data": [{"id": GLIMMER}]})
    _fake(scheduler_bin, "squeue", 'echo "RUNNING"')
    _write_record(tmp_path, endpoint)
    text = "\n".join(describe(AgentConfig(model=GLIMMER), tmp_path))
    assert "job 4242: running" in text
    assert f"[+] endpoint answers; serving: {GLIMMER}" in text


def test_describe_says_state_unknown_off_a_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # no squeue anywhere
    _write_record(tmp_path, "http://127.0.0.1:1/v1")
    text = "\n".join(describe(AgentConfig(model=GLIMMER), tmp_path))
    assert "state unknown" in text
    assert "not on PATH" in text


def test_describe_flags_a_record_whose_endpoint_is_silent(
    tmp_path: Path, scheduler_bin: Path
) -> None:
    _fake(scheduler_bin, "squeue", 'echo "PENDING"')
    _write_record(tmp_path, "http://127.0.0.1:1/v1")
    text = "\n".join(describe(AgentConfig(model=GLIMMER), tmp_path))
    assert "does not answer yet" in text


# -- CLI ---------------------------------------------------------------------


def _project(tmp_path: Path, extra: str = "") -> None:
    (tmp_path / "slab.toml").write_text(
        f"""\
[hpc]
cluster = "testcluster"
default_partition = "gpu"

[hpc.partitions.gpu]
time_limit = "08:00:00"
gres = "gpu:a100:4"

[agent]
model = "{GLIMMER}"

[agent.serve]
partition = "gpu"
tool_call_parser = "llama4_pythonic"
{extra}
"""
    )


def test_cli_serve_render_prints_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    result = runner.invoke(app, ["mason", "serve", "render", "-w", str(tmp_path / ".slab")])
    assert result.exit_code == 0, result.output
    assert "#SBATCH --gres=gpu:a100:4" in result.output
    assert f"vllm serve {GLIMMER}" in result.output


def test_cli_serve_render_reports_a_missing_parser_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        f'[hpc.partitions.gpu]\n\n[agent]\nmodel = "{GLIMMER}"\n'
        '\n[agent.serve]\npartition = "gpu"\n'
    )
    result = runner.invoke(app, ["mason", "serve", "render"])
    assert result.exit_code == 1
    assert "tool_call_parser" in result.output


def test_cli_serve_start_reports_the_job_and_how_to_watch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduler_bin: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    _fake(scheduler_bin, "sbatch", 'echo "606060"')
    result = runner.invoke(app, ["mason", "serve", "start", "-w", str(tmp_path / ".slab")])
    assert result.exit_code == 0, result.output
    assert "submitted job 606060" in result.output
    assert "slab mason serve status" in result.output


def test_cli_serve_start_wait_follows_the_job_to_a_live_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduler_bin: Path,
    llm_server: tuple[str, LlmScript],
) -> None:
    """--wait: submit, wait for the record, then wait for the model to load."""
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    endpoint, script = llm_server
    script.get_response = (200, {"data": [{"id": GLIMMER}]})
    root = tmp_path / ".slab"
    # A fake sbatch that does what a real serve job does: announce itself.
    record = record_path(root)
    _fake(
        scheduler_bin,
        "sbatch",
        f'mkdir -p "$(dirname {record})" && cat > {record} <<EOF\n'
        f'{{"endpoint": "{endpoint}", "model": "{GLIMMER}", "job_id": "717171",'
        f' "node": "gpu-07", "port": 8000}}\nEOF\necho 717171',
    )
    _fake(scheduler_bin, "squeue", 'echo "RUNNING"')
    result = runner.invoke(
        app, ["mason", "serve", "start", "--wait", "-w", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "gpu-07 announced" in result.output
    assert f"[+] {endpoint} answers; serving: {GLIMMER}" in result.output


def test_cli_serve_status_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduler_bin: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    root = tmp_path / ".slab"
    _write_record(root, "http://127.0.0.1:1/v1")
    _fake(scheduler_bin, "squeue", 'echo "RUNNING"')
    _fake(scheduler_bin, "scancel", "true")

    status = runner.invoke(app, ["mason", "serve", "status", "-w", str(root)])
    assert status.exit_code == 0, status.output
    assert "job 4242: running" in status.output
    assert "gpu-07" in status.output

    stopped = runner.invoke(app, ["mason", "serve", "stop", "-w", str(root)])
    assert stopped.exit_code == 0, stopped.output
    assert "cancel requested for job 4242" in stopped.output
    assert read_record(root) is None


def test_cli_doctor_names_where_the_endpoint_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm_server: tuple[str, LlmScript]
) -> None:
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    endpoint, script = llm_server
    script.get_response = (200, {"data": [{"id": GLIMMER}]})
    script.responses = [
        (
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {"name": "ping", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )
    ]
    root = tmp_path / ".slab"
    _write_record(root, endpoint)
    result = runner.invoke(app, ["mason", "doctor", "-w", str(root)])
    assert result.exit_code == 0, result.output
    assert "[job 4242 on gpu-07]" in result.output
    assert "native tool calls work" in result.output


def test_cli_doctor_points_at_serve_when_nothing_is_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A configured endpoint that no longer answers is usually last
    # allocation's node — the fix is a new serve job, so say so.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        f'[agent]\nmodel = "{GLIMMER}"\nendpoint = "http://127.0.0.1:1/v1"\n'
    )
    result = runner.invoke(app, ["mason", "doctor", "-w", str(tmp_path)])
    assert result.exit_code == 1
    assert "[[agent] endpoint]" in result.output
    assert "slab mason serve start" in result.output


def test_cli_doctor_does_not_nudge_when_an_endpoint_was_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Someone who typed --endpoint meant that server; 'start your own' is noise.
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    result = runner.invoke(
        app, ["mason", "doctor", "--endpoint", "http://127.0.0.1:1/v1", "-w", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "[--endpoint]" in result.output
    assert "slab mason serve start" not in result.output


def test_cli_doctor_reports_a_stale_record_as_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduler_bin: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    root = tmp_path / ".slab"
    _write_record(root, "http://127.0.0.1:1/v1")
    _fake(scheduler_bin, "squeue", "exit 1")
    _fake(scheduler_bin, "sacct", 'echo "COMPLETED|0:0"')
    result = runner.invoke(app, ["mason", "doctor", "-w", str(root)])
    assert result.exit_code == 1
    assert "ended as completed" in result.output
    assert "serve stop" in result.output


def test_a_goal_runs_against_an_endpoint_nobody_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm_server: tuple[str, LlmScript]
) -> None:
    """End to end: record -> discovery -> real HTTP client -> finished goal.

    Nothing here sets [agent] endpoint. The only reason Mason reaches the
    server is the record a serve job left behind — which is exactly the path a
    cluster session takes.
    """
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    endpoint, script = llm_server
    script.get_response = (200, {"data": [{"id": GLIMMER}]})
    report = "a0 = 3.615 A, verified (run 01abc)"
    script.responses = [
        (
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "finish",
                                        "arguments": json.dumps({"report": report}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 900, "completion_tokens": 40},
            },
        )
    ]
    root = tmp_path / ".slab"
    _write_record(root, endpoint)

    result = runner.invoke(app, ["mason", "run", "measure a0", "--auto", "-w", str(root)])
    assert result.exit_code == 0, result.output
    assert report in result.output
    # The request really went to the served endpoint, with the served model name.
    assert script.requests and script.requests[0]["model"] == GLIMMER


# -- config ------------------------------------------------------------------


def test_serve_section_loads_from_a_project_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, extra='port = 8123\nargs = ["--tensor-parallel-size 4"]\n')
    serve = load_config(tmp_path).agent.serve
    assert serve.port == 8123
    assert serve.args == ("--tensor-parallel-size 4",)
    assert serve.tool_call_parser == "llama4_pythonic"


def test_a_typo_in_the_serve_section_is_refused_with_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text("[agent.serve]\nparser = \"hermes\"\n")
    with pytest.raises(Exception, match="unknown key"):
        load_config(tmp_path)


def test_serve_variables_are_not_expanded_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [agent.serve] is compute-node shell: $SCRATCH must reach the node
    # verbatim, not be resolved on the login node (or refused as unset).
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        '[agent.serve]\nsetup = ["source $SLAB_NO_SUCH_VAR/bin/activate"]\n'
    )
    serve = load_config(tmp_path).agent.serve
    assert serve.setup == ("source $SLAB_NO_SUCH_VAR/bin/activate",)


# -- the vLLM preflight: fail loudly before announcing an endpoint ----------------------


def test_serve_script_checks_vllm_exists_before_announcing(tmp_path: Path) -> None:
    """A venv-less setup must die with a pointed message, not bash's mute
    'command not found' — and before the endpoint record is written, so a
    doomed job never announces itself."""
    script = render_serve_script(_agent(), HPC, tmp_path)
    guard = script.index("command -v vllm")
    assert "[agent.serve] setup lines activate the vLLM venv" in script
    assert guard < script.index("endpoint.json")
    assert guard < script.index("vllm serve")


def test_explicit_serve_command_gets_no_vllm_preflight(tmp_path: Path) -> None:
    """A maintainer's own command is opaque here: no check to get wrong."""
    script = render_serve_script(
        _agent(command='my-server --port "$port"'), HPC, tmp_path
    )
    assert "command -v vllm" not in script


# -- serve-job isolation: setup scope, port contract, cluster identity ------------------


def test_serve_job_excludes_hpc_global_setup_by_default(tmp_path: Path) -> None:
    """[hpc] setup exists to load ENGINE software; those module stacks fight
    the server's venv. The partition's own setup (GPU drivers) still applies."""
    hpc = HpcConfig.model_validate(
        {
            "setup": ["module load quantum-espresso/7.4"],
            "partitions": {"gpu": {"gres": "gpu:a100:4", "setup": ["module load cuda/12.4"]}},
        }
    )
    script = render_serve_script(_agent(), hpc, tmp_path)
    assert "quantum-espresso" not in script
    assert "module load cuda/12.4" in script
    opted_in = render_serve_script(_agent(include_hpc_setup=True), hpc, tmp_path)
    assert "module load quantum-espresso/7.4" in opted_in


def test_custom_serve_command_must_bind_the_announced_port(tmp_path: Path) -> None:
    """The record announces http://<node>:<port>/v1; a custom command that
    never references $port could leave a record no process answers."""
    with pytest.raises(ServeError, match=r"\$port"):
        render_serve_script(_agent(command="my-server --port 9000"), HPC, tmp_path)
    script = render_serve_script(
        _agent(command='my-server --port "$port"'), HPC, tmp_path
    )
    assert 'my-server --port "$port"' in script


def test_record_carries_cluster_and_stop_refuses_foreign_records(
    tmp_path: Path,
) -> None:
    """Job ids are per-cluster; on filesystems mounted across clusters,
    scancel on a foreign record's numeric id could kill someone else's job."""
    from slab.mason.serve import ServeRecord, record_path, stop

    hpc = HpcConfig.model_validate(
        {"cluster": "delta", "partitions": {"gpu": {"gres": "gpu:a100:4"}}}
    )
    script = render_serve_script(_agent(), hpc, tmp_path)
    assert '"cluster": "$cluster"' in script
    assert "cluster=delta" in script

    record = record_path(tmp_path)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        ServeRecord(
            endpoint="http://gpu-1:8000/v1",
            model=GLIMMER,
            job_id="424242",
            cluster="delta",
        ).model_dump_json()
    )
    with pytest.raises(ServeError, match="delta"):
        stop(tmp_path, cluster="omega")
    assert record.is_file()  # the record survives a refused stop


def test_describe_does_not_query_foreign_job_ids(tmp_path: Path) -> None:
    from slab.mason.serve import ServeRecord, describe, record_path

    record = record_path(tmp_path)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        ServeRecord(
            endpoint="http://127.0.0.1:1/v1",  # answers nothing, quickly
            model=GLIMMER,
            job_id="424242",
            cluster="delta",
        ).model_dump_json()
    )
    lines = describe(_agent(), tmp_path, cluster="omega")
    joined = "\n".join(lines)
    assert "belongs to cluster 'delta'" in joined
    assert "cluster:  delta" in joined
