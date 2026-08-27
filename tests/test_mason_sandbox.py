"""The sandbox: render is derivation, the bridge is real sockets, verify fails closed."""

import http.server
import json
import shutil
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mason.cli import app
from mason.config import AgentConfig
from mason.sandbox import (
    SandboxError,
    default_binds,
    forward,
    preflight,
    render_sandbox_script,
    sandbox_toml,
    verify,
)
from slab.config import HpcConfig, SlabConfig

runner = CliRunner()

_HPC = HpcConfig.model_validate(
    {"default_partition": "cpu", "partitions": {"cpu": {"time_limit": "04:00:00"}}}
)


def _slab_cfg(**tables: object) -> SlabConfig:
    return SlabConfig.model_validate(tables)


def _agent(**extra: object) -> AgentConfig:
    return AgentConfig.model_validate(
        {"model": "test-model", "sandbox": {"image": "/containers/slab.sif"}, **extra}
    )


def _render(tmp_path: Path, agent: AgentConfig, slab_cfg: SlabConfig) -> tuple[str, list[str]]:
    return render_sandbox_script(
        agent,
        _HPC,
        slab_cfg,
        tmp_path / "ws",
        tmp_path / "project",
        "relax Cu and report",
        toml_path=tmp_path / "sandbox" / "slab.toml",
    )


# -- rendering ----------------------------------------------------------------


def test_render_isolates_and_fails_closed(tmp_path: Path) -> None:
    script, _ = _render(tmp_path, _agent(), _slab_cfg())
    assert "--net --network none" in script
    assert "--containall --no-home --cleanenv" in script
    assert "mason sandbox verify" in script  # either proof failing aborts the job
    assert "mason run --auto" in script
    assert "'relax Cu and report'" in script
    assert 'sandbox bridge "$BRIDGE" "$UPSTREAM"' in script
    assert "socat" not in script  # both bridge halves are mason's own plumbing
    # A missing image aborts with a plain message, not an apptainer FATAL.
    assert 'no container image at $IMAGE' in script
    # The scheduler header comes from [hpc]; the payload never uses srun.
    assert "#SBATCH --partition=cpu" in script


def test_render_is_valid_bash(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    script, _ = _render(tmp_path, _agent(), _slab_cfg())
    check = subprocess.run(
        [bash, "-n", "/dev/stdin"], input=script, capture_output=True, text=True
    )
    assert check.returncode == 0, check.stderr


def test_binds_derive_from_the_config(tmp_path: Path) -> None:
    cfg = _slab_cfg(
        paths={"pseudos": "/shared/pseudos", "scratch": "/scratch/me"},
        engines={"rootstock": {"root": "/shared/rootstock"}},
    )
    binds, warnings = default_binds(tmp_path / "p", tmp_path / "ws", cfg)
    joined = "\n".join(binds)
    assert f"{tmp_path / 'p'}:{tmp_path / 'p'}:rw" in binds
    assert "/shared/pseudos:/shared/pseudos:ro" in binds
    assert "/scratch/me:/scratch/me:rw" in binds
    assert "/shared/rootstock:/shared/rootstock:ro" in binds
    assert ":ro" in joined  # the python environment rides along read-only
    assert warnings == []


def test_extra_binds_and_the_gaps_warned(tmp_path: Path) -> None:
    cfg = _slab_cfg(engines={"rootstock": {"cluster": "delta"}})
    agent = _agent(sandbox={"image": "/i.sif", "binds": ["/opt/qe:/opt/qe:ro"]})
    script, warnings = _render(tmp_path, agent, cfg)
    assert "--bind /opt/qe:/opt/qe:ro" in script
    assert any("cluster form" in w for w in warnings)
    assert any("no pseudopotentials will be visible" in w for w in warnings)


def test_render_requires_an_image(tmp_path: Path) -> None:
    agent = AgentConfig(model="test-model")
    with pytest.raises(SandboxError, match=r"\[agent.sandbox\] image"):
        _render(tmp_path, agent, _slab_cfg())


def test_sandbox_toml_strips_hpc_and_warns_about_srun(tmp_path: Path) -> None:
    cfg = _slab_cfg(
        hpc={"partitions": {"cpu": {}}},
        engines={"qe": {"command": "srun pw.x", "setup": ["module load qe"]}},
        paths={"pseudos": "/shared/pseudos"},
    )
    text, warnings = sandbox_toml(cfg, _agent(), tmp_path / "ws")
    assert "[hpc]" not in text
    assert "[engines.qe]" in text
    assert 'command = "srun pw.x"' in text
    assert "[workspace]" in text
    assert 'model = "test-model"' in text
    # The sandbox table itself, connection details, and serve stay out.
    assert "[agent.sandbox]" not in text
    assert "image" not in text
    assert any("srun" in w for w in warnings)
    assert any("module loads" in w for w in warnings)
    # What it emits, the loader accepts.
    import tomllib

    tomllib.loads(text)


# -- the bridge and the proofs ------------------------------------------------


class _Stub(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # http.server's naming contract
        body = json.dumps({"data": [{"id": "test-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class _UnixHTTPServer(socketserver.ThreadingUnixStreamServer):
    def get_request(self):  # type: ignore[no-untyped-def]
        request, _ = super().get_request()
        return request, ("127.0.0.1", 0)


_PORTS = iter(range(8907, 8957))


@pytest.fixture()
def bridged_stub() -> int:
    """A stub model server on a unix socket, bridged to a loopback port.

    Not under pytest's tmp_path: AF_UNIX paths cap at ~104 characters, and
    tmp_path routinely exceeds that. mkdtemp() under $TMPDIR stays short.
    Each test gets its own port, because forward() runs until its daemon
    thread dies with the process — a stale forwarder still owns its port.
    """
    import tempfile

    sock = str(Path(tempfile.mkdtemp()) / "llm.sock")
    server = _UnixHTTPServer(sock, _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = next(_PORTS)
    threading.Thread(target=forward, args=(sock, port), daemon=True).start()
    yield port
    server.shutdown()


def test_forward_relays_http_over_the_unix_socket(bridged_stub: int) -> None:
    import urllib.request

    with urllib.request.urlopen(
        f"http://127.0.0.1:{bridged_stub}/v1/models", timeout=5
    ) as response:
        payload = json.loads(response.read())
    assert payload["data"][0]["id"] == "test-model"


def test_verify_passes_when_dark_and_bridged(bridged_stub: int) -> None:
    names = verify(
        bridged_stub, probe_url="http://127.0.0.1:1", ready_timeout_s=10
    )
    assert names == ["test-model"]


def test_verify_refuses_a_reachable_internet(bridged_stub: int) -> None:
    reachable = f"http://127.0.0.1:{bridged_stub}/v1/models"
    with pytest.raises(SandboxError, match=r"not\s+isolated"):
        verify(bridged_stub, probe_url=reachable, ready_timeout_s=5)


def test_verify_refuses_a_dead_endpoint(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="did not answer"):
        verify(1, probe_url="http://127.0.0.1:1", ready_timeout_s=2)


# -- preflight and the CLI ----------------------------------------------------


def test_preflight_reports_the_missing_pieces(tmp_path: Path) -> None:
    agent = AgentConfig(model="m")  # no image configured
    rows = preflight(agent, tmp_path / "ws")
    marks = {message: mark for mark, message in rows}
    assert any("image is not set" in m and marks[m] == "-" for m in marks)
    assert any("no serve record" in m and marks[m] == "?" for m in marks)


def test_cli_render_writes_both_files_and_next_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "m"\n[agent.sandbox]\nimage = "/i.sif"\n'
        '[hpc]\ndefault_partition = "cpu"\n[hpc.partitions.cpu]\n'
    )
    result = runner.invoke(
        app, ["sandbox", "render", "do the thing", "-w", str(tmp_path / "ws")]
    )
    assert result.exit_code == 0, result.output
    script = (tmp_path / "sandbox" / "mason-sandbox.sbatch").read_text()
    toml_text = (tmp_path / "sandbox" / "slab.toml").read_text()
    assert "--network none" in script
    assert str(tmp_path / "sandbox" / "slab.toml") in script  # SLAB_CONFIG points at it
    assert "[hpc]" not in toml_text
    assert "read both files" in result.output


def test_cli_render_without_an_image_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text('[agent]\nmodel = "m"\n')
    result = runner.invoke(
        app, ["sandbox", "render", "goal", "-w", str(tmp_path / "ws")]
    )
    assert result.exit_code != 0
    assert "image" in result.output


def test_cli_check_exits_nonzero_when_requirements_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text('[agent]\nmodel = "m"\n')
    result = runner.invoke(app, ["sandbox", "check", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 1
    assert "[-]" in result.output


def test_roster_tables_cannot_override_sandbox() -> None:
    with pytest.raises(Exception, match="sandbox"):
        AgentConfig.model_validate({"roster": {"pi": {"sandbox": {"image": "x"}}}})


def test_qe_bin_is_bound_whole_install_and_ntasks_forwarded(tmp_path: Path) -> None:
    cfg = _slab_cfg(engines={"qe": {"bin": "/shared/qe-7.4/bin"}})
    binds, _ = default_binds(tmp_path / "p", tmp_path / "ws", cfg)
    assert "/shared/qe-7.4:/shared/qe-7.4:ro" in binds  # the prefix, not just bin/
    script, warnings = _render(tmp_path, _agent(), cfg)
    assert '--env SLURM_NTASKS="${SLURM_NTASKS:-1}"' in script
    assert not any("srun" in w for w in warnings)  # nothing to hand-edit


# -- the setup snapshot -------------------------------------------------------


def test_snapshot_runs_the_setup_and_captures_the_delta(tmp_path: Path) -> None:
    from mason.sandbox import snapshot_setup

    bin_dir = tmp_path / "lammps-2025" / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "lmp"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    # A name nothing else exports: a var already in the inherited env (the
    # engines tests leak OMP_NUM_THREADS) would vanish from the delta.
    snapshot = snapshot_setup(
        "lammps",
        (f'export PATH="{bin_dir}:$PATH"', "export SLAB_SNAPSHOT_PROBE=4"),
        "lmp",
    )
    assert snapshot.error is None
    assert snapshot.payload == str(fake)
    assert snapshot.env["SLAB_SNAPSHOT_PROBE"] == "4"
    # Only the ADDED component is kept, not the host's whole PATH.
    assert snapshot.path_prepends["PATH"] == (str(bin_dir),)
    lines = snapshot.setup_lines()
    assert "export SLAB_SNAPSHOT_PROBE=4" in lines
    assert f'export PATH="{bin_dir}${{PATH:+:$PATH}}"' in lines
    # The install prefix (parent of bin/) is what gets bound.
    from mason.sandbox import _snapshot_binds

    assert f"{bin_dir.parent}:{bin_dir.parent}:ro" in _snapshot_binds(snapshot)


def test_snapshot_reports_a_failing_setup(tmp_path: Path) -> None:
    from mason.sandbox import snapshot_setup

    snapshot = snapshot_setup("qe", ("false",), "pw.x")
    assert snapshot.error is not None
    assert snapshot.payload == ""


def test_snapshot_rewrites_the_rendered_toml(tmp_path: Path) -> None:
    from mason.sandbox import SetupSnapshot

    cfg = _slab_cfg(
        engines={"qe": {"command": "pw.x", "setup": ["module load qe/7.4"]}},
        paths={"pseudos": "/shared/pseudos"},
    )
    good = SetupSnapshot(
        "qe",
        "/apps/qe/bin/pw.x",
        {},
        ("/apps/mpi/lib",),
        path_prepends={"PATH": ("/apps/qe/bin",)},
    )
    text, warnings = sandbox_toml(cfg, _agent(), tmp_path / "ws", {"qe": good})
    assert "module load" not in text
    # The container's own PATH survives underneath the prepend.
    assert "/apps/qe/bin${PATH:+:$PATH}" in text
    assert any("snapshotted from the host" in w for w in warnings)
    assert not any("could not snapshot" in w for w in warnings)
    # A failed snapshot keeps the hand-configuration warning, with the cause.
    bad = SetupSnapshot("qe", "", {}, (), error="module: command not found")
    text, warnings = sandbox_toml(cfg, _agent(), tmp_path / "ws", {"qe": bad})
    assert "module load qe/7.4" in text
    assert any("could not snapshot" in w and "module: command not found" in w for w in warnings)


def test_snapshot_binds_reach_the_script_and_collapse(tmp_path: Path) -> None:
    from mason.sandbox import SetupSnapshot

    cfg = _slab_cfg(engines={"qe": {"command": "pw.x", "setup": ["module load qe"]}})
    snapshot = SetupSnapshot(
        "qe", "/apps/qe/bin/pw.x", {"PATH": "/apps/qe/bin"}, ("/apps/qe/lib", "/apps/mpi/lib")
    )
    script, _ = render_sandbox_script(
        _agent(),
        _HPC,
        cfg,
        tmp_path / "ws",
        tmp_path / "project",
        "goal",
        toml_path=tmp_path / "sandbox" / "slab.toml",
        snapshots={"qe": snapshot},
    )
    assert "--bind /apps/qe:/apps/qe:ro" in script
    assert "--bind /apps/mpi/lib:/apps/mpi/lib:ro" in script
    # /apps/qe/lib sits inside the /apps/qe bind and must not repeat.
    assert "--bind /apps/qe/lib:" not in script


def test_qe_pseudo_dir_is_bound_and_satisfies_the_pseudos_warning(tmp_path: Path) -> None:
    cfg = _slab_cfg(engines={"qe": {"pseudo_dir": "/shared/upf"}})
    binds, warnings = default_binds(tmp_path / "p", tmp_path / "ws", cfg)
    assert "/shared/upf:/shared/upf:ro" in binds
    assert not any("pseudopotentials" in w for w in warnings)


def test_bridge_and_forward_chain_end_to_end() -> None:
    """Client -> forward (loopback) -> unix socket -> bridge -> TCP stub.

    The full relay chain the rendered job assembles, both halves the real
    functions, no socat anywhere.
    """
    import tempfile
    import urllib.request
    from http.server import HTTPServer

    from mason.sandbox import bridge

    stub = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    sock = str(Path(tempfile.mkdtemp()) / "llm.sock")
    threading.Thread(
        target=bridge, args=(sock, f"127.0.0.1:{stub.server_address[1]}"), daemon=True
    ).start()
    for _ in range(50):
        if Path(sock).exists():
            break
        threading.Event().wait(0.1)
    port = next(_PORTS)
    threading.Thread(target=forward, args=(sock, port), daemon=True).start()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
        payload = json.loads(response.read())
    assert payload["data"][0]["id"] == "test-model"
    stub.shutdown()


def test_bridge_refuses_a_malformed_upstream() -> None:
    from mason.sandbox import bridge

    with pytest.raises(SandboxError, match="host:port"):
        bridge("/tmp/nope.sock", "not-an-endpoint")


def test_snapshot_resolves_the_bin_form_payload_by_absolute_path(tmp_path: Path) -> None:
    """With [engines.qe] bin, pw.x is off PATH by design; the setup lines
    exist for runtime libraries, and the snapshot must not demand that they
    export the binary."""
    from mason.sandbox import snapshot_engines

    bin_dir = tmp_path / "qe" / "bin"
    bin_dir.mkdir(parents=True)
    pw = bin_dir / "pw.x"
    pw.write_text("#!/bin/sh\n")
    pw.chmod(0o755)
    cfg = _slab_cfg(
        engines={"qe": {"bin": str(bin_dir), "setup": ["export SLAB_SNAPSHOT_QE=1"]}}
    )
    snapshot = snapshot_engines(cfg)["qe"]
    assert snapshot.error is None
    assert snapshot.payload == str(pw)
    assert snapshot.env["SLAB_SNAPSHOT_QE"] == "1"


def test_snapshot_failure_names_the_cause_not_the_stderr_noise(tmp_path: Path) -> None:
    from mason.sandbox import snapshot_setup

    # Noise on stderr plus a nonzero exit: the exit is the story.
    failing = snapshot_setup("qe", ("echo 'Loading requirement: x' >&2", "false"), "pw.x")
    assert failing.error is not None
    assert failing.error.startswith("setup exited 1")
    # Setup succeeds but the binary never appears: say that, not the noise.
    missing = snapshot_setup("qe", ("echo 'Loading requirement: x' >&2",), "slab-no-such-binary")
    assert missing.error is not None
    assert missing.error.startswith("'slab-no-such-binary' did not resolve after setup")
