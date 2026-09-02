"""``slab doctor``: the whole-stack preflight tells this machine's truth.

Most rows run for real against a temporary project; only the network and
subprocess edges are faked. The freshness check matters most: it must
pass on an untouched render and fail on any drift, because that row is
what retires the stale-sbatch failure class.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mason.cli import app as mason_app
from slab_stack import doctor
from slab_stack.cli import app

runner = CliRunner()


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal project of its own: cwd, config, memory, no registry."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.delenv("SLAB_SITE_CONFIG", raising=False)
    monkeypatch.setenv("SLAB_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "m"\n[hpc]\ndefault_partition = "cpu"\n[hpc.partitions.cpu]\n'
    )
    return tmp_path


def test_offline_doctor_reports_a_healthy_laptop(project: Path) -> None:
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 0, result.output
    assert "config: project" in result.output
    assert "[+] config [agent]: validates" in result.output
    assert "workspace: none yet" in result.output
    assert "[+] memory:" in result.output and "is writable" in result.output
    assert "[+] engines built-in:" in result.output
    assert "[=] endpoint: skipped (--offline)" in result.output
    assert "[=] sandbox: not configured" in result.output
    assert "[=] rendered job: none here" in result.output


def test_a_broken_config_is_a_failing_row(project: Path) -> None:
    (project / "slab.toml").write_text("[not-a-table]\nkey = 1\n")
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 1
    assert "[x] config" in result.output


def test_an_unreachable_endpoint_fails_the_doctor(project: Path) -> None:
    # Port 1 refuses immediately; the probe authenticates exactly as a
    # session would and reports the endpoint row as the failure.
    (project / "slab.toml").write_text(
        '[agent]\nmodel = "m"\nendpoint = "http://127.0.0.1:1/v1"\n'
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "endpoint:" in result.output
    assert "[x] endpoint:" in result.output


def _render(project: Path) -> None:
    (project / "slab.toml").write_text(
        '[agent]\nmodel = "m"\n[agent.sandbox]\nimage = "/i.sif"\n'
        '[hpc]\ndefault_partition = "cpu"\n[hpc.partitions.cpu]\n'
    )
    rendered = runner.invoke(
        mason_app, ["sandbox", "render", "measure a0", "--engine-tasks", "2"]
    )
    assert rendered.exit_code == 0, rendered.output


def test_an_untouched_render_is_fresh(project: Path) -> None:
    _render(project)
    result = runner.invoke(app, ["doctor", "--offline"])
    assert "[+] rendered job: sandbox/ matches a fresh render" in result.output
    # The sandbox preflight rows surface too (the image does not exist here).
    assert "sandbox:" in result.output


def test_a_mutated_render_is_stale(project: Path) -> None:
    _render(project)
    script = project / "sandbox" / "mason-sandbox.sbatch"
    script.write_text(script.read_text() + "# drifted\n")
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 1
    assert "[x] rendered job: sandbox/ differs from a fresh render" in result.output
    assert "slab mason sandbox launch" in result.output


def test_a_config_change_after_render_is_stale(project: Path) -> None:
    _render(project)
    (project / "slab.toml").write_text(
        '[agent]\nmodel = "m"\n[agent.sandbox]\nimage = "/other.sif"\n'
        '[hpc]\ndefault_partition = "cpu"\n[hpc.partitions.cpu]\n'
    )
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 1
    assert "[x] rendered job:" in result.output


def test_deep_probe_answers_with_a_real_engine() -> None:
    rows = doctor._deep_rows(["emt"], timeout_s=120.0)
    assert rows == [("+", rows[0][1])]
    assert "deep emt: single-point answers" in rows[0][1]


def test_deep_probe_reports_a_dead_worker() -> None:
    rows = doctor._deep_rows(["no-such-checkpoint"], timeout_s=120.0)
    assert rows[0][0] == "x"
    assert "deep no-such-checkpoint:" in rows[0][1]


def test_deep_probe_reports_a_hang_as_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hang(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="probe", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", hang)
    rows = doctor._deep_rows(["slow-checkpoint"], timeout_s=1.0)
    assert rows == [("x", "deep slow-checkpoint: no answer within 1s")]


def test_deep_without_checkpoints_says_so() -> None:
    assert doctor._deep_rows([], timeout_s=1.0) == [
        ("=", "deep: no declared checkpoints to probe")
    ]


def test_the_mp_snapshot_row_covers_all_three_states(
    project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    from conftest import build_mp_snapshot

    unconfigured = runner.invoke(app, ["doctor", "--offline"])
    assert "[=] mp snapshot: not configured" in unconfigured.output

    snapshot = build_mp_snapshot(tmp_path_factory.mktemp("data") / "mp-snapshot")
    (project / "slab.toml").write_text(
        f'[agent]\nmodel = "m"\n[builders.mp]\nroot = "{snapshot}"\n'
    )
    healthy = runner.invoke(app, ["doctor", "--offline"])
    assert healthy.exit_code == 0, healthy.output
    assert (
        f"[+] mp snapshot: release 2025.11.1, 4 materials at {snapshot}"
        in healthy.output
    )

    (snapshot / "metadata.sqlite").unlink()
    broken = runner.invoke(app, ["doctor", "--offline"])
    assert broken.exit_code == 1
    assert "[x] mp snapshot:" in broken.output
    assert "metadata.sqlite is missing" in broken.output


def test_the_gracemaker_row_covers_all_three_states(
    project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    unconfigured = runner.invoke(app, ["doctor", "--offline"])
    assert "[=] gracemaker: not configured" in unconfigured.output

    bin_dir = tmp_path_factory.mktemp("grace") / "bin"
    bin_dir.mkdir()
    for name, body in (("python", 'echo "0.5.1"'), ("gracemaker", "echo training")):
        script = bin_dir / name
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
    (project / "slab.toml").write_text(
        f'[agent]\nmodel = "m"\n[builders.gracemaker]\ncommand = "{bin_dir / "gracemaker"}"\n'
    )
    healthy = runner.invoke(app, ["doctor", "--offline"])
    assert healthy.exit_code == 0, healthy.output
    assert (
        f"[+] gracemaker: tensorpotential 0.5.1 via {bin_dir / 'gracemaker'}"
        in healthy.output
    )

    (bin_dir / "python").unlink()  # the environment stops answering the probe
    grace = bin_dir / "gracemaker"
    stat = grace.stat()
    os.utime(grace, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))  # re-key the probe memo
    broken = runner.invoke(app, ["doctor", "--offline"])
    assert broken.exit_code == 1
    assert "[x] gracemaker: configured, but the environment" in broken.output


def test_deep_probes_the_snapshot_and_names_a_truncated_transfer(
    project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    from conftest import build_mp_snapshot

    snapshot = build_mp_snapshot(
        tmp_path_factory.mktemp("data") / "mp-snapshot",
        extra_materials=({"material_id": "mp-777", "cif_path": "cifs/zz/gone.cif"},),
    )
    (project / "slab.toml").write_text(
        f'[agent]\nmodel = "m"\n[builders.mp]\nroot = "{snapshot}"\n'
    )
    result = runner.invoke(app, ["doctor", "--offline", "--deep"])
    assert "[+] deep mp: metadata.sqlite quick_check ok" in result.output
    assert "[x] deep mp: 1/5 sampled CIFs unresolvable" in result.output
    assert "transferred completely" in result.output
    assert result.exit_code == 1

    (snapshot / "cifs" / "zz").mkdir(parents=True)
    from ase.build import bulk
    from ase.io import write as ase_write

    ase_write(snapshot / "cifs" / "zz" / "gone.cif", bulk("Cu", "fcc", a=3.6))
    healed = runner.invoke(app, ["doctor", "--offline", "--deep"])
    assert "[+] deep mp: 5 sampled CIFs resolve and exist" in healed.output


def test_a_render_with_an_entry_agent_is_fresh(project: Path) -> None:
    """The freshness re-render must carry the recorded agent, or every
    planner job would read as stale."""
    (project / "slab.toml").write_text(
        '[agent]\nmodel = "m"\n[agent.sandbox]\nimage = "/i.sif"\n'
        '[hpc]\ndefault_partition = "cpu"\n[hpc.partitions.cpu]\n'
    )
    rendered = runner.invoke(mason_app, ["sandbox", "render", "measure a0", "--agent", "planner"])
    assert rendered.exit_code == 0, rendered.output
    result = runner.invoke(app, ["doctor", "--offline"])
    assert "[+] rendered job: sandbox/ matches a fresh render" in result.output
