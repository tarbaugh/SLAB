"""CLI tests via typer's CliRunner — every verb, happy and sad paths."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slab import Workspace
from slab.cli import app

runner = CliRunner()

HAPPY_SCRIPT = """\
from slab import check, converged, task

@task
def double(x):
    return 2 * x

y = double(21)

@check
def sane():
    return converged(0.01, below=0.05)
"""


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "ws"


def _seed_run(root: Path, *, name: str = "seeded", verified: bool = True) -> str:
    """Create a run with a check, task-free, directly through the library."""
    with Workspace(root) as ws:
        with ws.start_run(name=name, intent=f"intent of {name}") as run:
            run.keep("result", {"e": -1.5})
            run.check(lambda: verified, name="gate")
        return run.id


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("slab ")


# -- run -------------------------------------------------------------------------------


def test_run_happy(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "wf.py"
    script.write_text(HAPPY_SCRIPT)
    result = runner.invoke(app, ["run", str(script), "-w", str(root), "--intent", "cli smoke"])
    assert result.exit_code == 0, result.output
    assert "state=verified" in result.output
    assert "checks=1/1" in result.output
    assert "tasks=1" in result.output


def test_run_failing_script_exits_nonzero(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "boom.py"
    script.write_text("raise RuntimeError('kaboom')\n")
    result = runner.invoke(app, ["run", str(script), "-w", str(root)])
    assert result.exit_code == 1
    assert "kaboom" in result.output
    assert "status=failed" in result.output


def test_run_sys_exit_zero_succeeds(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "exit0.py"
    script.write_text(
        "import sys\n\ndef main():\n    return 0\n\n"
        "if __name__ == '__main__':\n    sys.exit(main())\n"
    )
    result = runner.invoke(app, ["run", str(script), "-w", str(root)])
    assert result.exit_code == 0, result.output
    assert "status=completed" in result.output


def test_run_sys_exit_nonzero_fails_honestly(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "exit3.py"
    script.write_text("import sys\nsys.exit(3)\n")
    result = runner.invoke(app, ["run", str(script), "-w", str(root)])
    assert result.exit_code == 1
    assert "status=failed" in result.output
    assert "sys.exit(3)" in result.output


def test_run_missing_script(root: Path) -> None:
    result = runner.invoke(app, ["run", str(root / "ghost.py"), "-w", str(root)])
    assert result.exit_code == 1
    assert "no such workflow script" in result.output


def test_run_self_managed_script_hint(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "own.py"
    script.write_text(
        "import sys\nfrom slab import Workspace\n"
        "with Workspace(sys.argv[1]) as ws, ws.start_run():\n    pass\n"
    )
    result = runner.invoke(app, ["run", str(script), "-w", str(root), str(tmp_path / "inner")])
    assert result.exit_code == 1
    assert "plain 'python own.py'" in result.output


# -- list ------------------------------------------------------------------------------


def test_list_empty(root: Path) -> None:
    result = runner.invoke(app, ["list", "-w", str(root)])
    assert result.exit_code == 0
    assert "no runs" in result.output


def test_list_rows_and_filters(root: Path) -> None:
    verified_id = _seed_run(root, name="good", verified=True)
    _seed_run(root, name="junk", verified=False)
    result = runner.invoke(app, ["list", "-w", str(root)])
    assert result.exit_code == 0
    assert "good" in result.output and "junk" in result.output
    assert "intent of good" in result.output

    filtered = runner.invoke(app, ["list", "-w", str(root), "--state", "verified"])
    assert "good" in filtered.output and "junk" not in filtered.output
    assert verified_id[:10] in filtered.output


def test_list_quiet_prints_full_ids(root: Path) -> None:
    run_id = _seed_run(root)
    result = runner.invoke(app, ["list", "-w", str(root), "-q"])
    assert result.output.split() == [run_id]


def test_list_rejects_bogus_state(root: Path) -> None:
    result = runner.invoke(app, ["list", "-w", str(root), "--state", "bogus"])
    assert result.exit_code == 1
    assert "LifecycleState" in result.output


# -- show ------------------------------------------------------------------------------


def test_show_renders_sections(root: Path) -> None:
    run_id = _seed_run(root, name="showme")
    result = runner.invoke(app, ["show", run_id[:8], "-w", str(root)])
    assert result.exit_code == 0
    out = result.output
    assert f"run {run_id}  showme" in out
    assert "state:   verified" in out
    assert "intent:  intent of showme" in out
    assert "checks:  1/1 passed" in out
    assert "[+] gate" in out
    assert "result  terminal" in out and "bytes" in out
    assert "quarantined -> verified" in out


def test_show_json(root: Path) -> None:
    run_id = _seed_run(root)
    result = runner.invoke(app, ["show", run_id, "-w", str(root), "--json"])
    assert result.exit_code == 0
    details = json.loads(result.output)
    assert details["run"]["id"] == run_id
    assert details["artifacts"][0]["bytes_available"] is True


def test_show_unknown_run(root: Path) -> None:
    _seed_run(root)
    result = runner.invoke(app, ["show", "zzzz", "-w", str(root)])
    assert result.exit_code == 1
    assert "no run matches" in result.output


# -- promote ---------------------------------------------------------------------------


def test_promote_verified_run(root: Path) -> None:
    run_id = _seed_run(root, verified=True)
    result = runner.invoke(
        app, ["promote", run_id[:8], "-w", str(root), "--reason", "best of batch"]
    )
    assert result.exit_code == 0
    assert result.output.startswith(f"promoted {run_id}")
    with Workspace(root) as ws:
        assert ws.runs.get(run_id).state.value == "promoted"
        assert ws.runs.history(run_id)[-1].reason == "best of batch"


def test_promote_unverified_needs_force(root: Path) -> None:
    run_id = _seed_run(root, verified=False)
    refused = runner.invoke(app, ["promote", run_id, "-w", str(root)])
    assert refused.exit_code == 1
    assert "force" in refused.output

    forced = runner.invoke(app, ["promote", run_id, "-w", str(root), "--force"])
    assert forced.exit_code == 0
    with Workspace(root) as ws:
        assert ws.runs.history(run_id)[-1].forced is True


# -- expire / gc -----------------------------------------------------------------------


def test_expire_older_than_zero(root: Path) -> None:
    _seed_run(root, verified=False)
    promoted = _seed_run(root, verified=True)
    runner.invoke(app, ["promote", promoted, "-w", str(root)])

    result = runner.invoke(app, ["expire", "-w", str(root), "--older-than", "0d"])
    assert result.exit_code == 0
    assert "1 run(s) expired" in result.output
    with Workspace(root) as ws:
        assert ws.runs.get(promoted).state.value == "promoted"  # untouchable


def test_expire_default_policy_keeps_fresh_runs(root: Path) -> None:
    _seed_run(root)
    result = runner.invoke(app, ["expire", "-w", str(root)])
    assert result.exit_code == 0
    assert "0 run(s) expired" in result.output


def test_expire_rejects_bad_duration(root: Path) -> None:
    result = runner.invoke(app, ["expire", "-w", str(root), "--older-than", "soon"])
    assert result.exit_code == 1
    assert "cannot parse duration" in result.output


def test_expire_include_running_flag(root: Path) -> None:
    from datetime import timedelta

    from slab import Run, utcnow

    with Workspace(root) as ws:
        dead = ws.runs.create(Run(name="killed", created_at=utcnow() - timedelta(days=400)))
        ws.runs.set_status(dead.id, "running")

    default = runner.invoke(app, ["expire", "-w", str(root), "--older-than", "0d"])
    assert "0 run(s) expired" in default.output  # protected by default

    forced = runner.invoke(
        app, ["expire", "-w", str(root), "--older-than", "0d", "--include-running"]
    )
    assert "1 run(s) expired" in forced.output
    with Workspace(root) as ws:
        recovered = ws.runs.get(dead.id)
        assert recovered.state.value == "expired"
        assert recovered.status.value == "failed"


def test_expire_with_policy_file(root: Path, tmp_path: Path) -> None:
    _seed_run(root, verified=False)
    policy = tmp_path / "p.json"
    policy.write_text(json.dumps({"quarantined": {"ttl_days": 1e-9}}))
    result = runner.invoke(app, ["expire", "-w", str(root), "--policy", str(policy)])
    assert "1 run(s) expired" in result.output


def test_gc_dry_run_then_real(root: Path) -> None:
    run_id = _seed_run(root, verified=False)
    runner.invoke(app, ["expire", "-w", str(root), "--older-than", "0d"])

    dry = runner.invoke(app, ["gc", "-w", str(root), "--dry-run"])
    assert dry.exit_code == 0
    assert "would drop 1 blob(s)" in dry.output

    real = runner.invoke(app, ["gc", "-w", str(root)])
    assert "dropped 1 blob(s)" in real.output
    with Workspace(root) as ws:
        ref = ws.runs.get_artifact(run_id, "result")
        assert not ws.artifacts.has(ref.hash)


def test_gc_warns_on_missing_demanded_bytes(root: Path) -> None:
    run_id = _seed_run(root, verified=True)
    runner.invoke(app, ["promote", run_id, "-w", str(root)])
    with Workspace(root) as ws:
        ws.artifacts.discard(ws.runs.get_artifact(run_id, "result").hash)
    result = runner.invoke(app, ["gc", "-w", str(root)])
    assert "WARNING" in result.output
    assert "missing" in result.output


# -- workspace resolution / mcp guard --------------------------------------------------


def test_workspace_env_var(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_run(root)
    monkeypatch.setenv("SLAB_WORKSPACE", str(root))
    result = runner.invoke(app, ["list"])
    assert "seeded" in result.output


def test_mcp_without_package_gives_hint(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "slab.mcp_server", None)
    result = runner.invoke(app, ["mcp", "-w", str(root)])
    assert result.exit_code == 1
    assert "pip install 'slab[mcp]'" in result.output


# -- engines ---------------------------------------------------------------------------


EMT_ENTRY = {
    "calculator": "ase.calculators.emt.EMT",
    "version": "ase-built-in",
    "description": "cluster-declared EMT for tests",
}


def _write_engines(tmp_path: Path, engines: dict, cluster: str = "delta") -> Path:
    path = tmp_path / "engines.json"
    path.write_text(json.dumps({"cluster": cluster, "engines": engines}))
    return path


def test_engines_list_without_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert "built-in: emt, lj, mace, qe, rootstock" in result.output
    assert "none configured" in result.output


def test_engines_list_with_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry = _write_engines(tmp_path, {"emt-cluster": EMT_ENTRY})
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert "registry [delta]" in result.output
    assert "emt-cluster" in result.output
    assert "ase-built-in" in result.output
    assert "cluster-declared EMT" in result.output


def test_engines_list_invalid_registry_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = tmp_path / "engines.json"
    bad.write_text(json.dumps({"engines": {"mace": EMT_ENTRY}}))  # shadows a built-in
    result = runner.invoke(app, ["engines", "list", "--registry", str(bad)])
    assert result.exit_code == 1
    assert "built-in engine name" in result.output


def test_engines_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys as _sys

    registry = _write_engines(
        tmp_path,
        {
            "good": {**EMT_ENTRY, "probe": [_sys.executable, "-c", "pass"]},
            "bad": {**EMT_ENTRY, "probe": [_sys.executable, "-c", "raise SystemExit(2)"]},
        },
    )
    result = runner.invoke(app, ["engines", "verify", "--registry", str(registry)])
    assert result.exit_code == 1
    assert "[+] good" in result.output
    assert "[x] bad" in result.output
    assert "1/2 engines verified" in result.output

    ok_registry = _write_engines(tmp_path, {"good": EMT_ENTRY})
    ok = runner.invoke(app, ["engines", "verify", "--registry", str(ok_registry)])
    assert ok.exit_code == 0
    assert "1/1 engines verified" in ok.output


def test_engines_verify_without_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    result = runner.invoke(app, ["engines", "verify"])
    assert result.exit_code == 1
    assert "no engine registry configured" in result.output


def test_engines_list_shows_rootstock_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("rootstock", reason="rootstock extra not installed")
    root = tmp_path / "rootstock-install"
    env_dir = root / "envs" / "fake-mace"
    env_dir.mkdir(parents=True)
    (env_dir / "env_source.py").write_text('CHECKPOINTS = {"fake-mace-checkpoint": "small"}\n')
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ROOTSTOCK_ROOT", str(root))
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert "rootstock checkpoints (usable directly as engine=)" in result.output
    assert "fake-mace: fake-mace-checkpoint" in result.output


# -- rendering details -----------------------------------------------------------------


def test_age_buckets() -> None:
    from datetime import timedelta

    from slab import utcnow
    from slab.cli import _age

    now = utcnow()
    assert _age(now).endswith("s")
    assert _age(now - timedelta(minutes=5)) == "5m"
    assert _age(now - timedelta(hours=3)) == "3h"
    assert _age(now - timedelta(days=12)) == "12d"


def test_show_failed_run_displays_error_and_tasks(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "wf.py"
    script.write_text(
        "from slab import task\n"
        "@task\ndef double(x):\n    return 2 * x\n"
        "double(4)\n"
        "raise RuntimeError('late failure')\n"
    )
    launched = runner.invoke(app, ["run", str(script), "-w", str(root)])
    assert launched.exit_code == 1
    assert "Traceback (most recent call last)" in launched.output  # evidence on stderr
    run_id = launched.output.splitlines()[-1].split()[1]

    shown = runner.invoke(app, ["show", run_id, "-w", str(root)])
    assert shown.exit_code == 0
    assert "error:   RuntimeError: late failure" in shown.output
    # no failed task explains this failure, so the run's own traceback renders
    # (between the error line and the created line)
    assert "Traceback (most recent call last)" in shown.output.split("created:")[0]
    assert "tasks:" in shown.output
    assert "1. double  completed" in shown.output


def test_run_prints_raw_traceback_when_failure_recording_dies(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw-traceback fallback (no failure record exists) must reach the
    terminal, and the exit code must be nonzero even though the run was never
    marked failed."""
    from slab.lifecycle import ExecutionStatus
    from slab.store import SQLiteRunStore

    real_set_status = SQLiteRunStore.set_status

    def dying_set_status(self, run_id, status, **kwargs):  # type: ignore[no-untyped-def]
        if ExecutionStatus(status) is ExecutionStatus.FAILED:
            raise OSError("disk full while recording the failure")
        return real_set_status(self, run_id, status, **kwargs)

    monkeypatch.setattr(SQLiteRunStore, "set_status", dying_set_status)
    script = tmp_path / "boom.py"
    script.write_text("raise RuntimeError('kaboom')\n")
    result = runner.invoke(app, ["run", str(script), "-w", str(root)])
    assert result.exit_code == 1
    assert "kaboom" in result.output
    assert "disk full" in result.output
    assert "status=running" in result.output  # honest about the stuck state


def test_show_renders_differing_run_failure_despite_failed_task(
    root: Path, tmp_path: Path
) -> None:
    """A task failure the script caught must not hide a different run failure:
    both tracebacks render, each under its owner."""
    script = tmp_path / "wf.py"
    script.write_text(
        "from slab import task\n"
        "@task\ndef explode():\n"
        "    raise RuntimeError('SCF diverged')\n"
        "try:\n"
        "    explode()\n"
        "except RuntimeError:\n"
        "    pass\n"
        "raise ValueError('post-processing failed: lattice constant NaN')\n"
    )
    launched = runner.invoke(app, ["run", str(script), "-w", str(root)])
    assert launched.exit_code == 1
    run_id = launched.output.splitlines()[-1].split()[1]

    shown = runner.invoke(app, ["show", run_id, "-w", str(root)])
    assert shown.exit_code == 0
    # run-level ValueError evidence renders up top (before 'created:')...
    head = shown.output.split("created:")[0]
    assert "ValueError: post-processing failed" in head
    assert "Traceback (most recent call last)" in head
    # ...and the task's own RuntimeError evidence renders under the task
    assert shown.output.count("Traceback (most recent call last)") == 2
    assert "RuntimeError: SCF diverged" in shown.output


def test_show_failed_task_renders_its_failure_evidence(root: Path, tmp_path: Path) -> None:
    """Task-level failures render under the task; the run does not repeat them."""
    script = tmp_path / "wf.py"
    script.write_text(
        "from slab import task\n"
        "@task\ndef explode():\n"
        "    e = RuntimeError('SCF diverged')\n"
        "    e.add_note('last residual: 3.2e-2')\n"
        "    raise e\n"
        "explode()\n"
    )
    launched = runner.invoke(app, ["run", str(script), "-w", str(root)])
    assert launched.exit_code == 1
    run_id = launched.output.splitlines()[-1].split()[1]

    shown = runner.invoke(app, ["show", run_id, "-w", str(root)])
    assert shown.exit_code == 0
    assert "1. explode  failed" in shown.output
    assert "last residual: 3.2e-2" in shown.output  # the note reaches the reader
    assert shown.output.count("Traceback (most recent call last)") == 1  # no duplication


def test_gc_rejects_bad_policy_file(root: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    result = runner.invoke(app, ["gc", "-w", str(root), "--policy", str(bad)])
    assert result.exit_code == 1


def test_gc_reports_orphans(root: Path) -> None:
    with Workspace(root) as ws:
        ws.artifacts.put_bytes(b"unclaimed scratch data")
    result = runner.invoke(app, ["gc", "-w", str(root)])
    assert "orphans (unreferenced, not deleted): 1" in result.output


# -- pseudos and protocols ---------------------------------------------------------------


def test_pseudos_list_empty_and_populated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from conftest import make_family

    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    empty = runner.invoke(app, ["pseudos", "list"])
    assert empty.exit_code == 0
    assert "no families installed" in empty.output

    make_family(tmp_path / "pseudos")
    populated = runner.invoke(app, ["pseudos", "list"])
    assert populated.exit_code == 0
    assert "SSSP/1.3.0/PBEsol/efficiency" in populated.output
    assert "2 elements" in populated.output


def test_pseudos_verify_happy_and_tampered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from conftest import make_family

    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    _family, directory = make_family(tmp_path / "pseudos")
    ok = runner.invoke(app, ["pseudos", "verify", "SSSP/1.3/PBEsol/efficiency"])
    assert ok.exit_code == 0
    assert "all 2 files match" in ok.output

    (directory / "Si.test.upf").write_text("tampered")
    bad = runner.invoke(app, ["pseudos", "verify", "SSSP/1.3/PBEsol/efficiency"])
    assert bad.exit_code == 1
    assert "checksum mismatch" in bad.output


def test_pseudos_verify_unknown_family_teaches_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    result = runner.invoke(app, ["pseudos", "verify", "SSSP/1.3/PBEsol/efficiency"])
    assert result.exit_code == 1
    assert "slab pseudos install" in result.output


def test_pseudos_install_refuses_unknown_kind() -> None:
    result = runner.invoke(app, ["pseudos", "install", "pseudo-dojo"])
    assert result.exit_code == 1
    assert "only 'sssp'" in result.output


def test_protocols_list_and_show() -> None:
    listed = runner.invoke(app, ["protocols", "list"])
    assert listed.exit_code == 0
    for name in ("fast", "balanced", "stringent"):
        assert name in listed.output

    shown = runner.invoke(app, ["protocols", "show", "balanced"])
    assert shown.exit_code == 0
    assert "kpoints_distance: 0.15" in shown.output
    assert "forc_conv_thr_ev_per_ang" in shown.output

    as_json = runner.invoke(app, ["protocols", "show", "balanced", "--json"])
    assert json.loads(as_json.output)["degauss"] == 0.02

    unknown = runner.invoke(app, ["protocols", "show", "extreme"])
    assert unknown.exit_code == 1
    assert "available: balanced, fast, stringent" in unknown.output


def test_engines_list_includes_protocols_and_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from conftest import make_family

    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    empty = runner.invoke(app, ["engines", "list"])
    assert empty.exit_code == 0
    assert "qe protocols: balanced, fast, stringent" in empty.output
    assert "pseudo families: none installed" in empty.output

    make_family(tmp_path / "pseudos")
    populated = runner.invoke(app, ["engines", "list"])
    assert "pseudo families: SSSP/1.3.0/PBEsol/efficiency" in populated.output
