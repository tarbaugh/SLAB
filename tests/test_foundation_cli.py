"""Foundation CLI tests via typer's CliRunner — every verb, happy and sad paths."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from foundation import Workspace
from foundation.cli import app

runner = CliRunner()


HAPPY_SCRIPT = """\
from foundation import check, converged, task

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
    assert result.output.startswith("foundation ")


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
        "import sys\nfrom foundation import Workspace\n"
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

    from foundation import Run, utcnow

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
    monkeypatch.setitem(__import__("sys").modules, "foundation.mcp_server", None)
    result = runner.invoke(app, ["mcp", "-w", str(root)])
    assert result.exit_code == 1
    assert "pip install 'slab-stack[mcp]'" in result.output


# -- rendering details -----------------------------------------------------------------


def test_age_buckets() -> None:
    from datetime import timedelta

    from foundation import utcnow
    from foundation.cli import _age

    now = utcnow()
    assert _age(now).endswith("s")
    assert _age(now - timedelta(minutes=5)) == "5m"
    assert _age(now - timedelta(hours=3)) == "3h"
    assert _age(now - timedelta(days=12)) == "12d"


def test_show_failed_run_displays_error_and_tasks(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "wf.py"
    script.write_text(
        "from foundation import task\n"
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
    from foundation.lifecycle import ExecutionStatus
    from foundation.store import SQLiteRunStore

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
        "from foundation import task\n"
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
        "from foundation import task\n"
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



# -- every verb reports workspace-open failures as error lines ---------------


def _future_workspace(tmp_path: Path) -> Path:
    """A runs.db written by a newer foundation than this one."""
    import sqlite3

    root = tmp_path / "future"
    root.mkdir()
    # contextlib.closing, not the connection's own context manager: sqlite3's
    # only wraps a transaction and leaves the handle open.
    from contextlib import closing

    with closing(sqlite3.connect(root / "runs.db")) as db:
        db.execute("PRAGMA user_version = 99")
    return root


@pytest.mark.parametrize(
    "argv",
    [
        ["list"],
        ["show", "01xxxxxxxx"],
        ["promote", "01xxxxxxxx"],
        ["expire", "--older-than", "0d"],
        ["gc"],
    ],
)
def test_a_future_schema_workspace_fails_as_an_error_line(
    tmp_path: Path, argv: list[str]
) -> None:
    """SchemaVersionError must print `error: ...`, never a traceback.

    Before this test, `expire` and `gc` opened the Workspace outside their
    try block, so exactly these two verbs crashed with a raw traceback while
    the rest reported cleanly.
    """
    result = runner.invoke(app, [*argv, "-w", str(_future_workspace(tmp_path))])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "error:" in result.output
    assert "schema version 99" in result.output


def test_a_workspace_path_that_is_a_file_fails_as_an_error_line(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("occupied")
    result = runner.invoke(app, ["gc", "-w", str(blocker)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "error:" in result.output
