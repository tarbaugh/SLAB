"""Tests for foundation._ops — the shared machinery behind the CLI and MCP server."""

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from foundation import DEFAULT_POLICY, FoundationError, Workspace
from foundation._ops import (
    launch_script,
    load_policy,
    parse_duration_days,
    resolve_root,
    run_details,
    run_summary,
    ttl_override_policy,
)

HAPPY_SCRIPT = """\
from foundation import check, converged, task

@task
def double(x):
    return 2 * x

y = double(21)
assert y == 42
print("computed", y)

@check
def sane():
    return converged(0.01, below=0.05)
"""

FAILING_SCRIPT = "raise RuntimeError('kaboom')\n"

ARGV_SCRIPT = """\
import sys
print("|".join(sys.argv[1:]))
"""

SELF_MANAGED_SCRIPT = """\
import sys
from foundation import Workspace

with Workspace(sys.argv[1]) as ws, ws.start_run(name="inner"):
    pass
"""


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "ws"


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


# -- resolve_root / durations / policy -------------------------------------------------


def test_resolve_root_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    assert resolve_root(tmp_path) == tmp_path
    assert resolve_root(None) == Path(".slab")
    monkeypatch.setenv("SLAB_WORKSPACE", str(tmp_path / "env-ws"))
    assert resolve_root(None) == tmp_path / "env-ws"
    assert resolve_root(tmp_path) == tmp_path  # explicit beats env


@pytest.mark.parametrize(
    ("text", "days"),
    [("30d", 30.0), ("12h", 0.5), ("36m", 0.025), ("90s", 90 / 86_400), ("1.5d", 1.5)],
)
def test_parse_duration(text: str, days: float) -> None:
    assert parse_duration_days(text) == pytest.approx(days)


def test_parse_duration_zero_means_now() -> None:
    assert 0 < parse_duration_days("0d") < 1e-5


@pytest.mark.parametrize("bad", ["", "30", "d30", "30w", "soon", "-5d"])
def test_parse_duration_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError, match="cannot parse duration"):
        parse_duration_days(bad)


def test_ttl_override_policy_touches_only_ttls() -> None:
    policy = ttl_override_policy(7)
    assert policy.quarantined.ttl_days == 7
    assert policy.verified.ttl_days == 7
    assert policy.promoted.keep == DEFAULT_POLICY.promoted.keep


def test_load_policy_default_when_absent(tmp_path: Path) -> None:
    assert load_policy(tmp_path) is DEFAULT_POLICY


def test_load_policy_from_workspace_file(tmp_path: Path) -> None:
    (tmp_path / "policy.json").write_text(json.dumps({"quarantined": {"ttl_days": 3}}))
    assert load_policy(tmp_path).quarantined.ttl_days == 3


def test_load_policy_explicit_path_and_errors(tmp_path: Path) -> None:
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({"verified": {"ttl_days": 5}}))
    assert load_policy(tmp_path, custom).verified.ttl_days == 5
    with pytest.raises(OSError):
        load_policy(tmp_path, tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"promoted": {"ttl_days": 9}}))
    with pytest.raises(ValidationError):
        load_policy(tmp_path, bad)


# -- summaries -------------------------------------------------------------------------


def test_run_details_shape(root: Path) -> None:
    with Workspace(root) as ws:
        with ws.start_run(name="detailed", intent="inspect me") as run:
            run.keep("result", {"e": -1.0})
            run.check(lambda: True, name="fine")
        details = run_details(ws, run.id[:8])

        assert details["run"]["name"] == "detailed"
        assert details["run"]["state"] == "verified"
        assert details["run"]["failure"] is None
        assert details["checks"] == [
            {
                "name": "fine",
                "kind": "custom",
                "passed": True,
                "message": "returned True",
                "observed": None,
                "expected": None,
            }
        ]
        (artifact,) = details["artifacts"]
        assert artifact["role"] == "terminal"
        assert artifact["bytes_available"] is True
        assert details["history"][0]["to"] == "verified"

        ws.artifacts.discard(artifact["hash"])
        assert run_details(ws, run.id)["artifacts"][0]["bytes_available"] is False

        summary = run_summary(ws.runs.get(run.id))
        assert summary["id"] == run.id and summary["status"] == "completed"
        assert "failure" not in summary  # listings stay compact; show carries evidence


def test_run_details_checks_carry_observed_and_expected(root: Path) -> None:
    """The structured values an agent computes a correction from — not just prose."""
    from foundation import converged

    with Workspace(root) as ws:
        with ws.start_run(name="measured") as run:
            run.check(lambda: converged(0.062, below=0.05, label="fmax"), name="forces")
        (check,) = run_details(ws, run.id)["checks"]
        assert check["passed"] is False
        assert check["observed"] == 0.062
        assert check["expected"] == {"below": 0.05}


def test_run_details_surfaces_failure_evidence(root: Path) -> None:
    """Failed run and failed task both carry their structured failure records."""
    from foundation import task

    @task
    def explode() -> None:
        error = RuntimeError("SCF diverged")
        error.add_note("last SCF residual: 3.2e-2")
        raise error

    with Workspace(root) as ws:
        with pytest.raises(RuntimeError), ws.start_run(name="doomed") as run:
            explode()
        details = run_details(ws, run.id)

        assert details["run"]["failure"]["type"] == "RuntimeError"
        (task_entry,) = details["tasks"]
        assert task_entry["status"] == "failed"
        assert task_entry["error"] == "RuntimeError: SCF diverged"
        assert task_entry["failure"]["notes"] == ["last SCF residual: 3.2e-2"]
        assert "SCF diverged" in task_entry["failure"]["traceback"]


# -- launch_script ---------------------------------------------------------------------


def test_launch_happy_script_verifies(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "wf.py", HAPPY_SCRIPT)
    result = launch_script(root, script, intent="smoke", capture_output=True)
    assert result["state"] == "verified"
    assert result["status"] == "completed"
    assert result["checks_passed"] == result["checks_total"] == 1
    assert result["tasks_recorded"] == 1
    assert result["name"] == "wf"
    assert "computed 42" in result["output"]
    assert "traceback" not in result
    assert "failure" not in result
    with Workspace(root) as ws:
        assert ws.runs.get(result["run_id"]).intent == "smoke"


def test_launch_failing_script_records_failure(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "boom.py", FAILING_SCRIPT)
    result = launch_script(root, script)
    assert result["status"] == "failed"
    assert result["state"] == "quarantined"
    assert result["failure"]["type"] == "RuntimeError"
    assert "kaboom" in result["failure"]["traceback"]
    assert "traceback" not in result  # the structured record replaces the raw text
    with Workspace(root) as ws:
        assert ws.runs.get(result["run_id"]).error == "RuntimeError: kaboom"


def test_launch_passes_argv(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "argv.py", ARGV_SCRIPT)
    result = launch_script(root, script, argv=("alpha", "beta"), capture_output=True)
    assert result["output"].strip() == "alpha|beta"


def test_launch_missing_script(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such workflow script"):
        launch_script(root, root / "ghost.py")


def test_launch_self_managed_script_gets_hint(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "own.py", SELF_MANAGED_SCRIPT)
    with pytest.raises(FoundationError, match=re.escape("plain 'python own.py'")):
        launch_script(root, script, argv=(str(tmp_path / "inner-ws"),))


def test_launch_name_override(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "wf.py", "x = 1\n")
    result = launch_script(root, script, name="custom-name")
    assert result["name"] == "custom-name"


def test_launch_sys_exit_zero_is_normal_completion(root: Path, tmp_path: Path) -> None:
    """The `sys.exit(main())` idiom with a zero code is everyday Python."""
    script = _write(
        tmp_path,
        "exit0.py",
        "import sys\n\ndef main():\n    print('did work')\n    return 0\n\n"
        "if __name__ == '__main__':\n    sys.exit(main())\n",
    )
    result = launch_script(root, script, capture_output=True)
    assert result["status"] == "completed"
    assert "traceback" not in result
    assert "did work" in result["output"]


def test_launch_sys_exit_nonzero_records_failure(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "exit2.py", "import sys\nsys.exit(2)\n")
    result = launch_script(root, script)
    assert result["status"] == "failed"
    assert "sys.exit(2)" in result["failure"]["message"]
    with Workspace(root) as ws:
        assert ws.runs.get(result["run_id"]).error == "ScriptExitError: script called sys.exit(2)"


def test_launch_machinery_failure_falls_back_to_raw_traceback(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When recording the failure itself fails (storage dies mid-crash), the
    run never gets a failure record — but the launcher must still deliver the
    evidence, as a raw traceback covering both the original error and what ate
    its record. The run honestly stays 'running' (expire --include-running is
    the recovery path)."""
    from foundation.lifecycle import ExecutionStatus
    from foundation.store import SQLiteRunStore

    real_set_status = SQLiteRunStore.set_status

    def dying_set_status(self, run_id, status, **kwargs):  # type: ignore[no-untyped-def]
        if ExecutionStatus(status) is ExecutionStatus.FAILED:
            raise OSError("disk full while recording the failure")
        return real_set_status(self, run_id, status, **kwargs)

    monkeypatch.setattr(SQLiteRunStore, "set_status", dying_set_status)
    script = _write(tmp_path, "boom.py", FAILING_SCRIPT)
    result = launch_script(root, script)
    assert "failure" not in result
    assert "kaboom" in result["traceback"]  # the original failure...
    assert "disk full" in result["traceback"]  # ...and what ate its record
    assert result["status"] == "running"


def test_launch_unwritable_workspace_surfaces_real_cause(root: Path, tmp_path: Path) -> None:
    """Storage failures must name the storage problem, not crash obscurely."""
    from foundation import StorageError

    script = _write(tmp_path, "wf.py", "x = 1\n")
    launch_script(root, script)  # create the workspace normally first
    db = root / "runs.db"
    db.chmod(0o444)
    try:
        with pytest.raises(StorageError) as excinfo:
            launch_script(root, script)
        assert "readonly" in str(excinfo.value).lower()
    finally:
        db.chmod(0o644)


def test_capture_includes_check_time_prints(root: Path, tmp_path: Path) -> None:
    """Checks evaluate at context exit; their prints must still be captured —
    under MCP, stdout is the protocol channel."""
    script = _write(
        tmp_path,
        "noisy.py",
        "from foundation import check\n"
        "print('BODY-PRINT')\n"
        "@check\ndef noisy_check():\n    print('CHECK-PRINT')\n    return True\n",
    )
    result = launch_script(root, script, capture_output=True)
    assert "BODY-PRINT" in result["output"]
    assert "CHECK-PRINT" in result["output"]
