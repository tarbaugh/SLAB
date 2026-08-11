"""Tests for slab._ops — the shared machinery behind the CLI and MCP server."""

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from slab import DEFAULT_POLICY, SlabError, Workspace
from slab._ops import (
    launch_script,
    load_policy,
    parse_duration_days,
    resolve_root,
    run_details,
    run_summary,
    ttl_override_policy,
)

HAPPY_SCRIPT = """\
from slab import check, converged, task

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
from slab import Workspace

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
        assert details["checks"] == [
            {"name": "fine", "kind": "custom", "passed": True, "message": "returned True"}
        ]
        (artifact,) = details["artifacts"]
        assert artifact["role"] == "terminal"
        assert artifact["bytes_available"] is True
        assert details["history"][0]["to"] == "verified"

        ws.artifacts.discard(artifact["hash"])
        assert run_details(ws, run.id)["artifacts"][0]["bytes_available"] is False

        summary = run_summary(ws.runs.get(run.id))
        assert summary["id"] == run.id and summary["status"] == "completed"


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
    with Workspace(root) as ws:
        assert ws.runs.get(result["run_id"]).intent == "smoke"


def test_launch_failing_script_records_failure(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "boom.py", FAILING_SCRIPT)
    result = launch_script(root, script)
    assert result["status"] == "failed"
    assert result["state"] == "quarantined"
    assert "kaboom" in result["traceback"]
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
    with pytest.raises(SlabError, match=re.escape("plain 'python own.py'")):
        launch_script(root, script, argv=(str(tmp_path / "inner-ws"),))


def test_launch_name_override(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "wf.py", "x = 1\n")
    result = launch_script(root, script, name="custom-name")
    assert result["name"] == "custom-name"
