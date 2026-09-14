"""Tests for foundation._ops — the shared machinery behind the CLI and MCP server."""

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from foundation import DEFAULT_POLICY, FoundationError, Run, Workspace
from foundation._ops import (
    launch_script,
    load_policy,
    parse_duration_days,
    promote_session,
    resolve_root,
    retire_session,
    run_commands,
    run_details,
    run_summary,
    sessions_summary,
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


# -- sessions --------------------------------------------------------------------------


def _run(ws: Workspace, name: str, *, verified: bool, session: str | None) -> str:
    with ws.start_run(name=name, session=session) as run:
        run.check(lambda: verified, name="gate")
    return run.id


def test_launch_script_stamps_the_session(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "wf.py", "x = 1\n")
    result = launch_script(root, script, session="chat-1")
    assert result["session"] == "chat-1"
    with Workspace(root) as ws:
        assert ws.runs.get(result["run_id"]).session == "chat-1"


def test_launch_script_falls_back_to_the_environment(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLAB_SESSION", "from-env")
    script = _write(tmp_path, "wf.py", "x = 1\n")
    assert launch_script(root, script)["session"] == "from-env"


def test_sessions_summary_shape(root: Path) -> None:
    with Workspace(root) as ws:
        _run(ws, "a", verified=True, session="chat-1")
        _run(ws, "b", verified=False, session="chat-1")
        _run(ws, "c", verified=True, session=None)
        summary = sessions_summary(ws)
    assert summary["unstamped"] == 1
    (row,) = summary["sessions"]
    assert (row["session"], row["runs"]) == ("chat-1", 2)
    assert row["breakdown"] == "1 quarantined, 1 verified"
    assert row["states"] == {"quarantined": 1, "verified": 1}


def test_sessions_summary_limit_keeps_the_unstamped_count_whole(root: Path) -> None:
    """A row limit must not make the unstamped tally look larger than it is."""
    with Workspace(root) as ws:
        _run(ws, "a", verified=True, session="chat-1")
        _run(ws, "b", verified=True, session="chat-2")
        _run(ws, "c", verified=True, session=None)
        summary = sessions_summary(ws, limit=1)
    assert len(summary["sessions"]) == 1
    assert summary["unstamped"] == 1


def test_promote_session_outcomes_are_reported_in_run_order(root: Path) -> None:
    with Workspace(root) as ws:
        first = _run(ws, "a", verified=True, session="chat-1")
        second = _run(ws, "b", verified=False, session="chat-1")
        result = promote_session(ws, "chat-1")
        assert [o["id"] for o in result["outcomes"]] == [first, second]  # oldest first
        assert [o["outcome"] for o in result["outcomes"]] == ["promoted", "skipped"]
        assert result["complete"] is False
        assert ws.runs.get(first).state.value == "promoted"


def test_promote_session_actor_is_recorded(root: Path) -> None:
    with Workspace(root) as ws:
        run_id = _run(ws, "a", verified=True, session="chat-1")
        promote_session(ws, "chat-1", actor="agent")
        assert ws.runs.history(run_id)[-1].actor == "agent"


def test_promote_session_of_only_permanent_runs_is_complete(root: Path) -> None:
    with Workspace(root) as ws:
        _run(ws, "a", verified=True, session="chat-1")
        assert promote_session(ws, "chat-1")["complete"] is True
        again = promote_session(ws, "chat-1")
        assert (again["already"], again["complete"]) == (1, True)


def test_promote_session_yields_to_a_concurrent_change(root: Path) -> None:
    """A run promoted (or expired) between the listing and the write is
    reported as skipped, never as a crash: the rest of the session still
    promotes, and rerunning the command settles it."""
    from foundation import IllegalTransitionError, LifecycleState

    with Workspace(root) as ws:
        run_id = _run(ws, "a", verified=True, session="chat-1")
        real = ws.runs.transition

        def racing(rid: str, to_state: object, **kwargs: object) -> object:
            ws.runs.transition = real  # only the first write loses
            raise IllegalTransitionError(
                LifecycleState.VERIFIED,
                LifecycleState.PROMOTED,
                detail="someone else moved it",
            )

        ws.runs.transition = racing  # type: ignore[method-assign]
        result = promote_session(ws, "chat-1")
        (outcome,) = result["outcomes"]
        assert outcome["outcome"] == "skipped"
        assert "someone else moved it" in outcome["detail"]
        assert result["complete"] is False
        assert ws.runs.get(run_id).state.value == "verified"


# -- retire_session ------------------------------------------------------------------


def _kept_run(ws: Workspace, name: str, *, session: str, payload: bytes) -> str:
    """A verified run with one artifact whose bytes are its own."""
    with ws.start_run(name=name, session=session) as run:
        run.keep("out", payload)
        run.check(lambda: True, name="gate")
    return run.id


def _outcomes(report: dict[str, object]) -> dict[str, str]:
    return {str(k["id"]): str(k["outcome"]) for k in report["kept"]}  # type: ignore[index,union-attr]


def test_retire_promotes_the_cited_and_expires_the_rest(root: Path) -> None:
    with Workspace(root) as ws:
        cited = _kept_run(ws, "eos", session="chat-1", payload=b"a" * 100)
        shakeout = _kept_run(ws, "smoke", session="chat-1", payload=b"b" * 40)
        draft = _run(ws, "draft", verified=False, session="chat-1")
        report = retire_session(ws, "chat-1", keep=[cited])
        assert _outcomes(report) == {cited: "promoted"}
        assert [e["id"] for e in report["expired"]] == [shakeout, draft]
        assert report["skipped"] == [] and report["purged"] == {}
        assert report["complete"] is True
        assert (report["runs_total"], report["runs_promoted"], report["runs_expired"]) == (3, 1, 2)
        assert report["bytes_promoted"] > report["bytes_expired"] > 0
        assert report["bytes_total"] == report["bytes_promoted"] + report["bytes_expired"]
        assert ws.runs.get(cited).state.value == "promoted"
        assert ws.runs.get(shakeout).state.value == "expired"
        assert ws.runs.get(draft).state.value == "expired"
        assert ws.runs.history(cited)[-1].reason == "cited by the finish of session chat-1"
        assert ws.runs.history(cited)[-1].actor == "system"
        assert ws.runs.history(shakeout)[-1].reason == "uncited by the finish of session chat-1"
        # The expired run keeps its row and its bytes: expire is a state change.
        assert ws.artifacts.has(ws.runs.get_artifact(shakeout, "out").hash)


@pytest.mark.parametrize(
    ("make", "detail"),
    [
        ("unverified", "not verified"),
        ("failed", "failed"),
        ("expired", "expired"),
        ("running", "running"),
        ("unknown", "unknown id"),
    ],
)
def test_retire_reports_a_cited_run_it_cannot_promote_and_never_forces(
    root: Path, make: str, detail: str
) -> None:
    with Workspace(root) as ws:
        if make == "unverified":
            run_id = _run(ws, "a", verified=False, session="chat-1")
        elif make == "failed":
            with pytest.raises(RuntimeError), ws.start_run(name="a", session="chat-1") as run:
                raise RuntimeError("boom")
            run_id = run.id
        elif make == "expired":
            run_id = _run(ws, "a", verified=True, session="chat-1")
            ws.runs.transition(run_id, "expired", reason="ttl")
        elif make == "running":
            active = ws.start_run(name="a", session="chat-1")
            run_id = active.__enter__().id
        else:
            _run(ws, "other", verified=True, session="chat-1")
            run_id = "01zzzzzzzzzzzzzzzzzzzzzzzz"
        report = retire_session(ws, "chat-1", keep=[run_id])
        (kept,) = report["kept"]
        assert kept["outcome"] == "skipped"
        assert kept["detail"].startswith(detail)
        assert report["complete"] is False
        if make != "unknown":
            assert ws.runs.get(run_id).state.value != "promoted"
        if make == "running":
            assert report["expired"] == []  # a cited run is never expired either
            active.__exit__(None, None, None)


def test_retire_promotes_an_anchor_from_an_earlier_session(root: Path) -> None:
    """Q2 cites the equation-of-state run Q1 made: promote it under Q2's finish."""
    with Workspace(root) as ws:
        anchor = _run(ws, "eos", verified=True, session="chat-1")
        own = _run(ws, "vacancy", verified=True, session="chat-2")
        report = retire_session(ws, "chat-2", keep=[anchor, own])
        assert _outcomes(report) == {anchor: "promoted", own: "promoted"}
        assert ws.runs.get(anchor).state.value == "promoted"
        assert ws.runs.history(anchor)[-1].reason == "cited by the finish of session chat-2"
        # Only chat-2's runs are candidates for expiry; chat-1's are not touched.
        other = _run(ws, "smoke", verified=True, session="chat-1")
        retire_session(ws, "chat-2", keep=[anchor])
        assert ws.runs.get(other).state.value == "verified"


def test_retire_skips_a_running_uncited_run_and_the_permanent_ones(root: Path) -> None:
    with Workspace(root) as ws:
        permanent = _run(ws, "kept", verified=True, session="chat-1")
        ws.runs.transition(permanent, "promoted", reason="by hand")
        gone = _run(ws, "old", verified=False, session="chat-1")
        ws.runs.transition(gone, "expired", reason="ttl")
        active = ws.start_run(name="live", session="chat-1")
        live = active.__enter__().id
        try:
            report = retire_session(ws, "chat-1")
        finally:
            active.__exit__(None, None, None)
        assert report["expired"] == []
        details = {s["id"]: s["detail"] for s in report["skipped"]}
        assert details[permanent] == "already permanent"
        assert details[gone] == "already expired"
        assert details[live].startswith("running")
        assert ws.runs.get(live).state.value == "quarantined"


def test_retire_is_idempotent(root: Path) -> None:
    with Workspace(root) as ws:
        cited = _run(ws, "a", verified=True, session="chat-1")
        _run(ws, "b", verified=True, session="chat-1")
        first = retire_session(ws, "chat-1", keep=[cited])
        assert (first["runs_promoted"], first["runs_expired"]) == (1, 1)
        again = retire_session(ws, "chat-1", keep=[cited])
        assert _outcomes(again) == {cited: "already"}
        assert (again["runs_promoted"], again["runs_expired"]) == (0, 0)
        assert [s["detail"] for s in again["skipped"]] == ["already expired"]
        assert again["complete"] is True
        assert len(ws.runs.history(cited)) == len(ws.runs.history(cited))


def test_retire_dry_run_writes_nothing(root: Path) -> None:
    with Workspace(root) as ws:
        cited = _run(ws, "a", verified=True, session="chat-1")
        other = _run(ws, "b", verified=True, session="chat-1")
        before = (len(ws.runs.history(cited)), len(ws.runs.history(other)))
        report = retire_session(ws, "chat-1", keep=[cited], mode="purge", dry_run=True)
        assert report["dry_run"] is True
        assert _outcomes(report) == {cited: "promoted"}
        assert [e["id"] for e in report["expired"]] == [other]
        assert report["purged"]["deleted"] == [] and report["purged"]["freed_bytes"] == 0
        assert ws.runs.get(cited).state.value == "verified"
        assert ws.runs.get(other).state.value == "verified"
        assert (len(ws.runs.history(cited)), len(ws.runs.history(other))) == before


def test_retire_mode_keep_expires_nothing(root: Path) -> None:
    with Workspace(root) as ws:
        cited = _run(ws, "a", verified=True, session="chat-1")
        other = _run(ws, "b", verified=True, session="chat-1")
        report = retire_session(ws, "chat-1", keep=[cited], mode="keep")
        assert report["expired"] == [] and report["skipped"] == []
        assert ws.runs.get(cited).state.value == "promoted"
        assert ws.runs.get(other).state.value == "verified"


def test_retire_purge_keeps_bytes_a_promoted_run_reaches_through_a_purged_sibling(
    root: Path,
) -> None:
    """The cited run consumed what an uncited sibling produced. Purging the
    sibling must not take the shared bytes: the promoted run still names them."""
    from foundation import task

    @task
    def make(seed: int) -> dict[str, int]:
        return {"seed": seed}

    @task
    def consume(data: dict[str, int]) -> int:
        return data["seed"] + 1

    with Workspace(root) as ws:
        with ws.start_run(name="producer", session="chat-1") as producer:
            produced = make(7)
            producer.keep("scratch", b"s" * 60)
            producer.check(lambda: True, name="gate")
        with ws.start_run(name="consumer", session="chat-1") as consumer:
            consume(produced)
            consumer.check(lambda: True, name="gate")
        (shared,) = ws.runs.list_tasks(producer.id)[0].outputs.values()
        scratch = ws.runs.get_artifact(producer.id, "scratch").hash
        assert ws.runs.list_tasks(consumer.id)[0].inputs["data"] == shared
        report = retire_session(ws, "chat-1", keep=[consumer.id], mode="purge")
        assert report["purged"]["deleted"] == [producer.id]
        assert report["purged"]["freed_bytes"] >= 60
        assert ws.artifacts.has(shared)
        assert not ws.artifacts.has(scratch)
        assert [r.name for r in ws.runs.list_runs()] == ["consumer"]
        assert ws.runs.get(consumer.id).state.value == "promoted"


def test_retire_of_a_session_that_made_no_runs_is_an_empty_report(root: Path) -> None:
    with Workspace(root) as ws:
        report = retire_session(ws, "never-ran")
        assert (report["runs_total"], report["kept"], report["expired"]) == (0, [], [])
        assert report["complete"] is True


def test_retire_rejects_an_unknown_mode(root: Path) -> None:
    with Workspace(root) as ws, pytest.raises(ValueError, match="keep, expire, or purge"):
        retire_session(ws, "chat-1", mode="delete")


# -- wait_for_run and liveness ----------------------------------------------------


def test_wait_for_run_returns_at_once_when_the_process_is_gone(tmp_path: Path) -> None:
    """A run whose recorded process died is not waited on: the wait marks it
    failed and returns process_gone without sleeping through its timeout."""
    import subprocess
    import sys
    import time

    from foundation._ops import wait_for_run
    from foundation.models import Run
    from foundation.runtime import Workspace, this_host

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    root = tmp_path / "ws"
    with Workspace(root) as ws:
        run = ws.runs.create(Run(name="killed"))
        ws.runs.set_status(run.id, "running", pid=child.pid, host=this_host())

    started = time.monotonic()
    waited = wait_for_run(root, run_id=run.id, timeout_s=30, poll_s=5)
    assert time.monotonic() - started < 5
    assert waited["outcome"] == "process_gone"
    assert waited["run"].status.value == "failed"
    assert waited["run"].error == (
        f"process {child.pid} on {this_host()} is gone; marked failed by wait_for_run"
    )
    assert waited["progress"].startswith("tasks:")
    # The next wait finds a finished record, not a dead one.
    assert wait_for_run(root, run_id=run.id, timeout_s=1)["outcome"] == "finished"


def test_wait_for_run_names_a_run_on_another_host_instead_of_guessing(
    tmp_path: Path,
) -> None:
    from foundation._ops import wait_for_run
    from foundation.models import Run
    from foundation.runtime import Workspace

    root = tmp_path / "ws"
    with Workspace(root) as ws:
        run = ws.runs.create(Run(name="remote"))
        ws.runs.set_status(run.id, "running", pid=1, host="another-node")
    waited = wait_for_run(root, run_id=run.id, timeout_s=0.2, poll_s=0.1)
    assert waited["outcome"] == "still_running"
    (listed, progress, liveness) = waited["running"][0]
    assert listed.id == run.id and progress.startswith("tasks:")
    assert liveness.startswith("process 1 on another-node, not this host")
    assert "not checked from here" in liveness
    with Workspace(root) as ws:
        assert ws.runs.get(run.id).status.value == "running"


def test_run_commands_collects_the_engine_commands_a_run_resolved(tmp_path: Path) -> None:
    """One entry per distinct command; tasks that share it are counted, cache hits too."""
    from foundation import task

    identity = {
        "engine": "lammps",
        "build": "gpu",
        "command": "mpirun -np {ntasks} lmp -k on g {gpus} -sf kk",
        "provenance": {
            "command": "mpirun -np 1 lmp -k on g 1 -sf kk",
            "envelope": {"ntasks": 1, "threads": 1, "gpus": 1},
        },
        "setup": ["module load lammps"],
        "version": "22 Jul 2025",
    }

    @task(cache_extra=lambda arguments: identity)
    def probe(x):
        return x

    @task(cache_extra=lambda arguments: {"builder": "atomsk", "command": "atomsk"})
    def build(x):
        return x

    @task
    def plain(x):
        return x

    with Workspace(tmp_path / "ws") as ws:
        with ws.start_run(name="commands") as run:
            probe(1)
            probe(2)
            probe(1)
            build(3)
            plain(4)
        entries = run_commands(ws, run.id)
        assert run_commands(ws, run.id) == entries
    assert [e["engine"] for e in entries] == ["lammps", "atomsk"]
    lammps, atomsk = entries
    assert lammps["run_id"] == run.id and lammps["seq"] == 1 and lammps["task"] == "probe"
    assert lammps["tasks"] == 3 and lammps["cache_hits"] == 1
    # The filled line is what ran; the template rides beside it.
    assert lammps["command"] == "mpirun -np 1 lmp -k on g 1 -sf kk"
    assert lammps["template"] == identity["command"]
    assert lammps["setup"] == ["module load lammps"]
    assert lammps["version"] == "22 Jul 2025"
    assert lammps["kokkos"]["enabled"] is True and lammps["kokkos"]["gpus"] == 1
    assert lammps["build"] == "gpu" and "route" not in lammps
    assert atomsk["tasks"] == 1 and "kokkos" not in atomsk and atomsk["setup"] == []
    assert "build" not in atomsk and "template" not in atomsk


def test_cancel_job_fails_the_runs_a_job_started(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run started under $SLURM_JOB_ID is stamped with the job id, so a
    cancel of that job with the workspace fails it. The sandbox render
    carries the variable into the container for exactly this."""
    import stat

    from foundation._ops import cancel_job
    from foundation.errors import IllegalStatusChangeError

    bin_dir = tmp_path / "fake-slurm"
    bin_dir.mkdir()
    script = bin_dir / "scancel"
    script.write_text("#!/bin/sh\ntrue\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")

    # In the job the process dies with the cancel and the block never exits.
    # Here it does exit, and the record is already failed: final, so refused.
    with (
        pytest.raises(IllegalStatusChangeError),
        Workspace(root) as ws,
        ws.start_run(name="relax") as active,
    ):
        assert ws.runs.get(active.id).job_id == "4242"
        summary = cancel_job("4242", workspace=root)
        assert [r["name"] for r in summary["runs_failed"]] == ["relax"]
    with Workspace(root) as ws:
        assert ws.runs.get(active.id).status.value == "failed"


def test_reap_and_cancel_remove_the_scratch_of_the_runs_they_fail(
    root: Path, tmp_path: Path, scratch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup happens at the moment a run is known dead: a reap removes the
    reaped runs' scratch, a cancel removes the cancelled job's runs'
    scratch, and both leave every other directory alone."""
    import stat

    from conftest import seed_scratch, vanished_pid
    from foundation._ops import cancel_job, cancel_lines
    from foundation.runtime import this_host

    bin_dir = tmp_path / "fake-slurm"
    bin_dir.mkdir()
    script = bin_dir / "scancel"
    script.write_text("#!/bin/sh\ntrue\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(tmp_path / "memory"))

    with Workspace(root) as ws:
        killed = ws.runs.create(Run(name="killed"))
        ws.runs.set_status(killed.id, "running", pid=vanished_pid(), host=this_host())
        jobbed = ws.runs.create(Run(name="in-job", job_id="4242"))
        ws.runs.set_status(jobbed.id, "running", pid=1, host="compute-7")
        bystander = seed_scratch(scratch_root, "slab-qe-bystander", marker=False)
        seed_scratch(scratch_root, "slab-qe-killed", run_id=killed.id)
        seed_scratch(scratch_root, "slab-qe-in-job", pid=1, host="compute-7", run_id=jobbed.id)
        assert [r.id for r in ws.reap_dead(caller="test")] == [killed.id]
    assert sorted(p.name for p in scratch_root.iterdir()) == [
        "slab-qe-bystander", "slab-qe-in-job"
    ]
    summary = cancel_job("4242", workspace=root)
    assert [r["name"] for r in summary["runs_failed"]] == ["in-job"]
    assert [Path(p).name for p in summary["scratch_removed"]] == ["slab-qe-in-job"]
    assert any(line.startswith("removed scratch ") for line in cancel_lines(summary))
    assert [p.name for p in scratch_root.iterdir()] == [bystander.name]


def test_retire_purge_mode_removes_the_purged_runs_scratch(
    root: Path, scratch_root: Path
) -> None:
    from conftest import seed_scratch

    with Workspace(root) as ws:
        with ws.start_run(name="shakeout", session="chat-1") as shakeout:
            seed_scratch(scratch_root, "slab-qe-shakeout", run_id=shakeout.id)
        with ws.start_run(name="result", session="chat-1") as result:
            seed_scratch(scratch_root, "slab-qe-result", run_id=result.id)
            result.check(lambda: True, name="gate")
        # A dry run expires nothing, so it purges nothing, so it sweeps nothing.
        dry = retire_session(ws, "chat-1", keep=[result.id], mode="purge", dry_run=True)
        assert dry["purged"]["scratch_removed"] == []
        assert sorted(p.name for p in scratch_root.iterdir()) == [
            "slab-qe-result", "slab-qe-shakeout"
        ]
        report = retire_session(ws, "chat-1", keep=[result.id], mode="purge")
        assert report["purged"]["deleted"] == [shakeout.id]
        assert [Path(p).name for p in report["purged"]["scratch_removed"]] == ["slab-qe-shakeout"]
    # The promoted run's scratch is a completed run's: the backstop sweep
    # takes it later, but the retire removes only what it purged.
    assert [p.name for p in scratch_root.iterdir()] == ["slab-qe-result"]


def test_a_job_and_a_child_never_inherit_the_run_id(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workflow script that submits a job or launches a child runs under a
    run of its own, and that run completes as soon as the script returns.
    Its id must not stamp the scratch the job or the child makes, or a
    sweep would remove a live calculation's directory."""
    import stat
    import subprocess

    from foundation._ops import launch_child, submit_job
    from slab.config import HpcConfig

    monkeypatch.setenv("SLAB_RUN_ID", "the-submitter")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/bin/sh\necho 777\n")
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
    hpc = HpcConfig.model_validate({"default_partition": "cpu", "partitions": {"cpu": {}}})
    job = submit_job(root, hpc=hpc, command="python wf.py", name="wf", session="chat-1")
    script = Path(job["script_path"]).read_text()
    assert "unset SLAB_RUN_ID\n" in script
    assert script.index("unset SLAB_RUN_ID") < script.index("export SLAB_SESSION=chat-1")

    seen: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> object:
        seen["env"] = kwargs["env"]
        raise OSError("stopped here: the environment was the question")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workflow = tmp_path / "wf.py"
    workflow.write_text("print('ok')\n")
    with Workspace(root) as ws:
        held = ws.reserve(ntasks=1, threads=1)
    with pytest.raises(FoundationError):
        launch_child(root, workflow, reservation=held, wait=False)
    assert "SLAB_RUN_ID" not in seen["env"]  # type: ignore[operator]


# -- dry run ---------------------------------------------------------------------------


DRY_RUN_SCRIPT = """\
import os
from foundation import check, task

print("workspace", os.environ["SLAB_WORKSPACE"])

@task
def double(x):
    return 2 * x

y = double(21)

@check
def sane():
    return y == 42
"""

DRY_RUN_KEYERROR_SCRIPT = """\
from foundation import task

@task
def double(x):
    return 2 * x

y = double(21)
result = {}
print(result["thermo"]["Temp"])
"""


def _store_counts(root: Path) -> tuple[int, int]:
    with Workspace(root) as ws:
        return len(ws.runs.list_runs(limit=1000)), sum(1 for _ in ws.artifacts.hashes())


def test_dry_run_touches_no_store_and_removes_the_throwaway(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rehearsal runs the script to its end in a throwaway workspace: the
    real store's run and blob counts do not move, the throwaway directory is
    gone afterwards, and $SLAB_WORKSPACE is what it was."""
    launch_script(root, _write(tmp_path, "real.py", HAPPY_SCRIPT))
    before = _store_counts(root)
    monkeypatch.setenv("SLAB_WORKSPACE", str(root))
    script = _write(tmp_path, "rehearsal.py", DRY_RUN_SCRIPT)
    report = launch_script(root, script, dry_run=True, capture_output=True)
    assert report["dry_run"] is True
    assert report["reached_end"] is True
    assert report["traceback"] is None
    assert report["lammps"] == []
    assert report["checks"] == [{"name": "sane", "passed": True, "message": "returned True"}]
    assert "LAMMPS integrated no steps" in report["checks_note"]
    assert report["outputs"] == []
    assert "run_id" not in report
    (line,) = [ln for ln in report["output"].splitlines() if ln.startswith("workspace ")]
    throwaway = Path(line.split(" ", 1)[1])
    assert throwaway.name.startswith("slab-dry-run-")
    assert throwaway != root and not throwaway.exists()
    assert _store_counts(root) == before
    import os

    assert os.environ["SLAB_WORKSPACE"] == str(root)


def test_dry_run_reports_a_failure_after_the_task(root: Path, tmp_path: Path) -> None:
    script = _write(tmp_path, "keyerror.py", DRY_RUN_KEYERROR_SCRIPT)
    report = launch_script(root, script, dry_run=True)
    assert report["reached_end"] is False
    assert "KeyError: 'thermo'" in report["traceback"]
    assert report["checks"] == []
    assert not (root / "runs.db").exists()  # the real store was never opened


def test_dry_run_uses_the_configured_scratch_root(
    scratch_root: Path, tmp_path: Path
) -> None:
    script = _write(tmp_path, "where.py", DRY_RUN_SCRIPT)
    report = launch_script(tmp_path / "ws", script, dry_run=True, capture_output=True)
    (line,) = [ln for ln in report["output"].splitlines() if ln.startswith("workspace ")]
    throwaway = Path(line.split(" ", 1)[1])
    assert throwaway.parent == scratch_root
    assert list(scratch_root.iterdir()) == []


def test_dry_run_of_a_missing_script(root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        launch_script(root, root / "ghost.py", dry_run=True)


def test_parse_dry_run_report_takes_the_last_marker() -> None:
    from foundation._ops import DRY_RUN_MARKER, parse_dry_run_report

    text = f"{DRY_RUN_MARKER}\n{{}}\nnoise\n{DRY_RUN_MARKER}\n{{\"reached_end\": false}}\n"
    assert parse_dry_run_report(text) == {"reached_end": False}
    assert parse_dry_run_report(f"{DRY_RUN_MARKER}\nnot json") is None


_DYING_LMP = """\
#!{python}
import sys
args = sys.argv[1:]
if "-h" in args:
    print("Large-scale Atomic/Molecular Massively Parallel Simulator - 22 Jul 2025")
    sys.exit(0)
with open(args[args.index("-log") + 1], "w") as log:
    log.write("LAMMPS (22 Jul 2025)\\npair_style nonsense\\n")
    log.write("ERROR: Unrecognized pair style 'nonsense' (src/force.cpp:275)\\n")
print("LAMMPS (22 Jul 2025)")
print("ERROR: Unrecognized pair style 'nonsense' (src/force.cpp:275)")
sys.exit(1)
"""


def _failing_dry_run(root: Path, tmp_path: Path, session: str | None = None) -> dict:
    """Rehearse a script whose one run_lammps fails; return the report."""
    import sys

    lmp = tmp_path / "dying-lmp"
    lmp.write_text(_DYING_LMP.format(python=sys.executable))
    lmp.chmod(0o755)
    script = _write(
        tmp_path,
        "dying.py",
        "from foundation.tasks import run_lammps\n"
        f"run_lammps('units metal\\npair_style nonsense\\n', label='md', command={str(lmp)!r})\n",
    )
    return launch_script(root, script, dry_run=True, session=session)


def test_a_failed_dry_run_leaves_a_record_in_the_real_workspace(
    root: Path, tmp_path: Path
) -> None:
    """The throwaway workspace is gone, and the failed LAMMPS files are not:
    the report names a dry-<stamp> record under the real workspace, and its
    log is the log LAMMPS wrote. The real store is still never opened."""
    from foundation._ops import dry_run_record

    report = _failing_dry_run(root, tmp_path, session="20260914-101500-4242")
    assert report["lammps"][0]["outcome"].startswith("ERROR: Unrecognized pair style")
    record = report["record"]
    assert record["id"].startswith("dry-")
    assert record["files"] == ["md-failed.in", "md-failed.log", "md-failed.screen"]
    directory, row = dry_run_record(root, record["id"])
    assert str(directory) == record["path"] and directory.parent == root / "dry-runs"
    assert row["kind"] == "dry_run" and row["stamp"] == record["id"].removeprefix("dry-")
    assert row["session"] == "20260914-101500-4242"
    assert "ERROR: Unrecognized pair style" in (directory / "md-failed.log").read_text()
    assert not (root / "runs.db").exists()
    with pytest.raises(FoundationError, match="no dry-run record dry-nope"):
        dry_run_record(root, "dry-nope")


def test_a_clean_dry_run_keeps_no_record(root: Path, tmp_path: Path) -> None:
    report = launch_script(root, _write(tmp_path, "clean.py", DRY_RUN_SCRIPT), dry_run=True)
    assert report["record"] is None
    assert not root.exists()


def test_purge_removes_the_dry_run_records_but_the_newest_conversations(
    root: Path, tmp_path: Path
) -> None:
    """A record is readable for the rest of its session: purge keeps the
    newest conversation's records, like its transcript, and removes the rest;
    --all-sessions removes them all."""
    from foundation._ops import dry_run_records
    from slab_stack._ops import purge_inventory

    current = "20260914-101500-4242"
    old = _failing_dry_run(root, tmp_path, session="20260913-090000-1111")["record"]
    mine = _failing_dry_run(root, tmp_path, session=current)["record"]
    sessions = root / "mason" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / f"{current}.jsonl").write_text("{}\n")

    def purge(*, dry_run: bool, all_sessions: bool = False):
        return purge_inventory(
            root,
            all_sessions=all_sessions,
            active=frozenset(),
            job_id_of=lambda name: None,
            dry_run=dry_run,
        )

    inventory = purge(dry_run=True)
    (category,) = [c for c in inventory.categories if c.name == "dry-run records"]
    assert category.items == [f"dry-runs/{old['id'].removeprefix('dry-')}"]
    assert category.bytes > 0
    assert [k.reason for k in inventory.kept if k.kind == "dry-run record"] == [
        "the newest conversation (--all-sessions removes it too)"
    ]
    assert len(dry_run_records(root)) == 2  # a dry run deletes nothing
    purge(dry_run=False)
    assert [row["id"] for _, row in dry_run_records(root)] == [mine["id"]]
    purge(dry_run=False, all_sessions=True)
    assert dry_run_records(root) == []


def test_launch_child_dry_run_returns_the_report_and_releases_the_reservation(
    root: Path, tmp_path: Path
) -> None:
    """The child rehearses under the slice, prints the report, and the
    reservation is gone when the parent's wait returns; no run exists."""
    from foundation._ops import launch_child

    script = _write(tmp_path, "child.py", DRY_RUN_SCRIPT)
    with Workspace(root) as ws:
        held = ws.reserve(ntasks=1, threads=1)
    result = launch_child(root, script, reservation=held, dry_run=True, wait=True)
    assert result["exit_code"] == 0, result["output"]
    assert result["dry_run"] is True and result["reached_end"] is True
    assert result["checks"][0]["passed"] is True
    assert result["reservation"] == held.id
    assert "--dry-run" in result["command"]
    with Workspace(root) as ws:
        assert ws.runs.list_runs() == []
        assert ws.runs.list_reservations() == []
