"""Session leases: who owns a run while it runs, and who settles it when nobody does.

A run used to be judged by its pid, its host, and its scheduler job, and
none of those answers "is the session that started this still working?"
from another process. A lease does. These tests pin the lease's life (open,
beat, close), the three verdicts a reader draws from it, and the places
that read it: the sweep, the wait, the prompt, the batch script, and the
operator's commands.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from foundation.models import Run, SessionLease, utcnow
from foundation.runtime import (
    Workspace,
    describe_liveness,
    job_deadline,
    lease_verdict,
    run_liveness,
    this_host,
)
from mason.config import MasonConfig
from mason.session import MasonSession

runner = CliRunner()


def _session(tmp_path: Path, **agent: object) -> MasonSession:
    config = MasonConfig.model_validate({"agent": {"model": "fake", **agent}})
    return MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent, auto_approve=True
    )


def _running(ws: Workspace, *, session: str | None = None, **stamp: object) -> Run:
    """One run at status running, stamped as the caller asks."""
    run = ws.runs.create(Run(name="left", session=session, job_id=stamp.pop("job_id", None)))
    return ws.runs.set_status(run.id, "running", **stamp)


# -- the lease's life ----------------------------------------------------------


def test_a_session_opens_a_lease_and_carries_the_jobs_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lease names the session, the harness, the agent, and this job; the
    deadline comes from the environment the sandbox script exports."""
    ends = datetime.now(UTC).replace(microsecond=0) + timedelta(hours=2)
    monkeypatch.setenv("SLAB_JOB_END", str(ends.timestamp()))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    session = _session(tmp_path)
    session.open_lease()
    with Workspace(session.workspace_root) as ws:
        lease = ws.runs.get_lease(session.session_id)
    assert lease is not None
    assert (lease.harness, lease.agent, lease.job_id) == ("mason", "pi", "4242")
    assert lease.deadline_at == ends
    assert lease.ended_at is None
    session.close("finished")


def test_job_deadline_reads_the_export_then_the_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SLAB_JOB_END wins, because --cleanenv strips the scheduler's own."""
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    monkeypatch.delenv("SLAB_JOB_END", raising=False)
    assert job_deadline() is None
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1789000000")
    assert job_deadline() == datetime.fromtimestamp(1789000000, tz=UTC)
    monkeypatch.setenv("SLAB_JOB_END", "2026-09-17T07:31:00+00:00")
    assert job_deadline() == datetime(2026, 9, 17, 7, 31, tzinfo=UTC)
    monkeypatch.setenv("SLAB_JOB_END", "N/A")  # squeue's answer for a job it cannot place
    assert job_deadline() == datetime.fromtimestamp(1789000000, tz=UTC)


def test_the_heartbeat_thread_and_a_wait_poll_both_beat(tmp_path: Path) -> None:
    """The thread stamps the lease on its own interval; a wait, which is the
    longest a session goes without a model call, stamps it at every poll."""
    from foundation import _ops

    session = _session(tmp_path, lease_beat_s=1)
    session.open_lease()
    with Workspace(session.workspace_root) as ws:
        opened = ws.runs.get_lease(session.session_id)
    assert opened is not None
    beat_thread = session._beat_thread
    assert beat_thread is not None and beat_thread.is_alive()
    beat_thread.join(timeout=3.0)  # the daemon runs until the lease closes
    with Workspace(session.workspace_root) as ws:
        beaten = ws.runs.get_lease(session.session_id)
    assert beaten is not None and beaten.heartbeat_at > opened.heartbeat_at

    with Workspace(session.workspace_root) as ws:
        ws.runs.beat_lease(session.session_id, at=utcnow() - timedelta(hours=1))
    _ops.wait_for_run(
        Path(session.workspace_root), session=session.session_id, timeout_s=0.0, grace_s=0.0
    )
    with Workspace(session.workspace_root) as ws:
        waited = ws.runs.get_lease(session.session_id)
    assert waited is not None
    assert (utcnow() - waited.heartbeat_at).total_seconds() < 60
    session.close("finished")


def test_closing_the_session_writes_the_reason_and_the_transcript_event(
    tmp_path: Path,
) -> None:
    """Every exit runs through close: the lease carries why, and so does the
    transcript, where --resume and 'slab mason report' read it."""
    session = _session(tmp_path)
    session.open_lease()
    session.close("the session stopped: finish")
    with Workspace(session.workspace_root) as ws:
        lease = ws.runs.get_lease(session.session_id)
    assert lease is not None and lease.end_reason == "the session stopped: finish"
    assert lease.ended_at is not None
    events = [
        json.loads(line)
        for line in session.transcript_path.read_text().splitlines()
        if line.strip()
    ]
    (ended,) = [e for e in events if e["type"] == "session_end"]
    assert ended["reason"] == "the session stopped: finish"
    assert ended["runs_ended"] == []
    # The beat thread stops with the lease, so nothing keeps writing.
    assert session._beat_thread is None


def test_a_session_ends_the_runs_it_was_still_executing(tmp_path: Path) -> None:
    """A run of the ending session is failed naming the session and the
    reason, and its process group is gone."""
    session = _session(tmp_path)
    session.open_lease()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    with Workspace(session.workspace_root) as ws:
        run = _running(ws, session=session.session_id, pid=child.pid, host=this_host())
    ended = session.close("terminated (SIGTERM)")
    assert [r.id for r in ended] == [run.id]
    with Workspace(session.workspace_root) as ws:
        settled = ws.runs.get(run.id)
    assert settled.status.value == "failed"
    assert settled.error == (
        f"session {session.session_id} ended (terminated (SIGTERM)) while the run was running"
    )
    # process_alive reaps the child on its way, so the status is gone with
    # it; that the process no longer exists is the fact under test.
    from slab.scratch import process_alive

    assert not process_alive(child.pid)


def test_sigterm_closes_the_lease_of_a_batch_session(tmp_path: Path) -> None:
    """The loop installs the handler, so a job that signals its processes
    before the time limit leaves a closed lease and no running run."""
    project = tmp_path / "project"
    project.mkdir()
    source = Path(__file__).resolve().parents[1] / "src"
    script = tmp_path / "job.py"
    script.write_text(
        "import os, signal, sys\n"
        f"sys.path.insert(0, {str(source)!r})\n"
        "from mason.config import MasonConfig\n"
        "from mason.loop import Mason\n"
        "from mason.session import MasonSession\n"
        "config = MasonConfig.model_validate({'agent': {'model': 'fake'}})\n"
        f"session = MasonSession({str(project)!r}, "
        f"workspace_root={str(tmp_path / 'ws')!r}, agent=config.agent)\n"
        "Mason(session, client=object())\n"
        "print(session.session_id, flush=True)\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
    )
    finished = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False
    )
    assert finished.returncode == -signal.SIGTERM, finished.stderr
    session_id = finished.stdout.strip().splitlines()[-1]
    with Workspace(tmp_path / "ws") as ws:
        lease = ws.runs.get_lease(session_id)
    assert lease is not None and lease.end_reason == "terminated (SIGTERM)"


# -- the verdicts a reader draws -----------------------------------------------


def _lease(**fields: object) -> SessionLease:
    return SessionLease(id="s1", host="n1", pid=7, **fields)


def test_the_three_dead_verdicts_and_the_live_one() -> None:
    past = utcnow() - timedelta(minutes=30)
    assert lease_verdict(_lease()) is None
    assert lease_verdict(_lease(ended_at=past, end_reason="time limit")) == "session-ended"
    assert lease_verdict(_lease(deadline_at=past)) == "deadline-passed"
    assert lease_verdict(_lease(heartbeat_at=past), silence_s=600) == "session-silent"
    # A lease that beat a moment ago is alive whatever its age.
    assert lease_verdict(_lease(started_at=past), silence_s=600) is None


def test_a_dead_lease_settles_its_runs_from_another_host_and_job(
    tmp_path: Path, scratch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case this exists for: the session died with its job two days ago,
    and the next session reads its runs from another host, in another job,
    with no scheduler to ask. Each run is failed, its reservation released,
    and its scratch removed."""
    from conftest import seed_scratch

    monkeypatch.setenv("SLURM_JOB_ID", "999")  # another job entirely
    long_ago = utcnow() - timedelta(days=2)
    with Workspace(tmp_path / "ws") as ws:
        ws.runs.open_lease("ended", host="other-node", pid=11, job_id="7")
        ws.runs.end_lease("ended", reason="time limit")
        ws.runs.open_lease(
            "overdue", host="other-node", pid=12, job_id="7", deadline_at=long_ago
        )
        ws.runs.open_lease("silent", host="other-node", pid=13, job_id="7")
        ws.runs.beat_lease("silent", at=long_ago)
        for name, pid in (("ended", 11), ("overdue", 12), ("silent", 13)):
            pending = ws.runs.create(Run(name=name, session=name, job_id="7"))
            held = ws.reserve(ntasks=1, budget=_cpu_budget(), host="other-node")
            run = ws.runs.claim_reservation(held.id, pending.id, host="other-node", pid=pid)
            assert run.run_id == pending.id
            seed_scratch(scratch_root, f"slab-{name}", run_id=pending.id)

        settled = ws.reap_dead(caller="the next session")
        assert sorted(str(r.session) for r in settled) == ["ended", "overdue", "silent"]
        errors = {str(r.session): str(r.error) for r in settled}
        assert "ended (time limit) at" in errors["ended"]
        assert "job ended at" in errors["overdue"]
        assert "last beat at" in errors["silent"]
        assert all(e.endswith("marked failed by the next session") for e in errors.values())
        assert ws.runs.list_reservations() == []
    assert not list(scratch_root.glob("slab-*"))


def _cpu_budget():
    from slab.resources import Budget

    return Budget(cpus=(0, 1, 2, 3), gpus=())


def test_a_beating_lease_keeps_its_runs_and_a_run_without_one_is_judged_as_before(
    tmp_path: Path,
) -> None:
    """A lease says nothing while it beats, so the run falls through to the
    verdicts that judge a process; a run with no lease row is untouched."""
    with Workspace(tmp_path / "ws") as ws:
        ws.runs.open_lease("live", host="other-node", pid=11, job_id="7")
        kept = _running(ws, session="live", pid=11, host="other-node", job_id="7")
        away = _running(ws, session=None, pid=1, host="other-node")
        assert ws.reap_dead(caller="t") == []
        assert ws.runs.get(kept.id).status.value == "running"
        assert ws.runs.get(away.id).status.value == "running"
        leases = ws.runs.leases()
    assert run_liveness(kept, job=None, leases=leases) == "other-job"
    assert run_liveness(away, job=None, leases=leases) == "elsewhere"


def test_the_scheduler_outranks_a_beating_lease() -> None:
    """A lease cannot beat from a job the scheduler calls ended: the clock is
    wrong, and the scheduler is right."""
    from slab.hpc import JobState

    run = Run(status="running", pid=11, host="n1", job_id="7", session="s1")
    leases = {"s1": _lease(job_id="7")}
    assert run_liveness(run, job=None, leases=leases) == "other-job"
    assert (
        run_liveness(run, job=None, leases=leases, job_states={"7": JobState.TIMEOUT})
        == "job-ended"
    )


def test_the_liveness_phrase_names_the_session_and_the_time() -> None:
    at = datetime(2026, 9, 17, 7, 31, tzinfo=UTC)
    run = Run(status="running", pid=11, host="n1", session="s1")
    leases = {"s1": _lease(ended_at=at, end_reason="time limit")}
    assert describe_liveness(run, host="n1", leases=leases) == (
        "session s1 ended (time limit) at 07:31 UTC"
    )
    overdue = {"s1": _lease(deadline_at=at)}
    assert describe_liveness(run, host="n1", leases=overdue) == (
        "session s1's job ended at 07:31 UTC; the process died with it"
    )


# -- the readers ---------------------------------------------------------------


def test_wait_for_run_settles_a_dead_sessions_run_instead_of_waiting(
    tmp_path: Path,
) -> None:
    """The 75 minutes a real lead spent waiting on a run whose job had ended
    two days earlier: the wait now names the verdict and returns at once."""
    from foundation import _ops

    root = tmp_path / "ws"
    with Workspace(root) as ws:
        ws.runs.open_lease("gone", host="other-node", pid=11, job_id="7")
        ws.runs.end_lease("gone", reason="time limit")
        run = _running(ws, session="gone", pid=11, host="other-node", job_id="7")
    waited = _ops.wait_for_run(root, run_id=run.id, session="mine", timeout_s=30.0)
    assert waited["outcome"] == "settled"
    assert waited["liveness"].startswith("session gone ended (time limit) at ")
    assert waited["run"].status.value == "failed"


def test_wait_for_run_caps_at_the_jobs_end_and_names_a_live_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wait past the job's end never returns, so it is capped at the end
    less the grace the batch script is signalled in. A run of another live
    session is polled once and reported with its owner."""
    from foundation import _ops

    monkeypatch.setenv("SLAB_JOB_END", str((datetime.now(UTC) + timedelta(minutes=4)).timestamp()))
    root = tmp_path / "ws"
    with Workspace(root) as ws:
        ws.runs.open_lease("theirs", host="other-node", pid=11, job_id="7")
        run = _running(ws, session="theirs", pid=11, host="other-node", job_id="7")
    waited = _ops.wait_for_run(
        root, run_id=run.id, session="mine", timeout_s=3600.0, poll_s=0.05
    )
    assert waited["outcome"] == "still_running"
    assert waited["capped_by"] == "the job's end"
    assert waited["timeout_s"] < 120  # four minutes less the three-minute grace
    assert waited["owner"] == "session theirs (mason) is still working on it"


def test_the_environment_block_and_the_free_line_carry_the_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent sizes the last wave against this."""
    from mason.prompts import environment_block, free_hint

    monkeypatch.setenv(
        "SLAB_JOB_END", str((datetime.now(UTC) + timedelta(hours=2, minutes=11)).timestamp())
    )
    session = _session(tmp_path)
    block = environment_block(session)
    assert "this job ends at " in block and "(2 h 10 min left)" in block
    assert "min left" not in free_hint(session)  # over an hour: the block says it once

    monkeypatch.setenv("SLAB_JOB_END", str((datetime.now(UTC) + timedelta(minutes=41)).timestamp()))
    session._deadline = ...  # read again
    assert "40 min left.]" in free_hint(session)

    monkeypatch.delenv("SLAB_JOB_END")
    session._deadline = ...
    assert "this job ends at" not in environment_block(session)


def test_the_sandbox_script_signals_itself_and_settles_the_jobs_leases(
    tmp_path: Path,
) -> None:
    """The container may die before the session closes its own lease, so the
    batch shell carries the fallback: a TERM trap that ends the job's leases."""
    from mason.config import AgentConfig
    from mason.sandbox import render_sandbox_script
    from slab.config import HpcConfig, SlabConfig

    agent = AgentConfig.model_validate(
        {"provider": "openai", "endpoint": "http://x/v1", "sandbox": {"image": "/x.sif"}}
    )
    hpc = HpcConfig.model_validate(
        {"default_partition": "cpu", "partitions": {"cpu": {"time_limit": "04:00:00"}}}
    )
    script, _warnings, _context = render_sandbox_script(
        agent, hpc, SlabConfig(), tmp_path, tmp_path, "goal", toml_path=tmp_path / "slab.toml"
    )
    assert "#SBATCH --signal=B:TERM@180" in script
    assert 'SLAB_JOB_END="${SLURM_JOB_END_TIME:-}"' in script
    assert 'squeue -h -j "$SLURM_JOB_ID" -o %e' in script
    assert 'sessions end --job "$SLURM_JOB_ID" --reason "time limit"' in script
    assert '--env SLAB_JOB_END="$SLAB_JOB_END"' in script
    written = tmp_path / "job.sh"
    written.write_text(script)
    checked = subprocess.run(
        ["bash", "-n", str(written)], capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, checked.stderr


# -- the operator's view -------------------------------------------------------


def test_sessions_list_end_and_sweep(tmp_path: Path) -> None:
    from foundation.cli import sessions_app

    root = tmp_path / "ws"
    with Workspace(root) as ws:
        ws.runs.open_lease("working", host="n1", pid=os.getpid(), agent="pi", job_id="7")
        _running(ws, session="working", pid=os.getpid(), host="n1", job_id="7")
        ws.runs.open_lease("silent", host="n1", pid=11, harness="mcp")
        ws.runs.beat_lease("silent", at=utcnow() - timedelta(hours=2))
        stranded = _running(ws, session="silent", pid=11, host="other-node")

    listed = runner.invoke(sessions_app, ["list", "-w", str(root)])
    assert listed.exit_code == 0, listed.output
    rows = {line.split()[0]: line.split() for line in listed.output.splitlines()[1:]}
    assert rows["working"][1:3] == ["mason", "pi"]
    assert rows["working"][3] == "7" and rows["working"][-1] == "alive"
    assert rows["silent"][1] == "mcp" and rows["silent"][-1] == "session-silent"
    assert rows["silent"][-2] == "1"  # one run still running behind it

    swept = runner.invoke(sessions_app, ["sweep", "-w", str(root)])
    assert swept.exit_code == 0, swept.output
    assert "1 run(s) settled" in swept.output
    with Workspace(root) as ws:
        assert ws.runs.get(stranded.id).status.value == "failed"

    ended = runner.invoke(
        sessions_app, ["end", "working", "-w", str(root), "--reason", "the operator said so"]
    )
    assert ended.exit_code == 0, ended.output
    assert "ended working (the operator said so)" in ended.output
    assert "while the run was running" in ended.output
    with Workspace(root) as ws:
        lease = ws.runs.get_lease("working")
        assert lease is not None and lease.end_reason == "the operator said so"
        assert ws.runs.list_runs(status="running") == []


def test_sessions_end_by_job_closes_every_lease_of_that_job(tmp_path: Path) -> None:
    """What the batch script's TERM trap runs when the job is out of time."""
    from foundation.cli import sessions_app

    root = tmp_path / "ws"
    with Workspace(root) as ws:
        ws.runs.open_lease("a", host="n1", pid=11, job_id="7")
        ws.runs.open_lease("b", host="n1", pid=12, job_id="7")
        ws.runs.open_lease("elsewhere", host="n1", pid=13, job_id="8")
    result = runner.invoke(
        sessions_app, ["end", "-w", str(root), "--job", "7", "--reason", "time limit"]
    )
    assert result.exit_code == 0, result.output
    with Workspace(root) as ws:
        assert [lease.end_reason for lease in ws.runs.list_leases(job_id="7")] == [
            "time limit",
            "time limit",
        ]
        other = ws.runs.get_lease("elsewhere")
        assert other is not None and other.ended_at is None
    again = runner.invoke(sessions_app, ["end", "-w", str(root), "--job", "9"])
    assert "no open lease for job 9" in again.output


def test_the_doctor_reports_runs_left_by_a_session_that_is_over(tmp_path: Path) -> None:
    from slab_stack import doctor

    root = tmp_path / "ws"
    with Workspace(root) as ws:
        ws.runs.open_lease("gone", host="n1", pid=11)
        ws.runs.end_lease("gone", reason="time limit")
        _running(ws, session="gone", pid=11, host="other-node")
    mark, text = doctor._leases_row(root)  # type: ignore[misc]
    assert mark == "="
    assert "1 session(s) over with 1 run(s) still running" in text
    with Workspace(root) as ws:
        ws.reap_dead(caller="t")
    mark, text = doctor._leases_row(root)  # type: ignore[misc]
    assert mark == "+" and "no run left by a session that is over" in text


def test_a_finished_run_closes_its_lease_and_the_report_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'slab mason run' ends its lease whatever stopped it, and
    'slab mason report' reads the reason out of the transcript."""
    import json as _json

    from mason.cli import app as mason_app
    from mason.client import ChatReply, ToolCall

    class _Finishing:
        def chat(self, messages: object, tools: object = None, **options: object) -> ChatReply:
            return ChatReply(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="f1",
                        name="finish",
                        arguments={"report": "done"},
                        arguments_raw=_json.dumps({"report": "done"}),
                    ),
                ),
                prompt_tokens=50,
                completion_tokens=10,
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mason.loop.client_from_config", lambda agent, keys=None: _Finishing())
    root = tmp_path / ".slab"
    result = runner.invoke(mason_app, ["run", "measure a0", "-w", str(root)])
    assert result.exit_code == 0, result.output

    with Workspace(root) as ws:
        (lease,) = ws.runs.list_leases()
    assert lease.harness == "mason"
    assert lease.end_reason == "the session stopped: finish"

    reported = runner.invoke(mason_app, ["report", "-w", str(root)])
    assert reported.exit_code == 0, reported.output
    assert "session ended: the session stopped: finish" in reported.output


def test_the_wait_tool_tells_the_agent_the_run_was_settled(tmp_path: Path) -> None:
    """What the lead reads instead of waiting: the verdict, and what to do."""
    import json as _json

    from mason.client import ToolCall
    from mason.tools import build_toolbox

    session = _session(tmp_path)
    with Workspace(session.workspace_root) as ws:
        ws.runs.open_lease("gone", host="other-node", pid=11, job_id="7")
        ws.runs.end_lease("gone", reason="time limit")
        run = _running(ws, session="gone", pid=11, host="other-node", job_id="7")
    box = build_toolbox(session)
    answer = box.dispatch(
        ToolCall(
            id="w1",
            name="wait_for_run",
            arguments={"run_id": run.id, "timeout_s": 30},
            arguments_raw=_json.dumps({"run_id": run.id, "timeout_s": 30}),
        )
    )
    assert "this run is over: session gone ended (time limit) at " in answer
    assert "launch again if the work is still wanted" in answer
