"""CLI tests for slab-stack: the destructive pair, exercised on real stores.

Every test seeds a real workspace (SQLite store, CAS, transcript files, job
files) and then runs the command through the CliRunner — no mocked layers,
because what these commands must get right is exactly what they touch.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from foundation.models import Run
from foundation.runtime import Workspace
from slab.hpc import SchedulerNotAvailableError
from slab_stack.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic default: this machine has no queue, so nothing is active."""

    def unavailable() -> frozenset[str]:
        raise SchedulerNotAvailableError("no squeue in tests")

    monkeypatch.setattr("slab_stack.cli.active_job_ids", unavailable)


def _seed_runs(root: Path) -> dict[str, str]:
    """One promoted run sharing a blob with one quarantined run + scratch."""
    with Workspace(root) as ws:
        keep = ws.runs.create(Run(name="keep-me"))
        gone = ws.runs.create(Run(name="failed-test"))
        shared = ws.artifacts.put_bytes(b"shared structure")
        scratch = ws.artifacts.put_bytes(b"wavecar")
        ws.runs.add_artifact(
            keep.id, name="s", role="terminal", hash=shared, size_bytes=16
        )
        ws.runs.add_artifact(
            gone.id, name="s", role="terminal", hash=shared, size_bytes=16
        )
        ws.runs.add_artifact(
            gone.id, name="w", role="intermediate", hash=scratch, size_bytes=7
        )
        ws.runs.transition(keep.id, "promoted", force=True)
        return {"keep": keep.id, "gone": gone.id, "shared": shared, "scratch": scratch}


def _seed_files(root: Path) -> None:
    sessions = root / "mason" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "20260826-120000-111.jsonl").write_text("{}\n")
    (sessions / "20260826-120000-111-crystal-1.jsonl").write_text("{}\n")
    (sessions / "20260827-090000-222.jsonl").write_text("{}\n")
    jobs = root / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "cu-relax-1244113.sbatch").write_text("#!/bin/bash\n")
    (jobs / "cu-relax-1244113.out").write_text("done\n")


def test_fast_forward_expires_everything_unpromoted(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    ids = _seed_runs(root)
    result = runner.invoke(app, ["fast-forward", "-w", str(root)])
    assert result.exit_code == 0
    assert "1 run(s) fast-forwarded to expired" in result.output
    with Workspace(root) as ws:
        assert ws.runs.get(ids["gone"]).state.value == "expired"
        assert ws.runs.get(ids["keep"]).state.value == "promoted"


def test_purge_deletes_rows_bytes_transcripts_and_job_files(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    ids = _seed_runs(root)
    _seed_files(root)
    assert runner.invoke(app, ["fast-forward", "-w", str(root)]).exit_code == 0
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes"])
    assert result.exit_code == 0
    with Workspace(root) as ws:
        assert [r.id for r in ws.runs.list_runs()] == [ids["keep"]]
        assert ws.artifacts.has(ids["shared"])  # the survivor still references it
        assert not ws.artifacts.has(ids["scratch"])
    # The newest conversation survives for --resume; the older one is gone
    # together with its delegation sibling. Job files are swept.
    remaining = sorted(p.name for p in (root / "mason" / "sessions").iterdir())
    assert remaining == ["20260827-090000-222.jsonl"]
    assert list((root / "jobs").iterdir()) == []
    assert "kept the newest conversation" in result.output


def test_purge_dry_run_deletes_nothing(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _seed_runs(root)
    _seed_files(root)
    assert runner.invoke(app, ["fast-forward", "-w", str(root)]).exit_code == 0
    result = runner.invoke(app, ["purge", "-w", str(root), "--dry-run"])
    assert result.exit_code == 0
    assert "would delete 1 run(s)" in result.output
    with Workspace(root) as ws:
        assert len(ws.runs.list_runs()) == 2
    assert len(list((root / "mason" / "sessions").iterdir())) == 3
    assert len(list((root / "jobs").iterdir())) == 2


def test_purge_all_sessions_removes_the_newest_too(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _seed_files(root)
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes", "--all-sessions"])
    assert result.exit_code == 0
    assert list((root / "mason" / "sessions").iterdir()) == []


def test_purge_confirmation_defaults_to_no(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _seed_runs(root)
    assert runner.invoke(app, ["fast-forward", "-w", str(root)]).exit_code == 0
    result = runner.invoke(app, ["purge", "-w", str(root)], input="n\n")
    assert result.exit_code != 0
    with Workspace(root) as ws:
        assert len(ws.runs.list_runs()) == 2  # nothing was deleted


def test_purge_keeps_files_of_jobs_still_in_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".slab"
    _seed_files(root)
    monkeypatch.setattr(
        "slab_stack.cli.active_job_ids", lambda: frozenset({"1244113"})
    )
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes"])
    assert result.exit_code == 0
    kept = sorted(p.name for p in (root / "jobs").iterdir())
    assert kept == ["cu-relax-1244113.out", "cu-relax-1244113.sbatch"]


def test_purge_never_touches_the_serve_record_or_its_job(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    mason = root / "mason"
    mason.mkdir(parents=True)
    (mason / "endpoint.json").write_text(
        json.dumps({"endpoint": "http://node:8000/v1", "model": "m", "job_id": "777"})
    )
    (mason / "serve-777.sbatch").write_text("#!/bin/bash\n")
    (mason / "serve-777.out").write_text("serving\n")
    (mason / "serve-500.out").write_text("an older, finished server\n")
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes"])
    assert result.exit_code == 0
    remaining = sorted(p.name for p in mason.iterdir())
    # The record and the recorded job's files stay; the finished one goes.
    assert remaining == ["endpoint.json", "serve-777.out", "serve-777.sbatch"]


def test_version_flag(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "slab-stack" in result.output
