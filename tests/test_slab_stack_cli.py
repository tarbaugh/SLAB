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
    # What a session leaves beside its transcripts: compaction summaries for
    # the conversation and its delegate, and review records for both.
    (sessions / "20260826-120000-111.compactions.md").write_text("# summary\n")
    (sessions / "20260826-120000-111-crystal-1.compactions.md").write_text("# summary\n")
    (sessions / "20260827-090000-222.compactions.md").write_text("# summary\n")
    reviews = root / "mason" / "reviews"
    reviews.mkdir(parents=True)
    (reviews / "20260826-120000-111-review-1.md").write_text("---\n---\n")
    (reviews / "20260826-120000-111-crystal-1-review-1.md").write_text("---\n---\n")
    (reviews / "20260827-090000-222-review-1.md").write_text("---\n---\n")
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
    assert remaining == ["20260827-090000-222.compactions.md", "20260827-090000-222.jsonl"]
    assert [p.name for p in (root / "mason" / "reviews").iterdir()] == [
        "20260827-090000-222-review-1.md"
    ]
    assert list((root / "jobs").iterdir()) == []
    assert "deleted expired runs: 1\n" in result.output
    assert "deleted transcripts: 2 (" in result.output
    assert "deleted sidecars: 4 (" in result.output
    assert "deleted job files: 2 (" in result.output
    assert (
        "kept transcript mason/sessions/20260827-090000-222.jsonl: the newest conversation "
        "(--all-sessions removes it too)" in result.output
    )


def test_purge_dry_run_deletes_nothing(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _seed_runs(root)
    _seed_files(root)
    assert runner.invoke(app, ["fast-forward", "-w", str(root)]).exit_code == 0
    result = runner.invoke(app, ["purge", "-w", str(root), "--dry-run"])
    assert result.exit_code == 0
    # The dry run prints the inventory: one line per category, its items under it.
    assert "would delete expired runs: 1\n" in result.output
    assert "  failed-test\n" in result.output
    assert "would delete blobs: 1 (7 bytes)\n" in result.output
    assert "  mason/reviews/20260826-120000-111-review-1.md\n" in result.output
    assert "would delete unrecognised: none\n" in result.output
    assert "would delete scratch: none\n" in result.output
    with Workspace(root) as ws:
        assert len(ws.runs.list_runs()) == 2
    assert len(list((root / "mason" / "sessions").iterdir())) == 6
    assert len(list((root / "mason" / "reviews").iterdir())) == 3
    assert len(list((root / "jobs").iterdir())) == 2


def test_purge_all_sessions_removes_the_newest_too(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _seed_files(root)
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes", "--all-sessions"])
    assert result.exit_code == 0
    assert list((root / "mason" / "sessions").iterdir()) == []
    assert list((root / "mason" / "reviews").iterdir()) == []


def test_purge_confirmation_defaults_to_no(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _seed_runs(root)
    assert runner.invoke(app, ["fast-forward", "-w", str(root)]).exit_code == 0
    result = runner.invoke(app, ["purge", "-w", str(root)], input="n\n")
    assert result.exit_code != 0
    assert "permanently delete expired runs: 1, blobs: 1 (7 bytes in all) from" in result.output
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


def test_purge_json_prints_the_inventory(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    _seed_runs(root)
    _seed_files(root)
    assert runner.invoke(app, ["fast-forward", "-w", str(root)]).exit_code == 0
    result = runner.invoke(app, ["purge", "-w", str(root), "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    inventory = json.loads(result.output)
    assert inventory["dry_run"] is True
    by_name = {c["name"]: c for c in inventory["categories"]}
    assert list(by_name) == [
        "stale locks", "transcripts", "sidecars", "unrecognised", "harness records",
        "job files", "dry-run records", "expired runs", "blobs", "scratch",
    ]
    assert by_name["blobs"]["bytes"] == 7
    assert by_name["job files"]["items"] == [
        "jobs/cu-relax-1244113.out",
        "jobs/cu-relax-1244113.sbatch",
    ]
    assert inventory["kept"][0]["kind"] == "transcript"


def test_fast_forward_reaps_a_dead_run_before_it_expires_it(tmp_path: Path) -> None:
    from conftest import vanished_pid
    from foundation.runtime import this_host

    root = tmp_path / ".slab"
    with Workspace(root) as ws:
        killed = ws.runs.create(Run(name="killed"))
        ws.runs.set_status(killed.id, "running", pid=vanished_pid(), host=this_host())
    result = runner.invoke(app, ["fast-forward", "-w", str(root)])
    assert result.exit_code == 0, result.output
    assert f"failed  {killed.id}  killed  its process is gone" in result.output
    assert "1 run(s) fast-forwarded to expired" in result.output
    with Workspace(root) as ws:
        after = ws.runs.get(killed.id)
        assert after.status.value == "failed" and after.state.value == "expired"


def test_purge_leaves_nothing_behind(
    tmp_path: Path, scratch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guarantee. After fast-forward --include-running and purge
    --all-sessions the workspace holds the promoted rows and the blobs
    they reach, the serve record of a live job, and nothing under the
    session, review, lock, record, and job directories; the scratch root
    holds only the directory of a live process."""
    import os

    from conftest import seed_scratch, vanished_pid
    from foundation.runtime import this_host

    root = tmp_path / ".slab"
    ids = _seed_runs(root)
    _seed_files(root)
    sessions = root / "mason" / "sessions"
    (sessions / "20260810-000000-7-md-expert-1.jsonl").write_text("{}\n")  # orphan sibling
    (sessions / "notes.txt").write_text("stray\n")  # unrecognised
    (root / "mason" / "reviews" / "20260701-000000-3-review-1.md").write_text("---\n")  # orphan
    (root / "sessions").mkdir()
    (root / "sessions" / "mcp-20260901-100000-5.jsonl").write_text("{}\n")  # harness record
    (root / "mason" / "locks").mkdir()
    (root / "mason" / "locks" / "1111111111111111.lock").write_text("pid 1\n")  # stale
    (root / "mason" / "endpoint.json").write_text(
        json.dumps({"endpoint": "http://node:8000/v1", "model": "m", "job_id": "777"})
    )
    (root / "mason" / "serve-500.out").write_text("an older, finished server\n")
    with Workspace(root) as ws:
        killed = ws.runs.create(Run(name="killed"))
        ws.runs.set_status(killed.id, "running", pid=vanished_pid(), host=this_host())
    seed_scratch(scratch_root, "slab-qe-killed", pid=vanished_pid(), run_id=killed.id)
    seed_scratch(scratch_root, "slab-qe-old", marker=False)  # no owner at all
    live = seed_scratch(scratch_root, "slab-qe-live")  # this process, still running
    monkeypatch.setattr("slab_stack.cli.active_job_ids", lambda: frozenset({"777"}))

    forward = runner.invoke(app, ["fast-forward", "-w", str(root), "--include-running"])
    assert forward.exit_code == 0, forward.output
    assert "2 run(s) fast-forwarded to expired" in forward.output
    result = runner.invoke(app, ["purge", "-w", str(root), "--all-sessions", "--yes"])
    assert result.exit_code == 0, result.output

    with Workspace(root) as ws:
        assert [r.id for r in ws.runs.list_runs()] == [ids["keep"]]
        assert ws.artifacts.has(ids["shared"]) and not ws.artifacts.has(ids["scratch"])
    files = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not p.name.startswith("runs.db")
    )
    assert files == [
        f"cas/{ids['shared'][:2]}/{ids['shared'][2:4]}/{ids['shared']}",
        "mason/endpoint.json",
    ]
    for empty in ("mason/sessions", "mason/reviews", "mason/locks", "sessions", "jobs"):
        assert list((root / empty).iterdir()) == [], empty
    assert [p.name for p in scratch_root.iterdir()] == [live.name]
    assert f"kept scratch {live}: process {os.getpid()} is alive on this host" in result.output
    # The killed run's scratch went at reap time, inside fast-forward; the
    # purge is the backstop for the directory nothing owned.
    assert "deleted scratch: 1 (" in result.output
    assert "deleted unrecognised: 2 (" in result.output
    assert "deleted harness records: 1 (" in result.output
    assert "deleted stale locks: 1 (" in result.output
    assert "kept job file" not in result.output  # the live server's record is not a job file


def test_purge_keeps_unrecognised_files_and_the_newest_record_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".slab"
    _seed_files(root)
    (root / "mason" / "sessions" / "notes.txt").write_text("stray\n")
    (root / "sessions").mkdir()
    (root / "sessions" / "mcp-20260901-100000-5.jsonl").write_text("{}\n")
    (root / "sessions" / "mcp-20260902-100000-6.jsonl").write_text("{}\n")
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes"])
    assert result.exit_code == 0, result.output
    assert (root / "mason" / "sessions" / "notes.txt").is_file()
    assert "kept unrecognised mason/sessions/notes.txt: no transcript claims it" in result.output
    assert [p.name for p in (root / "sessions").iterdir()] == ["mcp-20260902-100000-6.jsonl"]
    assert "deleted harness records: 1 (" in result.output


def test_purge_leaves_a_lock_taken_since_the_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock path is the project's, so a session that started while
    purge was busy holds the very file the inventory called stale. The
    lock is probed again just before it is unlinked."""
    import fcntl

    from slab_stack import _ops as stack_ops

    root = tmp_path / ".slab"
    locks = root / "mason" / "locks"
    locks.mkdir(parents=True)
    lock = locks / "1111111111111111.lock"
    lock.write_text("pid 1\n")
    handle = open(lock, "a+", encoding="utf-8")  # noqa: SIM115 - held across the purge
    probed = stack_ops.stale_locks

    def take_then_report(workspace_root: Path) -> list[Path]:
        found = probed(workspace_root)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # a session starts now
        return found

    monkeypatch.setattr(stack_ops, "stale_locks", take_then_report)
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes"])
    handle.close()
    assert result.exit_code == 0, result.output
    assert lock.is_file()
    assert "deleted stale locks: none" in result.output


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


# -- machine memory ----------------------------------------------------------


@pytest.fixture()
def memories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A memory store of this test's own, seeded with two facts a person confirmed."""
    from datetime import date

    from foundation import memory as memory_store

    root = tmp_path / "memory"
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(root))
    memory_store.write(
        "vllm-mamba-cache",
        "vLLM refuses hybrid-Mamba models at the default batch size.",
        "Lower max-num-seqs below the available Mamba cache blocks.",
        agent="pi",
        model="qwen3-30b",
        directory=root,
        evidence="checked by hand",
        confirmed=date.today(),
    )
    memory_store.write(
        "srun-in-sandbox", "srun cannot reach the controller here.", "Use mpirun.",
        agent="md-expert", directory=root,
        evidence="checked by hand",
        confirmed=date.today(),
    )
    return root


def test_memory_list_names_what_the_machine_knows(memories: Path) -> None:
    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 0, result.output
    assert "vllm-mamba-cache" in result.output
    assert "vLLM refuses hybrid-Mamba models" in result.output
    assert "md-expert" in result.output
    assert "2 memory(s)" in result.output
    assert str(memories) in result.output


def test_memory_list_json_carries_the_provenance(memories: Path) -> None:
    result = runner.invoke(app, ["memory", "list", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert [row["name"] for row in rows] == ["srun-in-sandbox", "vllm-mamba-cache"]
    assert rows[1]["agent"] == "pi" and rows[1]["model"] == "qwen3-30b"
    assert rows[1]["created"] == rows[1]["updated"]


def test_memory_list_on_an_empty_machine_says_where_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(tmp_path / "nothing"))
    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 0
    assert "no memories recorded yet" in result.output
    assert str(tmp_path / "nothing") in result.output


def test_memory_show_prints_the_file_verbatim(memories: Path) -> None:
    result = runner.invoke(app, ["memory", "show", "vllm-mamba-cache"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("---\n")
    assert "agent: pi" in result.output
    assert "Lower max-num-seqs" in result.output


def test_memory_show_of_an_unknown_name_lists_what_exists(memories: Path) -> None:
    result = runner.invoke(app, ["memory", "show", "ghost"])
    assert result.exit_code == 1
    assert "no memory named 'ghost'" in result.output
    assert "vllm-mamba-cache" in result.output


def test_memory_forget_confirms_before_deleting(memories: Path) -> None:
    from foundation import memory as memory_store

    refused = runner.invoke(app, ["memory", "forget", "srun-in-sandbox"], input="n\n")
    assert refused.exit_code == 1
    assert "srun cannot reach the controller here." in refused.output
    assert (memories / "srun-in-sandbox.md").is_file()

    accepted = runner.invoke(app, ["memory", "forget", "srun-in-sandbox"], input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert "forgot" in accepted.output
    assert list(memory_store.discover(memories)) == ["vllm-mamba-cache"]


def test_memory_forget_yes_skips_the_prompt(memories: Path) -> None:
    result = runner.invoke(app, ["memory", "forget", "vllm-mamba-cache", "--yes"])
    assert result.exit_code == 0, result.output
    assert not (memories / "vllm-mamba-cache.md").exists()


@pytest.fixture()
def _live(monkeypatch: pytest.MonkeyPatch) -> None:
    """The machine reports one piece of software, without probing any."""
    import slab._ops

    monkeypatch.setattr(slab._ops, "software_versions", lambda: {"lammps": "22Jul2025"})


@pytest.mark.usefixtures("_live")
def test_memory_add_wants_evidence_or_says_unverified(memories: Path) -> None:
    from foundation import memory as memory_store

    refused = runner.invoke(
        app, ["memory", "add", "fix-nph", "fix nph takes no temp.", "Seen once."]
    )
    assert refused.exit_code == 1
    assert "a memory needs evidence" in refused.output
    assert "fix-nph" not in memory_store.discover()

    claim = runner.invoke(
        app, ["memory", "add", "fix-nph", "fix nph takes no temp.", "Seen once.", "--unverified"]
    )
    assert claim.exit_code == 0, claim.output
    assert "recorded fix-nph (unverified)" in claim.output

    checked = runner.invoke(
        app,
        ["memory", "add", "fix-nph", "lammps fix nph takes a pressure keyword.", "-",
         "--evidence", "the lammps manual and one completed run"],
        input="Both iso and aniso ran.\n",
    )
    assert checked.exit_code == 0, checked.output
    assert checked.output.startswith("replaced fix-nph in ")
    memory = memory_store.discover()["fix-nph"]
    assert memory.unverified is False and memory.agent == "cli"
    assert memory.body() == "Both iso and aniso ran.\n"
    assert memory.against == {"lammps": "22Jul2025"}
    assert [v.body for v in memory_store.versions("fix-nph")] == ["Seen once."]

    listed = runner.invoke(app, ["memory", "list"])
    assert "[unverified]" not in listed.output


@pytest.mark.usefixtures("_live")
def test_memory_review_lists_what_a_person_should_check(memories: Path) -> None:
    from foundation import memory as memory_store

    clean = runner.invoke(app, ["memory", "review"])
    assert clean.exit_code == 0, clean.output
    assert "nothing to review (2 memory(s), all verified and current)" in clean.output

    memory_store.write(
        "gpu-build-works", "The gpu build's command works.", "It started.",
        agent="md-expert", unverified=True, directory=memories,
    )
    memory_store.write(
        "lammps-kokkos", "lammps kokkos wants one rank per gpu.", "Seen.",
        against={"lammps": "2Aug2023"}, evidence="run 01k2x7abcd", directory=memories,
    )
    result = runner.invoke(app, ["memory", "review"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("gpu-build-works") and lines[0].endswith("[unverified]")
    assert lines[1].startswith("lammps-kokkos")
    assert lines[1].endswith("[lammps was 2Aug2023, now 22Jul2025]")
    assert "2 of 4 memory(s) to review" in lines[-1]
    assert "vllm-mamba-cache" not in result.output

    rows = json.loads(runner.invoke(app, ["memory", "review", "--json"]).output)
    assert [(row["name"], row["reasons"]) for row in rows] == [
        ("gpu-build-works", ["unverified"]),
        ("lammps-kokkos", ["lammps was 2Aug2023, now 22Jul2025"]),
    ]
    assert rows[0]["agent"] == "md-expert"

    listed = runner.invoke(app, ["memory", "list"])
    assert "The gpu build's command works. [unverified]" in listed.output


def test_memory_review_rejudges_the_evidence_against_a_workspace(
    tmp_path: Path, memories: Path
) -> None:
    """The rule that evidence is a finished run reaches the memories written before it."""
    from foundation import memory as memory_store
    from foundation.models import Run
    from foundation.runtime import Workspace

    root = tmp_path / "ws"
    with Workspace(root) as ws:
        going = ws.runs.create(Run(name="melt"))
        ws.runs.set_status(going.id, "running")
        done = ws.runs.create(Run(name="quench"))
        ws.runs.set_status(done.id, "running")
        ws.runs.set_status(done.id, "completed")
    memory_store.write(
        "device-init-fails", "A node refuses to initialise its GPUs.", "Body.",
        agent="md-expert", evidence=f"run {going.id} died at once", directory=memories,
    )
    memory_store.write(
        "scratch-quota", "The scratch filesystem here fills at 80 percent.", "Body.",
        agent="md-expert", evidence=f"run {done.id} completed after the sweep",
        directory=memories,
    )
    result = runner.invoke(app, ["memory", "review", "--workspace", str(root)])
    assert result.exit_code == 0, result.output
    assert "device-init-fails" in result.output
    assert "no completed run confirms it" in result.output
    assert "scratch-quota" not in result.output
    # The memories seeded by the fixture cite no run at all, but a person
    # confirmed each by hand, so they are not judged against the runs.
    assert "1 of 4 memory(s) to review" in result.output

    rows = json.loads(
        runner.invoke(app, ["memory", "review", "--workspace", str(root), "--json"]).output
    )
    (row,) = [r for r in rows if r["name"] == "device-init-fails"]
    assert row["kind"] == "build"
    assert f"{going.id} does not (running" in row["reasons"][0]


def test_memory_review_finds_the_workspace_the_environment_and_the_config_name(
    tmp_path: Path, memories: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review reads the workspace every other command reads: the flag,
    then $SLAB_WORKSPACE, then [workspace] root, then ./.slab."""
    from foundation import memory as memory_store

    root = tmp_path / "ws"
    with Workspace(root) as ws:
        going = ws.runs.create(Run(name="melt"))
        ws.runs.set_status(going.id, "running")
    memory_store.write(
        "device-init-fails", "A node refuses to initialise its GPUs.", "Body.",
        agent="md-expert", evidence=f"run {going.id} died at once", directory=memories,
    )
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.delenv("SLAB_CONFIG", raising=False)
    monkeypatch.delenv("SLAB_SITE_CONFIG", raising=False)

    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    unjudged = runner.invoke(app, ["memory", "review"])
    assert unjudged.exit_code == 0, unjudged.output
    assert "no completed run confirms it" not in unjudged.output

    monkeypatch.setenv("SLAB_WORKSPACE", str(root))
    by_env = runner.invoke(app, ["memory", "review"])
    assert by_env.exit_code == 0, by_env.output
    assert f"{going.id} does not (running" in by_env.output

    monkeypatch.delenv("SLAB_WORKSPACE")
    (project / "slab.toml").write_text(f'[workspace]\nroot = "{root}"\n')
    by_config = runner.invoke(app, ["memory", "review"])
    assert by_config.exit_code == 0, by_config.output
    assert f"{going.id} does not (running" in by_config.output


def test_memory_review_lists_an_expired_outage_for_deletion(memories: Path) -> None:
    from datetime import date, timedelta

    from foundation import memory as memory_store

    memory_store.write(
        "device-init-fails", "A node refuses to initialise its GPUs.", "Body.",
        agent="md-expert", evidence="run 01k2x7abcd", kind="outage", where="n1",
        expires_at=date.today() - timedelta(days=1), directory=memories,
    )
    result = runner.invoke(app, ["memory", "review"])
    assert result.exit_code == 0, result.output
    assert "device-init-fails" in result.output
    assert "[expired outage, recorded " in result.output
    # It is still on the machine until the person removes it.
    assert "device-init-fails" in runner.invoke(app, ["memory", "list"]).output
    assert (memories / "device-init-fails.md").is_file()


@pytest.mark.usefixtures("_live")
def test_memory_confirm_records_the_evidence_and_restamps(memories: Path) -> None:
    from foundation import memory as memory_store

    memory_store.write(
        "lammps-kokkos", "lammps kokkos wants one rank per gpu.", "Seen.",
        agent="md-expert", against={"lammps": "2Aug2023"}, unverified=True,
        directory=memories,
    )
    missing = runner.invoke(app, ["memory", "confirm", "lammps-kokkos"])
    assert missing.exit_code != 0

    result = runner.invoke(
        app, ["memory", "confirm", "lammps-kokkos", "--evidence", "run 01k2x7abcd, 4 gpus"]
    )
    assert result.exit_code == 0, result.output
    assert "confirmed lammps-kokkos: evidence run 01k2x7abcd, 4 gpus" in result.output
    memory = memory_store.discover()["lammps-kokkos"]
    assert memory.unverified is False
    assert memory.against == {"lammps": "22Jul2025"}
    assert memory.agent == "md-expert" and memory.body() == "Seen.\n"
    assert "nothing to review" in runner.invoke(app, ["memory", "review"]).output

    ghost = runner.invoke(app, ["memory", "confirm", "ghost", "--evidence", "e"])
    assert ghost.exit_code == 1 and "no memory named 'ghost'" in ghost.output


def test_a_confirmed_memory_is_not_listed_again_and_the_warning_says_why(
    tmp_path: Path, memories: Path
) -> None:
    """A person who checked a fact by hand is not asked again at every review."""
    from datetime import date

    from foundation import memory as memory_store

    root = tmp_path / "ws"
    with Workspace(root) as ws:
        going = ws.runs.create(Run(name="melt"))
        ws.runs.set_status(going.id, "running")
        done = ws.runs.create(Run(name="quench"))
        ws.runs.set_status(done.id, "running")
        ws.runs.set_status(done.id, "completed")
    memory_store.write(
        "device-init-fails", "A node refuses to initialise its GPUs.", "Body.",
        agent="md-expert", evidence=f"run {going.id} died at once", directory=memories,
    )
    before = runner.invoke(app, ["memory", "review", "--workspace", str(root)])
    assert "device-init-fails" in before.output

    by_hand = runner.invoke(
        app, ["memory", "confirm", "device-init-fails", "--evidence", "checked by hand",
              "--workspace", str(root)]
    )
    assert by_hand.exit_code == 0, by_hand.output
    assert "confirmed device-init-fails: evidence checked by hand" in by_hand.output
    assert "warning: the evidence names no run id; the memory stands on your check alone" in (
        by_hand.output
    )
    memory = memory_store.discover()["device-init-fails"]
    assert memory.confirmed == date.today().isoformat()
    assert "confirmed by a person on " in memory.provenance()

    after = runner.invoke(app, ["memory", "review", "--workspace", str(root)])
    assert after.exit_code == 0, after.output
    assert "device-init-fails" not in after.output
    assert "nothing to review" in after.output

    # A run that never completed is named in the warning; a completed one
    # is not, because then the run confirms it.
    still_going = runner.invoke(
        app, ["memory", "confirm", "device-init-fails", "--evidence",
              f"run {going.id} died", "--workspace", str(root)]
    )
    assert f"warning: no completed run confirms it (evidence: {going.id} does not (running" in (
        still_going.output
    )
    finished = runner.invoke(
        app, ["memory", "confirm", "device-init-fails", "--evidence",
              f"run {done.id} completed", "--workspace", str(root)]
    )
    assert finished.exit_code == 0 and "warning:" not in finished.output
    # Without a workspace to read, the warning says where it looked.
    nowhere = runner.invoke(
        app, ["memory", "confirm", "device-init-fails", "--evidence",
              f"run {done.id} completed", "--workspace", str(tmp_path / "missing")]
    )
    assert "warning: no workspace at " in nowhere.output

    # An agent's later write is a new claim, and review judges it again.
    memory_store.write(
        "device-init-fails", "A node refuses to initialise its GPUs.", "Body.",
        agent="md-expert", evidence=f"run {going.id} died at once", directory=memories,
    )
    again = runner.invoke(app, ["memory", "review", "--workspace", str(root)])
    assert "device-init-fails" in again.output


def test_memory_confirm_moves_an_outages_expiry_out(memories: Path) -> None:
    from datetime import date, timedelta

    from foundation import memory as memory_store

    memory_store.write(
        "device-init-fails", "A node refuses to initialise its GPUs.", "Body.",
        agent="md-expert", evidence="run 01k2x7abcd", kind="outage", where="n1",
        expires_at=date.today() - timedelta(days=1), directory=memories,
    )
    assert "[expired outage" in runner.invoke(app, ["memory", "review"]).output

    result = runner.invoke(
        app, ["memory", "confirm", "device-init-fails", "--evidence", "still down today"]
    )
    assert result.exit_code == 0, result.output
    memory = memory_store.discover()["device-init-fails"]
    a_week = (date.today() + timedelta(days=memory_store.OUTAGE_DAYS)).isoformat()
    assert memory.expires_at == a_week and memory.where == "n1"
    assert f"expires {a_week}" in result.output
    assert "nothing to review" in runner.invoke(app, ["memory", "review"]).output

    dated = runner.invoke(
        app, ["memory", "confirm", "device-init-fails", "--evidence", "still down today",
              "--expires", "2030-01-01"]
    )
    assert dated.exit_code == 0, dated.output
    assert memory_store.discover()["device-init-fails"].expires_at == "2030-01-01"

    # Only an outage expires, so --expires on a build memory is refused.
    refused = runner.invoke(
        app, ["memory", "confirm", "srun-in-sandbox", "--evidence", "e", "--expires",
              "2030-01-01"]
    )
    assert refused.exit_code == 1 and "only an outage expires" in refused.output


def test_memory_purge_matches_globs_and_confirms(memories: Path) -> None:
    from foundation import memory as memory_store

    refused = runner.invoke(app, ["memory", "purge", "vllm-*"], input="n\n")
    assert refused.exit_code != 0
    assert "vllm-mamba-cache: vLLM refuses" in refused.output
    assert sorted(memory_store.discover()) == ["srun-in-sandbox", "vllm-mamba-cache"]

    accepted = runner.invoke(app, ["memory", "purge", "vllm-*"], input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert "1 of 2" in accepted.output
    assert "purged 1 memory(s)" in accepted.output
    assert list(memory_store.discover()) == ["srun-in-sandbox"]


def test_memory_purge_without_a_pattern_takes_everything(memories: Path) -> None:
    from foundation import memory as memory_store

    result = runner.invoke(app, ["memory", "purge", "--yes"])
    assert result.exit_code == 0, result.output
    assert "purged 2 memory(s)" in result.output
    assert memory_store.discover() == {}


def test_memory_purge_before_filters_by_date(memories: Path) -> None:
    from foundation import memory as memory_store

    (memories / "old-fact.md").write_text(
        "---\ndescription: An old fact.\ncreated: 2020-01-02\nupdated: 2020-01-02\n"
        "---\nBody.\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["memory", "purge", "--before", "2021-01-01", "--yes"])
    assert result.exit_code == 0, result.output
    assert "purged 1 memory(s)" in result.output
    assert sorted(memory_store.discover()) == ["srun-in-sandbox", "vllm-mamba-cache"]


def test_memory_purge_with_no_match_deletes_nothing(memories: Path) -> None:
    from foundation import memory as memory_store

    result = runner.invoke(app, ["memory", "purge", "ghost-*", "--yes"])
    assert result.exit_code == 0
    assert "nothing matched (memories here: 2)" in result.output
    assert len(memory_store.discover()) == 2


def test_memory_purge_refuses_a_malformed_date(memories: Path) -> None:
    result = runner.invoke(app, ["memory", "purge", "--before", "yesterday", "--yes"])
    assert result.exit_code == 1
    assert "YYYY-MM-DD" in result.output


def test_memory_forget_of_an_unknown_name_deletes_nothing(memories: Path) -> None:
    result = runner.invoke(app, ["memory", "forget", "ghost", "--yes"])
    assert result.exit_code == 1
    assert "no memory named 'ghost'" in result.output
    assert len(list(memories.glob("*.md"))) == 2


def test_memory_path_prints_the_directory(memories: Path) -> None:
    result = runner.invoke(app, ["memory", "path"])
    assert result.exit_code == 0
    assert result.output.strip() == str(memories)


def test_a_malformed_memory_is_reported_not_skipped(memories: Path) -> None:
    (memories / "broken.md").write_text("no frontmatter\n", encoding="utf-8")
    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 1
    assert "broken.md" in result.output
    assert "frontmatter" in result.output


def test_purge_leaves_the_machine_memory_alone(tmp_path: Path, memories: Path) -> None:
    root = tmp_path / ".slab"
    _seed_runs(root)
    result = runner.invoke(app, ["fast-forward", "--workspace", str(root)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["purge", "--workspace", str(root), "--yes"])
    assert result.exit_code == 0, result.output
    assert len(list(memories.glob("*.md"))) == 2


def test_memory_list_flags_what_changed_since(
    memories: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import slab._ops
    from foundation import memory as memory_store

    memory_store.write(
        "grace-gpu-growth", "gracemaker needs X.", "Body.", agent="pi",
        against={"gracemaker": "0.5.2"}, directory=memories,
        evidence="checked by hand",
    )
    monkeypatch.setattr(slab._ops, "software_versions", lambda: {"gracemaker": "0.6.0"})
    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 0, result.output
    assert "gracemaker needs X. [changed since: gracemaker was 0.5.2, now 0.6.0]" in result.output
    assert "vLLM refuses hybrid-Mamba models at the default batch size.\n" in result.output

    result = runner.invoke(app, ["memory", "list", "--json"])
    rows = {row["name"]: row for row in json.loads(result.output)}
    assert rows["grace-gpu-growth"]["against"] == {"gracemaker": "0.5.2"}
    assert rows["grace-gpu-growth"]["changed"] == ["gracemaker was 0.5.2, now 0.6.0"]
    assert rows["vllm-mamba-cache"]["against"] == {} and rows["vllm-mamba-cache"]["changed"] == []


def test_memory_list_of_unstamped_memories_probes_nothing(
    memories: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import slab._ops

    def refuse() -> dict[str, str]:
        raise AssertionError("no stamp, so no probe")

    monkeypatch.setattr(slab._ops, "software_versions", refuse)
    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 0, result.output
    assert "changed since" not in result.output


def test_benchmark_render_passes_the_entry_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mason.cli

    seen: dict[str, object] = {}

    def fake(goal: str, **kwargs: object) -> tuple[Path, str]:
        seen.update(kwargs, goal=goal)
        return tmp_path / "mason-sandbox.sbatch", "script"

    monkeypatch.setattr(mason.cli, "render_sandbox_files", fake)
    result = runner.invoke(app, ["benchmark", "render", "1", "--agent", "planner"])
    assert result.exit_code == 0, result.output
    assert seen["agent"] == "planner"
    assert "rendered Q1" in result.output


# -- slab runs: reap and fail -----------------------------------------------------


def _three_running(root: Path) -> dict[str, str]:
    """One run whose process exited, one alive (this test), one on another host."""
    import os
    import subprocess
    import sys

    from foundation.runtime import this_host

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    with Workspace(root) as ws:
        dead = ws.runs.create(Run(name="killed"))
        ws.runs.set_status(dead.id, "running", pid=child.pid, host=this_host())
        live = ws.runs.create(Run(name="live"))
        ws.runs.set_status(live.id, "running", pid=os.getpid(), host=this_host())
        away = ws.runs.create(Run(name="elsewhere"))
        ws.runs.set_status(away.id, "running", pid=1, host="another-node")
    return {"dead": dead.id, "live": live.id, "away": away.id, "pid": str(child.pid)}


REVERIFY_SCRIPT = """\
from foundation import check, task

@task
def simulate(script):
    return {{"rows": [300.0, 302.0]}}

result = simulate("run 1000")

@check
def thermostat_held():
    return abs(result["rows"][-1] - result["rows"][0]) < {tolerance}
"""


def test_runs_reverify_verifies_a_run_and_run_from_run_rehearses_on_its_results(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    wrong = tmp_path / "md.py"
    wrong.write_text(REVERIFY_SCRIPT.format(tolerance=1.0))
    fixed = tmp_path / "md_fixed.py"
    fixed.write_text(REVERIFY_SCRIPT.format(tolerance=5.0))
    launched = runner.invoke(app, ["run", str(wrong), "-w", str(root)])
    assert "state=quarantined" in launched.output, launched.output
    run_id = launched.output.split()[1]

    shown = runner.invoke(app, ["show", run_id, "-w", str(root)])
    assert "    [x] thermostat_held: returned False\n" in shown.output
    assert "        keys of result: rows\n" in shown.output
    assert "        source:\n" in shown.output

    refused = runner.invoke(app, ["run", str(fixed), "-w", str(root), "--from-run", run_id])
    assert refused.exit_code == 1 and "pass it with --dry-run" in refused.output
    rehearsed = runner.invoke(
        app, ["run", str(fixed), "-w", str(root), "--dry-run", "--from-run", run_id[:8]]
    )
    assert rehearsed.exit_code == 0, rehearsed.output
    assert f'"reading": "passed on the cached result of run {run_id}"' in rehearsed.output

    result = runner.invoke(app, ["runs", "reverify", run_id[:8], str(fixed), "-w", str(root)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith(f"run {run_id}: pass 2 verified, 1/1 checks passed")
    shown = runner.invoke(app, ["show", run_id, "-w", str(root)])
    assert "checks:  1/1 passed (pass 2)" in shown.output
    assert "earlier pass 1: 0/1" in shown.output


def test_runs_reap_marks_the_dead_and_names_what_it_did_not_check(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    ids = _three_running(root)
    result = runner.invoke(app, ["runs", "reap", "-w", str(root)])
    assert result.exit_code == 0, result.output
    assert f"failed  {ids['dead']}  killed  process {ids['pid']} on" in result.output
    assert "marked failed by slab runs reap" in result.output
    assert f"running {ids['live']}  live  process" in result.output
    assert "is alive" in result.output
    assert f"running {ids['away']}  elsewhere  process 1 on another-node, not this host" in (
        result.output
    )
    assert "1 run(s) marked failed" in result.output
    with Workspace(root) as ws:
        assert ws.runs.get(ids["dead"]).status.value == "failed"
        assert ws.runs.get(ids["live"]).status.value == "running"
        assert ws.runs.get(ids["away"]).status.value == "running"


def test_runs_fail_retires_one_run_and_refuses_a_live_process(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    ids = _three_running(root)
    result = runner.invoke(
        app, ["runs", "fail", ids["dead"], "--reason", "node rebooted", "-w", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert f"failed {ids['dead']}  killed  marked failed by the operator: node rebooted" in (
        result.output
    )
    assert "not checked" not in result.output

    result = runner.invoke(app, ["runs", "fail", ids["live"], "--reason", "x", "-w", str(root)])
    assert result.exit_code == 1
    assert "is alive; stop it first, or wait for it" in result.output
    with Workspace(root) as ws:
        assert ws.runs.get(ids["live"]).status.value == "running"

    result = runner.invoke(
        app, ["runs", "fail", ids["away"][:16], "--reason", "drained", "-w", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "marked failed by the operator: drained" in result.output
    assert "(not checked: process 1 on another-node, not this host" in result.output

    result = runner.invoke(app, ["runs", "fail", ids["dead"], "--reason", "again", "-w", str(root)])
    assert result.exit_code == 1
    assert "its status is 'failed', not 'running'" in result.output


def test_memory_list_marks_a_memory_about_slab(memories: Path) -> None:
    from foundation import memory as memory_store

    memory_store.write(
        "run-lammps-keys", "run_lammps result has n_rows, not rows.", "A count.",
        agent="pi", directory=memories,
        evidence="checked by hand",
    )
    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 0, result.output
    (line,) = [line for line in result.output.splitlines() if line.startswith("run-lammps-keys")]
    assert "[about slab]" in line
    other = next(line for line in result.output.splitlines() if line.startswith("srun-in"))
    assert "[about slab]" not in other
    rows = json.loads(runner.invoke(app, ["memory", "list", "--json"]).output)
    assert {row["name"]: row["about_slab"] for row in rows} == {
        "run-lammps-keys": True, "srun-in-sandbox": False, "vllm-mamba-cache": False,
    }


def _seed_job_runs(root: Path) -> dict[str, str]:
    """Two running runs stamped with jobs, as a sandbox job leaves them."""
    with Workspace(root) as ws:
        ended = ws.runs.create(Run(name="ended", job_id="4242"))
        held = ws.runs.reserve(
            host="compute-7", holder_pid=1, budget_cpus=range(2), budget_gpus=(),
            ntasks=1, job_id="4242",
        )
        ws.runs.claim_reservation(held.id, ended.id, host="compute-7", pid=1)
        lost = ws.runs.create(Run(name="lost", job_id="4243"))
        ws.runs.set_status(lost.id, "running", pid=1, host="compute-7")
    return {"ended": ended.id, "lost": lost.id}


@pytest.fixture
def _scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scheduler that says 4242 timed out and cannot place 4243."""
    from foundation import runtime
    from slab.hpc import JobState, JobStatus

    states = {"4242": JobState.TIMEOUT, "4243": JobState.UNDETERMINED}

    def answer(job_id: str) -> JobStatus:
        return JobStatus(job_id=job_id, state=states[job_id], raw=states[job_id].value)

    monkeypatch.setattr(runtime, "job_state", answer)


@pytest.mark.usefixtures("_scheduler")
def test_purge_marks_failed_the_runs_of_ended_jobs(tmp_path: Path) -> None:
    """The dry run names the runs it would mark failed and changes nothing;
    the purge marks them failed and releases their slices. A job the
    scheduler cannot place keeps its runs."""
    root = tmp_path / ".slab"
    ids = _seed_job_runs(root)
    dry = runner.invoke(app, ["purge", "-w", str(root), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert dry.output.startswith("would mark failed runs of ended jobs: 1\n")
    assert f"  {ids['ended']}  ended  job 4242\n" in dry.output
    with Workspace(root) as ws:
        assert ws.runs.get(ids["ended"]).status.value == "running"
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("marked failed runs of ended jobs: 1\n")
    with Workspace(root) as ws:
        failed = ws.runs.get(ids["ended"])
        assert failed.status.value == "failed"
        assert failed.error == "job 4242 is timeout; marked failed by slab purge"
        assert ws.runs.list_reservations() == []
        assert ws.runs.get(ids["lost"]).status.value == "running"


@pytest.mark.usefixtures("_scheduler")
def test_purge_job_takes_a_job_the_scheduler_cannot_place_as_ended(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    ids = _seed_job_runs(root)
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes", "--job", "4243"])
    assert result.exit_code == 0, result.output
    assert "marked failed runs of ended jobs: 2\n" in result.output
    with Workspace(root) as ws:
        assert ws.runs.get(ids["lost"]).error == "job 4243 ended; marked failed by slab purge"


def test_purge_job_is_refused_while_the_job_is_in_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".slab"
    ids = _seed_job_runs(root)
    monkeypatch.setattr("slab_stack.cli.active_job_ids", lambda: frozenset({"4243"}))
    result = runner.invoke(app, ["purge", "-w", str(root), "--yes", "--job", "4243"])
    assert result.exit_code == 1
    assert "job 4243 is still in the queue" in result.output
    assert "slab hpc cancel 4243" in result.output
    with Workspace(root) as ws:
        assert ws.runs.get(ids["lost"]).status.value == "running"


@pytest.mark.usefixtures("_scheduler")
def test_purge_confirmation_names_the_runs_it_marks_failed(tmp_path: Path) -> None:
    root = tmp_path / ".slab"
    ids = _seed_job_runs(root)
    result = runner.invoke(app, ["purge", "-w", str(root)], input="n\n")
    assert result.exit_code == 1
    assert "mark failed 1 run(s) of ended jobs and permanently delete" in result.output
    with Workspace(root) as ws:
        assert ws.runs.get(ids["ended"]).status.value == "running"
