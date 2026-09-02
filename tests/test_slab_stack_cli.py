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


# -- machine memory ----------------------------------------------------------


@pytest.fixture()
def memories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A memory store of this test's own, seeded with two facts."""
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
    )
    memory_store.write(
        "srun-in-sandbox", "srun cannot reach the controller here.", "Use mpirun.",
        agent="md-expert", directory=root,
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
