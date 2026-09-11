"""MCP server tests: tool registration and behavior through the real tool layer."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from foundation import Workspace
from foundation.mcp_server import build_server
from foundation.session_record import find_session_record

EXPECTED_TOOLS = {
    "list_runs",
    "list_sessions",
    "show_run",
    "promote_run",
    "promote_session",
    "retire_session",
    "expire_runs",
    "gc",
    "launch_workflow",
    "wait_for_run",
    "list_engines",
    "list_tasks",
    "describe_task",
    "search_materials",
    "get_material",
    "query_materials",
    "notebook",
    "plan",
    "list_memories",
    "recall",
    "remember",
    "list_skills",
    "skill",
    "report_results",
}
HPC_TOOLS = {"submit_job", "job_status", "cancel_job"}


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "ws"


def _call(server: Any, tool: str, args: dict[str, Any] | None = None) -> Any:
    """Invoke a tool through the real MCP tool layer; normalize across mcp versions."""
    result = asyncio.run(server.call_tool(tool, args or {}))
    if isinstance(result, tuple):  # some mcp versions: (content_blocks, structured)
        blocks, structured = result
    else:  # mcp 2.x: CallToolResult
        blocks = getattr(result, "content", [])
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        # the SDK wraps non-object results as {"result": ...}
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    texts = [b.text for b in blocks if getattr(b, "text", None)]
    return json.loads(texts[0]) if texts else None


def _seed(root: Path, *, verified: bool = True, session: str | None = None) -> str:
    with Workspace(root) as ws:
        with ws.start_run(name="seeded", intent="mcp test", session=session) as run:
            # content differs per outcome so gc tests don't hit shared-hash dedup
            run.keep("out", {"e": -2.0, "verified": verified})
            run.check(lambda: verified, name="gate")
        return run.id


def test_expected_tools_registered(root: Path) -> None:
    server = build_server(root)
    tools = asyncio.run(server.list_tools())
    by_name = {t.name: t for t in tools}
    assert set(by_name) == EXPECTED_TOOLS
    for tool in by_name.values():
        assert tool.description  # agents read these; empty would be a regression


def test_list_and_show(root: Path) -> None:
    run_id = _seed(root)
    server = build_server(root)

    runs = _call(server, "list_runs", {"state": "verified"})
    assert [r["id"] for r in runs] == [run_id]

    details = _call(server, "show_run", {"run_id": run_id[:8]})
    assert details["run"]["id"] == run_id
    assert details["checks"][0]["passed"] is True
    assert details["artifacts"][0]["bytes_available"] is True


def test_promote_expire_gc_flow(root: Path) -> None:
    keeper = _seed(root, verified=True)
    goner = _seed(root, verified=False)
    server = build_server(root)

    promoted = _call(server, "promote_run", {"run_id": keeper, "reason": "agent liked it"})
    assert promoted["state"] == "promoted"

    untouched = _call(server, "expire_runs", {})  # policy default: nothing is overdue
    assert untouched["count"] == 0

    expired = _call(server, "expire_runs", {"older_than": "0d"})
    assert expired["count"] == 1
    assert expired["expired"][0]["id"] == goner

    report = _call(server, "gc", {})
    assert len(report["dropped"]) == 1
    assert report["freed_bytes"] > 0

    with Workspace(root) as ws:
        assert ws.runs.get(keeper).state.value == "promoted"
        assert not ws.artifacts.has(ws.runs.get_artifact(goner, "out").hash)


def test_launch_workflow(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "wf.py"
    script.write_text(
        "from foundation import check, task\n"
        "@task\ndef triple(x):\n    return 3 * x\n"
        "print('result:', triple(5))\n"
        "@check\ndef ok():\n    return True\n"
    )
    server = build_server(root)
    result = _call(
        server,
        "launch_workflow",
        {"script_path": str(script), "intent": "launched by agent"},
    )
    assert result["state"] == "verified"
    assert "result: 15" in result["output"]
    with Workspace(root) as ws:
        assert ws.runs.get(result["run_id"]).intent == "launched by agent"


def test_launch_workflow_records_the_commands_the_run_resolved(root: Path, tmp_path: Path) -> None:
    """The harness session record says what ran, one command event per engine command."""
    script = tmp_path / "wf.py"
    script.write_text(
        "from foundation import task\n"
        "IDENTITY = {'engine': 'lammps', 'command': 'lmp -k on g 1 -sf kk'}\n"
        "@task(cache_extra=lambda arguments: IDENTITY)\n"
        "def probe(x):\n    return x\n"
        "probe(1)\n"
    )
    server = build_server(root, session="chat-cmd")
    launched = _call(server, "launch_workflow", {"script_path": str(script), "intent": "cmds"})
    _call(server, "wait_for_run", {"run_id": launched["run_id"]})
    events = find_session_record(root, "chat-cmd").events()
    launches = [e for e in events if e.get("type") == "command" and e["kind"] == "launch"]
    (launch,) = launches
    assert launch["tool"] == "launch_workflow" and launch["command"].startswith("slab run ")
    assert launch["sized"] is False and launch["resources"]["cpus"]  # the whole free budget
    commands = [e for e in events if e.get("type") == "command" and e["kind"] == "engine"]
    assert len(commands) == 1
    (command,) = commands
    assert command["kind"] == "engine" and command["tool"] == "launch_workflow"
    assert command["run_id"] == launched["run_id"] and command["task"] == "probe"
    assert command["command"] == "lmp -k on g 1 -sf kk" and command["kokkos"]["gpus"] == 1


def test_tool_errors_surface_helpfully(root: Path) -> None:
    _seed(root)
    server = build_server(root)
    with pytest.raises(Exception, match="no run matches"):
        _call(server, "show_run", {"run_id": "zzzz"})


def test_failure_evidence_reaches_the_agent(root: Path, tmp_path: Path) -> None:
    """An agent's debugging loop: launch fails -> the result carries the failure
    record -> show_run exposes the failed task's evidence too."""
    script = tmp_path / "boom.py"
    script.write_text(
        "from foundation import task\n"
        "@task\ndef explode():\n"
        "    e = RuntimeError('SCF diverged')\n"
        "    e.add_note('last residual: 3.2e-2')\n"
        "    raise e\n"
        "explode()\n"
    )
    server = build_server(root)
    result = _call(server, "launch_workflow", {"script_path": str(script), "intent": "doomed"})
    assert result["status"] == "failed"
    assert result["failure"]["type"] == "RuntimeError"
    assert "SCF diverged" in result["failure"]["traceback"]

    details = _call(server, "show_run", {"run_id": result["run_id"]})
    (task_entry,) = details["tasks"]
    assert task_entry["failure"]["notes"] == ["last residual: 3.2e-2"]


# -- sessions ----------------------------------------------------------------


def test_list_runs_filters_by_session(root: Path) -> None:
    mine = _seed(root, session="chat-1")
    _seed(root, verified=False, session="chat-2")
    server = build_server(root)
    assert [r["id"] for r in _call(server, "list_runs", {"session": "chat-1"})] == [mine]
    assert {r["session"] for r in _call(server, "list_runs", {})} == {"chat-1", "chat-2"}


def test_list_sessions_reports_counts_and_unstamped(root: Path) -> None:
    _seed(root, session="chat-1")
    _seed(root, verified=False, session="chat-1")
    _seed(root)  # no session
    server = build_server(root)

    summary = _call(server, "list_sessions", {})
    assert summary["unstamped"] == 1
    (row,) = summary["sessions"]
    assert (row["session"], row["runs"]) == ("chat-1", 2)
    assert row["breakdown"] == "1 quarantined, 1 verified"


def test_promote_session_reports_every_outcome(root: Path) -> None:
    verified = _seed(root, session="chat-1")
    unverified = _seed(root, verified=False, session="chat-1")
    server = build_server(root)

    result = _call(server, "promote_session", {"session": "chat", "reason": "the good batch"})
    assert (result["session"], result["promoted"], result["skipped"]) == ("chat-1", 1, 1)
    assert result["complete"] is False
    by_id = {o["id"]: o for o in result["outcomes"]}
    assert by_id[verified]["outcome"] == "promoted"
    assert by_id[unverified]["outcome"] == "skipped"

    forced = _call(server, "promote_session", {"session": "chat-1", "force": True})
    assert (forced["promoted"], forced["already"], forced["complete"]) == (1, 1, True)

    with Workspace(root) as ws:
        assert ws.runs.get(verified).state.value == "promoted"
        assert ws.runs.history(verified)[-1].actor == "agent"
        assert ws.runs.history(verified)[-1].reason == "the good batch"


def test_retire_session_retires_the_servers_own_session_and_records_it(
    root: Path, tmp_path: Path
) -> None:
    from foundation.session_record import find_session_record

    script = tmp_path / "wf.py"
    script.write_text("from foundation import check\n@check\ndef ok():\n    return True\n")
    anchor = _seed(root, session="earlier")
    server = build_server(root, project=tmp_path, session="chat-9")
    own = _call(server, "launch_workflow", {"script_path": str(script)})["run_id"]
    shakeout = _call(server, "launch_workflow", {"script_path": str(script)})["run_id"]

    dry = _call(server, "retire_session", {"run_ids": [anchor, own], "dry_run": True})
    assert dry["dry_run"] is True and dry["runs_promoted"] == 2
    with Workspace(root) as ws:
        assert ws.runs.get(anchor).state.value == "verified"

    report = _call(server, "retire_session", {"run_ids": [anchor, own]})
    assert report["session"] == "chat-9"
    by_id = {k["id"]: k["outcome"] for k in report["kept"]}
    assert by_id == {anchor: "promoted", own: "promoted"}
    assert [e["id"] for e in report["expired"]] == [shakeout]
    with Workspace(root) as ws:
        assert ws.runs.get(anchor).state.value == "promoted"
        assert ws.runs.history(anchor)[-1].actor == "agent"
        assert ws.runs.get(shakeout).state.value == "expired"
    events = find_session_record(root, "chat-9").events()
    retire = [e for e in events if e["type"] == "retire"]
    assert len(retire) == 1 and retire[0]["runs_expired"] == 1

    with pytest.raises(Exception, match="keep, expire, or purge"):
        _call(server, "retire_session", {"run_ids": [], "uncited": "drop"})


def test_promote_session_default_reason_names_the_session(root: Path) -> None:
    run_id = _seed(root, session="chat-1")
    server = build_server(root)
    _call(server, "promote_session", {"session": "chat-1"})
    with Workspace(root) as ws:
        assert ws.runs.history(run_id)[-1].reason == "promoted with session chat-1"


def test_promote_session_unknown_surfaces_helpfully(root: Path) -> None:
    _seed(root, session="chat-1")
    server = build_server(root)
    with pytest.raises(Exception, match="slab sessions"):
        _call(server, "promote_session", {"session": "nope"})


def test_materials_tools_answer_from_the_snapshot(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conftest import build_mp_snapshot

    snapshot = build_mp_snapshot(tmp_path / "mp-snapshot")
    (tmp_path / "slab.toml").write_text(f'[builders.mp]\nroot = "{snapshot}"\n')
    monkeypatch.chdir(tmp_path)
    server = build_server(root)
    rows = _call(
        server,
        "search_materials",
        {"filters": {"elements": ["Fe"], "energy_above_hull__lte": 0.05}},
    )
    assert [row["material_id"] for row in rows] == ["mp-13"]
    record = _call(server, "get_material", {"material_id": "mp-13"})
    assert record["formula_pretty"] == "Fe"
    assert Path(record["cif_file"]).is_file()
    overview = _call(server, "list_engines")
    assert overview["mp"]["materials"] == 4


def test_materials_tools_unconfigured_surface_the_fix(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    server = build_server(root)
    with pytest.raises(Exception, match=r"\[builders.mp\] root"):
        _call(server, "search_materials", {})
    with pytest.raises(Exception, match=r"\[builders.mp\] root"):
        _call(server, "get_material", {"material_id": "mp-149"})


# -- parity with the resident agent -------------------------------------------


def test_hpc_tools_appear_only_with_partitions(root: Path, tmp_path: Path) -> None:
    server = build_server(root, project=tmp_path)
    assert not HPC_TOOLS & {t.name for t in asyncio.run(server.list_tools())}
    (tmp_path / "slab.toml").write_text(
        '[hpc]\ndefault_partition = "cpu"\n[hpc.partitions.cpu]\n'
    )
    server = build_server(root, project=tmp_path)
    assert {t.name for t in asyncio.run(server.list_tools())} >= HPC_TOOLS
    # Off a cluster the scheduler refuses loudly, through the same error path.
    with pytest.raises(Exception, match="not on PATH"):
        _call(server, "job_status", {"job_id": "1"})


def test_tasks_are_listed_and_described(root: Path) -> None:
    server = build_server(root)
    names = {entry["name"] for entry in _call(server, "list_tasks")}
    assert {"relax", "relax_cell", "single_point"} <= names
    described = _call(server, "describe_task", {"name": "relax"})
    assert described["name"] == "relax" and "engine" in described["signature"]
    assert described["doc"]
    with pytest.raises(Exception, match="no task 'relox'; known:"):
        _call(server, "describe_task", {"name": "relox"})


def test_launched_runs_carry_the_server_session_and_wait_reports_them(
    root: Path, tmp_path: Path
) -> None:
    script = tmp_path / "wf.py"
    script.write_text("from foundation import check\n@check\ndef ok():\n    return True\n")
    server = build_server(root, project=tmp_path, session="mcp-test-1")
    launched = _call(server, "launch_workflow", {"script_path": str(script), "intent": "x"})
    assert launched["session"] == "mcp-test-1"
    waited = _call(server, "wait_for_run", {"run_id": launched["run_id"][:8]})
    assert waited["outcome"] == "finished"
    assert waited["run"]["id"] == launched["run_id"]
    assert waited["run"]["progress"].startswith("tasks:")
    by_name = _call(server, "wait_for_run", {"run_id": "wf"})
    assert by_name["outcome"] == "finished" and "resolved 'wf' by name" in by_name["note"]
    idle = _call(server, "wait_for_run", {"timeout_s": 0.5})
    assert idle["outcome"] == "none_running"
    assert [r["id"] for r in idle["runs"]] == [launched["run_id"]]
    empty = _call(build_server(root, project=tmp_path, session="mcp-test-2"),
                  "wait_for_run", {"timeout_s": 0.5})
    assert empty["outcome"] == "no_runs"


def test_notebook_and_plan_are_the_project_files(root: Path, tmp_path: Path) -> None:
    server = build_server(root, project=tmp_path)
    assert _call(server, "plan")["text"] == ""
    written = _call(server, "plan", {"content": "1. relax Cu"})
    assert written["text"] == "1. relax Cu\n"
    assert (tmp_path / "PLAN.md").read_text() == "1. relax Cu\n"
    noted = _call(server, "notebook", {"entry": "a0 = 3.60 Å (run ab12)", "heading": "lattice"})
    assert noted["recorded"] is True and noted["path"] == str(tmp_path / "NOTEBOOK.md")
    tail = _call(server, "notebook")["text"]
    assert tail.startswith("# Lab notebook") and " — lattice\n" in tail
    assert "(run ab12)" in tail


def test_machine_memory_round_trip(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(tmp_path / "memory"))
    server = build_server(root, project=tmp_path)
    assert _call(server, "list_memories") == []
    with pytest.raises(Exception, match="no memory named 'vllm-cache'"):
        _call(server, "recall", {"name": "vllm-cache"})
    written = _call(
        server,
        "remember",
        {"name": "vllm-cache", "description": "the vllm cache flag matters", "body": "Set it."},
    )
    assert written["name"] == "vllm-cache" and Path(written["path"]).is_file()
    listed = _call(server, "list_memories")
    assert [m["name"] for m in listed] == ["vllm-cache"]
    recalled = _call(server, "recall", {"name": "vllm-cache"})
    assert recalled["body"] == "Set it." and "mcp" in recalled["provenance"]
    assert recalled["changed_since"] == []
    with pytest.raises(Exception, match="name"):
        _call(server, "remember", {"name": "Bad Name", "description": "d", "body": "b"})


def test_skills_are_cataloged_loaded_and_recorded(root: Path, tmp_path: Path) -> None:
    (tmp_path / "skills" / "xrd").mkdir(parents=True)
    (tmp_path / "skills" / "xrd" / "SKILL.md").write_text(
        "---\nname: xrd\ndescription: Simulate a pattern.\n---\n# XRD\n\nRun it.\n"
    )
    (tmp_path / "skills" / "xrd" / "scripts").mkdir()
    (tmp_path / "skills" / "xrd" / "scripts" / "sim.py").write_text("print(1)\n")
    server = build_server(root, project=tmp_path, session="mcp-skills")
    catalog = {s["name"]: s for s in _call(server, "list_skills")}
    assert catalog["equation-of-state"]["source"] == "built-in"
    assert catalog["xrd"]["source"] == "project"
    loaded = _call(server, "skill", {"name": "xrd"})
    assert loaded["body"].startswith("# XRD") and loaded["files"] == ["SKILL.md", "scripts/sim.py"]
    assert loaded["digest"] == catalog["xrd"]["digest"]
    with pytest.raises(Exception, match="no skill named 'xrdd'; available skills: "):
        _call(server, "skill", {"name": "xrdd"})
    assert find_session_record(root, "mcp-skills").skills() == {"xrd": loaded["digest"]}


def test_query_materials_answers_from_the_snapshot(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conftest import build_mp_snapshot

    snapshot = build_mp_snapshot(tmp_path / "mp-snapshot")
    (tmp_path / "slab.toml").write_text(f'[builders.mp]\nroot = "{snapshot}"\n')
    monkeypatch.chdir(tmp_path)
    server = build_server(root, project=tmp_path)
    answer = _call(
        server,
        "query_materials",
        {"sql": "SELECT material_id FROM materials ORDER BY material_id LIMIT 2"},
    )
    assert len(answer["rows"]) == 2
    with pytest.raises(Exception, match="SELECT"):
        _call(server, "query_materials", {"sql": "DELETE FROM materials"})


def test_report_results_validates_then_records_the_answer(root: Path, tmp_path: Path) -> None:
    run_id = _seed(root, session="mcp-answer")
    server = build_server(root, project=tmp_path, session="mcp-answer")
    with pytest.raises(Exception, match="no numeric value"):
        _call(server, "report_results",
              {"results": {"a0": {"value": "3.6", "unit": "Å"}}, "run_ids": [run_id]})
    with pytest.raises(Exception, match="no run matches"):
        _call(server, "report_results",
              {"results": {"a0": {"value": 3.6, "unit": "Å"}}, "run_ids": ["zzzz"]})
    answer = _call(
        server,
        "report_results",
        {
            "results": {"a0": {"value": 3.6, "unit": "Å"}},
            "run_ids": [run_id[:8]],
            "summary": "done",
        },
    )
    assert answer["session"] == "mcp-answer" and answer["run_ids"] == [run_id]
    record = find_session_record(root, "mcp-ans")
    assert record.header()["client"] == "mcp"
    assert record.results()["results"] == {"a0": {"value": 3.6, "unit": "Å"}}
    assert record.results()["run_ids"] == [run_id]
    assert Path(answer["recorded"]) == root / "sessions" / "mcp-answer.jsonl"


# -- resource sizing --------------------------------------------------------------------


@pytest.fixture()
def no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS", "SLURM_NTASKS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_list_engines_reports_budget_and_free(root: Path, no_gpus: None) -> None:
    import os

    server = build_server(root)
    answer = _call(server, "list_engines")
    assert answer["budget"]["gpus"] == 0 and answer["budget"]["cpus"] >= 1
    assert answer["free"] == answer["budget"]
    with Workspace(root) as ws:
        ws.reserve(ntasks=1, holder_pid=os.getpid())
    after = _call(server, "list_engines")
    assert after["free"]["cpus"] == answer["budget"]["cpus"] - 1


def test_sized_launch_runs_as_a_child_and_the_run_carries_resources(
    root: Path, tmp_path: Path, no_gpus: None
) -> None:
    """A sized launch is a child 'foundation run --reservation': the run's record
    holds the slice, the child's environment carried it, and the reservation is
    released when the run ends."""
    script = tmp_path / "wf.py"
    script.write_text(
        "import os\n"
        "from foundation import check\n"
        "print('env', os.environ['SLAB_NTASKS'], os.environ['OMP_NUM_THREADS'],"
        " repr(os.environ['CUDA_VISIBLE_DEVICES']))\n"
        "@check\ndef ok():\n    return True\n"
    )
    server = build_server(root, project=tmp_path, session="mcp-sized")
    result = _call(
        server,
        "launch_workflow",
        {"script_path": str(script), "intent": "sized", "ntasks": 1, "threads": 1},
    )
    assert result["state"] == "verified" and result["exit_code"] == 0
    assert result["resources"]["ntasks"] == 1 and len(result["resources"]["cpus"]) == 1
    assert "env 1 1 ''" in result["output"]
    assert Path(result["log"]).exists()
    with Workspace(root) as ws:
        run = ws.runs.get(result["run_id"])
        assert run.session == "mcp-sized" and run.pid != os.getpid()
        assert run.resources["reservation"] == result["reservation"]
        assert ws.runs.list_reservations() == []


def test_unsized_launch_reserves_the_whole_free_budget_in_process(
    root: Path, tmp_path: Path, no_gpus: None
) -> None:
    script = tmp_path / "wf.py"
    script.write_text("from foundation import check\n@check\ndef ok():\n    return True\n")
    server = build_server(root, project=tmp_path)
    budget = _call(server, "list_engines")["budget"]
    result = _call(server, "launch_workflow", {"script_path": str(script), "intent": "x"})
    assert len(result["resources"]["cpus"]) == budget["cpus"]
    with Workspace(root) as ws:
        assert ws.runs.get(result["run_id"]).pid == os.getpid()
        assert ws.runs.list_reservations() == []


def test_a_launch_that_does_not_fit_is_refused_with_the_free_amounts(
    root: Path, tmp_path: Path, no_gpus: None
) -> None:
    script = tmp_path / "wf.py"
    script.write_text("pass\n")
    server = build_server(root, project=tmp_path)
    budget = _call(server, "list_engines")["budget"]
    with pytest.raises(Exception, match=r"1 gpu\(s\) asked, but only 0 of 0"):
        _call(server, "launch_workflow", {"script_path": str(script), "gpus": 1})
    with pytest.raises(Exception, match=rf"only {budget['cpus']} of {budget['cpus']} cpu"):
        _call(server, "launch_workflow", {"script_path": str(script), "ntasks": budget["cpus"] + 1})
    with Workspace(root) as ws:
        assert ws.runs.list_reservations() == []  # nothing leaked
        assert ws.runs.list_runs() == []


def test_a_failed_sized_launch_releases_its_reservation(
    root: Path, tmp_path: Path, no_gpus: None
) -> None:
    server = build_server(root, project=tmp_path)
    with pytest.raises(Exception, match="no such workflow script"):
        _call(server, "launch_workflow", {"script_path": str(tmp_path / "nope.py"), "ntasks": 1})
    with Workspace(root) as ws:
        assert ws.runs.list_reservations() == []


def test_submit_job_takes_a_size(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat

    (tmp_path / "slab.toml").write_text(
        "[hpc]\n"
        'default_partition = "gpu"\n'
        "[hpc.partitions.gpu]\n"
        'gres = "gpu:a100:4"\n'
        "[hpc.partitions.gpu.node]\n"
        "cpus = 64\n"
        "gpus = 4\n"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/bin/sh\necho 777\n")
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
    server = build_server(root, project=tmp_path, session="mcp-job")
    job = _call(
        server,
        "submit_job",
        {"command": "slab run md.py", "name": "md", "ntasks_per_node": 4, "gpus_per_node": 2},
    )
    assert job["job_id"] == "777"
    assert job["size"] == {
        "nodes": 1, "ntasks_per_node": 4, "cpus_per_task": 1, "gpus_per_node": 2, "mem": None,
    }
    kept = Path(job["script_path"]).read_text()
    assert "#SBATCH --ntasks-per-node=4\n" in kept and "#SBATCH --gres=gpu:a100:2\n" in kept
    with pytest.raises(Exception, match="gpus_per_node=5 exceeds the 4 gpus"):
        _call(server, "submit_job", {"command": "c", "name": "n", "ntasks_per_node": 1,
                                     "gpus_per_node": 5})
    with pytest.raises(Exception, match="pass ntasks_per_node"):
        _call(server, "submit_job", {"command": "c", "name": "n", "gpus_per_node": 1})
    events = find_session_record(root, "mcp-job").events()
    (event,) = [e for e in events if e.get("type") == "command" and e["kind"] == "job"]
    assert event["size"]["gpus_per_node"] == 2
