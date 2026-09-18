"""Machine memory as the agent meets it: the prompt block and the two tools."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from foundation import memory as memory_store
from foundation.models import Run
from mason.client import ToolCall
from mason.config import AgentConfig
from mason.prompts import system_messages
from mason.session import MasonSession
from mason.tools import build_toolbox
from slab.config import HpcConfig


def _utc_today() -> date:
    """Today in UTC, the day the memory store dates outages by."""
    return datetime.now(UTC).date()



@pytest.fixture()
def memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the store at a directory of this test's own."""
    root = tmp_path / "memory"
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(root))
    return root


def _session(tmp_path: Path, agent: AgentConfig | None = None, **kwargs: object) -> MasonSession:
    return MasonSession(
        tmp_path,
        agent=agent or AgentConfig(),
        hpc=HpcConfig(),
        workspace_root=tmp_path / ".slab",
        **kwargs,  # type: ignore[arg-type]
    )


def _call(tool: str, /, **arguments: object) -> ToolCall:
    # Positional-only: 'name' is an argument of both tools under test.
    return ToolCall(
        id="t1", name=tool, arguments=dict(arguments), arguments_raw=json.dumps(arguments)
    )


# -- the prompt --------------------------------------------------------------


def test_the_prompt_carries_the_catalog_when_memories_exist(
    tmp_path: Path, memory_root: Path
) -> None:
    memory_store.write(
        "vllm-mamba-cache",
        "vLLM refuses hybrid-Mamba models at the default batch size.",
        "Lower max-num-seqs.",
        directory=memory_root,
        evidence="checked by hand",
    )
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "# Memory" in content
    assert "- vllm-mamba-cache: vLLM refuses hybrid-Mamba models" in content
    # The trigger line only: the fact itself waits for recall.
    assert "Lower max-num-seqs." not in content


def test_empty_store_still_shows_the_memory_surface(tmp_path: Path, memory_root: Path) -> None:
    """A fresh machine's agent still needs to see 'this surface exists,
    call remember when you find a machine fact worth keeping' — otherwise
    the tool never gets called the first time a blocker is solved."""
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "# Memory" in content
    assert "No machine facts recorded on this machine yet" in content
    assert "remember" in content


def test_memory_false_removes_the_block_and_the_tools(
    tmp_path: Path, memory_root: Path
) -> None:
    memory_store.write(
        "a-fact", "A fact.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    session = _session(tmp_path, AgentConfig(memory=False))
    (content,) = [m["content"] for m in system_messages(session)]
    assert "# Memory" not in content
    box = build_toolbox(session)
    assert "recall" not in box.tools and "remember" not in box.tools
    # The store keeps what it holds; only this session is blind to it.
    assert list(memory_store.discover(memory_root)) == ["a-fact"]


def test_roster_tables_cannot_override_memory() -> None:
    with pytest.raises(Exception, match="memory"):
        AgentConfig.model_validate({"roster": {"pi": {"memory": False}}})


# -- recall ------------------------------------------------------------------


def test_recall_returns_the_fact_and_who_recorded_it(
    tmp_path: Path, memory_root: Path
) -> None:
    memory_store.write(
        "srun-in-sandbox",
        "srun cannot reach the controller from inside the sandbox.",
        "Use an mpirun-style command sized to the job's allocation.",
        agent="md-expert",
        model="qwen3-30b",
        directory=memory_root,
        evidence="checked by hand",
    )
    box = build_toolbox(_session(tmp_path))
    answer = box.dispatch(_call("recall", name="srun-in-sandbox"))
    assert "mpirun-style command" in answer
    assert "recorded by md-expert on " in answer
    assert "model qwen3-30b" in answer


def test_recall_of_an_unknown_name_lists_what_exists(
    tmp_path: Path, memory_root: Path
) -> None:
    memory_store.write(
        "a-fact", "A fact.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    box = build_toolbox(_session(tmp_path))
    answer = box.dispatch(_call("recall", name="no-such-fact"))
    assert "no memory named 'no-such-fact'" in answer
    assert "a-fact" in answer


def test_recall_says_so_when_the_machine_knows_nothing(
    tmp_path: Path, memory_root: Path
) -> None:
    box = build_toolbox(_session(tmp_path))
    assert "none recorded yet" in box.dispatch(_call("recall", name="anything"))


# -- remember ----------------------------------------------------------------


def test_remember_writes_the_fact_with_its_attribution(
    tmp_path: Path, memory_root: Path
) -> None:
    session = _session(tmp_path, auto_approve=True)
    session.agent = session.agent.model_copy(update={"model": "qwen3-30b"})
    session.agent_name = "md-expert"
    box = build_toolbox(session)
    answer = box.dispatch(
        _call(
            "remember",
            name="lammps-potentials",
            description="The lammps engine needs an absolute potential path here.",
            body="Relative pair_coeff paths resolve against the run's scratch, not the project.",
            evidence="a relative path failed and the absolute one ran",
        )
    )
    assert "recorded as memory 'lammps-potentials'" in answer

    memory = memory_store.discover(memory_root)["lammps-potentials"]
    assert memory.agent == "md-expert"
    assert memory.model == "qwen3-30b"
    assert "Relative pair_coeff paths" in memory.body()
    # The next session's prompt carries it without anything else happening.
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "- lammps-potentials: The lammps engine needs" in content


def test_remember_asks_before_it_writes(tmp_path: Path, memory_root: Path) -> None:
    asked: list[tuple[str, str]] = []

    def refuse(tool: str, preview: str) -> bool:
        asked.append((tool, preview))
        return False

    session = _session(tmp_path, approver=refuse)
    session.agent_name = "dft-expert"
    session._parent = _session(tmp_path)  # a delegated specialist is attributed
    box = build_toolbox(session)
    answer = box.dispatch(
        _call("remember", name="a-fact", description="A fact.", body="The whole fact.")
    )
    assert "was not approved" in answer
    assert memory_store.discover(memory_root) == {}

    (tool, preview) = asked[0]
    assert tool == "remember"
    # The human sees who asks and the full text they are about to publish
    # into every later session's prompt.
    assert preview.startswith("[dft-expert] ")
    assert "a-fact" in preview and "A fact." in preview and "The whole fact." in preview


def test_a_refused_memory_teaches_the_rule(tmp_path: Path, memory_root: Path) -> None:
    box = build_toolbox(_session(tmp_path, auto_approve=True))
    answer = box.dispatch(
        _call("remember", name="Not A Name", description="d", body="The fact.", evidence="e")
    )
    assert answer.startswith("not recorded: ")
    assert "lowercase alphanumerics" in answer
    assert not memory_root.exists()


def test_remember_then_recall_round_trips_through_the_transcript(
    tmp_path: Path, memory_root: Path
) -> None:
    session = _session(tmp_path, auto_approve=True)
    box = build_toolbox(session)
    box.dispatch(
        _call("remember", name="a-fact", description="A fact.", body="The fact.", evidence="e")
    )
    box.dispatch(_call("recall", name="a-fact"))
    events = [
        json.loads(line)
        for line in session.transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [e["type"] for e in events] == ["remember", "recall"]
    assert events[0]["path"].endswith("a-fact.md")


# -- version stamps ----------------------------------------------------------


def _versions(monkeypatch: pytest.MonkeyPatch, live: dict[str, str] | None) -> list[int]:
    """Make the machine report *live*; None makes the probe an error."""
    import slab._ops

    calls: list[int] = []

    def probe() -> dict[str, str]:
        calls.append(1)
        if live is None:
            raise AssertionError("the probe must not run")
        return dict(live)

    monkeypatch.setattr(slab._ops, "software_versions", probe)
    return calls


def test_remember_stamps_the_software_the_fact_names(
    tmp_path: Path, memory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _versions(monkeypatch, {"gracemaker": "0.6.0", "atomsk": "0.13.1"})
    session = _session(tmp_path, auto_approve=True)
    box = build_toolbox(session)
    answer = box.dispatch(
        _call(
            "remember",
            name="grace-gpu-growth",
            description="gracemaker needs TF_FORCE_GPU_ALLOW_GROWTH on the GPU nodes.",
            body="Without it the second fit on a node fails to allocate.",
            evidence="the second fit failed without it and ran with it",
        )
    )
    assert "(stamped against gracemaker 0.6.0)" in answer
    memory = memory_store.discover(memory_root)["grace-gpu-growth"]
    assert memory.against == {"gracemaker": "0.6.0"}
    # A second write in the same session reuses the probe.
    box.dispatch(
        _call("remember", name="b", description="An atomsk fact.", body="Body.", evidence="e")
    )
    assert memory_store.discover(memory_root)["b"].against == {"atomsk": "0.13.1"}
    assert calls == [1]


def test_the_prompt_flags_a_memory_whose_software_changed(
    tmp_path: Path, memory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_store.write(
        "grace-gpu-growth", "gracemaker needs X.", "Body.", against={"gracemaker": "0.5.2"},
        directory=memory_root,
        evidence="checked by hand",
    )
    memory_store.write(
        "vllm-cache", "vLLM refuses a big batch.", "Body.", directory=memory_root,
        evidence="checked by hand",
    )
    _versions(monkeypatch, {"gracemaker": "0.6.0"})
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert (
        "- grace-gpu-growth: gracemaker needs X. "
        "[changed since: gracemaker was 0.5.2, now 0.6.0]"
    ) in content
    assert "- vllm-cache: vLLM refuses a big batch.\n" in content + "\n"


def test_unstamped_memories_cost_no_probe(
    tmp_path: Path, memory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_store.write(
        "vllm-cache", "vLLM refuses a big batch.", "Body.", directory=memory_root,
        evidence="checked by hand",
    )
    _versions(monkeypatch, None)
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "- vllm-cache: vLLM refuses a big batch." in content
    box = build_toolbox(_session(tmp_path))
    assert "[recorded" in box.dispatch(_call("recall", name="vllm-cache"))


def test_recall_says_what_changed_since(
    tmp_path: Path, memory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_store.write(
        "grace-gpu-growth", "gracemaker needs X.", "The whole fact.",
        agent="pi", against={"gracemaker": "0.5.2"}, directory=memory_root,
        evidence="checked by hand",
    )
    _versions(monkeypatch, {"gracemaker": "0.6.0"})
    box = build_toolbox(_session(tmp_path))
    answer = box.dispatch(_call("recall", name="grace-gpu-growth"))
    assert "[recorded by pi on " in answer
    assert "against gracemaker 0.5.2, evidence: checked by hand]" in answer
    assert "[changed since: gracemaker was 0.5.2, now 0.6.0. Confirm the fact" in answer


def test_a_delegate_shares_the_parent_probe(
    tmp_path: Path, memory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _versions(monkeypatch, {"gracemaker": "0.6.0"})
    parent = _session(tmp_path)
    assert parent.software_versions() == {"gracemaker": "0.6.0"}
    child = parent.spawn("dft-expert", AgentConfig())
    assert child.software_versions() == {"gracemaker": "0.6.0"}
    assert calls == [1]


# -- what counts as evidence -------------------------------------------------


def _run(session: MasonSession, name: str, status: str, host: str | None = None) -> str:
    """One run in the session's workspace, left in *status*."""
    from foundation.runtime import Workspace

    with Workspace(session.workspace_root) as ws:
        run = ws.runs.create(Run(name=name))
        # The host is stamped when a run starts and stays with it after.
        ws.runs.set_status(run.id, "running", host=host)
        if status != "running":
            ws.runs.set_status(run.id, status)
        return run.id


def test_a_running_run_is_no_evidence_and_the_reply_names_it(
    tmp_path: Path, memory_root: Path
) -> None:
    session = _session(tmp_path, auto_approve=True)
    run_id = _run(session, "melt", "running")
    answer = build_toolbox(session).dispatch(
        _call(
            "remember",
            name="device-init-fails",
            description="A node refuses to initialise its GPUs.",
            body="Every launch there dies before the first step.",
            evidence=f"run {run_id} died at once",
        )
    )
    assert "marked unverified until a completed run confirms it" in answer
    assert f"{run_id} does not (running, so it has not confirmed anything)" in answer
    assert memory_store.discover(memory_root)["device-init-fails"].unverified is True


def test_a_completed_run_verifies_the_memory(tmp_path: Path, memory_root: Path) -> None:
    session = _session(tmp_path, auto_approve=True)
    run_id = _run(session, "melt", "completed")
    answer = build_toolbox(session).dispatch(
        _call(
            "remember",
            name="scratch-quota",
            description="The scratch filesystem here fills at 80 percent.",
            body="Writes fail above it until the sweep runs.",
            evidence=f"run {run_id} completed after the sweep",
        )
    )
    assert "unverified" not in answer
    assert f"{run_id} counts (completed)" in answer
    assert memory_store.discover(memory_root)["scratch-quota"].unverified is False


def test_a_dry_run_id_is_kept_in_the_text_and_counts_for_nothing(
    tmp_path: Path, memory_root: Path
) -> None:
    session = _session(tmp_path, auto_approve=True)
    answer = build_toolbox(session).dispatch(
        _call(
            "remember",
            name="loader-fails",
            description="The GPU build cannot find its libraries.",
            body="The loader error names the missing object.",
            evidence="dry-20260917-121314-ab12 kept the screen file",
        )
    )
    assert "dry-20260917-121314-ab12 does not (a dry run, which confirms nothing)" in answer
    memory = memory_store.discover(memory_root)["loader-fails"]
    assert memory.unverified is True
    assert "dry-20260917-121314-ab12" in (memory.evidence or "")


def test_an_outage_takes_the_host_from_its_evidence_and_expires(
    tmp_path: Path, memory_root: Path
) -> None:
    session = _session(tmp_path, auto_approve=True)
    run_id = _run(session, "melt", "completed", host="n1")
    answer = build_toolbox(session).dispatch(
        _call(
            "remember",
            name="device-init-fails",
            description="A node refuses to initialise its GPUs.",
            body="Every launch there dies before the first step.",
            evidence=f"run {run_id} failed to initialise the device",
            kind="outage",
        )
    )
    assert "recorded as an outage (outage recorded " in answer
    memory = memory_store.discover(memory_root)["device-init-fails"]
    assert memory.kind == "outage"
    assert memory.where == "n1"
    assert memory.expires_at is not None

    read = build_toolbox(session).dispatch(_call("recall", name="device-init-fails"))
    assert read.startswith("outage recorded ")
    assert "on n1;" in read.splitlines()[0]


def test_the_catalog_stops_carrying_an_outage_after_its_day(
    tmp_path: Path, memory_root: Path
) -> None:
    memory_store.write(
        "device-init-fails", "A node refuses to initialise its GPUs.", "Body.",
        evidence="run 01k2x7abcd", kind="outage", where="n1",
        expires_at=_utc_today() - timedelta(days=1), directory=memory_root,
    )
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "device-init-fails" not in content


def test_a_memory_that_restates_a_skill_is_refused_by_the_tool(
    tmp_path: Path, memory_root: Path
) -> None:
    session = _session(tmp_path, auto_approve=True)
    run_id = _run(session, "melt", "completed")
    answer = build_toolbox(session).dispatch(
        _call(
            "remember",
            name="cna-codes",
            description="The build shifts the cna/atom codes.",
            body="Here 1 is hcp and 2 is fcc, not the documented mapping.",
            evidence=f"run {run_id} showed it",
        )
    )
    assert answer.startswith("not recorded: this is documented behaviour")
    assert "two-phase-melting section 3" in answer
    assert memory_store.discover(memory_root) == {}


def test_the_memories_written_block_carries_the_kind_and_the_evidence_states(
    tmp_path: Path, memory_root: Path
) -> None:
    from mason.tools import memories_written_block

    session = _session(tmp_path, auto_approve=True)
    run_id = _run(session, "melt", "running", host="n1")
    box = build_toolbox(session)
    box.dispatch(
        _call(
            "remember",
            name="device-init-fails",
            description="A node refuses to initialise its GPUs.",
            body="Every launch there dies before the first step.",
            evidence=f"run {run_id} died at once",
            kind="outage",
        )
    )
    block = memories_written_block(session, session.memories_written)
    (_, line) = block.splitlines()
    assert line.startswith("- device-init-fails (pi, outage): A node refuses")
    # The host is the provenance of the citation, not its verification: a
    # run that has not finished still says where it is going wrong.
    assert "[on n1, expires " in line
    assert "[unverified]" in line
    assert f"[runs now: {run_id}: melt, running," in line


def test_an_outage_takes_its_host_from_the_failed_run_that_showed_it(
    tmp_path: Path, memory_root: Path
) -> None:
    """The plan's own case: CUDA fails to initialise, the run dies, and the
    outage must still name the node the run died on."""
    from mason.tools import memories_written_block

    session = _session(tmp_path, auto_approve=True)
    run_id = _run(session, "device-probe", "failed", host="n1")
    answer = build_toolbox(session).dispatch(
        _call(
            "remember",
            name="device-init-fails",
            description="A node refuses to initialise its GPUs.",
            body="Every launch there dies before the first step.",
            evidence=f"run {run_id} died at once on that node",
            kind="outage",
        )
    )
    assert "recorded as an outage (outage recorded " in answer
    assert " on n1; expires " in answer
    assert "it is marked unverified until a completed run confirms it" in answer
    assert f"{run_id} does not (failed, so it has not confirmed anything)" in answer
    written = memory_store.discover(memory_root)["device-init-fails"]
    assert written.where == "n1" and written.unverified is True
    (_, line) = memories_written_block(session, session.memories_written).splitlines()
    assert "[on n1, expires " in line and "[unverified]" in line


def test_an_outage_without_a_host_stamp_says_so(tmp_path: Path, memory_root: Path) -> None:
    from mason.tools import memories_written_block

    session = _session(tmp_path, auto_approve=True)
    run_id = _run(session, "device-probe", "completed")
    build_toolbox(session).dispatch(
        _call(
            "remember",
            name="device-init-fails",
            description="A node refuses to initialise its GPUs.",
            body="Every launch there dies before the first step.",
            evidence=f"run {run_id} showed it",
            kind="outage",
        )
    )
    assert memory_store.discover(memory_root)["device-init-fails"].where is None
    (_, line) = memories_written_block(session, session.memories_written).splitlines()
    assert "[on an unrecorded host, expires " in line
