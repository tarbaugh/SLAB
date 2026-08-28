"""Machine memory as the agent meets it: the prompt block and the two tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundation import memory as memory_store
from mason.client import ToolCall
from mason.config import AgentConfig
from mason.prompts import system_messages
from mason.session import MasonSession
from mason.tools import build_toolbox
from slab.config import HpcConfig


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
    )
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "# Memory" in content
    assert "- vllm-mamba-cache: vLLM refuses hybrid-Mamba models" in content
    # The trigger line only: the fact itself waits for recall.
    assert "Lower max-num-seqs." not in content


def test_an_empty_store_adds_no_section(tmp_path: Path, memory_root: Path) -> None:
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "# Memory" not in content


def test_memory_false_removes_the_block_and_the_tools(
    tmp_path: Path, memory_root: Path
) -> None:
    memory_store.write("a-fact", "A fact.", "Body.", directory=memory_root)
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
    )
    box = build_toolbox(_session(tmp_path))
    answer = box.dispatch(_call("recall", name="srun-in-sandbox"))
    assert "mpirun-style command" in answer
    assert "recorded by md-expert on " in answer
    assert "model qwen3-30b" in answer


def test_recall_of_an_unknown_name_lists_what_exists(
    tmp_path: Path, memory_root: Path
) -> None:
    memory_store.write("a-fact", "A fact.", "Body.", directory=memory_root)
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
        _call("remember", name="Not A Name", description="d", body="The fact.")
    )
    assert answer.startswith("not recorded: ")
    assert "lowercase alphanumerics" in answer
    assert not memory_root.exists()


def test_remember_then_recall_round_trips_through_the_transcript(
    tmp_path: Path, memory_root: Path
) -> None:
    session = _session(tmp_path, auto_approve=True)
    box = build_toolbox(session)
    box.dispatch(_call("remember", name="a-fact", description="A fact.", body="The fact."))
    box.dispatch(_call("recall", name="a-fact"))
    events = [
        json.loads(line)
        for line in session.transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [e["type"] for e in events] == ["remember", "recall"]
    assert events[0]["path"].endswith("a-fact.md")
