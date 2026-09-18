"""The prompt a lead reads: free resources at every planner step, and the notebook it inherits."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from foundation import Workspace
from foundation.project import notebook_append
from mason.client import ChatReply, ToolCall
from mason.config import MasonConfig
from mason.loop import Mason
from mason.prompts import WAVE_RULE, environment_block, free_hint
from mason.roster import discover_roster
from mason.session import MasonSession


class _Script:
    """Answers from a script and keeps every request it saw."""

    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: object = None, **_: Any) -> ChatReply:
        self.requests.append([dict(m) for m in messages])
        return self.replies.pop(0)


def _session(project: Path, **agent: object) -> MasonSession:
    config = MasonConfig.model_validate({"agent": {"model": "fake", **agent}})
    return MasonSession(
        project, workspace_root=project / ".slab", agent=config.agent, auto_approve=True
    )


def _call(name: str, **arguments: object) -> ChatReply:
    call = ToolCall(
        id=f"call_{name}", name=name, arguments=dict(arguments), arguments_raw=json.dumps(arguments)
    )
    return ChatReply(content=None, tool_calls=(call,), prompt_tokens=100, completion_tokens=10)


def _text(text: str) -> ChatReply:
    return ChatReply(content=text, prompt_tokens=100, completion_tokens=10)


# -- free resources at every planner step ----------------------------------------


def test_the_free_line_reads_the_store_now(tmp_path: Path, no_gpus: None) -> None:
    session = _session(tmp_path)
    line = free_hint(session)
    cpus = int(line.split("free right now: ")[1].split()[0])
    assert line.startswith("[harness: free right now: ")
    assert f"of {cpus} cpu(s), 0 of 0 gpu(s)" in line
    assert WAVE_RULE in line
    with Workspace(session.workspace_root) as ws:
        ws.reserve(ntasks=1, holder_pid=os.getpid())
    assert f"free right now: {cpus - 1} of {cpus} cpu(s)" in free_hint(session)


def test_the_resource_paragraph_states_the_wave_rule(tmp_path: Path, no_gpus: None) -> None:
    block = environment_block(_session(tmp_path))
    assert "Size a wave to the free budget: as many concurrent launches as free GPUs" in block


def test_every_planner_step_carries_what_is_free(tmp_path: Path, no_gpus: None) -> None:
    """A planner briefed two GPUs from an old intent while a third sat free."""
    roster = discover_roster(tmp_path)
    session = _session(tmp_path)

    class _TakesASlice(_Script):
        def chat(self, messages: list[dict[str, Any]], tools: object = None, **_: Any) -> ChatReply:
            if not self.requests:  # after the first hint was read, before the second
                with Workspace(session.workspace_root) as ws:
                    ws.reserve(ntasks=1, holder_pid=os.getpid())
            return super().chat(messages, tools)

    client = _TakesASlice([_call("list_runs"), _call("list_runs"), _text("planned")])
    Mason(session, client=client, spec=roster["planner"], roster=roster).run_turn("plan it")
    hints = [request[-1]["content"] for request in client.requests]
    assert len(hints) == 3
    assert all(hint.startswith("[harness: model call ") for hint in hints)
    assert all("free right now: " in hint and WAVE_RULE in hint for hint in hints)
    # Read at each step, so a slice taken mid-session shows at the next one.
    free = [int(hint.split("free right now: ")[1].split()[0]) for hint in hints]
    assert free[1] == free[2] == free[0] - 1


def test_a_lead_that_launches_reads_free_resources_itself(tmp_path: Path, no_gpus: None) -> None:
    client = _Script([_text("done")])
    Mason(_session(tmp_path), client=client).run_turn("look")  # the pi
    assert "free right now" not in client.requests[0][-1]["content"]


# -- the notebook a plan-writing card inherits ------------------------------------


def _notebook_with_an_old_probe(project: Path) -> None:
    notebook_append(
        project,
        "Coexistence probe at 1180 K stays two-phase; the model's melting point "
        "is near 1180 K, below experiment (run 01abc).",
        heading=f"{project.name} coexistence probe",
    )
    notebook_append(project, "Another campaign's note on Cu.", heading="Cu lattice")
    for n in range(8):
        notebook_append(project, f"step {n}: " + "x" * 500, heading=f"{project.name} step {n}")


def test_the_environment_carries_the_prior_findings(tmp_path: Path) -> None:
    project = tmp_path / "w-mlip"
    project.mkdir()
    _notebook_with_an_old_probe(project)
    block = environment_block(_session(project), inherit=True)
    prior = block.split("# Prior findings (notebook)")[1].split("# Lab notebook")[0]
    assert "whose heading names w-mlip" in prior
    assert "UTC — w-mlip coexistence probe\nCoexistence probe at 1180 K" in prior
    assert "Cu lattice" not in prior  # another campaign's heading
    assert "step 7" not in prior  # the latest entries show it already
    latest = block.split("# Lab notebook (latest entries)")[1]
    assert "step 7" in latest and "1180 K" not in latest
    assert "# Prior findings" not in environment_block(_session(project))


def test_a_small_notebook_needs_no_prior_block(tmp_path: Path) -> None:
    notebook_append(tmp_path, "a = 3.61 Å (run 01abc)", heading="Cu lattice")
    block = environment_block(_session(tmp_path), inherit=True)
    assert "# Prior findings" not in block and "a = 3.61 Å" in block


def test_a_planner_sees_the_prior_findings_in_its_first_prompt(
    tmp_path: Path, no_gpus: None
) -> None:
    project = tmp_path / "w-mlip"
    project.mkdir()
    _notebook_with_an_old_probe(project)
    roster = discover_roster(project)
    client = _Script([_text("planned")])
    Mason(_session(project), client=client, spec=roster["planner"], roster=roster).run_turn("go")
    system = client.requests[0][0]["content"]
    assert "# Prior findings (notebook)" in system
    assert "the model's melting point is near 1180 K" in system
    # A specialist does not write the plan and keeps the plain tail.
    worker = Mason(
        _session(tmp_path / "w-mlip"), client=_Script([]), spec=roster["worker"], roster=roster,
        depth=1,
    )
    assert "# Prior findings" not in worker.messages[0]["content"]


@pytest.fixture()
def no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS", "SLURM_NTASKS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_a_specialist_team_block_lists_the_helpers_only(tmp_path: Path) -> None:
    from mason.prompts import team_block

    roster = discover_roster(tmp_path)
    block = team_block(roster["md-expert"], roster, depth=1)
    assert block.startswith("# Your team\n")
    assert "Your team takes scripts, not studies." in block
    assert "- coding-expert: Writes, fixes, and checks scripts" in block
    assert "- worker:" not in block and "critic" not in block
    # A helper has no team, and a lead's block is unchanged by helpers.
    assert team_block(roster["coding-expert"], roster, depth=1) == ""
    assert "Specialists you can hand a scoped task to" in team_block(roster["pi"], roster)
    assert "- coding-expert:" in team_block(roster["pi"], roster)
