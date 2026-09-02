"""Delegation: the PI hands one scoped task down, one level, sequentially."""

import json
from pathlib import Path
from typing import Any

import pytest

from mason.client import ChatReply, LlmError, ToolCall
from mason.config import MasonConfig
from mason.loop import Mason
from mason.roster import discover_roster
from mason.session import MasonSession
from slab.config import HpcConfig


class FakeClient:
    """One shared script: parent and delegated child consume it in order."""

    def __init__(self, replies: list[ChatReply | Exception]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatReply:
        self.requests.append([dict(m) for m in messages])
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _session(tmp_path: Path, **agent: object) -> MasonSession:
    config = MasonConfig.model_validate({"agent": {"model": "fake", **agent}})
    return MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent,
        hpc=HpcConfig(), auto_approve=True,
    )


def _call(name: str, **arguments: object) -> ChatReply:
    return ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id=f"c_{name}", name=name, arguments=dict(arguments),
                arguments_raw=json.dumps(arguments),
            ),
        ),
        prompt_tokens=100,
        completion_tokens=10,
    )


def _text(text: str) -> ChatReply:
    return ChatReply(content=text, prompt_tokens=100, completion_tokens=10)


def _delegated_turn(tmp_path: Path, **agent: object) -> tuple[Mason, FakeClient, Any]:
    """One full PI -> md-expert -> PI round trip on a shared scripted client."""
    session = _session(tmp_path, **agent)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="report the melting feel"),
            _call("finish", report="MSD says solid (run ab12cd)"),
            _text("md-expert reports: solid (run ab12cd)"),
        ]
    )
    mason = Mason(session, client=client)
    result = mason.run_turn("is it molten?")
    return mason, client, result


def test_the_report_reaches_the_pi_with_a_harness_footer(tmp_path: Path) -> None:
    _mason, client, result = _delegated_turn(tmp_path)
    assert result.stop_reason == "answer"
    tool_result = next(
        m for m in client.requests[-1]
        if m.get("role") == "tool" and m.get("tool_call_id") == "c_delegate"
    )
    content = str(tool_result["content"])
    assert "MSD says solid (run ab12cd)" in content
    assert "[md-expert: finish after 1 step(s);" in content
    assert "transcript" in content


def test_the_child_ran_as_the_specialist(tmp_path: Path) -> None:
    _mason, client, _result = _delegated_turn(tmp_path)
    # Request 2 of 3 is the child's only call: its system prompt is the
    # specialist's, and the brief is the delegated task.
    child_system = client.requests[1][0]["content"]
    assert "molecular-dynamics specialist" in child_system
    assert "# Your team" not in child_system  # no onward delegation promised
    child_goal = client.requests[1][1]["content"]
    assert child_goal == "report the melting feel"


def test_usage_rolls_up_and_both_transcripts_exist(tmp_path: Path) -> None:
    mason, _client, _result = _delegated_turn(tmp_path)
    session = mason.session
    # 3 model calls x (100 + 10) all count at the session level.
    assert session.prompt_tokens == 300
    assert session.completion_tokens == 30
    child_path = session.transcript_path.with_name(
        f"{session.transcript_path.stem}-md-expert-1.jsonl"
    )
    assert child_path.is_file()
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    (delegated,) = [e for e in events if e["type"] == "delegate"]
    assert delegated["agent"] == "md-expert"
    assert delegated["stop"] == "finish"
    assert delegated["transcript"] == child_path.name


def test_resume_never_picks_a_delegation_transcript(tmp_path: Path) -> None:
    mason, _client, _result = _delegated_turn(tmp_path)
    session = mason.session
    child_path = session.transcript_path.with_name(
        f"{session.transcript_path.stem}-md-expert-1.jsonl"
    )
    # Make the child transcript the newest file; the parent must still win.
    child_path.touch()
    latest = session.latest_transcript()
    assert latest is not None
    assert latest.name == session.transcript_path.name


def test_a_child_harness_stop_is_reported_not_hidden(tmp_path: Path) -> None:
    session = _session(tmp_path, max_turns=1)  # the child inherits max_turns=1
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="loop forever"),
            _call("list_dir"),  # the child burns its single call
            # ...child stops at its budget; the PI would answer next, but its
            # own max_turns=1 ends the parent turn too with the budget text.
        ]
    )
    mason = Mason(session, client=client)
    result = mason.run_turn("go")
    assert result.stop_reason == "max_turns"
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    (delegated,) = [e for e in events if e["type"] == "delegate"]
    assert delegated["stop"] == "max_turns"


def test_a_crashing_child_client_becomes_tool_failure_evidence(tmp_path: Path) -> None:
    session = _session(tmp_path)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="x"),
            LlmError("the server went away"),
            _text("could not delegate; stopping"),
        ]
    )
    mason = Mason(session, client=client)
    result = mason.run_turn("go")
    assert result.stop_reason == "answer"
    tool_result = next(
        m for m in client.requests[-1]
        if m.get("role") == "tool" and m.get("tool_call_id") == "c_delegate"
    )
    assert "tool delegate failed: LlmError: the server went away" in str(tool_result["content"])


def test_unknown_and_self_targets_answer_with_the_team(tmp_path: Path) -> None:
    session = _session(tmp_path)
    client = FakeClient(
        [
            _call("delegate", agent="nobody", task="x"),
            _call("delegate", agent="pi", task="x"),
            _text("ok"),
        ]
    )
    mason = Mason(session, client=client)
    mason.run_turn("go")
    results = [
        str(m["content"])
        for request in client.requests
        for m in request
        if m.get("role") == "tool" and m.get("tool_call_id") == "c_delegate"
    ]
    assert any("no agent named 'nobody'" in r and "md-expert" in r for r in results)
    assert any("cannot delegate to yourself" in r for r in results)


def test_depth_one_agents_never_delegate_or_plan(tmp_path: Path) -> None:
    """The depth rule is structural: even the pi card, delegated to, loses both."""
    session = _session(tmp_path)
    roster = discover_roster(tmp_path)
    child_session = session.spawn("pi", session.agent)
    child = Mason(
        child_session, client=FakeClient([]), spec=roster["pi"], roster=roster, depth=1
    )
    assert "delegate" not in child.toolbox.tools
    assert "plan" not in child.toolbox.tools
    assert "notebook" in child.toolbox.tools


def test_the_delegation_switch_removes_the_tool_and_the_promise(tmp_path: Path) -> None:
    session = _session(tmp_path, delegation=False)
    mason = Mason(session, client=FakeClient([]))
    assert "delegate" not in mason.toolbox.tools
    (system,) = mason.messages
    assert "# Your team" not in system["content"]


def test_child_read_files_guard_is_fresh(tmp_path: Path) -> None:
    """A specialist must read a file before editing it, even if the PI read it."""
    target = tmp_path / "notes.txt"
    target.write_text("x = 1\n")
    session = _session(tmp_path)
    session.read_files.add(target)
    child_session = session.spawn("md-expert", session.agent)
    assert target not in child_session.read_files
    from mason.tools import build_toolbox

    box = build_toolbox(child_session, depth=1)
    call = ToolCall(
        id="e1", name="edit_file",
        arguments={"path": str(target), "old_string": "x = 1", "new_string": "x = 2"},
        arguments_raw="{}",
    )
    assert "staleness guard" in box.dispatch(call)


def test_child_approvals_carry_the_agent_name(tmp_path: Path) -> None:
    asked: list[tuple[str, str]] = []

    def approver(tool: str, preview: str) -> bool:
        asked.append((tool, preview))
        return False

    config = MasonConfig.model_validate({"agent": {"model": "fake"}})
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent,
        hpc=HpcConfig(), approver=approver,
    )
    child_session = session.spawn("dft-expert", session.agent)
    from mason.tools import build_toolbox

    box = build_toolbox(child_session, depth=1)
    call = ToolCall(
        id="w1", name="write_file",
        arguments={"path": str(tmp_path / "a.txt"), "content": "hi"}, arguments_raw="{}",
    )
    answer = box.dispatch(call)
    assert "not approved" in answer
    (record,) = asked
    assert record[1].startswith("[dft-expert] ")
    # The parent's own previews carry no bracket:
    parent_box = build_toolbox(session)
    parent_box.dispatch(call)
    assert not asked[-1][1].startswith("[")


def test_child_notebook_entries_are_attributed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    child_session = session.spawn("dft-expert", session.agent)
    child_session.notebook_append("the k-mesh converged at 6x6x6", heading="convergence")
    text = (tmp_path / "NOTEBOOK.md").read_text()
    assert "convergence [dft-expert]" in text


def test_children_derive_from_base_config_not_the_entry_table(tmp_path: Path) -> None:
    """[agent.roster.pi] must not leak into a specialist's effective config."""
    session = _session(
        tmp_path,
        temperature=0.4,
        roster={"pi": {"temperature": 0.9}, "md-expert": {"max_turns": 5}},
    )
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="t"),
            _call("finish", report="done"),
            _text("ok"),
        ]
    )
    mason = Mason(session, client=client)
    assert mason.session.agent.temperature == 0.9  # the entry override applied
    mason.run_turn("go")
    # The child's system prompt request came from a session whose agent was
    # base + md table: temperature stays 0.4, max_turns becomes 5.
    events_path = mason.session.transcript_path.with_name(
        f"{mason.session.transcript_path.stem}-md-expert-1.jsonl"
    )
    assert events_path.is_file()
    # Reach the child config through the recorded spawn: rebuild it the same way.
    from mason.config import roster_agent_config

    child_agent = roster_agent_config(session.base_agent, "md-expert")
    assert child_agent.temperature == 0.4
    assert child_agent.max_turns == 5


def test_flags_apply_to_children_too(tmp_path: Path) -> None:
    session = _session(tmp_path, roster={"md-expert": {"max_turns": 5}})
    session.flag_updates = {"max_turns": 2}
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="loop"),
            _call("list_dir"),
            _call("list_dir"),  # the child's second call hits the flag budget
            _text("child was stopped by the flag budget"),
        ]
    )
    mason = Mason(session, client=client)
    mason.run_turn("go")
    events = [
        json.loads(line)
        for line in mason.session.transcript_path.read_text().splitlines()
    ]
    (delegated,) = [e for e in events if e["type"] == "delegate"]
    assert delegated["stop"] == "max_turns"
    assert delegated["steps"] == 2


def test_the_shared_client_is_reused_for_matching_profiles(tmp_path: Path) -> None:
    """One server, one client object: the child saw the same FakeClient."""
    _mason, client, _result = _delegated_turn(tmp_path)
    # All three requests landed on the single shared client instance.
    assert len(client.requests) == 3


def test_a_differing_profile_builds_a_fresh_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []

    def fake_builder(agent: Any, keys: object = None) -> FakeClient:
        built.append(agent.model)
        return FakeClient([_call("finish", report="from the other model")])

    monkeypatch.setattr("mason.loop.client_from_config", fake_builder)
    session = _session(tmp_path, roster={"md-expert": {"model": "bigger-model"}})
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="t"),
            _text("ok"),
        ]
    )
    mason = Mason(session, client=client)
    result = mason.run_turn("go")
    assert result.stop_reason == "answer"
    assert built == ["bigger-model"]
    assert len(client.requests) == 2  # the parent's two calls only
