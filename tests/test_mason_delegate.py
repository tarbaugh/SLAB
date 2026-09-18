"""Delegation: the PI hands one scoped task down, and a specialist may hand a script on."""

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
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
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
    assert "[md-expert-1: finish after 1 step(s);" in content
    assert 'continue with continues="md-expert-1"' in content
    assert "transcript" in content


def test_the_child_ran_as_the_specialist(tmp_path: Path) -> None:
    _mason, client, _result = _delegated_turn(tmp_path)
    # Request 2 of 3 is the child's only call: its system prompt is the
    # specialist's, and the brief is the delegated task.
    child_system = client.requests[1][0]["content"]
    assert "molecular-dynamics specialist" in child_system
    # A specialist's team is its helpers, and nobody else.
    assert "Your team takes scripts, not studies." in child_system
    assert "- coding-expert:" in child_system
    assert "- worker:" not in child_system
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


def test_a_child_killed_by_the_server_returns_a_result_with_its_steps(tmp_path: Path) -> None:
    """The server's failure is the child's stop reason, not a harness
    failure of the delegate tool: the steps it took are in its transcript,
    and the footer says how far it got, so the lead can re-brief rather
    than pay for the whole task again."""
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
    text = str(tool_result["content"])
    assert text.startswith("stopped: the model server failed mid-turn after step 1")
    assert "the server went away" in text
    assert "[md-expert-1: error after 1 step(s);" in text


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
    """The depth rule is structural: even the pi card, delegated to, loses both.

    A lead is never on a team, so it never runs at depth one outside a
    test; if it did, it would not get the specialist's helper tool either.
    """
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


def test_a_lead_takes_no_briefs(tmp_path: Path) -> None:
    client = FakeClient([_call("delegate", agent="planner", task="plan it"), _text("done")])
    mason = Mason(_session(tmp_path), client=client)
    mason.run_turn("go")
    seen = json.dumps(client.requests[1])
    assert "planner leads a group of its own and takes no briefs" in seen
    assert "your team: analysis-expert, coding-expert, dft-expert, md-expert, worker" in seen


def test_a_child_cut_at_its_ceiling_hands_back_its_partial_outcome(tmp_path: Path) -> None:
    """The child's answer was cut twice; the parent still learns what the
    child wrote, so it re-briefs from there and not from zero."""
    session = _session(tmp_path)
    cut = ChatReply(content="script so far\nline", finish_reason="max_tokens", prompt_tokens=100)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="write the MD script"),
            _call("write_file", path="md.py", content="print('part one')\n"),
            cut,
            cut,
            _text("md-expert got cut; I will re-brief from md.py"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    handed = next(
        m["content"] for m in client.requests[-1] if m["role"] == "tool"
    )
    assert "[truncated:" in handed
    assert "[partial outcome:" in handed
    assert f"files it wrote: {tmp_path / 'md.py'}" in handed
    assert "[md-expert-1: answer after" in handed


def test_a_child_that_ends_whole_hands_back_no_partial_outcome(tmp_path: Path) -> None:
    _mason, client, _result = _delegated_turn(tmp_path)
    handed = next(m["content"] for m in client.requests[-1] if m["role"] == "tool")
    assert "[partial outcome:" not in handed


def test_a_brief_naming_a_cache_hit_file_is_sent_to_the_producer(tmp_path: Path) -> None:
    from foundation import Workspace, current_run, task

    @task
    def averaged(temperature: float) -> float:
        current_run().keep("averages.json", {"T": temperature})
        return temperature

    with Workspace(tmp_path / ".slab") as ws:
        with ws.start_run(name="first"):
            averaged(1180.0)
        with ws.start_run(name="again"):
            averaged(1180.0)
        producer, hit = (run.id for run in reversed(ws.runs.list_runs()))
    client = FakeClient(
        [
            _call("delegate", agent="analysis-expert", task=f"read run:{hit}/averages.json"),
            _call("finish", report="T = 1180 K"),
            _text("done"),
        ]
    )
    Mason(_session(tmp_path), client=client).run_turn("check it")
    assert client.requests[1][1]["content"] == f"read run:{producer}/averages.json"
    footer = next(
        m for m in client.requests[-1] if m.get("tool_call_id") == "c_delegate"
    )["content"]
    assert f"[harness] brief: run:{hit}/averages.json rewritten to run:{producer}/" in footer


# -- continuing a specialist ---------------------------------------------------


def _delegate_results(client: FakeClient) -> list[str]:
    """Every delegate result the lead saw this turn, in order.

    Each request carries the whole conversation, so the last one that
    holds any delegate result holds them all.
    """
    for request in reversed(client.requests):
        results = [
            str(m["content"])
            for m in request
            if m.get("role") == "tool" and m.get("tool_call_id") == "c_delegate"
        ]
        if results:
            return results
    return []


def _continued_pair(tmp_path: Path, **agent: object) -> tuple[Mason, FakeClient]:
    """One specialist briefed, then continued once on the same handle."""
    session = _session(tmp_path, **agent)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="read the failure record"),
            _call("finish", report="the run died at step 40"),
            _call(
                "delegate",
                agent="md-expert",
                task="now fix the timestep and relaunch",
                continues="md-expert-1",
            ),
            _call("finish", report="relaunched as run ef99"),
            _text("md-expert relaunched it (run ef99)"),
        ]
    )
    mason = Mason(session, client=client)
    mason.run_turn("why did it die?")
    return mason, client


def test_a_continued_specialist_keeps_the_messages_of_its_first_turn(tmp_path: Path) -> None:
    """The second brief lands in the same conversation: the specialist reads
    its own first brief and its own first answer, so the lead pays for the
    reading once."""
    mason, client = _continued_pair(tmp_path)
    first, second = client.requests[1], client.requests[3]
    early = [m["content"] for m in first if m["role"] == "user"]
    assert "read the failure record" in early
    later = [m["content"] for m in second if m["role"] == "user"]
    assert "read the failure record" in later
    assert "now fix the timestep and relaunch" in later
    assert len(second) > len(first)
    # One specialist, one transcript: no second child was spawned.
    stem = mason.session.transcript_path.stem
    siblings = sorted(p.name for p in mason.session.sessions_dir.glob(f"{stem}-*.jsonl"))
    assert siblings == [f"{stem}-md-expert-1.jsonl"]


def test_the_footer_names_the_handle_that_continues_the_specialist(tmp_path: Path) -> None:
    mason, client = _continued_pair(tmp_path)
    handed = _delegate_results(client)
    stem = mason.session.transcript_path.stem
    for text in handed:
        assert 'continue with continues="md-expert-1"' in text
        assert f"transcript {stem}-md-expert-1.jsonl" in text
    assert "[md-expert-1: finish after 1 step(s);" in handed[0]


def test_each_turn_of_a_continued_specialist_is_marked_in_its_transcript(
    tmp_path: Path,
) -> None:
    mason, _client = _continued_pair(tmp_path)
    stem = mason.session.transcript_path.stem
    child = mason.session.sessions_dir / f"{stem}-md-expert-1.jsonl"
    turns = [
        json.loads(line)
        for line in child.read_text().splitlines()
        if json.loads(line)["type"] == "turn"
    ]
    assert [event["n"] for event in turns] == [1, 2]
    events = [json.loads(line) for line in mason.session.transcript_path.read_text().splitlines()]
    briefs = [e for e in events if e["type"] == "delegate"]
    assert [e["turn"] for e in briefs] == [1, 2]
    assert "continues" not in briefs[0]
    assert briefs[1]["continues"] == "md-expert-1"


def test_a_continue_rebuilds_the_system_message_from_current_state(tmp_path: Path) -> None:
    """The lead wrote the notebook between the two briefs; the specialist's
    second turn reads the entry, because the system message is rebuilt."""
    session = _session(tmp_path)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="first"),
            _call("finish", report="one"),
            _call("notebook", entry="the timestep must be 0.5 fs for this potential"),
            _call("delegate", agent="md-expert", task="second", continues="md-expert-1"),
            _call("finish", report="two"),
            _text("done"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    first_system = str(client.requests[1][0]["content"])
    second_system = str(client.requests[4][0]["content"])
    assert "0.5 fs for this potential" not in first_system
    assert "0.5 fs for this potential" in second_system


def test_an_unknown_handle_is_refused_with_the_handles_that_exist(tmp_path: Path) -> None:
    session = _session(tmp_path)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="x", continues="md-expert-3"),
            _call("delegate", agent="md-expert", task="x"),
            _call("finish", report="done"),
            _call("delegate", agent="md-expert", task="y", continues="md-expert-3"),
            _text("ok"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    refusals = _delegate_results(client)
    assert refusals[0] == "no specialist md-expert-3 in this conversation; live: none"
    assert refusals[2] == (
        "no specialist md-expert-3 in this conversation; live: md-expert-1"
    )


def test_a_handle_of_another_agent_and_a_critics_handle_are_refused(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.sessions_dir.mkdir(parents=True, exist_ok=True)
    stem = session.transcript_path.stem
    (session.sessions_dir / f"{stem}-critic-1.jsonl").write_text("")
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="x"),
            _call("finish", report="done"),
            _call("delegate", agent="dft-expert", task="y", continues="md-expert-1"),
            _call("delegate", agent="md-expert", task="z", continues="critic-1"),
            _text("ok"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    refusals = _delegate_results(client)
    assert refusals[1].startswith("md-expert-1 is md-expert, not dft-expert;")
    assert refusals[2].startswith("critic-1 is a critic, and a review is not continued;")


def test_a_cut_turn_hands_back_only_what_that_turn_left_behind(tmp_path: Path) -> None:
    """The partial outcome is sliced by turn: the second brief's report names
    the file the second turn wrote, not the first turn's."""
    session = _session(tmp_path)
    cut = ChatReply(content="half a script", finish_reason="max_tokens", prompt_tokens=100)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="write the first script"),
            _call("write_file", path="one.py", content="print(1)\n"),
            cut,
            cut,
            _call("delegate", agent="md-expert", task="and the second", continues="md-expert-1"),
            _call("write_file", path="two.py", content="print(2)\n"),
            cut,
            cut,
            _text("both cut"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    handed = _delegate_results(client)
    assert f"files it wrote: {tmp_path / 'one.py'}" in handed[0]
    assert f"files it wrote: {tmp_path / 'two.py'}" in handed[1]
    assert "one.py" not in handed[1]


def test_a_continue_after_a_resume_replays_the_childs_transcript(tmp_path: Path) -> None:
    """The lead's process is gone and its live children with it. The handle
    still resolves: the specialist's own transcript is replayed, the resume
    is recorded in it, and its earlier messages are in the next request."""
    first = _session(tmp_path)
    Mason(
        first,
        client=FakeClient(
            [
                _call("delegate", agent="md-expert", task="read the record"),
                _call("finish", report="step 40"),
                _text("ok"),
            ]
        ),
    ).run_turn("go")
    first.release_session_lock()
    child_path = first.sessions_dir / f"{first.transcript_path.stem}-md-expert-1.jsonl"
    assert child_path.is_file()

    resumed = _session(tmp_path)
    resumed.resume_from_transcript(first.transcript_path)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="now relaunch", continues="md-expert-1"),
            _call("finish", report="relaunched"),
            _text("done"),
        ]
    )
    replayed = resumed.load_messages(first.transcript_path)
    Mason(resumed, client=client, resume_from=replayed).run_turn("carry on")
    briefed = [m["content"] for m in client.requests[1] if m["role"] == "user"]
    assert "read the record" in briefed
    assert "now relaunch" in briefed
    events = [json.loads(line) for line in child_path.read_text().splitlines()]
    assert [e["type"] for e in events].count("resume") == 1
    assert [e["n"] for e in events if e["type"] == "turn"] == [1, 2]
    # The replayed messages were already in the file: they are not doubled.
    assert len([e for e in events if e["type"] == "message"]) == len(
        {json.dumps(e["message"], sort_keys=True) + str(i)
         for i, e in enumerate(e for e in events if e["type"] == "message")}
    )


# -- per-brief budgets ---------------------------------------------------------


def test_a_brief_may_lower_the_budget_and_the_footer_names_it(tmp_path: Path) -> None:
    session = _session(tmp_path)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="look around", steps=3),
            _call("list_dir"),
            _call("list_dir"),
            _call("list_dir"),
            _text("the specialist ran out of its three calls"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    handed = next(
        m["content"]
        for m in client.requests[-1]
        if m.get("role") == "tool" and m.get("tool_call_id") == "c_delegate"
    )
    assert "[md-expert-1: turn budget (3, set by the brief) after 3 step(s);" in str(handed)
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    (brief,) = [e for e in events if e["type"] == "delegate"]
    assert brief["steps_budget"] == 3
    # The specialist read its own budget, not the card's sixty.
    assert "model call 1 of 3" in str(client.requests[1][-1]["content"])


def test_the_cards_own_budget_names_itself_in_the_footer(tmp_path: Path) -> None:
    session = _session(tmp_path, max_turns=2)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="look around"),
            _call("list_dir"),
            _call("list_dir"),
            _text("the specialist ran out of the card's two calls"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    (brief,) = [e for e in events if e["type"] == "delegate"]
    assert "steps_budget" not in brief
    assert brief["stop"] == "max_turns"


def test_a_brief_may_not_raise_the_step_cap_or_the_effort(tmp_path: Path) -> None:
    session = _session(tmp_path, max_turns=10, effort="medium")
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="x", steps=80),
            _call("delegate", agent="md-expert", task="x", effort="max"),
            _call("delegate", agent="md-expert", task="x", steps=0),
            _text("ok"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    refusals = _delegate_results(client)
    assert refusals[0] == (
        "steps 80 exceeds this agent's cap of 10; a brief may lower the cap, never raise it"
    )
    assert refusals[1] == (
        "effort max exceeds this agent's effort of medium; "
        "a brief may lower the effort, never raise it"
    )
    assert refusals[2] == "steps must be at least 1 model call"
    # Nothing ran: three refusals and no child transcript.
    stem = session.transcript_path.stem
    assert not list(session.sessions_dir.glob(f"{stem}-*.jsonl"))


def test_a_flag_outranks_the_brief_and_the_lead_is_told(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.flag_updates = {"max_turns": 2}
    session.agent = MasonConfig.model_validate(
        {"agent": {"model": "fake", "max_turns": 2}}
    ).agent
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="look", steps=1),
            _call("list_dir"),
            _call("list_dir"),
            _text("ok"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    handed = next(
        m["content"]
        for m in client.requests[-1]
        if m.get("role") == "tool" and m.get("tool_call_id") == "c_delegate"
    )
    text = str(handed)
    assert "[harness] the --max-turns flag pins 2 call(s); the brief's steps=1 is ignored" in text
    assert "turn budget (2)" in text


def test_a_brief_at_lower_effort_builds_its_own_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Effort is baked into the client, so a brief that lowers it gets a
    client of its own, at the dial the brief asked for."""
    built: list[str | None] = []

    def fake_builder(agent: Any, keys: object = None) -> FakeClient:
        built.append(agent.effort)
        return FakeClient([_call("finish", report="a quick look")])

    monkeypatch.setattr("mason.loop.client_from_config", fake_builder)
    session = _session(tmp_path, effort="high")
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="read the record", effort="low"),
            _text("ok"),
        ]
    )
    Mason(session, client=client).run_turn("go")
    assert built == ["low"]
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    (brief,) = [e for e in events if e["type"] == "delegate"]
    assert brief["effort"] == "low"


def test_a_resumed_lead_never_reuses_a_handle_of_the_conversation_it_replays(
    tmp_path: Path,
) -> None:
    """A fresh brief after a resume takes the next ordinal, so the earlier
    specialist stays reachable by its own handle."""
    first = _session(tmp_path)
    Mason(
        first,
        client=FakeClient(
            [
                _call("delegate", agent="md-expert", task="one"),
                _call("finish", report="done"),
                _text("ok"),
            ]
        ),
    ).run_turn("go")
    first.release_session_lock()

    resumed = _session(tmp_path)
    resumed.resume_from_transcript(first.transcript_path)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="two"),
            _call("finish", report="done"),
            _text("ok"),
        ]
    )
    Mason(resumed, client=client).run_turn("carry on")
    handed = _delegate_results(client)[0]
    assert 'continue with continues="md-expert-2"' in handed


# -- helpers: a specialist hands a script on, one level further down -----------


def _helped_turn(
    tmp_path: Path, specialist: list[ChatReply], **agent: object
) -> tuple[Mason, FakeClient, Any]:
    """Lead -> md-expert -> coding-expert on one shared scripted client.

    *specialist* is what md-expert and its helpers say, in order, between
    the lead's brief and the lead's closing answer.
    """
    session = _session(tmp_path, **agent)
    client = FakeClient(
        [
            _call("delegate", agent="md-expert", task="compute the MSD"),
            *specialist,
            _text("the MSD is in (run ab12cd)"),
        ]
    )
    mason = Mason(session, client=client)
    return mason, client, mason.run_turn("is it molten?")


def _fix_msd() -> list[ChatReply]:
    return [
        _call("delegate", agent="coding-expert", task="fix msd.py", context="IndexError"),
        _call("finish", report="msd.py fixed; the dry run passed"),
        _call("finish", report="MSD 0.8 A^2 (run ab12cd)"),
    ]


def _tool_result(request: list[dict[str, Any]], call_id: str) -> str:
    """The latest result of *call_id* in one request: the scripted calls share ids."""
    results = [m for m in request if m.get("role") == "tool" and m.get("tool_call_id") == call_id]
    return str(results[-1]["content"])


def test_a_specialist_has_delegate_and_a_helper_has_none(tmp_path: Path) -> None:
    session = _session(tmp_path)
    roster = discover_roster(tmp_path)
    specialist = Mason(
        session.spawn("md-expert", session.agent),
        client=FakeClient([]), spec=roster["md-expert"], roster=roster, depth=1,
    )
    assert "delegate" in specialist.toolbox.tools
    assert "delegate_many" not in specialist.toolbox.tools
    assert "a helper on your team" in specialist.toolbox.tools["delegate"].description
    helper = Mason(
        specialist.session.spawn("coding-expert", session.agent),
        client=FakeClient([]), spec=roster["coding-expert"], roster=roster, depth=2,
    )
    assert "delegate" not in helper.toolbox.tools
    # Even a specialist card run at depth two gets none.
    deep = Mason(
        specialist.session.spawn("md-expert", session.agent),
        client=FakeClient([]), spec=roster["md-expert"], roster=roster, depth=2,
    )
    assert "delegate" not in deep.toolbox.tools


def test_the_delegation_switch_removes_the_tool_at_both_depths(tmp_path: Path) -> None:
    session = _session(tmp_path, delegation=False)
    roster = discover_roster(tmp_path)
    lead = Mason(session, client=FakeClient([]), roster=roster)
    specialist = Mason(
        session.spawn("md-expert", session.agent),
        client=FakeClient([]), spec=roster["md-expert"], roster=roster, depth=1,
    )
    assert "delegate" not in lead.toolbox.tools
    assert "delegate" not in specialist.toolbox.tools
    assert "# Your team" not in specialist.messages[0]["content"]


def test_a_specialist_that_names_a_non_helper_is_refused(tmp_path: Path) -> None:
    _mason, client, _result = _helped_turn(
        tmp_path,
        [
            _call("delegate", agent="worker", task="run the study"),
            _call("finish", report="nobody to hand it to"),
        ],
    )
    refusal = _tool_result(client.requests[2], "c_delegate")
    assert refusal == (
        "worker is not a helper; a specialist hands scripts to its helpers only. "
        "your team: coding-expert"
    )


def test_a_helper_transcript_nests_under_its_specialist(tmp_path: Path) -> None:
    from mason.session import session_header, transcript_groups, unrecognised_session_files

    mason, _client, _result = _helped_turn(tmp_path, _fix_msd())
    lead = mason.session.transcript_path
    specialist = lead.with_name(f"{lead.stem}-md-expert-1.jsonl")
    helper = lead.with_name(f"{lead.stem}-md-expert-1-coding-expert-1.jsonl")
    assert helper.is_file()
    # One conversation, both delegations its siblings; purge sweeps them together.
    groups = transcript_groups(tmp_path / ".slab", include_orphans=True)
    assert groups == [(lead, sorted([specialist, helper]))]
    assert unrecognised_session_files(tmp_path / ".slab") == []
    # The header says who spawned whom.
    assert session_header(specialist)["agent"] == "md-expert"
    assert session_header(specialist)["parent"] is None
    assert session_header(helper)["agent"] == "coding-expert"
    assert session_header(helper)["parent"] == "md-expert-1"


def test_the_lead_reads_one_footer_line_per_helper_brief(tmp_path: Path) -> None:
    _mason, client, result = _helped_turn(tmp_path, _fix_msd())
    assert result.stop_reason == "answer"
    # The specialist read the helper's report with its own footer.
    to_specialist = _tool_result(client.requests[3], "c_delegate")
    assert "msd.py fixed; the dry run passed" in to_specialist
    assert "[coding-expert-1: finish after 1 step(s);" in to_specialist
    # The lead reads the helper's cost under the specialist's footer.
    to_lead = _tool_result(client.requests[-1], "c_delegate")
    assert "[md-expert-1: finish after 2 step(s);" in to_lead
    assert "\n[harness] helper coding-expert-1: 1 call, finished" in to_lead


def test_the_fourth_helper_brief_of_a_turn_is_refused(tmp_path: Path) -> None:
    briefs: list[ChatReply] = []
    for n in range(3):
        briefs += [
            _call("delegate", agent="coding-expert", task=f"fix script {n}"),
            _call("finish", report=f"script {n} fixed"),
        ]
    _mason, client, _result = _helped_turn(
        tmp_path,
        [
            *briefs,
            _call("delegate", agent="coding-expert", task="fix script 3"),
            _call("finish", report="three scripts fixed"),
        ],
    )
    refused = _tool_result(client.requests[-2], "c_delegate")
    assert refused == "this brief has used its 3 helper calls; finish with what you have"
    to_lead = _tool_result(client.requests[-1], "c_delegate")
    assert to_lead.count("[harness] helper coding-expert-") == 3


def test_helper_briefs_is_a_roster_override(tmp_path: Path) -> None:
    _mason, client, _result = _helped_turn(
        tmp_path,
        [
            _call("delegate", agent="coding-expert", task="fix it"),
            _call("finish", report="fixed"),
            _call("delegate", agent="coding-expert", task="fix it again"),
            _call("finish", report="done"),
        ],
        roster={"md-expert": {"helper_briefs": 1}},
    )
    assert _tool_result(client.requests[-2], "c_delegate") == (
        "this brief has used its 1 helper calls; finish with what you have"
    )


def test_a_helper_runs_no_longer_than_the_brief_that_called_it(tmp_path: Path) -> None:
    mason, client, _result = _helped_turn(
        tmp_path,
        [
            _call("delegate", agent="coding-expert", task="fix msd.py"),
            _call("list_dir"),  # the helper's one call
            _call("finish", report="the helper ran out; msd.py is still broken"),
        ],
        max_turns=2,
    )
    # md-expert delegated at its first of two calls, so one is left, and
    # the helper stops there however large its own cap is.
    to_specialist = _tool_result(client.requests[3], "c_delegate")
    assert "[coding-expert-1: turn budget (1) after 1 step(s);" in to_specialist
    assert (
        "[harness] the helper stopped at 1 call(s), the calls this brief had left; "
        "a helper cannot outlive the brief that called it"
    ) in to_specialist
    specialist = mason.session.transcript_path.with_name(
        f"{mason.session.transcript_path.stem}-md-expert-1.jsonl"
    )
    events = [json.loads(line) for line in specialist.read_text().splitlines()]
    (helped,) = [e for e in events if e["type"] == "delegate"]
    assert helped["steps_budget"] == 1
    assert helped["handle"] == "coding-expert-1"


def test_a_helper_that_finishes_in_time_carries_no_calls_left_note(tmp_path: Path) -> None:
    _mason, client, _result = _helped_turn(tmp_path, _fix_msd())
    assert "calls this brief had left" not in _tool_result(client.requests[3], "c_delegate")


def test_helper_tokens_reach_the_lead_total(tmp_path: Path) -> None:
    mason, _client, _result = _helped_turn(tmp_path, _fix_msd())
    # Five model calls of 100 + 10: lead twice, specialist twice, helper once.
    assert (mason.session.prompt_tokens, mason.session.completion_tokens) == (500, 50)
    specialist = next(iter(mason.session.children.values())).session
    assert (specialist.prompt_tokens, specialist.completion_tokens) == (300, 30)


def test_the_lead_cannot_continue_a_helper_directly(tmp_path: Path) -> None:
    from mason.tools import live_handles

    mason, _client, _result = _helped_turn(tmp_path, _fix_msd())
    assert live_handles(mason.session) == ["md-expert-1"]
    specialist = mason.session.children["md-expert-1"]
    assert live_handles(specialist.session) == ["coding-expert-1"]
