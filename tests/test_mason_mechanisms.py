"""Mechanism switches and the three harness conditions: a card plus a set.

Every mechanism has a switch the loop, the toolbox, and the prompt consult;
a condition names a card and a set. The tests here flip the switches with
a scripted backend and check that what the model sees follows them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mason.client import ChatReply, ToolCall
from mason.config import MasonConfig
from mason.loop import Mason
from mason.mechanisms import (
    ALL_MECHANISMS,
    CONDITIONS,
    MECHANISMS,
    ConditionError,
    check_mechanisms,
    conditions_table,
    effective,
    enabled,
    entry_card,
    harness_label,
    mechanisms_table,
    resolve,
)
from mason.prompts import CORE_PROMPT, core_prompt
from mason.roster import RosterError, discover_roster, hands, parse_agent_card
from mason.session import MasonSession


class FakeClient:
    """Answers from a script; records every request it saw."""

    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []
        self.options: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatReply:
        self.requests.append([dict(m) for m in messages])
        self.options.append(dict(options))
        return self.replies.pop(0)


def _session(tmp_path: Path, **agent: object) -> MasonSession:
    config = MasonConfig.model_validate({"agent": {"model": "fake", **agent}})
    return MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent, auto_approve=True
    )


def _arm(tmp_path: Path, condition: str, without: tuple[str, ...] = ()) -> MasonSession:
    """A session set up the way ``--condition`` sets one up."""
    chosen, on = resolve(condition, without)
    session = _session(tmp_path, mechanisms=sorted(on))
    session.condition = chosen.name
    session.ablated = without
    return session


def _tool_reply(name: str, **arguments: object) -> ChatReply:
    return ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id=f"call_{name}",
                name=name,
                arguments=dict(arguments),
                arguments_raw=json.dumps(arguments),
            ),
        ),
        prompt_tokens=100,
        completion_tokens=10,
    )


def _text(text: str) -> ChatReply:
    return ChatReply(content=text, prompt_tokens=100, completion_tokens=10)


def _call(tool: str, **arguments: object) -> ToolCall:
    return ToolCall(
        id="c1", name=tool, arguments=dict(arguments), arguments_raw=json.dumps(arguments)
    )


# -- the registry -------------------------------------------------------------


def test_every_mechanism_is_a_named_switch_with_evidence() -> None:
    names = [m.name for m in MECHANISMS]
    assert len(names) == len(set(names)) == 11
    assert all(n == n.lower() and " " not in n for n in names)
    assert all(m.does and m.evidence for m in MECHANISMS)
    assert set(names) == ALL_MECHANISMS
    assert mechanisms_table().count("\n") == len(MECHANISMS) + 1
    assert conditions_table().count("\n") == len(CONDITIONS) + 1
    # The ledger's last column is the measured effect, and no grid has run
    # yet, so every row says so rather than leaving the reader to guess.
    from mason.mechanisms import NOT_MEASURED

    header, _rule, *rows = mechanisms_table().splitlines()
    assert header == "| Switch | What it does | Evidence | Measured effect |"
    assert all(row.endswith(f"| {NOT_MEASURED} |") for row in rows)
    assert all(m.measured == NOT_MEASURED for m in MECHANISMS)


def test_a_condition_is_a_card_plus_a_mechanism_set() -> None:
    assert CONDITIONS["slab"].card == "pi" and CONDITIONS["slab"].mechanisms == ALL_MECHANISMS
    protocol = CONDITIONS["protocol"]
    assert protocol.card == "protocol"
    assert "skills" in protocol.mechanisms
    assert not protocol.mechanisms & {
        "check-gating",
        "failure-records",
        "critic-gate",
        "machine-memory",
        "delegation",
    }
    assert CONDITIONS["bare"].mechanisms == frozenset()
    assert entry_card("protocol") == "protocol"
    assert entry_card("protocol", "planner") == "planner" and entry_card(None) is None
    assert harness_label(None) == "slab"
    assert harness_label("slab", ["skills", "budget-hint"]) == "slab -budget-hint -skills"


def test_resolve_refuses_what_it_cannot_switch() -> None:
    with pytest.raises(ConditionError, match="no condition named 'aicc'"):
        resolve("aicc")
    with pytest.raises(ConditionError, match="no mechanism named 'budget'; the switches"):
        check_mechanisms(["budget"])
    with pytest.raises(ConditionError, match="'bare' does not run skills"):
        resolve("bare", ["skills"])
    _, on = resolve(None, ["budget-hint", "budget-hint"])
    assert on == ALL_MECHANISMS - {"budget-hint"}


def test_the_config_switch_is_validated_against_the_registry() -> None:
    with pytest.raises(ValidationError, match="no mechanism named 'skillz'"):
        MasonConfig.model_validate({"agent": {"mechanisms": ["skillz"]}})
    agent = MasonConfig.model_validate(
        {"agent": {"mechanisms": ["skills", "budget-hint", "skills"]}}
    ).agent
    assert agent.mechanisms == ("budget-hint", "skills")
    assert effective(agent) == ("budget-hint", "skills")
    assert MasonConfig().agent.mechanisms is None


def test_enabled_honors_the_older_flags_too() -> None:
    agent = MasonConfig.model_validate(
        {"agent": {"memory": False, "delegation": False, "clear_tool_results": False}}
    ).agent
    assert not enabled(agent, "machine-memory")
    assert not enabled(agent, "delegation")
    assert not enabled(agent, "context-hygiene")
    assert enabled(agent, "check-gating")
    assert set(effective(MasonConfig().agent)) == ALL_MECHANISMS
    with pytest.raises(ConditionError, match="no mechanism named 'budget'"):
        enabled(agent, "budget")


# -- the cards ----------------------------------------------------------------


def test_condition_cards_own_their_prompt_and_take_no_briefs(tmp_path: Path) -> None:
    roster = discover_roster(tmp_path)
    protocol, bare = roster["protocol"], roster["bare"]
    assert not protocol.core and not bare.core
    assert protocol.tools == frozenset(
        {"read_file", "write_file", "edit_file", "list_dir", "search", "shell", "skill", "finish"}
    )
    assert bare.tools == frozenset({"read_file", "write_file", "shell", "finish"})
    assert protocol.skills_scope == "all"
    assert all(spec.core for name, spec in roster.items() if name not in ("protocol", "bare"))
    # Neither is a hand: no lead can brief a harness arm.
    for lead in ("pi", "planner"):
        assert not {"protocol", "bare"} & set(hands(roster[lead], roster))


def test_a_card_without_the_core_runs_alone(tmp_path: Path) -> None:
    card = tmp_path / "solo.md"
    card.write_text("---\nname: solo\ndescription: d\ncore: false\ndelegates: true\n---\nbody\n")
    with pytest.raises(RosterError, match="'core: false' cannot be combined"):
        parse_agent_card(card, "project")
    card.write_text("---\nname: solo\ndescription: d\ncore: maybe\n---\nbody\n")
    with pytest.raises(RosterError, match="'core' must be true or false"):
        parse_agent_card(card, "project")
    card.write_text("---\nname: solo\ndescription: d\ncore: false\n---\nbody\n")
    assert parse_agent_card(card, "project").core is False


# -- the prompt ---------------------------------------------------------------


def test_the_core_prompt_follows_the_switches() -> None:
    assert core_prompt(ALL_MECHANISMS) == CORE_PROMPT
    without_gate = core_prompt(ALL_MECHANISMS - {"check-gating"})
    assert "launch_workflow" not in without_gate and "Long jobs belong" not in without_gate
    assert "Failures are evidence" in without_gate
    without_records = core_prompt(ALL_MECHANISMS - {"failure-records"})
    assert "Failures are evidence" not in without_records and "launch_workflow" in without_records
    for text in (without_gate, without_records, core_prompt([])):
        assert "Do not fabricate" in text and "# Tool discipline" in text


def test_bare_is_the_card_and_a_minimal_environment(tmp_path: Path) -> None:
    session = _arm(tmp_path, "bare")
    roster = discover_roster(tmp_path)
    mason = Mason(session, client=FakeClient([]), spec=roster["bare"], roster=roster)
    (system,) = mason.messages
    content = system["content"]
    assert content.startswith(roster["bare"].prompt)
    for absent in (
        "# How you work",
        "# Compute budget",
        "# Working bounds",
        "# Lab notebook",
        "# Your team",
        "Not available in this session",
        "# Software notes",
    ):
        assert absent not in content
    assert "project directory:" in content and "date:" in content
    assert set(mason.toolbox.tools) == {"read_file", "write_file", "shell", "finish"}


def test_protocol_keeps_the_skill_catalog_and_the_shell_path(tmp_path: Path) -> None:
    session = _arm(tmp_path, "protocol")
    roster = discover_roster(tmp_path)
    mason = Mason(session, client=FakeClient([]), spec=roster["protocol"], roster=roster)
    content = mason.messages[0]["content"]
    assert "PROVENANCE.md" in content and "python script.py" in content
    assert "equation-of-state" in content  # the catalog, skills: all
    assert "launch_workflow" not in content and "# How you work" not in content
    assert set(mason.toolbox.tools) == {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "search",
        "shell",
        "skill",
        "finish",
    }


def test_slab_is_mason_as_it_is(tmp_path: Path) -> None:
    session = _arm(tmp_path, "slab")
    mason = Mason(session, client=FakeClient([]))
    content = mason.messages[0]["content"]
    assert "# How you work" in content and "launch_workflow" in content
    assert {
        "launch_workflow",
        "show_run",
        "skill",
        "delegate",
        "review",
        "remember",
        "plan",
    } <= set(mason.toolbox.tools)


def test_switches_remove_tools_and_their_paragraphs(tmp_path: Path) -> None:
    session = _arm(tmp_path, "slab", ("check-gating", "skills", "machine-memory", "critic-gate"))
    mason = Mason(session, client=FakeClient([]))
    tools = set(mason.toolbox.tools)
    assert not tools & {
        "launch_workflow",
        "wait_for_run",
        "list_runs",
        "show_run",
        "read_artifact",
        "skill",
        "remember",
        "recall",
        "review",
    }
    assert {"delegate", "shell", "plan", "notebook"} <= tools
    content = mason.messages[0]["content"]
    assert "Verification-gated physics" not in content
    assert "Not available in this session" in content and "launch_workflow" in content


# -- the loop -----------------------------------------------------------------


def test_the_header_names_the_arm_and_its_switches(tmp_path: Path) -> None:
    session = _arm(tmp_path, "slab", ("budget-hint",))
    Mason(session, client=FakeClient([_text("ok")])).run_turn("hi")
    header = json.loads(session.transcript_path.read_text().splitlines()[0])
    assert header["condition"] == "slab" and header["ablated"] == ["budget-hint"]
    assert "budget-hint" not in header["mechanisms"]
    assert set(header["mechanisms"]) == ALL_MECHANISMS - {"budget-hint"}
    # A second workspace: two sessions in one second would share a transcript stamp.
    plain = _session(tmp_path / "plain")
    Mason(plain, client=FakeClient([_text("ok")])).run_turn("hi")
    header = json.loads(plain.transcript_path.read_text().splitlines()[0])
    assert header["condition"] is None and header["ablated"] == []
    assert set(header["mechanisms"]) == ALL_MECHANISMS


def test_the_budget_hint_is_absent_when_switched_off(tmp_path: Path) -> None:
    client = FakeClient([_text("ok")])
    session = _session(tmp_path, mechanisms=sorted(ALL_MECHANISMS - {"budget-hint"}))
    Mason(session, client=client).run_turn("go")
    assert [m["role"] for m in client.requests[0]] == ["system", "user"]
    session.release_session_lock()
    on = FakeClient([_text("ok")])
    Mason(_session(tmp_path), client=on).run_turn("go")
    assert [m["role"] for m in on.requests[0]] == ["system", "user", "user"]


def test_the_looking_hint_has_its_own_switch(tmp_path: Path) -> None:
    from mason.loop import _turn_hint

    assert _turn_hint(3, 10, 15, step_back=False) == "[step 3 of 10]"
    assert _turn_hint(3, 10, 15, budget=False).startswith("[15 consecutive steps")
    assert _turn_hint(3, 10, 15, budget=False, step_back=False) == ""
    # A look-only run under the switch: the sixteenth request carries no hint.
    (tmp_path / "f.txt").write_text("x\n")
    replies = [_tool_reply("read_file", path="f.txt")] * 16 + [_text("done")]
    client = FakeClient(list(replies))
    session = _session(
        tmp_path,
        mechanisms=sorted(ALL_MECHANISMS - {"looking-hint", "identical-result-annotation"}),
    )
    Mason(session, client=client).run_turn("look")
    hints = [m["content"] for m in client.requests[15] if m["role"] == "user"][-1]
    assert hints.startswith("[step 16 of") and "consecutive" not in hints


def test_the_identical_result_note_is_absent_when_switched_off(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("same\n")
    replies = [
        _tool_reply("read_file", path="f.txt"),
        _tool_reply("read_file", path="f.txt"),
        _text("done"),
    ]
    off = FakeClient(list(replies))
    session = _session(
        tmp_path, mechanisms=sorted(ALL_MECHANISMS - {"identical-result-annotation"})
    )
    Mason(session, client=off).run_turn("read it twice")
    second = [m for m in off.requests[2] if m.get("role") == "tool"][1]["content"]
    assert "[note:" not in second
    session.release_session_lock()
    on = FakeClient(list(replies))
    Mason(_session(tmp_path), client=on).run_turn("read it twice")
    second = [m for m in on.requests[2] if m.get("role") == "tool"][1]["content"]
    assert "[note: this is the same call as the previous step" in second


def test_adaptive_effort_off_keeps_the_configured_effort(tmp_path: Path) -> None:
    cut = ChatReply(content="", prompt_tokens=100, completion_tokens=10, finish_reason="max_tokens")
    off = FakeClient([cut, _text("short")])
    session = _session(
        tmp_path, effort="high", mechanisms=sorted(ALL_MECHANISMS - {"adaptive-effort"})
    )
    result = Mason(session, client=off).run_turn("go")
    assert result.text == "short" and "effort" not in off.options[1]
    session.release_session_lock()
    on = FakeClient([cut, _text("short")])
    Mason(_session(tmp_path, effort="high"), client=on).run_turn("go")
    assert on.options[1].get("effort") == "low"


def test_context_hygiene_off_never_clears_a_result(tmp_path: Path) -> None:
    # Forty lines under the read tool's per-line cap: the result stays large.
    big = ("x" * 200 + "\n") * 40
    (tmp_path / "big.txt").write_text(big)
    replies = [_tool_reply("read_file", path="big.txt")] * 4 + [_text("done")]
    off = FakeClient(list(replies))
    session = _session(
        tmp_path,
        clear_tool_results_at=0.01,
        keep_tool_results=1,
        mechanisms=sorted(ALL_MECHANISMS - {"context-hygiene", "identical-result-annotation"}),
    )
    Mason(session, client=off).run_turn("read")
    assert not any("cleared to save context" in str(m.get("content")) for m in off.requests[-1])
    (tmp_path / "on").mkdir()
    (tmp_path / "on" / "big.txt").write_text(big)
    on = FakeClient(list(replies))
    Mason(
        _session(
            tmp_path / "on",
            clear_tool_results_at=0.01,
            keep_tool_results=1,
            mechanisms=sorted(ALL_MECHANISMS - {"identical-result-annotation"}),
        ),
        client=on,
    ).run_turn("read")
    assert any("cleared to save context" in str(m.get("content")) for m in on.requests[-1])


# -- the tools ----------------------------------------------------------------


def test_failure_records_are_withheld_when_switched_off(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("raise RuntimeError('boom')\n")
    on = Mason(_session(tmp_path), client=FakeClient([]))
    result = on.toolbox.dispatch(_call("launch_workflow", script="bad.py", name="bad"))
    assert "status=failed" in result and "failure record:" in result
    run_id = result.split()[1].rstrip(":")
    shown = on.toolbox.dispatch(_call("show_run", run_id=run_id))
    assert '"failure"' in shown
    on.session.release_session_lock()
    off = Mason(
        _session(tmp_path, mechanisms=sorted(ALL_MECHANISMS - {"failure-records"})),
        client=FakeClient([]),
    )
    result = off.toolbox.dispatch(_call("launch_workflow", script="bad.py", name="bad2"))
    assert "status=failed" in result and "failure record:" not in result
    shown = off.toolbox.dispatch(_call("show_run", run_id=result.split()[1].rstrip(":")))
    assert '"failure"' not in shown and '"status": "failed"' in shown


def test_critic_gate_off_ungates_a_review_first_card(tmp_path: Path) -> None:
    roster = discover_roster(tmp_path)
    gated = Mason(_session(tmp_path), client=FakeClient([]), spec=roster["planner"], roster=roster)
    assert "review" in gated.toolbox.tools
    refusal = gated.toolbox.dispatch(_call("delegate", agent="worker", task="t"))
    assert refusal.startswith("refused: this card spends no compute")
    gated.session.release_session_lock()
    session = _session(tmp_path, mechanisms=sorted(ALL_MECHANISMS - {"critic-gate"}))
    free = Mason(session, client=FakeClient([]), spec=roster["planner"], roster=roster)
    assert "review" not in free.toolbox.tools
    assert not free.toolbox.dispatch(_call("delegate", agent="worker", task="t")).startswith(
        "refused: this card spends no compute"
    )


def test_context_hygiene_off_lifts_the_output_cap(tmp_path: Path) -> None:
    """The cap is hygiene's first layer, so the switch removes it too: with
    it off a long result reaches the model whole, and with it on the
    middle is cut with a marker."""
    from mason.tools import build_toolbox

    (tmp_path / "big.txt").write_text(("y" * 80 + "\n") * 100)  # 8,100 characters
    call = ToolCall(
        id="r", name="read_file", arguments={"path": "big.txt", "raw": True}, arguments_raw="{}"
    )
    off = _session(
        tmp_path,
        max_tool_output_chars=1_000,
        mechanisms=sorted(ALL_MECHANISMS - {"context-hygiene"}),
    )
    whole = build_toolbox(off).dispatch(call)
    assert len(whole) > 8_000 and "truncated" not in whole
    on = _session(tmp_path, max_tool_output_chars=1_000)
    cut = build_toolbox(on).dispatch(call)
    assert len(cut) < 1_200 and "characters truncated" in cut
