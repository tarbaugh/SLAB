"""The critic: a read-only card, the review tool, and the gate before compute."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mason.cli import app
from mason.client import ChatReply, ToolCall
from mason.config import MasonConfig
from mason.errors import MasonError
from mason.loop import Mason
from mason.reviews import digest, load_reviews, plan_is_approved, review_block
from mason.roster import RosterError, critics, discover_roster, hands, parse_agent_card
from mason.session import MasonSession
from mason.tools import READ_ONLY_TOOLS, TOOL_VOCABULARY, build_toolbox
from slab.config import HpcConfig

runner = CliRunner()

PLAN = "# Goal\nLattice constant of fcc Cu.\n\n1. relax_cell under emt; check fmax < 0.05\n"


class FakeClient:
    """One shared script: the lead and every child consume it in order."""

    def __init__(self, replies: list[ChatReply | Exception]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []
        self.tools: list[list[str]] = []

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatReply:
        self.requests.append([dict(m) for m in messages])
        self.tools.append([t["function"]["name"] for t in tools or []])
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


def _tool_results(client: FakeClient, call_id: str) -> list[str]:
    """The results of every call named *call_id*, from the last request's history."""
    return [
        str(m["content"])
        for m in client.requests[-1]
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    ]


def _write_card(root: Path, name: str, frontmatter: str, body: str = "A role.\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Does {name} things.\n{frontmatter}---\n{body}",
        encoding="utf-8",
    )
    return path


# -- the card -----------------------------------------------------------------


def test_the_builtin_roster_holds_a_critic_and_a_gated_planner(tmp_path: Path) -> None:
    roster = discover_roster(tmp_path)
    critic = roster["critic"]
    assert critic.reviews and not critic.delegates and not critic.review_first
    assert critic.skills_scope == "all"
    assert list(critics(roster)) == ["critic"]
    assert roster["planner"].review_first and not roster["pi"].review_first
    # A critic takes reviews, not briefs: it is on nobody's team.
    assert "critic" not in hands(roster["pi"], roster)
    assert "critic" not in hands(roster["planner"], roster)


def test_the_critic_is_read_only_by_construction(tmp_path: Path) -> None:
    """Whatever the session offers, a card that reviews keeps only the readers."""
    hpc = HpcConfig.model_validate({"default_partition": "cpu", "partitions": {"cpu": {}}})
    config = MasonConfig.model_validate({"agent": {"model": "fake"}})
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent, hpc=hpc,
        auto_approve=True,
    )
    roster = discover_roster(tmp_path)
    box = build_toolbox(session, roster["critic"], roster=roster)
    assert set(box.tools) <= READ_ONLY_TOOLS
    assert {"read_file", "show_run", "list_engines", "skill", "finish"} <= set(box.tools)
    assert not set(box.tools) & {
        "write_file", "edit_file", "shell", "launch_workflow", "submit_job",
        "notebook", "plan", "remember", "delegate", "review",
    }


def test_the_read_only_tools_are_the_ones_that_never_ask(tmp_path: Path) -> None:
    """READ_ONLY_TOOLS is vocabulary, and none of it passes the approval gate."""
    from conftest import build_mp_snapshot

    assert READ_ONLY_TOOLS < TOOL_VOCABULARY
    snapshot = build_mp_snapshot(tmp_path / "mp-snapshot")
    (tmp_path / "slab.toml").write_text(f'[builders.mp]\nroot = "{snapshot}"\n')
    hpc = HpcConfig.model_validate({"default_partition": "cpu", "partitions": {"cpu": {}}})
    session = MasonSession(tmp_path, workspace_root=tmp_path / ".slab", hpc=hpc)
    roster = discover_roster(tmp_path)
    box = build_toolbox(session, roster["pi"], roster=roster)
    assert set(box.tools) > READ_ONLY_TOOLS
    for name in READ_ONLY_TOOLS:
        tool = box.tools[name]
        assert not tool.requires_approval and tool.gate is None, name
    # And every writer is outside it.
    for name in ("write_file", "edit_file", "shell", "launch_workflow", "submit_job",
                 "cancel_job", "remember", "notebook", "plan", "delegate", "review"):
        assert name not in READ_ONLY_TOOLS


def test_a_reviewing_card_that_leads_or_launches_is_refused(tmp_path: Path) -> None:
    path = _write_card(tmp_path / "agents", "judge-lead", "reviews: true\ndelegates: true\n")
    with pytest.raises(RosterError, match="leads nothing"):
        parse_agent_card(path, "project")
    path = _write_card(tmp_path / "agents", "judge-gate", "reviews: true\nreview_first: true\n")
    with pytest.raises(RosterError, match="nothing to gate"):
        parse_agent_card(path, "project")
    path = _write_card(
        tmp_path / "agents", "judge-shell", "reviews: true\ntools: read_file shell finish\n"
    )
    with pytest.raises(RosterError, match="'shell'") as excinfo:
        parse_agent_card(path, "project")
    assert "read-only" in str(excinfo.value) and "list_runs" in str(excinfo.value)
    path = _write_card(
        tmp_path / "agents", "gate-mute", "review_first: true\ntools: read_file finish\n"
    )
    with pytest.raises(RosterError, match="add 'review' to its 'tools'"):
        parse_agent_card(path, "project")
    path = _write_card(tmp_path / "agents", "judge-typed", "review_first: yes please\n")
    with pytest.raises(RosterError, match="'review_first' must be true or false"):
        parse_agent_card(path, "project")


def test_a_reviewing_card_may_narrow_its_readers(tmp_path: Path) -> None:
    path = _write_card(
        tmp_path / "agents", "judge-narrow", "reviews: true\ntools: read_file finish\n"
    )
    spec = parse_agent_card(path, "project")
    box = build_toolbox(_session(tmp_path), spec)
    assert set(box.tools) == {"read_file", "finish"}


# -- the team and the tool ----------------------------------------------------


class _Idle:
    def chat(self, messages: object, tools: object = None) -> ChatReply:
        raise AssertionError("the model was called during construction")


def test_the_lead_sees_the_critic_apart_from_its_hands(tmp_path: Path) -> None:
    roster = discover_roster(tmp_path)
    mason = Mason(_session(tmp_path), client=_Idle(), spec=roster["planner"], roster=roster)
    assert {"review", "delegate"} <= set(mason.toolbox.tools)
    (system,) = mason.messages
    content = system["content"]
    hands_part, _, critics_part = content.partition("Critics you can hand")
    assert "- worker:" in hands_part and "- critic:" not in hands_part
    assert "- critic:" in critics_part and "review tool" in critics_part
    assert "- worker:" not in critics_part


def test_a_gated_card_with_only_a_critic_still_gets_the_team_block(tmp_path: Path) -> None:
    """A card that reviews first but delegates nothing has a review tool and
    a team block naming the critic alone."""
    _write_card(tmp_path / "agents", "careful", "review_first: true\n")
    roster = discover_roster(tmp_path)
    mason = Mason(_session(tmp_path), client=_Idle(), spec=roster["careful"], roster=roster)
    assert "review" in mason.toolbox.tools and "delegate" not in mason.toolbox.tools
    content = mason.messages[0]["content"]
    assert "# Your team" in content and "- critic:" in content and "- worker:" not in content


def test_the_review_tool_lives_under_the_delegation_switch(tmp_path: Path) -> None:
    mason = Mason(_session(tmp_path, delegation=False), client=_Idle())
    assert "review" not in mason.toolbox.tools
    assert "# Your team" not in mason.messages[0]["content"]


def test_a_gated_card_refuses_to_run_where_no_critic_can(tmp_path: Path) -> None:
    roster = discover_roster(tmp_path)
    without = {name: card for name, card in roster.items() if not card.reviews}
    with pytest.raises(MasonError, match="no card on the roster reviews"):
        Mason(_session(tmp_path), client=_Idle(), spec=roster["planner"], roster=without)
    _write_card(tmp_path / "agents", "careful", "review_first: true\n")
    roster = discover_roster(tmp_path)
    with pytest.raises(MasonError, match=r"\[agent\] delegation is off"):
        Mason(
            _session(tmp_path, delegation=False), client=_Idle(),
            spec=roster["careful"], roster=roster,
        )


def test_delegating_to_the_critic_is_refused_toward_the_review_tool(tmp_path: Path) -> None:
    client = FakeClient([_call("delegate", agent="critic", task="judge this"), _text("ok")])
    mason = Mason(_session(tmp_path), client=client)
    mason.run_turn("go")
    (answer,) = _tool_results(client, "c_delegate")
    assert "critic reviews and takes no briefs" in answer
    assert "review tool" in answer and "worker" in answer
    assert len(client.requests) == 2  # no child loop ran


# -- a review, recorded -------------------------------------------------------


def _reviewed(
    tmp_path: Path, verdict: str | None, *, plan: str = PLAN
) -> tuple[Mason, FakeClient]:
    (tmp_path / "PLAN.md").write_text(plan)
    finish: dict[str, object] = {"report": "1. blocking: step 1 names no k-mesh check."}
    if verdict is not None:
        finish["verdict"] = verdict
    client = FakeClient(
        [
            _call("review", subject="plan"),
            _call("finish", **finish),
            _text("noted"),
        ]
    )
    roster = discover_roster(tmp_path)
    mason = Mason(_session(tmp_path), client=client, spec=roster["planner"], roster=roster)
    mason.run_turn("plan the campaign")
    return mason, client


def test_a_review_round_trip_persists_the_verdict_and_the_findings(tmp_path: Path) -> None:
    mason, client = _reviewed(tmp_path, "revise")
    (answer,) = _tool_results(client, "c_review")
    assert answer.startswith("verdict: revise.")
    assert "1. blocking: step 1 names no k-mesh check." in answer
    assert "[critic: finish after 1 step(s);" in answer
    assert "[review recorded in " in answer
    # The critic's request: its own card, the plan in the brief, readers only.
    child_system = client.requests[1][0]["content"]
    assert "You are the critic of a SLAB research group" in child_system
    assert "Lattice constant of fcc Cu." in client.requests[1][1]["content"]
    assert set(client.tools[1]) <= READ_ONLY_TOOLS
    # The record stands alone.
    (review,) = load_reviews(mason.session)
    assert review.subject == "plan" and review.verdict == "revise"
    assert review.reviewer == "critic" and review.digest == digest(PLAN)
    assert review.findings == "1. blocking: step 1 names no k-mesh check."
    assert review.text.strip() == PLAN.strip()
    assert review.path.parent == tmp_path / ".slab" / "mason" / "reviews"
    assert review.path.name == f"{mason.session.transcript_path.stem}-review-1.md"
    assert review.transcript.endswith("-critic-1.jsonl")
    # And the transcript names it.
    events = [json.loads(line) for line in mason.session.transcript_path.read_text().splitlines()]
    (event,) = [e for e in events if e["type"] == "review"]
    assert event["verdict"] == "revise" and event["record"] == review.path.name
    assert event["agent"] == "critic" and event["subject"] == "plan"
    assert not mason.session.plan_approved


def test_a_review_without_a_verdict_approves_nothing(tmp_path: Path) -> None:
    mason, client = _reviewed(tmp_path, None)
    (answer,) = _tool_results(client, "c_review")
    assert answer.startswith("verdict: none.")
    (review,) = load_reviews(mason.session)
    assert review.verdict == "none"
    assert not mason.session.plan_approved


def test_an_approval_opens_the_gate_and_is_kept_for_that_plan_only(tmp_path: Path) -> None:
    mason, _client = _reviewed(tmp_path, "approve")
    assert mason.session.plan_approved
    assert plan_is_approved(mason.session)
    mason.session.release_session_lock()
    # A new session on the same, unchanged plan starts approved.
    roster = discover_roster(tmp_path)
    again = Mason(_session(tmp_path), client=_Idle(), spec=roster["planner"], roster=roster)
    assert again.session.plan_approved
    assert "# Latest review of the plan" in again.messages[0]["content"]
    assert "verdict: approve" in again.messages[0]["content"]
    again.session.release_session_lock()
    # An edited plan is a different plan.
    (tmp_path / "PLAN.md").write_text(PLAN + "2. relax_cell under qe\n")
    changed = Mason(_session(tmp_path), client=_Idle(), spec=roster["planner"], roster=roster)
    assert not changed.session.plan_approved
    assert "PLAN.md has changed since this review" in changed.messages[0]["content"]


def test_the_review_survives_compaction_where_the_plan_does(tmp_path: Path) -> None:
    mason, _client = _reviewed(tmp_path, "revise")
    block = review_block(mason.session)
    assert block.startswith("# Latest review of the plan")
    assert "verdict: revise; reviewer: critic" in block
    assert "k-mesh" in block
    (tmp_path / "PLAN.md").unlink()
    assert "PLAN.md has been removed since this review" in review_block(mason.session)


def test_executors_are_not_shown_the_review(tmp_path: Path) -> None:
    mason, _client = _reviewed(tmp_path, "revise")
    roster = discover_roster(tmp_path)
    child_session = mason.session.spawn("worker", mason.session.agent)
    worker = Mason(child_session, client=_Idle(), spec=roster["worker"], roster=roster, depth=1)
    assert "# Latest review" not in worker.messages[0]["content"]


# -- the gate -----------------------------------------------------------------


def test_the_planner_spends_no_compute_before_the_critic_approves(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text(PLAN)
    client = FakeClient(
        [
            _call("delegate", agent="worker", task="relax Cu"),  # refused, no child
            _call("review"),
            _call("finish", report="no blocking finding", verdict="approve"),
            _call("delegate", agent="worker", task="relax Cu"),
            _call("finish", report="a = 3.6 Å (run ab12cd)"),
            _text("done"),
        ]
    )
    roster = discover_roster(tmp_path)
    mason = Mason(_session(tmp_path), client=client, spec=roster["planner"], roster=roster)
    result = mason.run_turn("measure a")
    assert result.stop_reason == "answer"
    first, second = _tool_results(client, "c_delegate")
    assert first.startswith("refused: the plan has not been approved by the critic")
    assert "review" in first
    assert "a = 3.6 Å (run ab12cd)" in second and "[worker: finish" in second
    # The worker ran only after the approval: its request is the fifth.
    assert "worker of a SLAB research group" in client.requests[4][0]["content"]


def test_the_gate_names_a_missing_plan(tmp_path: Path) -> None:
    client = FakeClient([_call("delegate", agent="worker", task="x"), _text("ok")])
    roster = discover_roster(tmp_path)
    mason = Mason(_session(tmp_path), client=client, spec=roster["planner"], roster=roster)
    mason.run_turn("go")
    (answer,) = _tool_results(client, "c_delegate")
    assert answer.startswith("refused:") and "PLAN.md is empty" in answer


def test_the_gate_refuses_launches_before_asking_for_approval(tmp_path: Path) -> None:
    """A launch the gate will refuse never reaches the approver."""
    asked: list[str] = []

    def approver(tool: str, preview: str) -> bool:
        asked.append(tool)
        return True

    _write_card(tmp_path / "agents", "careful", "review_first: true\n")
    (tmp_path / "PLAN.md").write_text(PLAN)
    config = MasonConfig.model_validate({"agent": {"model": "fake"}})
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent,
        hpc=HpcConfig(), approver=approver,
    )
    roster = discover_roster(tmp_path)
    box = build_toolbox(session, roster["careful"], roster=roster)
    assert "launch_workflow" in box.tools and "review" in box.tools
    call = ToolCall(
        id="l1", name="launch_workflow", arguments={"script": "wf.py"}, arguments_raw="{}"
    )
    assert box.dispatch(call).startswith("refused: the plan has not been approved")
    assert asked == []
    # Once approved, the same call asks as it always did.
    session.plan_approved = True
    answer = box.dispatch(call)
    assert asked == ["launch_workflow"]
    assert not answer.startswith("refused: the plan")


def test_a_review_first_card_is_not_gated_when_delegated_to(tmp_path: Path) -> None:
    """The gate belongs to the turn owner: at depth one it does not exist."""
    (tmp_path / "PLAN.md").write_text(PLAN)
    roster = discover_roster(tmp_path)
    session = _session(tmp_path)
    child = session.spawn("planner", session.agent)
    box = build_toolbox(child, roster["planner"], roster=roster, depth=1)
    assert "review" not in box.tools and "delegate" not in box.tools


# -- subjects -----------------------------------------------------------------


def test_a_file_can_be_reviewed_and_the_fence_holds(tmp_path: Path) -> None:
    script = tmp_path / "wf.py"
    script.write_text("from foundation.tasks import relax\n")
    outside = tmp_path.parent / "elsewhere.py"
    outside.write_text("x = 1\n")
    client = FakeClient(
        [
            _call("review", subject=str(outside)),
            _call("review", subject="missing.py"),
            _call("review", subject="plan"),
            _call("review", subject=str(script), focus="the check"),
            _call("finish", report="1. advisory: no @check", verdict="approve"),
            _text("ok"),
        ]
    )
    mason = Mason(_session(tmp_path), client=client)  # the pi may review too
    mason.run_turn("go")
    fenced, missing, no_plan, reviewed = _tool_results(client, "c_review")
    assert fenced.startswith("refused:")
    assert missing.startswith("nothing to review:") and "is not a file" in missing
    assert no_plan.startswith("nothing to review: PLAN.md")
    assert reviewed.startswith("verdict: approve.")
    brief = client.requests[-2][1]["content"]
    assert "from foundation.tasks import relax" in brief
    assert "Focus from the lead: the check" in brief
    (review,) = load_reviews(mason.session)
    assert review.subject == str(script) and review.verdict == "approve"
    # A file's approval never opens the plan gate.
    assert not mason.session.plan_approved


def test_an_unknown_critic_name_lists_the_critics(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text(PLAN)
    client = FakeClient([_call("review", agent="judge"), _text("ok")])
    mason = Mason(_session(tmp_path), client=client)
    mason.run_turn("go")
    (answer,) = _tool_results(client, "c_review")
    assert answer == "no critic named 'judge'; the critics on the roster: critic"


# -- the CLI ------------------------------------------------------------------


def test_cli_roster_marks_the_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text('[agent]\nmodel = "m"\n')
    result = runner.invoke(app, ["roster"])
    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines()}
    assert lines["pi"].endswith("[delegates]")
    assert lines["planner"].endswith("[delegates, review first]")
    assert lines["critic"].endswith("[reviews]")
    assert not lines["worker"].rstrip().endswith("]")


def test_cli_read_renders_the_review_event(tmp_path: Path) -> None:
    mason, _client = _reviewed(tmp_path, "revise")
    result = runner.invoke(app, ["read", str(mason.session.transcript_path)])
    assert result.exit_code == 0, result.output
    assert "review by critic of plan: verdict revise" in result.output
