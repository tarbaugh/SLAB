"""The script-bug handoff: a Python bug in the agent's script goes to the helper.

Three tiers, one mechanism switch. The note names the helper, the second
failure of one script is briefed by the harness itself, and a finish over
an unfixed bug is refused once. Nothing here turns a script failure into a
harness failure, which is what the streak tests at the end prove.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundation._ops import launch_script
from mason.client import ChatReply, ToolCall
from mason.config import MasonConfig
from mason.loop import Mason
from mason.roster import discover_roster
from mason.scriptbugs import ScriptBug, script_bug, shell_script_bug
from mason.session import MasonSession
from mason.tools import Toolbox, build_toolbox, open_script_bugs
from slab.config import HpcConfig

# -- fixtures ----------------------------------------------------------------


class FakeClient:
    """A scripted backend; the parent and every child consume it in order."""

    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatReply:
        self.requests.append([dict(m) for m in messages])
        return self.replies.pop(0)


def _session(tmp_path: Path, **agent: object) -> MasonSession:
    config = MasonConfig.model_validate({"agent": {"model": "fake", **agent}})
    return MasonSession(
        tmp_path,
        workspace_root=tmp_path / ".slab",
        agent=config.agent,
        hpc=HpcConfig(),
        auto_approve=True,
    )


def _call(tool: str, /, **arguments: object) -> ToolCall:
    return ToolCall(
        id="t1", name=tool, arguments=dict(arguments), arguments_raw=json.dumps(arguments)
    )


def _reply(name: str, **arguments: object) -> ChatReply:
    return ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id=f"c_{name}",
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


BUGGY = "rows = [1.0, 2.0]\nprint('mean', sum(rows) / len(row))\n"
CLEAN = "rows = [1.0, 2.0]\nprint('mean', sum(rows) / len(rows))\n"


def _box(
    session: MasonSession,
    agent: str = "dft-expert",
    *,
    depth: int = 1,
    client: Any = None,
) -> Toolbox:
    """A toolbox for one card of the built-in roster, at *depth*."""
    roster = discover_roster(session.cwd)
    return build_toolbox(
        session,
        roster[agent],
        depth=depth,
        roster=roster,
        parent_client=client or FakeClient([]),
    )


def _events(session: MasonSession, type_: str) -> list[dict[str, Any]]:
    return session.recorded(type_)


# -- B: the classification on real result shapes ------------------------------


def _launch(tmp_path: Path, name: str, text: str, **kw: Any) -> dict[str, Any]:
    """Really run *text* as a workflow script and return the launch result."""
    workspace = tmp_path / ".store"
    script = tmp_path / f"{name}.py"
    script.write_text(text)
    return launch_script(workspace, script, capture_output=True, session="s1", **kw)


@pytest.mark.parametrize(
    ("name", "text", "exception"),
    [
        ("nameerror", BUGGY, "NameError"),
        ("syntaxerror", "def f(:\n    pass\n", "SyntaxError"),
        (
            "typeerror",
            "from foundation.tasks import fetch_structure\nfetch_structure(nope=1)\n",
            "TypeError",
        ),
        ("valueerror", "raise ValueError('a guard refused the arguments')\n", "ValueError"),
    ],
)
def test_a_python_failure_of_a_real_run_is_a_script_bug(
    tmp_path: Path, name: str, text: str, exception: str
) -> None:
    result = _launch(tmp_path, name, text)
    bug = script_bug(result, tmp_path / f"{name}.py")
    assert bug is not None
    assert bug.exception == exception
    assert bug.line is not None
    assert bug.traceback.startswith("Traceback")


def test_a_run_that_completed_is_no_script_bug(tmp_path: Path) -> None:
    assert script_bug(_launch(tmp_path, "fine", CLEAN), tmp_path / "fine.py") is None


def test_a_failed_check_is_no_script_bug(tmp_path: Path) -> None:
    result = _launch(
        tmp_path,
        "checked",
        "from foundation.runtime import check\n"
        "@check\n"
        "def always():\n"
        "    return False, 'the gate did not pass'\n"
        "print('ran')\n",
    )
    assert result["checks_passed"] == 0 and result["checks_total"] == 1
    assert result["status"] == "completed"
    assert script_bug(result, tmp_path / "checked.py") is None


def test_a_dry_run_that_stopped_is_a_script_bug(tmp_path: Path) -> None:
    result = _launch(tmp_path, "rehearsed", BUGGY, dry_run=True)
    assert result["reached_end"] is False
    bug = script_bug(result, tmp_path / "rehearsed.py")
    assert bug is not None and bug.exception == "NameError" and bug.line == 2


def test_a_dry_run_that_reached_its_end_is_no_script_bug(tmp_path: Path) -> None:
    result = _launch(tmp_path, "rehearsed", CLEAN, dry_run=True)
    assert result["reached_end"] is True
    assert script_bug(result, tmp_path / "rehearsed.py") is None


def test_an_engine_failure_is_no_script_bug() -> None:
    """A LAMMPS that ran and failed is a science question, not a code one."""
    engine = {
        "status": "failed",
        "failure": {
            "type": "LammpsScriptError",
            "message": "LAMMPS failed (exit 1):\n  ERROR: Unrecognized pair style",
            "traceback": (
                'Traceback (most recent call last):\n  File "/w/md.py", line 9, in <module>\n'
                "slab.errors.LammpsScriptError: LAMMPS failed (exit 1)"
            ),
        },
    }
    assert script_bug(engine, "/w/md.py") is None


@pytest.mark.parametrize("kind", ["ResourcesError", "TimeoutError", "KeyboardInterrupt"])
def test_a_refusal_a_timeout_and_an_interrupt_are_no_script_bugs(kind: str) -> None:
    result = {"status": "failed", "failure": {"type": kind, "message": "", "traceback": "x"}}
    assert script_bug(result, "/w/md.py") is None


def test_a_cancelled_run_carries_no_traceback_and_is_no_script_bug() -> None:
    cancelled = {"status": "failed", "error": "marked failed by the operator: job cancelled"}
    assert script_bug(cancelled, "/w/md.py") is None


def test_a_shell_python_traceback_is_a_script_bug() -> None:
    out = (
        "exit 1\nTraceback (most recent call last):\n"
        '  File "analysis.py", line 7, in <module>\n'
        "KeyError: 'temp'"
    )
    bug = shell_script_bug(out, "python analysis.py 300")
    assert bug is not None and bug.exception == "KeyError" and bug.line == 7


def test_a_shell_command_that_is_not_python_is_no_script_bug() -> None:
    assert shell_script_bug("exit 1\nERROR: no such file", "lmp -in md.in") is None


# -- tier 1: the note ---------------------------------------------------------


def _first_launch(tmp_path: Path, agent: str = "dft-expert", **session_kw: object) -> str:
    (tmp_path / "gap.py").write_text(BUGGY)
    session = _session(tmp_path, **session_kw)
    box = _box(session, agent, depth=0 if agent == "pi" else 1)
    return box.dispatch(_call("launch_workflow", script="gap.py"))


def test_the_note_names_the_helper_the_bug_and_the_line(tmp_path: Path) -> None:
    answer = _first_launch(tmp_path)
    assert "[script bug: NameError: name 'row' is not defined, in gap.py line 2." in answer
    assert "Brief coding-expert with the script path" in answer
    # The failure the agent asked for still comes first.
    assert answer.index("failure record:") < answer.index("[script bug:")


def test_the_lead_gets_the_note_too(tmp_path: Path) -> None:
    assert "[script bug:" in _first_launch(tmp_path, "pi")


def test_a_helper_gets_no_note(tmp_path: Path) -> None:
    """The coding helper is the one the note would send the work to."""
    assert "[script bug:" not in _first_launch(tmp_path, "coding-expert")


def test_the_bare_and_protocol_conditions_get_no_note(tmp_path: Path) -> None:
    from mason.mechanisms import CONDITIONS

    for name in ("bare", "protocol"):
        answer = _first_launch(
            tmp_path, "dft-expert", mechanisms=sorted(CONDITIONS[name].mechanisms)
        )
        assert "[script bug:" not in answer


def test_the_switch_off_leaves_the_result_alone(tmp_path: Path) -> None:
    from mason.mechanisms import ALL_MECHANISMS

    without = sorted(ALL_MECHANISMS - {"script-bug-handoff"})
    assert "[script bug:" not in _first_launch(tmp_path, "dft-expert", mechanisms=without)


def test_an_engine_failure_and_a_failed_check_get_no_note(tmp_path: Path) -> None:
    (tmp_path / "checked.py").write_text(
        "from foundation.runtime import check\n"
        "@check\n"
        "def always():\n"
        "    return False, 'the gate did not pass'\n"
    )
    session = _session(tmp_path)
    box = _box(session)
    assert "[script bug:" not in box.dispatch(_call("launch_workflow", script="checked.py"))


def test_the_note_is_absent_when_the_helper_cap_is_used_up(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    session = _session(tmp_path, helper_briefs=1)
    session.record({"type": "turn", "n": 1})
    session.record({"type": "delegate", "agent": "coding-expert", "task": "something else"})
    box = _box(session)
    first = box.dispatch(_call("launch_workflow", script="gap.py"))
    # The first failure still asks, because asking is free.
    assert "Brief coding-expert" in first
    second = box.dispatch(_call("launch_workflow", script="gap.py"))
    assert "the 1 helper call(s) of this brief are used up" in second
    assert "[coding-expert-" not in second


# -- tier 2: the automatic handoff -------------------------------------------


def _two_launches(tmp_path: Path, agent: str = "dft-expert", **session_kw: object):
    (tmp_path / "gap.py").write_text(BUGGY)
    session = _session(tmp_path, **session_kw)
    client = FakeClient([_text("I fixed line 2: len(row) -> len(rows)")])
    box = _box(session, agent, depth=0 if agent == "pi" else 1, client=client)
    first = box.dispatch(_call("launch_workflow", script="gap.py"))
    second = box.dispatch(_call("launch_workflow", script="gap.py"))
    return session, client, first, second


def test_the_second_failure_is_handed_over_with_the_hand_back(tmp_path: Path) -> None:
    _, client, _first, second = _two_launches(tmp_path)
    assert "so the harness briefed coding-expert" in second
    assert "I fixed line 2" in second
    assert "[coding-expert-1: answer after 1 step(s);" in second
    assert "read what coding-expert changed in gap.py, then launch it again" in second
    # Exactly one child turn ran.
    assert client.replies == []


def test_the_composed_brief_carries_the_script_the_launch_and_both_tracebacks(
    tmp_path: Path,
) -> None:
    _, client, _first, _second = _two_launches(tmp_path)
    # The last message of the child's request is the ephemeral turn hint.
    brief = str(client.requests[-1][-2]["content"])
    assert "Script: " in brief and "gap.py" in brief
    assert "launch_workflow script=gap.py" in brief
    assert brief.count("NameError: name 'row' is not defined") >= 2
    assert "Do not launch the production run." in brief
    assert "do not make it" in brief


def test_the_handoff_is_recorded_for_the_report(tmp_path: Path) -> None:
    session, _client, _first, _second = _two_launches(tmp_path)
    handoffs = _events(session, "script_bug_handoff")
    assert len(handoffs) == 1
    assert handoffs[0]["exception"] == "NameError"
    assert handoffs[0]["tier"] == 2
    assert handoffs[0]["agent"] == "coding-expert"
    assert handoffs[0]["transcript"].endswith("coding-expert-1.jsonl")
    # It counts as a brief, like any other helper brief.
    assert [e["agent"] for e in _events(session, "delegate")] == ["coding-expert"]


def test_a_brief_to_the_helper_in_between_suppresses_the_handoff(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    session = _session(tmp_path)
    box = _box(session)
    box.dispatch(_call("launch_workflow", script="gap.py"))
    session.record({"type": "delegate", "agent": "coding-expert", "task": "fix gap.py please"})
    second = box.dispatch(_call("launch_workflow", script="gap.py"))
    assert "so the harness briefed" not in second
    assert "Brief coding-expert with the script path" in second


def test_a_different_script_does_not_trigger_the_handoff(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    (tmp_path / "other.py").write_text(BUGGY)
    session = _session(tmp_path)
    box = _box(session)
    box.dispatch(_call("launch_workflow", script="gap.py"))
    second = box.dispatch(_call("launch_workflow", script="other.py"))
    assert "so the harness briefed" not in second
    assert "in other.py line 2" in second


def test_a_clean_run_in_between_clears_the_bug(tmp_path: Path) -> None:
    script = tmp_path / "gap.py"
    script.write_text(BUGGY)
    session = _session(tmp_path)
    box = _box(session)
    box.dispatch(_call("launch_workflow", script="gap.py"))
    script.write_text(CLEAN)
    box.dispatch(_call("launch_workflow", script="gap.py"))
    assert open_script_bugs(session, ["coding-expert"]) == []
    script.write_text(BUGGY)
    third = box.dispatch(_call("launch_workflow", script="gap.py"))
    assert "so the harness briefed" not in third


def test_a_helper_that_crashes_never_raises_into_the_parent(tmp_path: Path) -> None:
    """hold_errors: the child's fault comes back as its report."""

    class Crashing(FakeClient):
        def chat(self, messages: Any, tools: Any = None, **options: Any) -> ChatReply:
            raise RuntimeError("the helper's client broke")

    (tmp_path / "gap.py").write_text(BUGGY)
    session = _session(tmp_path)
    box = _box(session, client=Crashing([]))
    box.dispatch(_call("launch_workflow", script="gap.py"))
    second = box.dispatch(_call("launch_workflow", script="gap.py"))
    assert "the helper's client broke" in second
    assert not second.startswith("tool launch_workflow failed:")


def test_a_shell_python_bug_hands_over_the_same_way(tmp_path: Path) -> None:
    (tmp_path / "an.py").write_text("raise KeyError('temp')\n")
    session = _session(tmp_path)
    client = FakeClient([_text("fixed the key")])
    box = _box(session, client=client)
    import sys

    command = f"{sys.executable} an.py"
    first = box.dispatch(_call("shell", command=command))
    assert "[script bug: KeyError" in first
    second = box.dispatch(_call("shell", command=command))
    assert "so the harness briefed coding-expert" in second


# -- tier 3: the finish gate --------------------------------------------------


def _mason(tmp_path: Path, replies: list[ChatReply], agent: str = "pi", depth: int = 0):
    session = _session(tmp_path)
    roster = discover_roster(tmp_path)
    client = FakeClient(replies)
    return Mason(session, client=client, spec=roster[agent], roster=roster, depth=depth), session


def test_the_finish_is_refused_once_and_stands_when_repeated(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    mason, _ = _mason(
        tmp_path,
        [
            _reply("launch_workflow", script="gap.py"),
            _reply("finish", report="the gap is 1.1 eV"),
            _reply("finish", report="the gap is 1.1 eV"),
        ],
    )
    result = mason.run_turn("measure the gap")
    assert result.stop_reason == "finish"
    refusals = [
        m
        for m in mason.messages
        if isinstance(m.get("content"), str) and "finish refused:" in m["content"]
    ]
    assert len(refusals) == 1
    assert "gap.py line 2" in refusals[0]["content"]
    assert "Brief coding-expert" in refusals[0]["content"]


def test_the_finish_is_not_refused_after_a_clean_relaunch(tmp_path: Path) -> None:
    script = tmp_path / "gap.py"
    script.write_text(BUGGY)
    mason, _ = _mason(
        tmp_path,
        [
            _reply("launch_workflow", script="gap.py"),
            _reply("shell", command="true"),
            _reply("finish", report="done"),
        ],
    )

    calls = {"n": 0}
    original = mason.client.chat  # type: ignore[union-attr]

    def rewrite(*args: Any, **kwargs: Any) -> ChatReply:
        calls["n"] += 1
        if calls["n"] == 2:
            script.write_text(CLEAN)
            return _reply("launch_workflow", script="gap.py")
        return original(*args, **kwargs)

    mason.client.chat = rewrite  # type: ignore[union-attr,method-assign]
    result = mason.run_turn("measure the gap")
    assert result.stop_reason == "finish"
    assert not [m for m in mason.messages if "finish refused:" in str(m.get("content") or "")]


def test_the_finish_is_not_refused_after_a_brief_to_the_helper(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    mason, _ = _mason(
        tmp_path,
        [
            _reply("launch_workflow", script="gap.py"),
            _reply("delegate", agent="coding-expert", task="fix gap.py"),
            _text("the helper fixed it"),
            _reply("finish", report="done"),
        ],
    )
    result = mason.run_turn("measure the gap")
    assert result.stop_reason == "finish"
    assert not [m for m in mason.messages if "finish refused:" in str(m.get("content") or "")]


def test_a_helper_is_never_gated(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    mason, _ = _mason(
        tmp_path,
        [_reply("launch_workflow", script="gap.py"), _reply("finish", report="it still fails")],
        agent="coding-expert",
        depth=2,
    )
    result = mason.run_turn("fix gap.py")
    assert result.stop_reason == "finish"
    assert not [m for m in mason.messages if "finish refused:" in str(m.get("content") or "")]


def test_an_answer_at_the_root_is_gated_once(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    mason, _ = _mason(
        tmp_path,
        [
            _reply("launch_workflow", script="gap.py"),
            _text("the gap is 1.1 eV"),
            _text("the gap is 1.1 eV; the script was abandoned"),
        ],
    )
    result = mason.run_turn("measure the gap")
    assert result.stop_reason == "answer"
    assert "abandoned" in result.text
    assert len([m for m in mason.messages if "finish refused:" in str(m.get("content") or "")]) == 1


def test_the_gate_is_off_with_the_switch_off(tmp_path: Path) -> None:
    from mason.mechanisms import ALL_MECHANISMS

    (tmp_path / "gap.py").write_text(BUGGY)
    session = _session(tmp_path, mechanisms=sorted(ALL_MECHANISMS - {"script-bug-handoff"}))
    roster = discover_roster(tmp_path)
    client = FakeClient(
        [_reply("launch_workflow", script="gap.py"), _reply("finish", report="done")]
    )
    mason = Mason(session, client=client, spec=roster["pi"], roster=roster)
    assert mason.run_turn("measure the gap").stop_reason == "finish"


# -- A: the audit. a script bug is never a harness failure --------------------


def test_five_script_bugs_in_a_row_do_not_stop_the_turn(tmp_path: Path) -> None:
    """The streak counts harness failures; a traceback from the agent's own
    code is a tool result the model must read, so it never counts."""
    (tmp_path / "gap.py").write_text(BUGGY)
    replies = [_reply("launch_workflow", script="gap.py") for _ in range(5)]
    # The handoff briefs the helper on the second, fourth... failure.
    replies += [_text("still broken"), _text("still broken; abandoned")]
    mason, _ = _mason(tmp_path, [*replies, _text("x"), _text("x")])
    result = mason.run_turn("measure the gap")
    assert result.stop_reason != "error_streak"


def test_five_malformed_calls_still_stop_the_turn(tmp_path: Path) -> None:
    mason, _ = _mason(tmp_path, [_reply("no_such_tool") for _ in range(5)])
    assert mason.run_turn("go").stop_reason == "error_streak"


def test_a_launch_that_fails_is_never_marked_a_harness_failure(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    session = _session(tmp_path)
    roster = discover_roster(tmp_path)
    mason = Mason(session, client=FakeClient([]), spec=roster["pi"], roster=roster)
    _result, ok = mason._dispatch(_call("launch_workflow", script="gap.py"))
    assert ok is True


def test_a_delegated_child_whose_script_fails_returns_a_report(tmp_path: Path) -> None:
    """A script bug inside a specialist reaches the lead as a report."""
    (tmp_path / "gap.py").write_text(BUGGY)
    mason, _ = _mason(
        tmp_path,
        [
            _reply("delegate", agent="worker", task="run gap.py"),
            _reply("launch_workflow", script="gap.py"),
            _text("gap.py fails with a NameError on line 2"),
            _reply("finish", report="the script is broken; abandoned"),
        ],
    )
    result = mason.run_turn("run the script")
    assert result.stop_reason == "finish"
    assert "abandoned" in result.text


def test_a_background_run_that_failed_is_classified_on_the_wait(tmp_path: Path) -> None:
    (tmp_path / "gap.py").write_text(BUGGY)
    session = _session(tmp_path)
    box = _box(session)
    box.dispatch(_call("launch_workflow", script="gap.py", background=True, ntasks=1))
    answer = box.dispatch(_call("wait_for_run", timeout_s=60))
    assert "[script bug: NameError" in answer
    assert "in gap.py line 2" in answer


def test_the_record_carries_every_script_run(tmp_path: Path) -> None:
    script = tmp_path / "gap.py"
    script.write_text(BUGGY)
    session = _session(tmp_path)
    box = _box(session)
    box.dispatch(_call("launch_workflow", script="gap.py"))
    script.write_text(CLEAN)
    box.dispatch(_call("launch_workflow", script="gap.py"))
    runs = _events(session, "script_run")
    assert [bool(r["bug"]) for r in runs] == [True, False]
    assert runs[0]["bug"]["where"] == "gap.py line 2"


def test_the_where_of_a_bug_without_a_frame_is_the_file_name() -> None:
    assert ScriptBug("KeyError", "k", "/w/md.py", None, "").where == "md.py"


def test_an_engine_that_left_its_files_is_no_script_bug_whatever_the_type() -> None:
    # A task keeps the engine's files and says so in a note when the engine
    # ran and failed. The exception type may be a plain RuntimeError.
    failure = {
        "type": "RuntimeError",
        "message": "pw.x exited 3",
        "traceback": "Traceback (most recent call last):\n  File \"s.py\", line 4\nRuntimeError: x",
        "notes": ["engine files kept as artifacts: 'si-failed.pwo'"],
    }
    assert script_bug({"status": "failed", "failure": failure}, "s.py") is None
    failure["notes"] = ["relax failed after 3 steps"]
    assert script_bug({"status": "failed", "failure": failure}, "s.py") is not None


def test_every_engine_and_store_error_type_is_on_the_deny_list() -> None:
    # A new error type in slab.errors must be sorted here: an engine,
    # builder, scheduler, or store failure is no script bug.
    import inspect

    import slab.errors
    from foundation.errors import StorageError
    from mason.scriptbugs import NOT_A_SCRIPT_BUG

    declared = {
        name
        for name, value in inspect.getmembers(slab.errors, inspect.isclass)
        if issubclass(value, slab.errors.SlabError) and value is not slab.errors.SlabError
    }
    assert declared <= NOT_A_SCRIPT_BUG, sorted(declared - NOT_A_SCRIPT_BUG)
    assert StorageError.__name__ in NOT_A_SCRIPT_BUG


def test_an_answer_in_a_conversation_is_not_gated(tmp_path: Path) -> None:
    # A person at the keyboard reads the answer and decides what is next.
    (tmp_path / "gap.py").write_text(BUGGY)
    mason, _ = _mason(
        tmp_path,
        [_reply("launch_workflow", script="gap.py"), _text("the script failed; what next?")],
    )
    mason.session.interactive = True
    result = mason.run_turn("measure the gap")
    assert result.stop_reason == "answer" and "what next" in result.text
    assert not [m for m in mason.messages if "finish refused:" in str(m.get("content") or "")]
