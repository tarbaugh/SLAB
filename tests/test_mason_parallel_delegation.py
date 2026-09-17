"""A wave of independent briefs: the specialists run at the same time."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from mason.client import ChatReply, ToolCall
from mason.config import MasonConfig
from mason.loop import Mason
from mason.session import MasonSession
from slab.config import HpcConfig


class WaveClient:
    """One client for every child of a wave, scripted by what the brief says.

    Two specialists of one wave run in two threads, so a script consumed
    in call order would depend on how the threads interleave. This one
    keys each reply off the brief instead, and records when each child's
    calls started and ended so a test can see that they overlapped.
    """

    def __init__(
        self,
        scripts: dict[str, list[ChatReply | Exception]],
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.scripts = {key: list(value) for key, value in scripts.items()}
        self.barrier = barrier
        self.spans: dict[str, list[tuple[float, float]]] = {}
        self._lock = threading.Lock()

    def _key(self, messages: list[dict[str, Any]]) -> str:
        goal = next(m["content"] for m in messages if m.get("role") == "user")
        return next(key for key in self.scripts if key in str(goal))

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatReply:
        key = self._key(messages)
        started = time.monotonic()
        if self.barrier is not None:
            # Both children must be inside a call at once, or the wave is
            # not a wave: a sequential run would time out here.
            self.barrier.wait(timeout=10)
        with self._lock:
            answer = self.scripts[key].pop(0)
            self.spans.setdefault(key, []).append((started, time.monotonic()))
        if isinstance(answer, Exception):
            raise answer
        return answer


class LeadClient:
    """The lead's own scripted client; the children never share it."""

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


def _call(tool: str, **arguments: object) -> ChatReply:
    return ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id=f"c_{tool}",
                name=tool,
                arguments=dict(arguments),
                arguments_raw=json.dumps(arguments),
            ),
        ),
        prompt_tokens=100,
        completion_tokens=10,
    )


def _text(text: str) -> ChatReply:
    return ChatReply(content=text, prompt_tokens=100, completion_tokens=10)


def _wave_call(*briefs: tuple[str, str]) -> ChatReply:
    return _call(
        "delegate_many",
        briefs=[{"agent": agent, "task": task} for agent, task in briefs],
    )


def _serve(monkeypatch: pytest.MonkeyPatch, client: WaveClient) -> None:
    """Every child of a wave builds its own client; here they get this one."""
    monkeypatch.setattr("mason.loop.client_from_config", lambda agent, keys=None: client)


def _wave_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripts: dict[str, list[ChatReply | Exception]],
    briefs: tuple[tuple[str, str], ...],
    *,
    barrier: threading.Barrier | None = None,
    **agent: object,
) -> tuple[Mason, LeadClient, WaveClient, Any]:
    session = _session(tmp_path, **agent)
    children = WaveClient(scripts, barrier=barrier)
    _serve(monkeypatch, children)
    lead = LeadClient([_wave_call(*briefs), _text("both reports are in")])
    mason = Mason(session, client=lead)
    result = mason.run_turn("run the two independent steps")
    return mason, lead, children, result


_TWO = (("md-expert", "melt the cell"), ("dft-expert", "relax the cell"))
_TWO_SCRIPTS: dict[str, list[ChatReply | Exception]] = {
    "melt": [_call("finish", report="molten at 1200 K (run aa11bb)")],
    "relax": [_call("finish", report="a = 3.60 A (run cc22dd)")],
}


def test_the_specialists_of_a_wave_run_at_the_same_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The barrier only lets a call through once both children are inside
    # one, so a sequential wave would fail here instead of passing slowly.
    barrier = threading.Barrier(2)
    _mason, _lead, children, result = _wave_turn(
        tmp_path, monkeypatch, dict(_TWO_SCRIPTS), _TWO, barrier=barrier
    )
    assert result.stop_reason == "answer"
    (melt_start, melt_end), = children.spans["melt"]
    (relax_start, relax_end), = children.spans["relax"]
    assert melt_start < relax_end and relax_start < melt_end


def test_every_report_comes_back_in_brief_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mason, lead, _children, _result = _wave_turn(
        tmp_path, monkeypatch, dict(_TWO_SCRIPTS), _TWO
    )
    answer = next(
        m for m in lead.requests[-1] if m.get("tool_call_id") == "c_delegate_many"
    )
    text = str(answer["content"])
    assert "## md-expert (brief 1 of 2)" in text
    assert "## dft-expert (brief 2 of 2)" in text
    assert text.index("## md-expert") < text.index("## dft-expert")
    assert "molten at 1200 K (run aa11bb)" in text
    assert "a = 3.60 A (run cc22dd)" in text
    assert "[md-expert-1: finish after 1 step(s);" in text
    assert "wave: 2 briefs, " in text and "sequential would have been about " in text


def test_the_ordinals_and_transcript_names_follow_brief_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mason, _lead, _children, _result = _wave_turn(
        tmp_path, monkeypatch, dict(_TWO_SCRIPTS), _TWO
    )
    stem = mason.session.transcript_path.stem
    events = [
        json.loads(line)
        for line in mason.session.transcript_path.read_text().splitlines()
    ]
    delegated = [e for e in events if e["type"] == "delegate"]
    assert [e["agent"] for e in delegated] == ["md-expert", "dft-expert"]
    assert [e["transcript"] for e in delegated] == [
        f"{stem}-md-expert-1.jsonl",
        f"{stem}-dft-expert-2.jsonl",
    ]
    assert all(e["wave"] == 1 and e["parallel"] for e in delegated)
    for event in delegated:
        assert (mason.session.sessions_dir / event["transcript"]).is_file()


def test_a_second_wave_is_numbered_after_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts: dict[str, list[ChatReply | Exception]] = {
        "melt": [_call("finish", report="one"), _call("finish", report="three")],
        "relax": [_call("finish", report="two"), _call("finish", report="four")],
    }
    session = _session(tmp_path)
    _serve(monkeypatch, WaveClient(scripts))
    lead = LeadClient([_wave_call(*_TWO), _wave_call(*_TWO), _text("done")])
    mason = Mason(session, client=lead)
    mason.run_turn("two waves")
    events = [
        json.loads(line) for line in session.transcript_path.read_text().splitlines()
    ]
    waves = [e["wave"] for e in events if e["type"] == "delegate"]
    assert waves == [1, 1, 2, 2]


def test_one_failed_brief_leaves_its_sibling_report_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts: dict[str, list[ChatReply | Exception]] = {
        "melt": [RuntimeError("the client died\nsocket closed")],
        "relax": [_call("finish", report="a = 3.60 A (run cc22dd)")],
    }
    _mason, lead, _children, result = _wave_turn(tmp_path, monkeypatch, scripts, _TWO)
    assert result.stop_reason == "answer"
    text = str(
        next(m for m in lead.requests[-1] if m.get("tool_call_id") == "c_delegate_many")[
            "content"
        ]
    )
    assert "stopped: socket closed after step 1; the transcript holds the steps taken" in text
    assert "a = 3.60 A (run cc22dd)" in text
    assert "[md-expert-1: error after 1 step(s);" in text


def test_usage_totals_equal_the_sum_of_the_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mason, _lead, _children, _result = _wave_turn(
        tmp_path, monkeypatch, dict(_TWO_SCRIPTS), _TWO
    )
    # Two lead calls and one call per child, each 100 + 10.
    assert mason.session.prompt_tokens == 400
    assert mason.session.completion_tokens == 40


def test_the_memories_of_every_child_reach_the_lead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def remember(what: str) -> ChatReply:
        return _call(
            "remember",
            name=what,
            description=f"what {what} says",
            body="the machine does this",
            unverified=True,
        )

    scripts: dict[str, list[ChatReply | Exception]] = {
        "melt": [remember("melt-note"), _call("finish", report="one")],
        "relax": [remember("relax-note"), _call("finish", report="two")],
    }
    mason, _lead, _children, _result = _wave_turn(tmp_path, monkeypatch, scripts, _TWO)
    written = {entry["name"] for entry in mason.session.memories_written}
    assert written == {"melt-note", "relax-note"}


def test_concurrent_notebook_entries_stay_whole(tmp_path: Path) -> None:
    from foundation import project as project_files

    session = _session(tmp_path)
    children = [session.spawn(f"hand-{n}", session.agent) for n in range(4)]
    for child, name in zip(children, ("a", "b", "c", "d"), strict=True):
        child.agent_name = name

    def write(child: MasonSession) -> None:
        for n in range(50):
            child.notebook_append(
                f"line one of {child.agent_name}{n}\nline two of {child.agent_name}{n}",
                heading=f"{child.agent_name}{n}",
            )

    threads = [threading.Thread(target=write, args=(child,)) for child in children]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    text = project_files.notebook_path(tmp_path).read_text()
    blocks = [b for b in text.split("\n## ") if b and not b.startswith("# Lab notebook")]
    assert len(blocks) == 200
    for block in blocks:
        heading, *body = block.splitlines()
        who = heading.split(" — ")[1].split(" [")[0]
        assert heading.endswith(f" [{who[0]}]")
        assert [line for line in body if line.strip()] == [
            f"line one of {who}",
            f"line two of {who}",
        ]


def test_the_approver_is_asked_one_question_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []
    inside = threading.Lock()

    def approver(tool_name: str, preview: str) -> bool:
        if not inside.acquire(blocking=False):
            raise AssertionError("two specialists asked at once")
        try:
            asked.append(preview)
            time.sleep(0.01)  # a person reading the preview
        finally:
            inside.release()
        return True

    session = _session(tmp_path)
    session.auto_approve = False
    session.approver = approver
    scripts: dict[str, list[ChatReply | Exception]] = {
        "melt": [_call("shell", command="echo melt"), _call("finish", report="one")],
        "relax": [_call("shell", command="echo relax"), _call("finish", report="two")],
    }
    _serve(monkeypatch, WaveClient(scripts))
    lead = LeadClient([_wave_call(*_TWO), _text("done")])
    Mason(session, client=lead).run_turn("two steps")
    assert sorted(asked) == [
        "[dft-expert] echo relax",
        "[md-expert] echo melt",
    ]


def test_a_brief_to_an_unknown_agent_is_refused_before_any_thread_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[object] = []

    def factory(agent: object, keys: object = None) -> object:
        made.append(agent)
        raise AssertionError("no child should have started")

    monkeypatch.setattr("mason.loop.client_from_config", factory)
    session = _session(tmp_path)
    lead = LeadClient(
        [
            _wave_call(("md-expert", "melt the cell"), ("nobody", "relax the cell")),
            _text("refused"),
        ]
    )
    mason = Mason(session, client=lead)
    mason.run_turn("two steps")
    answer = str(
        next(m for m in lead.requests[-1] if m.get("tool_call_id") == "c_delegate_many")[
            "content"
        ]
    )
    assert answer.startswith("brief 2: no agent named 'nobody'; your team: ")
    assert made == []


def test_more_briefs_than_the_cap_are_refused_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, parallel_delegations=2)
    monkeypatch.setattr(
        "mason.loop.client_from_config",
        lambda agent, keys=None: pytest.fail("no child should have started"),
    )
    lead = LeadClient(
        [
            _call(
                "delegate_many",
                briefs=[
                    {"agent": "md-expert", "task": "one"},
                    {"agent": "md-expert", "task": "two"},
                    {"agent": "md-expert", "task": "three"},
                ],
            ),
            _text("refused"),
        ]
    )
    mason = Mason(session, client=lead)
    mason.run_turn("three steps")
    answer = str(
        next(m for m in lead.requests[-1] if m.get("tool_call_id") == "c_delegate_many")[
            "content"
        ]
    )
    assert "3 briefs is more than this session runs at once (the cap is 2" in answer
    assert "parallel_delegations" in answer


def test_one_brief_is_sent_back_to_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    lead = LeadClient([_wave_call(("md-expert", "melt the cell")), _text("refused")])
    Mason(session, client=lead).run_turn("one step")
    answer = str(
        next(m for m in lead.requests[-1] if m.get("tool_call_id") == "c_delegate_many")[
            "content"
        ]
    )
    assert answer.startswith("a wave carries at least two briefs")


def _tools(tmp_path: Path, **agent: object) -> set[str]:
    from mason.roster import discover_roster
    from mason.tools import build_toolbox

    session = _session(tmp_path, **agent)
    roster = discover_roster(tmp_path)
    return set(build_toolbox(session, roster["pi"], roster=roster, depth=0).tools)


def test_the_switch_and_the_cap_each_remove_the_tool(tmp_path: Path) -> None:
    from mason.mechanisms import ALL_MECHANISMS

    assert "delegate_many" in _tools(tmp_path)
    without = tuple(n for n in ALL_MECHANISMS if n != "parallel-delegation")
    assert "delegate_many" not in _tools(tmp_path, mechanisms=without)
    assert "delegate_many" not in _tools(tmp_path, parallel_delegations=1)
    # The sequential tool survives both.
    assert "delegate" in _tools(tmp_path, parallel_delegations=1)


def test_a_specialist_of_a_wave_has_no_wave_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mason, _lead, _children, _result = _wave_turn(
        tmp_path, monkeypatch, dict(_TWO_SCRIPTS), _TWO
    )
    stem = mason.session.transcript_path.stem
    child = mason.session.sessions_dir / f"{stem}-md-expert-1.jsonl"
    system = next(
        json.loads(line)
        for line in child.read_text().splitlines()
        if json.loads(line).get("type") == "message"
    )
    assert "delegate_many" not in json.dumps(system)


def test_the_team_block_names_the_tool_only_when_it_is_offered(tmp_path: Path) -> None:
    from mason.prompts import team_block
    from mason.roster import discover_roster

    roster = discover_roster(tmp_path)
    with_wave = team_block(roster["pi"], roster, parallel=True)
    without = team_block(roster["pi"], roster, parallel=False)
    assert "delegate_many" in with_wave
    assert "delegate_many" not in without


def test_an_interrupted_child_ends_its_turn_with_what_it_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    client = WaveClient({"melt": [_call("finish", report="never reached")]})
    mason = Mason(session, client=client, spec=None, depth=1)
    mason.stop = threading.Event()
    mason.stop.set()
    result = mason.run_turn("melt the cell")
    assert result.stop_reason == "error"
    assert "the lead interrupted the wave" in result.text
    assert client.spans == {}


def test_every_brief_keeps_its_share_of_the_result_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Capping the joined text instead would drop the middle section whole.
    scripts: dict[str, list[ChatReply | Exception]] = {
        "one": [_call("finish", report="first " + "a" * 4_000)],
        "two": [_call("finish", report="second " + "b" * 4_000)],
        "three": [_call("finish", report="third " + "c" * 4_000)],
    }
    session = _session(tmp_path, max_tool_output_chars=3_000)
    _serve(monkeypatch, WaveClient(scripts))
    lead = LeadClient(
        [
            _wave_call(
                ("md-expert", "step one"),
                ("dft-expert", "step two"),
                ("analysis-expert", "step three"),
            ),
            _text("done"),
        ]
    )
    Mason(session, client=lead).run_turn("three steps")
    text = str(
        next(m for m in lead.requests[-1] if m.get("tool_call_id") == "c_delegate_many")[
            "content"
        ]
    )
    assert len(text) <= 3_000
    for position, (agent, head) in enumerate(
        (("md-expert", "first"), ("dft-expert", "second"), ("analysis-expert", "third")),
        start=1,
    ):
        assert f"## {agent} (brief {position} of 3)" in text
        assert head in text
    assert text.count("characters truncated") == 3


def test_the_report_tallies_the_waves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mason.report import summarize
    from mason.session import transcript_groups

    mason, _lead, _children, _result = _wave_turn(
        tmp_path, monkeypatch, dict(_TWO_SCRIPTS), _TWO
    )
    ((transcript, siblings),) = transcript_groups(mason.session.workspace_root)
    summary = summarize(transcript, siblings)
    assert summary["waves"] == 1
    assert summary["parallel_briefs"] == 2
    assert len(summary["delegations"]) == 2
    assert summary["wave_saved_s"] >= 0.0
