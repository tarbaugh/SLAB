"""Turn-engine tests: a scripted fake backend drives every loop mechanism."""

import json
from pathlib import Path
from typing import Any

import pytest

from slab.config import SlabConfig
from slab.mason.client import ChatReply, ContextOverflowError, ToolCall
from slab.mason.loop import Mason
from slab.mason.session import MasonSession


class FakeClient:
    """Answers from a script; records every request it saw."""

    def __init__(self, replies: list[ChatReply | Exception]) -> None:
        self.replies = list(replies)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatReply:
        self.requests.append(([dict(m) for m in messages], tools))
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _session(tmp_path: Path, **agent: object) -> MasonSession:
    config = SlabConfig.model_validate({"agent": {"model": "fake", **agent}})
    return MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", config=config, auto_approve=True
    )


def _tool_reply(name: str, prompt_tokens: int = 100, **arguments: object) -> ChatReply:
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
        prompt_tokens=prompt_tokens,
        completion_tokens=10,
    )


def _text_reply(text: str, prompt_tokens: int = 100) -> ChatReply:
    return ChatReply(content=text, prompt_tokens=prompt_tokens, completion_tokens=10)


def test_tool_call_then_answer(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("materials\n")
    client = FakeClient(
        [
            _tool_reply("read_file", path="hello.txt"),
            _text_reply("the file says materials"),
        ]
    )
    mason = Mason(_session(tmp_path), client=client)
    result = mason.run_turn("what does hello.txt say?")
    assert result.stop_reason == "answer"
    assert result.text == "the file says materials"
    assert result.steps == 2
    # The tool result reached the model as a tool-role message:
    final_messages = client.requests[1][0]
    tool_messages = [m for m in final_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "materials" in tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"] == "call_read_file"


def test_finish_tool_closes_the_turn(tmp_path: Path) -> None:
    client = FakeClient([_tool_reply("finish", report="a0 = 3.615 A (run ab12)")])
    result = Mason(_session(tmp_path), client=client).run_turn("measure a0")
    assert result.finished
    assert result.stop_reason == "finish"
    assert result.text == "a0 = 3.615 A (run ab12)"


def test_max_turns_stops_loudly(tmp_path: Path) -> None:
    client = FakeClient([_tool_reply("list_dir") for _ in range(3)])
    mason = Mason(_session(tmp_path, max_turns=3), client=client)
    result = mason.run_turn("loop forever")
    assert result.stop_reason == "max_turns"
    assert "3-call budget" in result.text


def test_error_streak_aborts_with_evidence(tmp_path: Path) -> None:
    bad = ChatReply(
        content=None,
        tool_calls=tuple(
            ToolCall(
                id=f"bad_{i}",
                name="shell",
                arguments={},
                arguments_raw="{broken",
                arguments_error="arguments were not valid JSON",
            )
            for i in range(5)
        ),
        prompt_tokens=50,
        completion_tokens=5,
    )
    client = FakeClient([bad])
    result = Mason(_session(tmp_path), client=client).run_turn("go")
    assert result.stop_reason == "error_streak"
    assert "5 consecutive tool calls" in result.text


def test_domain_errors_do_not_count_toward_the_streak(tmp_path: Path) -> None:
    replies: list[ChatReply | Exception] = [
        _tool_reply("read_file", path=f"missing-{i}.txt") for i in range(6)
    ]
    replies.append(_text_reply("those files do not exist"))
    client = FakeClient(replies)
    result = Mason(_session(tmp_path), client=client).run_turn("read them")
    assert result.stop_reason == "answer"  # "no such file" is the model's problem


def test_system_prompt_carries_environment_and_conventions(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Always use PBEsol.\n")
    (tmp_path / "PLAN.md").write_text("1. relax Cu\n")
    client = FakeClient([_text_reply("ok")])
    Mason(_session(tmp_path), client=client).run_turn("hi")
    system = client.requests[0][0][0]
    assert system["role"] == "system"
    assert "You are Mason" in system["content"]
    assert "Always use PBEsol." in system["content"]
    assert "1. relax Cu" in system["content"]
    assert str(tmp_path) in system["content"]


def test_transcript_records_messages_usage_and_finish(tmp_path: Path) -> None:
    client = FakeClient(
        [
            _tool_reply("list_dir"),
            _tool_reply("finish", report="done"),
        ]
    )
    session = _session(tmp_path)
    Mason(session, client=client).run_turn("look around")
    events = [
        json.loads(line) for line in session.transcript_path.read_text().splitlines()
    ]
    kinds = [event["type"] for event in events]
    assert kinds.count("usage") == 2
    assert "finish" in kinds
    messages = [event["message"] for event in events if event["type"] == "message"]
    assert messages[0]["role"] == "user"
    roles = [m["role"] for m in messages]
    assert "assistant" in roles and "tool" in roles


def test_resume_replays_messages_after_fresh_system(tmp_path: Path) -> None:
    client = FakeClient([_text_reply("first answer")])
    session = _session(tmp_path)
    Mason(session, client=client).run_turn("first question")
    transcript = session.latest_transcript()
    assert transcript is not None
    replayed = session.load_messages(transcript)
    session2 = _session(tmp_path)
    client2 = FakeClient([_text_reply("second answer")])
    mason2 = Mason(session2, client=client2, resume_from=replayed)
    mason2.run_turn("second question")
    sent = client2.requests[0][0]
    assert sent[0]["role"] == "system"  # rebuilt fresh, not replayed
    contents = [str(m.get("content")) for m in sent]
    assert any("first question" in c for c in contents)
    assert any("first answer" in c for c in contents)


def test_compaction_folds_history_and_writes_notebook(tmp_path: Path) -> None:
    session = _session(tmp_path, context_window=4_096, compact_at=0.5)
    replies: list[ChatReply | Exception] = [
        _tool_reply("list_dir", prompt_tokens=tokens)
        for tokens in (200, 400, 900, 1_500, 2_500)  # the fifth crosses 2048
    ]
    # The compaction summarizer's answer, then a post-compaction (smaller) step:
    replies.append(_text_reply("STATE: listed the directory five times."))
    replies.append(_tool_reply("list_dir", prompt_tokens=300))
    replies.append(_text_reply("done looking"))
    client = FakeClient(replies)
    mason = Mason(session, client=client)
    result = mason.run_turn("inspect")
    assert result.stop_reason == "answer"
    notebook = (tmp_path / "NOTEBOOK.md").read_text()
    assert "context compaction" in notebook
    assert "listed the directory five times" in notebook
    # The compacted conversation carries the summary marker:
    compacted_request = client.requests[6][0]
    assert any(
        "[history compacted; working summary]" in str(m.get("content"))
        for m in compacted_request
    )
    # And the summarizer was called without tools:
    assert client.requests[5][1] is None


def test_context_overflow_from_server_forces_compaction(tmp_path: Path) -> None:
    session = _session(tmp_path, context_window=1_000_000)  # estimate never triggers
    replies: list[ChatReply | Exception] = [
        _tool_reply("list_dir") for _ in range(8)
    ]
    replies.append(ContextOverflowError("maximum context length exceeded"))
    replies.append(_text_reply("STATE: summary after overflow"))
    replies.append(_text_reply("recovered"))
    client = FakeClient(replies)
    result = Mason(session, client=client).run_turn("go")
    assert result.stop_reason == "answer"
    assert result.text == "recovered"
    assert "summary after overflow" in (tmp_path / "NOTEBOOK.md").read_text()


def test_fenced_protocol_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("fenced works\n")
    session = _session(tmp_path, tool_protocol="fenced")
    client = FakeClient(
        [
            _text_reply(
                'I will read it.\n```tool\n{"tool": "read_file", '
                '"arguments": {"path": "f.txt"}}\n```'
            ),
            _text_reply("it says: fenced works"),
        ]
    )
    result = Mason(session, client=client).run_turn("read f.txt")
    assert result.stop_reason == "answer"
    # No tools= in the request; the catalog lives in the system prompt instead.
    assert client.requests[0][1] is None
    system = client.requests[0][0][0]["content"]
    assert "Tool protocol (fenced)" in system
    assert "- read_file(" in system
    # The tool result went back as a labeled user message:
    followup = client.requests[1][0]
    assert any(
        m["role"] == "user" and "[tool result: read_file]" in str(m["content"])
        for m in followup
    )


def test_loose_text_call_is_executed_and_answered_as_user_message(tmp_path: Path) -> None:
    """llama3.1 sometimes writes the tool call as plain JSON text — run it anyway."""
    (tmp_path / "g.txt").write_text("loose works\n")
    client = FakeClient(
        [
            _text_reply(
                "I will read it now:\n"
                '{"name": "read_file", "parameters": {"path": "g.txt"}}'
            ),
            _text_reply("it says: loose works"),
        ]
    )
    result = Mason(_session(tmp_path), client=client).run_turn("read g.txt")
    assert result.stop_reason == "answer"
    assert result.text == "it says: loose works"
    followup = client.requests[1][0]
    # The result went back as a labeled user message (no server-side tool_calls
    # exist to pair a tool-role message with):
    assert any(
        m["role"] == "user" and "[tool result: read_file]" in str(m["content"])
        and "loose works" in str(m["content"])
        for m in followup
    )
    assert not any(m.get("role") == "tool" for m in followup)


def test_short_circuit_answers_every_tool_call(tmp_path: Path) -> None:
    """finish mid-list must not leave later tool_call ids unanswered."""
    reply = ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id="f1",
                name="finish",
                arguments={"report": "done"},
                arguments_raw='{"report": "done"}',
            ),
            ToolCall(id="c2", name="list_dir", arguments={}, arguments_raw="{}"),
        ),
        prompt_tokens=10,
        completion_tokens=5,
    )
    session = _session(tmp_path)
    mason = Mason(session, client=FakeClient([reply]))
    result = mason.run_turn("go")
    assert result.stop_reason == "finish"
    answered = {m["tool_call_id"] for m in mason.messages if m.get("role") == "tool"}
    assert answered == {"f1", "c2"}  # every declared call has a result


def test_missing_arguments_count_toward_the_error_streak(tmp_path: Path) -> None:
    bad = ChatReply(
        content=None,
        tool_calls=tuple(
            ToolCall(id=f"m{i}", name="read_file", arguments={}, arguments_raw="{}")
            for i in range(5)
        ),
        prompt_tokens=10,
        completion_tokens=5,
    )
    result = Mason(_session(tmp_path), client=FakeClient([bad])).run_turn("go")
    assert result.stop_reason == "error_streak"


def test_resume_transcripts_stay_self_contained(tmp_path: Path) -> None:
    """Resuming a resumed session must not amputate the first session."""
    session1 = _session(tmp_path)
    Mason(session1, client=FakeClient([_text_reply("first answer")])).run_turn("first question")
    replay1 = session1.load_messages(session1.transcript_path)

    session2 = _session(tmp_path)
    Mason(
        session2, client=FakeClient([_text_reply("second answer")]), resume_from=replay1
    ).run_turn("second question")

    session3 = _session(tmp_path)
    latest = session3.latest_transcript()
    assert latest == session2.transcript_path
    replay2 = session3.load_messages(latest)
    contents = " ".join(str(m.get("content")) for m in replay2)
    assert "first question" in contents and "first answer" in contents
    assert "second question" in contents and "second answer" in contents


def test_compaction_does_not_refire_without_progress(tmp_path: Path) -> None:
    """An over-budget conversation with nothing new must not re-summarize."""
    session = _session(tmp_path, context_window=4_096, compact_at=0.1)
    replies: list[ChatReply | Exception] = [
        _tool_reply("list_dir", prompt_tokens=3_000) for _ in range(3)
    ]
    replies.insert(1, _text_reply("STATE: folded once"))  # the one summarizer call
    replies.append(_text_reply("done"))
    client = FakeClient(replies)
    mason = Mason(session, client=client)
    # Pad history so the first compaction has something to fold.
    for i in range(8):
        mason.messages.append({"role": "user", "content": f"filler {i}"})
    result = mason.run_turn("go")
    assert result.stop_reason == "answer"
    assert (tmp_path / "NOTEBOOK.md").read_text().count("context compaction") == 1


def test_overflow_with_nothing_to_compact_is_a_teaching_error(tmp_path: Path) -> None:
    session = _session(tmp_path, context_window=1_000_000)
    client = FakeClient([ContextOverflowError("maximum context length is 4096")])
    with pytest.raises(ContextOverflowError, match="nothing left to compact"):
        Mason(session, client=client).run_turn("go")


def test_compaction_boundary_never_orphans_tool_messages(tmp_path: Path) -> None:
    session = _session(tmp_path)
    client = FakeClient([_text_reply("STATE: folded")])
    mason = Mason(session, client=client)
    # Build a long synthetic history ending in assistant+tool pairs.
    mason.messages.append({"role": "user", "content": "goal"})
    for i in range(10):
        mason.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            }
        )
        mason.messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "r" * 200})
    mason._compact()
    # Wherever the cut landed, no tool message may open the kept tail:
    kept = mason.messages
    first_tool = next(i for i, m in enumerate(kept) if m.get("role") == "tool")
    assert kept[first_tool - 1].get("tool_calls") is not None


def test_client_from_config_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from slab.errors import SlabError
    from slab.mason.loop import client_from_config

    config = SlabConfig()
    with pytest.raises(SlabError, match="no model configured"):
        client_from_config(config.agent)
    monkeypatch.delenv("MISSING_KEY_VAR", raising=False)
    config = SlabConfig.model_validate(
        {"agent": {"model": "m", "api_key_env": "MISSING_KEY_VAR"}}
    )
    with pytest.raises(SlabError, match=r"\$MISSING_KEY_VAR"):
        client_from_config(config.agent)
    monkeypatch.setenv("MISSING_KEY_VAR", "sk-123")
    client = client_from_config(config.agent)
    assert client.api_key == "sk-123"
