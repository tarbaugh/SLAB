"""CLI tests for mason: doctor against a live mock server, run/chat with a fake."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import LlmScript
from mason.cli import app
from mason.client import ChatReply, ToolCall

runner = CliRunner()


class FakeClient:
    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatReply:
        return self.replies.pop(0)


def _patch_client(monkeypatch: pytest.MonkeyPatch, replies: list[ChatReply]) -> None:
    fake = FakeClient(replies)
    monkeypatch.setattr("mason.loop.client_from_config", lambda agent: fake)


def _finish(report: str) -> ChatReply:
    return ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id="f1",
                name="finish",
                arguments={"report": report},
                arguments_raw=json.dumps({"report": report}),
            ),
        ),
        prompt_tokens=50,
        completion_tokens=10,
    )


def test_mason_run_one_shot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_client(monkeypatch, [_finish("a0 = 3.615 A (run ab12cd)")])
    result = runner.invoke(app, ["run", "measure a0", "-w", str(tmp_path / ".slab")])
    assert result.exit_code == 0
    assert "a0 = 3.615 A (run ab12cd)" in result.output
    assert "finish after 1 step(s)" in result.output


def test_mason_run_exit_code_reflects_harness_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    list_call = ChatReply(
        content=None,
        tool_calls=(
            ToolCall(id="c1", name="list_dir", arguments={}, arguments_raw="{}"),
        ),
        prompt_tokens=10,
        completion_tokens=5,
    )
    (tmp_path / "slab.toml").write_text('[agent]\nmodel = "fake"\nmax_turns = 2\n')
    _patch_client(monkeypatch, [list_call, list_call])
    result = runner.invoke(app, ["run", "loop", "-w", str(tmp_path / ".slab")])
    assert result.exit_code == 1
    assert "2-call budget" in result.output


def test_mason_run_unconfigured_model_fails_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "hi", "-w", str(tmp_path / ".slab")])
    assert result.exit_code == 1
    assert "no model configured" in result.output


def test_mason_chat_status_and_quit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_client(monkeypatch, [ChatReply(content="hello!", prompt_tokens=10, completion_tokens=2)])
    result = runner.invoke(
        app,
        ["chat", "-w", str(tmp_path / ".slab"), "--model", "fake"],
        input="hi\n/status\n/quit\n",
    )
    assert result.exit_code == 0
    assert "mason ready: fake" in result.output
    assert "hello!" in result.output
    assert "tokens: 10 prompt, 2 completion" in result.output


def test_mason_chat_resume_requires_a_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_client(monkeypatch, [])
    result = runner.invoke(
        app, ["chat", "--resume", "-w", str(tmp_path / ".slab"), "--model", "fake"]
    )
    assert result.exit_code == 1
    assert "nothing to resume" in result.output


def test_mason_doctor_healthy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm_server: tuple[str, LlmScript]
) -> None:
    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "llama3.1:8b"}]})
    script.responses.append(
        (
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"id": "p1", "function": {"name": "ping", "arguments": "{}"}}
                            ],
                        }
                    }
                ]
            },
        )
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["doctor", "--endpoint", url, "--model", "llama3.1:8b"]
    )
    assert result.exit_code == 0
    assert "[+] endpoint answers; 1 model(s) served" in result.output
    assert "[+] model 'llama3.1:8b' is served" in result.output
    assert "[+] native tool calls work" in result.output


def test_mason_doctor_flags_missing_model_and_no_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm_server: tuple[str, LlmScript]
) -> None:
    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "other-model"}]})
    script.responses.append(
        (200, {"choices": [{"message": {"content": "pong, verbally"}}]})
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--endpoint", url, "--model", "wanted"])
    assert result.exit_code == 1
    assert "[x] model 'wanted' not served; available: other-model" in result.output
    assert "tool_protocol" in result.output  # teaches the fenced fallback


def test_mason_doctor_unreachable_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mason.client.time.sleep", lambda s: None)
    result = runner.invoke(
        app, ["doctor", "--endpoint", "http://127.0.0.1:9/v1", "--model", "m"]
    )
    assert result.exit_code == 1
    assert "[x] endpoint:" in result.output


def test_mason_doctor_without_model_lists_served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm_server: tuple[str, LlmScript]
) -> None:
    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "a"}, {"id": "b"}]})
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--endpoint", url])
    assert result.exit_code == 1
    assert "no model configured; served here: a, b" in result.output


def test_ask_approval_refuses_on_noninteractive_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under sbatch/nohup stdin is /dev/null: the gate must refuse, not die."""
    from click.exceptions import Abort

    from mason.cli import _ask_approval

    def no_tty(*args: object, **kwargs: object) -> bool:
        raise Abort()

    monkeypatch.setattr("slab.cli.typer.confirm", no_tty)
    assert _ask_approval("write_file", "path='x'") is False


def test_mason_run_gets_the_refusing_gate_not_a_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mason run promises 'mutating tools are refused' without --auto — the
    session must carry the refuse-everything gate, never the terminal prompt."""
    from mason.cli import _mason_session

    monkeypatch.chdir(tmp_path)
    batch = _mason_session(
        tmp_path / "ws", auto=False, model=None, endpoint=None, interactive=False
    )
    assert batch.approver.__name__ == "_approve_nothing"
    chat = _mason_session(tmp_path / "ws", auto=False, model=None, endpoint=None)
    assert chat.approver.__name__ == "_ask_approval"
