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
    assert "mason ready: pi — fake" in result.output
    assert "hello!" in result.output
    assert "tokens: 10 prompt, 2 completion" in result.output
    # the id a person types into 'foundation promote --session'
    assert "session 20" in result.output


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


def test_provider_typo_is_refused_not_rerouted(tmp_path: Path) -> None:
    """CLI overrides go through the model's validation, so a mistyped
    provider is refused by name instead of silently probing the openai
    branch with an Anthropic model."""
    result = runner.invoke(
        app, ["doctor", "--provider", "anthorpic", "-w", str(tmp_path / ".slab")]
    )
    assert result.exit_code == 1
    assert "invalid --provider" in result.output
    assert "'openai' or 'anthropic'" in result.output


def test_max_turns_below_one_is_refused(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", "hi", "--max-turns", "0", "-w", str(tmp_path / ".slab")]
    )
    assert result.exit_code == 1
    assert "invalid --max-turns" in result.output


def _tool_probe_response() -> tuple[int, dict[str, Any]]:
    return (
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


def test_mason_doctor_probes_roster_connections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm_server: tuple[str, LlmScript]
) -> None:
    """A specialist pinned to a served model gets a [+]; the doctor stays green."""
    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "primary"}, {"id": "bigger"}]})
    script.responses.append(_tool_probe_response())
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "primary"\n\n[agent.roster.dft-expert]\nmodel = "bigger"\n'
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--endpoint", url])
    assert result.exit_code == 0
    assert "[+] dft-expert: model 'bigger' is served" in result.output


def test_mason_doctor_fails_a_specialist_pinned_to_an_unserved_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm_server: tuple[str, LlmScript]
) -> None:
    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "primary"}]})
    script.responses.append(_tool_probe_response())
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "primary"\n\n[agent.roster.dft-expert]\nmodel = "missing"\n'
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--endpoint", url])
    assert result.exit_code == 1
    assert "[x] dft-expert: model 'missing' not served" in result.output


def test_mason_doctor_flags_a_roster_table_without_a_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm_server: tuple[str, LlmScript]
) -> None:
    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "primary"}]})
    script.responses.append(_tool_probe_response())
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "primary"\n\n[agent.roster.nobody]\nmodel = "primary"\n'
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--endpoint", url])
    assert result.exit_code == 1
    assert "[x] roster: " in result.output
    assert "[agent.roster.nobody]" in result.output


# -- reasoning display ---------------------------------------------------------


def test_mason_chat_prints_reasoning_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_client(
        monkeypatch,
        [
            ChatReply(
                content="hello!",
                reasoning="a warm greeting back",
                prompt_tokens=10,
                completion_tokens=2,
            )
        ],
    )
    result = runner.invoke(
        app,
        ["chat", "-w", str(tmp_path / ".slab"), "--model", "fake"],
        input="hi\n/quit\n",
    )
    assert result.exit_code == 0
    assert "[reasoning] a warm greeting back" in result.output
    assert "hello!" in result.output


def test_show_reasoning_false_silences_the_chat_display(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "fake"\nshow_reasoning = false\n'
    )
    _patch_client(
        monkeypatch,
        [
            ChatReply(
                content="hello!", reasoning="hidden", prompt_tokens=10, completion_tokens=2
            )
        ],
    )
    result = runner.invoke(
        app, ["chat", "-w", str(tmp_path / ".slab")], input="hi\n/quit\n"
    )
    assert result.exit_code == 0
    assert "[reasoning]" not in result.output
    assert "hello!" in result.output


def test_mason_run_never_prints_reasoning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    reply = _finish("a0 = 3.615 A (run ab12cd)").model_copy(
        update={"reasoning": "checked the run record"}
    )
    _patch_client(monkeypatch, [reply])
    result = runner.invoke(app, ["run", "measure a0", "-w", str(tmp_path / ".slab")])
    assert result.exit_code == 0
    assert "[reasoning]" not in result.output
    assert "checked the run record" not in result.output


def test_echo_step_clips_long_traces(capsys: pytest.CaptureFixture[str]) -> None:
    from mason.cli import _STEP_PREVIEW_CHARS, _echo_step

    _echo_step("reasoning", "", "x" * (_STEP_PREVIEW_CHARS + 500))
    out = capsys.readouterr().out
    assert "[... 500 more characters in the transcript]" in out


def test_an_endpoint_flag_reaches_delegated_children(tmp_path: Path) -> None:
    """--endpoint joins flag_updates: a child re-resolves its endpoint, and
    without the flag it would rediscover the serve record's URL — in the
    sandbox, exactly the address the namespace cannot reach."""
    from mason.cli import _mason_session
    from mason.config import roster_agent_config

    session = _mason_session(
        tmp_path / "ws",
        auto=True,
        model=None,
        endpoint="http://127.0.0.1:8000/v1",
        interactive=False,
    )
    assert session.flag_updates["endpoint"] == "http://127.0.0.1:8000/v1"
    # The delegation path: roster config + re-asserted flags.
    from mason.config import override_agent

    effective = override_agent(
        roster_agent_config(session.base_agent, "dft-expert"), dict(session.flag_updates)
    )
    child = session.spawn("dft-expert", effective)
    assert child.endpoint == "http://127.0.0.1:8000/v1"


def test_mason_read_renders_a_transcript_for_humans(tmp_path: Path) -> None:
    import json

    transcript = tmp_path / "20260827-000000-1.jsonl"
    events = [
        {"at": "2026-08-27T10:00:00+00:00", "type": "message",
         "message": {"role": "user", "content": "relax Cu"}},
        {"at": "2026-08-27T10:00:05+00:00", "type": "usage",
         "prompt_tokens": 100, "completion_tokens": 20},
        {"at": "2026-08-27T10:00:05+00:00", "type": "reasoning", "text": "think " * 600},
        {"at": "2026-08-27T10:00:05+00:00", "type": "message",
         "message": {"role": "assistant", "content": None, "tool_calls": [
             {"id": "t1", "type": "function",
              "function": {"name": "shell", "arguments": '{"command": "ls"}'}}]}},
        {"at": "2026-08-27T10:00:06+00:00", "type": "message",
         "message": {"role": "tool", "tool_call_id": "t1", "content": "exit 0\nfiles"}},
        {"at": "2026-08-27T10:00:09+00:00", "type": "finish", "report": "done, run r1"},
    ]
    lines = [json.dumps(e) for e in events]
    lines.insert(3, "{broken")
    transcript.write_text("\n".join(lines) + "\n")

    result = runner.invoke(app, ["read", str(transcript)])
    assert result.exit_code == 0, result.output
    assert "=== user @ 10:00:00" in result.output
    assert "relax Cu" in result.output
    assert "[reasoning @ 10:00:05]" in result.output
    assert "--full shows them" in result.output  # long reasoning clipped
    assert '-> shell {"command": "ls"}' in result.output
    assert "exit 0" in result.output
    assert "=== final report @ 10:00:09" in result.output
    assert "[line 4: not valid JSON; skipped]" in result.output
    assert "[1 model call(s); tokens 100+20]" in result.output

    unclipped = runner.invoke(app, ["read", str(transcript), "--full"])
    assert "--full shows them" not in unclipped.output

    missing = runner.invoke(app, ["read", str(tmp_path / "nope.jsonl")])
    assert missing.exit_code != 0


# -- session stamps ----------------------------------------------------------


def test_mason_run_reports_and_exports_its_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The summary line names the session, and the exported variable stamps
    the runs the agent's own shell launches."""
    import os

    monkeypatch.chdir(tmp_path)
    _patch_client(monkeypatch, [_finish("done")])
    result = runner.invoke(app, ["run", "measure a0", "-w", str(tmp_path / ".slab")])
    assert result.exit_code == 0
    exported = os.environ["SLAB_SESSION"]
    assert f"session {exported}" in result.output
    transcripts = list((tmp_path / ".slab" / "mason" / "sessions").glob("*.jsonl"))
    assert [t.stem for t in transcripts] == [exported]


def test_launched_runs_carry_the_session_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A goal that launches a workflow lands a run promotable by session."""
    from foundation import Workspace

    monkeypatch.chdir(tmp_path)
    (tmp_path / "wf.py").write_text("x = 1\n")
    launch = ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id="l1",
                name="launch_workflow",
                arguments={"script": "wf.py", "intent": "stamped"},
                arguments_raw=json.dumps({"script": "wf.py", "intent": "stamped"}),
            ),
        ),
        prompt_tokens=1,
        completion_tokens=1,
    )
    _patch_client(monkeypatch, [launch, _finish("launched")])
    result = runner.invoke(app, ["run", "run wf", "-w", str(tmp_path / ".slab"), "--auto"])
    assert result.exit_code == 0, result.output
    with Workspace(tmp_path / ".slab") as ws:
        (run,) = ws.runs.list_runs()
        (row,) = ws.runs.list_sessions()
        assert run.session == row.session
        assert f"session {row.session}" in result.output
