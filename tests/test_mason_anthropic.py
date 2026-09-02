"""Anthropic provider tests: shape translation, parsing, refusals, wiring."""

import json
from typing import Any

import pytest

from conftest import LlmScript
from mason.anthropic import (
    AnthropicClient,
    ModelRefusalError,
    parse_reply,
    translate_messages,
)
from mason.client import ContextOverflowError, LlmError
from mason.config import MasonConfig

# -- message translation -----------------------------------------------------


def test_system_messages_leave_the_turn_list() -> None:
    system, turns = translate_messages(
        [
            {"role": "system", "content": "core prompt"},
            {"role": "user", "content": "hello"},
        ]
    )
    assert system == "core prompt"
    assert turns == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_tool_calls_and_results_become_content_blocks() -> None:
    _system, turns = translate_messages(
        [
            {"role": "user", "content": "read both"},
            {
                "role": "assistant",
                "content": "on it",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "b.py"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "aaa"},
            {"role": "tool", "tool_call_id": "c2", "content": "bbb"},
        ]
    )
    assistant = turns[1]
    assert assistant["role"] == "assistant"
    assert [block["type"] for block in assistant["content"]] == ["text", "tool_use", "tool_use"]
    assert assistant["content"][1]["input"] == {"path": "a.py"}
    # Both results must ride in ONE user message: splitting them teaches the
    # model to stop making parallel tool calls.
    assert len(turns) == 3
    results = turns[2]
    assert results["role"] == "user"
    assert [block["tool_use_id"] for block in results["content"]] == ["c1", "c2"]
    assert results["content"][0]["content"] == "aaa"


def test_consecutive_same_role_turns_are_merged() -> None:
    _system, turns = translate_messages(
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    )
    assert len(turns) == 1
    assert [block["text"] for block in turns[0]["content"]] == ["first", "second"]


def test_empty_assistant_turns_are_dropped() -> None:
    """An assistant turn with neither text nor tool calls is rejected upstream."""
    _system, turns = translate_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "still there?"},
        ]
    )
    assert [turn["role"] for turn in turns] == ["user"]
    assert len(turns[0]["content"]) == 2


def test_history_opening_on_an_assistant_turn_gets_a_user_turn_prepended() -> None:
    """Prepending beats dropping: a dropped tool_use would orphan its result."""
    _system, turns = translate_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
    )
    assert turns[0]["role"] == "user"
    assert turns[1]["content"][0]["type"] == "tool_use"
    assert turns[2]["content"][0]["type"] == "tool_result"


def test_malformed_tool_arguments_degrade_to_an_empty_object() -> None:
    _system, turns = translate_messages(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{oops"},
                    }
                ],
            },
        ]
    )
    assert turns[1]["content"][0]["input"] == {}


def test_empty_tool_output_still_carries_content() -> None:
    """A tool_result block with empty content is rejected by the API."""
    _system, turns = translate_messages(
        [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "c1", "content": ""},
        ]
    )
    blocks = turns[-1]["content"]
    assert blocks[0]["type"] == "tool_result"  # results lead their user turn
    assert blocks[0]["content"] == "(no output)"


def test_tool_results_lead_a_merged_user_turn() -> None:
    """The API requires tool_result blocks first in the user message."""
    _system, turns = translate_messages(
        [
            {"role": "user", "content": "plain text"},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
    )
    assert [block["type"] for block in turns[-1]["content"]] == ["tool_result", "text"]


# -- response parsing --------------------------------------------------------


def test_parse_reply_collects_text_thinking_and_tool_use() -> None:
    reply = parse_reply(
        {
            "content": [
                {"type": "thinking", "thinking": "considering"},
                {"type": "text", "text": "here goes"},
                {"type": "tool_use", "id": "toolu_1", "name": "shell", "input": {"command": "ls"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
    )
    assert reply.content == "here goes"
    assert reply.reasoning == "considering"
    assert reply.tool_calls[0].name == "shell"
    assert reply.tool_calls[0].arguments == {"command": "ls"}
    assert json.loads(reply.tool_calls[0].arguments_raw) == {"command": "ls"}
    assert reply.finish_reason == "tool_use"


def test_prompt_tokens_include_cached_input() -> None:
    """The compaction trigger needs the prompt's size, not its billed size."""
    reply = parse_reply(
        {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 500,
                "cache_read_input_tokens": 40_000,
                "cache_creation_input_tokens": 1_000,
                "output_tokens": 12,
            },
        }
    )
    assert reply.prompt_tokens == 41_500
    assert reply.completion_tokens == 12
    assert reply.cached_prompt_tokens == 40_000  # the cheap share, for the cost reading


def test_refusal_raises_instead_of_looking_like_an_empty_answer() -> None:
    with pytest.raises(ModelRefusalError, match="category: cyber"):
        parse_reply(
            {
                "content": [],
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "cyber"},
            }
        )


def test_missing_content_list_is_a_loud_error() -> None:
    with pytest.raises(LlmError, match="no content list"):
        parse_reply({"stop_reason": "end_turn"})


# -- live wire ---------------------------------------------------------------


def _message(content: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": content, **extra}


def test_request_shape_matches_the_messages_api(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append((200, _message([{"type": "text", "text": "ok"}])))
    client = AnthropicClient("claude-opus-5", "sk-ant-test", endpoint=url, effort="medium")
    client.chat(
        [
            {"role": "system", "content": "be careful"},
            {"role": "user", "content": "hello"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
    )
    sent = script.requests[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["max_tokens"] == 16_000  # required by the API; bounds thinking too
    assert sent["system"][0]["text"] == "be careful"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}  # prefix caching
    assert sent["tools"] == [
        {
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]
    assert sent["output_config"] == {"effort": "medium"}
    # Sampling parameters are rejected by current Claude models — never sent:
    assert "temperature" not in sent
    assert "top_p" not in sent and "top_k" not in sent


def test_auth_uses_the_x_api_key_and_version_headers(
    llm_server: tuple[str, LlmScript],
) -> None:
    url, script = llm_server
    script.responses.append((200, _message([{"type": "text", "text": "ok"}])))
    AnthropicClient("claude-opus-5", "sk-ant-secret", endpoint=url).chat([])
    headers = script.requests[0]["_headers"]
    assert headers["x-api-key"] == "sk-ant-secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert script.requests[0]["_auth"] is None  # not a Bearer-token API


def test_context_overflow_is_detected_from_anthropic_wording(
    llm_server: tuple[str, LlmScript],
) -> None:
    url, script = llm_server
    script.responses.append(
        (
            400,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 250000 tokens > 200000 maximum",
                },
            },
        )
    )
    with pytest.raises(ContextOverflowError):
        AnthropicClient("claude-opus-5", "k", endpoint=url).chat([])


def test_rate_limit_is_reported_plainly(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append(
        (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    )
    with pytest.raises(LlmError, match="rate-limited"):
        AnthropicClient("claude-opus-5", "k", endpoint=url).chat([])


def test_model_names_lists_ids(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "claude-opus-5"}, {"id": "claude-haiku-4-5"}]})
    assert AnthropicClient("claude-opus-5", "k", endpoint=url).model_names() == [
        "claude-opus-5",
        "claude-haiku-4-5",
    ]


def test_max_reply_tokens_is_configurable(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append((200, _message([{"type": "text", "text": "ok"}])))
    AnthropicClient("claude-opus-5", "k", endpoint=url, max_reply_tokens=64_000).chat([])
    assert script.requests[0]["max_tokens"] == 64_000


def test_provider_drives_a_verified_relax_end_to_end(
    tmp_path: Any, llm_server: tuple[str, LlmScript]
) -> None:
    """Everything but Anthropic's servers: real tools, real EMT, real checks.

    The scripted answers use the documented Messages wire shape, so this
    exercises the translation in both directions across a whole turn.
    """
    from mason.loop import Mason
    from mason.session import MasonSession

    url, script = llm_server
    workflow = "\n".join(
        [
            "from foundation import check, converged",
            "from foundation.tasks import relax",
            "from ase.build import bulk",
            "atoms = bulk('Cu', 'fcc', a=3.6)",
            "relaxed, info = relax(atoms, engine='emt', fmax=0.05, label='cu')",
            "print('energy (eV):', info['energy'])",
            "@check",
            "def forces_converged():",
            "    return converged(info['fmax'], below=0.05)",
            "",
        ]
    )
    script.responses.extend(
        [
            (
                200,
                _message(
                    [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "write_file",
                            "input": {"path": "wf.py", "content": workflow},
                        }
                    ],
                    stop_reason="tool_use",
                    usage={"input_tokens": 900, "output_tokens": 120},
                ),
            ),
            (
                200,
                _message(
                    [
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "launch_workflow",
                            "input": {"script": "wf.py", "intent": "laptop smoke test"},
                        }
                    ],
                    stop_reason="tool_use",
                    usage={"input_tokens": 1200, "output_tokens": 80},
                ),
            ),
            (
                200,
                _message(
                    [{"type": "text", "text": "the run is verified"}],
                    stop_reason="end_turn",
                    usage={"input_tokens": 1500, "output_tokens": 30},
                ),
            ),
        ]
    )
    config = MasonConfig.model_validate(
        {"agent": {"provider": "anthropic", "model": "claude-opus-5", "endpoint": url}}
    )
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent, auto_approve=True
    )
    result = Mason(session, client=AnthropicClient("claude-opus-5", "k", endpoint=url)).run_turn(
        "relax Cu"
    )
    assert result.stop_reason == "answer"

    from foundation.runtime import Workspace

    with Workspace(tmp_path / ".slab") as ws:
        runs = ws.runs.list_runs()
        assert len(runs) == 1
        assert runs[0].state.value == "verified"  # the checks actually ran
    # The tool result went back as a tool_result block inside a user turn:
    second_request = script.requests[1]
    first_block = second_request["messages"][-1]["content"][0]
    assert first_block["type"] == "tool_result"
    assert first_block["tool_use_id"] == "toolu_1"
    # ...and the prompt tokens the server reported reached the session:
    assert session.prompt_tokens == 900 + 1200 + 1500


# -- config wiring -----------------------------------------------------------


def test_provider_selects_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from mason.loop import client_from_config

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    config = MasonConfig.model_validate(
        {"agent": {"provider": "anthropic", "model": "claude-opus-5", "effort": "high"}}
    )
    client = client_from_config(config.agent)
    assert isinstance(client, AnthropicClient)
    assert client.endpoint == "https://api.anthropic.com/v1"
    assert client.effort == "high"

    openai_config = MasonConfig.model_validate({"agent": {"model": "llama3.1:8b"}})
    from mason.client import ChatClient

    assert isinstance(client_from_config(openai_config.agent), ChatClient)


def test_anthropic_without_a_key_refuses_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mason.errors import MasonError
    from mason.loop import client_from_config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = MasonConfig.model_validate(
        {"agent": {"provider": "anthropic", "model": "claude-opus-5"}}
    )
    with pytest.raises(MasonError, match=r"\$ANTHROPIC_API_KEY is not set"):
        client_from_config(config.agent)


def test_cli_doctor_speaks_to_the_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, llm_server: tuple[str, LlmScript]
) -> None:
    from typer.testing import CliRunner

    from mason.cli import app

    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "claude-opus-5"}]})
    script.responses.append(
        (
            200,
            _message(
                [{"type": "tool_use", "id": "toolu_1", "name": "ping", "input": {}}],
                stop_reason="tool_use",
            ),
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    result = CliRunner().invoke(
        app,
        [
            "doctor",
            "--provider", "anthropic",
            "--endpoint", url,
            "--model", "claude-opus-5",
        ],
    )
    assert result.exit_code == 0
    assert "provider: anthropic" in result.output
    assert "[+] model 'claude-opus-5' is served" in result.output
    assert "[+] native tool calls work" in result.output


def test_cli_doctor_refuses_anthropic_without_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from typer.testing import CliRunner

    from mason.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["doctor", "--provider", "anthropic"])
    assert result.exit_code == 1
    assert "$ANTHROPIC_API_KEY is not set" in result.output


def test_endpoint_defaults_are_per_provider() -> None:
    assert MasonConfig().agent.resolved_endpoint == "http://localhost:11434/v1"
    anthropic = MasonConfig.model_validate({"agent": {"provider": "anthropic"}}).agent
    assert anthropic.resolved_endpoint == "https://api.anthropic.com/v1"
    assert anthropic.resolved_api_key_env == "ANTHROPIC_API_KEY"
    override = MasonConfig.model_validate(
        {"agent": {"provider": "anthropic", "endpoint": "http://proxy/v1"}}
    ).agent
    assert override.resolved_endpoint == "http://proxy/v1"
