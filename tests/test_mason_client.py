"""Chat-client tests against a real local HTTP server (urllib path included)."""

from typing import Any

import pytest

from conftest import LlmScript
from slab.mason.client import (
    ChatClient,
    ContextOverflowError,
    LlmError,
    parse_fenced_calls,
)


def _reply(message: dict[str, Any], usage: dict[str, int] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"choices": [{"message": message, "finish_reason": "stop"}]}
    if usage:
        payload["usage"] = usage
    return payload


def test_chat_round_trip_with_tools_and_usage(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append(
        (
            200,
            _reply(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                        }
                    ],
                },
                usage={"prompt_tokens": 120, "completion_tokens": 15},
            ),
        )
    )
    client = ChatClient(url, "test-model", api_key="sk-test", temperature=0.5)
    reply = client.chat([{"role": "user", "content": "read a.py"}], tools=[{"type": "function"}])
    assert reply.tool_calls[0].name == "read_file"
    assert reply.tool_calls[0].arguments == {"path": "a.py"}
    assert reply.prompt_tokens == 120 and reply.completion_tokens == 15
    sent = script.requests[0]
    assert sent["model"] == "test-model"
    assert sent["temperature"] == 0.5
    assert sent["tools"] == [{"type": "function"}]
    assert sent["_auth"] == "Bearer sk-test"
    assert "tool_choice" not in sent  # Ollama does not support it


def test_malformed_arguments_survive_as_repair_evidence(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append(
        (
            200,
            _reply(
                {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "shell", "arguments": '{"command": broken'}}
                    ],
                }
            ),
        )
    )
    reply = ChatClient(url, "m").chat([])
    call = reply.tool_calls[0]
    assert call.id == "call_0"  # missing id synthesized
    assert call.arguments == {}
    assert call.arguments_error is not None and "not valid JSON" in call.arguments_error
    assert call.arguments_raw == '{"command": broken'


def test_pre_parsed_dict_arguments_are_normalized(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append(
        (
            200,
            _reply(
                {
                    "content": None,
                    "tool_calls": [{"function": {"name": "t", "arguments": {"x": 1}}}],
                }
            ),
        )
    )
    reply = ChatClient(url, "m").chat([])
    assert reply.tool_calls[0].arguments == {"x": 1}


def test_reasoning_content_is_captured_separately(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append(
        (200, _reply({"content": "answer", "reasoning_content": "chain of thought"}))
    )
    reply = ChatClient(url, "m").chat([])
    assert reply.content == "answer"
    assert reply.reasoning == "chain of thought"


def test_nested_error_shape_is_parsed(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append((404, {"error": {"message": "model 'x' not found"}}))
    with pytest.raises(LlmError, match="model 'x' not found") as excinfo:
        ChatClient(url, "x").chat([])
    assert "version prefix" in str(excinfo.value)  # 404 teaches the /v1 mistake


def test_vllm_top_level_error_shape_is_parsed(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append(
        (400, {"object": "error", "message": "something else entirely", "code": 400})
    )
    with pytest.raises(LlmError, match="something else entirely"):
        ChatClient(url, "m").chat([])


def test_context_overflow_400_raises_its_own_type(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append(
        (
            400,
            {
                "error": {
                    "message": "This model's maximum context length is 8192 tokens..."
                }
            },
        )
    )
    with pytest.raises(ContextOverflowError):
        ChatClient(url, "m").chat([])


def test_auth_error_teaches_api_key_env(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.responses.append((401, {"error": {"message": "invalid key"}}))
    with pytest.raises(LlmError, match="api_key_env"):
        ChatClient(url, "m").chat([])


def test_unreachable_endpoint_teaches_and_bounds_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("slab.mason.client.time.sleep", sleeps.append)
    client = ChatClient("http://127.0.0.1:9", "m", timeout_s=2)
    with pytest.raises(LlmError, match="is the model server running"):
        client.chat([])
    assert len(sleeps) == 2  # 3 attempts, 2 backoffs


def test_non_json_answer_is_a_loud_error(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    # A dict payload that json-encodes fine but lacks choices:
    script.responses.append((200, {"unexpected": True}))
    with pytest.raises(LlmError, match="no choices"):
        ChatClient(url, "m").chat([])


def test_model_names_lists_ids(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.get_response = (200, {"data": [{"id": "llama3.1:8b"}, {"id": "qwen3"}]})
    assert ChatClient(url, "m").model_names() == ["llama3.1:8b", "qwen3"]


def test_model_names_shape_guard(llm_server: tuple[str, LlmScript]) -> None:
    url, script = llm_server
    script.get_response = (200, {"models": []})
    with pytest.raises(LlmError, match="'data' list"):
        ChatClient(url, "m").model_names()


# -- fenced fallback protocol ------------------------------------------------


def test_fenced_calls_parse_and_report_malformed_blocks() -> None:
    content = (
        'thinking...\n```tool\n{"tool": "read_file", "arguments": {"path": "x"}}\n```\n'
        "and also\n```tool\nnot json\n```"
    )
    calls = parse_fenced_calls(content)
    assert calls[0].name == "read_file" and calls[0].arguments == {"path": "x"}
    assert calls[1].arguments_error is not None and "not valid JSON" in calls[1].arguments_error


def test_fenced_calls_require_the_documented_shape() -> None:
    calls = parse_fenced_calls('```tool\n{"name": "wrong_key"}\n```')
    assert calls[0].arguments_error is not None and '"tool"' in calls[0].arguments_error
    assert parse_fenced_calls("no blocks here") == ()
    assert parse_fenced_calls(None) == ()


def test_loose_calls_catch_llama_style_text_json() -> None:
    from slab.mason.client import parse_loose_calls

    known = frozenset({"slab_launch", "shell"})
    content = (
        "I will run it with slab_launch:\n\n"
        '{"name": "slab_launch", "parameters": {"script": "wf.py", "intent": "relax Cu"}}'
    )
    calls = parse_loose_calls(content, known)
    assert len(calls) == 1
    assert calls[0].name == "slab_launch"
    assert calls[0].arguments == {"script": "wf.py", "intent": "relax Cu"}


def test_loose_calls_ignore_prose_and_bad_shapes() -> None:
    from slab.mason.client import parse_loose_calls

    known = frozenset({"shell"})
    assert parse_loose_calls('the {"name": "field"} key is prose', known) == ()
    assert parse_loose_calls('{"name": "shell", "parameters": "ls"}', known) == ()
    assert parse_loose_calls('{"name": "shell", "parameters": {broken', known) == ()
    assert parse_loose_calls(None, known) == ()
    # The "arguments" spelling and the "tool" key work too:
    calls = parse_loose_calls('{"tool": "shell", "arguments": {"command": "ls"}}', known)
    assert calls[0].arguments == {"command": "ls"}


def test_loose_calls_never_execute_quoted_examples() -> None:
    from slab.mason.client import parse_loose_calls

    known = frozenset({"finish", "shell"})
    quoted = (
        "To end a task you would write:\n"
        "```json\n{\"name\": \"finish\", \"parameters\": {\"report\": \"...\"}}\n```\n"
        "and inline `{\"name\": \"shell\", \"parameters\": {\"command\": \"ls\"}}` too."
    )
    assert parse_loose_calls(quoted, known) == ()
    # But a bare (unquoted) call still parses:
    bare = 'Running now.\n{"name": "shell", "parameters": {"command": "ls"}}'
    assert parse_loose_calls(bare, known)[0].name == "shell"


def test_5xx_answers_are_retried_then_surfaced(
    llm_server: tuple[str, LlmScript], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, script = llm_server
    monkeypatch.setattr("slab.mason.client.time.sleep", lambda s: None)
    script.responses.append((503, {"error": {"message": "overloaded"}}))
    script.responses.append((503, {"error": {"message": "overloaded"}}))
    script.responses.append((200, _reply({"content": "recovered"})))
    reply = ChatClient(url, "m").chat([])
    assert reply.content == "recovered"
    assert len(script.requests) == 3


def test_loose_calls_surface_hallucinated_tool_names_for_teaching() -> None:
    """A well-shaped call naming an unknown tool reaches dispatch, whose
    'unknown tool' answer lists the real catalog — better than a dead end."""
    from slab.mason.client import parse_loose_calls

    known = frozenset({"shell"})
    calls = parse_loose_calls('{"name": "slab_run", "parameters": {"path": "x"}}', known)
    assert len(calls) == 1 and calls[0].name == "slab_run"
    # But a known-name call in the same text wins over the unknown one:
    both = parse_loose_calls(
        '{"name": "slab_run", "parameters": {}}\n{"name": "shell", "parameters": {}}', known
    )
    assert [call.name for call in both] == ["shell"]


def test_blackholed_connect_fails_fast_instead_of_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """urllib wraps connect-phase timeouts in URLError; treating that as a
    retryable connection failure would burn attempts x timeout in silence
    on a firewalled endpoint."""
    import urllib.error
    import urllib.request

    attempts: list[int] = []

    def blackholed(request: object, timeout: float = 0) -> object:
        attempts.append(1)
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", blackholed)
    client = ChatClient("http://gpu-9:8000/v1", "m", timeout_s=5.0)
    with pytest.raises(LlmError, match="timed out"):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 1  # no retries against a blackhole
