"""The Anthropic Messages API as a Mason backend: Claude for the thinking.

Mason's loop is provider-agnostic — it needs a :class:`~slab.mason.loop.ChatBackend`,
one method, OpenAI-shaped messages in and a :class:`~slab.mason.client.ChatReply`
out. This module implements that against ``POST /v1/messages``, so the same
harness — same tools, same notebook, same verification gates, same provenance —
runs on Claude instead of a locally served open model. The open-model path stays
first-class: compute nodes are often firewalled off the internet, and the GPUs
you already own are free at the margin.

The Messages API is *not* OpenAI-shaped, so this module owns the translation:

* **System messages become a top-level ``system`` parameter**, not a turn. The
  last system block carries ``cache_control``, so Mason's stable prompt prefix
  (identity, tool guidance, environment) is served from Anthropic's prompt cache
  turn after turn — the same prefix-stability that pays off on a vLLM server.
* **Tool calls are content blocks**: an assistant turn carries ``tool_use``
  blocks, and results come back as ``tool_result`` blocks inside a *user* turn.
  All results answering one assistant turn must ride in a **single** user
  message — splitting them teaches the model to stop calling tools in parallel.
* **No sampling parameters are ever sent.** ``temperature``, ``top_p``, and
  ``top_k`` are rejected outright by current Claude models (HTTP 400); steering
  is prompting's job here. ``[agent] temperature`` therefore applies only to the
  OpenAI-compatible provider, and ``[agent] effort`` is the equivalent knob.
* **``max_tokens`` is required and bounds thinking plus reply together.** Claude
  models think by default, so the ceiling is generous; a truncated turn is
  reported rather than passed off as a finished answer.
* **A refusal is a successful HTTP 200** with ``stop_reason: "refusal"`` and a
  possibly empty content list. Reading ``content[0]`` blindly would turn that
  into a silent empty answer, so it is raised as a typed error instead.
"""

from __future__ import annotations

import json
from typing import Any

from slab.mason.client import ChatReply, LlmError, ToolCall, request_json

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-opus-5"
# Claude thinks before answering, and max_tokens caps thinking + reply together;
# a laptop-sized default would truncate mid-answer on real agentic turns.
DEFAULT_MAX_TOKENS = 16_000
_RESUMED = "(conversation resumed)"


class ModelRefusalError(LlmError):
    """Claude's safety classifiers declined the request (``stop_reason: refusal``)."""


class AnthropicClient:
    """Chat backend speaking the Anthropic Messages API.

    Args:
        model: A Claude model id, e.g. ``claude-opus-5``.
        api_key: The API key. Required — Anthropic has no anonymous access.
        endpoint: Base URL including the version prefix (override for a proxy).
        max_reply_tokens: ``max_tokens`` per turn; bounds thinking *and* reply.
        effort: ``low``/``medium``/``high``/``xhigh``/``max`` — the depth and
            spend control that replaces sampling parameters here. None leaves
            the server default.
        timeout_s: Per-request timeout.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        endpoint: str = ANTHROPIC_ENDPOINT,
        max_reply_tokens: int | None = None,
        effort: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_reply_tokens = max_reply_tokens or DEFAULT_MAX_TOKENS
        self.effort = effort
        self.timeout_s = timeout_s

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatReply:
        """One Messages round trip, translated in both directions."""
        system, turns = translate_messages(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_reply_tokens,
            "messages": turns,
        }
        if system:
            # cache_control on the last system block caches tools + system:
            # they render before the conversation, and Mason keeps both stable
            # within a session.
            body["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if tools:
            body["tools"] = [_tool_spec(tool) for tool in tools]
        if self.effort is not None:
            body["output_config"] = {"effort": self.effort}
        payload = request_json(
            f"{self.endpoint}/messages",
            headers=self._headers,
            body=body,
            timeout_s=self.timeout_s,
            unreachable_hint="check network access to api.anthropic.com "
            "(HPC compute nodes are often firewalled — the open-model path works there)",
        )
        return parse_reply(payload)

    def model_names(self) -> list[str]:
        """Model ids ``GET /v1/models`` reports."""
        payload = request_json(
            f"{self.endpoint}/models",
            headers=self._headers,
            body=None,
            timeout_s=self.timeout_s,
            unreachable_hint="check network access to api.anthropic.com",
        )
        data = payload.get("data")
        if not isinstance(data, list):
            raise LlmError(f"{self.endpoint}/models answered without a 'data' list")
        return [str(item.get("id")) for item in data if isinstance(item, dict)]


def _tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
    """OpenAI ``{type, function:{...}}`` -> Anthropic ``{name, description, input_schema}``."""
    raw = tool.get("function")
    function: dict[str, Any] = raw if isinstance(raw, dict) else tool
    return {
        "name": function.get("name"),
        "description": function.get("description", ""),
        "input_schema": function.get("parameters")
        or {"type": "object", "properties": {}, "required": []},
    }


def translate_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI-shaped history -> ``(system_text, anthropic_messages)``.

    Four rules the Messages API enforces and this translation honors: system
    content is a parameter rather than a turn; tool results are ``tool_result``
    blocks in a *user* turn; every result answering one assistant turn rides in
    a single user message; and the first turn must be a user turn.

    Examples:
        >>> system, turns = translate_messages([
        ...     {"role": "system", "content": "be careful"},
        ...     {"role": "user", "content": "read a.py"},
        ...     {"role": "assistant", "content": None, "tool_calls": [
        ...         {"id": "c1", "type": "function",
        ...          "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        ...     {"role": "tool", "tool_call_id": "c1", "content": "x = 1"},
        ... ])
        >>> system
        'be careful'
        >>> turns[1]["content"][0]["type"], turns[1]["content"][0]["name"]
        ('tool_use', 'read_file')
        >>> turns[2]["role"], turns[2]["content"][0]["type"]
        ('user', 'tool_result')
    """
    system_parts: list[str] = []
    turns: list[tuple[str, list[dict[str, Any]]]] = []
    pending_results: list[dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            turns.append(("user", list(pending_results)))
            pending_results.clear()

    for message in messages:
        role = str(message.get("role", ""))
        if role == "system":
            text = _text_of(message.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id") or ""),
                    "content": _text_of(message.get("content")) or "(no output)",
                }
            )
            continue
        flush_results()
        if role == "assistant":
            blocks = _assistant_blocks(message)
            if blocks:  # a turn with no content at all is rejected
                turns.append(("assistant", blocks))
        else:
            text = _text_of(message.get("content"))
            if text:
                turns.append(("user", [{"type": "text", "text": text}]))
    flush_results()

    merged: list[dict[str, Any]] = []
    for role, blocks in turns:
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"].extend(blocks)
        else:
            merged.append({"role": role, "content": list(blocks)})
    for turn in merged:
        if turn["role"] == "user":
            # tool_result blocks must lead their user turn; merging adjacent
            # turns can otherwise put a text block in front of them.
            turn["content"].sort(key=lambda block: block.get("type") != "tool_result")
    if merged and merged[0]["role"] != "user":
        # The conversation must open on a user turn. Prepending beats dropping:
        # an assistant turn carrying tool_use whose tool_result follows would
        # otherwise leave that result orphaned, which the API rejects.
        merged.insert(0, {"role": "user", "content": [{"type": "text", "text": _RESUMED}]})
    return "\n\n".join(system_parts), merged


def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    text = _text_of(message.get("content"))
    if text:
        blocks.append({"type": "text", "text": text})
    raw_calls = message.get("tool_calls")
    for position, call in enumerate(raw_calls if isinstance(raw_calls, list) else []):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or f"call_{position}"),
                "name": str(function["name"]),
                "input": arguments if isinstance(arguments, dict) else {},
            }
        )
    return blocks


def _text_of(content: object) -> str:
    """Message content as plain text (list-of-blocks content is flattened)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return str(content)


def parse_reply(payload: dict[str, Any]) -> ChatReply:
    """Messages response -> :class:`ChatReply`, refusals raised, usage summed.

    ``usage.input_tokens`` counts only the *uncached* remainder, so the cache
    fields are added back: the loop's compaction trigger cares how big the
    prompt is, not how much of it was billed at full rate.
    """
    stop_reason = payload.get("stop_reason")
    if stop_reason == "refusal":
        details = payload.get("stop_details")
        category = details.get("category") if isinstance(details, dict) else None
        raise ModelRefusalError(
            "Claude declined this request"
            + (f" (category: {category})" if category else "")
            + " — safety classifiers can fire on benign security or life-science "
            "work; rephrase, or run this goal on the open-model provider"
        )
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise LlmError(f"messages answer has no content list: {json.dumps(payload)[:300]}")

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    calls: list[ToolCall] = []
    for position, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(str(block.get("text") or ""))
        elif kind == "thinking":
            thinking_parts.append(str(block.get("thinking") or ""))
        elif kind == "tool_use":
            arguments = block.get("input")
            arguments = arguments if isinstance(arguments, dict) else {}
            calls.append(
                ToolCall(
                    id=str(block.get("id") or f"toolu_{position}"),
                    name=str(block.get("name") or ""),
                    arguments=arguments,
                    arguments_raw=json.dumps(arguments),
                )
            )
    usage_raw = payload.get("usage")
    usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
    content = "\n".join(part for part in text_parts if part)
    return ChatReply(
        content=content or None,
        tool_calls=tuple(call for call in calls if call.name),
        reasoning="\n".join(part for part in thinking_parts if part) or None,
        finish_reason=None if stop_reason is None else str(stop_reason),
        prompt_tokens=_prompt_tokens(usage),
        completion_tokens=_int_or_none(usage.get("output_tokens")),
    )


def _prompt_tokens(usage: dict[str, Any]) -> int | None:
    """Total prompt size: uncached input plus everything served from cache."""
    parts = [
        _int_or_none(usage.get(key))
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    ]
    known = [part for part in parts if part is not None]
    return sum(known) if known else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None
