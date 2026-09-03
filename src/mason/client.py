"""A stdlib OpenAI-compatible chat client: one client for vLLM, Ollama, llama.cpp.

Every serious open-model server speaks the OpenAI chat-completions surface,
so Mason targets exactly that and nothing else: ``POST /chat/completions``
and ``GET /models`` under a configurable base URL, plain ``urllib``, no SDK
dependency. The deliberate omissions follow the least common denominator of
the servers we support: no ``tool_choice`` (Ollama ignores it), no ``n``,
no ``logit_bias``, no streaming (an agent loop consumes whole turns).

The response contract is defensive where open-model serving is loose:

* ``tool_calls[].function.arguments`` is a JSON *string*; when it does not
  parse, the call is kept with ``arguments_error`` set so the loop can hand
  the model a repair prompt instead of crashing.
* Missing tool-call ids are synthesized (some servers omit them).
* ``reasoning_content`` (vLLM reasoning parsers) is captured separately and
  never echoed back into history.
* Both error body shapes are parsed — nested ``{"error": {...}}``
  (OpenAI/Ollama) and top-level ``{"object": "error", ...}`` (vLLM) — and a
  400 whose message points at the context window raises
  :class:`ContextOverflowError` so the loop can compact and retry.
"""

from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mason.errors import MasonError

_RETRIES = 3
_BACKOFF_S = (1.0, 4.0)  # between attempts 1->2 and 2->3
_CONTEXT_OVERFLOW_HINTS = (
    "context length",
    "context window",
    "maximum context",
    "context size",  # llama.cpp's wording
    "prompt is too long",  # Anthropic's wording
    "too many tokens",
)


class LlmError(MasonError):
    """The model endpoint refused, misbehaved, or could not be reached."""


class ContextOverflowError(LlmError):
    """The request exceeded the model's context window (compact and retry)."""


class ToolCall(BaseModel):
    """One tool invocation the model asked for.

    ``arguments_raw`` preserves the exact string for echoing back into
    history; ``arguments_error`` is set (and ``arguments`` empty) when that
    string was not valid JSON — the loop reports it as the tool outcome so
    the model can repair itself.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_raw: str = ""
    arguments_error: str | None = None


class ChatReply(BaseModel):
    """One assistant turn, plus the token accounting the server reported."""

    model_config = ConfigDict(frozen=True)

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: The part of ``prompt_tokens`` the server read from its prompt cache,
    #: when it says (OpenAI-style ``prompt_tokens_details.cached_tokens``).
    #: Billed at a fraction of the rest, so a cost reading needs it.
    cached_prompt_tokens: int | None = None


#: ``[agent] effort`` values to the OpenAI ``reasoning_effort`` field. Every
#: value is the field's own name except ``max``, an Anthropic word, which
#: goes out as ``xhigh``, the field's top.
_REASONING_EFFORT = {"max": "xhigh"}


class ChatClient:
    """Blocking chat-completions client for any OpenAI-compatible ``/v1``.

    Args:
        endpoint: Base URL including the version prefix, e.g.
            ``http://localhost:11434/v1`` (Ollama) or
            ``http://gpu-node:8000/v1`` (vLLM).
        model: Model name as the server knows it.
        api_key: Sent as a Bearer token; vLLM deployments may require one,
            Ollama ignores it. Defaults to ``"slab"`` because the header
            must always be present.
        temperature: Sampling temperature for every request.
        timeout_s: Per-request timeout — generous by default, HPC inference
            can be slow.
        max_reply_tokens: ``max_tokens`` per turn, or None to let the
            server default.
        effort: ``none``/``low``/``medium``/``high``/``xhigh``/``max``, sent
            verbatim as the OpenAI ``reasoning_effort`` field (``max`` as
            ``xhigh``, the field's top). None sends no field, and the server
            then applies its own default. Servers differ in which values
            they accept: OpenAI reasoning models take the whole scale, vLLM
            and llama.cpp take what the model's chat template has a dial
            for, and a hosted Qwen3.8 endpoint took ``none``, ``low``, and
            ``medium`` only, with ``xhigh`` as the unset default. A server
            that does not know the field ignores it, and a thinking model
            then thinks as long as it likes. The transcript's usage lines
            tell which case a server is.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 600.0,
        max_reply_tokens: int | None = None,
        effort: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or "slab"
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_reply_tokens = max_reply_tokens
        self.effort = effort

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> ChatReply:
        """One chat-completions round trip; retries transport-level failures.

        Connection failures and 5xx answers are retried up to 3 attempts
        with backoff; HTTP 4xx are the server telling us something and are
        never retried (a context-window 400 raises
        :class:`ContextOverflowError` instead).

        *effort* and *max_tokens* override the instance's settings for this
        one call: the compaction summarizer is a model call that should
        think little and answer short, whatever the session's dial says.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
        reply_tokens = self.max_reply_tokens if max_tokens is None else max_tokens
        if reply_tokens is not None:
            body["max_tokens"] = reply_tokens
        wanted = self.effort if effort is None else effort
        if wanted is not None:
            body["reasoning_effort"] = _REASONING_EFFORT.get(wanted, wanted)
        payload = self._request("/chat/completions", body)
        return _parse_reply(payload)

    def model_names(self) -> list[str]:
        """The ids ``GET /models`` reports — the doctor's first question."""
        payload = self._request("/models", None)
        data = payload.get("data")
        if not isinstance(data, list):
            raise LlmError(f"{self.endpoint}/models answered without a 'data' list")
        return [str(item.get("id")) for item in data if isinstance(item, dict)]

    def _request(self, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        return request_json(
            f"{self.endpoint}{path}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            body=body,
            timeout_s=self.timeout_s,
            unreachable_hint=(
                "is the model server running? "
                "(vLLM: 'vllm serve MODEL'; Ollama: 'ollama serve')"
            ),
        )


def request_json(
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any] | None,
    timeout_s: float,
    unreachable_hint: str,
) -> dict[str, Any]:
    """One JSON round trip with the retry discipline both providers share.

    GET when *body* is None, POST otherwise. Connection failures and 5xx
    answers are retried up to 3 attempts with backoff; 4xx answers are the
    server telling us something and are never retried.
    """
    data = None if body is None else json.dumps(body).encode()
    last_error: LlmError | None = None
    for attempt in range(_RETRIES):
        request = urllib.request.Request(
            url, data=data, headers=headers, method="GET" if body is None else "POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return _decode_json(response.read(), url)
        except urllib.error.HTTPError as e:
            parsed = _http_error(e, url)
            if e.code < 500:
                raise parsed from e  # a 4xx is the server's answer, not weather
            last_error = parsed  # 5xx: retry as promised
        except TimeoutError as e:
            raise LlmError(
                f"no answer from {url} within {timeout_s:.0f}s — slow inference "
                f"is normal on HPC; raise [agent] request_timeout_s if this recurs"
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                # urllib wraps connect-phase timeouts in URLError, so a
                # blackholed endpoint (packet-dropping firewall) would land
                # in the retry path and burn attempts x timeout in silence.
                # A timeout is a timeout: fail fast, like the branch above.
                raise LlmError(
                    f"no answer from {url} within {timeout_s:.0f}s (the connection "
                    f"attempt itself timed out — firewalled endpoint or wrong "
                    f"node?) — {unreachable_hint}"
                ) from e
            last_error = LlmError(f"cannot reach {url}: {e.reason} — {unreachable_hint}")
        except (OSError, http.client.HTTPException) as e:
            # A connection dying mid-request/mid-read surfaces raw
            # (ConnectionResetError, RemoteDisconnected, IncompleteRead);
            # wrap and retry like any transport failure.
            last_error = LlmError(f"connection to {url} failed mid-request: {e!r}")
        if attempt < _RETRIES - 1:
            time.sleep(_BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)])
    assert last_error is not None
    raise last_error


def _decode_json(raw: bytes, url: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        head = raw[:200].decode(errors="replace")
        raise LlmError(f"{url} answered non-JSON ({e}): {head!r}") from e
    if not isinstance(payload, dict):
        raise LlmError(f"{url} answered JSON but not an object: {type(payload).__name__}")
    return payload


def _http_error(error: urllib.error.HTTPError, url: str) -> LlmError:
    """Turn an HTTP error into the most specific :class:`LlmError` we can."""
    try:
        raw = error.read()
    except Exception:  # pragma: no cover - defensive: unreadable error body
        raw = b""
    message = _error_message(raw) or raw[:500].decode(errors="replace") or "no error body"
    if error.code == 400 and any(hint in message.lower() for hint in _CONTEXT_OVERFLOW_HINTS):
        return ContextOverflowError(f"{url}: {message}")
    if error.code == 429:
        return LlmError(f"{url} rate-limited this request (429): {message}")
    if error.code in (401, 403):
        return LlmError(
            f"{url} refused authentication ({error.code}): {message} — set [agent] "
            f"api_key_env to the name of the environment variable holding your key"
        )
    if error.code == 404:
        return LlmError(
            f"{url} not found (404): {message} — the endpoint should include the "
            f"version prefix, e.g. http://localhost:11434/v1"
        )
    return LlmError(f"{url} answered {error.code}: {message}")


def _error_message(raw: bytes) -> str | None:
    """Both error shapes: nested ``{"error": {...}}`` and vLLM's top-level."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str) and error:
        return error
    if payload.get("message"):  # vLLM: {"object": "error", "message": ...}
        return str(payload["message"])
    return None


def _finish_reason(raw: object) -> str | None:
    """The server's finish reason in the loop's one vocabulary.

    OpenAI-compatible servers say ``length`` where the Messages API says
    ``max_tokens``; the loop tests for the latter, so a truncated reply on
    vLLM or the gateway was passed off as a finished answer before this.

    Examples:
        >>> _finish_reason("length")
        'max_tokens'
        >>> _finish_reason("stop")
        'stop'
        >>> _finish_reason(None) is None
        True
    """
    if raw is None:
        return None
    word = str(raw)
    return "max_tokens" if word == "length" else word


def _parse_reply(payload: dict[str, Any]) -> ChatReply:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmError(f"chat answer has no choices: {json.dumps(payload)[:300]}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LlmError("chat answer's first choice has no message object")
    usage_raw = payload.get("usage")
    usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return ChatReply(
        content=None if content is None else str(content),
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
        reasoning=None if reasoning is None else str(reasoning),
        finish_reason=_finish_reason(choices[0].get("finish_reason")),
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(usage.get("completion_tokens")),
        cached_prompt_tokens=_cached_prompt_tokens(usage),
    )


def _cached_prompt_tokens(usage: dict[str, Any]) -> int | None:
    """``prompt_tokens_details.cached_tokens`` when the server reports it.

    Examples:
        >>> _cached_prompt_tokens({"prompt_tokens_details": {"cached_tokens": 900}})
        900
        >>> _cached_prompt_tokens({"prompt_tokens": 10}) is None
        True
    """
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    return _int_or_none(details.get("cached_tokens"))


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _parse_tool_calls(raw: object) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "")
        if not name:
            continue
        arguments_raw = function.get("arguments")
        if isinstance(arguments_raw, dict):  # some servers pre-parse; normalize
            arguments_raw = json.dumps(arguments_raw)
        arguments_raw = str(arguments_raw or "{}")
        arguments: dict[str, Any] = {}
        arguments_error: str | None = None
        try:
            parsed = json.loads(arguments_raw)
            if isinstance(parsed, dict):
                arguments = parsed
            else:
                arguments_error = f"arguments must be a JSON object, got {type(parsed).__name__}"
        except json.JSONDecodeError as e:
            arguments_error = f"arguments were not valid JSON: {e}"
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{position}"),
                name=name,
                arguments=arguments,
                arguments_raw=arguments_raw,
                arguments_error=arguments_error,
            )
        )
    return tuple(calls)


_FENCED_ACTION = re.compile(r"```tool\s*\n(.*?)```", re.DOTALL)
_LOOSE_CALL_START = re.compile(r'\{\s*"(?:name|tool)"\s*:')
# Quoted material never executes: fenced code blocks (any language tag except
# our own ```tool protocol) and inline `code` spans are how a model QUOTES a
# call shape rather than making one.
_QUOTED_REGIONS = re.compile(r"```(?!tool\b).*?```|`[^`\n]*`", re.DOTALL)


def parse_loose_calls(content: str | None, known_names: frozenset[str]) -> tuple[ToolCall, ...]:
    """Tool calls a model wrote as plain-text JSON instead of native calls.

    Llama-family models served without a tool parser (and sometimes with one)
    emit ``{"name": ..., "parameters": {...}}`` in the message text — the
    llama3 JSON tool format leaking into content. Only a well-shaped object
    (a string name plus an *object* of arguments) counts; anything else is
    prose. Known tool names win; when the text only names unknown tools the
    calls are returned anyway, so dispatch can answer with the tool catalog
    instead of the turn dead-ending on a hallucinated name. This is the
    harness-side repair path the open-weight literature recommends over
    constrained decoding.

    Examples:
        >>> calls = parse_loose_calls(
        ...     'I will run it:\\n{"name": "shell", "parameters": {"command": "ls"}}',
        ...     frozenset({"shell"}),
        ... )
        >>> (calls[0].name, calls[0].arguments)
        ('shell', {'command': 'ls'})
        >>> parse_loose_calls('the {"name": "field"} key is prose', frozenset({"shell"}))
        ()
    """
    if not content:
        return ()
    content = _QUOTED_REGIONS.sub(" ", content)
    decoder = json.JSONDecoder()
    known: list[tuple[str, dict[str, Any]]] = []
    unknown: list[tuple[str, dict[str, Any]]] = []
    for match in _LOOSE_CALL_START.finditer(content):
        try:
            parsed, _end = decoder.raw_decode(content, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name") or parsed.get("tool")
        arguments = parsed.get("parameters", parsed.get("arguments"))
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            continue
        (known if name in known_names else unknown).append((name, arguments))
    chosen = known if known else unknown
    return tuple(
        ToolCall(
            id=f"loose_{position}",
            name=name,
            arguments=arguments,
            arguments_raw=json.dumps(arguments),
        )
        for position, (name, arguments) in enumerate(chosen)
    )


def parse_fenced_calls(content: str | None) -> tuple[ToolCall, ...]:
    """Tool calls written as fenced ````tool`` blocks in plain text.

    The fallback protocol for servers without a tool-call parser
    (mini-swe-agent's lesson: a text protocol is the great equalizer for
    open-weight models). Each block holds one JSON object
    ``{"tool": name, "arguments": {...}}``; malformed blocks become calls
    with ``arguments_error`` set so the loop can teach instead of ignore.

    Examples:
        >>> calls = parse_fenced_calls('do it\\n```tool\\n{"tool": "ls", "arguments": {}}\\n```')
        >>> (calls[0].name, calls[0].arguments)
        ('ls', {})
    """
    if not content:
        return ()
    calls: list[ToolCall] = []
    for position, block in enumerate(_FENCED_ACTION.findall(content)):
        text = block.strip()
        name = ""
        arguments: dict[str, Any] = {}
        error: str | None = None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
                name = parsed["tool"]
                if isinstance(parsed.get("arguments"), dict):
                    arguments = parsed["arguments"]
                elif "arguments" in parsed:
                    error = "\"arguments\" must be a JSON object"
            else:
                error = 'the block must be {"tool": "<name>", "arguments": {...}}'
        except json.JSONDecodeError as e:
            error = f"the block was not valid JSON: {e}"
        calls.append(
            ToolCall(
                id=f"fenced_{position}",
                name=name or "unknown",
                arguments=arguments,
                arguments_raw=text,
                arguments_error=error,
            )
        )
    return tuple(calls)
