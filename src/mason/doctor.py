"""The endpoint probes behind ``slab mason doctor``.

The CLI moved its probe body here so the whole-stack ``slab doctor`` can
run the same checks without a terminal: every probe emits its lines
through a callback and returns a failure count, and only the discovery
step raises — an unreachable endpoint is a finding, not an exception.
The emitted lines are the doctor's output, byte for byte.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mason.config import AgentConfig

from foundation.errors import FoundationError
from mason.errors import MasonError
from slab.errors import SlabError

Emit = Callable[[str], None]


def serve_hint(agent: AgentConfig, root: Path, origin: str, *, cluster: str = "") -> list[str]:
    """Why an unreachable endpoint might be unreachable, when we can tell."""
    from mason.serve import read_record
    from slab.hpc import job_state

    if agent.provider != "openai":
        return []
    try:
        record = read_record(root)
    except (MasonError, FoundationError, SlabError) as e:
        return [f"    (the endpoint record is unreadable: {e})"]
    if record is None:
        # A one-shot --endpoint meant that server, so 'start your own' is noise.
        # A *configured* endpoint that no longer answers is the opposite case:
        # it is usually last allocation's node, and serve start is the fix.
        if origin == "--endpoint":
            return []
        return [
            "    no server is recorded here; start one with 'slab mason serve start' "
            "(or point [agent] endpoint at a server you started yourself)"
        ]
    if not record.job_id:
        return [f"    a record exists ({record.endpoint}) but names no job to ask about"]
    if record.cluster and record.cluster != cluster:
        # A job id is only meaningful on its own cluster; asking this one
        # would describe an unrelated job that happens to share the number.
        return [
            f"    the record belongs to cluster {record.cluster!r}; job "
            f"{record.job_id} is not queried from here (job ids are per-cluster)"
        ]
    try:
        status = job_state(record.job_id)
    except (MasonError, FoundationError, SlabError) as e:
        return [f"    job {record.job_id}: state unknown — {e}"]
    if status.state.is_terminal:
        return [
            f"    job {record.job_id} ended as {status.state.value}; the record is "
            f"stale — 'slab mason serve stop' clears it"
        ]
    return [f"    job {record.job_id} is {status.state.value}; the model may still be loading"]


def _probe_key(agent: AgentConfig, *, label: str) -> tuple[str | None, str | None]:
    """The API key a probe must send, or the failure line naming the gap.

    The probe has to authenticate exactly as the session will. Sending
    nothing where the config names a key turns a working connection into a
    401 whose message tells you to configure what you already configured.
    """
    key_var = agent.resolved_api_key_env
    if key_var is None:
        return None, None
    api_key = os.environ.get(key_var)
    if not api_key:
        return None, f"[x] {label}${key_var} is not set — [agent] api_key_env names it"
    return api_key, None


def _client_for(agent: AgentConfig, endpoint: str, model: str, api_key: str | None) -> Any:
    from mason.client import ChatClient

    if agent.provider == "anthropic":
        from mason.anthropic import AnthropicClient

        assert api_key is not None  # resolved_api_key_env always names one here
        return AnthropicClient(model, api_key, endpoint=endpoint, timeout_s=60.0)
    return ChatClient(endpoint, model, api_key=api_key, timeout_s=60.0)


def run(
    agent: AgentConfig,
    root: Path,
    *,
    cluster: str,
    model: str | None = None,
    endpoint_forced: bool = False,
    emit: Emit,
) -> int:
    """The full endpoint report: discovery, probes, roster.

    Returns the failure count (0 = healthy). Raises the MasonError family
    only when endpoint discovery itself fails; every probe result — an
    unreachable endpoint included — is emitted and counted instead.
    """
    from mason.client import LlmError
    from mason.serve import discover_endpoint

    resolved_endpoint, origin = discover_endpoint(agent, root)
    if endpoint_forced:
        origin = "--endpoint"
    resolved_model = model or agent.model
    emit(f"provider: {agent.provider}")
    emit(f"endpoint: {resolved_endpoint}  [{origin}]")
    emit(f"model:    {resolved_model or '(not configured)'}")
    api_key, missing = _probe_key(agent, label="")
    if missing is not None:
        emit(missing)
        return 1
    client = _client_for(agent, resolved_endpoint, resolved_model or "unconfigured", api_key)
    failed = 0
    try:
        names = client.model_names()
        emit(f"[+] endpoint answers; {len(names)} model(s) served")
    except LlmError as e:
        emit(f"[x] endpoint: {e}")
        for line in serve_hint(agent, root, origin, cluster=cluster):
            emit(line)
        return 1
    if resolved_model is None:
        emit(f"[x] no model configured; served here: {', '.join(names) or 'none'}")
        return 1
    if resolved_model in names:
        emit(f"[+] model {resolved_model!r} is served")
    else:
        failed += 1
        emit(f"[x] model {resolved_model!r} not served; available: {', '.join(names)}")
    ping = {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Reply with a pong.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    try:
        reply = client.chat(
            [{"role": "user", "content": "Call the ping tool now."}], tools=[ping]
        )
        if reply.tool_calls:
            emit("[+] native tool calls work")
        else:
            failed += 1
            emit(
                "[x] the model answered without a tool call — the server may lack a "
                "tool-call parser; try [agent] tool_protocol = \"fenced\""
            )
    except LlmError as e:
        failed += 1
        emit(f"[x] tool-call probe: {e}")
    failed += _context_probe(client, emit=emit)
    primary = (agent.provider, resolved_endpoint, resolved_model)
    failed += _roster(agent, root, seen={primary}, emit=emit)
    return failed


#: The context probe sends this many filler words: about 6k tokens, which
#: is the size of Mason's fixed prefix (system prompt plus tool schemas)
#: on a laptop configuration, and beyond Ollama's default num_ctx.
_CONTEXT_PROBE_WORDS = 6_000


def _context_probe(client: Any, *, emit: Emit) -> int:
    """Send a prompt the size of Mason's own prefix and check the server saw it.

    Ollama truncates every prompt to its ``num_ctx`` (2048 or 4096 by
    default) and says nothing; the only trace is a ``usage.prompt_tokens``
    far below what was sent. A model served that way never sees Mason's
    instructions, and every session looks like a weak model when it is a
    blind one. Returns 1 when the server truncates, else 0.
    """
    from mason.client import LlmError

    filler = "word " * _CONTEXT_PROBE_WORDS
    try:
        reply = client.chat(
            [
                {"role": "system", "content": filler},
                {"role": "user", "content": "Reply with the single word ok."},
            ],
            None,
        )
    except LlmError as e:
        emit(f"[?] context probe: {e}")
        return 0
    counted = reply.prompt_tokens
    if counted is None:
        emit("[?] context probe: the server reports no usage, so truncation cannot be ruled out")
        return 0
    if counted * 2 < _CONTEXT_PROBE_WORDS:
        emit(
            f"[x] the server counted {counted} prompt tokens for a {_CONTEXT_PROBE_WORDS}-word "
            f"prompt: it truncates the context, and Mason's own prefix is larger than "
            f"that. Serve the model with a larger context: for Ollama, a Modelfile with "
            f"'PARAMETER num_ctx 32768' ('ollama create <name>-32k -f Modelfile'), or "
            f"OLLAMA_CONTEXT_LENGTH=32768 before starting the server"
        )
        return 1
    emit(f"[+] a {_CONTEXT_PROBE_WORDS}-word prompt arrives intact ({counted} tokens counted)")
    return 0



def effort_notes(agent: AgentConfig, roster: dict[str, Any]) -> list[str]:
    """Where ``effort`` is set but does not reach the agents that run.

    Two traps from one real campaign. ``[agent.roster.<name>]`` tables set
    the dial for the planner and the worker while ``[agent]`` leaves it
    unset, so the PI and the critic, which have no table, run at the
    server's default: on a hosted endpoint that default is the top of the
    scale. And a table for a card the PI never delegates to (a lead, such
    as the planner) applies only when a run starts with ``--agent`` for
    that card. Notes, not failures: the config is valid, just not doing
    what it looks like it does.
    """
    from mason.roster import critics, hands

    notes: list[str] = []
    tables = agent.roster or {}
    dialed = sorted(name for name, table in tables.items() if table.effort is not None)
    if dialed and agent.effort is None:
        notes.append(
            f"[?] effort: [agent.roster.{'/'.join(dialed)}] set effort but [agent] does "
            f"not, so every card without a table (pi, the critic) runs at the server's "
            f"default; set [agent] effort"
        )
    lead = roster.get("pi")
    if lead is not None:
        reachable = {"pi", *hands(lead, roster), *critics(roster)}
        for name in sorted(tables):
            if name in roster and name not in reachable:
                notes.append(
                    f"[?] {name}: pi never delegates to it, so its [agent.roster.{name}] "
                    f"table applies only when a run starts with --agent {name}"
                )
    return notes

def _roster(
    agent: AgentConfig,
    root: Path,
    *,
    seen: set[tuple[str, str, str | None]],
    emit: Emit,
) -> int:
    """Probe the roster's distinct model connections; return the failure count.

    A specialist pinned to an unserved model should fail the doctor, not
    the first delegation. Only ``[agent.roster.<name>]`` tables are probed —
    an agent without a table shares the primary connection checked above.
    """
    if not agent.roster:
        return 0
    from mason.client import LlmError
    from mason.config import roster_agent_config
    from mason.roster import check_overrides, discover_roster
    from mason.serve import discover_endpoint

    try:
        roster = discover_roster(Path.cwd())
        check_overrides(agent, roster)
    except (MasonError, FoundationError, SlabError) as e:
        emit(f"[x] roster: {e}")
        return 1
    for note in effort_notes(agent, roster):
        emit(note)
    failures = 0
    for name in sorted(agent.roster):
        effective = roster_agent_config(agent, name)
        try:
            endpoint, _origin = discover_endpoint(effective, root)
        except (MasonError, FoundationError, SlabError) as e:
            emit(f"[x] {name}: {e}")
            failures += 1
            continue
        key = (effective.provider, endpoint, effective.model)
        if key in seen:
            continue
        seen.add(key)
        if effective.model is None:
            emit(f"[x] {name}: no model configured for its connection")
            failures += 1
            continue
        api_key, missing = _probe_key(effective, label=f"{name}: ")
        if missing is not None:
            emit(missing)
            failures += 1
            continue
        client = _client_for(effective, endpoint, effective.model, api_key)
        try:
            names = client.model_names()
        except LlmError as e:
            emit(f"[x] {name}: endpoint {endpoint}: {e}")
            failures += 1
            continue
        if effective.model in names:
            emit(f"[+] {name}: model {effective.model!r} is served at {endpoint}")
        else:
            emit(
                f"[x] {name}: model {effective.model!r} not served at {endpoint}; "
                f"available: {', '.join(names) or 'none'}"
            )
            failures += 1
    return failures
