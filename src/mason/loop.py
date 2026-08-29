"""The turn engine: one ReAct loop, budgeted context, harness-owned discipline.

One :class:`Mason` holds one conversation. ``run_turn`` takes a user goal and
loops model call -> tool dispatch -> observation until the model answers in
plain text, calls ``finish``, or a harness limit stops it. The limits live in
code because prompts do not enforce invariants:

* ``max_turns`` model calls per goal — the runaway stop.
* A consecutive-tool-failure streak aborts the turn with the evidence in
  place (five malformed or crashing calls in a row is a stuck agent, not
  progress).
* Compaction triggers at ``compact_at`` x ``context_window`` — well below
  the window, because model quality degrades before the hard limit — and a
  context-overflow answer from the server forces one immediate compaction
  and retry. Compaction folds the middle of the conversation into a
  structured summary (state, verified results, failures observed, open
  questions), writes that summary into the lab notebook, and rebuilds the
  system message fresh so plan and notebook re-enter the context updated.

Both tool protocols run through the same loop: native OpenAI tool calls, or
the fenced-block text protocol for servers without a tool-call parser.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from mason.client import (
    ChatClient,
    ChatReply,
    ContextOverflowError,
    parse_fenced_calls,
    parse_loose_calls,
)
from mason.config import AgentConfig, override_agent, roster_agent_config
from mason.errors import MasonError
from mason.prompts import COMPACTION_PROMPT, system_messages, team_block
from mason.roster import AgentSpec, check_overrides, discover_roster, skills_for
from mason.session import MasonSession
from mason.skills import Skill, discover_skills
from mason.tools import Toolbox, build_toolbox

_ERROR_STREAK_LIMIT = 5
_COMPACTION_KEEP_MESSAGES = 6
_CHARS_PER_TOKEN = 4  # the usual rough estimate; real usage numbers override it
#: Ratio at which the budget hint escalates from a bare counter to a
#: land-the-plane instruction. 0.9 means the last 10% of turns carry the
#: stricter form.
_BUDGET_LAST_STRETCH = 0.9

StopReason = Literal["answer", "finish", "max_turns", "error_streak"]


def _budget_hint(step: int, max_turns: int) -> str:
    """One-line reminder appended (ephemerally) to every turn's request.

    Bare counter for the bulk of the run; a land-the-plane instruction once
    the run is past its last stretch, so a model that lost the plot on
    step 82 of 120 does not spend the rest reading library source.

    Examples:
        >>> _budget_hint(1, 120)
        '[step 1 of 120]'
        >>> _budget_hint(108, 120).startswith('[step 108 of 120] the budget')
        True
        >>> _budget_hint(3, 3).startswith('[step 3 of 3] the budget')
        True
    """
    line = f"[step {step} of {max_turns}]"
    if step >= max(1, int(max_turns * _BUDGET_LAST_STRETCH)):
        return (
            f"{line} the budget is nearly out. Stop opening new lines of "
            f"inquiry. Write what you have to the notebook and call finish "
            f"with the result you can defend, even if it names an incomplete "
            f"gate — a truthful partial answer beats a stopped-at-max_turns."
        )
    return line


class ChatBackend(Protocol):
    """What the loop needs from a model client (tests substitute a script)."""

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatReply: ...


class TurnResult(BaseModel):
    """How one ``run_turn`` ended."""

    model_config = ConfigDict(frozen=True)

    text: str
    stop_reason: StopReason
    steps: int

    @property
    def finished(self) -> bool:
        """True when the model declared the task done via the finish tool."""
        return self.stop_reason == "finish"


def connection_profile(agent: AgentConfig) -> tuple[object, ...]:
    """What must match for two agents to share one chat client.

    Everything the client constructor bakes in: a delegation whose
    specialist matches the parent on all of these reuses the parent's
    client (one connection, one server); anything else builds its own.
    """
    return (
        agent.provider,
        agent.resolved_endpoint,
        agent.model,
        agent.temperature,
        agent.effort,
        agent.max_reply_tokens,
        agent.request_timeout_s,
    )


def client_from_config(agent: AgentConfig) -> ChatBackend:
    """Build the client the config describes, refusing the unconfigured.

    The API key is read from the environment variable ``api_key_env``
    *names* — a set name whose variable is missing is a loud error, not an
    anonymous request that fails somewhere down the line. Anthropic has no
    anonymous access, so a missing key there is refused before any request.
    """
    if agent.model is None:
        served = (
            "e.g. claude-opus-5" if agent.provider == "anthropic" else "'mason doctor' lists"
        )
        raise MasonError(
            f"no model configured: set [agent] model in the slab config "
            f"('slab config init' writes a template; {served} what the endpoint serves)"
        )
    key_var = agent.resolved_api_key_env
    api_key: str | None = None
    if key_var is not None:
        api_key = os.environ.get(key_var)
        if api_key is None:
            raise MasonError(
                f"the {agent.provider} provider needs an API key: ${key_var} is not set "
                f"in the environment (name a different variable with [agent] api_key_env)"
            )
    if agent.provider == "anthropic":
        from mason.anthropic import AnthropicClient

        assert api_key is not None  # resolved_api_key_env always names one here
        return AnthropicClient(
            agent.model,
            api_key,
            endpoint=agent.resolved_endpoint,
            max_reply_tokens=agent.max_reply_tokens,
            effort=agent.effort,
            timeout_s=agent.request_timeout_s,
        )
    return ChatClient(
        agent.resolved_endpoint,
        agent.model,
        api_key=api_key,
        temperature=agent.temperature,
        timeout_s=agent.request_timeout_s,
        max_reply_tokens=agent.max_reply_tokens,
    )


class Mason:
    """One conversation with one agent of the roster (the PI by default).

    Args:
        session: The project/session state (paths, config, transcript).
        client: A chat backend; built from the session's config when omitted.
        toolbox: The tool set; built for the session's card when omitted.
        resume_from: Messages replayed from an earlier transcript (the system
            message is always rebuilt fresh — plan, notebook, and environment
            re-enter current, not as they were).
        skills: The full skill catalog; discovered from the project directory
            when omitted. The card's scope narrows it to what this agent sees.
        spec: The agent card to run as; ``None`` resolves the roster and uses
            ``pi``. ``[agent.roster.<name>]`` overrides apply to the session's
            config, and CLI flag overrides stay on top of them.
        roster: The full roster; discovered when omitted.
        depth: Delegation depth. At 0 the card's ``delegates`` flag can grant
            the ``delegate`` tool; below that it never does, and ``plan`` is
            withheld — the plan belongs to the turn owner.
    """

    def __init__(
        self,
        session: MasonSession,
        client: ChatBackend | None = None,
        toolbox: Toolbox | None = None,
        resume_from: list[dict[str, Any]] | None = None,
        skills: dict[str, Skill] | None = None,
        spec: AgentSpec | None = None,
        roster: dict[str, AgentSpec] | None = None,
        depth: int = 0,
    ) -> None:
        self.session = session
        self.roster = roster if roster is not None else discover_roster(session.cwd)
        if spec is None:
            spec = self.roster["pi"]  # the built-in layer guarantees pi exists
        self.spec = spec
        self.depth = depth
        if depth == 0:
            check_overrides(session.agent, self.roster)
            # One running loop per workspace; children run inside this lock.
            session.acquire_session_lock()
        session.agent_name = spec.name
        self._apply_roster_override()
        self.client: ChatBackend = (
            client if client is not None else client_from_config(session.agent)
        )
        self.all_skills = skills if skills is not None else discover_skills(session.cwd)
        self.skills = skills_for(spec, self.all_skills)
        self.toolbox = (
            toolbox
            if toolbox is not None
            else build_toolbox(
                session,
                spec,
                depth=depth,
                skills=self.all_skills,
                roster=self.roster,
                parent_client=self.client,
            )
        )
        self.fenced = session.agent.tool_protocol == "fenced"
        catalog = self.toolbox.catalog_text() if self.fenced else None
        self._team = team_block(spec, self.roster) if "delegate" in self.toolbox.tools else None
        self.messages: list[dict[str, Any]] = system_messages(
            session, spec, catalog, skills=self.skills, team=self._team
        )
        self._catalog = catalog
        self._last_prompt_tokens: int | None = None
        self._messages_at_last_compaction = 0
        # Repetition guard state: the last (name, arguments, result) triple
        # and how many consecutive times it has recurred unchanged.
        self._last_identical: tuple[str, str, str] | None = None
        self._repeat_streak = 0
        if resume_from:
            # Re-record the replayed history into THIS session's transcript so
            # each transcript stays self-contained: resuming a resumed session
            # must not amputate everything before the previous resume.
            self.session.record({"type": "resume", "messages": len(resume_from)})
            for message in resume_from:
                self._append(message)

    def _apply_roster_override(self) -> None:
        """``[agent.roster.<name>]`` under the flags: config per agent, flags on top.

        The session's config arrives with CLI flag overrides already applied;
        the card's roster table must sit *under* those flags, so the merge
        order is base, then the table, then the flags again. When anything
        changed, the endpoint is resolved again — which server is right
        depends on the provider and model — except a ``--endpoint`` flag,
        which stays pinned.
        """
        session = self.session
        effective = roster_agent_config(session.agent, self.spec.name)
        if session.flag_updates:
            effective = override_agent(effective, session.flag_updates)
        if effective == session.agent:
            return
        session.agent = effective
        pinned = session.endpoint if session.endpoint_origin == "--endpoint" else None
        session.resolve_endpoint(pinned)

    # -- the loop -------------------------------------------------------------

    def run_turn(self, user_text: str) -> TurnResult:
        """Drive one goal until an answer, a finish, or a harness stop."""
        self._append({"role": "user", "content": user_text})
        error_streak = 0
        max_turns = self.session.agent.max_turns
        for step in range(1, max_turns + 1):
            self._maybe_compact()
            reply = self._call_model(hint=_budget_hint(step, max_turns))
            calls = list(reply.tool_calls)
            from_text = False
            if not calls:
                # The text-protocol ladder: the documented fenced format, then
                # the llama-style {"name": ..., "parameters": ...} that open
                # models leak into content even when served with a tool parser.
                if self.fenced:
                    calls = list(parse_fenced_calls(reply.content))
                if not calls:
                    calls = list(parse_loose_calls(reply.content, frozenset(self.toolbox.tools)))
                from_text = bool(calls)
            self._append_assistant(reply, has_calls=bool(calls))
            self._observe_step(reply, interim=bool(calls) and not from_text)
            if not calls:
                text = reply.content or ""
                if reply.finish_reason == "max_tokens":
                    # A truncated answer must not be passed off as a finished
                    # one: the reply budget bounds thinking plus text together.
                    text += (
                        "\n\n[truncated: the model hit its reply-token ceiling; "
                        "raise [agent] max_reply_tokens or narrow the goal]"
                    )
                return TurnResult(text=text, stop_reason="answer", steps=step)
            for position, call in enumerate(calls):
                if call.name == "finish" and call.arguments_error is None:
                    if len(calls) > 1:
                        # A finish sharing its reply with other tool calls was
                        # written before their results existed — its report
                        # can only be a guess (open models emit exactly this,
                        # with placeholder text where the evidence should be).
                        self._append_tool_result(
                            call,
                            "finish not honored: it arrived in the same reply as "
                            "other tool calls, so its report was written before "
                            "their results existed; read the results, then call "
                            "finish alone",
                            as_text=from_text,
                        )
                        continue
                    report = str(call.arguments.get("report", "") or "").strip()
                    if not report:
                        # Every other tool gets required-argument validation in
                        # dispatch; finish is handled here, so it gets the same
                        # contract here. An empty report closes nothing.
                        self._append_tool_result(
                            call,
                            "finish not honored: the required 'report' argument is "
                            "missing or empty; call finish again with the full "
                            "report text",
                            as_text=from_text,
                        )
                        continue
                    self._append_tool_result(call, "task closed", as_text=from_text)
                    self._answer_unrun(calls[position + 1 :], from_text=from_text)
                    self.session.record({"type": "finish", "report": report})
                    return TurnResult(text=report, stop_reason="finish", steps=step)
                result, ok = self._dispatch(call)
                result = self._note_repetition(call, result)
                self._append_tool_result(call, result, as_text=from_text)
                error_streak = 0 if ok else error_streak + 1
                if error_streak >= _ERROR_STREAK_LIMIT:
                    self._answer_unrun(calls[position + 1 :], from_text=from_text)
                    return TurnResult(
                        text=(
                            f"stopped: {_ERROR_STREAK_LIMIT} consecutive tool calls "
                            f"failed at the harness level; the transcript holds the "
                            f"evidence — the last failure was: {result}"
                        ),
                        stop_reason="error_streak",
                        steps=step,
                    )
        return TurnResult(
            text=(
                f"stopped at the {self.session.agent.max_turns}-call budget for one "
                f"goal without a final answer; PLAN.md and NOTEBOOK.md hold the "
                f"state — continue with a narrower goal or raise [agent] max_turns"
            ),
            stop_reason="max_turns",
            steps=self.session.agent.max_turns,
        )

    def _note_repetition(self, call: Any, result: str) -> str:
        """Annotate a result identical to the same call's previous result.

        An open model in a loop can re-run one command dozens of times,
        reading each identical result as new information (a real 240-call
        session spent 82 calls on one byte-identical readelf pipeline).
        The model cannot see the sameness; the harness can. The call still
        executes every time — polling a queue legitimately repeats, and its
        result changes when it matters — but a repeat with an unchanged
        result carries an escalating note the model reads as evidence.
        """
        key = (call.name, call.arguments_raw, result)
        if key == self._last_identical:
            self._repeat_streak += 1
        else:
            self._last_identical = key
            self._repeat_streak = 1
            return result
        if self._repeat_streak == 2:
            return (
                f"{result}\n[note: this is the same call as the previous step, "
                f"and the result is identical]"
            )
        return (
            f"{result}\n[note: this exact call has now returned this exact result "
            f"{self._repeat_streak} times in a row. Repeating it cannot produce new "
            f"information. Record what you learned in the notebook, change the "
            f"approach, or finish with a report naming the blocker.]"
        )

    def _answer_unrun(self, calls: list[Any], *, from_text: bool) -> None:
        """Answer tool calls we short-circuited past.

        Every ``tool_calls`` entry in an assistant message must have a
        matching tool result or the history is malformed — servers reject it
        on the next request, and ``--resume`` would replay the breakage.
        """
        for call in calls:
            self._append_tool_result(
                call, "not run: the turn ended before this call", as_text=from_text
            )

    def _dispatch(self, call: Any) -> tuple[str, bool]:
        """Run one tool call; ``ok=False`` marks harness-level failures only.

        A domain answer like "no such file" is the model's problem to react
        to, not a harness failure; malformed arguments, missing required
        arguments, unknown tools, refused approvals, and crashed handlers all
        count toward the abort streak — they are the loops that never
        converge on their own.
        """
        result = self.toolbox.dispatch(call)
        hard_failure = (
            call.arguments_error is not None
            or call.name not in self.toolbox.tools
            or result.startswith(f"tool {call.name} failed:")
            or result.startswith(f"tool {call.name} not run:")
            or result.startswith(f"tool {call.name} was not approved")
        )
        return result, not hard_failure

    def _call_model(self, *, hint: str | None = None) -> ChatReply:
        tools = None if self.fenced else self.toolbox.specs()
        # An ephemeral system message tacked onto the end each turn — the
        # step-of-budget line, and stricter guidance near the ceiling.
        # Never persisted: the counter changes each turn and stale copies
        # in the transcript would mislead --resume.
        messages = self.messages
        if hint is not None:
            messages = [*messages, {"role": "system", "content": hint}]
        try:
            reply = self.client.chat(messages, tools)
        except ContextOverflowError as e:
            # The server knows the window better than our estimate: compact
            # once and retry. If there was nothing left to fold, retrying
            # would just repeat the overflow — say so instead.
            if not self._compact():
                raise ContextOverflowError(
                    f"{e} — and there is nothing left to compact: the system prompt, "
                    f"plan, notebook tail, and current goal already exceed the model's "
                    f"window. Shorten PLAN.md/AGENTS.md, or serve a larger context."
                ) from e
            messages = self.messages
            if hint is not None:
                messages = [*messages, {"role": "system", "content": hint}]
            reply = self.client.chat(messages, tools)
        self.session.count_usage(reply.prompt_tokens, reply.completion_tokens)
        self._last_prompt_tokens = reply.prompt_tokens
        self.session.record(
            {
                "type": "usage",
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
            }
        )
        return reply

    # -- message bookkeeping --------------------------------------------------

    def _append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.session.record({"type": "message", "message": message})

    def _observe_step(self, reply: ChatReply, *, interim: bool) -> None:
        """Live step output to whoever is watching (chat wires a printer).

        Interim assistant text is shown only for native tool calls: under
        the text protocols the content *is* the call markup, and the
        approval preview already shows the call.
        """
        observer = self.session.observer
        if observer is None:
            return
        attribution = self.session.attribution()
        if reply.reasoning:
            observer("reasoning", attribution, reply.reasoning)
        if interim and reply.content and reply.content.strip():
            observer("text", attribution, reply.content)

    def _append_assistant(self, reply: ChatReply, *, has_calls: bool) -> None:
        if reply.reasoning:
            # Its own event, not a message field: --resume replays message
            # events verbatim, and reasoning must never re-enter the
            # model's context (the client contract, mason.client).
            self.session.record({"type": "reasoning", "text": reply.reasoning})
        message: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if has_calls and not self.fenced and reply.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_raw},
                }
                for call in reply.tool_calls
            ]
        self._append(message)

    def _append_tool_result(self, call: Any, result: str, *, as_text: bool = False) -> None:
        # Calls parsed out of plain text have no server-side tool_calls to
        # answer, so their results go back as labeled user messages (a tool
        # message without a preceding tool_calls violates the protocol).
        if self.fenced or as_text:
            self._append(
                {"role": "user", "content": f"[tool result: {call.name}]\n{result}"}
            )
        else:
            self._append({"role": "tool", "tool_call_id": call.id, "content": result})

    # -- compaction -----------------------------------------------------------

    def _estimated_prompt_tokens(self) -> int:
        """The server's last count when we have one, chars/4 as the floor."""
        estimate = sum(len(json.dumps(m)) for m in self.messages) // _CHARS_PER_TOKEN
        if self._last_prompt_tokens is not None:
            return max(self._last_prompt_tokens, estimate // 2)
        return estimate

    def _maybe_compact(self) -> None:
        agent = self.session.agent
        if self._estimated_prompt_tokens() < int(agent.context_window * agent.compact_at):
            return
        # Over budget but nothing new since the last fold: compacting again
        # would re-summarize the summary every single step, burning a model
        # call per turn for no reduction. Let the turn proceed; a real
        # overflow is caught by the server and handled loudly.
        if len(self.messages) <= self._messages_at_last_compaction:
            return
        self._compact()

    def _compact(self) -> bool:
        """Fold the middle of the conversation into a structured summary.

        Returns False when there was nothing foldable (the caller decides
        whether that is fine or fatal).
        """
        keep = _COMPACTION_KEEP_MESSAGES
        if len(self.messages) <= 1 + keep + 2:  # nothing worth folding
            return False
        boundary = len(self.messages) - keep
        # A tool message must keep the assistant message that called it.
        while boundary > 1 and self.messages[boundary].get("role") == "tool":
            boundary -= 1
        folded = self.messages[1:boundary]
        if not folded:
            return False
        summary_reply = self.client.chat(
            [
                {"role": "system", "content": COMPACTION_PROMPT},
                {"role": "user", "content": _render_for_summary(folded)},
            ],
            None,
        )
        summary = (summary_reply.content or "").strip() or "(the summarizer said nothing)"
        self.session.count_usage(summary_reply.prompt_tokens, summary_reply.completion_tokens)
        # The summary already travels two ways: as a user message prepended
        # to the rebuilt conversation, and as a {type: compaction} event in
        # the transcript. The per-session file is a human debugging aid,
        # deliberately not the notebook — the notebook is what the AGENT
        # kept, this is what the HARNESS folded.
        self.session.compactions_append(summary)
        rebuilt = system_messages(
            self.session, self.spec, self._catalog, skills=self.skills, team=self._team
        )
        tail = self.messages[boundary:]
        self.messages = [
            *rebuilt,
            {
                "role": "user",
                "content": f"[history compacted; working summary]\n{summary}",
            },
            *tail,
        ]
        self._last_prompt_tokens = None
        self._messages_at_last_compaction = len(self.messages)
        self.session.record({"type": "compaction", "summary": summary})
        return True


def _render_for_summary(messages: list[dict[str, Any]], per_message_chars: int = 1_500) -> str:
    """A transcript rendering the summarizer can read; long entries clipped."""
    lines = []
    for message in messages:
        role = str(message.get("role", "?"))
        content = message.get("content")
        text = "" if content is None else str(content)
        for call in message.get("tool_calls", ()):
            function = call.get("function", {})
            text += f"\n[called {function.get('name')}({function.get('arguments', '')[:300]})]"
        if len(text) > per_message_chars:
            text = text[:per_message_chars] + " [...]"
        lines.append(f"--- {role} ---\n{text}")
    return "\n".join(lines)
