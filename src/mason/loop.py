"""The turn engine: one ReAct loop, budgeted context, harness-owned discipline.

One :class:`Mason` holds one conversation. ``run_turn`` takes a user goal and
loops model call -> tool dispatch -> observation until the model answers in
plain text, calls ``finish``, or a harness limit stops it. The limits live in
code because prompts do not enforce invariants:

* ``max_turns`` model calls per goal — the runaway stop.
* A consecutive-tool-failure streak aborts the turn with the evidence in
  place (five malformed or crashing calls in a row is a stuck agent, not
  progress).
* Tool-result clearing at ``clear_tool_results_at`` x ``context_window``:
  results older than the newest ``keep_tool_results`` become one-line
  placeholders (the calls stay, so they are restorable), in batches so the
  cached prompt prefix is rewritten rarely. Errors, skill texts, and plan
  updates are never cleared. This is the cheap layer, and the one the
  measurements favor: masking old observations matches summarization on
  solve rate at about half the cost.
* Compaction triggers at ``compact_at`` x ``context_window`` — well below
  the window, because model quality degrades before the hard limit — and a
  context-overflow answer from the server forces one immediate compaction
  and retry. Compaction folds the middle of the conversation into a
  structured summary (state, verified results, failures observed, open
  questions), writes that summary into the per-session compactions file,
  and rebuilds the system message fresh so plan and notebook re-enter the
  context updated.

Both tool protocols run through the same loop: native OpenAI tool calls, or
the fenced-block text protocol for servers without a tool-call parser.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from mason.client import (
    ChatClient,
    ChatReply,
    ContextOverflowError,
    parse_fenced_calls,
    parse_loose_calls,
    unfinished_call_name,
)
from mason.config import AgentConfig, override_agent, roster_agent_config
from mason.errors import MasonError
from mason.mechanisms import effective, enabled
from mason.prompts import COMPACTION_PROMPT, system_messages, team_block
from mason.reviews import plan_is_approved
from mason.roster import (
    AgentSpec,
    check_overrides,
    critics,
    discover_roster,
    hands,
    skills_for,
)
from mason.session import MasonSession
from mason.skills import Skill, discover_skills
from mason.tools import LOOKING_TOOLS, TOOL_VOCABULARY, Toolbox, build_toolbox
from slab._version import __version__

_ERROR_STREAK_LIMIT = 5
#: The identical call whose identical result is returned this many times in
#: one session gets told so: the earlier copies were cleared, and the model
#: fetched again instead of writing the fact down (a real session described
#: one task seven times and read one template six times).
_REFETCH_NOTE_AT = 3
#: How much of a cleared result its placeholder quotes back.
_CLEAR_HEAD_CHARS = 120
#: This many consecutive steps made only of looking tools, and the turn's
#: hint says so; it says so again every few steps after. One real campaign
#: spent 72 minutes and 80 % of its completion tokens in two such runs
#: with nothing in the loop to make it step back.
_LOOKING_HINT_AT = 15
_LOOKING_HINT_EVERY = 5
#: One step's reply budget when [agent] max_reply_tokens is unset, the
#: same as the Anthropic client's default. A thinking model's think block
#: counts toward it, so it caps one step's spend, not the campaign's; a
#: cut reply is marked and nudged once. Bounded below by the room the
#: window has left, because vLLM refuses prompt + max_tokens beyond it.
_DEFAULT_REPLY_TOKENS = 16_000
_REPLY_MARGIN_TOKENS = 1_024
_MIN_REPLY_TOKENS = 1_024
#: What replaces an older plan echo once a newer one arrives. The newest
#: stays verbatim (the recitation that keeps the goal in view); three
#: copies of a growing plan rode every prompt of one real campaign.
_PLAN_SUPERSEDED = (
    "[cleared: plan result superseded by the update at a later step; PLAN.md "
    "holds the current plan]"
)
_COMPACTION_KEEP_MESSAGES = 6
#: The summarizer's reply budget. A summary is a working state, not a
#: transcript: one real compaction ran to 14,000 tokens carrying the
#: working state of a dead end, most of it lost to the next fold anyway.
_COMPACTION_MAX_TOKENS = 4_096
#: What the model reads after a reply with no text and no tool call. Such a
#: reply is a fault (a thinking model that spent its budget in the think
#: block, or a template that emitted nothing), not an answer; one campaign
#: closed on exactly that after two and a half hours. The loop asks once.
_EMPTY_REPLY_NUDGE = (
    "[harness] your reply was empty: act with a tool call, or call finish with the report"
)
#: A reply the server cut at max_tokens carried nothing usable: with a
#: thinking model the whole budget went to the think block. Repeating the
#: identical request would repeat the cut, so the retry asks for brevity
#: and runs at low effort. One real critic pass spent 78 minutes and
#: 73,000 completion tokens over twelve steps, then lost its verdict this way.
_CUT_REPLY_NUDGE = (
    "[harness] your reply was cut at the reply-token ceiling before it ended, so "
    "none of it was received. Do not deliberate further: answer in a few lines, "
    "or call finish with a short report (for a review: the verdict and one line "
    "per finding)."
)
_TRUNCATED_MARK = (
    "\n\n[truncated: the reply hit the reply-token ceiling twice, the second time "
    "at low effort; the operator can raise [agent] max_reply_tokens or lower the "
    "effort for this agent]"
)
#: A cut reply that held text and no tool call was two thirds of an
#: answer or a script, not a spent think block. The text stays in the
#: history and the model resumes it; the join on return trims the
#: incomplete last line the model was told not to repeat. One planner on
#: 2026-09-10 briefed the same specialist three times because the brevity
#: nudge threw such replies away: about thirty minutes and 470,000 tokens.
_CONTINUE_REPLY_NUDGE = (
    "[harness] your reply was cut at the reply-token ceiling while you were writing "
    "text; what you wrote is kept above. Continue from your last complete line; do "
    "not repeat what you wrote."
)
#: A cut inside a tool call's arguments: the call is not run, and the
#: model is told which tool and how to fit the call in the ceiling.
_CUT_CALL_NUDGE = (
    "[harness] your reply was cut at the reply-token ceiling inside the arguments "
    "of your {name} call, so the call did not run. Write the file in parts: "
    "write_file the first part, then extend it with edit_file (replace its last "
    "line with that line plus the next part). Or shorten the arguments. Then call "
    "again."
)
#: Messages that must accumulate after a compaction before another may fire.
_COMPACTION_MIN_NEW_MESSAGES = 4
_CHARS_PER_TOKEN = 4  # the usual rough estimate; real usage numbers override it
#: Tool-result clearing: a result shorter than the floor is not worth the
#: cache invalidation a clearing costs, and a clearing that frees less than
#: the batch minimum is deferred so the prompt prefix is rewritten rarely,
#: in large steps, not every turn.
_CLEAR_FLOOR_CHARS = 400
_CLEAR_AT_LEAST_CHARS = 6_000
#: Results that must stay verbatim: a skill's instructions are consulted
#: for the rest of the task, and a plan update is the recitation that keeps
#: the goal in view.
_NEVER_CLEARED = frozenset({"skill", "plan", "delegate", "review"})
_CLEARED_MARK = "[cleared:"
_TEXT_RESULT_PREFIX = "[tool result: "
#: The truncation warning needs a prompt of real size to judge: below this
#: estimate, a server's count and the chars/4 guess disagree for ordinary
#: reasons.
_TRUNCATION_MIN_TOKENS = 3_000
#: Ratio at which the budget hint escalates from a bare counter to a
#: land-the-plane instruction. 0.9 means the last 10% of turns carry the
#: stricter form.
_BUDGET_LAST_STRETCH = 0.9

StopReason = Literal["answer", "finish", "max_turns", "error_streak", "error"]


def _looking_hint(looking: int) -> str:
    """The step-back sentence after a run of steps that only looked, or empty.

    Examples:
        >>> _looking_hint(14)
        ''
        >>> _looking_hint(15).startswith(' [15 consecutive steps have only read')
        True
        >>> _looking_hint(17)
        ''
        >>> _looking_hint(20).startswith(' [20 consecutive')
        True
    """
    if looking < _LOOKING_HINT_AT or (looking - _LOOKING_HINT_AT) % _LOOKING_HINT_EVERY:
        return ""
    return (
        f" [{looking} consecutive steps have only read and listed: nothing launched, "
        f"planned, noted, briefed, or finished. Step back: re-read PLAN.md, write what "
        f"you learned in the notebook, load the skill that covers this, or delegate.]"
    )


def _turn_hint(
    step: int, max_turns: int, looking: int = 0, *, budget: bool = True, step_back: bool = True
) -> str:
    """The ephemeral line each request ends with: the budget, and the step-back.

    Each half has its switch (``budget-hint``, ``looking-hint``); with both
    off the line is empty and the request carries no hint at all.

    Examples:
        >>> _turn_hint(1, 10).startswith('[harness: model call 1 of 10')
        True
        >>> _turn_hint(1, 10, 15, budget=False).startswith('[15 consecutive')
        True
        >>> _turn_hint(1, 10, 15, budget=False, step_back=False)
        ''
    """
    line = _budget_hint(step, max_turns) if budget else ""
    tail = _looking_hint(looking) if step_back else ""
    return (line + tail).strip()


def _budget_hint(step: int, max_turns: int) -> str:
    """One-line reminder appended (ephemerally) to every turn's request.

    The line names itself on every turn: a bare ``[step N of M]`` was read
    by a planner card as the progress of the MD run it was waiting on, for
    twenty turns. A land-the-plane instruction joins it once the run is
    past its last stretch, so a model that lost the plot on step 82 of 120
    does not spend the rest reading library source.

    Examples:
        >>> _budget_hint(1, 120)
        "[harness: model call 1 of 120 in this session's budget; not the progress of any run]"
        >>> _budget_hint(108, 120).startswith('[harness: model call 108 of 120')
        True
        >>> 'the budget is nearly out' in _budget_hint(108, 120)
        True
        >>> 'the budget is nearly out' in _budget_hint(3, 3)
        True
    """
    line = (
        f"[harness: model call {step} of {max_turns} in this session's budget; "
        f"not the progress of any run]"
    )
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
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> ChatReply: ...


def _retry_effort(configured: str | None) -> str:
    """The effort for the one retry after a cut reply: low, unless the agent
    already runs without reasoning.

    Examples:
        >>> _retry_effort("xhigh"), _retry_effort(None), _retry_effort("none")
        ('low', 'low', 'none')
    """
    return "none" if configured == "none" else "low"


class TurnResult(BaseModel):
    """How one ``run_turn`` ended."""

    model_config = ConfigDict(frozen=True)

    text: str
    stop_reason: StopReason
    steps: int
    #: The structured hand-back of a finish call: result name -> {value, unit},
    #: and the run ids the model cited for them. Empty for every other stop.
    results: dict[str, Any] = Field(default_factory=dict)
    run_ids: tuple[str, ...] = ()
    #: A review's verdict, ``approve`` or ``revise``, when finish carried one.
    verdict: str | None = None
    #: The answer was cut at the reply-token ceiling, on the retry too.
    truncated: bool = False

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


def _check_lead_can_delegate(
    spec: AgentSpec, agent: AgentConfig, roster: dict[str, AgentSpec]
) -> None:
    """Refuse a card that hands its work to a team it cannot reach.

    A card whose tool allowlist names ``delegate`` has no way to work
    without it: the planner, for one, has no shell and no launch tool. Run
    with delegation off or with nobody to delegate to, it would sit with
    the tools of a reader. Better to say so before the session lock is
    taken and the model is called.
    """
    if spec.tools is None or "delegate" not in spec.tools:
        return
    if not enabled(agent, "delegation"):
        raise MasonError(
            f"the {spec.name} card hands every step to its team, but [agent] "
            f"delegation is off; set delegation = true or run another card"
        )
    if not hands(spec, roster):
        raise MasonError(
            f"the {spec.name} card hands every step to its team, but no card on the "
            f"roster can take a brief (every other card delegates); add a card that "
            f"does not, such as the built-in worker"
        )


def _check_review_first(spec: AgentSpec, agent: AgentConfig, roster: dict[str, AgentSpec]) -> None:
    """Refuse a card that must be reviewed before compute but cannot be.

    A ``review_first`` card's launches and briefs are refused until a
    critic approves the plan. With delegation off, or with no critic on
    the roster, that approval can never arrive, and the card would sit
    with every compute tool refusing. Say so before the model is called.
    """
    if not spec.review_first or not enabled(agent, "critic-gate"):
        # With the critic gate switched off (an ablation), the card runs
        # ungated: there is nothing a critic would be needed for.
        return
    if not enabled(agent, "delegation"):
        raise MasonError(
            f"the {spec.name} card spends no compute before a critic approves the plan, "
            f"but [agent] delegation is off, so no critic can run; set delegation = "
            f"true or run another card"
        )
    if not critics(roster):
        raise MasonError(
            f"the {spec.name} card spends no compute before a critic approves the plan, "
            f"but no card on the roster reviews; add one, such as the built-in critic"
        )


def _take_api_key(key_var: str, keys: dict[str, str]) -> str | None:
    """The key named by *key_var*: from *keys* when read before, else from the
    environment, withdrawn from os.environ once read so a workflow script the
    model launches in-process (or a shell it drives) cannot print it back
    into the context and the transcript. *keys* is the session's store, so a
    delegate's client finds the key there after the withdrawal."""
    if key_var in keys:
        return keys[key_var]
    value = os.environ.pop(key_var, None)
    if value is not None:
        keys[key_var] = value
    return value


def client_from_config(agent: AgentConfig, keys: dict[str, str] | None = None) -> ChatBackend:
    """Build the client the config describes, refusing the unconfigured.

    The API key is read from the environment variable ``api_key_env``
    *names* — a set name whose variable is missing is a loud error, not an
    anonymous request that fails somewhere down the line. Anthropic has no
    anonymous access, so a missing key there is refused before any request.
    """
    if agent.model is None:
        served = "e.g. claude-opus-5" if agent.provider == "anthropic" else "'mason doctor' lists"
        raise MasonError(
            f"no model configured: set [agent] model in the slab config "
            f"('slab config init' writes a template; {served} what the endpoint serves)"
        )
    key_var = agent.resolved_api_key_env
    api_key: str | None = None
    if key_var is not None:
        api_key = _take_api_key(key_var, keys if keys is not None else {})
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
        effort=agent.effort,
    )


def _reap_at_session_start(session: MasonSession) -> None:
    """Mark failed the running runs whose process on this host is gone.

    A store that cannot be opened is not a reason to refuse the session:
    the first workspace tool call names the fault with its recovery.
    """
    import sqlite3

    from foundation.errors import FoundationError
    from foundation.runtime import Workspace

    try:
        with Workspace(session.workspace_root) as ws:
            ws.reap_dead(caller="session start")
    except (FoundationError, sqlite3.Error, OSError):
        return


def _retire_at_finish(session: MasonSession, run_ids: tuple[str, ...]) -> None:
    """Promote what the finish cites; expire what this session did not cite.

    The workspace retention policy's ``finish`` rule decides: with
    ``promote_cited`` off and ``uncited`` at ``keep`` nothing happens. The
    outcome lands in the transcript as a ``retire`` event, and a retire
    that fails records its error there. It never fails the campaign: the
    finish stands as reported.
    """
    import sqlite3

    from foundation import _ops
    from foundation.errors import FoundationError
    from foundation.runtime import Workspace
    from slab.errors import SlabError

    try:
        rule = _ops.load_policy(Path(session.workspace_root)).finish
        if not rule.promote_cited and rule.uncited == "keep":
            return
        with Workspace(session.workspace_root) as ws:
            report = _ops.retire_session(
                ws,
                session.session_id,
                keep=run_ids if rule.promote_cited else (),
                mode=rule.uncited,
                actor="finish",
            )
    except (FoundationError, SlabError, ValueError, sqlite3.Error, OSError) as e:
        session.record({"type": "retire", "error": str(e)})
        return
    session.record({"type": "retire", **report})


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
        expected_results: dict[str, str] | None = None,
    ) -> None:
        self.session = session
        #: The result names the goal asks finish to carry, each with its
        #: unit. They travel explicitly from the caller that knows the goal
        #: (a benchmark question, ``--expect``), never parsed back out of
        #: the goal text. None means the finish's result names go unchecked.
        self.expected_results = dict(expected_results) if expected_results else None
        self.roster = roster if roster is not None else discover_roster(session.cwd)
        #: The step the current or last turn reached; read by a parent whose
        #: child died of a transport error mid-turn, so the footer can say
        #: how far it got.
        self.steps_taken = 0
        if spec is None:
            spec = self.roster["pi"]  # the built-in layer guarantees pi exists
        self.spec = spec
        self.depth = depth
        if depth == 0:
            check_overrides(session.agent, self.roster)
            _check_lead_can_delegate(spec, session.agent, self.roster)
            _check_review_first(spec, session.agent, self.roster)
            # One running loop per workspace; children run inside this lock.
            session.acquire_session_lock()
            # A run left at status running by a hard-killed process is
            # marked failed before the first turn, so the record the agent
            # reads never carries a dead run as a live one.
            _reap_at_session_start(session)
            # An approval on record for the plan as it reads now still holds:
            # a resumed campaign does not pay for a second review.
            session.plan_approved = plan_is_approved(session)
        session.agent_name = spec.name
        self._apply_roster_override()
        self.client: ChatBackend = (
            client if client is not None else client_from_config(session.agent, session.api_keys)
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
        self._team = (
            team_block(
                spec,
                self.roster,
                delegate="delegate" in self.toolbox.tools,
                review="review" in self.toolbox.tools,
            )
            or None
        )
        # The latest review of the plan is shown to whoever acts on it: a
        # card that can ask for one, or one that may not launch without one.
        self._shows_review = "review" in self.toolbox.tools or spec.review_first
        self.messages: list[dict[str, Any]] = system_messages(
            session,
            spec,
            catalog,
            skills=self.skills,
            team=self._team,
            review=self._shows_review,
            absent_tools=self._absent_tools(),
        )
        self._catalog = catalog
        self._last_prompt_tokens: int | None = None
        self._messages_at_last_compaction = 0
        self._truncation_warned = False
        # Repetition guard state: the last (name, arguments, result) triple
        # and how many consecutive times it has recurred unchanged.
        self._last_identical: tuple[str, str, str] | None = None
        self._repeat_streak = 0
        # Session-wide: (tool, arguments) -> (digest of its result, times seen
        # with that same result), so a re-fetch of cleared content is named.
        # ... and where the last full copy of that result sits in the
        # messages (index, step), or None once compaction rebuilt them, so a
        # repeat whose copy is still in context can point at it.
        self._seen_results: dict[tuple[str, str], tuple[str, int, int | None, int]] = {}
        # Consecutive steps whose calls were all looking tools.
        self._looking_streak = 0
        self._effort_override: str | None = None
        if depth == 0:
            # The transcript says which model answered it, so a later reader
            # (a report, a benchmark score) trusts the record, not the config
            # that may have changed since. A resumed session is a new
            # transcript and gets its own header.
            self.session.record(
                {
                    "type": "session",
                    "resumed": bool(resume_from),
                    "cwd": str(session.cwd),
                    "agent": spec.name,
                    "model": session.agent.model,
                    "provider": session.agent.provider,
                    "endpoint": session.endpoint,
                    "endpoint_origin": session.endpoint_origin,
                    "compute_profile": session.compute_profile,
                    # The cpus and gpus the session may use, so a report can
                    # say what share of them its runs held.
                    "budget": dict(session.budget),
                    "max_turns": session.agent.max_turns,
                    # The reasoning dial and the code that ran, so a review
                    # can attribute a bloated or a truncated turn.
                    "effort": session.agent.effort,
                    "version": __version__,
                    # The harness arm and its switches, so a benchmark record
                    # says which mechanisms the campaign ran with.
                    "condition": session.condition,
                    "mechanisms": list(effective(session.agent)),
                    "ablated": list(session.ablated),
                }
            )
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

    def _results_mismatch(self, results: dict[str, Any]) -> str | None:
        """The refusal for a finish whose result names miss the expected ones.

        None when no names are expected, or when the finish names exactly
        them. The message names what the finish carried and what the goal
        asks for, unit included, so the next finish can match.
        """
        if self.expected_results is None or set(results) == set(self.expected_results):
            return None
        named = ", ".join(sorted(results)) if results else "nothing"
        asked = ", ".join(f"{name} in {unit}" for name, unit in self.expected_results.items())
        return (
            f"finish not honored: results name {named}; the goal asks for {asked}; "
            f"call finish again with results keyed exactly so"
        )

    def run_turn(self, user_text: str) -> TurnResult:
        """Drive one goal until an answer, a finish, or a harness stop."""
        self._append({"role": "user", "content": user_text})
        error_streak = 0
        empty_nudged = False
        cut_nudged = False
        # The text of a reply cut mid-answer, kept for the join once the
        # continuation arrives (case 2 of the cut-reply cases).
        cut_prefix: str | None = None
        continue_cut = enabled(self.session.agent, "continue-cut-reply")
        max_turns = self.session.agent.max_turns
        for step in range(1, max_turns + 1):
            self.steps_taken = step
            self._clear_tool_results()
            self._maybe_compact()
            hint = _turn_hint(
                step,
                max_turns,
                self._looking_streak,
                budget=enabled(self.session.agent, "budget-hint"),
                step_back=enabled(self.session.agent, "looking-hint"),
            )
            reply = self._call_model(hint=hint or None)
            cut = reply.finish_reason == "max_tokens"
            cut_call = self._cut_call_name(reply) if cut else None
            if cut_call is not None and continue_cut and not cut_nudged:
                # Case 3: the cut fell inside a tool call's arguments. The
                # partial call never runs; the history keeps the text only,
                # and the model reads which tool and how to fit the call.
                cut_nudged = True
                self._append_assistant(reply, has_calls=False)
                self._observe_step(reply, interim=False)
                self.session.record({"type": "cut", "case": 3, "continued": True})
                self._append(
                    {"role": "user", "content": _CUT_CALL_NUDGE.format(name=cut_call)}
                )
                continue
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
                if cut and not cut_nudged:
                    cut_nudged = True
                    case = 3 if cut_call is not None else 2 if text.strip() else 1
                    if case == 2 and continue_cut:
                        # Case 2: the cut fell mid-text. The text stands in
                        # the history; the model resumes it, and the join
                        # below puts the two halves together on return.
                        cut_prefix = text
                        self.session.record({"type": "cut", "case": 2, "continued": True})
                        self._append({"role": "user", "content": _CONTINUE_REPLY_NUDGE})
                        continue
                    # Case 1: the budget went to the think block and no text
                    # arrived. Ask once for a short answer at low effort; a
                    # second cut ends the turn below. With the switch off,
                    # every cut reply takes this path.
                    self.session.record({"type": "cut", "case": case, "continued": False})
                    self._append({"role": "user", "content": _CUT_REPLY_NUDGE})
                    if enabled(self.session.agent, "adaptive-effort"):
                        self._effort_override = _retry_effort(self.session.agent.effort)
                    continue
                if cut_prefix is not None:
                    text = _join_cut(cut_prefix, text)
                if not text.strip() and not empty_nudged and reply.finish_reason != "max_tokens":
                    # No text and no call is a fault, not an answer. Ask once;
                    # a second empty reply ends the turn below.
                    empty_nudged = True
                    self._append({"role": "user", "content": _EMPTY_REPLY_NUDGE})
                    continue
                if cut:
                    # A truncated answer must not be passed off as a finished one.
                    text += _TRUNCATED_MARK
                return TurnResult(text=text, stop_reason="answer", steps=step, truncated=cut)
            if cut and cut_call is not None:
                # A second cut inside a call, or one with the switch off: the
                # malformed call is answered with its JSON error, as before.
                self.session.record({"type": "cut", "case": 3, "continued": False})
            # A reply that went on to act was not the answer the cut half
            # started; the half is not joined onto whatever comes later.
            cut_prefix = None
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
                    # The structured hand-back travels as given: the loop never
                    # re-shapes a report, and a scorer refuses what it cannot read.
                    raw_results = call.arguments.get("results")
                    results = dict(raw_results) if isinstance(raw_results, dict) else {}
                    mismatch = self._results_mismatch(results)
                    if mismatch is not None:
                        # The goal named its result keys; a finish under other
                        # names is the drift the scorer would fail, so it is
                        # refused here, where the agent can still fix it.
                        self._append_tool_result(call, mismatch, as_text=from_text)
                        continue
                    self._append_tool_result(call, "task closed", as_text=from_text)
                    self._answer_unrun(calls[position + 1 :], from_text=from_text)
                    raw_ids = call.arguments.get("run_ids")
                    run_ids = tuple(str(r) for r in raw_ids) if isinstance(raw_ids, list) else ()
                    raw_verdict = call.arguments.get("verdict")
                    verdict = raw_verdict if raw_verdict in ("approve", "revise") else None
                    self.session.record(
                        {
                            "type": "finish",
                            "report": report,
                            "results": results,
                            "run_ids": list(run_ids),
                            "verdict": verdict,
                        }
                    )
                    if self.depth == 0:
                        # The finish is the completion-time act: the cited
                        # runs are promoted and the session's other runs
                        # expire. A delegated child's finish decides nothing;
                        # the parent's does, for the whole session.
                        _retire_at_finish(self.session, run_ids)
                    return TurnResult(
                        text=report,
                        stop_reason="finish",
                        steps=step,
                        results=results,
                        run_ids=run_ids,
                        verdict=verdict,
                    )
                try:
                    result, ok = self._dispatch(call)
                except BaseException:
                    # A KeyboardInterrupt during a slow tool (or a crash below
                    # the tool's own guard) must not leave the assistant's
                    # tool_calls unanswered: the next request would carry a
                    # protocol-invalid history, and --resume would replay it.
                    self._answer_unrun(calls[position:], from_text=from_text)
                    raise
                if enabled(self.session.agent, "identical-result-annotation"):
                    result = self._note_repetition(call, result)
                self._append_tool_result(call, result, as_text=from_text)
                if (
                    call.name == "plan"
                    and result.startswith("PLAN.md updated:")
                    and enabled(self.session.agent, "context-hygiene")
                ):
                    self._supersede_plan_echoes()
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
            looked = all(call.name in LOOKING_TOOLS for call in calls)
            self._looking_streak = self._looking_streak + 1 if looked else 0
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

        The second count is session-wide. Clearing turns an old result into
        a placeholder, and a model that needs the content again fetches it
        again, which is the design; a model that fetches the same content a
        third time is not reading it into the notebook, and is told so.
        """
        key = (call.name, call.arguments_raw, result)
        if key == self._last_identical:
            self._repeat_streak += 1
        else:
            self._last_identical = key
            self._repeat_streak = 1
        digest = hashlib.sha256(result.encode("utf-8")).hexdigest()
        seen = self._seen_results.get((call.name, call.arguments_raw))
        same = seen is not None and seen[0] == digest
        times = seen[1] + 1 if same and seen is not None else 1
        # A second identical body while the first is still in context is
        # pure cost: point at the copy instead. Shell results are exempt (a
        # command legitimately re-runs, and its first line is its exit
        # status). A cleared copy means the re-fetch is the design.
        shown = result
        if same and seen is not None:
            copy_index, copy_step = seen[2], seen[3]
            if (
                call.name != "shell"
                and len(result) >= _CLEAR_FLOOR_CHARS
                and copy_index is not None
                and self._result_intact(copy_index)
            ):
                shown = (
                    f"[unchanged: identical to this call's result at step {copy_step}, "
                    f"still in your context; {len(result)} characters not repeated]"
                )
                self._seen_results[(call.name, call.arguments_raw)] = (
                    digest,
                    times,
                    copy_index,
                    copy_step,
                )
            else:
                self._seen_results[(call.name, call.arguments_raw)] = (
                    digest,
                    times,
                    len(self.messages),
                    self.steps_taken,
                )
        else:
            self._seen_results[(call.name, call.arguments_raw)] = (
                digest,
                times,
                len(self.messages),
                self.steps_taken,
            )
        if self._repeat_streak == 2:
            return (
                f"{shown}\n[note: this is the same call as the previous step, "
                f"and the result is identical]"
            )
        if self._repeat_streak > 2:
            return (
                f"{shown}\n[note: this exact call has now returned this exact result "
                f"{self._repeat_streak} times in a row. Repeating it cannot produce new "
                f"information. Record what you learned in the notebook, change the "
                f"approach, or finish with a report naming the blocker.]"
            )
        if times >= _REFETCH_NOTE_AT:
            return (
                f"{shown}\n[note: this call has returned this same content {times} "
                f"times in this session. Each copy costs context: write what you need "
                f"from it into the notebook or the plan now, and stop fetching it.]"
            )
        return shown

    def _result_intact(self, index: int) -> bool:
        """Whether the tool result at *index* still carries its body."""
        if index >= len(self.messages):
            return False
        message = self.messages[index]
        if message.get("role") not in ("tool", "user"):
            return False
        content = str(message.get("content") or "")
        return _CLEARED_MARK not in content[:200] and not content.startswith("[unchanged:")

    def _supersede_plan_echoes(self) -> None:
        """Replace every plan echo but the newest with a one-line marker.

        The plan tool echoes the whole plan back so the goal stays in view;
        once a newer echo exists, the older ones are only cost. Runs when a
        plan result lands, whatever the prompt size.
        """
        names_by_id: dict[str, str] = {}
        echoes: list[int] = []
        for index, message in enumerate(self.messages):
            role = message.get("role")
            if role == "assistant":
                for call in message.get("tool_calls") or ():
                    names_by_id[str(call.get("id"))] = str(call["function"]["name"])
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            by_id = role == "tool" and names_by_id.get(str(message.get("tool_call_id"))) == "plan"
            by_text = role == "user" and content.startswith(f"{_TEXT_RESULT_PREFIX}plan]")
            if by_id or by_text:
                echoes.append(index)
        freed = 0
        cleared = 0
        for index in echoes[:-1]:
            message = self.messages[index]
            content = str(message["content"])
            if _CLEARED_MARK in content[:200]:
                continue
            prefix = ""
            if message.get("role") == "user":
                head, _, _ = content.partition("\n")
                prefix = head + "\n"
            message["content"] = f"{prefix}{_PLAN_SUPERSEDED}"
            freed += len(content)
            cleared += 1
        if cleared:
            self._last_prompt_tokens = None
            self.session.record(
                {"type": "clearing", "cleared": cleared, "chars": freed, "why": "plan superseded"}
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

    def _cut_call_name(self, reply: ChatReply) -> str | None:
        """The tool a reply cut at the ceiling was calling, or ``None``.

        A native call cut mid-arguments arrives with ``arguments_error``
        set (the JSON does not close); a text-protocol call leaves an
        open fence or an object that does not decode.
        """
        for call in reply.tool_calls:
            if call.arguments_error is not None:
                return call.name
        return unfinished_call_name(reply.content, fenced=self.fenced)

    def _call_model(self, *, hint: str | None = None) -> ChatReply:
        tools = None if self.fenced else self.toolbox.specs()
        # An ephemeral user message tacked onto the end each turn — the
        # step-of-budget line, and stricter guidance near the ceiling.
        # Never persisted: the counter changes each turn and stale copies
        # in the transcript would mislead --resume. Deliberately not
        # role="system": most chat templates (Qwen's included) accept a
        # system message only at position 0, and the server 400s otherwise.
        messages = self.messages
        if hint is not None:
            messages = [*messages, {"role": "user", "content": hint}]
        # A one-call effort override, set by the cut-reply retry and consumed
        # here, so the next ordinary step runs at the configured effort.
        options: dict[str, Any] = {"max_tokens": self._reply_budget()}
        if self._effort_override is not None:
            options["effort"] = self._effort_override
            self._effort_override = None
        try:
            reply = self.client.chat(messages, tools, **options)
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
                messages = [*messages, {"role": "user", "content": hint}]
            options["max_tokens"] = self._reply_budget()
            reply = self.client.chat(messages, tools, **options)
        self.session.count_usage(
            reply.prompt_tokens, reply.completion_tokens, reply.cached_prompt_tokens
        )
        self._warn_if_truncated(messages, tools, reply.prompt_tokens)
        self._last_prompt_tokens = reply.prompt_tokens
        self.session.record(
            {
                "type": "usage",
                "cached_prompt_tokens": reply.cached_prompt_tokens,
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "finish_reason": reply.finish_reason,
            }
        )
        return reply

    def _reply_budget(self) -> int:
        """One step's ``max_tokens``: the configured or default cap, bounded by
        the room the window has left after the prompt."""
        agent = self.session.agent
        budget = agent.max_reply_tokens or _DEFAULT_REPLY_TOKENS
        room = agent.context_window - self._estimated_prompt_tokens() - _REPLY_MARGIN_TOKENS
        return max(_MIN_REPLY_TOKENS, min(budget, room))

    def _warn_if_truncated(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        prompt_tokens: int | None,
    ) -> None:
        """Say so, once, when the server counted far fewer prompt tokens than
        were sent: it is truncating the context, and the model is answering
        without its instructions.

        Ollama does exactly this silently, to its ``num_ctx`` (2048 or 4096
        by default) while Mason's fixed prefix alone is several thousand
        tokens. A session run that way looks like a weak model; it is a
        blind one. The check is a warning, not a stop: a server's count and
        the chars/4 estimate can differ, so only a factor of two below the
        estimate, on a prompt of real size, counts.
        """
        if prompt_tokens is None or self._truncation_warned:
            return
        sent = sum(len(json.dumps(m)) for m in messages) + len(json.dumps(tools or []))
        estimate = sent // _CHARS_PER_TOKEN
        if estimate < _TRUNCATION_MIN_TOKENS or prompt_tokens * 2 >= estimate:
            return
        self._truncation_warned = True
        text = (
            f"the server counted {prompt_tokens} prompt tokens for a request of about "
            f"{estimate}: it is truncating the context, so the model cannot see its "
            f"instructions. Ollama truncates to its num_ctx; serve the model with a "
            f"larger context (a Modelfile with 'PARAMETER num_ctx 32768', or "
            f"OLLAMA_CONTEXT_LENGTH=32768 before starting the server) and run again."
        )
        self.session.record({"type": "warning", "text": text})
        observer = self.session.observer
        if observer is not None:
            observer("harness", self.session.attribution(), f"[warning] {text}")

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
            self._append({"role": "user", "content": f"[tool result: {call.name}]\n{result}"})
        else:
            self._append({"role": "tool", "tool_call_id": call.id, "content": result})

    def _absent_tools(self) -> list[str]:
        """Vocabulary tools this session does not offer, for the prompt to name."""
        return sorted(name for name in TOOL_VOCABULARY if name not in self.toolbox.tools)

    # -- tool-result clearing -------------------------------------------------

    def _clear_tool_results(self) -> None:
        """Replace old tool results with placeholders once the prompt is large.

        The lightest form of context hygiene, and the one the evidence
        favors: masking old observations matches LLM summarization on
        solve rate at about half the cost (the SWE-agent and OpenHands
        studies), so it runs first and compaction stays the rare fallback.
        The rules follow the shape of Anthropic's ``clear_tool_uses``: a
        trigger on prompt size, the newest results kept intact, oldest
        cleared first, some tools excluded, and a minimum batch so the
        cached prefix is not invalidated for a trivial gain. The call and
        its arguments stay in the history, so a cleared result is
        restorable by calling again; errors stay verbatim because the
        record of what went wrong is what stops a model repeating it.
        """
        agent = self.session.agent
        if not enabled(agent, "context-hygiene"):
            return
        if self._estimated_prompt_tokens() < int(
            agent.context_window * agent.clear_tool_results_at
        ):
            return
        names_by_id: dict[str, str] = {}
        results: list[int] = []  # indices of every tool-result message, in order
        clearable: list[tuple[int, str]] = []
        for index, message in enumerate(self.messages):
            role = message.get("role")
            if role == "assistant":
                for call in message.get("tool_calls") or ():
                    names_by_id[str(call.get("id"))] = str(call["function"]["name"])
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if role == "tool":
                name = names_by_id.get(str(message.get("tool_call_id")), "?")
            elif role == "user" and content.startswith(_TEXT_RESULT_PREFIX):
                name = content[len(_TEXT_RESULT_PREFIX) :].split("]", 1)[0]
            else:
                continue
            results.append(index)
            if (
                len(content) < _CLEAR_FLOOR_CHARS
                or _CLEARED_MARK in content[:120]
                or name in _NEVER_CLEARED
                or _is_error_result(name, content)
            ):
                continue
            clearable.append((index, name))
        protected = set(results[-agent.keep_tool_results :])
        chosen = [(index, name) for index, name in clearable if index not in protected]
        freed = sum(len(str(self.messages[index]["content"])) for index, _ in chosen)
        if not chosen or freed < _CLEAR_AT_LEAST_CHARS:
            return
        for index, name in chosen:
            message = self.messages[index]
            content = str(message["content"])
            prefix, body = "", content
            if message.get("role") == "user":
                head, _, body = content.partition("\n")
                prefix = head + "\n"
            # The first line survives: a signature, a count, an exit status.
            # It is often the one fact the model wanted, and it keeps a
            # cleared record from being fetched again just to be sure.
            began = body.strip().split("\n", 1)[0][:_CLEAR_HEAD_CHARS]
            placeholder = (
                f"{prefix}{_CLEARED_MARK} {name} result, {len(content)} characters, "
                f"cleared to save context; it began {began!r}; the call above shows "
                f"what was asked — call again if the content is needed]"
            )
            message["content"] = placeholder
        # The server's last count described the uncleared prompt.
        self._last_prompt_tokens = None
        self.session.record({"type": "clearing", "cleared": len(chosen), "chars": freed})
        observer = self.session.observer
        if observer is not None:
            observer(
                "harness",
                self.session.attribution(),
                f"[cleared {len(chosen)} old tool result(s), {freed} characters]",
            )

    # -- compaction -----------------------------------------------------------

    def _estimated_prompt_tokens(self) -> int:
        """The server's last count when we have one, chars/4 as the floor."""
        sent = sum(len(json.dumps(m)) for m in self.messages)
        if not self.fenced:
            sent += len(json.dumps(self.toolbox.specs()))  # the schemas ride every request
        estimate = sent // _CHARS_PER_TOKEN
        if self._last_prompt_tokens is not None:
            return max(self._last_prompt_tokens, estimate // 2)
        return estimate

    def _maybe_compact(self) -> None:
        agent = self.session.agent
        if self._estimated_prompt_tokens() < int(agent.context_window * agent.compact_at):
            return
        # Over budget but little new since the last fold: compacting again
        # would re-summarize the summary every step, burning a model call
        # per turn for no reduction (a small window whose system prompt and
        # kept tail already exceed the budget did exactly that). Let a few
        # steps accumulate; a real overflow is caught by the server and
        # handled loudly.
        if len(self.messages) < self._messages_at_last_compaction + _COMPACTION_MIN_NEW_MESSAGES:
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
            effort="low",
            max_tokens=_COMPACTION_MAX_TOKENS,
        )
        summary = (summary_reply.content or "").strip() or "(the summarizer said nothing)"
        self.session.count_usage(
            summary_reply.prompt_tokens,
            summary_reply.completion_tokens,
            summary_reply.cached_prompt_tokens,
        )
        # The summarizer's call is a model call like any other: the report
        # and the session's own counters must agree.
        self.session.record(
            {
                "type": "usage",
                "for": "compaction",
                "cached_prompt_tokens": summary_reply.cached_prompt_tokens,
                "prompt_tokens": summary_reply.prompt_tokens,
                "completion_tokens": summary_reply.completion_tokens,
                "finish_reason": summary_reply.finish_reason,
            }
        )
        # The summary already travels two ways: as a user message prepended
        # to the rebuilt conversation, and as a {type: compaction} event in
        # the transcript. The per-session file is a human debugging aid,
        # deliberately not the notebook — the notebook is what the AGENT
        # kept, this is what the HARNESS folded.
        self.session.compactions_append(summary)
        rebuilt = system_messages(
            self.session,
            self.spec,
            self._catalog,
            skills=self.skills,
            team=self._team,
            review=self._shows_review,
            absent_tools=self._absent_tools(),
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
        # The messages were rebuilt: no earlier result copy is where it was.
        self._seen_results = {
            key: (digest, times, None, step)
            for key, (digest, times, _index, step) in self._seen_results.items()
        }
        self.session.record({"type": "compaction", "summary": summary})
        return True


def _join_cut(prefix: str, rest: str) -> str:
    """The cut half and its continuation as one text.

    The continuation was asked to resume from the last complete line, so
    the prefix's unfinished last line is dropped before the join; a prefix
    with no line break is kept whole.

    Examples:
        >>> _join_cut("a = 1\\nb = 2\\nc = ", "c = 3\\n")
        'a = 1\\nb = 2\\nc = 3\\n'
        >>> _join_cut("one long sen", "tence")
        'one long sentence'
    """
    if prefix.endswith("\n") or "\n" not in prefix:
        return prefix + rest
    return prefix[: prefix.rfind("\n") + 1] + rest


def _is_error_result(name: str, content: str) -> bool:
    """A tool result that records a failure or a refusal, kept verbatim.

    Examples:
        >>> _is_error_result("shell", "tool shell failed: exit 2\\n...")
        True
        >>> _is_error_result("shell", "exit 0\\nall good")
        False
    """
    prefixes = (
        f"tool {name} failed:",
        f"tool {name} not run:",
        f"tool {name} was not approved",
        "refused",
    )
    return content.startswith(prefixes)


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
