"""The science review: flags an evaluator raises on a scored campaign.

The scorer says whether a campaign passed and why it did not. That is an
outcome, not a defect: "a0 outside the band" names nothing a revision can
edit. The review closes that gap. Two evaluators read the same campaign
evidence (the transcript, the runs it made, the finish it reported) and
raise **flags**. Every flag carries a ``rule`` (what went wrong), a
``target`` (what a revision edits to stop it), ``evidence`` (where in the
campaign), and a one-sentence ``note``. Targets are:

- ``skill:<name>`` — a skill's description, body, or bundled script;
- ``card:<agent>`` — the entry card's role prompt;
- ``tool:<name>`` — a tool's description or schema;
- ``prompt`` — the system prompt outside any card.

The **rules** are code: deterministic checks that run on every scored
campaign. The **referee** is a model that reads the evidence pack against
a rubric and argues with the procedure; it runs on request, because it
costs a model call. Both write the same flag shape into the campaign
record, beside ``passed`` and ``reason``.

A flag on a skill is raised against the skill's **digest**, the content
hash the ``skill`` tool records when the skill loads. That is what makes a
flag attributable to one revision and a fix verifiable: ``gate`` refuses
to call a skill revision validated until a campaign has run under the new
digest, passed at least as often as before, and stopped raising the flag.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foundation.errors import SessionNotFoundError
from mason.skills import Skill, discover_skills
from mason.tools import LOOKING_TOOLS

if TYPE_CHECKING:
    from foundation.runtime import Workspace
    from mason.loop import ChatBackend
    from slab_stack.benchmark import Question

#: Lifecycle states a run must have reached for the rules to leave it alone.
_PASSING_STATES = frozenset({"verified", "promoted", "archived"})

#: Who raised a flag.
RULES = "rules"
REFEREE = "referee"

#: The target of a flag that no skill, tool, or card can be blamed for.
PROMPT_TARGET = "prompt"

_RULE_NAME = re.compile(r"[^a-z0-9]+")
_ARGS_CHARS = 300
_RESULT_CHARS = 400
_SKILL_BODY_CHARS = 6000
_UNIT_ALIASES = {"a": "å", "angstrom": "å", "ang": "å", "ev/a": "ev/å", "ev/angstrom": "ev/å"}


class ReviewError(Exception):
    """The referee could not deliver a review; the message says why."""


@dataclass(frozen=True)
class Flag:
    """One attributable defect raised on a campaign.

    Examples:
        >>> flag = Flag("script-not-used", "skill:equation-of-state", "step 4", "no fit ran")
        >>> Flag.from_dict(flag.as_dict()) == flag
        True
    """

    rule: str
    target: str
    evidence: str
    note: str
    raised_by: str = RULES

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Flag:
        return cls(
            rule=str(raw.get("rule") or "unnamed"),
            target=str(raw.get("target") or PROMPT_TARGET),
            evidence=str(raw.get("evidence") or ""),
            note=str(raw.get("note") or ""),
            raised_by=str(raw.get("raised_by") or RULES),
        )

    @property
    def skill(self) -> str | None:
        """The skill name when the target is a skill, else None.

        Examples:
            >>> Flag("r", "skill:surface-energy", "", "").skill
            'surface-energy'
            >>> Flag("r", "card:pi", "", "").skill is None
            True
        """
        return self.target.removeprefix("skill:") if self.target.startswith("skill:") else None


# -- the campaign as evidence -------------------------------------------------


@dataclass
class Step:
    """One model call: the tool calls it made, the results they got, its cost."""

    index: int
    at: str | None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    #: The completion tokens the server billed for this step, when the
    #: transcript's usage event said.
    completion_tokens: int | None = None

    @property
    def looking(self) -> bool:
        """True when every call only reads: shell, files, listings, lookups."""
        return bool(self.calls) and all(name in LOOKING_TOOLS for name, _ in self.calls)


@dataclass
class SkillLoad:
    """One ``skill`` event: which skill, which revision, when."""

    name: str
    digest: str
    at: str | None
    step: int


@dataclass
class Campaign:
    """A transcript read as evidence: steps, skill loads, and the finish."""

    header: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    loads: list[SkillLoad] = field(default_factory=list)
    finish: dict[str, Any] | None = None

    @property
    def agent(self) -> str:
        return str(self.header.get("agent") or "pi")

    @property
    def effort(self) -> str | None:
        """The reasoning dial the header recorded, or None for an older transcript."""
        value = self.header.get("effort")
        return None if value is None else str(value)

    @property
    def loaded(self) -> dict[str, str]:
        """Skill name to the digest of the revision that loaded (last load wins)."""
        return {load.name: load.digest for load in self.loads}

    def shell_calls(self, *, after_step: int = 0) -> list[tuple[int, str, str]]:
        """``(step, command, result)`` for every shell call after *after_step*."""
        found: list[tuple[int, str, str]] = []
        for step in self.steps:
            if step.index <= after_step:
                continue
            for position, (name, args) in enumerate(step.calls):
                if name != "shell":
                    continue
                result = step.results[position] if position < len(step.results) else ""
                found.append((step.index, str(args.get("command") or ""), result))
        return found

    def skill_in_force(self, when: str | None) -> str | None:
        """The skill loaded most recently before *when* (an ISO timestamp)."""
        moment = _parse_time(when)
        if moment is None:
            return None
        current: str | None = None
        for load in self.loads:
            loaded_at = _parse_time(load.at)
            if loaded_at is not None and loaded_at <= moment:
                current = load.name
        return current


def _parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return loaded if isinstance(loaded, dict) else {"_raw": raw}
    return {}


def walk(transcript: Path) -> Campaign:
    """Read a transcript into a :class:`Campaign`; malformed lines are skipped.

    A usage event precedes the assistant message it paid for (the loop
    records the count before it appends the message), so the count waits
    for the next step.
    """
    campaign = Campaign()
    step_index = 0
    current: Step | None = None
    pending: list[Step] = []
    paid: int | None = None
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        at = event.get("at") if isinstance(event.get("at"), str) else None
        if kind == "session" and not campaign.header:
            campaign.header = event
        elif kind == "message":
            message = event.get("message") or {}
            role = message.get("role")
            if role == "assistant":
                step_index += 1
                current = Step(index=step_index, at=at, completion_tokens=paid)
                paid = None
                campaign.steps.append(current)
                for call in message.get("tool_calls") or []:
                    name = str((call.get("function") or {}).get("name") or "?")
                    current.calls.append((name, _arguments(call)))
                    pending.append(current)
            elif role == "tool" and pending:
                owner = pending.pop(0)
                owner.results.append(str(message.get("content") or ""))
        elif kind == "usage" and "for" not in event:
            raw = event.get("completion_tokens")
            paid = int(raw) if isinstance(raw, int) else None
        elif kind == "skill":
            campaign.loads.append(
                SkillLoad(
                    name=str(event.get("name") or ""),
                    digest=str(event.get("digest") or "unknown"),
                    at=at,
                    step=step_index,
                )
            )
        elif kind == "finish":
            campaign.finish = event
    return campaign


def loaded_skills(transcript: Path) -> dict[str, str]:
    """Skill name to digest for every skill the campaign loaded."""
    return walk(transcript).loaded


def session_runs(ws: Workspace, session: str) -> list[dict[str, Any]]:
    """Details of every run the session created; empty when it made none."""
    from foundation import _ops

    try:
        runs = ws.runs.list_runs(session=session, limit=200)
    except SessionNotFoundError:
        return []
    return [_ops.run_details(ws, run.id) for run in runs]


# -- the rules ----------------------------------------------------------------

#: This many consecutive looking steps is a loop, not reconnaissance. One
#: real campaign spent 72 minutes and 80 % of its completion tokens in two
#: such windows, rewriting a potential file that one keyword would have
#: loaded.
_PROGRESS_WINDOW = 15
#: A step billed this many completion tokens wrote no plan, file, note, or
#: report: the model thought at length and had little to show for it.
_HEAVY_STEP_TOKENS = 8_000
#: The tools whose call justifies a long think.
_WRITING_TOOLS = frozenset({"plan", "finish", "write_file", "edit_file", "notebook"})


def _script_names(skill: Skill) -> list[str]:
    scripts = skill.root / "scripts"
    if not scripts.is_dir():
        return []
    return sorted(p.name for p in scripts.iterdir() if p.is_file() and not p.name.startswith("."))


def _failed(result: str) -> bool:
    if result.startswith("tool shell failed:") or result.startswith("refused"):
        return True
    head, _, _ = result.partition("\n")
    return head.startswith("exit ") and head.split()[1:2] not in (["0"], [])


def _normal_unit(unit: str) -> str:
    lowered = " ".join(unit.split()).lower().replace("^", "")
    return _UNIT_ALIASES.get(lowered, lowered)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def rules(
    campaign: Campaign,
    question: Question,
    runs: Iterable[dict[str, Any]],
    catalog: dict[str, Skill],
) -> list[Flag]:
    """The deterministic evaluator: every rule that fires on this campaign.

    Rules, and what each one blames:

    - ``skill-not-loaded``: a skill the question lists never loaded. The
      skill's description did not trigger (``skill:<name>``).
    - ``script-not-used``: a loaded skill bundles scripts and no shell
      call after the load named one. The body did not make the agent run
      them (``skill:<name>``).
    - ``script-failed``: a shell call running a bundled script failed.
      The script, or the body's instructions for it (``skill:<name>``).
    - ``run-not-verified``: a run the session made never reached
      ``verified``. The skill in force when the run started, else the
      card (``skill:<name>`` or ``card:<agent>``).
    - ``unit-mismatch``: a reported unit is not the question's unit
      (``tool:finish``).
    - ``finish-incomplete``: no finish, no structured results, or no run
      ids. The card's reporting instructions (``card:<agent>``).
    - ``no-progress-loop``: fifteen or more consecutive steps only looked
      (shell, reads, listings), with no run, plan change, note, brief, or
      finish. The card's doctrine on when to step back (``card:<agent>``).
    - ``reasoning-heavy``: a step billed 8,000 or more completion tokens
      and wrote nothing. The dial, which the note names (``prompt``).
    """
    flags: list[Flag] = []
    card = f"card:{campaign.agent}"
    loads = {load.name: load for load in campaign.loads}

    for name in question.skills:
        if name not in loads:
            flags.append(
                Flag(
                    "skill-not-loaded",
                    f"skill:{name}",
                    "no skill event in the transcript",
                    f"The campaign never loaded {name}, so its description did not "
                    "trigger for this instruction.",
                )
            )

    for name, load in loads.items():
        skill = catalog.get(name)
        if skill is None:
            continue
        scripts = _script_names(skill)
        if not scripts:
            continue
        calls = campaign.shell_calls(after_step=load.step)
        used = [(step, command, result) for step, command, result in calls
                if any(script in command for script in scripts)]
        if not used:
            flags.append(
                Flag(
                    "script-not-used",
                    f"skill:{name}",
                    f"loaded at step {load.step}; {len(calls)} shell call(s) after it",
                    f"None of the bundled scripts ({', '.join(scripts)}) ran after the "
                    "skill loaded; the number, if any, came from improvised analysis.",
                )
            )
        for step, command, result in used:
            if _failed(result):
                flags.append(
                    Flag(
                        "script-failed",
                        f"skill:{name}",
                        f"step {step}: {command[:80]}",
                        _first_line(result)[:200] or "the call failed",
                    )
                )

    for run in runs:
        info = run.get("run") or {}
        state = str(info.get("state") or "")
        if state in _PASSING_STATES:
            continue
        run_id = str(info.get("id") or "")[:10]
        checks = run.get("checks") or []
        failed_checks = [str(c.get("name")) for c in checks if not c.get("passed")]
        if failed_checks:
            why = "checks failed: " + ", ".join(failed_checks)
        elif not checks:
            why = "no checks were registered, so nothing could verify it"
        else:
            why = str(info.get("error") or f"left in state {state}")
        in_force = campaign.skill_in_force(info.get("created_at") or info.get("started_at"))
        flags.append(
            Flag(
                "run-not-verified",
                f"skill:{in_force}" if in_force else card,
                f"run {run_id} ({state}, {info.get('status')})",
                f"Run {info.get('name')} never reached verified: {why}.",
            )
        )

    finish = campaign.finish or {}
    raw_results = finish.get("results")
    results: dict[str, Any] = raw_results if isinstance(raw_results, dict) else {}
    raw_ids = finish.get("run_ids")
    run_ids: list[Any] = raw_ids if isinstance(raw_ids, list) else []
    if not campaign.finish:
        missing = "the campaign never called finish"
    elif not results:
        missing = "the finish carried no structured results"
    elif not run_ids:
        missing = "the finish cited no run ids"
    else:
        missing = ""
    if missing:
        flags.append(
            Flag(
                "finish-incomplete",
                card,
                "finish",
                f"{missing}; the card's reporting instructions did not hold.",
            )
        )
    for name, unit in question.results.items():
        entry = results.get(name)
        reported = entry.get("unit") if isinstance(entry, dict) else None
        if isinstance(reported, str) and _normal_unit(reported) != _normal_unit(unit):
            flags.append(
                Flag(
                    "unit-mismatch",
                    "tool:finish",
                    "finish",
                    f"{name} was reported in {reported!r}; the question asks for {unit!r}.",
                )
            )

    for first, last, spent in _looking_windows(campaign.steps):
        flags.append(
            Flag(
                "no-progress-loop",
                card,
                f"steps {first}-{last}",
                f"{last - first + 1} consecutive steps only looked (shell, reads, listings) "
                f"with no run, plan change, note, brief, or finish; {spent} completion "
                f"tokens went into them.",
            )
        )
    heavy = [
        step
        for step in campaign.steps
        if (step.completion_tokens or 0) >= _HEAVY_STEP_TOKENS
        and not any(name in _WRITING_TOOLS for name, _ in step.calls)
    ]
    if heavy:
        dial = campaign.effort or "unset"
        peak = max(step.completion_tokens or 0 for step in heavy)
        flags.append(
            Flag(
                "reasoning-heavy",
                "prompt",
                "steps " + ", ".join(str(step.index) for step in heavy),
                f"{len(heavy)} step(s) billed {_HEAVY_STEP_TOKENS:,}+ completion tokens "
                f"(peak {peak:,}) and wrote no plan, file, note, or report; effort was "
                f"{dial}.",
            )
        )
    return flags


def _looking_windows(steps: list[Step]) -> list[tuple[int, int, int]]:
    """``(first, last, completion_tokens)`` for every maximal run of looking
    steps at least ``_PROGRESS_WINDOW`` long.

    Examples:
        >>> looks = [Step(i, None, [("shell", {})], completion_tokens=10) for i in range(1, 17)]
        >>> _looking_windows(looks)
        [(1, 16, 160)]
        >>> plan = Step(9, None, [("plan", {})])
        >>> _looking_windows(looks[:8] + [plan] + looks[9:])
        []
    """
    windows: list[tuple[int, int, int]] = []
    run: list[Step] = []

    def close() -> None:
        if len(run) >= _PROGRESS_WINDOW:
            spent = sum(step.completion_tokens or 0 for step in run)
            windows.append((run[0].index, run[-1].index, spent))
        run.clear()

    for step in steps:
        if step.looking:
            run.append(step)
        else:
            close()
    close()
    return windows


# -- the referee --------------------------------------------------------------

RUBRIC = """\
You are the referee of a computational materials science benchmark. A research
agent ran one campaign to answer a fixed question. You receive the evidence: the
instruction, the verdict a scorer already reached, the procedure the agent should
have followed (the skills it was given), a digest of every step it took, the runs
it made, and the report it finished with.

Your job is adversarial. Find the defects in the science procedure that a
revision of the agent's instructions could prevent. Judge the procedure, not
the number: convergence declared or assumed, structure relaxed at the engine that
produced the number, the scan bracketing its minimum, slab thickness and vacuum,
supercell size for a defect, equilibration and sampling for dynamics, a held-out
set that was really held out, units and conversions stated, uncertainty
estimated, the bundled script used instead of improvised analysis, every number
tied to a verified run.

Each defect is one flag with four fields:
- "rule": a short kebab-case name for the defect, e.g. "kpoints-not-converged".
- "target": what a revision edits to prevent it. Choose exactly one of the
  allowed targets listed in the evidence.
- "evidence": where in the campaign you saw it: a step number, a run id, or
  "finish".
- "note": one sentence a maintainer can act on.

Raise only defects you can point to in the evidence. Do not restate the
scorer's verdict as a flag. Answer with one JSON object and nothing else:
{"flags": [{"rule": "...", "target": "...", "evidence": "...", "note": "..."}]}
An empty list is a valid answer.
"""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15] + " [... truncated]"


def allowed_targets(campaign: Campaign, question: Question) -> list[str]:
    """The targets a referee may blame: listed and loaded skills, the card,
    every tool the campaign called, and the prompt."""
    skills = sorted(set(question.skills) | set(campaign.loaded))
    tools = sorted({name for step in campaign.steps for name, _ in step.calls} | {"finish"})
    return (
        [f"skill:{name}" for name in skills]
        + [f"card:{campaign.agent}"]
        + [f"tool:{name}" for name in tools]
        + [PROMPT_TARGET]
    )


def evidence_pack(
    campaign: Campaign,
    question: Question,
    record: dict[str, Any],
    runs: Iterable[dict[str, Any]],
    catalog: dict[str, Skill],
) -> str:
    """The referee's brief: everything it may cite, and nothing it may not."""
    lines = [
        "# Instruction",
        question.instruction,
        "",
        "# Scorer's verdict",
        "passed" if record.get("passed") else f"failed: {record.get('reason')}",
        "",
        "# Allowed targets",
        *(f"- {target}" for target in allowed_targets(campaign, question)),
        "",
        "# Procedure the agent was given",
    ]
    for name in sorted(set(question.skills) | set(campaign.loaded)):
        skill = catalog.get(name)
        if skill is None:
            lines.append(f"## skill {name}: not in the catalog")
            continue
        loaded = "loaded" if name in campaign.loaded else "listed for the question, never loaded"
        lines.append(f"## skill {name} ({loaded})")
        lines.append(_truncate(skill.body().strip(), _SKILL_BODY_CHARS))
        lines.append("")
    lines.append("# Steps")
    for step in campaign.steps:
        if not step.calls:
            lines.append(f"step {step.index}: text reply")
            continue
        for position, (name, args) in enumerate(step.calls):
            shown = json.dumps(args, ensure_ascii=False)
            lines.append(f"step {step.index}: {name} {_truncate(shown, _ARGS_CHARS)}")
            if position < len(step.results):
                lines.append("  -> " + _truncate(step.results[position], _RESULT_CHARS))
    lines.append("")
    lines.append("# Runs the session made")
    any_run = False
    for run in runs:
        any_run = True
        info = run.get("run") or {}
        lines.append(
            f"- run {str(info.get('id'))[:10]} {info.get('name')}: state {info.get('state')}, "
            f"status {info.get('status')}"
        )
        for task in run.get("tasks") or []:
            engine = ((task.get("recipe") or {}).get("params") or {}).get("engine")
            lines.append(f"  task {task.get('name')} ({task.get('status')}, engine {engine})")
        for check in run.get("checks") or []:
            verdict = "passed" if check.get("passed") else "FAILED"
            lines.append(f"  check {check.get('name')}: {verdict}")
    if not any_run:
        lines.append("none")
    lines.append("")
    lines.append("# Finish")
    finish = campaign.finish or {}
    if not finish:
        lines.append("the campaign never called finish")
    else:
        lines.append(f"results: {json.dumps(finish.get('results') or {}, ensure_ascii=False)}")
        lines.append(f"run_ids: {finish.get('run_ids') or []}")
        lines.append("report:")
        lines.append(str(finish.get("report") or ""))
    return "\n".join(lines)


def _json_object(text: str) -> dict[str, Any]:
    """The first JSON object in a reply, fences and prose around it ignored."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ReviewError("the referee answered without a JSON object")
    try:
        loaded = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ReviewError(f"the referee's JSON does not parse: {e}") from e
    if not isinstance(loaded, dict):
        raise ReviewError("the referee's JSON is not an object")
    return loaded


def parse_referee(reply: str, allowed: Iterable[str], card: str) -> list[Flag]:
    """Flags from the referee's reply. A target outside *allowed* is
    re-attributed to the card, and the note keeps what the referee named.

    Examples:
        >>> [flag] = parse_referee(
        ...     '{"flags": [{"rule": "Slab Too Thin", "target": "skill:surface-energy",'
        ...     ' "evidence": "step 3", "note": "four layers"}]}',
        ...     ["skill:surface-energy"], "card:pi",
        ... )
        >>> flag.rule, flag.target, flag.raised_by
        ('slab-too-thin', 'skill:surface-energy', 'referee')
    """
    allowed = set(allowed)
    raw_flags = _json_object(reply).get("flags")
    if not isinstance(raw_flags, list):
        raise ReviewError("the referee's JSON has no 'flags' list")
    flags: list[Flag] = []
    for raw in raw_flags:
        if not isinstance(raw, dict):
            continue
        rule = _RULE_NAME.sub("-", str(raw.get("rule") or "unnamed").strip().lower()).strip("-")
        target = str(raw.get("target") or "").strip()
        note = str(raw.get("note") or "").strip()
        if target not in allowed:
            note = f"(the referee named target {target or 'nothing'!r}) {note}".strip()
            target = card
        flags.append(
            Flag(
                rule=rule or "unnamed",
                target=target,
                evidence=str(raw.get("evidence") or "").strip(),
                note=note,
                raised_by=REFEREE,
            )
        )
    return flags


def referee(
    client: ChatBackend,
    campaign: Campaign,
    question: Question,
    record: dict[str, Any],
    runs: Iterable[dict[str, Any]],
    catalog: dict[str, Skill],
) -> list[Flag]:
    """Ask the referee model for its flags on one campaign.

    Raises:
        ReviewError: The model answered with nothing the parser accepts.
    """
    pack = evidence_pack(campaign, question, record, list(runs), catalog)
    reply = client.chat(
        [{"role": "system", "content": RUBRIC}, {"role": "user", "content": pack}]
    )
    return parse_referee(
        reply.content or "", allowed_targets(campaign, question), f"card:{campaign.agent}"
    )


def referee_client(
    root: Path,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    provider: str | None = None,
) -> ChatBackend:
    """A chat client for the referee from the ``[agent]`` config, with overrides.

    The same client the campaign itself would use, so a referee on a
    cluster reaches the served model through the same discovered endpoint.
    """
    from mason.config import override_agent
    from mason.loop import client_from_config
    from mason.session import MasonSession

    session = MasonSession(workspace_root=root)
    updates: dict[str, object] = {}
    if model is not None:
        updates["model"] = model
    if provider is not None:
        updates["provider"] = provider
    if updates:
        session.agent = override_agent(session.agent, updates)
    session.resolve_endpoint(endpoint)
    return client_from_config(session.agent, session.api_keys)


# -- review one campaign ------------------------------------------------------


def review(
    root: Path,
    transcript: Path,
    question: Question,
    record: dict[str, Any],
    *,
    ws: Workspace,
    catalog: dict[str, Skill] | None = None,
    referee_client: ChatBackend | None = None,
) -> None:
    """Run the evaluators on a scored campaign and write their flags into *record*.

    The rules always run. The referee runs when a client is given; a
    referee that fails leaves the rules' flags in place and records the
    failure under ``referee_error``, so a dark endpoint never blanks a
    review.
    """
    campaign = walk(transcript)
    if catalog is None:
        catalog = discover_skills(Path.cwd())
    runs = session_runs(ws, transcript.stem)
    flags = rules(campaign, question, runs, catalog)
    reviewed = [RULES]
    record["referee_model"] = None
    record["referee_error"] = None
    if referee_client is not None:
        record["referee_model"] = getattr(referee_client, "model", None)
        try:
            flags.extend(referee(referee_client, campaign, question, record, runs, catalog))
            reviewed.append(REFEREE)
        except ReviewError as e:
            record["referee_error"] = str(e)
    record["flags"] = [flag.as_dict() for flag in flags]
    record["reviewed_by"] = reviewed


# -- the ledger and the gate --------------------------------------------------

OPEN = "open"
PENDING = "pending"
UNKNOWN = "unknown"


def flags_of(record: dict[str, Any]) -> list[Flag]:
    raw = record.get("flags")
    return [Flag.from_dict(f) for f in raw if isinstance(f, dict)] if isinstance(raw, list) else []


def raised_against(flag: Flag, record: dict[str, Any]) -> str | None:
    """The skill digest a flag was raised against, when its target is a skill."""
    if flag.skill is None:
        return None
    digest = (record.get("skills") or {}).get(flag.skill)
    return str(digest) if digest else None


def flag_status(flag: Flag, record: dict[str, Any], catalog: dict[str, Skill]) -> str:
    """``open`` when the target is unchanged since the flag was raised,
    ``pending`` when a skill revision exists that no campaign has validated,
    ``unknown`` when the skill is not in the catalog or carried no digest."""
    name = flag.skill
    if name is None:
        return OPEN
    skill = catalog.get(name)
    against = raised_against(flag, record)
    if skill is None or against is None or against == "unknown":
        return UNKNOWN
    return OPEN if skill.digest == against else PENDING


def ledger(
    records: Iterable[dict[str, Any]], catalog: dict[str, Skill]
) -> list[dict[str, Any]]:
    """Every flag of the latest record per cell, with its status: the defect list."""
    from slab_stack.benchmark import latest_by_cell

    rows: list[dict[str, Any]] = []
    for (model, machine, key), record in latest_by_cell(records).items():
        for flag in flags_of(record):
            rows.append(
                {
                    **flag.as_dict(),
                    "status": flag_status(flag, record, catalog),
                    "against": raised_against(flag, record),
                    "question": record.get("question"),
                    "key": key,
                    "model": model,
                    "machine": machine,
                    "session": record.get("session"),
                }
            )
    rows.sort(key=lambda r: (r["target"], r["rule"], r["key"], r["model"], r["machine"]))
    return rows


@dataclass(frozen=True)
class GateCell:
    """One (question, model, machine) cell's verdict on a skill revision."""

    question: int
    key: str
    model: str
    machine: str
    verdict: str
    detail: str
    validated: bool


@dataclass(frozen=True)
class GateReport:
    skill: str
    digest: str
    cells: tuple[GateCell, ...]

    @property
    def validated(self) -> bool:
        return bool(self.cells) and all(cell.validated for cell in self.cells)


def _digest_of(record: dict[str, Any], skill: str) -> str | None:
    digest = (record.get("skills") or {}).get(skill)
    return None if digest in (None, "", "unknown") else str(digest)


def gate(skill: str, records: Iterable[dict[str, Any]], catalog: dict[str, Skill]) -> GateReport:
    """Whether the catalog's revision of *skill* is validated by the benchmark.

    For every cell (question, model, machine) that ever recorded a campaign
    for a question that lists the skill or that loaded it, the newest
    record under the current digest is compared with the newest record
    under any other revision. The revision is validated in that cell when a
    record under it exists, it passes if the earlier one passed, and it
    raises no flag against the skill. No cell at all means nothing has
    exercised the skill, which is not validation either.
    """
    from slab_stack.benchmark import QUESTIONS

    found = catalog.get(skill)
    if found is None:
        raise ReviewError(f"no skill named {skill!r} in the catalog")
    digest = found.digest
    listed = {q.key: q for q in QUESTIONS if skill in q.skills}
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("key"))
        if key not in listed and skill not in (record.get("skills") or {}):
            continue
        cell = (key, str(record.get("model")), str(record.get("machine")))
        cells.setdefault(cell, []).append(record)
    verdicts: list[GateCell] = []
    for (key, model, machine), history in sorted(cells.items()):
        number = int(history[-1].get("question") or 0)
        current = next((r for r in reversed(history) if _digest_of(r, skill) == digest), None)
        previous = next((r for r in reversed(history) if _digest_of(r, skill) != digest), None)
        if current is None:
            verdicts.append(
                GateCell(
                    number, key, model, machine, "not validated",
                    f"no scored campaign loaded revision {digest}", False,
                )
            )
            continue
        open_flags = sorted({f.rule for f in flags_of(current) if f.skill == skill})
        before = (
            sorted({f.rule for f in flags_of(previous) if f.skill == skill})
            if previous is not None
            else []
        )
        closed = [rule for rule in before if rule not in open_flags]
        if previous is not None and previous.get("passed") and not current.get("passed"):
            verdicts.append(
                GateCell(
                    number, key, model, machine, "regressed",
                    f"passed under the earlier revision, now fails: {current.get('reason')}",
                    False,
                )
            )
        elif open_flags:
            verdicts.append(
                GateCell(
                    number, key, model, machine, "still flagged",
                    "raises " + ", ".join(open_flags)
                    + (f"; closed {', '.join(closed)}" if closed else ""),
                    False,
                )
            )
        else:
            outcome = "passes" if current.get("passed") else f"fails ({current.get('reason')})"
            detail = f"{outcome}, no flag against the skill"
            if closed:
                detail += f"; closed {', '.join(closed)}"
            verdicts.append(GateCell(number, key, model, machine, "validated", detail, True))
    return GateReport(skill=skill, digest=digest, cells=tuple(verdicts))


__all__ = [
    "OPEN",
    "PENDING",
    "PROMPT_TARGET",
    "REFEREE",
    "RUBRIC",
    "RULES",
    "UNKNOWN",
    "Campaign",
    "Flag",
    "GateCell",
    "GateReport",
    "ReviewError",
    "SkillLoad",
    "Step",
    "allowed_targets",
    "evidence_pack",
    "flag_status",
    "flags_of",
    "gate",
    "ledger",
    "loaded_skills",
    "parse_referee",
    "raised_against",
    "referee",
    "referee_client",
    "review",
    "rules",
    "session_runs",
    "walk",
]
