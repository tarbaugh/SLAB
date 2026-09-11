"""A session digest: arithmetic over one transcript's event stream.

``slab mason report`` answers the questions a person asks after a campaign
without reading the transcript end to end: how many steps went where, what
the session launched, where the friction was, and whether it finished.
Everything here is counting — no model is involved, and a report on a
transcript that is still being appended to simply describes what has
happened so far.

The event vocabulary is the one :meth:`mason.session.MasonSession.record`
writes: ``session`` (the header naming the model that answered),
``message``, ``reasoning``, ``skill``, ``compaction``, ``finish``, ``retire``,
``resume``, and ``usage``. A malformed line is counted and skipped — a
report must describe a damaged transcript, not refuse it.
"""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foundation.runtime import Workspace

# The launch surfaces: the first call to either is the moment a session
# stops preparing and starts computing.
_LAUNCH_TOOLS = frozenset({"launch_workflow", "submit_job"})

# A tool result that reports the harness's own friction, by the two
# conventions the toolbox uses: the file fence and approval gate say
# "refused: ...", and a tool that raised says "tool <name> failed: ...".
_ERRORED = re.compile(r"^tool \S+ failed: ")

_FINISH_HEAD_CHARS = 200


def _delegation_agent(conversation_stem: str, sibling: Path) -> str:
    """The agent name inside ``<stem>-<agent>-<n>.jsonl``."""
    tail = sibling.stem.removeprefix(f"{conversation_stem}-")
    name, _, ordinal = tail.rpartition("-")
    return name if name and ordinal.isdigit() else tail


def _span_seconds(started: str | None, ended: str | None) -> float | None:
    if not started or not ended:
        return None
    try:
        return (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        return None


def _budget_of(header: dict[str, Any]) -> dict[str, int] | None:
    """The ``budget: {cpus, gpus}`` counts of a header, or None before they existed."""
    raw = header.get("budget")
    if not isinstance(raw, dict):
        return None
    try:
        return {"cpus": int(raw.get("cpus") or 0), "gpus": int(raw.get("gpus") or 0)}
    except (TypeError, ValueError):
        return None


def _tally(transcript: Path) -> dict[str, Any]:
    """Counts for one transcript file; the shared core of the summary."""
    steps = prompt_tokens = completion_tokens = cached_prompt_tokens = 0
    peak_prompt_tokens = 0
    malformed = compactions = clearings = resumes = refusals = errored = 0
    started: str | None = None
    ended: str | None = None
    tools: Counter[str] = Counter()
    refused_tools: Counter[str] = Counter()
    errored_tools: Counter[str] = Counter()
    skills: list[str] = []
    warnings: list[str] = []
    first_launch_step: int | None = None
    finish_head: str | None = None
    finish_report: str | None = None
    finish_results: dict[str, Any] = {}
    finish_run_ids: list[str] = []
    finished = False
    retire: dict[str, Any] | None = None
    commands: Counter[str] = Counter()
    header: dict[str, Any] = {}
    # Tool results carry no name, but they answer the most recent
    # assistant message's calls in order.
    pending: deque[str] = deque()

    for line in transcript.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(event, dict):  # valid JSON, but not an event
            malformed += 1
            continue
        at = event.get("at")
        if isinstance(at, str):
            started = started or at
            ended = at
        kind = event.get("type")
        if kind == "session" and not header:
            header = event
        elif kind == "usage":
            steps += 1
            prompt = int(event.get("prompt_tokens") or 0)
            prompt_tokens += prompt
            peak_prompt_tokens = max(peak_prompt_tokens, prompt)
            completion_tokens += int(event.get("completion_tokens") or 0)
            cached_prompt_tokens += int(event.get("cached_prompt_tokens") or 0)
        elif kind == "clearing":
            clearings += 1
        elif kind == "warning":
            warnings.append(str(event.get("text") or ""))
        elif kind == "message":
            message = event.get("message") or {}
            role = message.get("role")
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    name = str((call.get("function") or {}).get("name") or "?")
                    tools[name] += 1
                    pending.append(name)
                    if name in _LAUNCH_TOOLS and first_launch_step is None:
                        first_launch_step = steps
            elif role == "tool":
                name = pending.popleft() if pending else "?"
                content = str(message.get("content") or "")
                if content.startswith("refused"):
                    refusals += 1
                    refused_tools[name] += 1
                elif _ERRORED.match(content):
                    errored += 1
                    errored_tools[name] += 1
        elif kind == "skill":
            skills.append(str(event.get("name")))
        elif kind == "compaction":
            compactions += 1
        elif kind == "resume":
            resumes += 1
        elif kind == "finish":
            finished = True
            report_text = str(event.get("report") or "").strip()
            finish_head = report_text[:_FINISH_HEAD_CHARS] or None
            finish_report = report_text or None
            raw_results = event.get("results")
            finish_results = dict(raw_results) if isinstance(raw_results, dict) else {}
            raw_ids = event.get("run_ids")
            finish_run_ids = [str(r) for r in raw_ids] if isinstance(raw_ids, list) else []
        elif kind == "retire":
            retire = {k: v for k, v in event.items() if k not in ("at", "type")}
        elif kind == "command":
            commands[str(event.get("kind") or "?")] += 1

    return {
        "model": header.get("model"),
        "provider": header.get("provider"),
        "endpoint_origin": header.get("endpoint_origin"),
        "compute_profile": header.get("compute_profile"),
        "agent": header.get("agent"),
        "condition": header.get("condition"),
        "mechanisms": header.get("mechanisms"),
        "budget": _budget_of(header),
        "steps": steps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "peak_prompt_tokens": peak_prompt_tokens,
        "clearings": clearings,
        "started": started,
        "ended": ended,
        "span_s": _span_seconds(started, ended),
        "tools": dict(tools.most_common()),
        "refusals": refusals,
        "refused_tools": dict(refused_tools.most_common()),
        "errored_calls": errored,
        "errored_tools": dict(errored_tools.most_common()),
        "remember": tools.get("remember", 0),
        "recall": tools.get("recall", 0),
        "skills": skills,
        "warnings": warnings,
        "compactions": compactions,
        "resumes": resumes,
        "malformed_lines": malformed,
        "first_launch_step": first_launch_step,
        "finish": {
            "reported": finished,
            "head": finish_head,
            "report": finish_report,
            "results": finish_results,
            "run_ids": finish_run_ids,
        },
        "retire": retire,
        "commands": dict(commands),
    }


def summarize(
    transcript: Path,
    siblings: list[Path] | None = None,
    runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Digest one conversation transcript, delegation siblings included.

    *siblings* are the ``<stem>-<agent>-<n>.jsonl`` files the conversation's
    delegations wrote (:func:`mason.session.transcript_groups` finds them).
    Their steps and tokens are reported per child and rolled into totals.

    *runs* are the session's run rows from :func:`session_runs`. With them
    the summary carries the resource hours those runs held
    (:func:`utilisation`); without them those fields are None, because
    "held nothing" and "not looked up" are different answers.
    """
    summary = _tally(transcript)
    summary["transcript"] = str(transcript)
    summary["session"] = transcript.stem

    delegations: list[dict[str, Any]] = []
    total_steps = summary["steps"]
    total_prompt = summary["prompt_tokens"]
    total_completion = summary["completion_tokens"]
    total_cached = summary["cached_prompt_tokens"]
    for sibling in siblings or []:
        child = _tally(sibling)
        delegations.append(
            {
                "agent": _delegation_agent(transcript.stem, sibling),
                "transcript": str(sibling),
                "steps": child["steps"],
                "prompt_tokens": child["prompt_tokens"],
                "completion_tokens": child["completion_tokens"],
            }
        )
        total_steps += child["steps"]
        total_prompt += child["prompt_tokens"]
        total_completion += child["completion_tokens"]
        total_cached += child["cached_prompt_tokens"]
    summary["delegations"] = delegations
    summary["total_steps"] = total_steps
    summary["total_prompt_tokens"] = total_prompt
    summary["total_completion_tokens"] = total_completion
    summary["total_cached_prompt_tokens"] = total_cached
    summary.update(
        utilisation(summary, runs) if runs is not None else dict.fromkeys(UTILISATION_KEYS)
    )
    return summary


UTILISATION_KEYS = (
    "cpu_hours_held",
    "gpu_hours_held",
    "wall_hours",
    "runs_sized",
    "runs_unsized",
    "runs_open",
    "utilisation",
)


def utilisation(summary: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The cpu-hours and gpu-hours a session's runs held, against its budget.

    A run holds its slice (the ``resources`` on its record) from
    ``started_at`` to ``finished_at``, so its cpu-hours held are the size of
    the slice's cpu list times that span, and its gpu-hours the same over
    the gpu list. "Held" is the word: nothing here measures whether the
    cores were busy. The sums go over the runs that have both a slice and a
    closed span. ``runs_unsized`` counts runs without a slice (never
    reserved, or from before the store kept one) and ``runs_open`` counts
    sized runs that have not finished; both contribute nothing.

    ``utilisation`` is held over the budget times the session's wall time,
    per resource: ``{"cpu": 0.13, "gpu": 0.26}``. It is None when the
    header records no budget, and a resource the budget has none of reads
    None inside it. ``wall_hours`` is the transcript span.

    Examples:
        >>> rows = [
        ...     {"resources": {"cpus": [0, 1], "gpus": ["0"]}, "seconds": 1800.0},
        ...     {"resources": {"cpus": [2], "gpus": []}, "seconds": 3600.0},
        ...     {"resources": None, "seconds": 10.0},
        ...     {"resources": {"cpus": [3], "gpus": []}, "seconds": None},
        ... ]
        >>> held = utilisation({"budget": {"cpus": 4, "gpus": 2}, "span_s": 7200.0}, rows)
        >>> held["cpu_hours_held"], held["gpu_hours_held"], held["wall_hours"]
        (2.0, 0.5, 2.0)
        >>> held["runs_sized"], held["runs_unsized"], held["runs_open"]
        (2, 1, 1)
        >>> held["utilisation"]
        {'cpu': 0.25, 'gpu': 0.125}
        >>> utilisation({"budget": None, "span_s": 7200.0}, rows)["utilisation"] is None
        True
    """
    cpu_seconds = gpu_seconds = 0.0
    sized = unsized = open_ = 0
    for run in runs:
        slice_ = run.get("resources")
        if not isinstance(slice_, dict):
            unsized += 1
            continue
        seconds = run.get("seconds")
        if seconds is None:
            open_ += 1
            continue
        sized += 1
        cpu_seconds += len(slice_.get("cpus") or ()) * float(seconds)
        gpu_seconds += len(slice_.get("gpus") or ()) * float(seconds)
    span_s = summary.get("span_s")
    wall_hours = float(span_s) / 3600 if span_s is not None else None
    budget = summary.get("budget")
    shares: dict[str, float | None] | None = None
    if budget is not None:
        shares = {}
        for resource, held in (("cpu", cpu_seconds), ("gpu", gpu_seconds)):
            capacity = int(budget.get(f"{resource}s") or 0)
            shares[resource] = (
                held / (capacity * float(span_s)) if capacity and span_s else None
            )
    return {
        "cpu_hours_held": cpu_seconds / 3600,
        "gpu_hours_held": gpu_seconds / 3600,
        "wall_hours": wall_hours,
        "runs_sized": sized,
        "runs_unsized": unsized,
        "runs_open": open_,
        "utilisation": shares,
    }


def session_runs(ws: Workspace, session: str) -> list[dict[str, Any]]:
    """The runs a session created, as report rows; empty when it made none.

    Each row carries the slice the run held (``resources``, None for a run
    that was never reserved), its ``started`` and ``ended`` timestamps, and
    the ``seconds`` between them (None until the run has both), which is
    what :func:`utilisation` sums.

    ``list_runs`` refuses an unknown session id, but for a report "this
    session launched nothing" is an answer, not an error.
    """
    from foundation.errors import SessionNotFoundError

    try:
        runs = ws.runs.list_runs(session=session, limit=100)
    except SessionNotFoundError:
        return []
    rows = []
    for run in runs:
        started = run.started_at.isoformat() if run.started_at else None
        ended = run.finished_at.isoformat() if run.finished_at else None
        rows.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state.value,
                "status": run.status.value,
                "resources": dict(run.resources) if run.resources else None,
                "started": started,
                "ended": ended,
                "seconds": _span_seconds(started, ended),
            }
        )
    return rows
