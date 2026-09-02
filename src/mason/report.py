"""A session digest: arithmetic over one transcript's event stream.

``slab mason report`` answers the questions a person asks after a campaign
without reading the transcript end to end: how many steps went where, what
the session launched, where the friction was, and whether it finished.
Everything here is counting — no model is involved, and a report on a
transcript that is still being appended to simply describes what has
happened so far.

The event vocabulary is the one :meth:`mason.session.MasonSession.record`
writes: ``session`` (the header naming the model that answered),
``message``, ``reasoning``, ``skill``, ``compaction``, ``finish``,
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

    return {
        "model": header.get("model"),
        "provider": header.get("provider"),
        "endpoint_origin": header.get("endpoint_origin"),
        "compute_profile": header.get("compute_profile"),
        "agent": header.get("agent"),
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
    }


def summarize(transcript: Path, siblings: list[Path] | None = None) -> dict[str, Any]:
    """Digest one conversation transcript, delegation siblings included.

    *siblings* are the ``<stem>-<agent>-<n>.jsonl`` files the conversation's
    delegations wrote (:func:`mason.session.transcript_groups` finds them).
    Their steps and tokens are reported per child and rolled into totals.
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
    return summary


def session_runs(ws: Workspace, session: str) -> list[dict[str, str]]:
    """The runs a session created, as report rows; empty when it made none.

    ``list_runs`` refuses an unknown session id, but for a report "this
    session launched nothing" is an answer, not an error.
    """
    from foundation.errors import SessionNotFoundError

    try:
        runs = ws.runs.list_runs(session=session, limit=100)
    except SessionNotFoundError:
        return []
    return [
        {
            "id": run.id,
            "name": run.name,
            "state": run.state.value,
            "status": run.status.value,
        }
        for run in runs
    ]
